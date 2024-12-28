# %%
from copy import deepcopy
from multiprocessing import Pool
from random import sample as random_sample
from random import seed
import time, logging, typing
from time import localtime, strftime
from sklearn.datasets import load_iris as load_iris_raw
import numpy as np
from tqdm.auto import tqdm
from z3 import *
from utils.dictionary_iris import *
from utils.encoding_iris import *
from utils import *

def load_iris() -> tuple[TImageBatch,TLabelBatch,TImageBatch,TLabelBatch]:
    # Parameter setting
    cats = [*range(10)]

    # General variables
    inputs_list = []  # To keep training images
    labels_list = []  # To keep training labels
    test_inputs_list = []  # To keep test images
    test_labels_list = []  # To keep test labels

    iris = load_iris_raw()
    raw_data_max = np.quantile(iris.data, 0.95)  # which is np.max(iris.data)

    raw_inputs, raw_labels = iris.data, iris.target # type: ignore
    raw_inputs = np.array(raw_inputs)
    for i in range(len(raw_labels)):
        if raw_labels[i] in cats:
            inputs_list.append(np.floor((raw_data_max - raw_inputs[i].reshape(n_layer_neurons[0], 1)) * (num_steps-1) / raw_data_max).astype(int))
            labels_list.append(cats.index(raw_labels[i]))

    raw_inputs, raw_labels = iris.data, iris.target # type: ignore
    raw_inputs = np.array(raw_inputs)
    for i in range(len(raw_labels)):
        if raw_labels[i] in cats:
            # images_test.append(TTT[i].reshape(28,28).astype(int))
            test_inputs_list.append(np.floor((raw_data_max - raw_inputs[i].reshape(n_layer_neurons[0], 1)) * (num_steps-1) / raw_data_max).astype(int))
            test_labels_list.append(cats.index(raw_labels[i]))

    #images contain values within [0,(num_steps-1)]
    inputs = typing.cast(TImageBatch, np.asarray(inputs_list))
    labels = typing.cast(TLabelBatch, np.asarray(labels_list))
    inputs_test = typing.cast(TImageBatch, np.asarray(test_inputs_list))
    labels_test = typing.cast(TLabelBatch, np.asarray(test_labels_list))
    
    return inputs, labels, inputs_test, labels_test

mgrid = np.mgrid[0:layer_shapes[0][0], 0:layer_shapes[0][1]]
def forward(weights_list:TWeightList, input:TImage):
    SpikeImage = np.zeros((layer_shapes[0][0],layer_shapes[0][1],(num_steps-1)+1))
    firingTime = []
    Spikes = []
    X = []
    for layer, neuron_of_layer in enumerate(n_layer_neurons[1:]):
        firingTime.append(np.asarray(np.zeros(neuron_of_layer)))
        Spikes.append(np.asarray(np.zeros((layer_shapes[layer + 1][0], layer_shapes[layer + 1][1], (num_steps-1)))))
        X.append(np.asarray(np.mgrid[0:layer_shapes[layer + 1][0], 0:layer_shapes[layer + 1][1]]))
    
    SpikeList = [SpikeImage] + Spikes
    
    SpikeImage[mgrid[0], mgrid[1], input] = 1
    for layer in range(len(n_layer_neurons)-1):
        Voltage = np.cumsum(np.tensordot(weights_list[layer], SpikeList[layer]), 1)
        Voltage[:, (num_steps-1)-1] = threshold + 1
        firingTime[layer] = np.argmax(Voltage > threshold, axis=1).astype(float) + 1
        # in layer 0, max time is (num_steps-1)-1, but in layer 1, max time is (num_steps-1), so we clamp it.
        firingTime[layer][firingTime[layer] > (num_steps-1)-1] = (num_steps-1)-1
        Spikes[layer][...] = 0
        Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(n_layer_neurons[layer+1], 1).astype(int)] = 1 # All neurons spike only once.
    
    V = int(np.argmin(firingTime[-1]))
    return V

def prepare_weights() -> TWeightList:
    if train:
        raise NotImplementedError("The model must be trained from S4NN.")
    else:
        # weights_list = np.load("mnist_weights_best.npy", allow_pickle=True)
        model_dir_path = f"models/{(num_steps)}_{'_'.join(str(i) for i in n_layer_neurons)}"
        weights_list = []
        for layer in range(len(n_layer_neurons) - 1):
            weights_list.append(np.load(os.path.join(model_dir_path, f"weights_{layer}.npy")))
        info('Model loaded')

    if not test: return weights_list
    inputs, labels, inputs_test, labels_test = load_iris()
    correct = 0
    for i, (image, target) in (pbar:=tqdm(enumerate(zip(inputs,labels), start=1), total=len(inputs))):
        predicted = forward(weights_list, image)
        if predicted == target:
            correct += 1
        pbar.desc = f"Acc {correct/i*100:.2f}, predicted {predicted}, target {target}"
    info(f"Total correctly classified test set images: {correct/len(inputs)*100:.3f}")
    return weights_list

def run_test(cfg:CFG):
    log_name = f"{strftime('%m%d%H%M', localtime())}_{cfg.log_name}_{(num_steps-1)}_{'_'.join(str(l) for l in n_layer_neurons)}_delta{cfg.deltas}.log"
    logging.basicConfig(filename="log/" + log_name, level=logging.INFO)
    info(cfg)

    seed(cfg.seed)
    np.random.seed(cfg.seed)
    # torch.manual_seed(cfg.seed)
    # torch.use_deterministic_algorithms(True)

    weights_list = prepare_weights()
    
    # mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    # test_loader = DataLoader(mnist_test, batch_size=1, shuffle=True, drop_last=True)
    
    images, *_ = load_iris()
    
    info('Data is loaded')
    
    S = Solver()
    # spike_indicators = gen_spikes()
    spike_times = gen_spike_times()
    print(weights_list)
    weights = gen_weights(weights_list)
    
    # Load equations.
    eqn_path = f'eqn/eqn_{(num_steps-1)}_{"_".join([str(i) for i in n_layer_neurons])}.txt'
    if not load_expr or not os.path.isfile(eqn_path):
        node_eqns = gen_node_eqns(weights, spike_times)
        S.add(node_eqns)
        # if cfg.np_level == 1:
        #     node_eqns.extend(gen_dnp_v2(weights, spike_indicators, potentials))
        # elif cfg.np_level == 2:
        #     node_eqns.extend(gen_gnp(weights, spike_indicators))
        if save_expr:
            try:
                with open(eqn_path, 'w') as f:
                    f.write(S.sexpr())
                    info("Node equations are saved.")
            except:
                pdb.set_trace(header="Failed to save node eqns.")
    else:
        S.from_file(eqn_path)
    info("Solver is loaded.")

    samples_no_list:List[int] = []
    sampled_imgs:List[TImage] = []
    orig_preds:List[int] = []
    for sample_no in random_sample([*range(len(images))], k=num_procs):
        info(f"sample {sample_no} is drawn.")
        samples_no_list.append(sample_no)
        img:TImage = images[sample_no]
        sampled_imgs.append(img) # type: ignore
        orig_preds.append(forward(weights_list, img))
    info(f"Sampling is completed with {num_procs} samples.")
    # data, target = next(iter(test_loader))
    # inp = spikegen.rate(data, (num_steps-1)=(num_steps-1)) # type: ignore
    # op = net.forward(inp.view((num_steps-1), -1))[0]
    # label = int(torch.cat(op).sum(dim=0).argmax())
    # info(f'single input ran in {time.time()-tx} sec')

    # For each delta
    for delta in cfg.deltas:
        global check_sample
        def check_sample(sample:Tuple[int, TImage, int]):
            sample_no, img, orig_pred = sample
            orig_neuron = (orig_pred, 0)
            tx = time.time()
            
            # # Input property terms
            prop = []
            # max_delta_per_neuron = min(1, delta)
            # max_delta_per_neuron = delta
            input_layer = 0
            deltas_list = []
            delta_pos = IntVal(0)
            delta_neg = IntVal(0)
            def relu(x): return If(x>0, x, 0)
            for in_neuron in get_layer_neurons_iter(input_layer):
                ## Try to avoid using abs, it makes z3 extremely slow.
                delta_pos += relu(spike_times[in_neuron, input_layer] - int(img[in_neuron]))
                delta_neg += relu(int(img[in_neuron]) - spike_times[in_neuron, input_layer])
                # neuron_spktime_delta = (
                #     typecast(ArithRef,
                #              Abs(spike_times[in_neuron, input_layer] - int(img[in_neuron]))))
                # prop.append(neuron_spktime_delta <= max_delta_per_neuron)
                # deltas_list.append(neuron_spktime_delta)
                # prop.append(spike_times[in_neuron,input_layer] == int(img[in_neuron]))
                # print(img[in_neuron], end = '\t')
            prop.append((delta_pos + delta_neg) <= delta)
            # prop.append(Sum(deltas_list) <= delta)
            info(f"Inputs Property Done in {time.time() - tx} sec")

            # Output property
            tx = time.time()
            op = []
            last_layer = len(n_layer_neurons)-1
            for out_neuron in get_layer_neurons_iter(last_layer):
                if out_neuron != orig_neuron:
                    # It is equal to Not(spike_times[out_neuron, last_layer] >= spike_times[orig_neuron, last_layer]),
                    # we are checking p and Not(q) and q = And(q1, q2, ..., qn)
                    # so Not(q) is Or(Not(q1), Not(q2), ..., Not(qn))
                    op.append(
                        spike_times[out_neuron, last_layer] <= spike_times[orig_neuron, last_layer]
                    )
            op = Or(op)
            info(f'Output Property Done in {time.time() - tx} sec')

            tx = time.time()
            S_instance = deepcopy(S)
            info(f'Network Encoding read in {time.time() - tx} sec')
            S_instance.add(op)
            S_instance.add(prop)
            info(f'Total model ready in {time.time() - tx}')

            info('Query processing starts')
            # set_param(verbose=2)
            # set_param("parallel.enable", True)
            tx = time.time()
            result = S_instance.check()
            info(f'Checking done in time {time.time() - tx}')
            if result == sat:
                info(f'Not robust for sample {sample_no} and delta={delta}')
            elif result == unsat:
                info(f'Robust for sample {sample_no} and delta={delta}')
            else:
                info(f'Unknown at sample {sample_no} for reason {S_instance.reason_unknown()}')
            # pdb.set_trace()
            return result
        
        samples = zip(samples_no_list, sampled_imgs, orig_preds)
        if mp:
            with Pool(num_procs) as pool:
                pool.map(check_sample, samples)
                pool.close()
                pool.join()
        else:
            for sample in samples:
                check_sample(sample)

    info("")
