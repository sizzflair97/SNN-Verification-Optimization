# %%
from copy import deepcopy
from multiprocessing import Pipe, Pool
from random import sample as random_sample
from random import seed
import time, logging, typing
from time import localtime, strftime
from typing import Any

from mnist import MNIST
# from snntorch import spikegen
# from snntorch import functional as SF
import numpy as np
# import torch
# import torch.nn as nn
# import snntorch as snn
from tqdm.auto import tqdm
from z3 import *
# from collections import defaultdict
from utils import *
from utils.dictionary_mnist import *
from utils.encoding_mnist import *
# from utils import MnistNet as Net

# transform = transforms.Compose([
#             transforms.Resize((28, 28)),
#             transforms.Grayscale(),
#             transforms.ToTensor(),
#             transforms.Normalize((0,), (1,))])


def load_mnist() -> Tuple[TImageBatch,TLabelBatch,TImageBatch,TLabelBatch]:
    # Parameter setting
    GrayLevels = 255  # Image GrayLevels
    cats = [*range(10)]

    # General variables
    images = []  # To keep training images
    labels = []  # To keep training labels
    images_test = []  # To keep test images
    labels_test = []  # To keep test labels

    # loading MNIST dataset
    mndata = MNIST('data/mnist/MNIST/raw/')
    # mndata.gz = False

    Images, Labels = mndata.load_training()
    Images = np.array(Images)
    for i in range(len(Labels)):
        if Labels[i] in cats:
            images.append(np.floor((GrayLevels - Images[i].reshape(28, 28)) * num_steps / GrayLevels).astype(int))
            labels.append(cats.index(Labels[i]))
    Images, Labels = mndata.load_testing()
    Images = np.array(Images)
    for i in range(len(Labels)):
        if Labels[i] in cats:
            # images_test.append(TTT[i].reshape(28,28).astype(int))
            images_test.append(np.floor((GrayLevels - Images[i].reshape(28, 28)) * num_steps / GrayLevels).astype(int))
            labels_test.append(cats.index(Labels[i]))

    del Images, Labels

    #images contain values within [0,num_steps]
    images = typing.cast(TImageBatch, np.asarray(images))
    labels = typing.cast(TLabelBatch, np.asarray(labels))
    images_test = typing.cast(TImageBatch, np.asarray(images_test))
    labels_test = typing.cast(TLabelBatch, np.asarray(labels_test))
    
    return images, labels, images_test, labels_test

mgrid = np.mgrid[0:28, 0:28]
def forward(weights_list:TWeightList, img:TImage):
    SpikeImage = np.zeros((28,28,num_steps+1))
    firingTime = []
    Spikes = []
    X = []
    for layer, neuron_of_layer in enumerate(n_layer_neurons[1:]):
        firingTime.append(np.asarray(np.zeros(neuron_of_layer)))
        Spikes.append(np.asarray(np.zeros((layer_shapes[layer + 1][0], layer_shapes[layer + 1][1], num_steps + 1))))
        X.append(np.asarray(np.mgrid[0:layer_shapes[layer + 1][0], 0:layer_shapes[layer + 1][1]]))
    
    SpikeList = [SpikeImage] + Spikes
    
    SpikeImage[mgrid[0], mgrid[1], img] = 1
    for layer in range(len(n_layer_neurons)-1):
        Voltage = np.cumsum(np.tensordot(weights_list[layer], SpikeList[layer]), 1)
        Voltage[:, num_steps] = threshold + 1
        firingTime[layer] = np.argmax(Voltage > threshold, axis=1).astype(float)
        firingTime[layer][firingTime[layer] > num_steps-1] = num_steps
        Spikes[layer][...] = 0
        Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(n_layer_neurons[layer+1], 1).astype(int)] = 1 # All neurons spike only once.
    # minFiringTime = firingTime[len(n_layer_neurons)-1 - 1].min()
    # if minFiringTime == num_steps:
    #     V = np.argmax(Voltage[:, num_steps - 3])
    #     # V = 0
    # else:
    V = int(np.argmin(firingTime[-1])) # acc is about 96.80%
    return V

def prepare_weights() -> TWeightList:
    if train:
        raise NotImplementedError("The model must be trained from S4NN.")
    else:
        # weights_list = np.load("mnist_weights_best.npy", allow_pickle=True)
        model_dir_path = f"models/{num_steps}_{'_'.join(str(i) for i in n_layer_neurons)}"
        weights_list = []
        for layer in range(len(n_layer_neurons) - 1):
            weights_list.append(np.load(os.path.join(model_dir_path, f"weights_{layer}.npy")))
        info('Model loaded')

    if not test: return weights_list
    images, labels, images_test, labels_test = load_mnist()
    correct = 0
    for i, (image, target) in (pbar:=tqdm(enumerate(zip(images,labels), start=1), total=len(images))):
        predicted = forward(weights_list, image)
        if predicted == target:
            correct += 1
        pbar.desc = f"Acc {correct/i*100:.2f}, predicted {predicted}, target {target}"
    info(f"Total correctly classified test set images: {correct}/{len(images)}")
    return weights_list

def run_test(cfg:CFG):
    log_name = f"{strftime('%m%d%H%M', localtime())}_{cfg.log_name}_{num_steps}_{'_'.join(str(l) for l in n_layer_neurons)}_delta{cfg.deltas}.log"
    logging.basicConfig(filename="log/" + log_name, level=logging.INFO)
    info(cfg)

    seed(cfg.seed)
    np.random.seed(cfg.seed)
    # torch.manual_seed(cfg.seed)
    # torch.use_deterministic_algorithms(True)

    weights_list = prepare_weights()
    
    # mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    # test_loader = DataLoader(mnist_test, batch_size=1, shuffle=True, drop_last=True)
    
    images, labels, images_test, labels_test = load_mnist()
    
    info('Data is loaded')
    
    S = Solver()
    # spike_indicators = gen_spikes()
    spike_times = gen_spike_times()
    weights = gen_weights(weights_list)
    
    # Load equations.
    eqn_path = f'eqn/eqn_{num_steps}_{"_".join([str(i) for i in n_layer_neurons])}.txt'
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
    # inp = spikegen.rate(data, num_steps=num_steps) # type: ignore
    # op = net.forward(inp.view(num_steps, -1))[0]
    # label = int(torch.cat(op).sum(dim=0).argmax())
    # info(f'single input ran in {time.time()-tx} sec')

    # For each delta
    for delta in cfg.deltas:
        global check_sample
        def check_sample(sample:Tuple[int, TImage, int]):
            sample_no, img, orig_pred = sample
            orig_neuron = (orig_pred, 0)
            tx = time.time()
            # # Prior knowledge terms
            # input_neurons = product(range(layer_shapes[0][0]), range(layer_shapes[0][1]))
            # prior_knowledge = [[],[]]
            # for in_neuron in input_neurons:
            #     spiketime_equality = (
            #         typecast(BoolRef,
            #                  spike_times[in_neuron,0] == int(img[in_neuron])))
            #     prior_knowledge[0].append(spiketime_equality)
            # out_neurons = product(range(layer_shapes[-1][0]), range(layer_shapes[-1][1]))
            # for out_neuron in out_neurons:
            #     if out_neuron != orig_neuron:
            #         prior_knowledge[1].append(
            #             spike_times[out_neuron, len(n_layer_neurons)-1] >= spike_times[orig_neuron, len(n_layer_neurons)-1] + 1
            #         )
            # prior_knowledge = Implies(And(prior_knowledge[0]), And(prior_knowledge[1]))
            
            # # Input property terms
            # prop = [prior_knowledge]
            prop = []
            # max_delta_per_neuron = min(1, delta)
            max_delta_per_neuron = delta
            input_neurons = product(range(layer_shapes[0][0]), range(layer_shapes[0][1]))
            input_layer = 0
            deltas_list = []
            for in_neuron in input_neurons:
                neuron_spktime_delta = (
                    typecast(ArithRef,
                             Abs(spike_times[in_neuron, input_layer] - int(img[in_neuron]))))
                prop.append(neuron_spktime_delta <= max_delta_per_neuron)
                deltas_list.append(neuron_spktime_delta)
                # prop.append(spike_times[in_neuron,0] == int(img[in_neuron]))
            prop.append(Sum(deltas_list) <= delta)
            info(f"Inputs Property Done in {time.time() - tx} sec")

            # Output property
            tx = time.time()
            op = []
            out_neurons = product(range(layer_shapes[-1][0]), range(layer_shapes[-1][1]))
            last_layer = len(n_layer_neurons)-1
            for out_neuron in out_neurons:
                if out_neuron != orig_neuron:
                    # It is equal to Not(spike_times[out_neuron, last_layer] >= spike_times[orig_neuron, last_layer]),
                    # we are checking p and Not(q) and q = And(q1, q2, ..., qn)
                    # so Not(q) is Or(Not(q1), Not(q2), ..., Not(qn))
                    op.append(
                        spike_times[out_neuron, last_layer] < spike_times[orig_neuron, last_layer]
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
            set_param(verbose=2)
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



# %%
