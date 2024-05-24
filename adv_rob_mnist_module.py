# %%
from multiprocessing import Pipe, Pool
from random import sample as random_sample
from random import seed
import functools, time, logging, json, typing
from time import localtime, strftime
from typing import Any

from mnist import MNIST
from snntorch import spikegen
from snntorch import functional as SF
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from tqdm.auto import tqdm
from z3 import *
from collections import defaultdict
from utils import *
from utils.dictionary_mnist import *
from utils.encoding_mnist import *
from utils import MnistNet as Net

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
        firingTime[layer] = np.argmax(Voltage > threshold, axis=1).astype(float) + 1
        firingTime[layer][firingTime[layer] > num_steps] = num_steps
        Spikes[layer][:, :, :] = 0
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
        weights_list = np.load("mnist_weights_best.npy", allow_pickle=True)
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
    log_name = f"{strftime('%m%d%H%M', localtime())}_{cfg.log_name}.log"
    logging.basicConfig(filename="log/" + log_name, level=logging.INFO)
    info(cfg)

    seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)

    weights_list = prepare_weights()
    
    # mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    # test_loader = DataLoader(mnist_test, batch_size=1, shuffle=True, drop_last=True)
    
    images, labels, images_test, labels_test = load_mnist()
    
    info('Data is loaded')
    
    S = Solver()
    spike_indicators = gen_spikes()
    spike_times = gen_spike_times()
    weights = gen_weights(weights_list)
    if load_expr:
        S.from_file(rf'{location}\eqn\eqn_{num_steps}_{"_".join([str(i) for i in n_layer_neurons])}.txt')
    else:
        node_eqns = gen_node_eqns(weights, spike_indicators, spike_times)
        S.add(node_eqns)
        # if cfg.np_level == 1:
        #     node_eqns.extend(gen_dnp_v2(weights, spike_indicators, potentials))
        # elif cfg.np_level == 2:
        #     node_eqns.extend(gen_gnp(weights, spike_indicators))
        try:
            with open(rf'eqn\eqn_{num_steps}_{"_".join([str(i) for i in n_layer_neurons])}.txt', 'w') as f:
                f.write(S.sexpr())
                info("Node equations are saved.")
        except:
            pdb.set_trace(header="Failed to save node eqns.")
    info("Solver is loaded.")

    inp_vec:List[TImage] = []
    label_vec:List[int] = []
    for sample_no in random_sample([*range(len(images))], k=num_procs):
        img:TImage = images[sample_no]
        inp_vec.append(img) # type: ignore
        label_vec.append(forward(weights_list, img))
    info(f"Sampling is completed with {num_procs} samples.")
    # data, target = next(iter(test_loader))
    # inp = spikegen.rate(data, num_steps=num_steps) # type: ignore
    # op = net.forward(inp.view(num_steps, -1))[0]
    # label = int(torch.cat(op).sum(dim=0).argmax())
    # info(f'single input ran in {time.time()-tx} sec')

    # For each delta
    for delta in cfg.deltas:
        global check_sample
        def check_sample(sample:Tuple[TImage, int]):
            img, orig_pred = sample
            # Input property
            tx = time.time()
            s = [[] for _ in range(num_steps)]
            sv = [Int(f's_{i + 1}') for i in range(num_steps)]
            prop = []
            input_neurons = product(range(layer_shapes[0][0]), range(layer_shapes[0][1]))
            for in_neuron in input_neurons:
                prop.append(spike_times[in_neuron, 0] - img[in_neuron] <= delta)
                
            # for timestep, spike_train in enumerate(img):
            #     for i, spike in enumerate(spike_train.view(n_layer_neurons[0])):
            #         if spike == 1:
            #             s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 0.0, 1.0))
            #         else:
            #             s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 1.0, 0.0))
            # prop = [sv[i] == sum(s[i]) for i in range(num_steps)]
            # prop.append(sum(sv) <= delta)
            # info(prop[0])
            info(f"Inputs Property Done in {time.time() - tx} sec")

            # Output property
            tx = time.time()
            op = []
            orig_neuron = (orig_pred, 0)
            out_neurons = product(range(layer_shapes[-1][0]), range(layer_shapes[-1][1]))
            for out_neuron in out_neurons:
                if out_neuron == orig_neuron:
                    continue
                op.append(
                    spike_times[out_neuron, len(n_layer_neurons)-1] >= spike_times[orig_neuron, len(n_layer_neurons)-1] + 1
                )
            # intend_sum = sum([2 * spike_indicators[(label, 2, timestep + 1)] for timestep in range(num_steps)])
            # for t in range(n_layer_neurons[-1]):
            #     if t != label:
            #         op.append(
            #             Not(intend_sum > sum([2 * spike_indicators[(t, 2, timestep + 1)] for timestep in range(num_steps)]))
            #         )
            info(f'Output Property Done in {time.time() - tx} sec')

            tx = time.time()
            S = Solver()
            S.from_file(rf'{location}\eqn\eqn_{num_steps}_{"_".join([str(i) for i in n_layer_neurons])}.txt')
            # S.from_file(f'{location}\\eqn\\eqn_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.txt')
            info(f'Network Encoding read in {time.time() - tx} sec')
            # S.add(op + prop)
            S.add(op + prop)
            info(f'Total model ready in {time.time() - tx}')

            info('Query processing starts')
            tx = time.time()
            result = S.check()
            info(f'Checking done in time {time.time() - tx}')
            if result == sat:
                info(f'Not robust for sample and delta={delta}')
            else:
                info(f'Robust for sample and delta={delta}')
        
        samples = zip(inp_vec, label_vec)
        if mp:
            with Pool(num_procs) as pool:
                pool.map(check_sample, samples)
        for sample in samples:
            check_sample(sample)
            # # Input property
            # tx = time.time()
            # s = [[] for _ in range(num_steps)]
            # sv = [Int(f's_{i + 1}') for i in range(num_steps)]
            # prop = []
            # for timestep, spike_train in enumerate(inp):
            #     for i, spike in enumerate(spike_train.view(layers[0])):
            #         if spike == 1:
            #             s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 0.0, 1.0))
            #         else:
            #             s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 1.0, 0.0))
            # prop = [sv[i] == sum(s[i]) for i in range(num_steps)]
            # prop.append(sum(sv) <= dt)
            # # info(prop[0])
            # info(f"Inputs Property Done in {time.time() - tx} sec")

            # # Output property
            # tx = time.time()
            # op = []
            # intend_sum = sum([2 * spike_indicators[(label, 2, timestep + 1)] for timestep in range(num_steps)])
            # for t in range(layers[-1]):
            #     if t != label:
            #         op.append(
            #             Not(intend_sum > sum([2 * spike_indicators[(t, 2, timestep + 1)] for timestep in range(num_steps)]))
            #         )
            # info(f'Output Property Done in {time.time() - tx} sec')

            # tx = time.time()
            # S = Solver()
            # # S.from_file(f'{location}\\eqn\\eqn_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.txt')
            # info(f'Network Encoding read in {time.time() - tx} sec')
            # # S.add(op + prop)
            # S.add(op + node_eqns + prop)
            # info(f'Total model ready in {time.time() - tx}')

            # info('Query processing starts')
            # tx = time.time()
            # result = S.check()
            # info(f'Checking done in time {time.time() - tx}')
            # if result == sat:
            #     info(f'Not robust for sample and delta={dt}')
            # else:
            #     info(f'Robust for sample and delta={dt}')
        # # Input property
        # tx = time.time()
        # s = [[] for _ in range(num_steps)]
        # sv = [Int(f's_{i + 1}') for i in range(num_steps)]
        # prop = []
        # for timestep, spike_train in enumerate(inp):
        #     for i, spike in enumerate(spike_train.view(layers[0])):
        #         if spike == 1:
        #             s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 0.0, 1.0))
        #         else:
        #             s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 1.0, 0.0))
        # prop = [sv[i] == sum(s[i]) for i in range(num_steps)]
        # prop.append(sum(sv) <= dt)
        # # info(prop[0])
        # info(f"Inputs Property Done in {time.time() - tx} sec")

        # # Output property
        # tx = time.time()
        # op = []
        # intend_sum = sum([2 * spike_indicators[(label, 2, timestep + 1)] for timestep in range(num_steps)])
        # for t in range(layers[-1]):
        #     if t != label:
        #         op.append(
        #             Not(intend_sum > sum([2 * spike_indicators[(t, 2, timestep + 1)] for timestep in range(num_steps)]))
        #         )
        # info(f'Output Property Done in {time.time() - tx} sec')

        # tx = time.time()
        # S = Solver()
        # # S.from_file(f'{location}\\eqn\\eqn_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.txt')
        # info(f'Network Encoding read in {time.time() - tx} sec')
        # # S.add(op + prop)
        # S.add(op + node_eqns + prop)
        # info(f'Total model ready in {time.time() - tx}')

        # info('Query processing starts')
        # tx = time.time()
        # result = S.check()
        # info(f'Checking done in time {time.time() - tx}')
        # if result == sat:
        #     info(f'Not robust for sample and delta={dt}')
        # else:
        #     info(f'Robust for sample and delta={dt}')

    # delta_v = {d: 0 for d in cfg.deltas}
    # for delta in cfg.deltas:
    #     avt = 0
        
    #     global check_sample
    #     def check_sample(sample:Tuple[int, Tensor]) -> Tuple[float, int, str]:
    #         sample_no:int; sample_spike:Tensor;
    #         sample_no, sample_spike = sample
    #         res, label_var, control = forward_net(sample_spike.view(num_steps, -1), spike_indicators, assign+node_eqn+pot_init)
    #         if res in {'unsat','unknown'}:
    #             info(f'Could not find model at delta = {delta}, sample = {sample_no}')
    #             return -1, delta, res
    #         del res
            
    #         control = control.model()
    #         prop = gen_delta_reuse(cfg, sample_spike, spike_indicators, potentials, delta, control)
    #         # Output property
    #         #tx = time.time()
    #         op = []
    #         label = control[label_var].as_long() # type: ignore
            
    #         S = Solver()
    #         intend_sum = sum([2 * spike_indicators[(label, 2, timestep)] for timestep in range(1, num_steps+1)])
    #         for t in range(num_output):
    #             if t != label:
    #                 op.append(
    #                     Not(intend_sum > sum([2 * spike_indicators[(t, 2, timestep)] for timestep in range(1, num_steps+1)]))
    #                 )
    #         S.add(assign+node_eqn+pot_init+prop+op)
            
    #         tx = time.time()
    #         res:Literal["sat", "unsat", "unknown"] = str(S.check()) # type: ignore
    #         del S
    #         tss = time.time()-tx
    #         info(f'Completed for delta = {delta}, sample = {sample_no} in {tss} sec as {res}')
    #         return tss, delta, res
        
    #     sample_spks = [spikegen.rate(torch.tensor(sample, dtype=torch.float), num_steps=num_steps) # type: ignore
    #                    for sample in samples]
        
    #     if mp:
    #         with Pool(processes=num_procs) as pool:
    #             tss_lst = pool.map(check_sample, enumerate(sample_spks))
    #         for tss, delta, res in tss_lst:
    #             avt += tss
    #             delta_v[delta] += 1 if res == "unsat" else 0
    #         avt /= len(sample_spks)
    #     else:
    #         for sample_no, sample_spike in enumerate(sample_spks):
    #             tss, delta, res = check_sample((sample_no, sample_spike))
    #             avt = (avt*sample_no + tss)/(sample_no+1)
    #             delta_v[delta] += 1 if res == "unsat" else 0
    #     info(f'Completed for delta = {delta} with {delta_v[delta]} in avg time {avt} sec')
    #     del check_sample

    info("")



# %%
