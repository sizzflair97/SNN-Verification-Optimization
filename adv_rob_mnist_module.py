# %%
from copy import deepcopy
from multiprocessing import Pipe, Pool
from random import sample as random_sample
from random import seed
import time, logging, typing
from time import localtime, strftime
from typing import Any, Generator

from mnist import MNIST
# from snntorch import spikegen
# from snntorch import functional as SF
import numpy as np
import cupy as cp

# import torch
# import torch.nn as nn
# import snntorch as snn
from tqdm.auto import tqdm
from z3 import *
# from collections import defaultdict
from utils.dictionary_mnist import *
from utils.encoding_mnist import *
from utils.config import *
from utils.etc import *
from utils.debug_utils import *
import matplotlib.pyplot as plt

# transform = transforms.Compose([
#             transforms.Resize((28, 28)),
#             transforms.Grayscale(),
#             transforms.ToTensor(),
#             transforms.Normalize((0,), (1,))])

# cp.cuda.Device(0).use()
# np = cp

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
            images.append(np.floor((GrayLevels - Images[i].reshape(28, 28)) * (num_steps-1) / GrayLevels).astype(int))
            labels.append(cats.index(Labels[i]))
    Images, Labels = mndata.load_testing()
    Images = np.array(Images)
    for i in range(len(Labels)):
        if Labels[i] in cats:
            # images_test.append(TTT[i].reshape(28,28).astype(int))
            images_test.append(np.floor((GrayLevels - Images[i].reshape(28, 28)) * (num_steps-1) / GrayLevels).astype(int))
            labels_test.append(cats.index(Labels[i]))

    del Images, Labels

    #images contain values within [0,num_steps]
    images = typing.cast(TImageBatch, np.asarray(images))
    labels = typing.cast(TLabelBatch, np.asarray(labels))
    images_test = typing.cast(TImageBatch, np.asarray(images_test))
    labels_test = typing.cast(TLabelBatch, np.asarray(labels_test))
    
    return images, labels, images_test, labels_test

mgrid = np.mgrid[0:28, 0:28]
def forward(weights_list:TWeightList,
            img:TImage,
            layers_firing_time_return:list[np.ndarray]|None=None):
    # Return by reference at firing_time_ptr.
    SpikeImage = np.zeros((28,28,num_steps+1))
    firingTime:list[np.ndarray] = []
    Spikes = []
    X = []
    for layer, neuron_of_layer in enumerate(n_layer_neurons[1:]):
        firingTime.append(np.asarray(np.zeros(neuron_of_layer)))
        Spikes.append(np.asarray(np.zeros((layer_shapes[layer + 1][0], layer_shapes[layer + 1][1], num_steps))))
        X.append(np.asarray(np.mgrid[0:layer_shapes[layer + 1][0], 0:layer_shapes[layer + 1][1]]))
    
    SpikeList = [SpikeImage] + Spikes
    
    SpikeImage[mgrid[0], mgrid[1], img] = 1
    for layer in range(len(n_layer_neurons)-1):
        Voltage = np.cumsum(np.tensordot(weights_list[layer], SpikeList[layer]), 1)
        Voltage[:, num_steps-1] = threshold + 1
        firingTime[layer] = np.argmax(Voltage > threshold, axis=1).astype(float) + 1
        # in layer 0, max time is num_steps-1, but in layer 1, max time is num_steps, so we clamp it.
        firingTime[layer][firingTime[layer] > num_steps-1] = num_steps-1
        Spikes[layer][...] = 0
        Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(n_layer_neurons[layer+1], 1).astype(int)] = 1 # All neurons spike only once.
    
    # print(np.max(firingTime))
    # minFiringTime = firingTime[len(n_layer_neurons)-1 - 1].min()
    # if minFiringTime == num_steps:
    #     V = np.argmax(Voltage[:, num_steps - 3])
    #     # V = 0
    # else:
    V = int(np.argmin(firingTime[-1]))
    if layers_firing_time_return is not None:
        layers_firing_time_return[:] = firingTime[:]
    return V

gamma = 2
target = np.zeros((n_layer_neurons[-1],))
def backward(weights_list:TWeightList,
             layers_firing_time:list[np.ndarray],
             image:TImage,
             label:int):
    weights_list = [x.copy() for x in weights_list]
    global target
    # Computing the relative target firing times
    min_firing = min(layers_firing_time[-1])
    if min_firing == num_steps - 1:
        target[:] = min_firing
        target[label] = min_firing - gamma
        target = target.astype(int)
    else:
        target[:] = layers_firing_time[-1][:]
        to_change = (layers_firing_time[-1] - min_firing) < gamma
        target[to_change] = min(min_firing + gamma, num_steps - 1)
        target[label] = min_firing
    
    # Backward path
    layer = len(n_layer_neurons) - 2  # Output layer
    
    delta_o = (target - layers_firing_time[layer]) / (num_steps-1)  # Error in the ouput layer

    # Gradient normalization
    norm = np.linalg.norm(delta_o)
    if (norm != 0):
        delta_o = delta_o / norm

    # Updating hidden-output weights
    hasFired_o = layers_firing_time[layer - 1] < layers_firing_time[layer][:,
                                            np.newaxis]  # To find which hidden neurons has fired before the ouput neurons
    weights_list[layer][:, :, 0] -= (delta_o[:, np.newaxis] * hasFired_o * lr[layer])  # Update hidden-ouput weights
    weights_list[layer] -= lr[layer] * lamda[layer] * weights_list[layer]  # Weight regularization

    # Backpropagating error to hidden neurons
    delta_h = (np.multiply(delta_o[:, np.newaxis] * hasFired_o, weights_list[layer][:, :, 0])).sum(
        axis=0)  # Backpropagated errors from ouput layer to hidden layer

    layer = len(n_layer_neurons) - 3  # Hidden layer
    
    # Gradient normalization
    norm = np.linalg.norm(delta_h)
    if (norm != 0):
        delta_h = delta_h / norm
    # Updating input-hidden weights
    hasFired_h = image < layers_firing_time[layer][:, np.newaxis,
                                        np.newaxis]  # To find which input neurons has fired before the hidden neurons
    weights_list[layer] -= lr[layer] * delta_h[:, np.newaxis, np.newaxis] * hasFired_h  # Update input-hidden weights
    weights_list[layer] -= lr[layer] * lamda[layer] * weights_list[layer]  # Weight regularization
    
    return weights_list

def test_weights(weights_list:TWeightList) -> None:
    images, labels, *_ = load_mnist()
    correct = 0
    for i, (image, target) in (pbar:=tqdm(enumerate(zip(images,labels), start=1), total=len(images))):
        predicted = forward(weights_list, image)
        if predicted == target:
            correct += 1
        pbar.desc = f"Acc {correct/i*100:.2f}, predicted {predicted}, target {target}"
    info(f"Total correctly classified test set images: {correct/len(images)*100:.3f}")

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

    if test:
        test_weights(weights_list)
    return weights_list

def run_test(cfg:CFG):
    log_name = f"{cfg.log_name}_{num_steps}_{'_'.join(str(l) for l in n_layer_neurons)}_delta{cfg.deltas}.log"
    logging.basicConfig(filename="log/" + log_name, level=logging.INFO)
    info(cfg)

    seed(cfg.seed)
    np.random.seed(cfg.seed)
    # torch.manual_seed(cfg.seed)
    # torch.use_deterministic_algorithms(True)

    weights_list = prepare_weights()
    
    # mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    # test_loader = DataLoader(mnist_test, batch_size=1, shuffle=True, drop_last=True)
    
    images, labels, *_ = load_mnist()
    
    info('Data is loaded')
    
    if cfg.z3:
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
        for sample_no in random_sample([*range(len(images))], k=cfg.num_samples):
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
                info(f"Checking done in time {time.time() - tx}")
                if result == sat:
                    info(f"Not robust for sample {sample_no} and delta={delta}")
                elif result == unsat:
                    info(f"Robust for sample {sample_no} and delta={delta}")
                else:
                    info(f"Unknown at sample {sample_no} for reason {S_instance.reason_unknown()}")
                info("")
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
    else:
        # Recursively find available adversarial attacks.
        def search_perts(img:TImage, delta:int, loc:int=0, pert:TImage|None=None) -> Generator[TImage,None,None]:
            # Initial case
            if pert is None:
                pert = np.zeros_like(img, dtype=img.dtype)
                
            # Last case
            if delta == 0:
                yield img + pert
            # Search must be terminated at the end of image.
            elif loc < n_layer_neurons[0]:
                loc_2d = (loc//layer_shapes[0][1], loc%layer_shapes[0][1])
                orig_time = int(img[loc_2d])
                # Clamp delta at current location
                available_deltas = range(-min(orig_time, delta), min((num_steps-1)-orig_time, delta)+1)
                for delta_at_neuron in available_deltas:
                    new_pert = pert.copy()
                    new_pert[loc_2d] += delta_at_neuron
                    yield from search_perts(img,
                                          delta-abs(delta_at_neuron),
                                          loc+1,
                                          new_pert)

        samples_no_list:List[int] = []
        sampled_imgs:List[TImage] = []
        sampled_labels:List[int] = []
        orig_preds:List[int] = []
        for sample_no in random_sample([*range(len(images))], k=cfg.num_samples):
            info(f"sample {sample_no} is drawn.")
            samples_no_list.append(sample_no)
            img:TImage = images[sample_no]
            label = labels[sample_no]
            sampled_imgs.append(img)
            sampled_labels.append(label)
            orig_preds.append(forward(weights_list, img))
        info(f"Sampling is completed with {num_procs} samples.")

        # For each delta
        for delta in cfg.deltas:
            global check_sample_non_smt
            def check_sample_non_smt(sample:Tuple[int, TImage, int, int],
                                     adv_train:bool=False,
                                     weights_list:TWeightList=weights_list):
                sample_no, img, label, orig_pred = sample
                
                info('Query processing starts')
                tx = time.time()
                sat_flag:bool = False
                adv_spk_times = []
                n_counterexamples = 0
                for pertd_img in search_perts(img, delta):
                    pert_pred = forward(weights_list, pertd_img, spk_times:=[])
                    adv_spk_times.append(spk_times)
                    last_layer_spk_times = spk_times[-1]
                    not_orig_mask = [x for x in range(n_layer_neurons[-1]) if x!=pert_pred]
                    # It is equal to Not(spike_times[out_neuron, last_layer] >= spike_times[orig_neuron, last_layer]),
                    # we are checking p and Not(q) and q = And(q1, q2, ..., qn)
                    # so Not(q) is Or(Not(q1), Not(q2), ..., Not(qn))
                    if np.any(last_layer_spk_times[not_orig_mask] <= last_layer_spk_times[orig_pred]):
                        sat_flag = True
                        if not adv_train:
                            break
                        n_counterexamples += 1
                info(f"Checking done in time {time.time() - tx}")
                if sat_flag:
                    if adv_train:
                        info(f"Not robust for sample {sample_no} and delta={delta} with {n_counterexamples} counterexamples.")
                        info(f"Start adversarial training.")
                        updated_weights_list = weights_list
                        for spk_times in adv_spk_times:
                            updated_weights_list = backward(updated_weights_list, spk_times, img, label)
                        test_weights(updated_weights_list)
                        new_orig_pred = forward(updated_weights_list, img)
                        new_sample = (*sample[:3],new_orig_pred)
                        info(f"Completed adversarial training. Checking robustness again.")
                        check_sample_non_smt(new_sample, adv_train=False, weights_list=updated_weights_list)
                    else:
                        info(f"Not robust for sample {sample_no} and delta={delta}")
                elif sat_flag == False:
                    info(f"Robust for sample {sample_no} and delta={delta}.")
                info("")
                return sat_flag
            
            samples = zip(samples_no_list, sampled_imgs, sampled_labels, orig_preds)
            if mp:
                with Pool(num_procs) as pool:
                    pool.map(check_sample_non_smt, samples)
                    pool.close()
                    pool.join()
            else:
                for sample in samples:
                    check_sample_non_smt(sample)

        info("")


# %%
