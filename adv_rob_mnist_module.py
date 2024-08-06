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
def forward(weights_list:TWeightList, img:TImage, out_layer:np.ndarray|None=None):
    SpikeImage = np.zeros((28,28,num_steps+1))
    firingTime = []
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
    if out_layer is not None:
        out_layer[:] = firingTime[-1]
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
    info(f"Total correctly classified test set images: {correct/len(images)*100:.3f}")
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
    
    images, *_ = load_mnist()
    
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
                orig_time = img[loc_2d]
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
        orig_preds:List[int] = []
        for sample_no in random_sample([*range(len(images))], k=num_procs):
            info(f"sample {sample_no} is drawn.")
            samples_no_list.append(sample_no)
            img:TImage = images[sample_no]
            sampled_imgs.append(img) # type: ignore
            orig_preds.append(forward(weights_list, img))
        info(f"Sampling is completed with {num_procs} samples.")

        # For each delta
        for delta in cfg.deltas:
            global check_sample_numpy_backend
            def check_sample_numpy_backend(sample:Tuple[int, TImage, int]):
                sample_no, img, orig_pred = sample
                
                # Output property
                info('Query processing starts')
                tx = time.time()
                sat_flag:bool = False
                for pertd_img in search_perts(img, delta):
                    last_layer_spk_times = np.zeros((n_layer_neurons[-1],))
                    pert_pred = forward(weights_list, pertd_img, last_layer_spk_times)
                    not_orig_mask = [x for x in range(n_layer_neurons[-1]) if x!=pert_pred]
                    # It is equal to Not(spike_times[out_neuron, last_layer] >= spike_times[orig_neuron, last_layer]),
                    # we are checking p and Not(q) and q = And(q1, q2, ..., qn)
                    # so Not(q) is Or(Not(q1), Not(q2), ..., Not(qn))
                    if orig_pred != pert_pred:
                        info(f"Orig pred: {orig_pred}, Pert pred: {pert_pred}")
                    if np.any(last_layer_spk_times[not_orig_mask] <= last_layer_spk_times[orig_pred]):
                        sat_flag = True
                        break
                info(f"Checking done in time {time.time() - tx}")
                if sat_flag:
                    info(f"Not robust for sample {sample_no} and delta={delta}")
                elif sat_flag == False:
                    info(f"Robust for sample {sample_no} and delta={delta}")
                info("")
                return sat_flag
            
            samples = zip(samples_no_list, sampled_imgs, orig_preds)
            if mp:
                with Pool(num_procs) as pool:
                    pool.map(check_sample_numpy_backend, samples)
                    pool.close()
                    pool.join()
            else:
                for sample in samples:
                    check_sample_numpy_backend(sample)

        info("")


# %%
