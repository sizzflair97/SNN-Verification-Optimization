from copy import deepcopy
from multiprocessing import Pool
from random import sample as random_sample
from random import seed
import time, logging, typing
import numpy as np
from collections.abc import Generator
from z3 import *
from utils.dictionary_mnist import *
from utils.encoding_mnist import *
from utils.config import CFG
from utils.debug import info
from utils.mnist_net import forward, backward, test_weights, prepare_weights
from mnist import MNIST

def load_mnist() -> tuple[TImageBatch,TLabelBatch,TImageBatch,TLabelBatch]:
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
            images_test.append(np.floor((GrayLevels - Images[i].reshape(28, 28)) * (num_steps-1) / GrayLevels).astype(int))
            labels_test.append(cats.index(Labels[i]))

    del Images, Labels

    #images contain values within [0,num_steps]
    images = typing.cast(TImageBatch, np.asarray(images))
    labels = typing.cast(TLabelBatch, np.asarray(labels))
    images_test = typing.cast(TImageBatch, np.asarray(images_test))
    labels_test = typing.cast(TLabelBatch, np.asarray(labels_test))
    
    return images, labels, images_test, labels_test

def run_test(cfg:CFG):
    log_name = f"{cfg.log_name}_{num_steps}_{'_'.join(str(l) for l in n_layer_neurons)}_delta{cfg.deltas}.log"
    logging.basicConfig(filename="log/" + log_name, level=logging.INFO)
    info(cfg)

    seed(cfg.seed)
    np.random.seed(cfg.seed)

    weights_list = prepare_weights(subtype="mnist", load_data_func=load_mnist)
    images, labels, *_ = load_mnist()
    
    info('Data is loaded')
    
    if cfg.z3:
        S = Solver()
        spike_times = gen_spike_times()
        weights = gen_weights(weights_list)
        
        # Load equations.
        eqn_path = f'eqn/eqn_{num_steps}_{"_".join([str(i) for i in n_layer_neurons])}.txt'
        if not load_expr or not os.path.isfile(eqn_path):
            node_eqns = gen_node_eqns(weights, spike_times)
            S.add(node_eqns)
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

        samples_no_list:list[int] = []
        sampled_imgs:list[TImage] = []
        orig_preds:list[int] = []
        for sample_no in random_sample([*range(len(images))], k=cfg.num_samples):
            info(f"sample {sample_no} is drawn.")
            samples_no_list.append(sample_no)
            img:TImage = images[sample_no]
            sampled_imgs.append(img) # type: ignore
            orig_preds.append(forward(weights_list, img))
        info(f"Sampling is completed with {num_procs} samples.")

        # For each delta
        for delta in cfg.deltas:
            global check_sample
            def check_sample(sample:tuple[int, TImage, int]):
                sample_no, img, orig_pred = sample
                orig_neuron = (orig_pred, 0)
                tx = time.time()
                
                # Input property terms
                prop:list[BoolRef] = []
                input_layer = 0
                delta_pos = IntVal(0)
                delta_neg = IntVal(0)
                def relu(x:Any): return If(x>0, x, 0)
                for in_neuron in get_layer_neurons_iter(input_layer):
                    # Try to avoid using abs, it makes z3 extremely slow.
                    delta_pos += relu(spike_times[in_neuron, input_layer] - int(img[in_neuron]))
                    delta_neg += relu(int(img[in_neuron]) - spike_times[in_neuron, input_layer])
                prop.append((delta_pos + delta_neg) <= delta)
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
                S_instance.add(op) # type: ignore
                S_instance.add(prop) # type: ignore
                info(f'Total model ready in {time.time() - tx}')

                info('Query processing starts')
                # set_param(verbose=2)
                # set_param("parallel.enable", True)
                tx = time.time()
                result = S_instance.check() # type: ignore
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

        samples_no_list:list[int] = []
        sampled_imgs:list[TImage] = []
        sampled_labels:list[int] = []
        orig_preds:list[int] = []
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
            def check_sample_non_smt(sample:tuple[int, TImage, int, int],
                                     adv_train:bool=False,
                                     weights_list:TWeightList=weights_list):
                sample_no, img, label, orig_pred = sample
                
                info('Query processing starts')
                tx = time.time()
                sat_flag:bool = False
                adv_spk_times:list[list[np.ndarray[Any, np.dtype[np.float_]]]] = []
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
                        test_weights(updated_weights_list, load_mnist)
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
