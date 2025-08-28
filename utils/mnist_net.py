import os
from typing import Any, LiteralString
from collections.abc import Callable, Generator, Iterable
from tqdm.auto import tqdm

from .config import CFG
from .dictionary_mnist import *
from .debug import info

mgrid = np.mgrid[0:28, 0:28]
def forward(cfg:CFG,
            weights_list:TWeightList,
            img:TImage,
            layers_firing_time_return:list[np.ndarray[Any, np.dtype[np.float_]]]|None = None,
            voltage_return:list[np.ndarray[Any, np.dtype[np.float_]]]|None = None) -> int:
    # Return by reference at firing_time_ptr.
    n_layer_neurons = cfg.n_layer_neurons
    layer_shapes = cfg.layer_shapes
    SpikeImage = np.zeros((28,28,num_steps+1))
    firingTime:list[np.ndarray[Any, np.dtype[np.float_]]] = []
    Spikes:list[np.ndarray[Any, np.dtype[np.float_]]] = []
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
        if voltage_return is not None:
            voltage_return.append(Voltage)
        firingTime[layer] = np.argmax(Voltage > threshold, axis=1).astype(float) + 1
        # in layer 0, max time is num_steps-1, but in layer 1, max time is num_steps, so we clamp it.
        firingTime[layer][firingTime[layer] > num_steps-1] = num_steps-1
        Spikes[layer][...] = 0
        Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(n_layer_neurons[layer+1], 1).astype(int)] = 1 # All neurons spike only once.
    
    V = int(np.argmin(firingTime[-1]))
    if layers_firing_time_return is not None:
        layers_firing_time_return[:] = firingTime[:]
        
    return V

gamma = 2
def backward(cfg:CFG,
             weights_list:TWeightList,
             layers_firing_time:list[np.ndarray[Any, np.dtype[np.float_]]],
             image:TImage,
             label:int) -> TWeightList:
    n_layer_neurons = cfg.n_layer_neurons
    weights_list = [x.copy() for x in weights_list]
    target = np.zeros((n_layer_neurons[-1],))
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


datafunc = Callable[[], tuple[TImageBatch,TLabelBatch,TImageBatch,TLabelBatch]]
sample_typing = tuple[np.ndarray[tuple[Literal[28],Literal[28]], np.dtype[np.int_]], int]

def test_weights(cfg:CFG, weights_list:TWeightList,
                 load_data_func:datafunc) -> None:
    images, labels, *_ = load_data_func()
    correct = 0
    pbar:Iterable[sample_typing] = tqdm(zip(images,labels), total=len(images))
    for i, (image, target) in enumerate(pbar, start=1):
        predicted = forward(cfg, weights_list, image)
        if predicted == target:
            correct += 1
        pbar.desc = f"Acc {correct/i*100:.2f}, predicted {predicted}, target {target}"
    info(f"Total correctly classified test set images: {correct/len(images)*100:.3f}")



def prepare_weights(cfg:CFG, subtype:Literal["mnist", "fmnist"],load_data_func:datafunc|None = None) -> TWeightList:
    n_layer_neurons = cfg.n_layer_neurons
    if train:
        raise NotImplementedError("The model must be trained from S4NN.")
    else:
        subtype_prefix = [] if subtype == "mnist" else ["fm"]
        model_dir_path = "models/" + "_".join(subtype_prefix + [f"{num_steps}", *(str(i) for i in n_layer_neurons)])
        f"models/fm_{num_steps}_{'_'.join(str(i) for i in n_layer_neurons)}"
        weights_list:TWeightList = []
        for layer in range(len(n_layer_neurons) - 1):
            weights_list.append(np.load(os.path.join(model_dir_path, f"weights_{layer}.npy")))
        info('Model loaded')

    if test:
        assert load_data_func is not None, "Data loading function must be provided for testing."
        test_weights(cfg, weights_list, load_data_func)
    return weights_list