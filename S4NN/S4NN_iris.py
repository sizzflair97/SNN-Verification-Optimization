###########################################################################################
# The implementation of the supervied spiking neural network named S4NN presented in      #
# "S4NN: temporal backpropagation for spiking neural networks with one spike per neuron"  #
# by S. R. Kheradpisheh and T. Masquelier                                                 #
# The paper is availbale at: https://arxiv.org/abs/1910.09495.                            #  
#                                                                                         #
# To run the codes on cpu set GPU=False.                                                  #
# An ipynb version of the code compatible with Google coLab is also available.            # 
###########################################################################################

# Imports
from __future__ import division

# if you have a CUDA-enabled GPU the set the GPU flag as True
GPU=True

import os
import pdb
import time
from mnist import MNIST
from sklearn.datasets import load_iris
import numpy as np
from tqdm.auto import tqdm
import os.path as osp
if GPU:
    import cupy as cp  # You need to have a CUDA-enabled GPU to use this package!
else:
    cp=np

# Parameter setting
thr = [1, 100]  # The threshold of hidden and output neurons
lr = [0.05, 0.05]  # The learning rate of hidden and ouput neurons
lamda = [0.000001, 0.000001]  # The regularization penalty for hidden and ouput neurons
b = [5, 50]  # The upper bound of weight initializations for hidden and ouput neurons
a = [0, 0]  # The lower bound of weight initializations for hidden and ouput neurons
Nepoch = 100  # The maximum number of training epochs
NumOfClasses = 3  # Number of classes
Nlayers = 2  # Number of layers
NhidenNeurons = 10  # Number of hidden neurons
Dropout = [0, 0]
tmax = 5  # Simulatin time
gamma = 1  # The gamma parameter in the relative target firing calculation

# General settings
loading = False  # Set it as True if you want to load a pretrained model
LoadFrom = "weights.npy"  # The pretrained model
saving = True  # Set it as True if you want to save the trained model
best_perf = 0
Nnrn = [NhidenNeurons, NumOfClasses]  # Number of neurons at hidden and output layers
# cats = [4, 1, 0, 7, 9, 2, 3, 5, 8, 6]  # Reordering the categoriescats
cats = [*range(10)]  # Reordering the categories

# General variables
inputs_list = []  # To keep training images
labels_list = []  # To keep training labels
test_inputs_list = []  # To keep test images
test_labels_list = []  # To keep test labels
W = []  # To hold the weights of hidden and output layers
firingTime = []  # To hold the firing times of hidden and output layers
Spikes = []  # To hold the spike trains of hidden and output layers
X = []  # To be used in converting firing times to spike trains
target = cp.zeros([NumOfClasses])  # To keep the target firing times of current image
FiringFrequency = []  # to count number of spikes each neuron emits during an epoch

# loading MNIST dataset
iris = load_iris()
raw_data_max = np.quantile(iris.data, 0.95)  # which is np.max(iris.data)

raw_inputs, raw_labels = iris.data, iris.target # type: ignore
raw_inputs = np.array(raw_inputs)
for i in range(len(raw_labels)):
    if raw_labels[i] in cats:
        inputs_list.append(np.floor((raw_data_max - raw_inputs[i].reshape(4, 1)) * tmax / raw_data_max).astype(int))
        labels_list.append(cats.index(raw_labels[i]))

raw_inputs, raw_labels = iris.data, iris.target # type: ignore
raw_inputs = np.array(raw_inputs)
for i in range(len(raw_labels)):
    if raw_labels[i] in cats:
        # images_test.append(TTT[i].reshape(28,28).astype(int))
        test_inputs_list.append(np.floor((raw_data_max - raw_inputs[i].reshape(4, 1)) * tmax / raw_data_max).astype(int))
        test_labels_list.append(cats.index(raw_labels[i]))

del raw_inputs, raw_labels

TInput = np.ndarray[tuple[int,int,int], np.dtype[np.int_]]
TLabel = np.ndarray[tuple[int], np.dtype[np.int_]]
inputs:TInput = cp.asarray(inputs_list)
labels:TLabel = cp.asarray(labels_list)
test_inputs:TInput = cp.asarray(test_inputs_list)
test_labels:TLabel = cp.asarray(test_labels_list)

# Building the model
layerSize = [[inputs[0].shape[0], inputs[0].shape[1]], [NhidenNeurons, 1], [NumOfClasses, 1]]
x = cp.mgrid[0:layerSize[0][0], 0:layerSize[0][1]]  # To be used in converting raw image into a spike image
SpikeImage = cp.zeros((layerSize[0][0], layerSize[0][1], tmax + 1))  # To keep spike image

# Initializing the network
np.random.seed(0)
for layer in range(Nlayers):
    W.append(cp.asarray(
        (b[layer] - a[layer]) * np.random.random_sample((Nnrn[layer], layerSize[layer][0], layerSize[layer][1])) + a[
            layer]))
    firingTime.append(cp.asarray(np.zeros(Nnrn[layer])))
    Spikes.append(cp.asarray(np.zeros((layerSize[layer + 1][0], layerSize[layer + 1][1], tmax + 1))))
    X.append(cp.asarray(np.mgrid[0:layerSize[layer + 1][0], 0:layerSize[layer + 1][1]]))
if loading:
    if GPU:
        W = np.load(LoadFrom, allow_pickle=True)
    else:
        for i in range(len(W)):
            W[i] = cupy.asnumpy(W[i])
SpikeList = [SpikeImage] + Spikes

# Start learning
for epoch in tqdm(range(Nepoch), desc="Epoch"):
    start_time = time.time()
    correct = cp.zeros(NumOfClasses)
    FiringFrequency = cp.zeros((NhidenNeurons))

    # Start an epoch
    for iteration in (pbar:=tqdm(range(len(inputs)), leave=False)):
        # converting input image into spiking image
        SpikeImage[:, :, :] = 0
        SpikeImage[x[0], x[1], inputs[iteration]] = 1

        # Feedforward path
        for layer in range(Nlayers):
            Voltage = cp.cumsum(cp.tensordot(W[layer], SpikeList[layer]), 1)  # Computing the voltage
            Voltage[:, tmax] = thr[layer] + 1  # Forcing the fake spike
            firingTime[layer] = cp.argmax(Voltage > thr[layer], axis=1).astype(
                float) + 1  # Findign the first threshold crossing
            firingTime[layer][firingTime[layer] > tmax] = tmax  # Forcing the fake spike
            pbar.desc = f"{min(firingTime[layer])}"

            Spikes[layer][:, :, :] = 0
            Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(Nnrn[layer], 1).astype(
                int)] = 1  # converting firing times to spikes

        FiringFrequency = FiringFrequency + (firingTime[0] < tmax)  # FiringFrequency is used to find dead neurons

        # Computing the relative target firing times
        winner = np.argmin(firingTime[Nlayers - 1])
        minFiring = min(firingTime[layer])
        if minFiring == tmax:
            target[:] = minFiring
            target[labels[iteration]] = minFiring - gamma
            target = target.astype(int)
        else:
            target[:] = firingTime[layer][:]
            toChange = (firingTime[layer] - minFiring) < gamma
            target[toChange] = min(minFiring + gamma, tmax)
            target[labels[iteration]] = minFiring

        # Backward path
        layer = Nlayers - 1  # Output layer

        delta_o = (target - firingTime[layer]) / tmax  # Error in the ouput layer

        # Gradient normalization
        norm = cp.linalg.norm(delta_o)
        if (norm != 0):
            delta_o = delta_o / norm

        if Dropout[layer] > 0:
            firingTime[layer][cp.asarray(np.random.permutation(Nnrn[layer])[:Dropout[layer]])] = tmax

        # Updating hidden-output weights
        hasFired_o = firingTime[layer - 1] < firingTime[layer][:,
                                             cp.newaxis]  # To find which hidden neurons has fired before the ouput neurons
        W[layer][:, :, 0] -= (delta_o[:, cp.newaxis] * hasFired_o * lr[layer])  # Update hidden-ouput weights
        W[layer] -= lr[layer] * lamda[layer] * W[layer]  # Weight regularization

        # Backpropagating error to hidden neurons
        delta_h = (cp.multiply(delta_o[:, cp.newaxis] * hasFired_o, W[layer][:, :, 0])).sum(
            axis=0)  # Backpropagated errors from ouput layer to hidden layer

        layer = Nlayers - 2  # Hidden layer

        # Gradient normalization
        norm = cp.linalg.norm(delta_h)
        if (norm != 0):
            delta_h = delta_h / norm
        # Updating input-hidden weights
        hasFired_h = inputs[iteration] < firingTime[layer][:, cp.newaxis,
                                         cp.newaxis]  # To find which input neurons has fired before the hidden neurons
        W[layer] -= lr[layer] * delta_h[:, cp.newaxis, cp.newaxis] * hasFired_h  # Update input-hidden weights
        W[layer] -= lr[layer] * lamda[layer] * W[layer]  # Weight regularization

    # # Evaluating on test samples
    # correct = 0
    # for iteration in tqdm(range(len(images_test)), desc="Eval Testset", leave=False):
    #     SpikeImage[:, :, :] = 0
    #     SpikeImage[x[0], x[1], images_test[iteration]] = 1
    #     for layer in range(Nlayers):
    #         Voltage = cp.cumsum(cp.tensordot(W[layer], SpikeList[layer]), 1)
    #         Voltage[:, tmax] = thr[layer] + 1
    #         firingTime[layer] = cp.argmax(Voltage > thr[layer], axis=1).astype(float) + 1
    #         firingTime[layer][firingTime[layer] > tmax] = tmax
    #         Spikes[layer][:, :, :] = 0
    #         Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(Nnrn[layer], 1).astype(int)] = 1
    #     minFiringTime = firingTime[Nlayers - 1].min()
    #     if minFiringTime == tmax:
    #         V = np.argmax(Voltage[:, tmax - 3])
    #         if V == labels_test[iteration]:
    #             correct += 1
    #     else:
    #         # if firingTime[layer][labels_test[iteration]] == minFiringTime:
    #         if np.argmin(firingTime[-1]) == labels[iteration]:
    #             correct += 1
    # testPerf = correct / len(images_test)

    # Evaluating on train samples
    correct = 0
    for iteration in tqdm(range(len(inputs)), desc="Eval Trainset", leave=False):
        SpikeImage[:, :, :] = 0
        SpikeImage[x[0], x[1], inputs[iteration]] = 1
        for layer in range(Nlayers):
            Voltage = cp.cumsum(cp.tensordot(W[layer], SpikeList[layer]), 1)
            Voltage[:, tmax] = thr[layer] + 1
            firingTime[layer] = cp.argmax(Voltage > thr[layer], axis=1).astype(float) + 1
            firingTime[layer][firingTime[layer] > tmax] = tmax
            Spikes[layer][:, :, :] = 0
            Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(Nnrn[layer], 1).astype(int)] = 1
        minFiringTime = firingTime[Nlayers - 1].min()
        if minFiringTime == tmax:
            V = np.argmax(Voltage[:, tmax - 3])
            if V == labels[iteration]:
                correct += 1
        else:
            # if firingTime[layer][labels[iteration]] == minFiringTime:
            if np.argmin(firingTime[-1]) == labels[iteration]:
                correct += 1
        # if np.argmin(firingTime[-1]) == labels[iteration]:
        #     correct += 1
    trainPerf = correct / len(inputs)

    # print('epoch= ', epoch, 'Perf_train= ', trainPerf, 'Perf_test= ', testPerf)
    print('epoch= ', epoch, 'Perf_train= ', trainPerf)
    print("--- %s seconds ---" % (time.time() - start_time))

    # To save the weights
    if saving:
        # for layer in range(Nlayers):
        #     np.save(f"weights_{layer}", W[layer].get())
        # print("weights are saved.")
        # if testPerf > best_perf:
        #     np.save("weights_best", [W[i].get() for i in range(Nlayers)], allow_pickle=True)
        #     best_perf = testPerf
        save_dir_path = osp.join(
            osp.dirname(__file__),
            f"../models/{tmax}_4_{NhidenNeurons}_{NumOfClasses}")
        if not osp.isdir(save_dir_path):
            os.mkdir(save_dir_path)
        if trainPerf > best_perf:
            for layer in range(Nlayers):
                np.save(osp.join(save_dir_path,f"weights_{layer}"), W[layer].get())
            print("best weights are saved.")
            best_perf = trainPerf

    # To find and reset dead neurons
    ResetCheck = FiringFrequency < 0.001 * len(inputs)
    ToReset = [i for i in range(NhidenNeurons) if ResetCheck[i]]
    for i in ToReset:
        W[0][i] = cp.asarray((b[0] - a[0]) * np.random.random_sample((layerSize[0][0], layerSize[0][1])) + a[0])  # r
