import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import defaultdict
from z3 import *
from tqdm.auto import tqdm
import time

import matplotlib.pyplot as plt
import numpy as np
import itertools

batch_size = 128
data_path = '../data/mnist/'
location = '..'
neurons_in_layers = [28*28, 20, 10]
num_steps = 10
beta = 0.95
dtype = torch.float
delta = [1]

class Net(nn.Module):
    def __init__(self, layers=(2, 2, 2), loss_value=1.0):
        super().__init__()

        # Initialize layers
        self.layer_count = layers
        self.fc_layers = nn.ModuleList()
        self.leaky_layers = nn.ModuleList()

        for layer_num in range(len(layers)-1):
            self.fc_layers.append(nn.Linear(layers[layer_num], layers[layer_num+1], bias=False))
            self.leaky_layers.append(snn.Leaky(beta=loss_value))

    def merge(self, layer_no, neuron_list):
        n = layer_no - 1
        weights1, weights2 = self.layers[n][0].weight.data, self.layers[n+1][0].weight.data.T

        merged_input_weights = []
        merged_output_weights = []

        new_weights1 = []
        new_weights2 = []

        for i in range(len(weights1)):
            if i in neuron_list:
                merged_input_weights.append(weights1[i])
            else:
                new_weights1.append(weights1[i])

        for i in range(len(weights2)):
            if i in neuron_list:
                merged_output_weights.append(weights2[i])
            else:
                new_weights2.append(weights2[i])

        if len(self.layers[n][1].beta.shape) == 0:
            new_thresh = self.layers[n][1].beta
        else:
            new_thresh = []
            merge_calc = []

            for i in range(self.layers[n][1].beta.shape[0]):
                if i in neuron_list:
                    merge_calc.append(self.layers[n][1].beta[i])
                else:
                    new_thresh.append(self.layers[n][1].beta[i])

            new_thresh.append(np.sum(merge_calc) / len(merge_calc))
            new_thresh = torch.tensor(new_thresh)

        new_weights1.append(np.sum(merged_input_weights) / len(merged_input_weights))
        new_weights2.append(np.sum(merged_output_weights))

        layer_count = self.layer_count
        layer_count[layer_no] -= (len(neuron_list)-1)

        new_net = Net(layer_count)
        for i in range(len(self.layers)-1):
            if i != n and i != n+1:
                new_net.layers[i] = self.layers[i]
            else:
                new_net.layers[i][0].weight.data = torch.stack(new_weights1)
                new_net.layers[i+1][0].weight.data = torch.stack(new_weights2).T
                new_net.layers[i][1].beta = new_thresh

        return new_net

    def forward(self, x, return_all=False):

        # Initialize hidden states at t=0
        mem_list = [
            layer.init_leaky() for layer in self.leaky_layers
        ]

        list_of_spikes = []
        list_of_potentials = []
        last_spikes = []
        last_pot = []
        for step in range(num_steps):
            input_spikes = x[step]

            output_spikes = []
            output_potentials = []

            for num, (layer, leak) in enumerate(zip(self.fc_layers, self.leaky_layers)):
                cur = layer(input_spikes)
                spk, mem_list[num] = leak(cur, mem_list[num])

                output_spikes.append(spk)
                output_potentials.append(mem_list[num])
                input_spikes = cur

            last_spikes.append(output_spikes[-1])
            last_pot.append(output_potentials[-1])

            list_of_spikes.append(output_spikes)
            list_of_potentials.append(output_potentials)

        if return_all:
            return list_of_spikes, list_of_potentials
        else:
            return last_spikes, last_pot