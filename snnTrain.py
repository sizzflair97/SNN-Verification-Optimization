import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import defaultdict
from z3 import *
import time

import matplotlib.pyplot as plt
import numpy as np
import itertools

batch_size = 128
data_path = '/data/mnist'
location = 'C:\\Users\\soham\\PycharmProjects\\Z3py'
neurons_in_layers = [28*28, 100, 10]
num_steps = 10
beta = 0.95
dtype = torch.float

transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)


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

if __name__ == '__main__':


    # Load the network onto CUDA if available
    net = Net(neurons_in_layers, loss_value=beta)

    load = False
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

    num_epochs = 1
    loss_hist = []
    test_loss_hist = []
    counter = 0
    if not load:
        # Outer training loop
        for epoch in range(num_epochs):
            iter_counter = 0
            train_batch = iter(train_loader)

            # Minibatch training loop
            for data, targets in train_batch:
                data = data
                targets = targets

                spike_data = spikegen.rate(data, num_steps=num_steps)

                # forward pass
                net.train()
                spk_rec, mem_rec = net(spike_data.view(num_steps, batch_size, -1), return_all=True)

                # initialize the loss & sum over time
                loss_val = torch.zeros((1), dtype=dtype)
                for step in range(num_steps):
                    loss_val += loss(mem_rec[step][-1], targets)

                # Gradient calculation + weight update
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                # Store loss history for future plotting
                loss_hist.append(loss_val.item())

        #torch.save(net, f'/Models/model_{num_hidden}.pth')
        torch.save(net, f'{location}\\Models\\model_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.pth')
        print("Model Saved")
    else:
        net = torch.load(f'{location}\\Models\\model_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.pth')
        print('Model loaded')

    total = 0
    correct = 0
    test_batch_size = 1
    # drop_last switched to False to keep all samples
    test_loader = DataLoader(mnist_test, batch_size=test_batch_size, shuffle=True, drop_last=False)
    test_log = True
    tt = []
    if test_log:
        with torch.no_grad():
            net.eval()
            c = 1
            for data, targets in test_loader:
                t = time.time()
                data = data
                targets = targets

                test_spike_data = spikegen.rate(data, num_steps=num_steps)

                # forward pass
                test_spk, _ = net(test_spike_data.view(num_steps, test_batch_size, -1))
                # calculate total accuracy
                predicted = torch.cat(test_spk).sum(dim=0).argmax()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                tt.append(time.time()-t)
                c += 1

        print(f"Total correctly classified test set images: {correct}/{total} in avg.time {sum(tt)/len(tt)}")
    #=============================================

    test_inputs = 20

    spike_c = {}
    for j in range(1, len(neurons_in_layers)):
        for i in range(neurons_in_layers[j]):
            spike_c[(i, j)] = 1

    for _ in range(test_inputs):
        inp = torch.tensor(np.random.randint(0, 2, (num_steps, neurons_in_layers[0])), dtype=torch.float)
        a, b = net(inp, return_all=True)
        for ts in a:
            for j, spi in enumerate(ts):
                for i, val in enumerate(spi):
                    if i == 1:
                        spike_c[(i, j)] = 0

    #=============================================

    layers = neurons_in_layers
    input_real = False

    spike_indicators = {}
    for t in range(num_steps):
        for j, m in enumerate(layers):
            if j == 0:
                if input_real:
                    for i in range(m):
                        spike_indicators[(i, j, t + 1)] = Real(f'x_{i}_{j}_{t + 1}')
                else:
                    for i in range(m):
                        spike_indicators[(i, j, t + 1)] = Bool(f'x_{i}_{j}_{t + 1}')
            else:
                for i in range(m):
                    spike_indicators[(i, j, t+1)] = Bool(f'x_{i}_{j}_{t+1}')

    print("Spikes created")
    potentials = {}
    for t in range(1, num_steps+1):
        for j, m in enumerate(layers):
            if j == 0:
                continue
            for i in range(m):
                potentials[(i, j, t)] = Real(f'P_{i}_{j}_{t}')

    print("Potentials created")
    weights = defaultdict(float)
    for k in range(0, len(neurons_in_layers)-1):
        w = net.fc_layers[k].weight
        for j in range(len(w)):
            for i in range(len(w[j])):
                weights[(i, j, k)] = float(w[j][i])

    print("Weights initialized")
    # ================================================

    # Node eqn
    node_eqn = []

    for t in range(1, num_steps+1):
        print(f"Started for t={t}")
        for j, m in enumerate(layers):
            if j == 0:
                continue
            tim = time.time()
            for i in range(m):
                if t == 1:
                    tim2 = time.time()
                    S = sum(
                        [spike_indicators[(k, j - 1, t)] * weights[(k, i, j - 1)] for k in range(layers[j - 1])]
                    )
                    node_eqn.append(
                        And(
                            Implies(
                                S >= 1,
                                And(spike_indicators[(i, j, t)], potentials[(i, j, t)] == S - 1)
                            ),
                            Implies(
                                S < 1,
                                And(Not(spike_indicators[(i, j, t)]), potentials[(i, j, t)] == S)
                            )
                        )
                    )
                else:
                    tim2 = time.time()
                    S = sum([spike_indicators[(k, j - 1, t)] * weights[(k, i, j - 1)] for k in range(layers[j - 1])]) + beta * \
                        potentials[(i, j, t - 1)]
                    node_eqn.append(
                        And(
                            Implies(
                                S >= 1,
                                And(spike_indicators[(i, j, t)], potentials[(i, j, t)] == S - 1)
                            ),
                            Implies(
                                S < 1,
                                And(Not(spike_indicators[(i, j, t)]), potentials[(i, j, t)] == S)
                            )
                        )
                    )

            print(f"Complete for t={t}, layer={j} in time {time.time()-tim}")

    print("Node_eqn completed")

    S = Solver()
    S.add(node_eqn)
    f = open(f'{location}\\eqn\\eqn_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.txt', 'w')
    f.write(S.sexpr())
    f.close()
    print('Equations Saved')
