import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import defaultdict
from z3 import *
from mnist_net import *
from tqdm.auto import tqdm
import time

import matplotlib.pyplot as plt
import numpy as np
import itertools

transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

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
        torch.save(net.state_dict(), f'{location}/models/model_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.pth')
        print("Model Saved")
    else:
        net_dict = torch.load(f'{location}/models/model_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.pth')
        net.load_state_dict(net_dict)
        print('Model loaded')

    total = 0
    correct = 0
    test_batch_size = 1
    # drop_last switched to False to keep all samples
    test_loader = DataLoader(mnist_train, batch_size=test_batch_size, shuffle=True, drop_last=False)
    test_log = True
    tt = []
    if test_log:
        with torch.no_grad():
            net.eval()
            c = 1
            for data, targets in tqdm(test_loader):
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
    with open(f'{location}/models/model_accs.txt', 'a') as f:
        f.write(f"Rate_{num_steps}_{neurons_in_layers}: {correct/total:.3f}\n")
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
    f = open(f'{location}/eqn/eqn_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.txt', 'w')
    f.write(S.sexpr())
    f.close()
    print('Equations Saved')
