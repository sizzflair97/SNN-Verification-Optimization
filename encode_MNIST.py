import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from snntorch import spikegen

from z3 import *
from collections import defaultdict
import functools


beta = 1
num_steps = 5
num_hidden = 2
num_input = 2
num_output = 2
input_real = False
data_path = '/data/mnist'


def compare(x, y):
    xx, yy = int(x.name().split('_')[-1]), int(y.name().split('_')[-1])
    return xx-yy


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_input, num_hidden, bias=False)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_output, bias=False)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk1_rec = []
        mem1_rec = []
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk1_rec, dim=0), torch.stack(mem1_rec, dim=0), torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


load = True
num_k = 500
if load:
    net = torch.load(f'C:\\Users\\soham\\PycharmProjects\\Z3py\\Models\\model_{num_k}.pth')
    num_input = net.fc1.in_features
    num_hidden = net.fc2.in_features
    num_output = net.fc2.out_features

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))])

    #mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    test_loader = DataLoader(mnist_test, batch_size=1, shuffle=True, drop_last=True)
    d, t = next(iter(test_loader))
    inp = spikegen.rate(d, num_steps=num_steps)


else:
    net = Net()
    net.fc1.weight.data = torch.tensor([[0.7, -0.2], [-0.2, 0.7]])
    net.fc2.weight.data = torch.tensor([[-0.2, 0.7], [0.7, -0.2]])

    inp = torch.tensor([[1, 0]] * num_steps, dtype=torch.float)


op = net.forward(inp.view(num_steps, -1))
#=============================================

layers = [num_input, num_hidden, num_output]
delta = 1
print("Model Simulated")

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
w1 = net.fc1.weight
for j in range(len(w1)):
    for i in range(len(w1[j])):
        weights[(i, j, 0)] = float(w1[j][i])

w2 = net.fc2.weight
for j in range(len(w2)):
    for i in range(len(w2[j])):
        weights[(i, j, 1)] = float(w2[j][i])
print("Weights initialized")
#================================================
'''
pot_init = []
for j, m in enumerate(layers):
    if j == 0:
        continue
    for i in range(m):
        pot_init.append(potentials[(i, j, 0)] == 0)
print("Potential initialized")
'''

assign = []
for timestep, spike_train in enumerate(inp):
    for i, spike in enumerate(spike_train.view(num_input)):
        if spike == 1:
            assign.append(spike_indicators[(i, 0, timestep + 1)])
        else:
            assign.append(Not(spike_indicators[(i, 0, timestep + 1)]))

print("Inputs Assigned")


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
                #torch.save(node_eqn, f'eqn/formulas_all_{t}.pkl')
            #print(f'Done for node {i} in {time.time()-tim2}')
        print(f"Complete for t={t}, layer={j} in time {time.time()-tim}")
        #torch.save(node_eqn, f'eqn/formulas_all_{t}.pkl')
        #del node_eqn
        #node_eqn = {}
print("Node_eqn completed")

S = Solver()
opq = [Not(spike_indicators[0, 2, t+1]) for t in range(num_steps)]
S.add(node_eqn)
f = open(f'C:\\Users\\soham\\PycharmProjects\\Z3py\\eqn\\eqn_{num_k}.txt', 'w')
f.write(S.sexpr())
f.close()
'''
S2 = Solver()
#S2.add(node_eqn+pot_init+assign)
S2.add(node_eqn+pot_init+assign)
tx = time.time()
S2.check()
print(time.time()-tx)
m = S2.model()

m = S.model()
for k in range(10):
    names = []
    for i in m.decls():
        t = i.name().split('_')
        if t[0] == 'x' and t[1] == f'{k}' and t[2] == '2':
            names.append(i)
    for i in sorted(names, key=functools.cmp_to_key(compare)):
        print(f'{i}->{m[i]}')
    input()
    print()
'''

'''
with open('temp.txt',mode='w') as f:
    f.write(s2)
'''

'''
s = Solver()
tx = time.time()
s.from_file('temp.txt')
print(f'Total time for model {time.time()-tx}')

tx = time.time()
s.check()
print(f'Total time for checking {time.time()-tx}')

'''