import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
import time

import sys

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
    return xx - yy


def get_pert(inp, delta):
    inp_pert = np.copy(inp)
    for _ in range(delta):
        row = np.random.randint(low=0, high=num_steps)
        column = np.random.randint(low=0, high=inp.shape[1])
        inp_pert[row][column] = 1 - inp_pert[row][column]
    return inp_pert


def flip(inp, r, c):
    pert_inp = np.copy(inp)
    pert_inp[r][c] = 1 - pert_inp[r][c]
    return pert_inp


sys.setrecursionlimit(10 ** 5)


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

        return torch.stack(spk1_rec, dim=0), torch.stack(mem1_rec, dim=0), torch.stack(spk2_rec, dim=0), torch.stack(
            mem2_rec, dim=0)


num_k = 10

net = torch.load(f'C:\\Users\\soham\\PycharmProjects\\Z3py\\Models\\model_{num_k}.pth')
num_input = net.fc1.in_features
num_hidden = net.fc2.in_features
num_output = net.fc2.out_features

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])

# mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=1, shuffle=True, drop_last=True)

layers = [num_input, num_hidden, num_output]

# delta = [1, 10, 15, 20, 25]
#deltas = [10, 50]
num_pert = num_input*num_steps

tx = time.time()
spike_indicators = {}
for t in range(num_steps):
    for j, m in enumerate(layers):
        if j == 1:
            continue
        else:
            for i in range(m):
                spike_indicators[(i, j, t + 1)] = Bool(f'x_{i}_{j}_{t + 1}')

print(f"Spike variables created in {time.time() - tx} sec")

data, target = next(iter(test_loader))
inp = spikegen.rate(data, num_steps=num_steps).view(num_steps, -1)
op_net = net.forward(inp)[2]
label = int(op_net.sum(dim=0).argmax())
r, c = 0, 0
timel = []
pert_c = 0

for pno in range(num_pert):
    # inp_pert = get_pert(inp, delta) # subtract this time from total
    c += 1
    if c >= num_input:
        c = 0
        r += 1
        if r >= num_steps:
            break
    inp_pert = flip(inp, r, c)

    # Assign pert image
    tx = time.time()
    assign = []
    for timestep, spike_train in enumerate(inp_pert):
        for i, spike in enumerate(spike_train):
            if spike == 1:
                assign.append(spike_indicators[(i, 0, timestep + 1)])
            else:
                assign.append(Not(spike_indicators[(i, 0, timestep + 1)]))
    #print("Assignment for inp_pert done")
    '''
    # Output Prop
    op = []
    intend_sum = sum([2 * spike_indicators[(label, 2, timestep + 1)] for timestep in range(num_steps)])
    for t in range(num_output):
        if t != label:
            op.append(
                Not(intend_sum > sum([2 * spike_indicators[(t, 2, timestep + 1)] for timestep in range(num_steps)]))
            )
    '''
    # S.push()
    # print('Model pushed')
    S = Solver()  # Encoding time should be done once. #################################
    with open(f'C:\\Users\\soham\\PycharmProjects\\Z3py\\eqn\\eqn_{num_k}.txt') as f:
        S.from_string(f.read())
    # S.from_file(f'C:\\Users\\soham\\PycharmProjects\\Z3py\\eqn\\eqn_{k}.txt')
    #print('Model read from file')
    S.add(assign)
    #print('Pert assignment added')
    # print('Model Ready')
    # tx = time.time()
    res = S.check()
    # print(time.time()-tx)
    #print(f'Model checked res={res}')

    m = S.model()
    opc = [0 for count in range(num_output)]
    for k in range(10):
        for timestep in range(1, num_steps+1):
            if str(m[spike_indicators[(k, 2, timestep)]]) == 'True':
                opc[k] += 1
    pert_op = np.array(opc).argmax()
    timel.append(time.time()-tx)
    if label == pert_op:
        pert_c += 1
    else:
        break

    if (pno + 1) % 1 == 0:
        print(f'Done for {pno + 1} perturbations in avg {np.mean(timel)} sec and {pert_c} matches')

    # S.pop()
    del S

if num_pert == pert_c+1:
    print(f"For hidden = {num_k}, all perturbations have same output in avg time {np.mean(timel)} sec")
else:
    print(f'For hidden = {num_k}, the network is not adversarially robust')
print()
