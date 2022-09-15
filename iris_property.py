import functools
import time

from sklearn import datasets
from snntorch import spikegen
from snntorch import functional as SF
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn

from z3 import *
from collections import defaultdict

shuffle = True
beta = 0.95
num_steps = 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train = False
file_name = 'model_iris.pth'


def compare(x, y):
    xx, yy = int(x.name().split('_')[-1]), int(y.name().split('_')[-1])
    return xx-yy


num_input = 4
num_hidden = 5
num_output = 3


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
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


iris = datasets.load_iris()

iris_data = iris.data / iris.data.max(axis=0)
iris_targets = iris.target

if shuffle:
    assert len(iris_data) == len(iris_data)
    perm = np.random.permutation(len(iris_data))
    iris_data, iris_targets = iris_data[perm], iris_targets[perm]


num_epochs = 1
loss_hist = []
test_loss_hist = []
counter = 0

if train:
    net = Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))
    #loss = nn.CrossEntropyLoss()
    loss = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    # Outer training loop
    for epoch in range(num_epochs):
        iter_counter = 0

        # Minibatch training loop
        for number in range(len(iris_targets)):
            data = torch.tensor(iris_data[number], dtype=torch.float)
            #targets = torch.tensor([0 if i != iris_targets[number] else 1 for i in range(max(iris_targets)+1)],dtype=torch.float)
            targets = torch.tensor([iris_targets[number]])

            # make spike trains
            data_spike = spikegen.rate(data, num_steps=num_steps)

            # forward pass
            net.train()
            spk_rec, mem_rec = net(data_spike.view(num_steps, -1))

            # initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=torch.float)
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            if counter % 20 == 0:
                print(f"Epoch {epoch}, Iteration {iter_counter}")
            counter += 1
            iter_counter += 1
    print("Saving model.pth")
    torch.save(net, file_name)
else:
    net = torch.load(file_name)
    print(f"Model loaded with t={num_steps}")

check = True
if check:
    acc = 0
    perm = np.random.permutation(len(iris_data))
    test_data, test_targets = torch.tensor(iris_data[perm][:100], dtype=torch.float), torch.tensor(iris_targets[perm][:100])
    for i, data in enumerate(test_data):
        spike_data = spikegen.rate(data, num_steps=num_steps)
        spk_rec, mem_rec = net(spike_data.view(num_steps, -1))
        idx = np.argmax(spk_rec.sum(dim=0).detach().numpy())
        if idx == test_targets[i]:
            #print(f'match for {test_targets[i]}')
            acc += 1
        else:
            #print(f'Not match for {test_targets[i]}')
            pass
    print(f'Accuracy of the model : {acc}%')

print()

# SMT encoding

# take a random input and make it into a spike train
layers = [num_input, num_hidden, num_output]
spike_indicators = {}
for t in range(num_steps):
    for j, m in enumerate(layers):
        for i in range(m):
            spike_indicators[(i, j, t+1)] = Bool(f'x_{i}_{j}_{t+1}')

potentials = {}
for t in range(num_steps+1):
    for j, m in enumerate(layers):
        if j == 0:
            continue
        for i in range(m):
            potentials[(i, j, t)] = Real(f'P_{i}_{j}_{t}')

weights = defaultdict(float)
w1 = net.fc1.weight
for j in range(len(w1)):
    for i in range(len(w1[j])):
        weights[(i, j, 0)] = float(w1[j][i])
w2 = net.fc2.weight
for j in range(len(w2)):
    for i in range(len(w2[j])):
        weights[(i, j, 1)] = float(w2[j][i])

#=====================================================
# Potential Initializations
pot_init = []
for j, m in enumerate(layers):
    if j == 0:
        continue
    for i in range(m):
        pot_init.append(potentials[(i, j, 0)] == 0)

# Assign Inputs
'''
assign = []
for i, spikes_t in enumerate(sample_spike):
    for j, spike in enumerate(spikes_t):
        if spike == 1:
            assign.append(spike_indicators[(j, 0, i+1)])
        else:
            assign.append(Not(spike_indicators[(j, 0, i + 1)]))
'''

# Node eqn
node_eqn = []
for t in range(1, num_steps+1):
    for j, m in enumerate(layers):
        if j == 0:
            continue

        for i in range(m):
            S = sum([spike_indicators[(k, j-1, t)]*weights[(k, i, j-1)] for k in range(layers[j-1])]) + potentials[(i, j, t-1)]
            node_eqn.append(
                And(
                    Implies(
                        S >= 1.0,
                        And(spike_indicators[(i, j, t)], potentials[(i, j, t)] == S - 1)
                    ),
                    Implies(
                        S < 1.0,
                        And(Not(spike_indicators[(i, j, t)]), potentials[(i, j, t)] == beta*S)
                    )
                )
            )
            #print(f'==========================================================\nAdded equation {(i,j,t)}')


# Sum of spikes
S1 = sum([2.0 * spike_indicators[0, 2, timestep+1] for timestep in range(num_steps)])
S2 = sum([2.0 * spike_indicators[1, 2, timestep+1] for timestep in range(num_steps)])
S3 = sum([2.0 * spike_indicators[2, 2, timestep+1] for timestep in range(num_steps)])

print('#'*40)
S = Solver()
S.add(pot_init+node_eqn)

# Property 1
# The network can output label 1
p1 = [S1 > S2, S1 > S3]

S.add(p1)
tx = time.time()
res = S.check()
p1t = time.time()-tx
if res == sat:
    print(f'The input network satisfies property 1. Checked in {p1t}')
else:
    print(f'The input network does not satisfy property 1. Checked in {p1t}')

print('#'*40)
S = Solver()
S.add(pot_init+node_eqn)

# Property 2
# The network can output label 2
p2 = [S2 > S1, S2 > S3]

S.add(p2)
tx = time.time()
res = S.check()
p2t = time.time()-tx
if res == sat:
    print(f'The input network satisfies property 2. Checked in {p2t}')
else:
    print(f'The input network does not satisfy property 2. Checked in {p2t}')


print('#'*40)
S = Solver()
S.add(pot_init+node_eqn)

# Property 3
# The network can output label 3
p3 = [S3 > S1, S3 > S2]

S.add(p3)
tx = time.time()
res = S.check()
p3t = time.time()-tx
if res == sat:
    print(f'The input network satisfies property 3. Checked in {p3t}')
else:
    print(f'The input network does not satisfy property 3. Checked in {p3t}')

print('#'*40)
S = Solver()
S.add(pot_init+node_eqn)

# Property 4
# Total output spikes must be less than 75%
p4 = [S1+S2+S3 > num_steps*num_output*0.75]

S.add(p4)
tx = time.time()
res = S.check()
p4t = time.time()-tx

if res != sat:
    print(f'The input network satisfies property 4. Checked in {p4t}')
else:
    print(f'The input network does not satisfy property 4. Checked in {p4t}')

print('#'*40)
