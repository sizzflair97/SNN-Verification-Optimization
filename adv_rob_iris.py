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
num_steps = 25
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train = True
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
    print("Model loaded")

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

#S.push()
#print("Equations Created")

num_samples = 5
samples = iris_data[np.random.choice(range(len(iris_data)), num_samples)]
deltas = [1,2,3]

delta_v = {d: 0 for d in deltas}

for delta in deltas:
    avt = 0
    for sample_no, sample in enumerate(samples):
        sample_spike = spikegen.rate(torch.tensor(sample, dtype=torch.float), num_steps=num_steps)

        spk_rec, mem_rec = net(sample_spike.view(num_steps, -1))
        label = int(spk_rec.sum(dim=0).argmax())

        S = Solver()
        # S.add(assign+node_eqn+pot_init)
        S.add(node_eqn + pot_init)

        sum_val = []
        for timestep, spike_train in enumerate(sample_spike):
            for i, spike in enumerate(spike_train.view(num_input)):
                if spike == 1:
                    sum_val.append(If(spike_indicators[(i, 0, timestep + 1)], 0.0, 1.0))
                else:
                    sum_val.append(If(spike_indicators[(i, 0, timestep + 1)], 1.0, 0.0))
        prop = [sum(sum_val) <= delta]
        S.add(prop)
        '''
        s = [[] for i in range(num_steps)]
        sv = [Int(f's_{i + 1}') for i in range(num_steps)]
        prop = []
        for timestep, spike_train in enumerate(sample_spike):
            for i, spike in enumerate(spike_train.view(num_input)):
                if spike == 1:
                    s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 0.0, 1.0))
                else:
                    s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 1.0, 0.0))
        prop = [sv[i] == sum(s[i]) for i in range(num_steps)]
        prop.append(sum(sv) <= delta)
        # print(prop[0])
        #print(f"Inputs Property Done in {time.time() - tx} sec")
        '''

        # Output property
        #tx = time.time()
        op = []
        intend_sum = sum([2 * spike_indicators[(label, 2, timestep + 1)] for timestep in range(num_steps)])
        for t in range(num_output):
            if t != op:
                op.append(
                    Not(intend_sum > sum([2 * spike_indicators[(t, 2, timestep + 1)] for timestep in range(num_steps)]))
                )
        #print(f'Output Property Done in {time.time() - tx} sec')
        S.add(op)
        tx = time.time()
        res = S.check()
        if str(res) == 'unsat':
            delta_v[delta] += 1
        else:
            '''
            sadv = np.zeros((num_steps, num_input), dtype=float)
            m = S.model()
            for tt in range(num_steps):
                for k in range(num_input):
                    sadv[tt][k] = 1 if str(m[spike_indicators[(k, 0, tt + 1)]]) == 'True' else 0
            print()
            '''
            pass
        del S
        tss = time.time()-tx
        print(f'Completed for delta = {delta}, sample = {sample_no} in {tss} sec as {res}')
        avt = (avt*sample_no + tss)/(sample_no+1)
    print(f'Completed for delta = {delta} with {delta_v[delta]} in avg time {avt} sec')


'''
m = S.model()
for k in range(num_output):
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



print()