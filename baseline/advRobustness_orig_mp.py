import time
from snnTrain import Net
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from snntorch import spikegen
from multiprocessing import Pool

from z3 import *
from collections import defaultdict
import functools

neurons_in_layers = [28*28, 20, 10]
num_steps = 10
data_path = '../data/mnist'
delta = [1]
location = '.'

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])


print('Reading Model')
net = torch.load(f'Models/model_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.pth')
print('Model loaded')

print('Loading data')
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=1, shuffle=True, drop_last=True)
print('data loaded')

tx = time.time()
spike_indicators = {}
for t in range(num_steps):
    for j, m in enumerate(neurons_in_layers):
        if j == 1:
            continue
        else:
            for i in range(m):
                spike_indicators[(i, j, t+1)] = Bool(f'x_{i}_{j}_{t+1}')
print(f"Spikes created in {time.time()-tx} sec")

# tx = time.time()
# data, target = next(iter(test_loader))
# inp = spikegen.rate(data, num_steps=num_steps)
# op = net.forward(inp.view(num_steps, -1))[0]
# label = int(torch.cat(op).sum(dim=0).argmax())
# print(f'single input ran in {time.time()-tx} sec')

samples_list = [41905, 7296, 1639, 48598, 18024, 16049, 14628,
                9144, 48265, 6717, 44348, 48540, 58469, 35741]

def check_sample(sample_no:int):
    tx = time.time()
    data, target = mnist_train[sample_no]
    inp = spikegen.rate(data, num_steps=num_steps)
    op = net.forward(inp.view(num_steps, -1))[0]
    label = int(torch.cat(op).sum(dim=0).argmax())
    print(f'single input ran in {time.time()-tx} sec')
    
    # Input property
    tx = time.time()
    s = [[] for i in range(num_steps)]
    sv = [Int(f's_{i + 1}') for i in range(num_steps)]
    prop = []
    for timestep, spike_train in enumerate(inp):
        for i, spike in enumerate(spike_train.view(neurons_in_layers[0])):
            if spike == 1:
                s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 0.0, 1.0))
            else:
                s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 1.0, 0.0))
    prop = [sv[i] == sum(s[i]) for i in range(num_steps)]
    prop.append(sum(sv) <= dt)
    # print(prop[0])
    print(f"Inputs Property Done in {time.time() - tx} sec")

    # Output property
    tx = time.time()
    op = []
    intend_sum = sum([2 * spike_indicators[(label, 2, timestep + 1)] for timestep in range(num_steps)])
    for t in range(neurons_in_layers[-1]):
        if t != label:
            op.append(
                Not(intend_sum > sum([2 * spike_indicators[(t, 2, timestep + 1)] for timestep in range(num_steps)]))
            )
    print(f'Output Property Done in {time.time() - tx} sec')

    tx = time.time()
    S = Solver()
    S.from_file(f'{location}/eqn/eqn_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.txt')
    print(f'Network Encoding read in {time.time() - tx} sec')
    S.add(op + prop)
    print(f'Total model ready in {time.time() - tx}')

    print('Query processing starts')
    tx = time.time()
    result = S.check()
    print(f'Checking done in time {time.time() - tx}')
    if result == sat:
        print(f'Not robust for sample {sample_no} and delta={dt}')
    else:
        print(f'Robust for sample and delta={dt}')

# For each delta
for dt in delta:
    with Pool(processes=len(samples_list)) as pool:
        pool.map(check_sample, samples_list)
        pool.close()
        pool.join()
        