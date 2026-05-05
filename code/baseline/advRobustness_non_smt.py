import pdb, logging
from time import strftime, localtime
from typing import Generator

import numpy as np
from snnTrain import Net
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from snntorch import spikegen
from multiprocessing import Pool
from mnist import MNIST

from z3 import *
from mnist_net import *
from collections import defaultdict
import functools
import sys

sys.setrecursionlimit(2**20)

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])


log_name = f"{strftime('%m%d%H%M', localtime())}_mnist_baseline_np_{num_steps}_{'_'.join(str(l) for l in neurons_in_layers)}_delta_{tuple(delta)}.log"
logging.basicConfig(filename=f"../log/{log_name}", level=logging.INFO)
stdout = print
print = lambda x: logging.getLogger().info(x) or stdout(x)

print(f"neurons in layers {neurons_in_layers}, number of steps {num_steps}")

print('Reading Model')
net_dict = torch.load(f'{location}/models/model_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.pth')
net = Net(layers=neurons_in_layers)
net.load_state_dict(net_dict)
print('Model loaded')

print('Loading data')
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=1, shuffle=True, drop_last=True)
print('data loaded')

samples_list = [41905, 7296, 1639, 48598, 18024, 16049, 14628,
                9144, 48265, 6717, 44348, 48540, 58469, 35741]


# Recursively find available adversarial attacks.
def search_perts(spk_train:torch.Tensor, delta:int, loc:int=0) -> Generator[torch.Tensor,None,None]:
    # Last case
    if delta == 0:
        yield spk_train
    # Search must be terminated at the end of image.
    elif loc < neurons_in_layers[0] * num_steps:
        loc_4d = (loc//neurons_in_layers[0], 0, loc%neurons_in_layers[0]//28, loc%28)

        new_spk_train = spk_train.clone().detach()
        yield from search_perts(new_spk_train,
                                delta,
                                loc+1)

        new_spk_train[loc_4d] = 1 - new_spk_train[loc_4d]
        yield from search_perts(new_spk_train,
                                delta-1,
                                loc+1)


def check_sample(sample_no:int):
    tx = time.time()
    data, target = mnist_train[sample_no]
    inp = spikegen.rate(data, num_steps=num_steps)
    op = net.forward(inp.view(num_steps, -1))[0]
    orig_pred = int(torch.stack(op, dim=0).sum(dim=0).argmax())
    print(f'single input ran in {time.time()-tx} sec')
    
    print('Query processing starts')
    tx = time.time()
    
    sat_flag = False
    for pertd_inp in search_perts(inp, dt):
        pertd_op = net.forward(pertd_inp.view(num_steps, -1))[0]
        spk_counts = torch.stack(pertd_op, dim=0).sum(dim=0)
        not_orig_mask = [x for x in range(neurons_in_layers[-1]) if x != orig_pred]
        if torch.any(spk_counts[not_orig_mask] >= spk_counts[orig_pred]):
            sat_flag = True
            break
    
    print(f'Checking done in time {time.time() - tx}')
    if sat_flag:
        print(f'Not robust for sample {sample_no} and delta={dt}')
    else:
        print(f'Robust for sample {sample_no} and delta={dt}')
    print("")

# For each delta
for dt in delta:
    with Pool(processes=len(samples_list)) as pool:
        pool.map(check_sample, samples_list)
        pool.close()
        pool.join()