# %%
from multiprocessing import Pipe, Pool
import functools, time, logging, json
from time import localtime, strftime
from typing import Any
from snntorch import spikegen
from snntorch import functional as SF
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from tqdm.auto import tqdm
from z3 import *
from collections import defaultdict
from utils import *
from utils.dictionary_mnist import *
from utils.encoding_mnist import *
from utils import MnistNet as Net

transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
    
def info(msg:Any):
    print(msg) or logging.getLogger().info(msg) # type: ignore

def prepare_net() -> Net:
    # Load the network onto CUDA if available
    

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

    net = Net(neurons_in_layers, loss_value=beta)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

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

                spike_data = spikegen.rate(data, num_steps=num_steps) # type: ignore

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
    #     torch.save(net, f'{location}\\Models\\model_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.pth')
    #     print("Model Saved")
    # else:
    #     net = torch.load(f'{location}\\Models\\model_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.pth')
    #     print('Model loaded')

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
    return net

def run_test(cfg:CFG):
    log_name = f"{strftime('%m%d%H%M', localtime())}_{cfg.log_name}.log"
    logging.basicConfig(filename="log/" + log_name, level=logging.INFO)
    info(cfg)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)

    net = prepare_net()
    
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

    tx = time.time()
    data, target = next(iter(test_loader))
    inp = spikegen.rate(data, num_steps=num_steps)
    op = net.forward(inp.view(num_steps, -1))[0]
    label = int(torch.cat(op).sum(dim=0).argmax())
    print(f'single input ran in {time.time()-tx} sec')

    # For each delta
    for dt in delta:

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
        S.from_file(f'{location}\\eqn\\eqn_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.txt')
        print(f'Network Encoding read in {time.time() - tx} sec')
        S.add(op + prop)
        print(f'Total model ready in {time.time() - tx}')

        print('Query processing starts')
        tx = time.time()
        result = S.check()
        print(f'Checking done in time {time.time() - tx}')
        if result == sat:
            print(f'Not robust for sample and delta={dt}')
        else:
            print(f'Robust for sample and delta={dt}')
    
    # take a random input and make it into a spike train
    spike_indicators = gen_spikes()
    potentials = gen_potentials()
    weights = gen_weights(net)
    pot_init = gen_initial_potentials(potentials)

    #Node eqns
    assign:List[BoolRef] = []
    node_eqn:List[Union[BoolRef,Literal[False]]] = []
    #In neuron pruning, indicators and potentials are modified.
    if cfg.np_level == 1:
        node_eqn.extend(gen_dnp_v2(weights, spike_indicators, potentials))
    elif cfg.np_level == 2:
        node_eqn.extend(gen_gnp(weights, spike_indicators))
    elif cfg.np_level == 3:
        node_eqn.extend(gen_gnp_v2(weights, spike_indicators))
    
    node_eqn.extend(gen_node_eqns(weights, spike_indicators, potentials))

    #Randomly draw samples
    samples = iris_data[np.random.choice(range(len(iris_data)), cfg.num_samples)] # type: ignore
    info(samples)

    delta_v = {d: 0 for d in cfg.deltas}
    for delta in cfg.deltas:
        avt = 0
        
        global check_sample
        def check_sample(sample:Tuple[int, Tensor]) -> Tuple[float, int, str]:
            sample_no:int; sample_spike:Tensor;
            sample_no, sample_spike = sample
            res, label_var, control = forward_net(sample_spike.view(num_steps, -1), spike_indicators, assign+node_eqn+pot_init)
            if res in {'unsat','unknown'}:
                info(f'Could not find model at delta = {delta}, sample = {sample_no}')
                return -1, delta, res
            del res
            
            control = control.model()
            prop = gen_delta_reuse(cfg, sample_spike, spike_indicators, potentials, delta, control)
            # Output property
            #tx = time.time()
            op = []
            label = control[label_var].as_long() # type: ignore
            
            S = Solver()
            intend_sum = sum([2 * spike_indicators[(label, 2, timestep)] for timestep in range(1, num_steps+1)])
            for t in range(num_output):
                if t != label:
                    op.append(
                        Not(intend_sum > sum([2 * spike_indicators[(t, 2, timestep)] for timestep in range(1, num_steps+1)]))
                    )
            S.add(assign+node_eqn+pot_init+prop+op)
            
            tx = time.time()
            res:Literal["sat", "unsat", "unknown"] = str(S.check()) # type: ignore
            del S
            tss = time.time()-tx
            info(f'Completed for delta = {delta}, sample = {sample_no} in {tss} sec as {res}')
            return tss, delta, res
        
        sample_spks = [spikegen.rate(torch.tensor(sample, dtype=torch.float), num_steps=num_steps) # type: ignore
                       for sample in samples]
        
        if mp:
            with Pool(processes=num_procs) as pool:
                tss_lst = pool.map(check_sample, enumerate(sample_spks))
            for tss, delta, res in tss_lst:
                avt += tss
                delta_v[delta] += 1 if res == "unsat" else 0
            avt /= len(sample_spks)
        else:
            for sample_no, sample_spike in enumerate(sample_spks):
                tss, delta, res = check_sample((sample_no, sample_spike))
                avt = (avt*sample_no + tss)/(sample_no+1)
                delta_v[delta] += 1 if res == "unsat" else 0
        info(f'Completed for delta = {delta} with {delta_v[delta]} in avg time {avt} sec')
        del check_sample

    print()


