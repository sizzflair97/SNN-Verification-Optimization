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

def prepare_net() -> Net:
    # Load the network onto CUDA if available
    

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

    net = Net(layers, loss_value=beta).to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

    loss_hist = []
    test_loss_hist = []
    counter = 0
    if train:
        # Outer training loop
        for epoch in tqdm(range(num_epochs)):
            iter_counter = 0
            train_batch = iter(train_loader)

            # Minibatch training loop
            for data, targets in (pbar:=tqdm(train_batch, leave=False)):
                data = data.to(device)
                targets = targets.to(device)

                spike_data = spikegen.rate(data, num_steps=num_steps) # type: ignore

                # forward pass
                net.train()
                spk_rec, mem_rec = net(spike_data.view(num_steps, batch_size, -1), return_all=True)

                # initialize the loss & sum over time
                loss_val = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    loss_val += loss(mem_rec[step][-1], targets)

                # Gradient calculation + weight update
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                # Store loss history for future plotting
                loss_hist.append(loss_val.item())
    
    info("Net is prepared.")

        #torch.save(net, f'/Models/model_{num_hidden}.pth')
    #     torch.save(net, f'{location}\\Models\\model_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.pth')
    #     info("Model Saved")
    # else:
    #     net = torch.load(f'{location}\\Models\\model_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.pth')
    #     info('Model loaded')

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
            for data, targets in (pbar:=tqdm(test_loader)):
                t = time.time()
                data = data.to(device)
                targets = targets.to(device)

                test_spike_data = spikegen.rate(data, num_steps=num_steps) # type: ignore

                # forward pass
                list_of_spikes:List[List[Tensor]]
                list_of_spikes, _ = net(test_spike_data.view(num_steps, test_batch_size, -1))
                last_spikes = net.extract_last_spikes(list_of_spikes)
                # calculate total accuracy
                # predicted = torch.cat(test_spk).sum(dim=0).argmax()
                predicted = torch.cat(last_spikes).sum(dim=0).argmax()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                tt.append(time.time()-t)
                c += 1

        info(f"Total correctly classified test set images: {correct}/{total} in avg.time {sum(tt)/len(tt)}")
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
    info('Data is loaded')
    
    tx = time.time()
    spike_indicators = gen_spikes()
    potentials = gen_potentials()
    weights = gen_weights(net)
    node_eqns = gen_node_eqns(weights, spike_indicators, potentials)
    if cfg.np_level == 1:
        node_eqns.extend(gen_dnp_v2(weights, spike_indicators, potentials))
    elif cfg.np_level == 2:
        node_eqns.extend(gen_gnp(weights, spike_indicators))

    return

    tx = time.time()
    inp_vec:List[Tensor] = []
    label_vec:List[int] = []
    for _, (data, target) in zip(range(num_procs), iter(test_loader)):
        inp_vec.append(spikegen.rate(data.to(device), num_steps=num_steps).detach()) # type: ignore
        list_of_spikes:List[List[Tensor]] = net.forward(inp_vec[-1].view(num_steps, -1))[0]
        last_spikes = net.extract_last_spikes(list_of_spikes)
        label_vec.append(int(torch.cat(last_spikes).sum(dim=0).argmax()))
    info(f"Sampling is completed with {num_procs} samples.")
    # data, target = next(iter(test_loader))
    # inp = spikegen.rate(data, num_steps=num_steps) # type: ignore
    # op = net.forward(inp.view(num_steps, -1))[0]
    # label = int(torch.cat(op).sum(dim=0).argmax())
    # info(f'single input ran in {time.time()-tx} sec')

    # For each delta
    for dt in cfg.deltas:
        global check_sample
        def check_sample(sample:Tuple[List[List[float]], int]):
            inp, label = sample
            inp = torch.tensor(inp)
            # Input property
            tx = time.time()
            s = [[] for _ in range(num_steps)]
            sv = [Int(f's_{i + 1}') for i in range(num_steps)]
            prop = []
            for timestep, spike_train in enumerate(inp):
                for i, spike in enumerate(spike_train.view(layers[0])):
                    if spike == 1:
                        s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 0.0, 1.0))
                    else:
                        s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 1.0, 0.0))
            prop = [sv[i] == sum(s[i]) for i in range(num_steps)]
            prop.append(sum(sv) <= dt)
            # info(prop[0])
            info(f"Inputs Property Done in {time.time() - tx} sec")

            # Output property
            tx = time.time()
            op = []
            intend_sum = sum([2 * spike_indicators[(label, 2, timestep + 1)] for timestep in range(num_steps)])
            for t in range(layers[-1]):
                if t != label:
                    op.append(
                        Not(intend_sum > sum([2 * spike_indicators[(t, 2, timestep + 1)] for timestep in range(num_steps)]))
                    )
            info(f'Output Property Done in {time.time() - tx} sec')

            tx = time.time()
            S = Solver()
            # S.from_file(f'{location}\\eqn\\eqn_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.txt')
            info(f'Network Encoding read in {time.time() - tx} sec')
            # S.add(op + prop)
            S.add(op + node_eqns + prop)
            info(f'Total model ready in {time.time() - tx}')

            info('Query processing starts')
            tx = time.time()
            result = S.check()
            info(f'Checking done in time {time.time() - tx}')
            if result == sat:
                info(f'Not robust for sample and delta={dt}')
            else:
                info(f'Robust for sample and delta={dt}')
        
        samples = ((inp.cpu().tolist(), label) for inp, label in zip(inp_vec, label_vec))
        if mp:
            with Pool(num_procs) as pool:
                pool.map(check_sample, samples)
        for sample in samples:
            check_sample(sample)
            # # Input property
            # tx = time.time()
            # s = [[] for _ in range(num_steps)]
            # sv = [Int(f's_{i + 1}') for i in range(num_steps)]
            # prop = []
            # for timestep, spike_train in enumerate(inp):
            #     for i, spike in enumerate(spike_train.view(layers[0])):
            #         if spike == 1:
            #             s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 0.0, 1.0))
            #         else:
            #             s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 1.0, 0.0))
            # prop = [sv[i] == sum(s[i]) for i in range(num_steps)]
            # prop.append(sum(sv) <= dt)
            # # info(prop[0])
            # info(f"Inputs Property Done in {time.time() - tx} sec")

            # # Output property
            # tx = time.time()
            # op = []
            # intend_sum = sum([2 * spike_indicators[(label, 2, timestep + 1)] for timestep in range(num_steps)])
            # for t in range(layers[-1]):
            #     if t != label:
            #         op.append(
            #             Not(intend_sum > sum([2 * spike_indicators[(t, 2, timestep + 1)] for timestep in range(num_steps)]))
            #         )
            # info(f'Output Property Done in {time.time() - tx} sec')

            # tx = time.time()
            # S = Solver()
            # # S.from_file(f'{location}\\eqn\\eqn_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.txt')
            # info(f'Network Encoding read in {time.time() - tx} sec')
            # # S.add(op + prop)
            # S.add(op + node_eqns + prop)
            # info(f'Total model ready in {time.time() - tx}')

            # info('Query processing starts')
            # tx = time.time()
            # result = S.check()
            # info(f'Checking done in time {time.time() - tx}')
            # if result == sat:
            #     info(f'Not robust for sample and delta={dt}')
            # else:
            #     info(f'Robust for sample and delta={dt}')
        # # Input property
        # tx = time.time()
        # s = [[] for _ in range(num_steps)]
        # sv = [Int(f's_{i + 1}') for i in range(num_steps)]
        # prop = []
        # for timestep, spike_train in enumerate(inp):
        #     for i, spike in enumerate(spike_train.view(layers[0])):
        #         if spike == 1:
        #             s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 0.0, 1.0))
        #         else:
        #             s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 1.0, 0.0))
        # prop = [sv[i] == sum(s[i]) for i in range(num_steps)]
        # prop.append(sum(sv) <= dt)
        # # info(prop[0])
        # info(f"Inputs Property Done in {time.time() - tx} sec")

        # # Output property
        # tx = time.time()
        # op = []
        # intend_sum = sum([2 * spike_indicators[(label, 2, timestep + 1)] for timestep in range(num_steps)])
        # for t in range(layers[-1]):
        #     if t != label:
        #         op.append(
        #             Not(intend_sum > sum([2 * spike_indicators[(t, 2, timestep + 1)] for timestep in range(num_steps)]))
        #         )
        # info(f'Output Property Done in {time.time() - tx} sec')

        # tx = time.time()
        # S = Solver()
        # # S.from_file(f'{location}\\eqn\\eqn_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.txt')
        # info(f'Network Encoding read in {time.time() - tx} sec')
        # # S.add(op + prop)
        # S.add(op + node_eqns + prop)
        # info(f'Total model ready in {time.time() - tx}')

        # info('Query processing starts')
        # tx = time.time()
        # result = S.check()
        # info(f'Checking done in time {time.time() - tx}')
        # if result == sat:
        #     info(f'Not robust for sample and delta={dt}')
        # else:
        #     info(f'Robust for sample and delta={dt}')

    # delta_v = {d: 0 for d in cfg.deltas}
    # for delta in cfg.deltas:
    #     avt = 0
        
    #     global check_sample
    #     def check_sample(sample:Tuple[int, Tensor]) -> Tuple[float, int, str]:
    #         sample_no:int; sample_spike:Tensor;
    #         sample_no, sample_spike = sample
    #         res, label_var, control = forward_net(sample_spike.view(num_steps, -1), spike_indicators, assign+node_eqn+pot_init)
    #         if res in {'unsat','unknown'}:
    #             info(f'Could not find model at delta = {delta}, sample = {sample_no}')
    #             return -1, delta, res
    #         del res
            
    #         control = control.model()
    #         prop = gen_delta_reuse(cfg, sample_spike, spike_indicators, potentials, delta, control)
    #         # Output property
    #         #tx = time.time()
    #         op = []
    #         label = control[label_var].as_long() # type: ignore
            
    #         S = Solver()
    #         intend_sum = sum([2 * spike_indicators[(label, 2, timestep)] for timestep in range(1, num_steps+1)])
    #         for t in range(num_output):
    #             if t != label:
    #                 op.append(
    #                     Not(intend_sum > sum([2 * spike_indicators[(t, 2, timestep)] for timestep in range(1, num_steps+1)]))
    #                 )
    #         S.add(assign+node_eqn+pot_init+prop+op)
            
    #         tx = time.time()
    #         res:Literal["sat", "unsat", "unknown"] = str(S.check()) # type: ignore
    #         del S
    #         tss = time.time()-tx
    #         info(f'Completed for delta = {delta}, sample = {sample_no} in {tss} sec as {res}')
    #         return tss, delta, res
        
    #     sample_spks = [spikegen.rate(torch.tensor(sample, dtype=torch.float), num_steps=num_steps) # type: ignore
    #                    for sample in samples]
        
    #     if mp:
    #         with Pool(processes=num_procs) as pool:
    #             tss_lst = pool.map(check_sample, enumerate(sample_spks))
    #         for tss, delta, res in tss_lst:
    #             avt += tss
    #             delta_v[delta] += 1 if res == "unsat" else 0
    #         avt /= len(sample_spks)
    #     else:
    #         for sample_no, sample_spike in enumerate(sample_spks):
    #             tss, delta, res = check_sample((sample_no, sample_spike))
    #             avt = (avt*sample_no + tss)/(sample_no+1)
    #             delta_v[delta] += 1 if res == "unsat" else 0
    #     info(f'Completed for delta = {delta} with {delta_v[delta]} in avg time {avt} sec')
    #     del check_sample

    info("")


