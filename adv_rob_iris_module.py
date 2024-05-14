# %%
from multiprocessing import Pipe, Pool
import functools, time, logging, json
from time import localtime, strftime
from sklearn import datasets
from snntorch import spikegen
from snntorch import functional as SF
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from tqdm.auto import tqdm
from z3 import *
from collections import defaultdict
from utils import *
from utils.dictionary_iris import *
from utils.encoding_iris import *
from utils import IrisNet as Net

def prepare_net(iris_data:np.ndarray, iris_targets:np.ndarray) -> Net:
    loss_hist = []
    seq = np.random.permutation(len(iris_targets))
    iris_data = iris_data[seq]
    iris_targets = iris_targets[seq]
    
    if train:
        net = Net()
        optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
        loss = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2) # type: ignore
        encoding_func = spikegen.rate

        # Outer training loop
        for epoch in range(num_epochs):
            iter_counter = 0

            # Minibatch training loop
            for number in (pbar:=tqdm(range(len(iris_targets)))):
                data = torch.tensor(iris_data[number], dtype=torch.float)
                targets = torch.tensor(iris_targets[number], dtype=torch.long)

                # make spike trains
                data_spike:Tensor = encoding_func(data, num_steps=num_steps) # type: ignore # for latency encoding
                # data_spike = encoding_func(data, num_steps=num_steps) # type: ignore # for rate encoding

                # forward pass
                net.train()
                spk_rec, mem_rec = net(data_spike.view(num_steps, -1))

                # initialize the loss & sum over time
                loss_val = loss(spk_rec[:,None,:], targets[None])
                # loss_val = torch.zeros((1), dtype=torch.float)
                # for step in range(num_steps):
                #     loss_val += loss(mem_rec[-1][step], targets[None])

                # Gradient calculation + weight update
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                
                # Store loss history for future plotting
                loss_hist.append(loss_val.item())

                try:
                    pbar.desc = (f"Epoch {epoch}, Iteration {iter_counter} "+
                                 f"LogLoss {-float('inf') if loss_hist[-1]==0 else math.log(loss_hist[-1]):.3f}, MeanSpk {net.mean_spks:.2f}")
                except ValueError:
                    print(loss_hist[-1])
                iter_counter += 1
            
            info((net.fc2.weight@net.fc1.weight)*((num_steps+1)*beta**num_steps-beta**(num_steps+1)+1)/(1-beta)**2)
        # info("Saving model.pth")
        # torch.save(net, file_name)
    else:
        net = torch.load(file_name)
        info("Model loaded")

    acc = 0
    test_data, test_targets = torch.tensor(iris_data, dtype=torch.float), torch.tensor(iris_targets)
    with torch.no_grad():
        for i, data in (pbar:=tqdm(enumerate(test_data))):
            spike_data:Tensor = encoding_func(data, num_steps=num_steps) # type: ignore
            spk_rec, mem_rec = net(spike_data.view(num_steps, -1))
            idx = torch.sum(spk_rec, dim=0).argmax() # rate encoding prediction
            # print(idx, test_targets[i], flush=True)
            pbar.desc = f"{idx}, {test_targets[i]}"
            if idx == test_targets[i]:
                acc += 1
        info(f'Accuracy of the model : {100*acc/len(test_data):.2f}%\n')
    return net

def run_test(cfg:CFG):
    log_name = f"{strftime('%m%d%H%M', localtime())}_{cfg.log_name}.log"
    logging.basicConfig(filename="log/" + log_name, level=logging.INFO)
    info(cfg)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)

    #Load dsets
    iris = datasets.load_iris()
    iris_data = iris.data / iris.data.max(axis=0) # type: ignore
    iris_targets = iris.target # type: ignore
    if shuffle:
        assert len(iris_data) == len(iris_data)
        perm = np.random.permutation(len(iris_data))
        iris_data, iris_targets = iris_data[perm], iris_targets[perm]
    
    #Load or train snn iris net
    net = prepare_net(iris_data, iris_targets)
    
    # take a random input and make it into a spike train
    spike_indicators = gen_spikes()
    potentials = gen_potentials()
    weights = gen_weights([net.fc1.weight, net.fc2.weight])
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


