# %%
import functools
import time
from time import localtime, strftime
from sklearn import datasets
from snntorch import spikegen
from snntorch import functional as SF
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from z3 import *
from collections import defaultdict
from utils import *
import logging

def run_test(_cfg:cfg):
    logging.basicConfig(filename=f"log/TEST_{strftime('%m%d%H%M', localtime())}_{_cfg.log_name}.log", level=logging.INFO)
    logger = logging.getLogger()
    info = lambda msg: print(msg) or logger.info(msg) # type: ignore
    info(_cfg)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

    iris = datasets.load_iris()

    iris_data = iris.data / iris.data.max(axis=0)
    iris_targets = iris.target

    if shuffle:
        assert len(iris_data) == len(iris_data)
        perm = np.random.permutation(len(iris_data))
        iris_data, iris_targets = iris_data[perm], iris_targets[perm]

    loss_hist = []
    test_loss_hist = []
    counter = 0

    if train:
        net = Net()
        optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))
        #loss = nn.CrossEntropyLoss()
        loss = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2) # type: ignore

        # Outer training loop
        for epoch in range(num_epochs):
            iter_counter = 0

            # Minibatch training loop
            for number in range(len(iris_targets)):
                data = torch.tensor(iris_data[number], dtype=torch.float)
                #targets = torch.tensor([0 if i != iris_targets[number] else 1 for i in range(max(iris_targets)+1)],dtype=torch.float)
                targets = torch.tensor([iris_targets[number]])

                # make spike trains
                data_spike = spikegen.rate(data, num_steps=num_steps) # type: ignore

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
        # print("Saving model.pth")
        info("Saving model.pth")
        torch.save(net, file_name)
    else:
        net = torch.load(file_name)
        # print("Model loaded")
        info("Model loaded")

    check = True
    if check:
        acc = 0
        perm = np.random.permutation(len(iris_data))
        test_data, test_targets = torch.tensor(iris_data[perm][:100], dtype=torch.float), torch.tensor(iris_targets[perm][:100])
        for i, data in enumerate(test_data):
            spike_data = spikegen.rate(data, num_steps=num_steps) # type: ignore
            spk_rec, mem_rec = net(spike_data.view(num_steps, -1))
            idx = np.argmax(spk_rec.sum(dim=0).detach().numpy())
            if idx == test_targets[i]:
                acc += 1
            else:
                pass
        info(f'Accuracy of the model : {acc}%')

    info("")
    
    # take a random input and make it into a spike train
    spike_indicators = gen_s_indicator()
    potentials = gen_p_indicator()
    weights = gen_w_indicator([net.fc1.weight, net.fc2.weight])
    pot_init = gen_initial_potential_term(potentials)

    assign:List[BoolRef] = []

    # Node eqn
    node_eqn:List[BoolRef] = []
    if _cfg.use_DNP:
        node_eqn += gen_DNP(weights, spike_indicators)
    node_eqn += gen_node_eqn(weights, spike_indicators, potentials)

    samples = iris_data[np.random.choice(range(len(iris_data)), _cfg.num_samples)]
    info(samples)

    delta_v = {d: 0 for d in _cfg.deltas}

    for delta in _cfg.deltas:
        avt = 0
        for sample_no, sample in enumerate(samples):
            sample_spike = spikegen.rate(torch.tensor(sample, dtype=torch.float), num_steps=num_steps) # type: ignore

            spk_rec, mem_rec = net(sample_spike.view(num_steps, -1)) # epsilon 1~5
            label = int(spk_rec.sum(dim=0).argmax())

            S = Solver()
            S.add(assign+node_eqn+pot_init)
            # S.add(node_eqn + pot_init)

            sum_val = []
            prop = []
            reuse_flag = True
            for timestep, spike_train in enumerate(sample_spike):
                #Variables to calculate the total perturbation.
                for i, spike in enumerate(spike_train.view(num_input)):
                    if spike == 1:
                        sum_val.append(If(spike_indicators[(i, 0, timestep + 1)], 0.0, 1.0))
                    else:
                        sum_val.append(If(spike_indicators[(i, 0, timestep + 1)], 1.0, 0.0))
                    #Flip flag if there is any perturbation
                    reuse_flag = And(
                        reuse_flag,
                        spike_indicators[(i, 0, timestep + 1)]==spike.bool().item()
                        )
                
                #If Accumulation of Delta until current timestep is 0, reuse y_hat of non-perturbated spike.
                if _cfg.reuse_level != 0:
                    _reuse_targets = []
                    for _out_node in range(layers[2]):
                        _reuse_targets.append(
                            spike_indicators[(_out_node, 2, timestep+1)]\
                                == spk_rec[timestep, _out_node].bool().item()
                            )
                        if _cfg.reuse_level == 1: continue
                        _reuse_targets.append(
                            potentials[(_out_node, 2, timestep+1)]\
                                == mem_rec[timestep, _out_node].item()
                        )
                    prop.append(
                        Implies(
                            reuse_flag,
                            And(_reuse_targets)
                            )
                        )
                    
            prop.append(sum(sum_val) <= delta)
            S.add(prop)

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
            del S
            tss = time.time()-tx
            # print(f'Completed for delta = {delta}, sample = {sample_no} in {tss} sec as {res}')
            info(f'Completed for delta = {delta}, sample = {sample_no} in {tss} sec as {res}')
            avt = (avt*sample_no + tss)/(sample_no+1)
        # print(f'Completed for delta = {delta} with {delta_v[delta]} in avg time {avt} sec')
        info(f'Completed for delta = {delta} with {delta_v[delta]} in avg time {avt} sec')

    print()


