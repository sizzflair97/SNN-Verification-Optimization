# %%
import functools, time, logging, json
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

def info(msg:Any):
    print(msg) or logging.getLogger().info(msg) # type: ignore

def prepare_net(iris_data, iris_targets) -> Net:
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
                    loss_val += loss(mem_rec[-1][step], targets)

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
    return net

def run_test(cfg:CFG, log_name:Union[None, str]=None):
    if log_name == None:
        log_name = f"TEST_{strftime('%m%d%H%M', localtime())}_{cfg.log_name}.log"
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
    spike_indicators = gen_s_indicator()
    potentials = gen_p_indicator()
    weights = gen_w_indicator([net.fc1.weight, net.fc2.weight])
    pot_init = gen_initial_potential_term(potentials)

    #Node eqns
    assign:List[BoolRef] = []
    node_eqn:List[BoolRef] = gen_node_eqns(weights, spike_indicators, potentials)
    if cfg.use_DNP:
        node_eqn += gen_DNP(weights, spike_indicators)

    #Randomly draw samples
    samples = iris_data[np.random.choice(range(len(iris_data)), cfg.num_samples)] # type: ignore
    info(samples)


    delta_v = {d: 0 for d in cfg.deltas}
    for delta in cfg.deltas:
        avt = 0
        for sample_no, sample in enumerate(samples):
            sample_spike = spikegen.rate(torch.tensor(sample, dtype=torch.float), num_steps=num_steps) # type: ignore
            
            label, control = forward_net(sample_spike.view(num_steps, -1), spike_indicators, assign+node_eqn+pot_init)
            # spk_rec, mem_rec = net(sample_spike.view(num_steps, -1)) # epsilon 1~5
            # label = int(spk_rec.sum(dim=0).argmax())
            prop = gen_delta_reuse(cfg, sample_spike, spike_indicators, potentials, delta, control)
            # Output property
            #tx = time.time()
            op = []
            intend_sum = sum([2 * spike_indicators[(control[label].as_long(), 2, timestep + 1)] for timestep in range(num_steps)]) # type: ignore
            for t in range(num_output):
                if t != op:
                    op.append(
                        Not(intend_sum > sum([2 * spike_indicators[(t, 2, timestep + 1)] for timestep in range(num_steps)]))
                    )
            #print(f'Output Property Done in {time.time() - tx} sec')
            
            S = Solver()
            S.add(assign+node_eqn+pot_init+prop+op)
            tx = time.time()
            res = S.check()
            if str(res) == 'unsat':
                delta_v[delta] += 1
            # else:
            #     model = S.model()
            #     dump(model, spike_indicators, potentials)
            #     pdb.set_trace()
            del S
            tss = time.time()-tx
            # print(f'Completed for delta = {delta}, sample = {sample_no} in {tss} sec as {res}')
            info(f'Completed for delta = {delta}, sample = {sample_no} in {tss} sec as {res}')
            avt = (avt*sample_no + tss)/(sample_no+1)
        # print(f'Completed for delta = {delta} with {delta_v[delta]} in avg time {avt} sec')
        info(f'Completed for delta = {delta} with {delta_v[delta]} in avg time {avt} sec')

    print()


