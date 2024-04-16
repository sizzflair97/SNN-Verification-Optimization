from math import floor, log
import uuid
from torch import Tensor
from z3 import *
from typing import Any, Dict, Literal, Tuple, List, DefaultDict, Union
from collections import defaultdict
from functools import reduce
from .dictionary import *
from .config import CFG
from . import z3_utils_hakank as z3utils
import pdb

def gen_s_indicator():
    spike_indicators:SType = {}
    for t in range(1, num_steps+1):
        for j, m in enumerate(layers):
            for i in range(m):
                spike_indicators[(i, j, t)] = Bool(f'x_{i}_{j}_{t+1}')
    return spike_indicators

def gen_p_indicator():
    potentials:PType = {}
    for t in range(num_steps+1):
        for j, m in enumerate(layers):
            if j == 0:
                continue
            for i in range(m):
                potentials[(i, j, t)] = Real(f'P_{i}_{j}_{t}')
    return potentials

def gen_w_indicator(weights_list:List[Tensor]):
    weights:WType = defaultdict(float)
    for l, w in enumerate(weights_list):
        for j in range(len(w)):
            for i in range(len(w[j])):
                weights[(i, j, l)] = w[j][i].item()
    return weights

def gen_initial_potential_term(potentials:PType):
    pot_init:List[Union[BoolRef, Any]] = []
    for j, m in enumerate(layers):
        if j == 0:
            continue
        for i in range(m):
            pot_init.append(potentials[(i, j, 0)] == 0)
    return pot_init

def gen_latency_encoding_props(spike_indicators:SType):
    eqn:List[BoolRef] = []
    for idx_node in range(layers[0]):
        # for timestep in range(1, num_steps+1):
        #     eqn.append(
        #         Implies(spike_indicators[(idx_node, 0, timestep)],
        #                 And([Not(spike_indicators[(idx_node, 0, other)])
        #                     for other in [*range(1,timestep)]+[*range(timestep+1,num_steps+1)]]))) # type: ignore
        eqn.append(
            Sum([1*spike_indicators[(idx_node, 0, timestep)] for timestep in range(1, num_steps+1)]) == 1 # type: ignore
        )
    return eqn

def gen_DNP(weights:WType, spike_indicators:SType):
    node_eqn:List[BoolRef] = []
    total_dns = 0
    dns = []
    for in_layer, (n_in_nodes, n_out_nodes) in enumerate(zip(layers[:-1], layers[1:])):
        prev_dns = dns # List to save dead neurons of prev. layer.
        dns = [] # List to save dead neurons of current layer.
        for i in range(n_out_nodes):
            # do not calc neurons to get max_current in prev_dns, because there are only dead neurons.
            S_max = sum(max(0, weights[(k, i, in_layer)]) for k in range(n_in_nodes) if (k, in_layer) not in prev_dns)
            if S_max == 0 or 1-threshold*(1-beta)/(S_max) <= 0:
                total_dns += 1
                dns.append((i, in_layer+1)) # save dead neuron: (node_idx, layer_idx)
                node_eqn.append(
                    Not(Or([spike_indicators[(i, in_layer+1, t)]
                            for t in range(1, num_steps+1)]))) # type: ignore
            # Previous Implementation. Equal but more terms.
            # node_eqn.append(
            #     Implies(
            #         Sum([If(weights[(k, i, in_layer)]>=0, weights[(k, i, in_layer)], 0)
            #              for k in range(n_in_nodes)])
            #         < threshold * (1-beta),
            #         Not(Or([spike_indicators[(i, in_layer+1, t)]
            #                 for t in range(1, num_steps+1)]))))
    print(f"Total Dead Neurons: {total_dns}")
    return node_eqn

def gen_GNP(weights:WType, spike_indicators:SType):
    node_eqn:List[BoolRef] = []
    total_dns = 0
    dns = []
    for in_layer, (n_in_nodes, n_out_nodes) in enumerate(zip(layers[:-1], layers[1:])):
        prev_dns = dns # List to save dead neurons of prev. layer.
        dns = [] # List to save dead neurons of current layer.
        for i in range(n_out_nodes):
            S_max = sum(max(0, weights[(k, i, in_layer)]) for k in range(n_in_nodes) if (k, in_layer) not in prev_dns)
            if S_max == 0 or (score:=1-threshold*(1-beta)/(S_max)) <= 0:
                total_dns += 1
                dns.append((i, in_layer+1)) # save dead neuron: (node_idx, layer_idx)
                node_eqn.append(
                    Not(Or([spike_indicators[(i, in_layer+1, t)]
                            for t in range(1, num_steps+1)]))) # type: ignore
                continue
            n_max = floor(log(score, beta))
            for t in range(1, num_steps+1):
                node_eqn.append(
                    Implies(spike_indicators[(i, in_layer+1, t)],
                            And([Not(spike_indicators[(i, in_layer+1, tp)])
                                for tp in range(t+1, min(t+n_max, num_steps+1))]))
                )
    return node_eqn

def gen_node_eqns(weights:WType, spike_indicators:SType, potentials:PType):
    node_eqn:List[BoolRef] = []
    for t in range(1, num_steps+1):
        for j, m in enumerate(layers[1:], start=1):
            for i in range(m):
                dP2i_lst = [spike_indicators[(k, j-1, t)]*weights[(k, i, j-1)] for k in range(layers[j-1])]
                S = sum(dP2i_lst) + beta*potentials[(i, j, t-1)] # type: ignore # epsilon_1
                reset = S >= threshold
                node_eqn.append(
                    And(Implies(reset,
                                And(spike_indicators[(i, j, t)], potentials[(i, j, t)] == S - 1)), # epsilon_2 & epsilon_4
                        Implies(Not(reset),
                                And(Not(spike_indicators[(i, j, t)]), potentials[(i, j, t)] == S)))) # type: ignore # epsilon_3 & epsilon_5
    return node_eqn

def argmax_left(s:Solver, x:List[ArithRef], ix:ArithRef, max_val:ArithRef):
    z3utils.maximum(s, max_val, x)
    n = len(x)
    for _i in range(n):
        _ne_left = And([max_val != _l for _l in x[:_i]])
        s.add(
            Implies(And(_ne_left,
                        x[_i] == max_val),
                    _i==ix))

def node_first_spike_time(x:List[BoolRef]):
    fst = Int(f"Fst_{uuid.uuid4().int}")
    eqn:List[BoolRef] = []
    flags:List[BoolRef] = []
    flag = False
    idx = 0
    while idx < len(x):
        prev_flag = flag
        flag = Or(flag,(x[idx]==True)) # flip flag if there is spike.
        flags.append(flag) # type: ignore
        eqn.append(Implies(prev_flag!=flag, fst==idx)) # flipped flag means first spike is in idx.
    eqn.append(Implies(Not(flag), fst==num_steps+1)) # first spike time is num_steps+1 if there is no spike in x.
    return fst, eqn
    
def first_spike_time(solver:Solver, spike_indicators:SType):
    #Save output spikes
    fsts:List[ArithRef] = []
    eqn:List[BoolRef] = []
    for i in range(layers[-1]):
        fst_i, eqn_i = node_first_spike_time([spike_indicators[(i, len(layers)-1, timestep)] for timestep in range(1, num_steps+1)])
        fsts.append(fst_i)
        eqn += eqn_i
    val_id = uuid.uuid4().int
    
    z3utils.argmax(solver, fsts, time:=Int(f'FirstSpikeTime_{val_id}'), model_out_max:=Int(f"OutMax_{val_id}"))
    return time, model_out_max

def gen_latency_output_validity(solver:Solver, spike_indicators:SType):
    eqn:List[BoolRef] = []
    for idx_node in range(layers[-1]):
        timeseries_of_node = [spike_indicators[(idx_node, 0, timestep)] for timestep in range(1, num_steps+1)]
        
        eqn.append(
            Sum([1*spike_indicators[(idx_node, 0, timestep)] for timestep in range(1, num_steps+1)]) == 1 # type: ignore
        )
    return eqn

def forward_net(sample_spike:torch.Tensor,
                spike_indicators:SType,
                encodings:List[BoolRef]):
    #solver preprocess
    solver = Solver()
    solver.add(encodings)
    
    #make spike input encoding
    for timestep, spike_train in enumerate(sample_spike, start=1):
        for i, spike in enumerate(spike_train.view(num_input)):
            solver.add(spike_indicators[(i, 0, timestep)]
                       == bool(spike.item()))
    label, model_out_max = first_spike_time(solver, spike_indicators)
    return label, solver.check(), solver.model()

def gen_delta_reuse(cfg:CFG,
                    sample_spike:Tensor,
                    spike_indicators:SType,
                    potentials:PType,
                    delta:int,
                    control:ModelRef):
    sum_val = []
    prop:List[BoolRef] = []
    reuse_flag = True
    for timestep, spike_train in enumerate(sample_spike):
        #Variables to calculate the total perturbation.
        for i, spike in enumerate(spike_train.view(num_input)):
            sum_val.append(If(spike_indicators[(i, 0, timestep + 1)] == bool(spike.item()), 0.0, 1.0))
            #Flip flag if there is any perturbation
            reuse_flag = And(reuse_flag,
                             spike_indicators[(i, 0, timestep + 1)] == spike.bool().item())
        
        #If Accumulation of Delta until current timestep is 0, reuse y_hat and potential of non-perturbated spike.
        if cfg.reuse_level != 0:
            _reuse_targets = []
            for node_i in range(layers[2]):
                _reuse_targets.append(
                    spike_indicators[(node_i, len(layers)-1, timestep+1)]\
                        == control[spike_indicators[(node_i, len(layers)-1, timestep+1)]]
                    )
            if cfg.reuse_level != 1:
                for layer_i, n_nodes in enumerate(layers[1:], 1):
                    for node_i in range(n_nodes):
                        _reuse_targets.append(
                            potentials[(node_i, layer_i, timestep+1)]
                            == control[potentials[(node_i, layer_i, timestep+1)]]) # type: ignore
            prop.append(Implies(
                reuse_flag,
                And(_reuse_targets)))
            
    prop.append(sum(sum_val) <= delta) # type: ignore
    return prop

def gen_delta_latency_reuse(cfg:CFG,
                    sample_spike:Tensor,
                    spike_indicators:SType,
                    delta:int):
    sum_val:List[ArithRef] = []
    prop:List[BoolRef] = []
    assert len(sample_spike.shape) == 2
    for node_i, spike_sequence in enumerate(sample_spike.T):
        orig = spike_sequence.argmax().item()
        terms = []
        for timestep, spike in enumerate(spike_sequence, start=1):
            terms.append(If(spike_indicators[(node_i, 0, timestep)], timestep, 0))
        sum_val.append(
            Abs(Sum(terms) - orig) # type: ignore
        )
            
    prop.append(sum(sum_val) <= delta) # type: ignore
    return prop