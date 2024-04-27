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

def gen_initial_potential_terms(potentials:PType):
    pot_init:List[Union[BoolRef, Any]] = []
    for j, m in enumerate(layers):
        if j == 0:
            continue
        for i in range(m):
            pot_init.append(potentials[(i, j, 0)] == 0)
    return pot_init

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
            # Previous Implementation. Equal but with more terms.
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

def maximum(v, x) -> List[BoolRef]:
    eqns:List[BoolRef] = [Or([v == x[i] for i in range(len(x))])] # type: ignore
    for i in range(len(x)):
        eqns.append(v >= x[i]) # and it's the greatest
    return eqns

def argmax_left(vector:List[ArithRef], max_index:ArithRef, max_val:ArithRef) -> List[BoolRef]:
    eqns = maximum(max_val, vector)
    n = len(vector)
    for i in range(n):
        _ne_left = And([max_val != _l for _l in vector[:i]])
        eqns.append(
            Implies(And(_ne_left,
                        vector[i] == max_val),
                    i==max_index))
    return eqns

def forward_net(sample_spike:torch.Tensor,
                spike_indicators:SType,
                encodings:List[BoolRef]) -> \
                    Tuple[Literal["sat", "unsat", "unknown"], ArithRef, ModelRef]:
    #solver preprocess
    solver = Solver()
    solver.add(encodings)
    
    #make spike input encoding
    spk_outs:List[ArithRef] = [0] * layers[-1] # type: ignore
    for _timestep, _spike_train in enumerate(sample_spike):
        for _i, _spike in enumerate(_spike_train.view(num_input)):
            solver.add(spike_indicators[(_i, 0, _timestep+1)]
                       == bool(_spike.item()))
        for _i in range(layers[-1]):
            spk_outs[_i] += If(spike_indicators[(_i, len(layers)-1, _timestep+1)], 1, 0)
    
    #add argmax encoding
    max_label_spk = Int('Max_Label_Spike')
    label = Int('Label')
    solver.add(argmax_left(spk_outs, label, max_label_spk))
    return str(solver.check()), label, solver.model() # type: ignore

def gen_delta_reuse(cfg:CFG,
                    sample_spike:Tensor,
                    spike_indicators:SType,
                    potentials:PType,
                    delta:int,
                    control:ModelRef):
    delta_vector = []
    prop:List[BoolRef] = []
    reuse_flag = True
    for timestep, spike_train in enumerate(sample_spike):
        #Variables to calculate the total perturbation.
        for i, spike in enumerate(spike_train.view(num_input)):
            delta_vector.append(If(spike_indicators[(i, 0, timestep + 1)] == bool(spike.item()), 0.0, 1.0))
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
            
    prop.append(sum(delta_vector) <= delta) # type: ignore
    return prop