from torch import Tensor
from z3 import *
from typing import Any, Dict, Tuple, List, DefaultDict, Union
from collections import defaultdict

from .dictionary import *

def gen_s_indicator():
    spike_indicators = {}
    for t in range(num_steps):
        for j, m in enumerate(layers):
            for i in range(m):
                spike_indicators[(i, j, t+1)] = Bool(f'x_{i}_{j}_{t+1}')
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
                weights[(i, j, l)] = float(w[j][i])
    return weights

def gen_initial_potential_term(potentials:PType):
    pot_init:List[Union[BoolRef, Any]] = []
    for j, m in enumerate(layers):
        if j == 0:
            continue
        for i in range(m):
            pot_init.append(potentials[(i, j, 0)] == 0)
    return pot_init

def gen_DNP(weights:WType, spike_indicators:SType):
    node_eqn:List[BoolRef] = []
    for _in_layer, (_in_nnodes, _out_nnodes) in enumerate(zip(layers[:-1], layers[1:])):
        for i in range(_out_nnodes):
            node_eqn.append(
                Implies(
                    Sum(
                        [If(weights[(k, i, _in_layer)]>=0, weights[(k, i, _in_layer)], 0)
                         for k in range(_in_nnodes)])
                    < 1 * (1-0.95), # threshold * (1-lambda)
                    Not(Or(
                        [spike_indicators[(i, _in_layer+1, t)]
                         for t in range(1, num_steps+1)]))
                )
            )
    return node_eqn

def gen_node_eqn(weights:WType, spike_indicators:SType, potentials:PType):
    node_eqn:List[BoolRef] = []
    for t in range(1, num_steps+1):
        for j, m in enumerate(layers):
            if j == 0:
                continue

            for i in range(m):
                S = sum([spike_indicators[(k, j-1, t)]*weights[(k, i, j-1)] for k in range(layers[j-1])]) + beta*potentials[(i, j, t-1)] # type: ignore # epsilon_1
                node_eqn.append(
                    And(
                        Implies(
                            S >= 1.0,
                            And(spike_indicators[(i, j, t)], potentials[(i, j, t)] == S - 1) # epsilon_2 & epsilon_4
                        ),
                        Implies(
                            S < 1.0,
                            And(Not(spike_indicators[(i, j, t)]), potentials[(i, j, t)] == S) # epsilon_3 & epsilon_5
                        )
                    ) # type: ignore
                )
    return node_eqn