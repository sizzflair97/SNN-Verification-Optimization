from itertools import product
from math import floor, log
from types import MappingProxyType
import typing
import numpy as np
from torch import Tensor
from tqdm.auto import tqdm
from z3 import *
from typing import Any, Dict, Literal, Set, Tuple, List, DefaultDict, Union
from typing import cast as typecast
from collections import defaultdict
from .dictionary_mnist import *
from .config import CFG
from .mnist_net import MnistNet as Net
from .debug_utils import info
import pdb

def gen_spikes() -> TSpike:
    spike_indicators = typecast(TSpike, {})
    for timestep in tqdm(range(1, num_steps+1), desc="Generating spikes"):
        for layer, n_layer_neuron in enumerate(n_layer_neurons):
            layer_neurons = product(range(layer_shapes[layer][0]), range(layer_shapes[layer][1]))
            for layer_neuron in layer_neurons:
                spike_indicators[layer_neuron, layer, timestep] = Bool(f'bSpk_{layer_neuron}_{layer}_{timestep}')
    info("Spikes are generated.")
    return spike_indicators

def gen_spike_times() -> TSpikeTime:
    spike_times = typecast(TSpikeTime, {})
    for layer, n_layer_neuron in enumerate(n_layer_neurons):
        layer_neurons = product(range(layer_shapes[layer][0]), range(layer_shapes[layer][1]))
        for layer_neuron in layer_neurons:
            spike_times[layer_neuron, layer] = Int(f'dSpkTime_{layer_neuron}_{layer}')
    return spike_times

def gen_weights(weights_list:TWeightList) -> TWeight:
    weights = typecast(TWeight, {})
    for in_layer in range(len(n_layer_neurons)-1):
        layer_weight = weights_list[in_layer]
        out_layer = in_layer+1
        for out_neuron in range(n_layer_neurons[out_layer]):
            in_neurons = product(range(layer_shapes[in_layer][0]), range(layer_shapes[in_layer][1]))
            for in_neuron in in_neurons:
                weights[in_neuron, out_neuron, in_layer] = float(layer_weight[out_neuron, *in_neuron])
    info("Weights are generated.")
    return weights

def gen_node_eqns(weights:TWeight, spike_times:TSpikeTime):
    node_eqn:List[BoolRef|bool] = []
    for layer,n_layer_neuron in enumerate(n_layer_neurons):
        neurons = product(range(layer_shapes[layer][0]),
                          range(layer_shapes[layer][1]))
        for neuron in neurons:
            term = typecast(BoolRef,
                            And(spike_times[neuron, layer] <= num_steps,
                                spike_times[neuron, layer] >= 0))
            node_eqn.append(term)
    
    for out_layer, n_out_layer_neurons in enumerate(n_layer_neurons[1:], start=1):
        in_layer = out_layer-1
        for out_neuron_pos in tqdm(range(n_out_layer_neurons),
                                   desc="Generating node equations. Nodes"):
            out_neuron = (out_neuron_pos,0) # We only use position 0 in dimension 1 for layer output.
            time_cumulated_potential:ArithRef = RealVal(0)
            flag = False
            # Does not include last step: [0,num_steps-1]
            for timestep in tqdm(range(num_steps), desc="Timestep", leave=False):
                in_neurons = product(
                    range(layer_shapes[in_layer][0]),
                    range(layer_shapes[in_layer][1]))
                for in_neuron in in_neurons:
                    time_cumulated_potential += If(
                        spike_times[in_neuron, in_layer] == timestep,
                        weights[in_neuron, out_neuron_pos, in_layer], 0)
                over_threshold = time_cumulated_potential >= threshold
                spike_condition = And(Not(flag),
                                      over_threshold)
                flag = Or(flag,
                          over_threshold)
                term = typecast(BoolRef,
                                spike_condition == (spike_times[out_neuron, out_layer]==timestep))
                node_eqn.append(term)
            # Force spike in last timestep.
            term = typecast(
                    BoolRef,
                    Not(flag)
                    == (spike_times[out_neuron, out_layer] == num_steps))
            # node_eqn.append(term)
    info("Node equations are generated.")
    return node_eqn

# def gen_node_eqns(weights:TWeight, spike_indicators:TSpike, spike_times:TSpikeTime):
#     node_eqn:List[BoolRef|bool] = []
#     for out_layer, n_out_layer_neurons in enumerate(n_layer_neurons[1:], start=1):
#         in_layer = out_layer-1
#         for out_neuron_pos in tqdm(range(n_out_layer_neurons),
#                                    desc="Generating node equations. Nodes"):
#             out_neuron = (out_neuron_pos,0) # We only use position 0 in dimension 1 for layer output.
#             time_cumulated_potential:ArithRef = RealVal(0)
#             for timestep in tqdm(range(1, num_steps+1), desc="Timestep", leave=False):
#                 current:ArithRef = RealVal(0)
#                 in_neurons = product(
#                     range(layer_shapes[in_layer][0]),
#                     range(layer_shapes[in_layer][1]))
#                 for in_neuron in in_neurons:
#                     current += (spike_indicators[in_neuron, in_layer, timestep] 
#                                 * weights[in_neuron, out_neuron_pos, in_layer])
#                 spike_condition = ((time_cumulated_potential < threshold) if timestep == num_steps
#                                    else And(time_cumulated_potential < threshold,
#                                             time_cumulated_potential + current >= threshold))
#                 term = typecast(
#                     BoolRef,
#                     And(spike_condition
#                         == spike_indicators[out_neuron, out_layer, timestep],
#                         spike_indicators[out_neuron, out_layer, timestep]
#                         == (spike_times[out_neuron, out_layer] == timestep)))
#                 node_eqn.append(term)
#                 time_cumulated_potential += current
#             #Force spike in last timestep.
#             node_eqn.append(
#                 And((time_cumulated_potential < threshold)
#                     == spike_indicators[out_neuron, out_layer, timestep],
#                     spike_indicators[out_neuron, out_layer, timestep]
#                     == (spike_times[out_neuron, out_layer] == timestep)))
#             # pdb.set_trace()
#     info("Node equations are generated.")
#     return node_eqn

def gen_output_props(): pass

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
