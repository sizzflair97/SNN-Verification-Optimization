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
    ttfses = typecast(TSpikeTime, {})
    for layer, n_layer_neuron in enumerate(n_layer_neurons):
        layer_neurons = product(range(layer_shapes[layer][0]), range(layer_shapes[layer][1]))
        for layer_neuron in layer_neurons:
            ttfses[layer_neuron, layer] = Int(f'dSpkTime_{layer_neuron}_{layer}')
    return ttfses

def gen_potentials() -> TPotential:
    potentials = typecast(TPotential, {})
    for timestep in tqdm(range(num_steps+1), desc="Generating potentials"):
        for layer, n_layer_neuron in enumerate(n_layer_neurons):
            if layer == 0:
                continue
            layer_neurons = product(range(layer_shapes[layer][0]), range(layer_shapes[layer][1]))
            for layer_neuron in layer_neurons:
                potentials[layer_neuron, layer, timestep] = Real(f'rPot_{layer_neuron}_{layer}_{timestep}') if timestep != 0 else RealVal(0)
    info("Potentials are generated.")
    return potentials

def gen_weights(weights_list:TWeightList):
    weights = typecast(TWeight, defaultdict(float))
    for in_layer in range(len(n_layer_neurons)-1):
        layer_weight = weights_list[in_layer]
        out_layer = in_layer+1
        for out_neuron in range(n_layer_neurons[out_layer]):
            in_neurons = product(range(layer_shapes[in_layer][0]), range(layer_shapes[in_layer][1]))
            for in_neuron in in_neurons:
                weights[in_neuron, out_neuron, in_layer] = float(layer_weight[out_neuron, *in_neuron])
    info("Weights are generated.")
    return weights

def gen_node_eqns(weights:TWeight, spike_indicators:TSpike, spike_times:TSpikeTime):
    node_eqn:List[BoolRef|bool] = []
    for out_layer, n_out_layer_neurons in enumerate(n_layer_neurons[1:], start=1):
        in_layer = out_layer-1
        for out_neuron_pos in tqdm(range(n_out_layer_neurons), desc="Generating node equations. Nodes"):
            out_neuron = (out_neuron_pos,0) # We only use position 0 in dimension 1 for layer output.
            time_potential:List[ArithRef] = [RealVal(0)]
            for timestep in tqdm(range(1, num_steps+1), desc="Timestep", leave=False):
                curr_vec = typecast(List[ArithRef], [])
                in_neurons = product(range(layer_shapes[in_layer][0]), range(layer_shapes[in_layer][1]))
                for in_neuron in in_neurons:
                    curr_vec.append(spike_indicators[in_neuron, in_layer, timestep]*weights[in_neuron, out_neuron_pos, in_layer]) # type: ignore
                curr = typecast(ArithRef,sum(curr_vec)) # epsilon_1
                time_potential.append(curr + time_potential[-1])
                node_eqn.append(
                    Implies(And(time_potential[-2] < threshold,
                                time_potential[-1] >= threshold),
                            And(spike_indicators[out_neuron, out_layer, timestep],
                                spike_times[out_neuron, out_layer] == timestep))
                )
            
            node_eqn.append(
                Implies(time_potential[-1] < threshold,
                        And(spike_indicators[out_neuron, out_layer, num_steps],
                            spike_times[out_neuron, out_layer] == num_steps)) # We always use position 0 in dimension 1.
            )
    info("Node equations are generated.")
    return node_eqn

def gen_output_props(): pass

def gen_dnp(weights:TWeight, spike_indicators:TSpike):
    node_eqn:List[BoolRef] = []
    total_dns = 0
    dns = []
    for in_layer, (n_in_nodes, n_out_nodes) in enumerate(zip(n_layer_neurons[:-1], n_layer_neurons[1:])):
        prev_dns = dns # List to save dead neurons of prev. layer.
        dns = [] # List to save dead neurons of current layer.
        for i in range(n_out_nodes):
            # do not calc neurons to get max_current in prev_dns, because there are only dead neurons.
            curr_max = sum(max(0, weights[(k, i, in_layer)]) for k in range(n_in_nodes) if (k, in_layer) not in prev_dns)
            if curr_max <= 0 or 1-threshold*(1-beta)/(curr_max) <= 0:
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
    info(f"Total Dead Neurons: {total_dns}")
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
