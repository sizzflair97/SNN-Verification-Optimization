from itertools import product
from tqdm.auto import tqdm
from z3 import *
from typing import List
from typing import cast as typecast
from .dictionary_iris import *
from .debug import info
import pdb

# def gen_spikes() -> TSpike:
#     spike_indicators = typecast(TSpike, {})
#     for timestep in tqdm(range(1, num_steps+1), desc="Generating spikes"):
#         for layer, n_layer_neuron in enumerate(n_layer_neurons):
#             layer_neurons = product(range(layer_shapes[layer][0]), range(layer_shapes[layer][1]))
#             for layer_neuron in layer_neurons:
#                 spike_indicators[layer_neuron, layer, timestep] = Bool(f'bSpk_{layer_neuron}_{layer}_{timestep}')
#     info("Spikes are generated.")
#     return spike_indicators

def get_layer_neurons_iter(layer:int):
    return product(range(layer_shapes[layer][0]), range(layer_shapes[layer][1]))

def gen_spike_times() -> TSpikeTime:
    """_summary_
    Generate spike times z3 Int terms. layer 0 is input layer and last layer is output layer.
    
    Returns:
        TSpikeTime: Dictionary including z3 Int terms.
    """
    spike_times = typecast(TSpikeTime, {})
    for layer, _ in enumerate(n_layer_neurons):
        for layer_neuron in get_layer_neurons_iter(layer):
            spike_times[layer_neuron, layer] = Int(f'dSpkTime_{layer_neuron}_{layer}')
    return spike_times

def gen_weights(weights_list:TWeightList) -> TWeight:
    weights = typecast(TWeight, {})
    print(num_steps, weights_list[0].shape, weights_list[1].shape)
    for in_layer in range(len(n_layer_neurons)-1):
        layer_weight = weights_list[in_layer]
        out_layer = in_layer+1
        for out_neuron in range(n_layer_neurons[out_layer]):
            in_neurons = get_layer_neurons_iter(in_layer)
            for in_neuron in in_neurons:
                weights[in_neuron, out_neuron, in_layer] = float(layer_weight[out_neuron, *in_neuron])
    info("Weights are generated.")
    return weights

def gen_node_eqns(weights:TWeight, spike_times:TSpikeTime) -> List[BoolRef|bool]:
    node_eqn:List[BoolRef|bool] = []
    for layer, _ in enumerate(n_layer_neurons):
        for neuron in tqdm(get_layer_neurons_iter(layer)):
            # out layer cannot spike in first "layer" steps.
            node_eqn.extend([spike_times[neuron, layer]>=layer, spike_times[neuron, layer]<=num_steps-1])
            
    for out_layer, n_out_layer_neurons in enumerate(n_layer_neurons[1:], start=1):
        in_layer = out_layer-1
        for out_neuron_pos in tqdm(range(n_out_layer_neurons),
                                   desc="Generating node equations. Nodes"):
            out_neuron = (out_neuron_pos,0) # We only use position 0 in dimension 1 for layer output.
            flag:List[BoolRef|bool] = [False]
            # Does not include last step: [0,num_steps-1]
            for timestep in tqdm(range(out_layer, num_steps-1), desc="Timestep", leave=False):
                time_cumulated_potential = []
                for in_neuron in get_layer_neurons_iter(in_layer):
                    time_cumulated_potential.append(If(
                        spike_times[in_neuron, in_layer] <= (timestep-1),
                        weights[in_neuron, out_neuron_pos, in_layer], 0))
                over_threshold = Sum(time_cumulated_potential) >= threshold
                spike_condition = And(Not(Or(flag)), over_threshold)
                flag.append(over_threshold)
                term = typecast(BoolRef,
                                spike_condition == (spike_times[out_neuron, out_layer]==timestep))
                node_eqn.append(term)
            # Force spike in last timestep.
            term = typecast(
                    BoolRef,
                    Not(Or(flag))
                    == (spike_times[out_neuron, out_layer] == num_steps-1))
            node_eqn.append(term)
    info("Node equations are generated.")
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

# It assumes there is at least one True.
# it encodes ttfs range too.
def ttfs(vector:List[BoolRef|bool], ttfs_index:ArithRef) -> List[BoolRef|bool]:
    n = len(vector)
    eqns:List[BoolRef|bool] = [typecast(BoolRef, Or([i==ttfs_index for i in range(n)]))]
    for i in range(n):
        eqns.append(
            (i==ttfs_index) == And(
                vector[i],
                And([Not(vector[j]) for j in range(i)])
                ))
    return eqns
