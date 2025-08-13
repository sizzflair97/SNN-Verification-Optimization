from itertools import product
from tqdm.auto import tqdm
from z3 import *
from typing import overload
from typing import cast as typecast
from .dictionary_mnist import *
from .debug import info

def get_layer_neurons_iter(layer: int) -> Iterable[tuple[int, int]]:
    return product(range(layer_shapes[layer][0]), range(layer_shapes[layer][1]))

def gen_spike_times() -> TSpikeTime:
    """_summary_
    Generate spike times z3 Int terms. layer 0 is input layer and last layer is output layer.

    Returns:
        TSpikeTime: Dictionary including z3 Int terms.
    """
    spike_times = TSpikeTime()
    for layer, _ in enumerate(n_layer_neurons):
        for neuron in get_layer_neurons_iter(layer):
            spike_times[neuron, layer] = Int(f"dSpkTime_{neuron}_{layer}")
    return spike_times

def gen_weights(weights_list: TWeightList) -> TWeight:
    weights = TWeight()
    print(num_steps, weights_list[0].shape, weights_list[1].shape)
    for prev_layer in range(len(n_layer_neurons) - 1):
        layer_weight = weights_list[prev_layer]
        post_layer = prev_layer + 1
        for post_neuron in range(n_layer_neurons[post_layer]):
            prev_neurons = get_layer_neurons_iter(prev_layer)
            for prev_neuron in prev_neurons:
                weights[prev_neuron, post_neuron, prev_layer] = float(
                    layer_weight[post_neuron, *prev_neuron]
                )
    info("Weights are generated.")
    return weights


def gen_node_eqns(weights: TWeight, spike_times: TSpikeTime) -> list[BoolRef | bool]:
    tau: int = 1
    node_eqn = list[BoolRef | bool]()
    for layer, _ in enumerate(n_layer_neurons):
        for neuron in tqdm(get_layer_neurons_iter(layer)):
            # out layer cannot spike in first "layer" steps.
            node_eqn.extend(
                [
                    spike_times[neuron, layer] >= layer * tau,
                    spike_times[neuron, layer] <= num_steps - 1,
                ]
            )

    for post_layer, n_out_layer_neurons in enumerate(n_layer_neurons[1:], start=1):
        prev_layer = post_layer - 1
        for post_neuron_pos in tqdm(
            range(n_out_layer_neurons), desc="Generating node equations. Nodes"
        ):
            post_neuron = (
                post_neuron_pos,
                0,
            )  # We only use position 0 in dimension 1 for layer output.
            flag = list[BoolRef | bool]([False])
            # Does not include last step: [0,num_steps-1]
            for timestep in tqdm(
                range(post_layer, num_steps - 1), desc="Timestep", leave=False
            ):
                time_cumulated_potential = []
                for in_neuron in get_layer_neurons_iter(prev_layer):
                    time_cumulated_potential.append(
                        If(
                            spike_times[in_neuron, prev_layer] <= (timestep - 1),
                            weights[in_neuron, post_neuron_pos, prev_layer],
                            0,
                        )
                    )
                over_threshold = Sum(time_cumulated_potential) >= threshold
                spike_condition = And(Not(Or(flag)), over_threshold)
                flag.append(over_threshold)
                term = typecast(
                    BoolRef,
                    spike_condition == (spike_times[post_neuron, post_layer] == timestep),
                )
                node_eqn.append(term)
            # Force spike in last timestep.
            term = typecast(
                BoolRef,
                Not(Or(flag)) == (spike_times[post_neuron, post_layer] == num_steps - 1),
            )
            node_eqn.append(term)
    info("Node equations are generated.")
    return node_eqn