from math import floor, log
from types import MappingProxyType
import uuid
from torch import Tensor
from z3 import *
from typing import Any, Dict, Literal, Set, Tuple, List, DefaultDict, Union
from collections import defaultdict
from .dictionary_mnist import *
from .config import CFG
from .mnist_net import MnistNet as Net
import pdb

def gen_spikes() -> SpkType:
    spike_indicators:SpkType = SpkType({})
    for t in range(num_steps):
        for j, m in enumerate(neurons_in_layers):
            if j == 0:
                for i in range(m):
                    spike_indicators[(i, j, t + 1)] = Bool(f'bSpk_{i}_{j}_{t + 1}')
            else:
                for i in range(m):
                    spike_indicators[(i, j, t+1)] = Bool(f'bSpk_{i}_{j}_{t+1}')

    return spike_indicators

def gen_potentials() -> PotType:
    potentials:PotType = PotType({})
    for t in range(num_steps+1):
        for j, m in enumerate(neurons_in_layers):
            if j == 0:
                continue
            for i in range(m):
                potentials[(i, j, t)] = Real(f'rPot_{i}_{j}_{t}')
    return potentials

def gen_weights(net:Net):
    weights:WeightType = WeightType(defaultdict(float))
    for k in range(0, len(neurons_in_layers)-1):
        w = net.fc_layers[k].weight
        for j in range(len(w)):
            for i in range(len(w[j])):
                weights[(i, j, k)] = float(w[j][i])
    return weights

def gen_initial_potentials(potentials:PotType):
    pot_init:List[Union[BoolRef, Any]] = []
    for j, m in enumerate(layers):
        if j == 0:
            continue
        for i in range(m):
            pot_init.append(potentials[(i, j, 0)] == 0)
    return pot_init

def gen_dnp(weights:WeightType, spike_indicators:SpkType):
    node_eqn:List[BoolRef] = []
    total_dns = 0
    dns = []
    for in_layer, (n_in_nodes, n_out_nodes) in enumerate(zip(layers[:-1], layers[1:])):
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
    print(f"Total Dead Neurons: {total_dns}")
    return node_eqn

def gen_dnp_v2(weights:WeightType, spike_indicators:SpkType, potentials:PotType):
    node_eqn:List[BoolRef] = []
    total_dns = 0
    dns = []
    for in_layer, (n_in_nodes, n_out_nodes) in enumerate(zip(layers[:-1], layers[1:])):
        prev_dns = dns # List to save dead neurons of prev. layer.
        dns = [] # List to save dead neurons of current layer.
        for i in range(n_out_nodes):
            # do not calc neurons to get max_current in prev_dns, because there are only dead neurons.
            curr_max = sum(max(0, weights[(k, i, in_layer)]) for k in range(n_in_nodes) if (k, in_layer) not in prev_dns)
            if curr_max <= 0 or 1-threshold*(1-beta)/(curr_max) <= 0:
                total_dns += 1
                dns.append((i, in_layer+1)) # save dead neuron: (node_idx, layer_idx)
                # node_eqn.append(
                #     Not(Or([spike_indicators[(i, in_layer+1, t)]
                #             for t in range(1, num_steps+1)]))) # type: ignore
                for t in range(1, num_steps+1):
                    spike_indicators[(i, in_layer+1, t)] = False
                    potentials.pop((i, in_layer+1, t))

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

def gen_gnp(weights:WeightType, spike_indicators:SpkType):
    node_eqn:List[BoolRef] = []
    total_dns = 0
    dns = []
    for in_layer, (n_in_nodes, n_out_nodes) in enumerate(zip(layers[:-1], layers[1:])):
        prev_dns = dns # List to save dead neurons of prev. layer.
        dns = [] # List to save dead neurons of current layer.
        for i in range(n_out_nodes):
            curr_max = sum(max(0, weights[(k, i, in_layer)]) for k in range(n_in_nodes) if (k, in_layer) not in prev_dns)
            if curr_max <= 0 or (score:=1-threshold*(1-beta)/(curr_max)) <= 0:
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

def gen_gnp_v2(weights:WeightType, spike_indicators:SpkType):
    node_eqn:List[Union[BoolRef,Literal[False]]] = []
    # inactivities = gen_inactivities()
    
    # for timestep in range(1, num_steps+1):
    #     for i_layer, n_nodes_in_layer in enumerate(layers[1:], start=1):
    #         for i_node in range(n_nodes_in_layer):
    #             node_eqn.append(Implies(inactivities[(i_node,i_layer,timestep)], Not(spike_indicators[(i_node,i_layer,timestep)]))) # if stable, then off spike
    
    for timestep in range(1, num_steps+1):
        for in_layer, (n_in_nodes, n_out_nodes) in enumerate(zip(layers[:-1], layers[1:])):
            for i_out_node in range(n_out_nodes):
                S_max = Sum([
                        # If(in_layer!=0 and Not(spike_indicators[(i_in_node,in_layer,timestep)]), # type: ignore
                        # 0,
                        # max(0,weights[(i_in_node, i_out_node, in_layer)]))
                        max(0,weights[(i_in_node, i_out_node, in_layer)])
                        for i_in_node in range(n_in_nodes)
                    ])
                if timestep == 1:
                    stability = S_max/(1-beta)<=threshold #a term to check a neuron is inactive
                    node_eqn.append(
                        Implies(stability,
                                # And([inactivities[(i_out_node,in_layer+1,tp)] for tp in range(1, num_steps+1)])
                                And([Not(spike_indicators[(i_out_node,in_layer+1,tp)]) for tp in range(1, num_steps+1)])
                        )
                    )
                #     continue
                # n_real_max:ArithRef = Real(f"n_max_"+"_".join(map(str,(i_out_node,in_layer+1,timestep))))
                # n_max:ArithRef = ToInt(n_real_max)
                # n_max = If(n_real_max == n_max, n_max-1, n_max) # type: ignore
                # node_eqn.append(beta**n_real_max == 1 - threshold*(1-beta)/S_max)
                # temporary_inactivities = And(
                #     [Implies(tp<timestep+n_max,
                #             #  inactivities[(i_out_node, in_layer+1, tp)]) for tp in range(timestep+1, num_steps+1)]
                #              spike_indicators[(i_out_node, in_layer+1, tp)]) for tp in range(timestep+1, num_steps+1)]
                #     )
                # node_eqn.append(
                #     Implies(
                #         spike_indicators[(i_out_node, in_layer+1, timestep)],
                #         temporary_inactivities
                #         )
                #     )
    return node_eqn

def gen_input_layer_current_bound(weights:WeightType, spike_indicators:SpkType, spike_orig:Tensor, delta:int):
    node_eqn:List[BoolRef] = []
    total_dns = 0
    dns = []
    for i_out_node in range(layers[1]):
        weight_vector_i = [weights[(k,i_out_node,0)] for k in range(layers[0])]
        assert len(weight_vector_i) == spike_orig.shape[1], "Weight and spike dimension mismatch."
        idx_descending_abs = sorted(range(len(weight_vector_i)), key=lambda i:abs(weight_vector_i[i]),reverse=True)
        pos_factor, neg_factor = [], []
        for t in range(num_steps):
            S_min = S_max = sum(weight_vector_i[k]*spike_orig[t,k] for k in range(layers[0]))
            for i in idx_descending_abs:
                score = (2*float(spike_orig[t,i])-1) * weight_vector_i[i]
                if score > 0: pos_factor.append(abs(weight_vector_i[i]))
                elif score < 0: neg_factor.append(abs(weight_vector_i[i]))
            
            for d in range(min(delta, len(pos_factor))):
                S_max += pos_factor[d]
            for d in range(min(delta, len(neg_factor))):
                S_min -= neg_factor[d]
                
            if S_max <= 0 or S_max<=threshold*(1-beta):
                total_dns += 1
                dns.append((i_out_node, 1)) # save dead neuron: (node_idx, layer_idx)
                node_eqn.append(
                    Not(Or([spike_indicators[(i_out_node, 1, t)]
                            for t in range(1, num_steps+1)]))) # type: ignore
    print(f"Total Dead Neurons: {total_dns}")
    return node_eqn, dns

def get_bound(w_vector:List[float], x_vector_orig:List[float], delta:int, is_first_layer:bool) -> Tuple[float, float]:
    # def get_bound_hidden(w_vector:List[float], x_vector_orig:List[float], delta:Union[int, float]) -> Tuple[float, float]:
    def get_bound_hidden(w_vector:List[float]) -> Tuple[float, float]:
        # assert len(w_vector) == len(x_vector_orig), f"Weight and spike dimension mismatch. w_vector:{len(w_vector)}, x_vector_orig:{len(x_vector_orig)}"
        # idx_descending_abs = sorted(range(len(w_vector)), key=lambda i:abs(w_vector[i]),reverse=True)
        # pos_factor, neg_factor = [], []
        # S_min = S_max = sum(w_vector[k]*x_vector_orig[k] for k in range(layers[0]))
        # for i in idx_descending_abs:
        #     score = (2*float(x_vector_orig[i])-1) * w_vector[i]
        #     if score > 0: pos_factor.append(abs(w_vector[i]))
        #     elif score < 0: neg_factor.append(abs(w_vector[i]))
        
        # for d in range(int(min(delta, len(pos_factor)))):
        #     S_max += pos_factor[d]
        # for d in range(int(min(delta, len(neg_factor)))):
        #     S_min -= neg_factor[d]
            
        return sum(filter(lambda x:x>0, w_vector)), sum(filter(lambda x:x<0, w_vector))
    def get_bound_input(w_vector:List[float], x_vector_orig:List[float], delta:int) -> Tuple[float, float]:
        assert len(w_vector) == len(x_vector_orig), f"Weight and spike dimension mismatch. w_vector:{len(w_vector)}, x_vector_orig:{len(x_vector_orig)}"
        idx_descending_abs = sorted(range(len(w_vector)), key=lambda i:abs(w_vector[i]),reverse=True)
        pos_factor, neg_factor = [], []
        S_min = S_max = sum(w_vector[k]*x_vector_orig[k] for k in range(layers[0]))
        for i in idx_descending_abs:
            score = (2*float(x_vector_orig[i])-1) * w_vector[i]
            if score > 0: pos_factor.append(abs(w_vector[i]))
            elif score < 0: neg_factor.append(abs(w_vector[i]))
        
        for d in range(min(delta, len(pos_factor))):
            S_max += pos_factor[d]
        for d in range(min(delta, len(neg_factor))):
            S_min -= neg_factor[d]
        
        return S_min, S_max
    if is_first_layer:
        return get_bound_input(w_vector, x_vector_orig, delta)
    return get_bound_hidden(w_vector)
    
def gen_node_eqns(weights:WeightType, spike_indicators:SpkType, potentials:PotType):
    node_eqn:List[Union[BoolRef,Literal[False]]] = []
    for t in range(1, num_steps+1):
        for j, m in enumerate(neurons_in_layers[1:], start=1):
            for i in range(m):
                if not (i,j,t) in potentials: continue
                curr_vec = [spike_indicators[(k, j-1, t)]*weights[(k, i, j-1)] for k in range(neurons_in_layers[j-1])]
                curr = sum(curr_vec) + beta*potentials[(i, j, t-1)] # type: ignore # epsilon_1
                node_eqn.append(
                    (curr >= threshold) == spike_indicators[(i, j, t)]
                )
                node_eqn.append(
                    potentials[(i, j, t)] == If(spike_indicators[(i, j, t)], curr-1, curr)
                    # And(Implies(spike_indicators[(i, j, t)],
                    #             potentials[(i, j, t)] == curr - 1), # epsilon_2 & epsilon_4
                    #     Implies(Not(spike_indicators[(i, j, t)]),
                    #             potentials[(i, j, t)] == curr))) # type: ignore # epsilon_3 & epsilon_5
                )# node_eqn = []

    return node_eqn

def gen_node_eqns_bounded(weights:WeightType, spike_indicators:SpkType, potentials:PotType, spk_orig:Tensor, delta:int):
    node_eqn:List[BoolRef] = []
    dead_neurons = []
    inactive_neurons = []
    for t in range(1, num_steps+1):
        for j, m in enumerate(layers[1:], start=1):
            for i in range(m):
                w_vector = [weights[(k, i, j-1)] for k in range(layers[j-1])]
                S_min:float; S_max:float
                S_min, S_max = get_bound(w_vector, spk_orig[t-1].float().tolist(), delta, j==1)
                # if j == 1:
                #     S_min, S_max = get_bound(w_vector, spk_orig[t-1].float().tolist(), delta)
                # else:
                #     S_min, S_max = sum(filter(lambda x:x>0, w_vector)), sum(filter(lambda x:x<0, w_vector))
                
                dP2i_lst = [spike_indicators[(k, j-1, t)]*weights[(k, i, j-1)] for k in range(layers[j-1])]
                S = sum(dP2i_lst) + beta*potentials[(i, j, t-1)] # type: ignore # epsilon_1
                node_eqn.append(And(S_min <= S, S_max >= S)) # type: ignore
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
                spike_indicators:SpkType,
                encodings:List[Union[BoolRef,Literal[False]]]) -> \
                    Tuple[Literal["sat", "unsat", "unknown"], ArithRef, Solver]:
    #solver preprocess
    solver = Solver()
    solver.add(encodings)
    
    #make spike input encoding
    spk_outs:List[ArithRef] = [0] * layers[-1] # type: ignore
    for timestep, input_spikes in enumerate(sample_spike):
        for i_input_node, spike in enumerate(input_spikes.view(num_input)):
            solver.add(spike_indicators[(i_input_node, 0, timestep+1)]
                       == bool(spike.item()))
        for i_input_node in range(layers[-1]):
            spk_outs[i_input_node] += If(spike_indicators[(i_input_node, len(layers)-1, timestep+1)], 1, 0)
    
    #add argmax encoding
    max_label_spk = Int('Max_Label_Spike')
    label = Int('Label')
    solver.add(argmax_left(spk_outs, label, max_label_spk))
    return str(solver.check()), label, solver # type: ignore

def gen_delta_reuse(cfg:CFG,
                    sample_spike:Tensor,
                    spike_indicators:SpkType,
                    potentials:PotType,
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