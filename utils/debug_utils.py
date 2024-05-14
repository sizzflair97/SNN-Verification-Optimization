from typing import Any, List
import numpy as np
from z3 import ModelRef, RatNumRef
from .dictionary_iris import *
import json, logging

def info(msg:Any):
    print(msg) or logging.getLogger().info(msg) # type: ignore

def dump(_model:ModelRef,
         spike_indicators:Dict[NeuronTuple, BoolRef],
         potentials:Dict[NeuronTuple, ArithRef],
         filename:str="dump.json"):
    _dump_dict = {}
    
    
    _spike_indicator_lst:List[np.ndarray] = []
    for _n_nodes in layers:
        _spike_indicator_lst.append(
            np.zeros(
                (_n_nodes, num_steps),
                dtype=bool
            )
        )
    for k, v in spike_indicators.items():
        _idx_node, _idx_layer, _idx_timestep = k
        _spike_indicator_lst[_idx_layer][_idx_node, _idx_timestep-1] = bool(_model[v])
    for i, _ in enumerate(_spike_indicator_lst):
        _spike_indicator_lst[i] = _spike_indicator_lst[i].tolist()
    _dump_dict["spike_indicators"] = _spike_indicator_lst
    
    
    _potentials_lst:List[np.ndarray] = []
    for _n_nodes in layers:
        _potentials_lst.append(
            np.zeros(
                (_n_nodes, num_steps+1),
                dtype=np.float64
            )
        )
    for k, v in potentials.items():
        _idx_node, _idx_layer, _idx_timestep = k
        _p:RatNumRef = _model[v] # type: ignore
        _potentials_lst[_idx_layer][_idx_node, _idx_timestep] = eval(_p.as_string())
    for i, _ in enumerate(_potentials_lst):
        _potentials_lst[i] = _potentials_lst[i].tolist()
    _dump_dict["potentials"] = _potentials_lst
    
    
    with open(filename, "w") as fp:
        json.dump(_dump_dict, fp)