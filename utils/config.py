from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .dictionary_mnist import TImageBatch, TLabelBatch, image_len

@dataclass
class CFG:
    log_name:str = ""
    seed:int = 42
    #Delta-perturbation params
    num_samples:int = 14
    deltas:tuple[int,...] = (1,)
    z3:bool = False
    milp:bool = False
    adv_attack:bool = False
    subtype:Literal["mnist", "fmnist"] = "mnist"
    load_data_func:Callable[["CFG",], tuple[TImageBatch,TLabelBatch,TImageBatch,TLabelBatch]] = None
    n_layer_neurons:tuple[int, ...] = (image_len*image_len, 10, 10) # [28*28, 100, 10] default
    layer_shapes:tuple[tuple[int, int], ...] = ((image_len,image_len), (10, 1), (10, 1))
    num_steps:int = 5