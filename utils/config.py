from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .load import load_mnist
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
    subtype:Literal["mnist", "fmnist"] = "mnist"
    load_data_func:Callable[[], tuple[TImageBatch,TLabelBatch,TImageBatch,TLabelBatch]] = load_mnist
    n_layer_neurons = (image_len*image_len, 10, 10) # [28*28, 100, 10] default
    layer_shapes = ((image_len,image_len), (10, 1), (10, 1))