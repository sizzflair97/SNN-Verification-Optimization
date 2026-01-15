from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Self

from .dictionary_mnist import TImageBatch, TLabelBatch, image_len

# class SingletonMeta(type):
#     _instances = {}

#     def __call__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             instance = super().__call__(*args, **kwargs)
#             cls._instances[cls] = instance
#         return cls._instances[cls]

@dataclass
class CFG:
    log_name:str
    subtype:Literal["mnist", "fmnist"]
    load_data_func:Callable[[Self,], tuple[TImageBatch,TLabelBatch,TImageBatch,TLabelBatch]]
    manual_indices:list[int]|None = None
    seed:int = 42
    
    #Delta-perturbation params
    num_samples:int = 14
    deltas:tuple[int,...] = (1,)
    z3:bool = False
    milp:bool = False
    prefix_set_match:bool = False
    adv_attack:bool = False
    n_layer_neurons:tuple[int, ...] = (image_len*image_len, 10, 10) # [28*28, 100, 10] default
    layer_shapes:tuple[tuple[int, int], ...] = ((image_len,image_len), (10, 1), (10, 1))
    num_steps:int = 5