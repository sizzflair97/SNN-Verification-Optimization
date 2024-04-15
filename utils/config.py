from dataclasses import dataclass, field
from typing import List, Literal, Tuple

@dataclass
class CFG:
    log_name:str = "M2"
    np_level:int = 0
    reuse_level:Literal[0,1,2] = 1
    seed:int = 42
    #Delta-perturbation params
    num_samples:int = 15
    deltas:Tuple[int,...] = (1,2,3)