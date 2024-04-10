from dataclasses import dataclass, field
from typing import List, Literal, Tuple

@dataclass
class CFG:
    log_name:str = "M2"
    use_DNP:bool = True
    reuse_level:Literal[0,1,2] = 1
    seed:int = 42
    # eps:float = 1e-1
    #Delta-perturbation params
    num_samples:int = 15
    deltas:Tuple[int,...] = (1,2,3)