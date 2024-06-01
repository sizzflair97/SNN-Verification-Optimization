from dataclasses import dataclass, field
from typing import List, Literal, Tuple

@dataclass
class CFG:
    log_name:str = ""
    seed:int = 42
    #Delta-perturbation params
    num_samples:int = 15
    deltas:Tuple[int,...] = (1,2,3)