from dataclasses import dataclass
from typing import Literal

TLogName = Literal["Control", "DNP", "DNP_M1", "DNP_M2", "M1", "M2"]

@dataclass
class cfg:
    log_name:TLogName = "DNP_M1"
    use_DNP:bool = True
    reuse_level:Literal[0,1,2] = 1

    #Delta-perturbation params
    num_samples = 15
    deltas = [1,2,3]