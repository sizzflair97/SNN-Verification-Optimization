from dataclasses import dataclass

@dataclass
class CFG:
    log_name:str = ""
    seed:int = 42
    #Delta-perturbation params
    num_samples:int = 14
    deltas:tuple[int,...] = (1,)
    z3:bool = False