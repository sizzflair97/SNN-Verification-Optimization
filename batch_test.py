from utils import *
from adv_rob_iris_module import run_test
from numpy import arange
from argparse import ArgumentParser

deltas = (1,2,3)
cfgs = [
    # CFG("Control", np_level=0, deltas=deltas),
    CFG("Manual_Control", np_level=0, deltas=deltas),
    CFG("Manual_DNP", np_level=1, deltas=deltas),
    # CFG("GNP", np_level=2, deltas=deltas),
]

def parse():
    parser = ArgumentParser()
    parser.add_argument("-p", "--prefix", dest="prefix", type=str)
    parser.add_argument("--np_level", dest="np_level", type=int)
    parser.add_argument("--reuse_level", dest="reuse_level", type=int)
    parser.add_argument("--seed", dest="seed", type=int, default=42)
    parser.add_argument("--delta_max", dest="delta_max", type=int, default=3)
    return parser.parse_args()

if __name__ == "__main__":
    try:
        parser = parse()
        log_name:str = ""
        assert all(getattr(parser, s) is not None for s in "np_level reuse_level prefix".split())
        if parser.np_level == 1: log_name += "DNP"
        elif parser.np_level == 2: log_name += "GNP"
        if parser.reuse_level: log_name += ("_" if log_name else "") + f"M{parser.reuse_level}"
        if log_name == "": log_name = "Control"
        if parser.prefix: log_name = parser.prefix + "_" + log_name 
        run_test(CFG(log_name=log_name,
                     np_level=parser.np_level,
                     reuse_level=parser.reuse_level,
                     seed=parser.seed,
                     deltas=tuple(range(0, parser.delta_max+1))))
    except AssertionError as e:
        for cfg in cfgs:
            run_test(cfg)
