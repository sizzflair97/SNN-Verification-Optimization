from utils import *
from adv_rob_iris_module import run_test
from numpy import arange
from argparse import ArgumentParser

deltas = (1,2,3)
cfgs = [
    # CFG("Control", np_level=0, deltas=deltas),
    # CFG("DNP", np_level=1, deltas=deltas),
    CFG("GNP", np_level=2, deltas=deltas),
]

def parse():
    parser = ArgumentParser()
    parser.add_argument("-p", dest="prefix", type=str)
    parser.add_argument("--np_level", dest="np", type=int)
    parser.add_argument("--reuse_level", dest="reuse_level", type=int)
    parser.add_argument("--seed", dest="seed", type=int, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    try:
        parser = parse()
        log_name:str = ""
        assert parser.np and parser.reuse_level and parser.prefix
        if parser.np: log_name += "DNP"
        if parser.reuse_level: log_name += ("_" if log_name else "") + f"M{parser.reuse_level}"
        if log_name == "": log_name = "Control"
        log_name = parser.prefix + "_" + log_name
        run_test(CFG(log_name,
                     parser.np,
                     parser.reuse_level,
                     parser.seed))
    except AssertionError as e:
        for cfg in cfgs:
            run_test(cfg)