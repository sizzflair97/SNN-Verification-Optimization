from utils import *
from adv_rob_iris_module import run_test
from numpy import arange
<<<<<<< HEAD
from argparse import ArgumentParser

deltas = (1,2,3)
cfgs = [
    # CFG("Control", np_level=0, deltas=deltas),
    # CFG("DNP", np_level=1, deltas=deltas),
    CFG("GNP", np_level=2, deltas=deltas),
=======
from argparse import ArgumentError, ArgumentParser
from functools import reduce
cfgs = [
    CFG("Control", False, 0, deltas=(1,)),
    CFG("DNP", True, 0, deltas=(1,)),
>>>>>>> mnist
]

def parse():
    parser = ArgumentParser()
    parser.add_argument("-p", dest="prefix", type=str)
    parser.add_argument("--np_level", dest="np_level", type=int)
    parser.add_argument("--reuse_level", dest="reuse_level", type=int)
    parser.add_argument("--seed", dest="seed", type=int, default=42)
    parser.add_argument("--delta_max", dest="delta_max", type=int, default=3)
    return parser.parse_args()

if __name__ == "__main__":
    try:
        parser = parse()
        log_name:str = ""
<<<<<<< HEAD
        assert all(hasattr(parser, attr) for attr in "np_level reuse_level seed prefix".split())
        if parser.np_level: log_name += "DNP"
        if parser.reuse_level: log_name += ("_" if log_name else "") + f"M{parser.reuse_level}"
        if log_name == "": log_name = "Control"
        if parser.prefix: log_name = parser.prefix + "_" + log_name 
        run_test(CFG(log_name,
                     parser.np_level,
                     parser.reuse_level,
                     parser.seed))
=======
        assert reduce(lambda x, y: x and y,
                      [hasattr(parser, x)
                           for x in "dnp prefix reuse_level".split()])
        if parser.dnp: log_name += "DNP"
        if parser.reuse_level: log_name += ("_" if log_name else "") + f"M{parser.reuse_level}"
        if log_name == "": log_name = "Control"
        log_name = parser.prefix + "_" + log_name + f"_{parser.seed}"
        run_test(CFG(log_name=log_name,
                     use_DNP=parser.dnp,
                     reuse_level=parser.reuse_level,
                     seed=parser.seed,
                     deltas=tuple(range(1,parser.delta_max+1))))
>>>>>>> mnist
    except AssertionError as e:
        for cfg in cfgs:
            run_test(cfg)
