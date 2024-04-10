from multiprocessing import Pool
from utils import *
from adv_rob_iris_module import run_test
from numpy import arange
from argparse import ArgumentError, ArgumentParser
from functools import reduce
cfgs = [
    CFG("Control", False, 0, deltas=(1,)),
    CFG("DNP", True, 0, deltas=(1,)),
]

def parse():
    parser = ArgumentParser()
    parser.add_argument("-p", dest="prefix", type=str)
    parser.add_argument("-d", dest="dnp", action="store_true")
    parser.add_argument("--reuse_level", dest="reuse_level", type=int)
    parser.add_argument("--seed", dest="seed", type=int, default=42)
    parser.add_argument("--delta_max", dest="delta_max", type=int, default=3)
    return parser.parse_args()

if __name__ == "__main__":
    try:
        parser = parse()
        log_name:str = ""
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
    except AssertionError as e:
        if mp:
            pool = Pool(processes=min(len(cfgs), 8))
            pool.map(run_test, cfgs)
            pool.close()
            pool.join()
        else:
            for cfg in cfgs:
                run_test(cfg)