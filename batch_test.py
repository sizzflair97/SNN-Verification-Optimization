from multiprocessing import Pool
from utils import *
from adv_rob_iris_module import run_test
from numpy import arange
from argparse import ArgumentError, ArgumentParser
cfgs = [
    CFG("Control", False, 0),
    CFG("DNP", True, 0),
    CFG("DNP_M2", True, 2),
    CFG("M2", False, 2)
]

def parse():
    parser = ArgumentParser()
    parser.add_argument("-p", dest="prefix", type=str)
    parser.add_argument("-d", dest="dnp", action="store_true")
    parser.add_argument("--reuse_level", dest="reuse_level", type=int)
    return parser.parse_args()

if __name__ == "__main__":
    try:
        parser = parse()
        log_name:str = ""
        if parser.dnp: log_name += "DNP"
        if parser.reuse_level: log_name += ("_" if log_name else "") + f"M{parser.reuse_level}"
        if log_name == "": log_name = "Control"
        log_name = parser.prefix + "_" + log_name
        run_test(CFG(log_name,
                     parser.dnp,
                     parser.reuse_level))
    except ArgumentError as e:
        if mp:
            pool = Pool(processes=min(len(cfgs), 8))
            pool.map(run_test, cfgs)
            pool.close()
            pool.join()
        else:
            for cfg in cfgs:
                run_test(cfg)