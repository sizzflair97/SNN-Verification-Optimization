from multiprocessing import Pool
from utils import *
from adv_rob_iris_module import run_test
from numpy import arange
cfgs = [
    CFG("Control", False, 0),
    CFG("DNP", True, 0),
    CFG("DNP_M2", True, 2),
    CFG("M2", False, 2)
]

if __name__ == "__main__":
    if mp:
        pool = Pool(processes=min(len(cfgs), 8))
        pool.map(run_test, cfgs)
        pool.close()
        pool.join()
    else:
        for cfg in cfgs:
            run_test(cfg)