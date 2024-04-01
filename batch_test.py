from multiprocessing import Pool
from utils import *
from adv_rob_iris_module import run_test
cfgs = [
    cfg("Control", False, 0),
    cfg("DNP", True, 0),
    cfg("DNP_M1", True, 1),
    cfg("DNP_M2", True, 2),
    cfg("M1", False, 1),
    cfg("M2", False, 2)
]

if __name__ == "__main__":
    pool = Pool(processes=len(cfgs))
    pool.map(run_test, cfgs)
    pool.close()
    pool.join()