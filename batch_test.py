from utils import *
from adv_rob_iris_module import run_test as run_test_iris
from adv_rob_mnist_module import run_test as run_test_mnist
from numpy import arange
from argparse import ArgumentParser, Namespace

deltas = (1,2,3)
cfgs = [
    CFG("Manual_Control", np_level=0, deltas=deltas),
    CFG("Manual_DNP", np_level=1, deltas=deltas),
]

TestType = Literal["iris", "mnist"]
def run_test(cfg:CFG, test_type:TestType="mnist"):
    if test_type == "iris": return run_test_iris(cfg)
    elif test_type == "mnist": return run_test_mnist(cfg)

def parse():
    parser = ArgumentParser()
    parser.add_argument("-p", "--prefix", dest="prefix", type=str)
    parser.add_argument("--np_level", dest="np_level", type=int)
    parser.add_argument("--reuse_level", dest="reuse_level", type=int)
    parser.add_argument("--seed", dest="seed", type=int, default=42)
    parser.add_argument("--delta_max", dest="delta_max", type=int, default=3)
    return parser.parse_args()

def prepare_log_name(parser:Namespace) -> str:
    words = []
    
    if parser.prefix: words.append(parser.prefix)
    
    # if parser.np_level == 1: words.append("DNP")
    # elif parser.np_level == 2: words.append("GNP")
    
    if parser.reuse_level: words.append(f"M{parser.reuse_level}")
    
    if not words: words.append("Control")
    
    return '_'.join(words)

if __name__ == "__main__":
    parser = parse()
    if all(getattr(parser, s) is not None for s in "np_level reuse_level prefix".split()):
        run_test(CFG(log_name=prepare_log_name(parser),
                     np_level=parser.np_level,
                     reuse_level=parser.reuse_level,
                     seed=parser.seed,
                     deltas=tuple(range(1, parser.delta_max+1))))
    else:
        for cfg in cfgs:
            run_test(cfg)
