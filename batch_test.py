from utils import *
from adv_rob_iris_module import run_test as run_test_iris
from adv_rob_mnist_module import run_test as run_test_mnist
from numpy import arange
from argparse import ArgumentParser, Namespace

deltas = (1,)
cfgs = [
    CFG("Manual_Control", deltas=deltas),
]

TestType = Literal["iris", "mnist"]
def run_test(cfg:CFG, test_type:TestType="mnist"):
    match test_type:
        case "iris": return run_test_iris(cfg)
        case "mnist": return run_test_mnist(cfg)
        case _: raise NotImplementedError

def parse():
    parser = ArgumentParser()
    parser.add_argument("-p", "--prefix", dest="prefix", type=str)
    parser.add_argument("--seed", dest="seed", type=int, default=42)
    parser.add_argument("--delta_max", dest="delta_max", type=int, default=1)
    return parser.parse_args()

def prepare_log_name(parser:Namespace) -> str:
    words = []
    
    if parser.prefix: words.append(parser.prefix)
    
    if not words: words.append("Control")
    
    return '_'.join(words)

if __name__ == "__main__":
    parser = parse()
    if all(hasattr(parser, s) is not None for s in "prefix".split()):
        run_test(CFG(log_name=prepare_log_name(parser),
                     seed=parser.seed,
                    #  deltas=tuple(range(1, parser.delta_max+1))))
                     deltas=(parser.delta_max,)))
    else:
        for cfg in cfgs:
            run_test(cfg)
