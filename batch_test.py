from utils import *
from adv_rob_iris_module import run_test as run_test_iris
from adv_rob_mnist_module import run_test as run_test_mnist
from numpy import arange
from argparse import ArgumentParser, Namespace, BooleanOptionalAction

deltas = (1,)

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
    parser.add_argument("--test-type", dest="test_type", type=str)
    parser.add_argument("--numpy-backend", dest="numpy_backend", action=BooleanOptionalAction, default=False)
    return parser.parse_args()

def prepare_log_name(parser:Namespace) -> str:
    words = []
    
    if hasattr(parser, "test_type"): words.append(parser.test_type)
    
    if hasattr(parser, "prefix"): words.append(parser.prefix)
    
    if not words: raise NotImplementedError
    
    return '_'.join(words)

if __name__ == "__main__":
    parser = parse()
    if all(hasattr(parser, s) for s in "prefix test_type numpy_backend".split()):
        run_test(CFG(log_name=prepare_log_name(parser),
                     seed=parser.seed,
                    #  deltas=tuple(range(1, parser.delta_max+1))))
                     deltas=(parser.delta_max,),
                     numpy_backend=parser.numpy_backend),
                 test_type=parser.test_type)
    else:
        print("Not appropriate arguments.")
