from utils import *
from adv_rob_iris_module import run_test as run_test_iris
from adv_rob_mnist_module import run_test as run_test_mnist
from numpy import arange
from argparse import ArgumentParser, Namespace, BooleanOptionalAction

required_arguments_lst = "test_type prefix".split()

TestType = Literal["iris", "mnist"]
def run_test(cfg:CFG, test_type:TestType="mnist"):
    match test_type:
        case "iris": return run_test_iris(cfg)
        case "mnist": return run_test_mnist(cfg)
        case _: raise NotImplementedError("Test type must be either 'iris' or 'mnist'.")

def parse():
    parser = ArgumentParser()
    parser.add_argument("-p", "--prefix", dest="prefix", type=str)
    parser.add_argument("--seed", dest="seed", type=int, default=42)
    parser.add_argument("--delta_max", dest="delta_max", type=int, default=1)
    parser.add_argument("--test-type", dest="test_type", type=str)
    parser.add_argument("--z3", dest="z3", action=BooleanOptionalAction, default=True)
    return parser.parse_args()

def prepare_log_name(parser:Namespace) -> str:
    words = []
    if hasattr(parser, "test_type"): words.append(parser.test_type)
    if hasattr(parser, "prefix"): words.append(parser.prefix)
    words.append("z3" if parser.z3 else "np")
    return '_'.join(words)

if __name__ == "__main__":
    parser = parse()
    if all(hasattr(parser, s) for s in required_arguments_lst):
        run_test(CFG(log_name=prepare_log_name(parser),
                     seed=parser.seed,
                    #  deltas=tuple(range(1, parser.delta_max+1))))
                     deltas=(parser.delta_max,),
                     z3=parser.z3),
                 test_type=parser.test_type)
    else:
        ValueError("Not appropriate arguments.")
