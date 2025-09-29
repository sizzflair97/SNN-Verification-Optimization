from typing import Literal, LiteralString
from utils.config import CFG
from utils.load import load_mnist, load_fmnist
from adv_rob_mnist_module import run_test as run_test_mnist
from argparse import ArgumentParser, Namespace
from time import strftime, localtime

required_arguments:list[LiteralString] = "test_type prefix".split()

TestType = Literal["mnist", "fmnist"]

def parse():
    parser = ArgumentParser()
    parser.add_argument("-p", "--prefix", dest="prefix", type=str)
    parser.add_argument("--seed", dest="seed", type=int, default=42)
    parser.add_argument("--delta-max", dest="delta_max", type=int, default=1)
    parser.add_argument("--repeat", dest="repeat", type=int, default=1)
    parser.add_argument("--num-samples", dest="num_samples", type=int, default=14)
    parser.add_argument("--n-hidden-neurons", dest="n_hidden_neurons", type=int, default=10)
    parser.add_argument("--num-steps", dest="num_steps", type=int, default=5)
    parser.add_argument("--test-type", dest="test_type", type=str)
    parser.add_argument("--z3", dest="z3", action="store_true", default=False)
    parser.add_argument("--milp", dest="milp", action="store_true", default=False)
    parser.add_argument("--np", dest="np", action="store_true", default=False)
    parser.add_argument("--adv", dest="adv", action="store_true", default=False)

    return parser.parse_args()

def prepare_log_name(parser:Namespace) -> str:
    words = [strftime('%m%d%H%M', localtime())]
    if getattr(parser, "repeat") > 1: words.append(f"rep_{parser.repeat}")
    if hasattr(parser, "test_type"): words.append(parser.test_type)
    if hasattr(parser, "prefix"): words.append(parser.prefix)
    
    prefix: str
    assert not (parser.z3 == parser.milp == True)
    if parser.z3: prefix = "z3"
    elif parser.milp: prefix = "milp"
    elif parser.np:
        prefix = "np"
        if parser.adv: prefix += "-adv"
    else: raise ValueError("Invalid solver type.")
    words.append(prefix)
    words.append(str(parser.num_steps))
    return '_'.join(words)

if __name__ == "__main__":
    parser = parse()
    
    if getattr(parser, "repeat") < 1:
        raise ValueError("repeat must be greater than 0.")
    
    match parser.test_type:
        case "mnist":
            load_data_func = load_mnist
        case "fmnist":
            load_data_func = load_fmnist
        case _: raise NotImplementedError(f"Test type must be in {TestType}.")
    
    if all(hasattr(parser, s) for s in required_arguments):
        for iteration in range(parser.repeat):
            run_test_mnist(CFG(log_name=prepare_log_name(parser),
                        subtype=parser.test_type,
                        load_data_func=load_data_func,
                        seed=parser.seed,
                        num_samples=parser.num_samples,
                        deltas=(parser.delta_max,),
                        z3=parser.z3,
                        milp=parser.milp,
                        adv_attack=parser.adv,
                        n_layer_neurons=(28*28, parser.n_hidden_neurons, 10),
                        layer_shapes=((28,28), (parser.n_hidden_neurons,1), (10,1)),
                        num_steps=parser.num_steps))
    else:
        raise ValueError("Not appropriate arguments.")
