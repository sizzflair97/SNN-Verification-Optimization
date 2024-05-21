from typing import DefaultDict, Dict, List, Literal, NewType, Tuple, Union
from z3 import ArithRef, BoolRef, FPRef, FPNumRef
import torch

#Model default params
batch_size = 128
location = '.'
layers = [28*28, 100, 10] # [28*28, 100, 10] default
beta = 0.95
dtype = torch.float
num_steps = 10
data_path = 'data/mnist'
delta = [1]
num_epochs = 3
train = True
threshold = 1.0

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
mp = True #
num_procs:int = 2 # None -> auto-detect # of cores.

#Code Function typing
NodeIdx = int; LayerIdx = int; TimeIdx = int
InNodeIdx = int; OutNodeIdx = int; InLayerIdx = int

NeuronTuple = Tuple[NodeIdx, LayerIdx, TimeIdx] # Represents neuron spatiotemporal position, (node, layer, timestep).
SynapseTuple = Tuple[InNodeIdx, OutNodeIdx, InLayerIdx]

SpkType = NewType("SpkType", Dict[NeuronTuple, Union[BoolRef,bool]])
InactivityType = NewType("InactivityType", Dict[NeuronTuple, BoolRef])
PotType = NewType("PotType", Dict[NeuronTuple, ArithRef])
CurrType = NewType('CurrType', Dict[NeuronTuple, ArithRef])
WeightType = NewType("WeightType", DefaultDict[SynapseTuple, float])