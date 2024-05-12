from typing import DefaultDict, Dict, List, Literal, NewType, Tuple, Union
from z3 import ArithRef, BoolRef, FPRef, FPNumRef
import torch

#Model default params
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
mp = True #
num_procs:Union[int,None] = 3 # None -> auto-detect # of cores.
shuffle = True # 
train = True # 
num_epochs = 25 # 1 default
file_name = 'model_iris.pth'
layers = (4, 16, 3) # 4, 5, 3 default
num_input, num_hidden, num_output = layers
beta = 0.9
threshold = 1.0
num_steps = 25 # 25 default

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