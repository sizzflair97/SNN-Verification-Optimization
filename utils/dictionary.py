from typing import DefaultDict, Dict, List, Tuple, Union
from z3 import ArithRef, BoolRef, FPRef, FPNumRef
import torch

#Model default params
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
mp = False
shuffle = True
train = False
num_epochs = 1
file_name = 'model_iris.pth'
num_input = 4
num_hidden = 5
num_output = 3
layers = [num_input, num_hidden, num_output]
beta = 0.95
threshold = 1.0
num_steps = 25

#Code Function typing
NodeIdx = int; LayerIdx = int; TimeIdx = int
InNodeIdx = int; OutNodeIdx = int; InLayerIdx = int

NeuronTuple = Tuple[NodeIdx, LayerIdx, TimeIdx]
SynapseTuple = Tuple[InNodeIdx, OutNodeIdx, InLayerIdx]

SType = Dict[NeuronTuple, BoolRef]
PType = Dict[NeuronTuple, ArithRef]
WType = DefaultDict[SynapseTuple, float]