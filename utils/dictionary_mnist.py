from typing import Annotated, DefaultDict, Dict, List, Literal, Tuple, Union
import numpy as np
from z3 import ArithRef, BoolRef, FPRef, FPNumRef
import torch

#Model default params
batch_size = 128
location = '.'
image_len = 28
n_layer_neurons = (image_len*image_len, 10, 10) # [28*28, 100, 10] default
layer_shapes = ((image_len,image_len), (n_layer_neurons[-2], 1), (n_layer_neurons[-1], 1))
beta = 1
dtype = torch.float
num_steps = 5
data_path = 'data/mnist'
delta = [1]
num_epochs = 300
train = False
test = True
load_expr = False
save_expr = False
threshold = 100

# Training params
lr = [.2, .2]  # The learning rate of hidden and ouput neurons
lamda = [0.000001, 0.000001]  # The regularization penalty for hidden and ouput neurons

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
mp = False #
num_procs:int = 14

#Code Function typing
NodeIdx = Tuple[int, int]; LayerIdx = int; TimeIdx = int # To define neuron states
InNodeIdx = Tuple[int, int]; OutNodeIdx = int; InLayerIdx = int # To define weights. The output neuron position in dimension 1 is always 1, so we ignore it.

Node_Layer_Time = Annotated[Tuple[NodeIdx, LayerIdx, TimeIdx], "Represents neuron spatiotemporal position, (node, layer, timestep)."]
In_Out_InLayer = Annotated[Tuple[InNodeIdx, OutNodeIdx, InLayerIdx], "Represents weight, (innode, outnode, layer)."]

TSpike = Dict[Node_Layer_Time, BoolRef|bool]
TPotential = Dict[Node_Layer_Time, ArithRef]
TCurrent = Dict[Node_Layer_Time, ArithRef]
TWeight = Dict[In_Out_InLayer, float]
TSpikeTime = Dict[Tuple[NodeIdx, LayerIdx], ArithRef]

TW1Shape = Tuple[Literal[400], Literal[28], Literal[28]]
TW2Shape = Tuple[Literal[10], Literal[400], Literal[1]]
TWeightShape = Union[TW1Shape, TW2Shape]
TWeightList = List[np.ndarray[TWeightShape, np.dtype[np.float_]]] # single quote trick to avoid numpy._DTypeMeta error.

TImage = np.ndarray[Tuple[Literal[28],Literal[28]], np.dtype[np.int_]]
TImageBatch = np.ndarray[Tuple[Literal[60000],Literal[28],Literal[28]], np.dtype[np.int_]] # single quote trick to avoid numpy._DTypeMeta error.
TLabelBatch = np.ndarray[Tuple[Literal[60000]], np.dtype[np.int_]]