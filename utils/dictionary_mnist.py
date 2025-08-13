from typing import Annotated, Literal
import numpy as np
from z3 import ArithRef, BoolRef
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
mp = True #
num_procs:int = 14

#Code Function typing
NodeIdx = tuple[int, int]; LayerIdx = int; TimeIdx = int # To define neuron states
InNodeIdx = tuple[int, int]; OutNodeIdx = int; InLayerIdx = int # To define weights. The output neuron position in dimension 1 is always 1, so we ignore it.

Neuron_Layer_Time = Annotated[tuple[NodeIdx, LayerIdx, TimeIdx], "Represents neuron spatiotemporal position, (node, layer, timestep)."]
In_Out_InLayer = Annotated[tuple[InNodeIdx, OutNodeIdx, InLayerIdx], "Represents weight, (innode, outnode, layer)."]

TSpike = dict[Neuron_Layer_Time, BoolRef|bool]
TPotential = dict[Neuron_Layer_Time, ArithRef]
TCurrent = dict[Neuron_Layer_Time, ArithRef]
TWeight = dict[In_Out_InLayer, float]
TSpikeTime = dict[tuple[NodeIdx, LayerIdx], ArithRef]

TWeightShape = tuple[Literal[400], Literal[28], Literal[28]] | tuple[Literal[10], Literal[400], Literal[1]]
TWeightList = list[np.ndarray[TWeightShape, np.dtype[np.float_]]]

TImage = np.ndarray[tuple[Literal[28],Literal[28]], np.dtype[np.int_]]
TImageBatch = np.ndarray[tuple[Literal[60000],Literal[28],Literal[28]], np.dtype[np.int_]]
TLabelBatch = np.ndarray[tuple[Literal[60000]], np.dtype[np.int_]]