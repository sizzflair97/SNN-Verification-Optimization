# Towards Efficient Formal Verification of Spiking Neural Networks
Implemented with python==3.12. Further informations are provided in requirements.txt

## Files
Baseline is from https://github.com/Soham-Banerjee/SMT-Encoding-for-Spiking-Neural-Network.
baseline/snnTrain.py - Trains MNIST baseline model.
baseline/snnTrainFMNIST.py - Trains FashionMNIST baseline model.
baseline/advRobustness.py - Perform formal verification of MNIST baseline model.
baseline/advRobustnessFMNIST.py - Perform formal verification of FashionMNIST baseline model.

Temporal model training implementation is from https://github.com/SRKH/S4NN.
S4NN/S4NN.ipynb - Trains MNIST temporal model.
S4NN/S4NN_fmnist.ipynb - Trains FashionMNIST temporal model.
batch_test.py - Perform formal verification about temporal models.

## ANN Train
```shell
python -m utils.ann --n-hidden-neurons 512
```

## SNN Verification
```shell
python batch_test.py -p latency --delta-max 2 --test-type mnist --np --num-samples 14 --n-hidden-neurons 512 --num-steps 256 --adv --repeat 1
```