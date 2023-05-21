# SMT-Encoding-for-Spiking-Neural-Network
Implementation demonstrating SMT encoding for Spiking Neural Networks used for Formal Verification and Adverserial Robustness checking. 

## Installation
The tool uses python to run the SMT encoding of the networks along with the solver.\
\
[Python](https://www.python.org/downloads/) installation\
\
Installing all dependencies
```
pip install requirement.txt
```

## File List

snnTrain.py -- Train an SNN save the model along with it's SMT encoding\
advRobustness -- Check Adv. Robustness for the trained and encoded SNN
adv_rob_iris.py -- Adversarial robustness for SNN-IRIS\
\
Models -- Folder containing the trained SNN models\
eqn -- Folder containing the encoded SNN models\

## Usage
1) Train SNN
Configure the following parameters and run snnTrain.py
```
batch_size = 128 # Training batch size
data_path = '/data/mnist' # Location to save data
location = 'C:\\Users\\soham\\PycharmProjects\\Z3py' # Directory of project
neurons_in_layers = [28*28, 100, 10] # List defining the architecture of the SNN
num_steps = 10 # Number of timesteps
beta = 0.95 # Decay (lambda)
```

2) Check Robustness
Configuire the following parameters and run advRobustness.py
```
neurons_in_layers = [28*28, 100, 10] # List defining the architecture of the SNN
num_steps = 10 # Number of timesteps
data_path = '/data/mnist' # Location to save data
delta = [1] # List of deltas to check
location = 'C:\\Users\\soham\\PycharmProjects\\Z3py' # Directory of project
```
