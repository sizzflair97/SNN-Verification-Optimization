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

The files and folders are as follows;
mnist_snn.py -- Train an SNN on the MNIST dataset and save the model\
encode_MNIST.py -- Encode a trained SNN and save the encoding in a seperate file\
adv_rob_MNIST.py -- Adversarial robustness for SNN-MNIST\
adv_rob_iris.py -- Adversarial robustness for SNN-IRIS\
iris_property.py -- Check properties for SNN-IRIS\
\
Models -- Folder containing the trained SNN models\
eqn -- Folder containing the encoded SNN models
