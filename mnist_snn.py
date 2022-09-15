import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools

batch_size = 128
data_path = '/data/mnist'

dtype = torch.float
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)


num_inputs = 28*28
num_hidden = 10
num_outputs = 10


num_steps = 25
beta = 0.95


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden, bias=False)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs, bias=False)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

# Load the network onto CUDA if available
net = Net()

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(num_steps, batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer():
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")


load = False
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

num_epochs = 1
loss_hist = []
test_loss_hist = []
counter = 0
if not load:
    # Outer training loop
    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)

        # Minibatch training loop
        for data, targets in train_batch:
            data = data
            targets = targets

            spike_data = spikegen.rate(data, num_steps=num_steps)

            # forward pass
            net.train()
            #spk_rec, mem_rec = net(data.view(batch_size, -1))
            spk_rec, mem_rec = net(spike_data.view(num_steps, batch_size, -1))

            # initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=dtype)
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            # Test set
            with torch.no_grad():
                net.eval()
                test_data, test_targets = next(iter(test_loader))
                test_data = test_data
                test_targets = test_targets

                test_spike_data = spikegen.rate(test_data, num_steps=num_steps)
                # Test set forward pass
                test_spk, test_mem = net(test_spike_data.view(num_steps, batch_size, -1))

                # Test set loss
                test_loss = torch.zeros((1), dtype=torch.float)
                for step in range(num_steps):
                    test_loss += loss(test_mem[step], test_targets)
                test_loss_hist.append(test_loss.item())

                # Print train/test loss/accuracy
                if counter % 50 == 0:
                    print(f"Epoch {epoch}, Iteration {iter_counter}")
                    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
                    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
                    print_batch_accuracy(spike_data, targets, train=True)
                    print_batch_accuracy(test_spike_data, test_targets, train=False)
                    print("\n")
                counter += 1
                iter_counter += 1

    #torch.save(net, f'/Models/model_{num_hidden}.pth')
    torch.save(net, f'C:\\Users\\soham\\PycharmProjects\\Z3py\\Models\\model_{num_hidden}.pth')
    print("Model Saved")
else:
    net = torch.load('model.pth')
    print('Model loaded')

total = 0
correct = 0
test_batch_size = 100
# drop_last switched to False to keep all samples
test_loader = DataLoader(mnist_test, batch_size=test_batch_size, shuffle=True, drop_last=False)

with torch.no_grad():
    net.eval()
    c = 1
    for data, targets in test_loader:
        data = data
        targets = targets

        test_spike_data = spikegen.rate(data, num_steps=num_steps)

        # forward pass
        test_spk, _ = net(test_spike_data.view(num_steps, test_batch_size, -1))
        # calculate total accuracy
        _, predicted = test_spk.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        print(f'Batch {c} done')
        c += 1

print(f"Total correctly classified test set images: {correct}/{total}")
print()