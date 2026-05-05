"""S4NN training with configurable multi-hidden-layer support, numpy backend.

Adapted from S4NN/S4NN.py for multi-hidden experiments. Saves weights in the
format expected by `utils.mnist_net.prepare_weights`:
  models/{T}_{n_0}_{n_1}_..._{n_L}/weights_{layer}.npy
"""
from __future__ import division
import os
import os.path as osp
import time
import argparse

import numpy as np
from mnist import MNIST
from tqdm.auto import tqdm

# -------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--hidden", type=int, nargs="+", default=[30, 30],
                    help="Hidden-layer widths (e.g., --hidden 30 30 for two hidden layers)")
parser.add_argument("--tmax", type=int, default=4, help="num_steps - 1")
parser.add_argument("--epochs", type=int, default=15)
_S4NN_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_S4NN_DIR)
parser.add_argument("--mnist-root", type=str,
                    default=os.path.join(_CODE_DIR, "data/mnist/MNIST/raw/"))
parser.add_argument("--save-root", type=str,
                    default=os.path.join(_CODE_DIR, "models/"))
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

NumOfClasses = 10
hidden = list(args.hidden)
Nnrn = hidden + [NumOfClasses]                    # neurons per layer (excluding input)
Nlayers = len(Nnrn)                               # number of weight layers
tmax = args.tmax
GrayLevels = 255
gamma = 3
Nepoch = args.epochs

# Per-layer hyperparams (broadcast first to all if not enough provided)
def expand(x, default):
    if len(x) >= Nlayers:
        return x[:Nlayers]
    return x + [default] * (Nlayers - len(x))

thr   = expand([100], 100)
lr    = expand([0.2], 0.2)
lamda = expand([1e-6], 1e-6)
b     = expand([5, 50], 50)              # last layer gets larger weight init
a     = expand([0], 0)
Dropout = expand([0], 0)

print(f"Architecture: 784 -> {' -> '.join(map(str, Nnrn))}")
print(f"T = {tmax + 1}, epochs = {Nepoch}")

# -------------------------------------------------------------------
# Load MNIST (TTFS-encoded: pixel value -> spike time)
mndata = MNIST(args.mnist_root)
Images, Labels = mndata.load_training()
Images = np.array(Images)
images = []
labels = []
for i in range(len(Labels)):
    images.append(np.floor((GrayLevels - Images[i].reshape(28, 28)) * tmax / GrayLevels).astype(int))
    labels.append(Labels[i])
images = np.asarray(images)
labels = np.asarray(labels)

Images, Labels = mndata.load_testing()
Images = np.array(Images)
images_test = []
labels_test = []
for i in range(len(Labels)):
    images_test.append(np.floor((GrayLevels - Images[i].reshape(28, 28)) * tmax / GrayLevels).astype(int))
    labels_test.append(Labels[i])
images_test = np.asarray(images_test)
labels_test = np.asarray(labels_test)

# -------------------------------------------------------------------
# Initialize layers
layerSize = [[28, 28]] + [[n, 1] for n in Nnrn]
W = []
firingTime = []
Spikes = []
X = []
np.random.seed(args.seed)
for layer in range(Nlayers):
    shape = (Nnrn[layer], layerSize[layer][0], layerSize[layer][1])
    W.append((b[layer] - a[layer]) * np.random.random_sample(shape) + a[layer])
    firingTime.append(np.zeros(Nnrn[layer]))
    Spikes.append(np.zeros((layerSize[layer + 1][0], layerSize[layer + 1][1], tmax + 1)))
    X.append(np.mgrid[0:layerSize[layer + 1][0], 0:layerSize[layer + 1][1]])

x_grid = np.mgrid[0:28, 0:28]
SpikeImage = np.zeros((28, 28, tmax + 1))
SpikeList = [SpikeImage] + Spikes

# Per-layer target firing storage
target = np.zeros(NumOfClasses)

# -------------------------------------------------------------------
# Save dir
save_dir_path = osp.join(args.save_root, f"{tmax + 1}_784_{'_'.join(map(str, Nnrn))}")
os.makedirs(save_dir_path, exist_ok=True)
print(f"Save dir: {save_dir_path}")

best_perf = 0.0
test_perf_history = []
for epoch in tqdm(range(Nepoch), desc=f"T={tmax+1} h={hidden} epoch"):
    start_time = time.time()
    correct = 0
    total = 0
    FiringFrequency = np.zeros(Nnrn[0])

    perm = np.random.permutation(len(images))
    for it in tqdm(perm, desc="iter", leave=False):
        # Forward
        SpikeImage[:, :, :] = 0
        SpikeImage[x_grid[0], x_grid[1], images[it]] = 1
        for layer in range(Nlayers):
            voltage = np.cumsum(np.tensordot(W[layer], SpikeList[layer]), 1)
            voltage[:, tmax] = thr[layer] + 1
            ft = (np.argmax(voltage > thr[layer], axis=1).astype(float) + 1)
            ft[ft > tmax] = tmax
            firingTime[layer] = ft
            Spikes[layer][:, :, :] = 0
            Spikes[layer][X[layer][0], X[layer][1], ft.reshape(Nnrn[layer], 1).astype(int)] = 1

        FiringFrequency += (firingTime[0] < tmax)

        winner = int(np.argmin(firingTime[Nlayers - 1]))
        if winner == labels[it]:
            correct += 1
        total += 1

        # Target firing time computation (last-layer)
        minFiring = float(firingTime[Nlayers - 1].min())
        if minFiring == tmax:
            target[:] = minFiring
            target[labels[it]] = minFiring - gamma
        else:
            target[:] = firingTime[Nlayers - 1][:]
            toChange = (firingTime[Nlayers - 1] - minFiring) < gamma
            target[toChange] = min(minFiring + gamma, tmax)
            target[labels[it]] = minFiring
        target_int = target.astype(int)

        # Backprop layer-by-layer (S4NN-style temporal BP)
        # Output layer
        delta_out = (firingTime[Nlayers - 1] - target_int) / tmax  # shape (n_out,)
        # Backprop to last hidden using simple chain (S4NN paper eqn)
        # We simplify: hidden gradient is (W^T @ delta_out) using the weights
        # appropriate to the timestep at which hidden's spike contributes.
        deltas = [None] * Nlayers
        deltas[Nlayers - 1] = delta_out
        for L in range(Nlayers - 2, -1, -1):
            # Gradient flowing from layer L+1 back to layer L's neurons
            ft_next = firingTime[L + 1].astype(int)
            # For each neuron in layer L+1, only weights from layer L spikes that
            # arrived BEFORE ft_next contribute. We treat the sign of dV/dW as +1
            # for spikes-arrived-by-then.
            ft_L = firingTime[L].astype(int)
            # Build "did this layer L neuron's spike contribute to L+1 neuron's firing?"
            # = 1 if ft_L[h] < ft_next[h_next]
            # (n_{L+1}, n_L)
            W_eff = W[L + 1].squeeze(axis=2) if W[L + 1].ndim == 3 else W[L + 1]
            # contrib_mask shape (n_{L+1}, n_L): True if ft_L[h] < ft_next[h_next]
            contrib_mask = (ft_L[None, :] < ft_next[:, None])
            grad_L = (deltas[L + 1][:, None] * W_eff * contrib_mask).sum(axis=0)
            deltas[L] = grad_L

        # Weight updates
        for L in range(Nlayers):
            ft_post = firingTime[L].astype(int)
            # pre-synaptic spikes are SpikeList[L] of shape (n_pre_dim, ..., tmax+1)
            pre = SpikeList[L]
            # Sum over time up to ft_post for each post neuron
            for n_idx in range(Nnrn[L]):
                t_post = ft_post[n_idx]
                if t_post < 1:
                    continue
                # contribution = pre fired at t < t_post
                pre_contrib = pre[..., :t_post].any(axis=-1).astype(float)
                W[L][n_idx] -= lr[L] * deltas[L][n_idx] * pre_contrib
            # L1 weight regularization
            W[L] -= lr[L] * lamda[L] * np.sign(W[L])

    train_perf = correct / total

    # Test (subset for speed)
    correct_te = 0
    test_n = min(2000, len(images_test))
    test_idx = np.random.permutation(len(images_test))[:test_n]
    for it in test_idx:
        SpikeImage[:, :, :] = 0
        SpikeImage[x_grid[0], x_grid[1], images_test[it]] = 1
        for layer in range(Nlayers):
            voltage = np.cumsum(np.tensordot(W[layer], SpikeList[layer]), 1)
            voltage[:, tmax] = thr[layer] + 1
            ft = (np.argmax(voltage > thr[layer], axis=1).astype(float) + 1)
            ft[ft > tmax] = tmax
            firingTime[layer] = ft
            Spikes[layer][:, :, :] = 0
            Spikes[layer][X[layer][0], X[layer][1], ft.reshape(Nnrn[layer], 1).astype(int)] = 1
        winner = int(np.argmin(firingTime[Nlayers - 1]))
        if winner == labels_test[it]:
            correct_te += 1
    test_perf = correct_te / test_n
    test_perf_history.append(test_perf)
    print(f"epoch {epoch}: train={train_perf:.4f}  test={test_perf:.4f}  ({time.time()-start_time:.1f}s)")

    if test_perf > best_perf:
        best_perf = test_perf
        for layer in range(Nlayers):
            # Reshape weights to expected format
            # weights_0: (n_h1, 28, 28); weights_k for k>=1: (n_{k+1}, n_k, 1)
            np.save(osp.join(save_dir_path, f"weights_{layer}"), W[layer])
        print(f"  ↳ best test={best_perf:.4f}, saved")

    # Reset dead neurons (first hidden layer only)
    if Nlayers >= 1:
        ResetCheck = FiringFrequency < 0.001 * len(images)
        for i in range(Nnrn[0]):
            if ResetCheck[i]:
                W[0][i] = (b[0] - a[0]) * np.random.random_sample((28, 28)) + a[0]

print(f"Final best test accuracy: {best_perf:.4f}")
print(f"Test perf history: {[f'{x:.3f}' for x in test_perf_history]}")
