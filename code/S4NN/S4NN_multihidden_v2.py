"""S4NN training, multi-hidden, numpy backend (faithful port of S4NN.py).

Backprop preserves: (target - actual)/tmax error sign, per-layer
gradient L2-normalization, and hasFired pre-spike indicators.

Generalization to multi-hidden:
  - Output layer L:
      delta_L = (target - ft_L)/tmax, normalized
      hasFired[L][o, h] = ft_{L-1}[h] < ft_L[o]
      W[L][:, :, 0] -= lr * delta_L[:, None] * hasFired[L]
      delta_{L-1} = (delta_L[:, None] * hasFired[L] * W[L][:, :, 0]).sum(axis=0)
  - Intermediate hidden layer ell (1 <= ell <= L-1):
      delta_ell normalized
      hasFired[ell][h_post, h_pre] = ft_{ell-1}[h_pre] < ft_ell[h_post]
      W[ell][:, :, 0] -= lr * delta_ell[:, None] * hasFired[ell]
      delta_{ell-1} = (delta_ell[:, None] * hasFired[ell] * W[ell][:, :, 0]).sum(axis=0)
  - First hidden (ell = 0):
      delta_0 normalized
      hasFired[0][h, x, y] = image[x, y] < ft_0[h]
      W[0] -= lr * delta_0[:, None, None] * hasFired[0]
"""
from __future__ import division
import os
import os.path as osp
import time
import argparse

import numpy as np
from mnist import MNIST
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--hidden", type=int, nargs="+", default=[30, 30])
parser.add_argument("--tmax", type=int, default=4)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--lr-scale", type=float, default=1.0)
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

thr = [100] * Nlayers
lr  = [0.2 * args.lr_scale] * Nlayers
lamda = [1e-6] * Nlayers
b = [5] * (Nlayers - 1) + [50]
a = [0] * Nlayers

print(f"Architecture: 784 -> {' -> '.join(map(str, Nnrn))}, T={tmax+1}, epochs={Nepoch}")

# ---- Data ----
mndata = MNIST(args.mnist_root)
Images, Labels = mndata.load_training()
Images = np.array(Images)
images = np.array([np.floor((GrayLevels - im.reshape(28, 28)) * tmax / GrayLevels).astype(int)
                   for im in Images])
labels = np.array(Labels)
Images_te, Labels_te = mndata.load_testing()
Images_te = np.array(Images_te)
images_test = np.array([np.floor((GrayLevels - im.reshape(28, 28)) * tmax / GrayLevels).astype(int)
                        for im in Images_te])
labels_test = np.array(Labels_te)

# ---- Model ----
np.random.seed(args.seed)
layerSize = [[28, 28]] + [[n, 1] for n in Nnrn]
W = []
firingTime = [None] * Nlayers
Spikes = []
X = []
for layer in range(Nlayers):
    shape = (Nnrn[layer], layerSize[layer][0], layerSize[layer][1])
    W.append((b[layer] - a[layer]) * np.random.random_sample(shape) + a[layer])
    Spikes.append(np.zeros((layerSize[layer + 1][0], layerSize[layer + 1][1], tmax + 1)))
    X.append(np.mgrid[0:layerSize[layer + 1][0], 0:layerSize[layer + 1][1]])

x_grid = np.mgrid[0:28, 0:28]
SpikeImage = np.zeros((28, 28, tmax + 1))
SpikeList = [SpikeImage] + Spikes

save_dir_path = osp.join(args.save_root, f"{tmax + 1}_784_{'_'.join(map(str, Nnrn))}")
os.makedirs(save_dir_path, exist_ok=True)
print(f"Save dir: {save_dir_path}")

best_perf = 0.0
test_perf_history = []


def forward_pass(image_int):
    """Run forward; populate firingTime[*], Spikes[*]."""
    SpikeImage[:, :, :] = 0
    SpikeImage[x_grid[0], x_grid[1], image_int] = 1
    for layer in range(Nlayers):
        voltage = np.cumsum(np.tensordot(W[layer], SpikeList[layer]), 1)
        voltage[:, tmax] = thr[layer] + 1
        ft = (np.argmax(voltage > thr[layer], axis=1).astype(float) + 1)
        ft[ft > tmax] = tmax
        firingTime[layer] = ft
        Spikes[layer][:, :, :] = 0
        Spikes[layer][X[layer][0], X[layer][1], ft.reshape(Nnrn[layer], 1).astype(int)] = 1


def compute_target(ft_out, lab):
    """Relative target firing time as in S4NN paper."""
    target = np.zeros(NumOfClasses)
    minFiring = float(ft_out.min())
    if minFiring == tmax:
        target[:] = minFiring
        target[lab] = minFiring - gamma
    else:
        target[:] = ft_out[:]
        toChange = (ft_out - minFiring) < gamma
        target[toChange] = min(minFiring + gamma, tmax)
        target[lab] = minFiring
    return target.astype(int)


for epoch in tqdm(range(Nepoch), desc=f"T={tmax+1} h={hidden}"):
    start = time.time()
    correct = 0
    FiringFrequency = np.zeros(Nnrn[0])
    perm = np.random.permutation(len(images))

    for it in tqdm(perm, desc=f"epoch{epoch}", leave=False):
        forward_pass(images[it])
        FiringFrequency += (firingTime[0] < tmax)
        if int(np.argmin(firingTime[Nlayers - 1])) == labels[it]:
            correct += 1
        target = compute_target(firingTime[Nlayers - 1], labels[it])

        # -------- Backprop --------
        # Output layer L = Nlayers - 1
        L = Nlayers - 1
        delta = (target - firingTime[L]) / tmax
        nrm = np.linalg.norm(delta)
        if nrm > 0:
            delta = delta / nrm

        # hasFired[L][o, h] : ft_{L-1}[h] < ft_L[o]
        ft_pre = firingTime[L - 1]
        hf_L = ft_pre[np.newaxis, :] < firingTime[L][:, np.newaxis]   # (n_L, n_{L-1})
        W[L][:, :, 0] -= lr[L] * delta[:, np.newaxis] * hf_L
        W[L] -= lr[L] * lamda[L] * W[L]

        # Backprop into hidden layers (L-1, L-2, ..., 0)
        # delta_below: shape matches Nnrn[layer-1]
        delta_below = (delta[:, np.newaxis] * hf_L * W[L][:, :, 0]).sum(axis=0)  # (n_{L-1},)

        for layer in range(L - 1, -1, -1):
            nrm = np.linalg.norm(delta_below)
            if nrm > 0:
                delta_below = delta_below / nrm

            if layer == 0:
                # Input -> hidden_1; hasFired[0][h, x, y]: image[x,y] < ft_0[h]
                hf = images[it][np.newaxis, :, :] < firingTime[0][:, np.newaxis, np.newaxis]
                W[0] -= lr[0] * delta_below[:, np.newaxis, np.newaxis] * hf
                W[0] -= lr[0] * lamda[0] * W[0]
                # No further backprop after layer 0
            else:
                # Hidden_{layer-1} -> Hidden_{layer}
                # hasFired[layer][h_post, h_pre]: ft_{layer-1}[h_pre] < ft_{layer}[h_post]
                ft_pre_l = firingTime[layer - 1]
                hf = ft_pre_l[np.newaxis, :] < firingTime[layer][:, np.newaxis]
                W[layer][:, :, 0] -= lr[layer] * delta_below[:, np.newaxis] * hf
                W[layer] -= lr[layer] * lamda[layer] * W[layer]
                # Compute delta for the layer below
                delta_below = (delta_below[:, np.newaxis] * hf * W[layer][:, :, 0]).sum(axis=0)

    train_perf = correct / len(images)

    # ---- Test (subset 2000) ----
    correct_te = 0
    test_n = min(2000, len(images_test))
    test_idx = np.random.permutation(len(images_test))[:test_n]
    for it in test_idx:
        forward_pass(images_test[it])
        if int(np.argmin(firingTime[Nlayers - 1])) == labels_test[it]:
            correct_te += 1
    test_perf = correct_te / test_n
    test_perf_history.append(test_perf)
    print(f"epoch {epoch}: train={train_perf:.4f} test={test_perf:.4f} ({time.time()-start:.1f}s)")

    if test_perf > best_perf:
        best_perf = test_perf
        for layer in range(Nlayers):
            np.save(osp.join(save_dir_path, f"weights_{layer}"), W[layer])
        print(f"  saved (best test={best_perf:.4f})")

    # Reset dead first-hidden neurons
    ResetCheck = FiringFrequency < 0.001 * len(images)
    for i in range(Nnrn[0]):
        if ResetCheck[i]:
            W[0][i] = (b[0] - a[0]) * np.random.random_sample((28, 28)) + a[0]

print(f"\nFinal best test accuracy: {best_perf:.4f}")
print(f"History: {[f'{x:.3f}' for x in test_perf_history]}")
