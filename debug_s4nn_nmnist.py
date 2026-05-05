#!/usr/bin/env python3
"""Targeted debug of S4NN N-MNIST trainer.

Mirrors S4NN_nmnist.py forward + backward exactly, but:
  - tiny run: 200 iterations, then reproduce forward on the SAME weights
    via utils.mnist_net.forward() to compare.
  - prints winner/label/correct for first 10 iters.
  - reports prediction distribution on a held-out test slice after training.
"""
import os, sys
os.environ.setdefault("CUPY_CACHE_DIR", "/tmp/cupy_cache_dbg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cupy as cp
from utils.config import CFG
from utils.load import load_nmnist
from utils.mnist_net import forward

cp.cuda.Device(0).use()

NhidenNeurons = 100
NumOfClasses = 10
Nlayers = 2
num_steps = 5
tmax = num_steps - 1
lr = [0.2, 0.2]
lamda = [1e-6, 1e-6]
b = [2.0, 6.0]
a = [0, 0]
thr = [100.0, 100.0]
gamma = 2
N_ITERS = 200

np.random.seed(0)

cfg = CFG(log_name="dbg", subtype="nmnist", load_data_func=load_nmnist,
          seed=0, num_samples=1, deltas=(1,),
          n_layer_neurons=(2048, NhidenNeurons, NumOfClasses),
          layer_shapes=((64, 32), (NhidenNeurons, 1), (NumOfClasses, 1)),
          num_steps=num_steps)
x_tr, y_tr, x_te, y_te = load_nmnist(cfg)
images = cp.asarray(x_tr[:N_ITERS])
labels = cp.asarray(y_tr[:N_ITERS])
print(f"train slice: images={images.shape} labels[:10]={labels[:10].get()}")

Nnrn = [NhidenNeurons, NumOfClasses]
layerSize = [[64, 32], [NhidenNeurons, 1], [NumOfClasses, 1]]
SpikeImage = cp.zeros((layerSize[0][0], layerSize[0][1], tmax + 1))
Spikes = [cp.zeros((layerSize[l+1][0], layerSize[l+1][1], tmax+1)) for l in range(Nlayers)]
SpikeList = [SpikeImage] + Spikes
firingTime = [cp.zeros(Nnrn[l]) for l in range(Nlayers)]
X = [cp.asarray(np.mgrid[0:layerSize[l+1][0], 0:layerSize[l+1][1]]) for l in range(Nlayers)]
x = cp.mgrid[0:layerSize[0][0], 0:layerSize[0][1]]

W = []
for layer in range(Nlayers):
    W.append(cp.asarray((b[layer] - a[layer]) * np.random.random_sample(
        (Nnrn[layer], layerSize[layer][0], layerSize[layer][1])) + a[layer]))

target = cp.zeros([NumOfClasses])
correct = 0

print("\n=== First 10 iters: ===")
for iteration in range(N_ITERS):
    SpikeImage[:, :, :] = 0
    SpikeImage[x[0], x[1], images[iteration]] = 1

    for layer in range(Nlayers):
        Voltage = cp.cumsum(cp.tensordot(W[layer], SpikeList[layer]), 1)
        Voltage[:, tmax] = thr[layer] + 1
        firingTime[layer] = cp.argmax(Voltage > thr[layer], axis=1).astype(float) + 1
        firingTime[layer][firingTime[layer] > tmax] = tmax
        Spikes[layer][:, :, :] = 0
        Spikes[layer][X[layer][0], X[layer][1],
                      firingTime[layer].reshape(Nnrn[layer], 1).astype(int)] = 1

    winner = np.argmin(firingTime[1])
    is_correct = bool(winner == labels[iteration])
    if is_correct:
        correct += 1

    if iteration < 10:
        ft1 = firingTime[1].get()
        print(f"iter {iteration}: winner={int(winner)}  label={int(labels[iteration])}  "
              f"correct={is_correct}  firingTime[1]={ft1.tolist()}")

    # --- S4NN weight update ---
    minFiring = min(firingTime[Nlayers - 1])
    if minFiring == tmax:
        target[:] = minFiring
        target[labels[iteration]] = minFiring - gamma
        target = target.astype(int)
    else:
        target[:] = firingTime[Nlayers - 1][:]
        toChange = (firingTime[Nlayers - 1] - minFiring) < gamma
        target[toChange] = min(minFiring + gamma, tmax)
        target[labels[iteration]] = minFiring

    layer = Nlayers - 1
    delta_o = (target - firingTime[layer]) / tmax
    norm = cp.linalg.norm(delta_o)
    if norm != 0:
        delta_o = delta_o / norm
    hasFired_o = firingTime[layer - 1] < firingTime[layer][:, cp.newaxis]
    W[layer][:, :, 0] -= (delta_o[:, cp.newaxis] * hasFired_o * lr[layer])
    W[layer] -= lr[layer] * lamda[layer] * W[layer]

    delta_h = (cp.multiply(delta_o[:, cp.newaxis] * hasFired_o, W[layer][:, :, 0])).sum(axis=0)
    layer = Nlayers - 2
    norm = cp.linalg.norm(delta_h)
    if norm != 0:
        delta_h = delta_h / norm
    hasFired_h = images[iteration] < firingTime[layer][:, cp.newaxis, cp.newaxis]
    W[layer] -= lr[layer] * delta_h[:, cp.newaxis, cp.newaxis] * hasFired_h
    W[layer] -= lr[layer] * lamda[layer] * W[layer]

print(f"\nOnline 'correct' = {correct}/{N_ITERS} = {100*correct/N_ITERS:.1f}%")

# Verify with utils.mnist_net.forward() on 200 test samples using these weights
weights_np = [W[0].get(), W[1].get().reshape(NumOfClasses, NhidenNeurons, 1)]
# W[1] is (10, 100, 1) already — correct shape
weights_np = [W[0].get(), W[1].get()]
print(f"W[0] shape: {weights_np[0].shape}, W[1] shape: {weights_np[1].shape}")

correct_indep = 0
preds = []
for i in range(200):
    ft = []
    pred = forward(cfg, weights_np, x_te[i], ft)
    preds.append(int(pred))
    if pred == y_te[i]:
        correct_indep += 1
from collections import Counter
print(f"Independent forward() on 200 test samples: {correct_indep}/200 = {100*correct_indep/200:.1f}%")
print(f"Prediction distribution: {sorted(Counter(preds).items())}")
print(f"Label distribution: {sorted(Counter(y_te[:200].tolist()).items())}")
