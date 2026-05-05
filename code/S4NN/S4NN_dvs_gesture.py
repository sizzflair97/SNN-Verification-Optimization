"""S4NN training for DVS Gesture (IBM DVS128, 11 classes, 128x128x2 event frames).

Adapted from S4NN_nmnist.py. Input is (256, 128) TTFS-encoded: two polarity
channels stacked vertically (ON rows [0:128), OFF rows [128:256)).

Saves weights to ../models/dg_{T}_{input_size}_{H}_11/.
Usage:
  CUPY_CACHE_DIR=/tmp/cupy_cache python3 S4NN_dvs_gesture.py \
      --num-steps 5 --hidden 100 --epochs 50 [--no-gpu]
"""
from __future__ import division
import argparse
import os
import os.path as osp
import sys
import time

import numpy as np
from tqdm.auto import tqdm

ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from utils.config import CFG
from utils.load import load_dvs_gesture

parser = argparse.ArgumentParser()
parser.add_argument("--num-steps", type=int, default=5)
parser.add_argument("--hidden", type=int, default=100)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, nargs=2, default=[0.2, 0.2])
parser.add_argument("--lamda", type=float, nargs=2, default=[1e-6, 1e-6])
parser.add_argument("--b", type=float, nargs=2, default=[0.5, 6.0],
                    help="upper bound of initial weights [hidden, output]; "
                         "smaller hidden-layer bound for 32768-input DVS Gesture")
parser.add_argument("--thr", type=float, nargs=2, default=[100.0, 100.0])
parser.add_argument("--no-gpu", action="store_true")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

if not args.no_gpu:
    import cupy as cp
    cp.cuda.Device(0).use()
else:
    cp = np

NhidenNeurons = args.hidden
Nepoch = args.epochs
NumOfClasses = 11
Nlayers = 2
lr = args.lr
lamda = args.lamda
b = args.b
a = [0, 0]
tmax = args.num_steps - 1
gamma = 2
best_perf = 0
saving = True

Nnrn = [NhidenNeurons, NumOfClasses]

np.random.seed(args.seed)

print(f"Loading DVS Gesture (T={args.num_steps}) ...")
cfg = CFG(
    log_name="dvs_gesture_s4nn_train",
    subtype="dvs_gesture",
    load_data_func=load_dvs_gesture,
    seed=args.seed,
    num_samples=1,
    deltas=(1,),
    n_layer_neurons=(2 * 128 * 128, NhidenNeurons, NumOfClasses),
    layer_shapes=((256, 128), (NhidenNeurons, 1), (NumOfClasses, 1)),
    num_steps=args.num_steps,
)
x_train_np, y_train_np, x_test_np, y_test_np = load_dvs_gesture(cfg)
print(f"train: {x_train_np.shape}, test: {x_test_np.shape}, "
      f"val_range=[{x_train_np.min()},{x_train_np.max()}]")

images = cp.asarray(x_train_np)
labels = cp.asarray(y_train_np)
images_test = cp.asarray(x_test_np)
labels_test = cp.asarray(y_test_np)

layerSize = [[images[0].shape[0], images[0].shape[1]],
             [NhidenNeurons, 1],
             [NumOfClasses, 1]]
input_size = layerSize[0][0] * layerSize[0][1]
x = cp.mgrid[0:layerSize[0][0], 0:layerSize[0][1]]
SpikeImage = cp.zeros((layerSize[0][0], layerSize[0][1], tmax + 1))

W = []
firingTime = []
Spikes = []
X = []
for layer in range(Nlayers):
    W.append(cp.asarray(
        (b[layer] - a[layer]) * np.random.random_sample(
            (Nnrn[layer], layerSize[layer][0], layerSize[layer][1])) + a[layer]))
    firingTime.append(cp.asarray(np.zeros(Nnrn[layer])))
    Spikes.append(cp.asarray(np.zeros((layerSize[layer + 1][0], layerSize[layer + 1][1], tmax + 1))))
    X.append(cp.asarray(np.mgrid[0:layerSize[layer + 1][0], 0:layerSize[layer + 1][1]]))
SpikeList = [SpikeImage] + Spikes

save_dir_path = osp.join(
    ROOT, f"models/dg_{args.num_steps}_{input_size}_{NhidenNeurons}_{NumOfClasses}"
)
os.makedirs(save_dir_path, exist_ok=True)
acc_log_path = osp.join(save_dir_path, "train_log.txt")

target = cp.zeros([NumOfClasses])

for epoch in tqdm(range(Nepoch), desc=f"T={args.num_steps} H={NhidenNeurons}"):
    start_time = time.time()
    FiringFrequency = cp.zeros((NhidenNeurons,))
    correct = 0
    total = 0
    perm = np.random.permutation(len(images))
    for iteration_idx in range(len(images)):
        iteration = int(perm[iteration_idx])
        SpikeImage[:, :, :] = 0
        SpikeImage[x[0], x[1], images[iteration]] = 1

        for layer in range(Nlayers):
            Voltage = cp.cumsum(cp.tensordot(W[layer], SpikeList[layer]), 1)
            Voltage[:, tmax] = args.thr[layer] + 1
            firingTime[layer] = cp.argmax(Voltage > args.thr[layer], axis=1).astype(float) + 1
            firingTime[layer][firingTime[layer] > tmax] = tmax
            Spikes[layer][:, :, :] = 0
            Spikes[layer][X[layer][0], X[layer][1],
                          firingTime[layer].reshape(Nnrn[layer], 1).astype(int)] = 1

        FiringFrequency = FiringFrequency + (firingTime[0] < tmax)

        winner = np.argmin(firingTime[1])
        if winner == labels[iteration]:
            correct += 1
        total += 1
        minFiring = min(firingTime[layer])
        if minFiring == tmax:
            target[:] = minFiring
            target[labels[iteration]] = minFiring - gamma
            target = target.astype(int)
        else:
            target[:] = firingTime[layer][:]
            toChange = (firingTime[layer] - minFiring) < gamma
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

    trainPerf = correct / total

    test_correct = 0
    for iteration in range(len(images_test)):
        SpikeImage[:, :, :] = 0
        SpikeImage[x[0], x[1], images_test[iteration]] = 1
        for layer in range(Nlayers):
            Voltage = cp.cumsum(cp.tensordot(W[layer], SpikeList[layer]), 1)
            Voltage[:, tmax] = args.thr[layer] + 1
            firingTime[layer] = cp.argmax(Voltage > args.thr[layer], axis=1).astype(float) + 1
            firingTime[layer][firingTime[layer] > tmax] = tmax
            Spikes[layer][:, :, :] = 0
            Spikes[layer][X[layer][0], X[layer][1],
                          firingTime[layer].reshape(Nnrn[layer], 1).astype(int)] = 1
        if np.argmin(firingTime[-1]) == labels_test[iteration]:
            test_correct += 1
    testPerf = test_correct / len(images_test)

    dt = time.time() - start_time
    msg = f"epoch={epoch} train_acc={trainPerf:.4f} test_acc={testPerf:.4f} elapsed={dt:.1f}s"
    print(msg, flush=True)
    with open(acc_log_path, "a") as f:
        f.write(msg + "\n")

    if saving and testPerf > best_perf:
        for layer in range(Nlayers):
            arr = W[layer].get() if hasattr(W[layer], "get") else W[layer]
            np.save(osp.join(save_dir_path, f"weights_{layer}"), arr)
        print(f"  best saved (test_acc={testPerf:.4f}) -> {save_dir_path}")
        best_perf = testPerf

    ResetCheck = FiringFrequency < 0.001 * len(images)
    ToReset = [i for i in range(NhidenNeurons) if bool(ResetCheck[i])]
    for i in ToReset:
        W[0][i] = cp.asarray(
            (b[0] - a[0]) * np.random.random_sample((layerSize[0][0], layerSize[0][1])) + a[0])

print(f"Training done. Best test_acc={best_perf:.4f}. Weights -> {save_dir_path}")
