# Adversarial Robustness Verification for Spiking Neural Networks at Scale

Reference implementation for the paper *Adversarial Robustness Verification
for Spiking Neural Networks at Scale* (NeurIPS 2026 submission).

The verifier checks the local adversarial robustness of TTFS (time-to-first-spike)
spiking neural networks against an L¹-bounded perturbation of input spike
times. The pipeline is sound and complete (verdicts match an exhaustive DFS
oracle on every tested configuration). Three sound layers compose into one
depth-first BnB driver:

1. **Incremental voltage caching** — rank-1 update of the hidden potential
   on each branch/unbranch (Sec. 4.1).
2. **Voltage-margin temporal-causality filter** — drops pixels whose weight
   is too small to cross any neuron's threshold margin (Sec. 4.2).
3. **Budget-Coupled IBP (BC-IBP)** — solves a 0/1-knapsack DP at every
   `(h, t)` to certify-prune whole subtrees under the joint L¹ budget
   (Sec. 4.3).

The same engine also exposes Z3 (QF-LIRA) and PuLP/CBC (big-M MILP)
encodings as principled sound-and-complete baselines (Sec. 3).

## Repository layout

```text
code/
├── adv_rob_mnist_module.py     Verifier core: Z3 / MILP / NumPy BnB drivers
├── bnb_ibp.py                  BC-IBP and β-branching primitives
├── batch_test.py               Single-run entry point (one (dataset, n_h, Δ))
├── train_relu_to_ttfs.py       ReLU→TTFS conversion for multi-hidden experiments (App. D)
├── mnist_download.py           Fetch raw MNIST under data/mnist/MNIST/raw/
├── requirements.txt
│
├── *_benchmark.py              One driver per paper row (see "Reproducing experiments")
├── temporal_sparsity_analysis.py   Per-sample temporal-concentration C aggregation (Fig. 4)
├── test_bnb_ibp.py             BC-IBP soundness regression (single-hidden)
├── test_multilayer_bcibp.py    BC-IBP soundness regression (multi-hidden, App. D)
├── result_analysis.ipynb       Tables/figures from bench_results/ JSONs
├── theoretical_analysis.ipynb  Perturbation-space scaling plot (Theorem 1)
│
├── utils/                      CFG, dataset loaders, forward simulator, ANN trainer
├── baseline/                   Banerjee et al. SMT baseline (rate-coded)
├── S4NN/                       TTFS training (Kheradpisheh & Masquelier, 2020)
├── snntorch/                   Vendored copy used by neuromorphic loaders
├── data/                       Datasets (gitignored — see Datasets)
└── models/                     Trained weights (gitignored — see Training)
```

The LaTeX source lives in the sibling `paper/` directory.

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.12, Linux, AMD EPYC 7763. The verifier is single-thread CPU
by design (`OMP_NUM_THREADS=1` is set on import). A GPU is only useful for the
PyTorch ReLU pre-train in [train_relu_to_ttfs.py](train_relu_to_ttfs.py).

## Datasets

```text
data/
├── mnist/MNIST/raw/        python mnist_download.py
├── FashionMNIST/           torchvision auto-download (first run)
├── cifar-10-batches-py/    torchvision auto-download (first run)
├── nmnist/                 https://www.garrickorchard.com/datasets/n-mnist
├── dvsgesture/             https://research.ibm.com/interactive/dvsgesture
└── cifar10_dvs/            https://figshare.com/articles/CIFAR10-DVS_New/4724671
```

Loaders live in [utils/load.py](utils/load.py).

## Training

Single-hidden TTFS networks are trained with the original
[S4NN](S4NN/) recipe; one script per dataset:

```bash
python S4NN/S4NN.py              # MNIST  (T=5,  n_h ∈ {10,…,500})
python S4NN/S4NN_fmnist.py       # FashionMNIST (T=256)
python S4NN/S4NN_cifar.py        # CIFAR-10
python S4NN/S4NN_nmnist.py       # N-MNIST
python S4NN/S4NN_dvs_gesture.py  # DVS Gesture
python S4NN/S4NN_cifar10_dvs.py  # CIFAR10-DVS
```

Each writes weights into `models/<T>_<n_in>_<n_h>_<n_out>/` (the layout that
[utils/mnist_net.py](utils/mnist_net.py) `prepare_weights()` expects). The
threshold is fixed at θ=100 across all experiments.

The multi-hidden network of Appendix D is built by ReLU→TTFS conversion:

```bash
python train_relu_to_ttfs.py --hidden 100 100 --tmax 4 --epochs 5
# writes models/5_784_100_100_10_relu2ttfs_thr1.0/weights_*.npy
```

## Single verification run

[batch_test.py](batch_test.py) is the canonical entry point.

```bash
# MNIST, T=5, n_h=200, Δ=2, full BC-IBP pipeline
SNN_BNB_LEGACY_ACTIVE_SET=1 SNN_BNB_EFFICIENT=0 \
SNN_BNB_IBP=1 SNN_BNB_IBP_COUPLED=1 SNN_BNB_IBP_EVERY=100 \
python batch_test.py -p mnist_demo --test-type mnist --np \
    --n-hidden-neurons 200 --num-steps 5 --delta-max 2 \
    --num-samples 10 --seed 42
```

Per-sample logs land in `log/<prefix>*.log`; a benchmark driver post-processes
them into a JSON under `bench_results/`.

### CLI flags ([batch_test.py](batch_test.py))

| Flag | Meaning |
| --- | --- |
| `--test-type` | One of `mnist`, `fmnist`, `cifar`, `nmnist`, `dvs_gesture`, `cifar10_dvs` |
| `--n-hidden-neurons` | Hidden width n_h |
| `--num-steps` | Time horizon T |
| `--delta-max` | L¹ perturbation budget Δ |
| `--num-samples` | Number of test samples (default 14, paper uses 10) |
| `--seed` | Sample-selection seed (paper: 42) |
| `--np` | NumPy/Numba BnB backend (the paper's pipeline) |
| `--milp` | MILP encoding via PuLP/CBC (Sec. 3.2) |
| `--z3` | SMT encoding via Z3 (Sec. 3.1) |
| `--adv` | Run a quick adversarial search before verification (early-exit on hits) |

### Verifier environment variables

| Variable | Effect |
| --- | --- |
| `SNN_BNB_LEGACY_ACTIVE_SET=1` | Sound voltage-margin filter (always set this for Δ ≥ 2) |
| `SNN_BNB_EFFICIENT=0` | Disable the heuristic active-set (only sound at Δ=1) |
| `SNN_BNB_IBP=1` | Enable BC-IBP |
| `SNN_BNB_IBP_COUPLED=1` | Use the budget-coupled knapsack DP (Sec. 4.3); `=0` reproduces the uncoupled IBP ablation row |
| `SNN_BNB_IBP_EVERY=K` | Call BC-IBP every K visited nodes (paper: κ=100) |
| `SNN_BNB_BETA=1` | β-branching on BC-IBP (App. D, multi-hidden) |
| `SNN_BNB_BETA_DEPTH` / `SNN_BNB_BETA_TOPK` | β recursion depth / branching width |

The paper's headline configuration (Tables 2–3) is
`SNN_BNB_LEGACY_ACTIVE_SET=1 SNN_BNB_EFFICIENT=0 SNN_BNB_IBP=1 SNN_BNB_IBP_COUPLED=1 SNN_BNB_IBP_EVERY=100`.

## Reproducing the paper experiments

Each driver writes a timestamped `bench_results/<name>_<ts>.json` and per-sample
logs in `log/`. All drivers use seed 42, 10 samples per configuration, and a
300 s per-sample timeout.

| Driver | Paper artifact |
| --- | --- |
| [realistic_large_benchmark.py](realistic_large_benchmark.py) | Tables 1–2 (small-N) and Fig. 3 — Z3 / MILP / DFS / BnB at n_h ∈ {100, 200, 300, 500}, Δ=1 |
| [delta_high_benchmark.py](delta_high_benchmark.py) | Table 2 right block — Δ ∈ {2, 3, 4} sweep |
| [fmnist_benchmark.py](fmnist_benchmark.py) | Table 4(a) — FashionMNIST, T=256, n_h ∈ {512, 1024}, Δ=2 |
| [cifar_benchmark.py](cifar_benchmark.py) | Table 4(a) — CIFAR-10, T=5, n_h=512, Δ ∈ {1, 2} |
| [nmnist_benchmark.py](nmnist_benchmark.py) | Table 4(b) — N-MNIST, n_h=100, Δ ∈ {1, 2} |
| [dvs_gesture_benchmark.py](dvs_gesture_benchmark.py) | Table 4(b) — DVS Gesture, n_h=100 |
| [dvs_gesture500_benchmark.py](dvs_gesture500_benchmark.py) | Table 4(b) — DVS Gesture, n_h=500 |
| [cifar10_dvs_benchmark.py](cifar10_dvs_benchmark.py) | Table 4(b) — CIFAR10-DVS, n_h=100 |
| [temporal_sparsity_analysis.py](temporal_sparsity_analysis.py) | Fig. 4 — temporal concentration C vs. BC-IBP time |

The two notebooks consume these JSONs:

- [result_analysis.ipynb](result_analysis.ipynb) renders Tables 1–4 and Fig. 3.
- [theoretical_analysis.ipynb](theoretical_analysis.ipynb) plots the rate-vs-temporal
  perturbation-space separation of Theorem 1.

The κ-sweep of Table 3 reuses the BnB engine: re-run
[batch_test.py](batch_test.py) on the n_h=500 / Δ=2 configuration with
`SNN_BNB_IBP_EVERY` ∈ {10, 20, 50, 100, 200, 500, 1000}.

## Soundness regressions

```bash
python test_bnb_ibp.py           # single-hidden BC-IBP vs. forward simulation
python test_multilayer_bcibp.py  # multi-hidden BC-IBP vs. exhaustive DFS oracle
```

`test_multilayer_bcibp.py` reproduces Table 7 (Appendix D) on the
ReLU-converted 784→100→100→10 network.

## Acknowledgements

- The SMT baseline in [baseline/](baseline/) is adapted from
  [Soham-Banerjee/SMT-Encoding-for-Spiking-Neural-Network](https://github.com/Soham-Banerjee/SMT-Encoding-for-Spiking-Neural-Network)
  (Banerjee et al., 2023).
- The temporal-coded training in [S4NN/](S4NN/) is from
  [SRKH/S4NN](https://github.com/SRKH/S4NN)
  (Kheradpisheh & Masquelier, 2020).
- [snntorch/](snntorch/) is a vendored copy of
  [jeshraghian/snntorch](https://github.com/jeshraghian/snntorch),
  used only for the neuromorphic dataset loaders.
