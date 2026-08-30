# Adversarial Robustness Verification for Spiking Neural Networks at Scale

Reference implementation for the paper *Adversarial Robustness Verification
for Spiking Neural Networks at Scale* (NeurIPS 2026 submission, double-blind).

The verifier checks local adversarial robustness of TTFS (time-to-first-spike)
spiking neural networks against an L¹-bounded perturbation of input spike
times. Every reported verdict is sound and complete (matches an exhaustive
DFS oracle on every tested configuration).

The pipeline composes four sound primitives into one depth-first BnB
driver:

1. **Incremental voltage caching** — rank-1 update of the hidden
   potential on each branch/unbranch (paper §4).
2. **Voltage-margin temporal-causality filter** — drops pixels whose
   weight is below every threshold margin (paper §4).
3. **Why not linear envelopes?** — vacuous-envelope theorem ruling out
   CROWN-style bounds on the step activation (paper §5.1; Fig. 1).
4. **Budget-Coupled IBP (BC-IBP)** — 0/1-knapsack DP per `(h, t)`
   that certifies pruning under the joint L¹ budget (paper §5.2).
5. **Cross-Layer Exact Bound (CLEB)** — input-perturbation enumeration
   for multi-hidden networks at Δ ≤ 2; sound and tight by Theorem 4
   (paper §5.3).

Z3 (QF-LIRA) and PuLP/CBC (big-M MILP) encodings are included as
principled sound-and-complete baselines (paper §3).

## Repository layout

```text
code/
├── bnb_ibp.py                    BC-IBP and multi-hidden bound primitives
├── bnb_ibp_pair.py               CLEB (cross-layer enumeration, GPU vectorised)
├── adv_rob_mnist_module.py       Verifier core: Z3 / MILP / NumPy BnB drivers
├── batch_test.py                 Single-run entry point ((dataset, n_h, Δ))
├── train_relu_to_ttfs.py         Single-scalar ReLU→TTFS conversion (App. C.7)
│
├── *_benchmark.py                One driver per paper row (see below)
├── cleb_benchmark.py             Multi-hidden CLEB / oracle benchmark (Table 4)
├── temporal_sparsity_analysis.py Per-sample C aggregation (Fig. 3)
├── compute_input_distrib_metrics.py  Dataset-level C / entropy summaries
│
├── test_bnb_ibp.py               BC-IBP soundness regression (single-hidden)
├── test_multilayer_bcibp.py      BC-IBP soundness regression (multi-hidden)
│
├── result_analysis.ipynb         Renders Tables 1–4 from bench_results JSONs
├── theoretical_analysis.ipynb    Theorem 1 perturbation-space plot
│
├── utils/                        CFG, dataset loaders, forward simulator
├── baseline/                     SMT baseline of Banerjee et al. (2023)
├── S4NN/                         TTFS training (Kheradpisheh & Masquelier)
├── snntorch/                     Vendored neuromorphic dataset loaders
├── data/                         Datasets (gitignored — see Datasets)
├── models/                       Trained weights (gitignored — see Training)
└── archive_legacy/               Earlier conversion / pair-bound attempts
                                  not used by the paper.
```

The LaTeX source lives in the sibling `paper/` directory.

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.12, Linux, AMD EPYC 7763. The verifier itself is
single-thread CPU by design (`OMP_NUM_THREADS=1` is set on import).
A CUDA GPU is required only for the multi-hidden CLEB enumeration
(`bnb_ibp_pair.ibp_prove_robust_multilayer_exact_gpu`) and the optional
PyTorch ReLU pre-train.

## Datasets

```text
data/
├── mnist/MNIST/raw/              python mnist_download.py
├── FashionMNIST/                 torchvision auto-download (first run)
├── cifar-10-batches-py/          torchvision auto-download (first run)
├── nmnist/                       https://www.garrickorchard.com/datasets/n-mnist
├── dvsgesture/                   https://research.ibm.com/interactive/dvsgesture
└── cifar10_dvs/                  https://figshare.com/articles/CIFAR10-DVS_New/4724671
```

Loaders live in [utils/load.py](utils/load.py). Event-driven datasets
are TTFS-encoded once and cached as `*_ttfs_T{T}.npz` next to the raw
files; subsequent runs reuse the cache.

## Training

Single-hidden TTFS networks use the original
[S4NN](S4NN/) recipe; one driver per dataset:

```bash
python S4NN/S4NN.py              # MNIST  (T=5, n_h ∈ {10, …, 500})
python S4NN/S4NN_fmnist.py       # FashionMNIST (T=256)
python S4NN/S4NN_cifar.py        # CIFAR-10
python S4NN/S4NN_nmnist.py       # N-MNIST
python S4NN/S4NN_dvs_gesture.py  # DVS Gesture
python S4NN/S4NN_cifar10_dvs.py  # CIFAR10-DVS
```

Each writes weights into `models/<T>_<n_in>_<n_h>_<n_out>/` (the layout
that [utils/mnist_net.py](utils/mnist_net.py) `prepare_weights()`
expects). The threshold is fixed at θ = 100 across all experiments.

The two multi-hidden TTFS fixtures used in Table 4 are produced by
ReLU→TTFS conversion (paper App. C.7):

```bash
python train_relu_to_ttfs.py --hidden 100 100 --tmax 4 --epochs 5  --scale 100
python train_relu_to_ttfs.py --hidden 200 200 --tmax 4 --epochs 30 --scale 100
```

These write `models/5_784_100_100_10/` and `models/5_784_200_200_10/`
respectively.

## CLI flags

`batch_test.py` exposes the verifier:

| Flag | Meaning |
| --- | --- |
| `--test-type` | One of `mnist`, `fmnist`, `cifar`, `nmnist`, `dvs_gesture`, `cifar10_dvs` |
| `--n-hidden-neurons` | Hidden width n_h |
| `--num-steps` | Time horizon T |
| `--delta-max` | L¹ perturbation budget Δ |
| `--num-samples` | Number of test samples (paper uses 10) |
| `--seed` | Sample-selection seed (paper: 42) |
| `--np` | NumPy/Numba BnB backend (the paper's pipeline) |
| `--milp` | MILP encoding via PuLP/CBC (paper §3) |
| `--z3` | SMT encoding via Z3 (paper §3) |

Verifier environment variables (defaults shown in parentheses):

| Variable | Default | Effect |
| --- | --- | --- |
| `SNN_BNB_LEGACY_ACTIVE_SET` | 0 | Sound voltage-margin filter (set 1 for Δ ≥ 2) |
| `SNN_BNB_EFFICIENT` | 1 | Heuristic active-set (set 0 to use only the sound filter) |
| `SNN_BNB_IBP` | 0 | Enable BC-IBP |
| `SNN_BNB_IBP_COUPLED` | 0 | Use the budget-coupled knapsack DP (paper §5.2); =0 reproduces the uncoupled IBP ablation row |
| `SNN_BNB_IBP_EVERY` | 100 | Call BC-IBP every K visits (paper: κ = 100) |

The paper's headline configuration (Tables 2–3) is
`SNN_BNB_LEGACY_ACTIVE_SET=1 SNN_BNB_EFFICIENT=0 SNN_BNB_IBP=1 SNN_BNB_IBP_COUPLED=1 SNN_BNB_IBP_EVERY=100`.

## Reproducing the paper experiments

All drivers use `--seed 42`, 10 samples per configuration, and a
300 s per-sample timeout. Each writes a JSON to `bench_results/`.

| Paper artefact | Driver / command |
| --- | --- |
| **Table 1** (small-N SMT vs. MILP vs. DFS) | `python batch_test.py -p small_N10  --test-type mnist --np --milp --z3 --num-steps 5 --delta-max 1 --n-hidden-neurons 10  --num-samples 14` and `--n-hidden-neurons 20`. |
| **Table 2** (MNIST scaling, Δ ∈ {1, …, 4}, n_h ∈ {100, 200, 300, 500}) | `python realistic_large_benchmark.py` (Δ = 1) and `python delta_high_benchmark.py` (Δ ≥ 2). |
| **Table 3 (a)** FMNIST | `python fmnist_benchmark.py` |
| **Table 3 (a)** CIFAR-10 | `python cifar_benchmark.py` |
| **Table 3 (b)** N-MNIST | `python nmnist_benchmark.py` |
| **Table 3 (b)** DVS Gesture, n_h = 100 | `python dvs_gesture_benchmark.py` |
| **Table 3 (b)** DVS Gesture, n_h = 500 | `python dvs_gesture500_benchmark.py` |
| **Table 3 (b)** CIFAR10-DVS | `python cifar10_dvs_benchmark.py` |
| **Table 4** (multi-hidden CLEB) | `python cleb_benchmark.py --hidden 100 100 --tmax 4 --delta {1,2}` and `--hidden 200 200`. |
| **Table 5** (App. C.2 small-N detail) | Same as Table 1; `result_analysis.ipynb` aggregates. |
| **Table 6** (App. C.3 κ sweep on sample 9144) | `for K in 10 20 50 100 200 500 1000; do SNN_BNB_IBP_EVERY=$K python batch_test.py -p kappa_$K --test-type mnist --np --n-hidden-neurons 500 --delta-max 2 --num-samples 1 --manual-indices 9144 --num-steps 5; done` |
| **Table 7** (App. C.4 per-sample n_h = 500, Δ = 2) | Subset of `delta_high_benchmark.py` output, rendered by `result_analysis.ipynb`. |
| **Figure 1** (vacuous envelope) | TikZ figure in [paper/main.tex](../paper/main.tex), no script. |
| **Figure 2** (App. C.5 SMT/MILP scalability) | `result_analysis.ipynb` reads the `realistic_bench_*.json` files. |
| **Figure 3** (App. C.6 temporal-sparsity scatter) | `python temporal_sparsity_analysis.py` |
| **Figure 4** (verifier overview) | TikZ figure in `paper/main.tex`, no script. |

`compute_input_distrib_metrics.py` reports per-dataset C and entropy
summaries underlying §7.3.

## Soundness regressions

```bash
python test_bnb_ibp.py             # single-hidden BC-IBP vs. forward simulation
python test_multilayer_bcibp.py    # multi-hidden BC-IBP vs. exhaustive DFS oracle
python cleb_benchmark.py --hidden 100 100 --delta 1 --n-samples 10
                                   # also serves as CLEB soundness check
```

The multi-hidden tests reproduce the soundness validation of
Appendix C.7.

## Acknowledgements (third-party code)

- The SMT baseline in [baseline/](baseline/) is adapted from the
  open-source release of Banerjee et al. (2023) — see paper.
- The TTFS training in [S4NN/](S4NN/) is from the original S4NN release
  (Kheradpisheh & Masquelier, 2020).
- [snntorch/](snntorch/) is a vendored copy used only for the
  neuromorphic dataset loaders.

All other code is original.

## Continuous-time LIF BnB

The LIF verifier is separate from the cumulative-IF pipeline:

- `lif_bnb_ibp.py` implements sound alpha-PSP interval bounds, subtree MCKP,
  NumPy/Torch backends, adaptive time cells, and recursive multilayer propagation.
- `lif_bnb.py` implements complete finite-domain DFS/BnB. A `robust` verdict is
  emitted only after every canonical integer time-shift assignment is either
  evaluated exactly or removed by a sound bound.
- `lif_bnb_experiments.py` runs resumable h350 paper-protocol experiments and
  writes one fsynced JSON object per completed job under `bench_results/lif/`.

The reference checkpoint is
`/data/SNN-Verification-Optimization/models/lif_goltz_ct_784_350_10_seed0/best.pt`.
The default runner requires the epoch-26 SHA-256
`a7c7c660857bec105abad959fa546de840f63634687ed19fc838546c7c257eac`
and snapshots it to `bench_results/lif/checkpoints/<sha256>.pt` before workers
start. This prevents a concurrently running trainer from mixing checkpoints.
For its `[0.15, 2.0]` input interval and the paper's `T=5` protocol, one
integer shift is `eta=(2.0-0.15)/(5-1)=0.4625`.

```bash
PYTHONPATH=code /opt/conda/envs/snn-verification/bin/python code/test_lif_bnb.py
PYTHONPATH=code /opt/conda/envs/snn-verification/bin/python code/lif_bnb_experiments.py --suite pilot --backend torch
PYTHONPATH=code /opt/conda/envs/snn-verification/bin/python code/lif_bnb_experiments.py --suite table1 --workers 8
PYTHONPATH=code /opt/conda/envs/snn-verification/bin/python code/lif_bnb_experiments.py --suite scaling --workers 8
PYTHONPATH=code /opt/conda/envs/snn-verification/bin/python code/lif_bnb_experiments.py --suite kappa --workers 7
PYTHONPATH=code /opt/conda/envs/snn-verification/bin/python code/lif_bnb_experiments.py --suite incidence --workers 8
```

Verdicts are `robust`, `not_robust`, `timeout`, `numerical_unknown`, or
`error`. Every `not_robust` row contains a full integer shift vector, sparse
shift map, perturbed spike times, L1 cost, and independently replayed
checkpoint prediction. SMT/MILP rows and experiments lacking a LIF checkpoint
are explicitly represented in `coverage_manifest.json` as `N/A` or
`pending_checkpoint`; they are never silently compared with a discretized
surrogate.
