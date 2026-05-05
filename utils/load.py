from . import CFG
from .dictionary_mnist import TImageBatch, TLabelBatch
from .debug import info
from mnist import MNIST
from torchvision.datasets import FashionMNIST, CIFAR10
import numpy as np
import typing
import glob
import os
import zipfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from snntorch.spikevision._utils import load_ATIS_bin


def load_mnist(cfg: CFG) -> tuple[TImageBatch, TLabelBatch, TImageBatch, TLabelBatch]:
    # Parameter setting
    num_steps = cfg.num_steps
    GrayLevels = 255  # Image GrayLevels
    cats = [*range(10)]

    # General variables
    images = []  # To keep training images
    labels = []  # To keep training labels
    images_test = []  # To keep test images
    labels_test = []  # To keep test labels

    # loading MNIST dataset
    mndata = MNIST("data/mnist/MNIST/raw/")

    Images, Labels = mndata.load_training()
    Images = np.array(Images)
    for i in range(len(Labels)):
        if Labels[i] in cats:
            images.append(np.floor((GrayLevels - Images[i].reshape(28, 28)) * (num_steps - 1) / GrayLevels).astype(int))
            labels.append(cats.index(Labels[i]))
    Images, Labels = mndata.load_testing()
    Images = np.array(Images)
    for i in range(len(Labels)):
        if Labels[i] in cats:
            images_test.append(
                np.floor((GrayLevels - Images[i].reshape(28, 28)) * (num_steps - 1) / GrayLevels).astype(int)
            )
            labels_test.append(cats.index(Labels[i]))

    del Images, Labels

    # images contain values within [0,num_steps]
    images = typing.cast(TImageBatch, np.asarray(images))
    labels = typing.cast(TLabelBatch, np.asarray(labels))
    images_test = typing.cast(TImageBatch, np.asarray(images_test))
    labels_test = typing.cast(TLabelBatch, np.asarray(labels_test))

    return images, labels, images_test, labels_test


def load_fmnist(cfg: CFG) -> tuple[TImageBatch, TLabelBatch, TImageBatch, TLabelBatch]:
    # Parameter setting
    num_steps = cfg.num_steps
    GrayLevels = 255  # Image GrayLevels
    cats = [*range(10)]

    # General variables
    images = []  # To keep training images
    labels = []  # To keep training labels
    images_test = []  # To keep test images
    labels_test = []  # To keep test labels

    # loading FMNIST dataset
    fmdata = FashionMNIST("./data/", train=True, download=True)

    Images, Labels = fmdata.data.numpy(), fmdata.targets.numpy()
    Images = np.array(Images)
    for i in range(len(Labels)):
        if Labels[i] in cats:
            images.append(
                np.floor((GrayLevels - Images[i].reshape(28, 28).astype(int)) * (num_steps - 1) / GrayLevels).astype(
                    int
                )
            )
            labels.append(cats.index(Labels[i]))
    Images, Labels = fmdata.data.numpy(), fmdata.targets.numpy()
    Images = np.array(Images)
    for i in range(len(Labels)):
        if Labels[i] in cats:
            # images_test.append(TTT[i].reshape(28,28).astype(int))
            images_test.append(
                np.floor((GrayLevels - Images[i].reshape(28, 28).astype(int)) * (num_steps - 1) / GrayLevels).astype(
                    int
                )
            )
            labels_test.append(cats.index(Labels[i]))

    del Images, Labels

    # images contain values within [0,num_steps]
    images = typing.cast(TImageBatch, np.asarray(images))
    labels = typing.cast(TLabelBatch, np.asarray(labels))
    images_test = typing.cast(TImageBatch, np.asarray(images_test))
    labels_test = typing.cast(TLabelBatch, np.asarray(labels_test))

    return images, labels, images_test, labels_test


# ---------- N-MNIST (real DVS-recorded MNIST) ----------

_NMNIST_CROP = 32  # snntorch convention: crop 34x34 → 32x32
_NMNIST_DIR = Path("data/nmnist")


def _events_to_ttfs(ts: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray, num_steps: int) -> np.ndarray:
    """Convert one event stream to a (2*32, 32) TTFS image.

    Two polarity channels stacked vertically: ON rows [0:32), OFF rows [32:64).
    Each pixel value = first-event time bucket in [0, num_steps-2]; num_steps-1 = no event.
    """
    img = np.full((2 * _NMNIST_CROP, _NMNIST_CROP), num_steps - 1, dtype=int)
    if len(ts) == 0:
        return img

    valid = (x < _NMNIST_CROP) & (y < _NMNIST_CROP)
    ts, x, y, p = ts[valid], x[valid], y[valid], p[valid]
    if len(ts) == 0:
        return img

    ts_max = int(ts.max())
    if ts_max == 0:
        ts_norm = np.zeros_like(ts, dtype=int)
    else:
        ts_norm = np.floor(ts.astype(np.float64) * (num_steps - 2) / ts_max).astype(int)

    sort_idx = np.argsort(ts, kind="stable")
    ts_s = ts_norm[sort_idx]
    x_s = x[sort_idx].astype(np.int64)
    y_s = y[sort_idx].astype(np.int64)
    p_s = p[sort_idx].astype(np.int64)

    pix_id = p_s * (_NMNIST_CROP * _NMNIST_CROP) + y_s * _NMNIST_CROP + x_s
    _unique_pix, first_idx = np.unique(pix_id, return_index=True)
    first_pix = pix_id[first_idx]
    ch = first_pix // (_NMNIST_CROP * _NMNIST_CROP)
    yy = (first_pix % (_NMNIST_CROP * _NMNIST_CROP)) // _NMNIST_CROP
    xx = first_pix % _NMNIST_CROP
    rows = (ch * _NMNIST_CROP + yy).astype(int)
    cols = xx.astype(int)
    img[rows, cols] = ts_s[first_idx]
    return img


def _ensure_nmnist_extracted() -> None:
    for zip_name, sub in (("Train.zip", "Train"), ("Test.zip", "Test")):
        zpath = _NMNIST_DIR / zip_name
        sub_dir = _NMNIST_DIR / sub
        if not sub_dir.exists():
            if not zpath.exists():
                raise FileNotFoundError(f"{zpath} not found. Download N-MNIST first.")
            info(f"Extracting {zpath} ...")
            with zipfile.ZipFile(zpath) as z:
                z.extractall(_NMNIST_DIR)


def _load_nmnist_split(split_dir: Path, num_steps: int) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for digit in range(10):
        bin_files = sorted(glob.glob(str(split_dir / str(digit) / "*.bin")))
        for bf in bin_files:
            ts, x, y, p = load_ATIS_bin(bf)
            images.append(_events_to_ttfs(ts, x, y, p, num_steps))
            labels.append(digit)
    return np.asarray(images), np.asarray(labels)


def load_nmnist(cfg: CFG) -> tuple[TImageBatch, TLabelBatch, TImageBatch, TLabelBatch]:
    """Load N-MNIST as TTFS-encoded images of shape (2*32, 32) = (64, 32).

    Pixel value ∈ [0, num_steps-1] = first-event time bucket per (channel, x, y).
    Channels stacked vertically: ON polarity rows [0:32), OFF polarity rows [32:64).
    """
    num_steps = cfg.num_steps
    _ensure_nmnist_extracted()

    cache_path = _NMNIST_DIR / f"nmnist_ttfs_T{num_steps}.npz"
    if cache_path.exists():
        info(f"Loading cached N-MNIST TTFS from {cache_path}")
        z = np.load(cache_path)
        return (
            typing.cast(TImageBatch, z["x_train"]),
            typing.cast(TLabelBatch, z["y_train"]),
            typing.cast(TImageBatch, z["x_test"]),
            typing.cast(TLabelBatch, z["y_test"]),
        )

    info("Encoding N-MNIST train events → TTFS ...")
    x_train, y_train = _load_nmnist_split(_NMNIST_DIR / "Train", num_steps)
    info("Encoding N-MNIST test events → TTFS ...")
    x_test, y_test = _load_nmnist_split(_NMNIST_DIR / "Test", num_steps)
    np.savez_compressed(cache_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    info(f"Cached N-MNIST TTFS to {cache_path} ({x_train.shape[0]} train / {x_test.shape[0]} test)")

    return (
        typing.cast(TImageBatch, x_train),
        typing.cast(TLabelBatch, y_train),
        typing.cast(TImageBatch, x_test),
        typing.cast(TLabelBatch, y_test),
    )


# ---------- DVS Gesture (IBM DVS128) ----------

_DVSGESTURE_CROP = 128
_DVSGESTURE_DIR = Path("data/dvsgesture/DvsGesture")
_DVSGESTURE_CACHE = Path("data/dvsgesture")

# Train/test split by subject id: users 1-23 train, 24-29 test
# (matches snntorch / torchneuromorphic convention).
_TRAIN_USERS = range(1, 24)
_TEST_USERS = range(24, 30)


def _dvs_events_to_ttfs(
    ts: np.ndarray, xs: np.ndarray, ys: np.ndarray, ps: np.ndarray, num_steps: int,
) -> np.ndarray:
    """Convert event stream in [0, t_max] to a (2*128, 128) TTFS image.

    Two polarity channels stacked vertically: ON rows [0:128), OFF rows [128:256).
    Pixel value = first-event time bucket in [0, num_steps-2]; num_steps-1 = no event.
    """
    img = np.full((2 * _DVSGESTURE_CROP, _DVSGESTURE_CROP), num_steps - 1, dtype=int)
    if len(ts) == 0:
        return img
    valid = (xs < _DVSGESTURE_CROP) & (ys < _DVSGESTURE_CROP)
    ts, xs, ys, ps = ts[valid], xs[valid], ys[valid], ps[valid]
    if len(ts) == 0:
        return img
    ts_max = int(ts.max())
    if ts_max == 0:
        ts_norm = np.zeros_like(ts, dtype=int)
    else:
        ts_norm = np.floor(ts.astype(np.float64) * (num_steps - 2) / ts_max).astype(int)
    sort_idx = np.argsort(ts, kind="stable")
    ts_s = ts_norm[sort_idx]
    xs_s = xs[sort_idx].astype(np.int64)
    ys_s = ys[sort_idx].astype(np.int64)
    ps_s = ps[sort_idx].astype(np.int64)
    pix_id = ps_s * (_DVSGESTURE_CROP * _DVSGESTURE_CROP) + ys_s * _DVSGESTURE_CROP + xs_s
    _unique_pix, first_idx = np.unique(pix_id, return_index=True)
    first_pix = pix_id[first_idx]
    ch = first_pix // (_DVSGESTURE_CROP * _DVSGESTURE_CROP)
    yy = (first_pix % (_DVSGESTURE_CROP * _DVSGESTURE_CROP)) // _DVSGESTURE_CROP
    xx = first_pix % _DVSGESTURE_CROP
    rows = (ch * _DVSGESTURE_CROP + yy).astype(int)
    cols = xx.astype(int)
    img[rows, cols] = ts_s[first_idx]
    return img


def _load_dvsgesture_split(user_ids, num_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Parse .aedat files for the given users, slice per gesture segment, TTFS-encode."""
    from snntorch.spikevision._utils import aedat_to_events

    imgs: list[np.ndarray] = []
    labels: list[int] = []
    for uid in user_ids:
        user_files = sorted(_DVSGESTURE_DIR.glob(f"user{uid:02d}_*.aedat"))
        for ad in user_files:
            try:
                data, lab_arr = aedat_to_events(str(ad))
            except Exception as exc:
                info(f"skipping {ad.name}: {exc}")
                continue
            if data.size == 0:
                continue
            # data cols: [timestamp_us, x, y, polarity]
            ts = data[:, 0]
            xs = data[:, 1]
            ys = data[:, 2]
            ps = data[:, 3]
            # lab_arr rows: [class_id(1..11), start_us, end_us]
            for row in lab_arr:
                cls, tstart, tend = int(row[0]), int(row[1]), int(row[2])
                if cls < 1 or cls > 11:
                    continue
                mask = (ts >= tstart) & (ts < tend)
                if not mask.any():
                    continue
                seg_ts = ts[mask] - tstart
                seg_xs = xs[mask]
                seg_ys = ys[mask]
                seg_ps = ps[mask]
                img = _dvs_events_to_ttfs(seg_ts, seg_xs, seg_ys, seg_ps, num_steps)
                imgs.append(img)
                labels.append(cls - 1)
    return np.asarray(imgs), np.asarray(labels)


def load_dvs_gesture(cfg: CFG) -> tuple[TImageBatch, TLabelBatch, TImageBatch, TLabelBatch]:
    """Load DVS Gesture as TTFS-encoded images of shape (2*128, 128) = (256, 128).

    11 hand gesture classes, 1176 train / 288 test samples (by-subject split).
    Pixel value ∈ [0, num_steps-1]: first-event time bucket per (channel, x, y).
    """
    num_steps = cfg.num_steps
    cache_path = _DVSGESTURE_CACHE / f"dvsgesture_ttfs_T{num_steps}.npz"
    if cache_path.exists():
        info(f"Loading cached DVS Gesture TTFS from {cache_path}")
        z = np.load(cache_path)
        return (
            typing.cast(TImageBatch, z["x_train"]),
            typing.cast(TLabelBatch, z["y_train"]),
            typing.cast(TImageBatch, z["x_test"]),
            typing.cast(TLabelBatch, z["y_test"]),
        )
    if not _DVSGESTURE_DIR.exists():
        raise FileNotFoundError(
            f"{_DVSGESTURE_DIR} not found. Extract DvsGesture.tar.gz into "
            f"{_DVSGESTURE_CACHE}/ first."
        )
    info("Encoding DVS Gesture train events → TTFS (this takes a few minutes) ...")
    x_train, y_train = _load_dvsgesture_split(_TRAIN_USERS, num_steps)
    info("Encoding DVS Gesture test events → TTFS ...")
    x_test, y_test = _load_dvsgesture_split(_TEST_USERS, num_steps)
    np.savez_compressed(cache_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    info(f"Cached DVS Gesture TTFS to {cache_path} ({x_train.shape[0]} train / {x_test.shape[0]} test)")
    return (
        typing.cast(TImageBatch, x_train),
        typing.cast(TLabelBatch, y_train),
        typing.cast(TImageBatch, x_test),
        typing.cast(TLabelBatch, y_test),
    )


# ---------- CIFAR10-DVS (Li et al. 2017, event-based CIFAR-10) ----------

_CIFAR10DVS_CROP = 128
_CIFAR10DVS_DIR = Path("data/cifar10_dvs/CIFAR10DVS")
_CIFAR10DVS_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _read_aedat2(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a DVS128 AEDAT 2.0 file. Returns (ts, x, y, polarity) arrays.
    DVS128 bit layout (big-endian 8-byte records):
        address : 32-bit; y=[8:14], x=[1:7], polarity=bit 0
        timestamp : 32-bit microseconds
    """
    with open(filename, "rb") as f:
        # Skip header lines beginning with '#' (ASCII 0x23).
        while True:
            pos = f.tell()
            line = f.readline()
            if not line or not line.startswith(b"#"):
                f.seek(pos)
                break
        raw = np.frombuffer(f.read(), dtype=">u4")
    if raw.size < 2:
        return (np.empty(0, dtype=np.uint32),) * 4
    raw = raw.reshape(-1, 2)
    addr = raw[:, 0]; ts = raw[:, 1]
    # DVS128 masks (jAER convention)
    x = (addr & 0x00FE) >> 1
    y = (addr & 0x7F00) >> 8
    p = (addr & 0x1)
    return ts, x, y, p


def _events_to_ttfs_128(
    ts: np.ndarray, xs: np.ndarray, ys: np.ndarray, ps: np.ndarray, num_steps: int,
) -> np.ndarray:
    """Same as N-MNIST encoder but 128x128 with polarity stacked vertically:
    ON rows [0:128), OFF rows [128:256). Shape (256, 128)."""
    img = np.full((2 * _CIFAR10DVS_CROP, _CIFAR10DVS_CROP), num_steps - 1, dtype=int)
    if len(ts) == 0:
        return img
    valid = (xs < _CIFAR10DVS_CROP) & (ys < _CIFAR10DVS_CROP)
    ts, xs, ys, ps = ts[valid], xs[valid], ys[valid], ps[valid]
    if len(ts) == 0:
        return img
    ts_max = int(ts.max())
    if ts_max == 0:
        ts_norm = np.zeros_like(ts, dtype=int)
    else:
        ts_norm = np.floor(ts.astype(np.float64) * (num_steps - 2) / ts_max).astype(int)
    sort_idx = np.argsort(ts, kind="stable")
    ts_s = ts_norm[sort_idx]
    xs_s = xs[sort_idx].astype(np.int64)
    ys_s = ys[sort_idx].astype(np.int64)
    ps_s = ps[sort_idx].astype(np.int64)
    pix_id = ps_s * (_CIFAR10DVS_CROP * _CIFAR10DVS_CROP) + ys_s * _CIFAR10DVS_CROP + xs_s
    _u, first_idx = np.unique(pix_id, return_index=True)
    first_pix = pix_id[first_idx]
    ch = first_pix // (_CIFAR10DVS_CROP * _CIFAR10DVS_CROP)
    yy = (first_pix % (_CIFAR10DVS_CROP * _CIFAR10DVS_CROP)) // _CIFAR10DVS_CROP
    xx = first_pix % _CIFAR10DVS_CROP
    rows = (ch * _CIFAR10DVS_CROP + yy).astype(int)
    cols = xx.astype(int)
    img[rows, cols] = ts_s[first_idx]
    return img


def load_cifar10_dvs(cfg: CFG) -> tuple[TImageBatch, TLabelBatch, TImageBatch, TLabelBatch]:
    """Load CIFAR10-DVS (Li et al., 2017) as TTFS-encoded (256, 128) images.

    Expects each class's .aedat files under
    `data/cifar10_dvs/CIFAR10DVS/<class_name>/*.aedat`.
    1000 samples per class × 10 classes = 10000 total.
    80/20 train/test split (by sample index within class) for reproducibility.
    """
    num_steps = cfg.num_steps
    cache_path = Path("data/cifar10_dvs") / f"cifar10dvs_ttfs_T{num_steps}.npz"
    if cache_path.exists():
        info(f"Loading cached CIFAR10-DVS TTFS from {cache_path}")
        z = np.load(cache_path)
        return (
            typing.cast(TImageBatch, z["x_train"]),
            typing.cast(TLabelBatch, z["y_train"]),
            typing.cast(TImageBatch, z["x_test"]),
            typing.cast(TLabelBatch, z["y_test"]),
        )

    if not _CIFAR10DVS_DIR.exists():
        raise FileNotFoundError(
            f"{_CIFAR10DVS_DIR} not found. Extract CIFAR10-DVS class zips into "
            f"{_CIFAR10DVS_DIR}/<class_name>/*.aedat first."
        )

    info("Encoding CIFAR10-DVS events → TTFS ...")
    x_train_l: list = []; y_train_l: list = []
    x_test_l: list = []; y_test_l: list = []
    for cls_idx, cls_name in enumerate(_CIFAR10DVS_CLASSES):
        cls_dir = _CIFAR10DVS_DIR / cls_name
        if not cls_dir.exists():
            info(f"WARN: class dir {cls_dir} missing; skipping class {cls_name}")
            continue
        files = sorted(cls_dir.glob("*.aedat"))
        info(f"  class {cls_idx} ({cls_name}): {len(files)} samples")
        for i, ad in enumerate(files):
            try:
                ts, xs, ys, ps = _read_aedat2(str(ad))
            except Exception as exc:
                info(f"skip {ad.name}: {exc}")
                continue
            img = _events_to_ttfs_128(ts, xs, ys, ps, num_steps)
            # 80/20 split by sample index (deterministic)
            if i < 800:
                x_train_l.append(img); y_train_l.append(cls_idx)
            else:
                x_test_l.append(img); y_test_l.append(cls_idx)

    x_train = np.asarray(x_train_l); y_train = np.asarray(y_train_l)
    x_test = np.asarray(x_test_l); y_test = np.asarray(y_test_l)
    np.savez_compressed(cache_path, x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test)
    info(f"Cached CIFAR10-DVS TTFS to {cache_path} "
         f"({x_train.shape[0]} train / {x_test.shape[0]} test)")

    return (
        typing.cast(TImageBatch, x_train),
        typing.cast(TLabelBatch, y_train),
        typing.cast(TImageBatch, x_test),
        typing.cast(TLabelBatch, y_test),
    )


def load_cifar(cfg: CFG) -> tuple[TImageBatch, TLabelBatch, TImageBatch, TLabelBatch]:
    """Load CIFAR-10 dataset with S4NN-compatible spike time encoding.

    Images are reshaped to (3*32, 32) = (96, 32) matching S4NN_cifar.py convention.
    Spike times are computed as floor((GrayLevels - pixel) * (num_steps-1) / GrayLevels).
    """
    num_steps = cfg.num_steps
    GrayLevels = 255
    cats = [*range(10)]

    images = []
    labels = []
    images_test = []
    labels_test = []

    # Load CIFAR-10 training set
    cifar_train = CIFAR10("./data/", train=True, download=True)
    Images_train = cifar_train.data  # (50000, 32, 32, 3) uint8
    Labels_train = np.array(cifar_train.targets)

    for i in range(len(Labels_train)):
        if Labels_train[i] in cats:
            # Reshape (32, 32, 3) -> (3*32, 32) = (96, 32) matching S4NN convention
            img = Images_train[i].transpose(2, 0, 1).reshape(3 * 32, 32).astype(int)
            images.append(
                np.floor((GrayLevels - img) * (num_steps - 1) / GrayLevels).astype(int)
            )
            labels.append(cats.index(Labels_train[i]))

    # Load CIFAR-10 test set
    cifar_test = CIFAR10("./data/", train=False, download=True)
    Images_test = cifar_test.data  # (10000, 32, 32, 3) uint8
    Labels_test = np.array(cifar_test.targets)

    for i in range(len(Labels_test)):
        if Labels_test[i] in cats:
            img = Images_test[i].transpose(2, 0, 1).reshape(3 * 32, 32).astype(int)
            images_test.append(
                np.floor((GrayLevels - img) * (num_steps - 1) / GrayLevels).astype(int)
            )
            labels_test.append(cats.index(Labels_test[i]))

    images = typing.cast(TImageBatch, np.asarray(images))
    labels = typing.cast(TLabelBatch, np.asarray(labels))
    images_test = typing.cast(TImageBatch, np.asarray(images_test))
    labels_test = typing.cast(TLabelBatch, np.asarray(labels_test))

    return images, labels, images_test, labels_test
