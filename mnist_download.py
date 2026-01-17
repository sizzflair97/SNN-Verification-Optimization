import os
import urllib.request
import gzip
import shutil

BASE_URL = "http://yann.lecun.com/exdb/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

ROOT = "data/mnist/MNIST/raw"


def download_and_extract(url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    filename = url
    gz_path = os.path.join(dest_dir, filename)
    out_path = gz_path.replace(".gz", "")

    if os.path.exists(out_path):
        print(f"[SKIP] {out_path} already exists")
        return

    print(f"[DOWNLOAD] {filename}")
    # urllib.request.urlretrieve(url, gz_path)

    print(f"[EXTRACT] {filename}")
    with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(gz_path)


def main():
    for fname in FILES:
        download_and_extract(fname, ROOT)

    print("\nMNIST raw files are ready:")
    for f in os.listdir(ROOT):
        print(" -", f)


if __name__ == "__main__":
    main()
