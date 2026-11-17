"""Download, verify, pre-process and save MNIST (digits 1-9) + sample Sudoku puzzles.

Run from the project root:
    python -m data.make_data

What it does (fully reproducible, fixed seed = config.SEED):
1. Downloads the 4 official MNIST IDX files into data/raw/ (skipped if present
   and the MD5 checksum matches). Several mirrors are tried because the
   original yann.lecun.com server frequently refuses automated downloads.
2. Parses the IDX binary format with NumPy (no torchvision needed).
3. Splits the official 60,000 training images into 50,000 train / 10,000
   validation using a seeded permutation, THEN removes digit 0 from every
   split. The official 10,000 test images are kept untouched (only 0s removed).
4. Saves data/processed/mnist_1to9.npz  (uint8 pixels; normalised to [0,1]
   by load_processed()).
5. Generates a small set of sample Sudoku puzzles + mosaic images into
   data/puzzles/ for the demo and the Streamlit app.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

import config

FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
}

# The official source is listed first; the others are byte-identical mirrors
# (verified with the MD5 checksums above).
MIRRORS = [
    "http://yann.lecun.com/exdb/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_mnist(raw_dir: Path = config.RAW_DIR) -> dict[str, Path]:
    """Download the 4 MNIST files if they are missing or corrupted."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, (fname, md5) in FILES.items():
        target = raw_dir / fname
        if target.exists() and _md5(target) == md5:
            print(f"[data] {fname}: already present (checksum OK) - skipping download")
            paths[key] = target
            continue
        errors = []
        for mirror in MIRRORS:
            url = mirror + fname
            try:
                print(f"[data] downloading {url}")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (P7 university project)"})
                tmp = target.with_suffix(".part")
                with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
                    shutil.copyfileobj(resp, out)
                if _md5(tmp) != md5:
                    tmp.unlink(missing_ok=True)
                    raise ValueError("MD5 checksum mismatch")
                tmp.replace(target)
                break
            except Exception as exc:  # noqa: BLE001 - try the next mirror
                print(f"[data]   failed ({exc}); trying next mirror")
                errors.append(f"{url}: {exc}")
        else:
            raise RuntimeError(
                "Could not download " + fname + " from any mirror:\n  " + "\n  ".join(errors)
                + f"\nDownload it manually and place it in {raw_dir}"
            )
        paths[key] = target
    return paths


def read_idx(path: Path) -> np.ndarray:
    """Parse a (gzipped) IDX file into a NumPy array."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    arr = np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim)
    return arr.reshape(dims)


def preprocess(paths: dict[str, Path], seed: int = config.SEED) -> dict[str, np.ndarray]:
    """Create the train/val/test splits for digits 1-9 (see module docstring)."""
    x_all = read_idx(paths["train_images"])
    y_all = read_idx(paths["train_labels"]).astype(np.int64)
    x_test = read_idx(paths["test_images"])
    y_test = read_idx(paths["test_labels"]).astype(np.int64)
    assert x_all.shape == (60_000, 28, 28) and x_test.shape == (10_000, 28, 28), "unexpected MNIST shapes"

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x_all))
    train_idx = perm[: config.N_TRAIN_SPLIT]
    val_idx = perm[config.N_TRAIN_SPLIT: config.N_TRAIN_SPLIT + config.N_VAL_SPLIT]
    test_idx = np.arange(len(x_test))

    out: dict[str, np.ndarray] = {}
    for name, xs, ys, idx in (("train", x_all, y_all, train_idx), ("val", x_all, y_all, val_idx),
                              ("test", x_test, y_test, test_idx)):
        keep = idx[ys[idx] != 0]                     # exclude digit 0
        out[f"x_{name}"] = xs[keep]                  # uint8 28x28
        out[f"y_{name}"] = ys[keep]                  # digit label 1..9
        out[f"idx_{name}"] = keep                    # index in the original MNIST file
    return out


def load_processed(path: Path = config.PROCESSED_FILE) -> dict[str, np.ndarray]:
    """Load the processed dataset; images are returned as float32 in [0,1]."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run:  python -m data.make_data")
    raw = np.load(path)
    data = {k: raw[k] for k in raw.files}
    for split in ("train", "val", "test"):
        data[f"x_{split}"] = data[f"x_{split}"].astype(np.float32) / 255.0
    return data


def save_sample_puzzles(data: dict[str, np.ndarray], n: int = 6, seed: int = config.SEED) -> None:
    """Generate demo puzzles (JSON + PNG mosaic) from MNIST *test* digits."""
    from data.puzzle_gen import generate_puzzle, render_mosaic, sample_digit_images

    config.PUZZLE_DIR.mkdir(parents=True, exist_ok=True)
    x_test = (data["x_test"]).astype(np.float32)
    if x_test.max() > 1.0:
        x_test = x_test / 255.0
    index = []
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        n_givens = [30, 32, 35][i % 3]
        puzzle, solution = generate_puzzle(n_givens, rng, unique=True)
        imgs, src = sample_digit_images(puzzle, x_test, data["y_test"], rng)
        mosaic = render_mosaic(imgs)
        name = f"puzzle_{i:02d}"
        from PIL import Image
        Image.fromarray(mosaic).save(config.PUZZLE_DIR / f"{name}.png")
        record = {"name": name, "seed": seed + i, "n_givens": int(n_givens), "puzzle": puzzle,
                  "solution": solution, "mnist_test_index": src}
        (config.PUZZLE_DIR / f"{name}.json").write_text(json.dumps(record))
        index.append({"name": name, "n_givens": int(n_givens)})
    (config.PUZZLE_DIR / "index.json").write_text(json.dumps(index, indent=2))
    print(f"[data] saved {n} sample puzzles to {config.PUZZLE_DIR}")


def main() -> int:
    config.ensure_dirs()
    paths = download_mnist()
    if config.PROCESSED_FILE.exists():
        print(f"[data] {config.PROCESSED_FILE.name} already exists - re-using it")
        data = dict(np.load(config.PROCESSED_FILE))
    else:
        data = preprocess(paths)
        np.savez_compressed(config.PROCESSED_FILE, **data)
        print(f"[data] saved {config.PROCESSED_FILE}")
    for split in ("train", "val", "test"):
        y = data[f"y_{split}"]
        counts = np.bincount(y, minlength=10)[1:]
        print(f"[data] {split:5s}: {len(y):6d} images, per-digit counts 1..9 = {counts.tolist()}")
    save_sample_puzzles(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
