"""Build PUMA's Sudoku caches (train_mdm.npy, test_mdm.npy) in parallel.

Produces exactly what data/sudoku.py would write on first use, using the same
per-example converter, but spreads the ~1.8M conversions over CPU workers.
Usage: python scripts/preprocess_sudoku_cache.py data/sudoku_new [num_workers]
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocess_sudoku import sudoku_example_to_162

_SRC = None


def _init(path):
    global _SRC
    _SRC = np.load(path, mmap_mode="r")


def _convert(bounds):
    lo, hi = bounds
    out = np.zeros((hi - lo, 162))
    bad = 0
    for j, i in enumerate(range(lo, hi)):
        out[j], meta = sudoku_example_to_162(_SRC[i])
        bad += int(not meta["givens_ok"])
    return lo, out, bad


def build(src, dst, workers, chunk=5000):
    n = np.load(src, mmap_mode="r").shape[0]
    result = np.zeros((n, 162))
    bad_total = 0
    chunks = [(lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]
    with Pool(workers, initializer=_init, initargs=(src,)) as pool:
        for k, (lo, out, bad) in enumerate(pool.imap_unordered(_convert, chunks)):
            result[lo:lo + len(out)] = out
            bad_total += bad
            if k % 50 == 0:
                print(f"{os.path.basename(dst)}: {k + 1}/{len(chunks)} chunks", flush=True)
    sol = result[:, 81:]
    assert ((sol >= 1) & (sol <= 9)).all(), "incomplete solution grid in cache"
    tmp = dst + ".tmp.npy"
    np.save(tmp, result)
    os.replace(tmp, dst)
    print(f"saved {dst} shape={result.shape} givens_violations={bad_total}", flush=True)


if __name__ == "__main__":
    data_dir = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    for src, dst in [("sudoku-test-data.npy", "test_mdm.npy"), ("sudoku-train-data.npy", "train_mdm.npy")]:
        dst = os.path.join(data_dir, dst)
        if os.path.exists(dst):
            print(f"exists, skipping: {dst}")
            continue
        build(os.path.join(data_dir, src), dst, workers)
