"""Precompute per-suite LIBERO action q01/q99 stats into a JSON cache.

Usage:
    python tools/compute_libero_stats.py \
        --data-root /workspace/tingting/LIBERO/libero/datasets \
        --suite libero_10 \
        --out vlm4vla/data/libero_stats/libero_10.json
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np

from vlm4vla.data.libero_constants import (
    libero_gripper_transform, noop_mask, GRIPPER_DIM,
)


def compute_stats_for_actions(actions: np.ndarray) -> dict:
    actions = np.asarray(actions, dtype=np.float64)
    q01 = np.quantile(actions, 0.01, axis=0)
    q99 = np.quantile(actions, 0.99, axis=0)
    mask = [True] * actions.shape[1]
    mask[GRIPPER_DIM] = False  # leave gripper dim un-normalized
    return {"action": {"q01": q01.tolist(), "q99": q99.tolist(), "mask": mask}}


def collect_suite_actions(data_root: str, suite: str) -> np.ndarray:
    import h5py
    files = sorted(glob.glob(os.path.join(data_root, suite, "*.hdf5")))
    if not files:
        raise FileNotFoundError(
            f"No HDF5 files under {os.path.join(data_root, suite)}")
    chunks = []
    for fp in files:
        with h5py.File(fp, "r") as h:
            for demo in h["data"].keys():
                raw = np.asarray(h["data"][demo]["actions"], dtype=np.float64)
                raw = libero_gripper_transform(raw)
                raw = raw[noop_mask(raw)]
                if len(raw):
                    chunks.append(raw)
    return np.concatenate(chunks, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    actions = collect_suite_actions(a.data_root, a.suite)
    stats = compute_stats_for_actions(actions)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {a.out}  ({len(actions)} action frames)")


if __name__ == "__main__":
    main()
