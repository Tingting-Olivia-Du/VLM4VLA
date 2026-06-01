"""Pure-torch LIBERO HDF5 dataset, drop-in replacement for RLDSDataset.

Reproduces the OXE libero transform + no_noops + Q99 normalization, then
slides a (window_size + fwd_pred_next_n) window over each demo trajectory,
yielding the 5 fields consumed by ActionPredictionDataset.batch_transform.
"""
from __future__ import annotations
import glob
import itertools
import json
import os
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from vlm4vla.data.libero_constants import (
    libero_gripper_transform, noop_mask, normalize_q99,
)


class LiberoHDF5Dataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: str,
        data_mix: str,                 # suite name, e.g. "libero_10"
        image_size: int,
        window_size: int = 1,
        fwd_pred_next_n: int = 1,
        stats_path: Optional[str] = None,
        window_sample: str = "sliding",
        left_pad: bool = False,
        train: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_root_dir = data_root_dir
        self.suite = data_mix
        self.image_size = image_size
        self.window_size = window_size
        self.fwd_pred_next_n = fwd_pred_next_n
        self.window_sample = window_sample
        self.left_pad = left_pad
        self.train = train

        suite_dir = os.path.join(data_root_dir, self.suite)
        self.files = sorted(glob.glob(os.path.join(suite_dir, "*.hdf5")))
        if not self.files:
            raise FileNotFoundError(f"No HDF5 files under {suite_dir}")

        if stats_path is None:
            stats_path = os.path.join(
                os.path.dirname(__file__), "libero_stats", f"{self.suite}.json")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Stats cache missing: {stats_path}. Run "
                f"tools/compute_libero_stats.py --suite {self.suite} first.")
        with open(stats_path) as f:
            s = json.load(f)["action"]
        self.q01 = np.array(s["q01"], dtype=np.float64)
        self.q99 = np.array(s["q99"], dtype=np.float64)
        self.norm_mask = np.array(s["mask"], dtype=bool)

    # ---- windowing layer (implemented in Task 4) -------------------------
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError("Windowing layer not yet implemented; use iter_trajectories().")

    # ---- trajectory layer ------------------------------------------------
    def iter_trajectories(self) -> Iterator[Dict[str, Any]]:
        import h5py
        for fp in self.files:
            with h5py.File(fp, "r") as h:
                lang = json.loads(
                    h["data"].attrs["problem_info"])["language_instruction"]
                for demo in h["data"].keys():
                    g = h["data"][demo]
                    actions = np.asarray(g["actions"], dtype=np.float64)
                    agent = np.asarray(g["obs"]["agentview_rgb"])    # (T,H,W,3) uint8
                    wrist = np.asarray(g["obs"]["eye_in_hand_rgb"])
                    actions = libero_gripper_transform(actions)
                    keep = noop_mask(actions)
                    actions, agent, wrist = actions[keep], agent[keep], wrist[keep]
                    if len(actions) == 0:
                        continue
                    actions = normalize_q99(
                        actions, self.q01, self.q99, self.norm_mask)
                    yield {
                        "language": str(lang),
                        "actions": actions,
                        "images": agent.astype(np.uint8),
                        "gripper_images": wrist.astype(np.uint8),
                    }
