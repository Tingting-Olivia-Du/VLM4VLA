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

        if window_sample != "sliding":
            raise NotImplementedError(
                f"LiberoHDF5Dataset only supports window_sample='sliding', "
                f"got {window_sample!r}.")

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

    # ---- window layer ----------------------------------------------------
    def _windows(self, traj: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        W, N = self.window_size, self.fwd_pred_next_n
        # Yield W+N frames per window: convert_action drops the last (-> W+N-1,
        # matching its assert) while convert_image consumes all W+N. Mirrors the
        # original RLDS window (window_size + future_action_window).
        span = W + N
        actions = traj["actions"]
        images = traj["images"]
        grip = traj["gripper_images"]
        T = len(actions)
        for start in range(T):                # sliding, stride 1; start always valid
            idx = list(range(start, start + span))
            valid = [i for i in idx if i < T]
            # right-pad by repeating the last valid frame; mask marks padding
            pad = span - len(valid)
            sel = valid + [valid[-1]] * pad
            mask = np.array([1] * len(valid) + [0] * pad, dtype=np.int64)
            yield {
                "task_description": traj["language"],
                "action": actions[sel],
                "episode_mask": mask,
                "images": images[sel],
                "gripper_images": grip[sel],
            }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rank, world = self._rank_world()
        flat = (w for traj in self.iter_trajectories() for w in self._windows(traj))
        # Round-robin shard across ranks (mirrors RLDS _RLDSDatasetByRank).
        # Shard sizes can differ by 1 across ranks; the trainer must use
        # drop_last=True to avoid a DDP all-reduce hang on the ragged last step.
        for item in itertools.islice(flat, rank, None, world):
            yield item

    @staticmethod
    def _rank_world():
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

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
