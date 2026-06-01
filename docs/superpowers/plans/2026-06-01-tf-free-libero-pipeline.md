# TF-Free LIBERO Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace VLM4VLA's tensorflow/RLDS data source with a pure-torch reader of LIBERO HDF5 files, so VLA ablation training runs in the py3.12 `vlmbench` env, leaving the model/action-head/training logic unchanged.

**Architecture:** A preprocessing script computes per-suite action `q01`/`q99` stats into a JSON cache. A new `LiberoHDF5Dataset` (pure-torch `IterableDataset`) reads HDF5 trajectories, reproduces the OXE libero gripper transform + no_noops filter + Q99 normalization + sliding-window chunking, and yields windowed slices. A thin `LiberoActionPredictionDataset` adapter feeds the existing (unchanged) `batch_transform` the same 5 fields the old `OpenVLADataset` did. Configs point at the new dataset class and local HDF5 path.

**Tech Stack:** Python 3.12, PyTorch 2.7.1, h5py, numpy, Pillow. No tensorflow.

**Env note:** All commands run with the vlmbench interpreter. Set once:
```bash
export VP=/workspace/tingting/envs/vlmbench/bin/python
cd /workspace/tingting/VLM4VLA
```
Branch: `feat/tf-free-libero-pipeline` (already created).

---

## File Structure

- **Create** `vlm4vla/data/libero_constants.py` — pure functions: gripper transform, no_noops mask, Q99 normalize. No I/O. Independently testable.
- **Create** `tools/compute_libero_stats.py` — CLI: scan a suite's HDF5, write `vlm4vla/data/libero_stats/<suite>.json`. Uses `libero_constants`.
- **Create** `vlm4vla/data/libero_hdf5_dataset.py` — `LiberoHDF5Dataset(IterableDataset)`: read HDF5 → transform → window → yield 5-field dicts. Uses `libero_constants` + stats cache.
- **Create** `vlm4vla/data/libero_action_prediction_dataset.py` — `LiberoActionPredictionDataset(ActionPredictionDataset, LiberoHDF5Dataset)`: yields via `batch_transform`.
- **Modify** `vlm4vla/data/__init__.py` — register the new class.
- **Create** `tests/data/test_libero_constants.py`, `tests/data/test_libero_hdf5_dataset.py`.
- **Modify** `3DBENCH/configs/vla_ablation/*.json` — `type` + `data_root_dir`; `strategy: ddp`.

**Contract the dataset must satisfy** (from `base_action_prediction_dataset.py::__call__`):
Each yielded item is a window of `W = window_size` history frames + `N = fwd_pred_next_n` future frames (so `W+N` total, plus the +1 the train path drops in `convert_action`). Fields:
- `task_description`: `str`
- `action`: `np.ndarray (W+N+1, 7)` float (Q99-normalized; gripper dim left binary)
- `episode_mask`: `np.ndarray (W+N+1,)` of 0/1 (1 = real frame, 0 = padding)
- `images`: `np.ndarray (W+N+1, H, W, 3)` uint8 — `Image.fromarray`-able frames
- `gripper_images`: `np.ndarray (W+N+1, H, W, 3)` uint8 or `None`

(`convert_image` indexes `images[window_size-1]` and iterates all frames; `convert_action` does `action[:-1]` in train mode — hence the `+1`.)

---

### Task 1: Action constants module (pure functions)

**Files:**
- Create: `vlm4vla/data/libero_constants.py`
- Test: `tests/data/test_libero_constants.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_libero_constants.py
import numpy as np
from vlm4vla.data.libero_constants import (
    libero_gripper_transform, noop_mask, normalize_q99,
)


def test_gripper_transform_flips_and_clips():
    # gripper col raw in [-1,1]; clip to [0,1] then invert -> 1-x
    actions = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.0],
                        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0,  1.0],
                        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0,  0.4]], dtype=np.float64)
    out = libero_gripper_transform(actions)
    # first 6 dims untouched
    np.testing.assert_allclose(out[:, :6], actions[:, :6])
    # gripper: clip(-1,0,1)=0 -> invert 1.0 ; clip(1)=1 -> invert 0.0 ; 0.4 -> 0.6
    np.testing.assert_allclose(out[:, 6], [1.0, 0.0, 0.6])


def test_noop_mask_flags_near_zero_frames():
    actions = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],   # noop (pos+rot ~0)
                        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],   # real
                       dtype=np.float64)
    keep = noop_mask(actions, eps=1e-4)
    assert keep.tolist() == [False, True]


def test_normalize_q99_maps_to_unit_range_and_skips_masked_dim():
    actions = np.array([[-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [ 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    q01 = np.array([-2, -1, -1, -1, -1, -1, 0], dtype=np.float64)
    q99 = np.array([ 2,  1,  1,  1,  1,  1, 1], dtype=np.float64)
    mask = np.array([True, True, True, True, True, True, False])  # gripper unnormalized
    out = normalize_q99(actions, q01, q99, mask)
    # dim0: -2->-1, 2->1
    np.testing.assert_allclose(out[:, 0], [-1.0, 1.0])
    # gripper dim (masked) unchanged
    np.testing.assert_allclose(out[:, 6], [1.0, 0.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$VP -m pytest tests/data/test_libero_constants.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vlm4vla.data.libero_constants'`

- [ ] **Step 3: Write minimal implementation**

```python
# vlm4vla/data/libero_constants.py
"""Pure, TF-free reproductions of the OXE 'libero' action transforms.

Mirrors prismatic/.../oxe/transforms.py::libero_dataset_transform and
prismatic/.../rlds/utils/data_utils.py::normalize_action_and_proprio
(BOUNDS_Q99), so HDF5-sourced actions match the original RLDS pipeline.
"""
import numpy as np

GRIPPER_DIM = 6  # 7-dim action: [dx,dy,dz, drx,dry,drz, gripper]


def libero_gripper_transform(actions: np.ndarray) -> np.ndarray:
    """Keep dims 0..5; gripper = invert(clip(g, 0, 1)) so +1=open, 0=close."""
    actions = np.asarray(actions, dtype=np.float64).copy()
    g = np.clip(actions[:, GRIPPER_DIM], 0.0, 1.0)
    actions[:, GRIPPER_DIM] = 1.0 - g
    return actions


def noop_mask(actions: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """True for frames whose pos+rot (dims 0..5) are not all near zero."""
    motion = np.abs(actions[:, :GRIPPER_DIM]).sum(axis=1)
    return motion > eps


def normalize_q99(actions: np.ndarray, q01: np.ndarray, q99: np.ndarray,
                  mask: np.ndarray) -> np.ndarray:
    """Map [q01,q99] -> [-1,1] and clip, per dim; leave masked dims untouched."""
    actions = np.asarray(actions, dtype=np.float64).copy()
    q01 = np.asarray(q01, dtype=np.float64)
    q99 = np.asarray(q99, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    denom = np.where((q99 - q01) == 0, 1.0, q99 - q01)
    normed = 2.0 * (actions - q01) / denom - 1.0
    normed = np.clip(normed, -1.0, 1.0)
    out = np.where(mask[None, :], normed, actions)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$VP -m pytest tests/data/test_libero_constants.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add vlm4vla/data/libero_constants.py tests/data/test_libero_constants.py
git commit -m "feat(data): TF-free libero action transforms (gripper/noop/q99)"
```

---

### Task 2: Stats precompute script

**Files:**
- Create: `tools/compute_libero_stats.py`
- Test: `tests/data/test_compute_libero_stats.py`

- [ ] **Step 1: Write the failing test** (drives a testable `compute_stats_for_actions` core, separate from CLI/I/O)

```python
# tests/data/test_compute_libero_stats.py
import json
import numpy as np
from tools.compute_libero_stats import compute_stats_for_actions


def test_compute_stats_returns_q01_q99_and_gripper_mask():
    rng = np.random.RandomState(0)
    actions = rng.uniform(-1, 1, size=(1000, 7))
    stats = compute_stats_for_actions(actions)
    assert set(stats["action"].keys()) == {"q01", "q99", "mask"}
    q01 = np.array(stats["action"]["q01"])
    q99 = np.array(stats["action"]["q99"])
    assert (q01 < q99).all()
    # gripper dim (index 6) is masked False so Q99 skips it
    assert stats["action"]["mask"][6] is False
    assert all(stats["action"]["mask"][:6])
    # round-trips through JSON
    json.loads(json.dumps(stats))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$VP -m pytest tests/data/test_compute_libero_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.compute_libero_stats'`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/compute_libero_stats.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$VP -m pytest tests/data/test_compute_libero_stats.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Generate the real cache for libero_10 and sanity-check**

Run:
```bash
$VP tools/compute_libero_stats.py \
  --data-root /workspace/tingting/LIBERO/libero/datasets \
  --suite libero_10 \
  --out vlm4vla/data/libero_stats/libero_10.json
$VP -c "import json; s=json.load(open('vlm4vla/data/libero_stats/libero_10.json'))['action']; import numpy as np; assert (np.array(s['q01'])<np.array(s['q99'])).all(); print('OK', s['q01'][:3], s['q99'][:3])"
```
Expected: `Wrote vlm4vla/data/libero_stats/libero_10.json (...)` then `OK ...`

- [ ] **Step 6: Commit**

```bash
git add tools/compute_libero_stats.py tests/data/test_compute_libero_stats.py vlm4vla/data/libero_stats/libero_10.json
git commit -m "feat(data): libero q01/q99 stats precompute + libero_10 cache"
```

---

### Task 3: HDF5 trajectory reader (no windowing yet)

**Files:**
- Create: `vlm4vla/data/libero_hdf5_dataset.py`
- Test: `tests/data/test_libero_hdf5_dataset.py`

- [ ] **Step 1: Write the failing test** (reads one real demo, checks per-trajectory output before windowing)

```python
# tests/data/test_libero_hdf5_dataset.py
import numpy as np
from vlm4vla.data.libero_hdf5_dataset import LiberoHDF5Dataset

DATA_ROOT = "/workspace/tingting/LIBERO/libero/datasets"
STATS = "vlm4vla/data/libero_stats/libero_10.json"


def _ds(**kw):
    base = dict(data_root_dir=DATA_ROOT, data_mix="libero_10", image_size=224,
                window_size=1, fwd_pred_next_n=4, stats_path=STATS, train=True)
    base.update(kw)
    return LiberoHDF5Dataset(**base)


def test_iter_trajectories_yields_clean_fields():
    ds = _ds()
    traj = next(ds.iter_trajectories())
    assert isinstance(traj["language"], str) and len(traj["language"]) > 0
    a = traj["actions"]
    assert a.ndim == 2 and a.shape[1] == 7
    # normalized dims 0..5 within [-1,1]; gripper dim is 0/1-ish
    assert a[:, :6].min() >= -1.0001 and a[:, :6].max() <= 1.0001
    img = traj["images"]
    assert img.dtype == np.uint8 and img.shape[1:] == (128, 128, 3)
    grip = traj["gripper_images"]
    assert grip.shape == img.shape
    # actions and frames aligned in length
    assert len(a) == len(img) == len(grip)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$VP -m pytest tests/data/test_libero_hdf5_dataset.py::test_iter_trajectories_yields_clean_fields -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vlm4vla.data.libero_hdf5_dataset'`

- [ ] **Step 3: Write minimal implementation** (trajectory layer only; window layer added in Task 4)

```python
# vlm4vla/data/libero_hdf5_dataset.py
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
        s = json.load(open(stats_path))["action"]
        self.q01 = np.array(s["q01"], dtype=np.float64)
        self.q99 = np.array(s["q99"], dtype=np.float64)
        self.norm_mask = np.array(s["mask"], dtype=bool)

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$VP -m pytest tests/data/test_libero_hdf5_dataset.py::test_iter_trajectories_yields_clean_fields -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add vlm4vla/data/libero_hdf5_dataset.py tests/data/test_libero_hdf5_dataset.py
git commit -m "feat(data): LiberoHDF5Dataset trajectory reader (transform+q99)"
```

---

### Task 4: Sliding-window + `__iter__` yielding the 5 fields

**Files:**
- Modify: `vlm4vla/data/libero_hdf5_dataset.py`
- Test: `tests/data/test_libero_hdf5_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/data/test_libero_hdf5_dataset.py
def test_iter_yields_windowed_5_fields():
    ds = _ds(window_size=1, fwd_pred_next_n=4)
    item = next(iter(ds))
    W, N = 1, 4
    expected_len = W + N + 1  # batch_transform drops the last in train mode
    assert isinstance(item["task_description"], str)
    assert item["action"].shape == (expected_len, 7)
    assert item["episode_mask"].shape == (expected_len,)
    assert set(np.unique(item["episode_mask"])).issubset({0, 1})
    assert item["images"].shape == (expected_len, 128, 128, 3)
    assert item["images"].dtype == np.uint8
    assert item["gripper_images"].shape == (expected_len, 128, 128, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$VP -m pytest tests/data/test_libero_hdf5_dataset.py::test_iter_yields_windowed_5_fields -v`
Expected: FAIL with `TypeError`/`StopIteration` (no `__iter__` yet)

- [ ] **Step 3: Add the window layer + `__iter__`** (append these methods to `LiberoHDF5Dataset`)

```python
    # ---- window layer ----------------------------------------------------
    def _windows(self, traj: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        W, N = self.window_size, self.fwd_pred_next_n
        span = W + N + 1                      # +1: train path drops last action
        actions = traj["actions"]
        images = traj["images"]
        grip = traj["gripper_images"]
        T = len(actions)
        for start in range(0, T):             # sliding, stride 1
            idx = list(range(start, start + span))
            valid = [i for i in idx if i < T]
            if not valid:
                break
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
        for item in itertools.islice(flat, rank, None, world):
            yield item

    @staticmethod
    def _rank_world():
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$VP -m pytest tests/data/test_libero_hdf5_dataset.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add vlm4vla/data/libero_hdf5_dataset.py tests/data/test_libero_hdf5_dataset.py
git commit -m "feat(data): sliding-window __iter__ yielding 5 batch_transform fields"
```

---

### Task 5: `LiberoActionPredictionDataset` adapter + registry

**Files:**
- Create: `vlm4vla/data/libero_action_prediction_dataset.py`
- Modify: `vlm4vla/data/__init__.py`
- Test: `tests/data/test_libero_action_prediction_dataset.py`

- [ ] **Step 1: Write the failing test** (adapter resolves via registry and produces model-ready dict keys)

```python
# tests/data/test_libero_action_prediction_dataset.py
import vlm4vla.data as vdata


def test_registered_in_namespace():
    assert hasattr(vdata, "LiberoActionPredictionDataset")
    assert "LiberoActionPredictionDataset" in vdata.__all__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$VP -m pytest tests/data/test_libero_action_prediction_dataset.py -v`
Expected: FAIL with `AssertionError` (attr missing)

- [ ] **Step 3: Write the adapter** (mirrors `OpenVLADataset`'s pattern in `openvla_action_prediction_dataset.py`)

```python
# vlm4vla/data/libero_action_prediction_dataset.py
from typing import Any, Dict

from vlm4vla.data.base_action_prediction_dataset import ActionPredictionDataset
from vlm4vla.data.libero_hdf5_dataset import LiberoHDF5Dataset


class LiberoActionPredictionDataset(ActionPredictionDataset, LiberoHDF5Dataset):
    """TF-free LIBERO equivalent of OpenVLADataset.

    Reads HDF5 via LiberoHDF5Dataset and emits items through the unchanged
    ActionPredictionDataset.batch_transform (5-field contract).
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        ActionPredictionDataset.__init__(self, **kwargs)
        if self.organize_type == "interleave":
            kwargs["window_sample"] = "sliding"
            kwargs["left_pad"] = False
        elif self.organize_type == "segment":
            kwargs["window_sample"] = "range"
            kwargs["left_pad"] = True
        else:
            raise ValueError("organize type must be interleave or segment")
        LiberoHDF5Dataset.__init__(self, **kwargs)

    def __iter__(self) -> Dict[str, Any]:
        for item in LiberoHDF5Dataset.__iter__(self):
            yield self.batch_transform(
                task_description=item["task_description"],
                action=item["action"],
                episode_mask=item["episode_mask"],
                images=item["images"],
                gripper_images=item["gripper_images"],
            )
```

- [ ] **Step 4: Register in the data namespace**

Modify `vlm4vla/data/__init__.py` to:

```python
from .calvin_dataset import DiskCalvinDataset
from .openvla_action_prediction_dataset import OpenVLADataset
from .libero_action_prediction_dataset import LiberoActionPredictionDataset
from .real_dataset import RealDataset

__all__ = [
    "DiskCalvinDataset",
    "OpenVLADataset",
    "LiberoActionPredictionDataset",
    "RealDataset",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `$VP -m pytest tests/data/test_libero_action_prediction_dataset.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Commit**

```bash
git add vlm4vla/data/libero_action_prediction_dataset.py vlm4vla/data/__init__.py tests/data/test_libero_action_prediction_dataset.py
git commit -m "feat(data): LiberoActionPredictionDataset adapter + registry"
```

---

### Task 6: Point e8_head config at the new dataset

**Files:**
- Modify: `3DBENCH/configs/vla_ablation/e8_head.json`

- [ ] **Step 1: Apply config edits**

Run:
```bash
$VP - <<'PY'
import json
p = "/workspace/tingting/3DBENCH/configs/vla_ablation/e8_head.json"
c = json.load(open(p))
for key in ("train_dataset", "val_dataset"):
    c[key]["type"] = "LiberoActionPredictionDataset"
    c[key]["data_root_dir"] = "/workspace/tingting/LIBERO/libero/datasets"
    c[key]["data_mix"] = "libero_10"
c["trainer"]["strategy"] = "ddp"   # already set during debug; keep explicit
json.dump(c, open(p, "w"), indent=4, ensure_ascii=False)
print("type:", c["train_dataset"]["type"], "| mix:", c["train_dataset"]["data_mix"])
PY
```
Expected: `type: LiberoActionPredictionDataset | mix: libero_10`

- [ ] **Step 2: Commit**

```bash
git -C /workspace/tingting/3DBENCH add configs/vla_ablation/e8_head.json
git -C /workspace/tingting/3DBENCH commit -m "config(vla_ablation): point e8_head at LiberoActionPredictionDataset"
```

(Note: 3DBENCH is a separate git repo from VLM4VLA — commit there.)

---

### Task 7: End-to-end single-GPU smoke test of e8_head

**Files:** none (verification task)

- [ ] **Step 1: Run e8_head on one GPU until first training steps**

Run:
```bash
cd /workspace/tingting/VLM4VLA && \
CUDA_VISIBLE_DEVICES=0 PATH="/workspace/tingting/envs/vlmbench/bin:$PATH" \
WANDB_MODE=offline timeout 600 torchrun --nnodes 1 --node_rank 0 \
  --nproc_per_node 1 --master_addr 127.0.0.1 --master_port 6070 \
  main.py /workspace/tingting/3DBENCH/configs/vla_ablation/e8_head.json \
  --gpus 1 --num_nodes 1 2>&1 | grep -vE "Requirement already|nvidia-" | tail -60
```
Expected: program reaches `trainer.fit`, prints the trainable/frozen param summary, and logs a few training steps with a **finite loss** (no `ModuleNotFoundError`, no `NotImplementedError`, no tensorflow/prismatic import). `timeout` killing a *running* train loop is success.

- [ ] **Step 2: If a shape/contract error appears**

Use `superpowers:systematic-debugging`. Most likely spots: window length vs `convert_action`'s `action[:-1]` (Task 4 `span`), or image dtype for `Image.fromarray` (must be uint8 HWC). Fix in `libero_hdf5_dataset.py`, re-run Step 1.

- [ ] **Step 3: Record result**

Append a one-line outcome (pass + first loss value, or the failure) to the plan's "Execution log" below. No commit needed (verification only).

---

### Task 8: Roll out to all 20 ablation configs

**Files:**
- Modify: `3DBENCH/configs/vla_ablation/e*.json` (the other 19)

- [ ] **Step 1: Apply the same edits to all configs**

Run:
```bash
$VP - <<'PY'
import glob, json
for p in sorted(glob.glob("/workspace/tingting/3DBENCH/configs/vla_ablation/e*.json")):
    c = json.load(open(p))
    for key in ("train_dataset", "val_dataset"):
        if key in c:
            c[key]["type"] = "LiberoActionPredictionDataset"
            c[key]["data_root_dir"] = "/workspace/tingting/LIBERO/libero/datasets"
            c[key]["data_mix"] = "libero_10"
    c["trainer"]["strategy"] = "ddp"
    json.dump(c, open(p, "w"), indent=4, ensure_ascii=False)
    print("updated", p.split("/")[-1])
PY
```
Expected: 20 `updated e*.json` lines.

- [ ] **Step 2: Verify no config still references the old pipeline**

Run:
```bash
grep -l "OpenVLADataset\|deepspeed_stage_2\|modified_libero_rlds" \
  /workspace/tingting/3DBENCH/configs/vla_ablation/*.json || echo "CLEAN"
```
Expected: `CLEAN`

- [ ] **Step 3: Spot-check one full config (e0_full) e2e**

Run the Task 7 Step 1 command but with `e0_full.json` and `--master_port 6071`.
Expected: reaches training steps with finite loss.

- [ ] **Step 4: Commit**

```bash
git -C /workspace/tingting/3DBENCH add configs/vla_ablation/
git -C /workspace/tingting/3DBENCH commit -m "config(vla_ablation): roll TF-free LIBERO pipeline to all experiments"
```

---

## Execution log

(filled in during execution)

- Task 7:
- Task 8 Step 3:

---

## Self-Review

**Spec coverage:**
- Problem/root cause → context for plan (no task needed). ✓
- `compute_libero_stats.py` → Task 2. ✓
- `LiberoHDF5Dataset` (read/gripper/noop/q99/window/resize/yield) → Tasks 1,3,4. ✓
  - *Note:* spec mentioned resize to `image_size`; the existing `convert_image`/`image_fn` already resizes via the model image processor, and configs use native 128→processor. The dataset yields native 128² uint8 frames (verified contract); no separate resize needed. Documented in Task 3.
- `LiberoActionPredictionDataset` + registry → Task 5. ✓
- Config change (type/root/ddp), e8 first then all 20 → Tasks 6,8. ✓
- "Untouched" model/batch_transform/main.py → no task modifies them. ✓
- Error handling (missing stats/dir, short traj, image asserts) → in Task 3 impl + Task 4 masking. ✓
- Testing (unit, stats sanity, e2e) → Tasks 1–5 unit, Task 2 Step 5 sanity, Tasks 7–8 e2e. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type consistency:** `libero_gripper_transform`/`noop_mask`/`normalize_q99`/`GRIPPER_DIM` defined in Task 1 and used identically in Tasks 2–3. The yielded dict keys (`task_description/action/episode_mask/images/gripper_images`) match `batch_transform.__call__` and are consumed unchanged in Task 5. Trajectory-layer keys (`language/actions/images/gripper_images`) are internal to `iter_trajectories`/`_windows` and consistent across Tasks 3–4. ✓
