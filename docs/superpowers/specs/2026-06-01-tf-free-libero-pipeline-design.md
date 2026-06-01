# TF-Free LIBERO Data Pipeline for VLM4VLA Ablations

**Date:** 2026-06-01
**Status:** Approved design, pending spec review
**Branch:** `feat/tf-free-libero-pipeline`

## Problem

The VLA ablation training (`3DBENCH/configs/vla_ablation/e*.json`, launched via
`3DBENCH/scripts/run_vla_ablation.sh`) crashes. The visible error is a torchrun
`ChildFailedError`, which hides the real cause. Systematic debugging through the
crash chain revealed the root issue:

**The `vlmbench` conda env is Python 3.12, but VLM4VLA's data pipeline depends on
`tensorflow==2.15.0`, which does not support Python 3.12** (TF only ships py3.12
wheels from 2.16+). The env was created by `3DBENCH/scripts/setup_env.sh` for
3DBENCH (py3.12 + LeRobot/hf-libero), never for VLM4VLA's older OpenVLA/RLDS stack.

The data pipeline reaches `tensorflow` transitively:
`OpenVLADataset` → `RLDSDataset` → `from prismatic.vla.datasets.rlds...` →
`import dlimp` → `tensorflow`. Both `dlimp` (empty submodule dir) and a
py3.12-compatible `tensorflow==2.15.0` are unobtainable here.

The intended training data `/workspace/tingting/modified_libero_rlds` (TFDS/RLDS
format) does not exist locally. However, **the raw LIBERO data IS present** as
HDF5 at `/workspace/tingting/LIBERO/libero/datasets/<suite>/*.hdf5`.

### Constraint

The user requires the **VLM4VLA architecture** (its `act_head`/FCDecoder, window /
`fwd_pred_next_n` action chunking). 3DBENCH's plain-SFT `06_finetune_qwen.py` is
not an acceptable substitute.

## Goal

Make VLM4VLA's VLA ablation training run in the `vlmbench` (py3.12) env by
**replacing only the data pipeline** — swap the tensorflow/TFDS/dlimp/prismatic
RLDS source for a pure-torch dataset that reads LIBERO HDF5 directly — while
leaving the model, action head, and training logic byte-for-byte unchanged.

## Why a clean swap is possible

The data pipeline talks to the model through a narrow contract. In
`vlm4vla/data/openvla_action_prediction_dataset.py`, `OpenVLADataset.__iter__`
pulls an `rlds_batch` from the TF pipeline and feeds exactly **5 fields** into the
(TF-free) `batch_transform`:

```python
yield self.batch_transform(
    task_description = rlds_batch["task"]["language_instruction"].decode(),  # str
    action           = rlds_batch["action"],                                 # (W+N, 7) float
    episode_mask     = rlds_batch["chunk_mask"],                             # (W+N,) bool/int
    images           = rlds_batch["observation"]["image_primary"],          # (W+N, H, W, 3) uint8
    gripper_images   = None,                                                 # or (W+N, H, W, 3)
)
```

Everything from `batch_transform` downstream (tokenize → model → action head →
loss) is TF-free and **already loads successfully in the no-TF env** — verified:
in debugging, the program got past model construction and wandb init, crashing
only at the `import prismatic` line in the data layer. The TF dependency is
isolated entirely on the data side of this 5-field interface.

`batch_transform` (`base_action_prediction_dataset.py`) does its own chunking
internally (`convert_action`/`convert_image` with `fwd_pred_next_n`,
`window_size`). It expects each yielded item to be a **windowed trajectory slice**
of `window_size + fwd_pred_next_n` frames, not the whole episode. The second-stage
action normalization (`normalize_action(norm_min=-0.65, norm_max=0.65)`) lives
here and uses **config constants, not dataset statistics** — so it is reused
unchanged.

## What the original RLDS pipeline does to actions (must be reproduced)

Three transforms, all reproducible without TF:

1. **OXE `libero_dataset_transform`** (`prismatic/.../oxe/transforms.py`): keep
   `action[:, :6]` as-is; gripper dim `action[:, 6] = invert(clip(x, 0, 1))`
   (flip so +1=open, 0=close). ~5 lines.
2. **`no_noops` filter**: drop near-zero ("no-op") action frames.
3. **`BOUNDS_Q99` normalization** (`prismatic/.../rlds/utils/data_utils.py`): map
   each action dim from `[q01, q99]` → `[-1, 1]` and clip, **except the gripper
   dim** (kept as binary). q01/q99 are per-dataset statistics, computed at TFDS
   build time in the original. **Not bundled for LIBERO** → must be computed.

The downstream `convert_action` then applies the second-stage
`normalize_action(-0.65, 0.65, maintain_last=True)` on top — unchanged.

## Design

### Components (isolated, minimal footprint)

1. **`tools/compute_libero_stats.py`** (new, preprocessing script)
   - Scans `LIBERO/libero/datasets/<suite>/*.hdf5` for a given suite
     (e.g. `libero_10`), applies the gripper transform + no_noops filter, then
     computes per-dim `q01`/`q99` over all actions.
   - Writes a cache JSON, e.g. `vlm4vla/data/libero_stats/<suite>.json`
     `{"action": {"q01": [...7], "q99": [...7], "mask": [...7]}}` (gripper dim
     `mask=false` so Q99 skips it, matching `normalize_action_and_proprio`).
   - Run once before training; idempotent.

2. **`vlm4vla/data/libero_hdf5_dataset.py`** (new) — `LiberoHDF5Dataset`,
   a pure-torch `IterableDataset` replacing `RLDSDataset`. Per demo trajectory:
   1. Read `actions (T,7)`, `obs/agentview_rgb`, `obs/eye_in_hand_rgb`, language
      from HDF5 metadata: `json.loads(h["data"].attrs["problem_info"])
      ["language_instruction"]` (verified present, e.g. "turn on the stove and
      put the moka pot on it"). Each demo (`data/demo_*`) is one trajectory.
   2. Gripper transform (step 1 above).
   3. no_noops filter (step 2).
   4. Q99 normalize using the cached stats (step 3).
   5. Slide a window of `window_size + fwd_pred_next_n` over the trajectory
      (`window_sample` "sliding"/"range", `left_pad` per `organize_type`),
      building `chunk_mask`. Reproduces RLDS chunk semantics.
   6. Resize images to `image_size`.
   7. Yield the 5-field dict (or tuple matching the `batch_transform` call).
   - Rank/world-size sharding handled like `_RLDSDatasetByRank` (islice by rank).

3. **`LiberoActionPredictionDataset`** — a sibling of `OpenVLADataset` that mixes
   `ActionPredictionDataset` + `LiberoHDF5Dataset` and yields via the same
   `batch_transform(...)` 5-field call. Registered in `vlm4vla/data/__init__.py`
   so `getattr(vlm4vla.data, type)` resolves it.

4. **Config change**: in `3DBENCH/configs/vla_ablation/*.json`, set
   `train_dataset.type` / `val_dataset.type` →
   `LiberoActionPredictionDataset`, `data_root_dir` →
   `/workspace/tingting/LIBERO/libero/datasets`, keep `data_mix` as the suite
   (`libero_10`). Apply to e8 first, then all 20 after e2e passes.

### Untouched (explicitly out of scope)

- The model, `act_head`/FCDecoder, `batch_transform` and everything downstream.
- `main.py` training logic. (The already-added `vla_ablation` task_name branch
  and the `strategy: ddp` config change from debugging are kept — they are
  prerequisites, documented below.)

### Data flow

```
LIBERO HDF5 ──[compute_libero_stats.py]──► <suite>.json (q01/q99 cache)
     │                                              │
     ▼                                              ▼
LiberoHDF5Dataset: read → gripper xf → no_noops → Q99 → window+chunk → resize
     │
     ▼ (5 fields: task_description, action, episode_mask, images, gripper_images)
batch_transform (UNCHANGED) → tokenize → model → action head → loss
```

## Prerequisite fixes already applied during debugging (keep)

- Installed into `vlmbench`: `lightning 2.6.5`, `flash-attn 2.8.3` (prebuilt
  wheel), `openvla`/`prismatic` (editable, `--no-deps`). torch untouched (2.7.1).
- `main.py`: added `elif "vla_ablation" in task_name` branch (avoids
  `NotImplementedError`).
- `e8_head.json`: `trainer.strategy` `deepspeed_stage_2` → `ddp` (deepspeed not
  installed; main.py natively supports ddp). To be applied to all configs.

## Error handling

- Missing stats cache → clear error telling the user to run
  `compute_libero_stats.py` for that suite.
- Missing/empty HDF5 dir or suite → explicit `FileNotFoundError` with the path.
- Trajectory shorter than `window_size + fwd_pred_next_n` → skip with a counted
  warning (don't silently drop).
- Image shape/dtype assertions before yield.

## Testing

1. **Unit**: instantiate `LiberoHDF5Dataset` on `libero_10`, pull one item;
   assert the 5 fields' shapes/dtypes/ranges (action ∈ [-1,1] except gripper;
   images uint8 at `image_size`; language non-empty; window length =
   `window_size + fwd_pred_next_n`).
2. **Stats sanity**: assert computed q01 < q99 per dim; gripper dim masked out.
3. **End-to-end**: single-GPU run of `e8_head` reaches `trainer.fit`, produces a
   few training steps with finite loss and a wandb curve.
4. **(Optional) cross-check**: if the TF pipeline is ever runnable, compare action
   values for one shared demo to confirm gripper-xf + Q99 reproduction.

## Out of scope / future

- Multi-suite `data_mix` mixtures (only single-suite needed now).
- proprio/state inputs (VLM4VLA ablation configs don't consume them).
- Restoring the TF/RLDS path (intentionally bypassed).
