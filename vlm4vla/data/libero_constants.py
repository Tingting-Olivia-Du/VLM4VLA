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
