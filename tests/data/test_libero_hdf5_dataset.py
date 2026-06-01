from pathlib import Path

import numpy as np
from vlm4vla.data.libero_hdf5_dataset import LiberoHDF5Dataset

DATA_ROOT = "/workspace/tingting/LIBERO/libero/datasets"
STATS = str(Path(__file__).resolve().parents[2] / "vlm4vla/data/libero_stats/libero_10.json")


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


def test_iter_yields_windowed_5_fields():
    ds = _ds(window_size=1, fwd_pred_next_n=4)
    item = next(iter(ds))
    W, N = 1, 4
    expected_len = W + N  # 5 fields yielded at length W+N; convert_action drops one downstream
    assert isinstance(item["task_description"], str)
    assert item["action"].shape == (expected_len, 7)
    assert item["episode_mask"].shape == (expected_len,)
    assert set(np.unique(item["episode_mask"])).issubset({0, 1})
    assert item["images"].shape == (expected_len, 128, 128, 3)
    assert item["images"].dtype == np.uint8
    assert item["gripper_images"].shape == (expected_len, 128, 128, 3)

    # first window starts at trajectory index 0
    traj0 = next(_ds(window_size=1, fwd_pred_next_n=4).iter_trajectories())
    np.testing.assert_array_equal(item["action"][0], traj0["actions"][0])
