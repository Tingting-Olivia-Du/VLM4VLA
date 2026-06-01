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
    # dim 1: value at q99 bound (1.0) should map to exactly 1.0;
    #         midpoint value (0.0) with q01=-1,q99=1 maps to 0.0.
    actions = np.array([[-2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [ 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    q01 = np.array([-2, -1, -1, -1, -1, -1, 0], dtype=np.float64)
    q99 = np.array([ 2,  1,  1,  1,  1,  1, 1], dtype=np.float64)
    mask = np.array([True, True, True, True, True, True, False])  # gripper unnormalized
    out = normalize_q99(actions, q01, q99, mask)
    # dim0: -2->-1, 2->1
    np.testing.assert_allclose(out[:, 0], [-1.0, 1.0])
    # dim1 exercises non-gripper path: q99 bound 1.0 -> 1.0; midpoint 0.0 -> 0.0
    np.testing.assert_allclose(out[0, 1], 1.0)   # at q99 bound
    np.testing.assert_allclose(out[1, 1], 0.0)   # midpoint (q01+q99)/2
    # gripper dim (masked) unchanged
    np.testing.assert_allclose(out[:, 6], [1.0, 0.0])


def test_normalize_q99_handles_zero_range_denominator():
    # dim 2 has q99 == q01 (zero range); implementation substitutes denom=1.0 and clips.
    # The output for that dim must be finite and clipped into [-1, 1].
    actions = np.array([[0.0, 0.5, 999.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.5, -999.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    q01 = np.array([0.0, -1.0, 0.5, -1.0, -1.0, -1.0, 0.0], dtype=np.float64)
    q99 = np.array([0.0,  1.0, 0.5,  1.0,  1.0,  1.0, 1.0], dtype=np.float64)
    mask = np.ones(7, dtype=bool)
    mask[6] = False  # gripper unnormalized
    # Must not raise and must not produce inf or nan
    out = normalize_q99(actions, q01, q99, mask)
    assert np.isfinite(out).all(), "normalize_q99 produced inf/nan for zero-range dim"
    # dim 2 output must be clipped into [-1, 1]
    assert np.all(out[:, 2] >= -1.0) and np.all(out[:, 2] <= 1.0)
