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
