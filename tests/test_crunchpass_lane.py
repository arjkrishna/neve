"""RL_IMPROV_18 v1c — tests for the success-conditioned crunchpass lane.
Run inside the training container with the branch's replaybuffer mounted
(same mount pattern as tests/test_incremental_buffer.py)."""
import os
import shutil
import tempfile

import numpy as np

from eve_rl.replaybuffer.pervanillastep import PERVanillaStep

LA0, LA2, R93 = 42, 44, 93
OBS = 125


class FakeEp:
    """Episode with controllable crunch-signature steps."""

    def __init__(self, n, seed, reached=False, crunch_steps=(), obs_dim=OBS):
        rng = np.random.RandomState(seed)
        self.is_demo = False
        self.reached_target_daughter = reached
        self.episode_return = float(rng.uniform(-8, 5))
        self.flat_obs = []
        for i in range(n + 1):
            o = rng.uniform(-0.1, 0.1, obs_dim).astype(np.float32)
            o[R93] = 0.4                      # wide lumen by default
            if i in crunch_steps:
                o[LA0], o[LA2], o[R93] = 0.6, -0.5, 0.170   # signature
            else:
                o[LA0], o[LA2] = 0.05, 0.05                  # disengaged
            self.flat_obs.append(o)
        self.actions = [rng.uniform(-1, 1, 4).astype(np.float32) for _ in range(n)]
        self.rewards = [np.float32(rng.uniform(-0.1, 0.1)) for _ in range(n)]
        self.terminals = [np.float32(0.0)] * (n - 1) + [np.float32(1.0)]

    def __len__(self):
        return len(self.actions)


def make_buf(capacity=4096, frac=0.25):
    return PERVanillaStep(
        capacity=capacity, batch_size=32,
        crunchpass_fraction=frac,
        crunchpass_la0_index=LA0, crunchpass_la2_index=LA2,
        crunchpass_radius_index=R93,
    )


def test_membership_success_conditioned():
    buf = make_buf()
    buf.push(FakeEp(50, seed=0, reached=True, crunch_steps=range(10, 20)))
    buf.push(FakeEp(50, seed=1, reached=False, crunch_steps=range(10, 20)))
    buf.push(FakeEp(50, seed=2, reached=True))       # success, no crunch
    flags = buf.is_crunchpass[:150]
    assert flags[10:20].all(), "crunch steps of a SUCCESS must be flagged"
    assert flags[:10].sum() == 0 and flags[20:50].sum() == 0
    assert flags[50:100].sum() == 0, "crunch steps of a FAILURE must NOT"
    assert flags[100:150].sum() == 0, "non-crunch success steps must NOT"
    assert buf.crunchpass_tree.total() > 0
    print("test_membership_success_conditioned PASS")


def test_sampling_fraction_and_lane_purity():
    buf = make_buf(frac=0.25)
    for s in range(4):
        buf.push(FakeEp(100, seed=s, reached=True, crunch_steps=range(30, 60)))
    for s in range(4, 8):
        buf.push(FakeEp(100, seed=s))
    for _ in range(10):
        b = buf.sample()
        idx = b.indices.numpy()
        n_lane = int(buf.is_crunchpass[idx].sum())
        # lane quota = round(0.25*32) = 8; general draws may add a few more
        assert n_lane >= 8, f"lane under-sampled: {n_lane}"
    print("test_sampling_fraction_and_lane_purity PASS")


def test_priority_maintenance():
    buf = make_buf()
    buf.push(FakeEp(100, seed=0, reached=True, crunch_steps=range(0, 50)))
    b = buf.sample()
    buf.update_priorities(b.indices.numpy(), np.random.rand(32) * 3)
    # lane tree leaves for flagged slots must track the main tree
    for i in range(0, 50):
        main = float(buf.tree.tree[i + buf.capacity - 1])
        lane = float(buf.crunchpass_tree.tree[i + buf.capacity - 1])
        assert abs(main - lane) < 1e-9, (i, main, lane)
    print("test_priority_maintenance PASS")


def test_incremental_roundtrip_preserves_lane():
    d = tempfile.mkdtemp()
    try:
        buf = make_buf()
        buf.push(FakeEp(80, seed=0, reached=True, crunch_steps=range(20, 40)))
        buf.push(FakeEp(80, seed=1))
        buf.save_incremental_to_dir(d)
        fresh = make_buf()
        n = fresh.load_incremental_from_dir(d)
        assert n == 160
        np.testing.assert_array_equal(
            buf.is_crunchpass[:160], fresh.is_crunchpass[:160]
        )
        np.testing.assert_allclose(
            buf.crunchpass_tree.tree, fresh.crunchpass_tree.tree
        )
        b = fresh.sample()
        assert int(fresh.is_crunchpass[b.indices.numpy()].sum()) >= 8
        print("test_incremental_roundtrip_preserves_lane PASS")
    finally:
        shutil.rmtree(d)


def test_default_off_legacy():
    buf = PERVanillaStep(capacity=1024, batch_size=32)
    buf.push(FakeEp(60, seed=0, reached=True, crunch_steps=range(0, 30)))
    assert buf.crunchpass_tree is None
    assert buf.is_crunchpass.sum() == 0
    b = buf.sample()
    assert len(b.indices) == 32
    print("test_default_off_legacy PASS")


def test_rail_filter_excludes_lane():
    os.environ["EVE_CLEAN_RAIL_MAX"] = "0.15"
    try:
        buf = make_buf()
        ep = FakeEp(60, seed=0, reached=True, crunch_steps=range(10, 30))
        ep.actions = [np.full(4, 0.99, dtype=np.float32) for _ in range(60)]
        buf.push(ep)                      # railed success -> reached flipped
        assert buf.is_crunchpass[:60].sum() == 0, (
            "rail-poisoned success must stay out of the crunchpass lane"
        )
        print("test_rail_filter_excludes_lane PASS")
    finally:
        del os.environ["EVE_CLEAN_RAIL_MAX"]


if __name__ == "__main__":
    test_membership_success_conditioned()
    test_sampling_fraction_and_lane_purity()
    test_priority_maintenance()
    test_incremental_roundtrip_preserves_lane()
    test_default_off_legacy()
    test_rail_filter_excludes_lane()
    print("ALL CRUNCHPASS LANE TESTS PASS")
