"""RL_IMPROV_16 — tests for the incremental replay-buffer save/load and
the resume budget math. Run inside the training container with the
worktree's eve_rl mounted:

    docker run --rm -i \
      -v <worktree>/eve_rl/eve_rl/replaybuffer/pervanillastep.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/pervanillastep.py \
      -v <worktree>/eve_rl/eve_rl/replaybuffer/pervanillashared.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/pervanillashared.py \
      -v <worktree>/tests/test_incremental_buffer.py:/tmp/t.py \
      eve-training-fixed python3 /tmp/t.py
"""
import os
import shutil
import tempfile

import numpy as np

from eve_rl.replaybuffer.pervanillastep import PERVanillaStep


class FakeEp:
    def __init__(self, n, seed, reached=False, obs_dim=16):
        rng = np.random.RandomState(seed)
        self.is_demo = False
        self.reached_target_daughter = reached
        self.episode_return = float(rng.uniform(-8, 5))
        self.flat_obs = [
            rng.uniform(-1, 1, obs_dim).astype(np.float32) for _ in range(n + 1)
        ]
        self.actions = [
            rng.uniform(-1, 1, 4).astype(np.float32) for _ in range(n)
        ]
        self.rewards = [np.float32(rng.uniform(-0.1, 0.1)) for _ in range(n)]
        self.terminals = [np.float32(0.0)] * (n - 1) + [np.float32(1.0)]

    def __len__(self):
        return len(self.actions)


def make_buf(capacity):
    return PERVanillaStep(
        capacity=capacity, batch_size=32, balanced_fraction=0.3,
        stuck_fraction=0.15, stuck_slack_index=5, stuck_contact_index=7,
        stuck_slack_thresh=0.5, stuck_contact_thresh=0.5,
    )


def assert_equivalent(a, b, n):
    """Byte-faithfulness: every live slot + all drift-state equal."""
    assert len(a.buffer) == len(b.buffer) == n, (len(a.buffer), len(b.buffer))
    for i in range(n):
        for j in range(4):
            np.testing.assert_array_equal(
                np.asarray(a.buffer[i][j]), np.asarray(b.buffer[i][j])
            )
    np.testing.assert_array_equal(a.tree.tree, b.tree.tree)
    np.testing.assert_array_equal(a.is_demo, b.is_demo)
    np.testing.assert_array_equal(a.is_clean, b.is_clean)
    np.testing.assert_array_equal(a.is_stuck, b.is_stuck)
    np.testing.assert_array_equal(a.episode_returns, b.episode_returns)
    assert a.position == b.position
    assert a.max_priority == b.max_priority
    assert a._sample_count == b._sample_count
    assert a._total_pushed == b._total_pushed
    if a.clean_tree is not None:
        np.testing.assert_allclose(a.clean_tree.tree, b.clean_tree.tree)
    if a.stuck_tree is not None:
        np.testing.assert_allclose(a.stuck_tree.tree, b.stuck_tree.tree)


def test_roundtrip_no_wrap():
    d = tempfile.mkdtemp()
    try:
        buf = make_buf(4096)
        for s in range(5):
            buf.push(FakeEp(100, seed=s, reached=(s % 2 == 0)))
        n1 = buf.save_incremental_to_dir(d)          # "eval 1"
        assert n1 == 500
        for s in range(5, 8):
            buf.push(FakeEp(100, seed=s))
        for _ in range(3):                            # priorities drift
            b = buf.sample()
            buf.update_priorities(b.indices.numpy(), np.random.rand(32) * 3)
        n2 = buf.save_incremental_to_dir(d)          # "eval 2"
        assert n2 == 300
        fresh = make_buf(4096)
        n = fresh.load_incremental_from_dir(d)
        assert n == 800
        assert_equivalent(buf, fresh, 800)
        # and a fresh save from the LOADED buffer adds nothing new
        assert fresh.save_incremental_to_dir(d) == 0
        print("test_roundtrip_no_wrap PASS")
    finally:
        shutil.rmtree(d)


def test_roundtrip_with_wrap():
    d = tempfile.mkdtemp()
    try:
        cap = 500
        buf = make_buf(cap)
        buf.push(FakeEp(300, seed=10, reached=True))
        buf.save_incremental_to_dir(d)
        buf.push(FakeEp(300, seed=11))                # wraps at 500
        buf.save_incremental_to_dir(d)
        buf.push(FakeEp(150, seed=12, reached=True))  # more wrap
        b = buf.sample()
        buf.update_priorities(b.indices.numpy(), np.random.rand(32) * 2)
        buf.save_incremental_to_dir(d)
        assert buf._total_pushed == 750 and len(buf.buffer) == cap
        fresh = make_buf(cap)
        n = fresh.load_incremental_from_dir(d)
        assert n == cap
        assert_equivalent(buf, fresh, cap)
        print("test_roundtrip_with_wrap PASS")
    finally:
        shutil.rmtree(d)


def test_span_exceeds_capacity_clamp():
    d = tempfile.mkdtemp()
    try:
        cap = 400
        buf = make_buf(cap)
        buf.push(FakeEp(300, seed=20))
        buf.push(FakeEp(300, seed=21))   # 600 unsaved > cap: oldest 200 gone
        n = buf.save_incremental_to_dir(d)
        assert n == cap, n               # clamped to what the ring holds
        fresh = make_buf(cap)
        assert fresh.load_incremental_from_dir(d) == cap
        assert_equivalent(buf, fresh, cap)
        print("test_span_exceeds_capacity_clamp PASS")
    finally:
        shutil.rmtree(d)


def test_legacy_roundtrip_and_counter_seed():
    d = tempfile.mkdtemp()
    try:
        buf = make_buf(4096)
        for s in range(4):
            buf.push(FakeEp(100, seed=30 + s, reached=True))
        data = buf.export_all()
        assert int(data["total_pushed"]) == 400
        legacy = {k: v for k, v in data.items() if k != "total_pushed"}
        fresh = make_buf(4096)
        fresh.import_all(legacy)                     # old-format file
        assert fresh._total_pushed == 400            # n < capacity path
        # incremental save after a legacy load bootstraps one full chunk
        n = fresh.save_incremental_to_dir(d)
        assert n == 400
        again = make_buf(4096)
        assert again.load_incremental_from_dir(d) == 400
        assert_equivalent(fresh, again, 400)
        print("test_legacy_roundtrip_and_counter_seed PASS")
    finally:
        shutil.rmtree(d)


def test_partial_dir_refuses():
    d = tempfile.mkdtemp()
    try:
        buf = make_buf(1000)
        buf.push(FakeEp(200, seed=40))
        buf.save_incremental_to_dir(d)
        buf.push(FakeEp(200, seed=41))
        buf.save_incremental_to_dir(d)
        # delete the FIRST chunk -> live slots 0..199 uncovered
        first = sorted(
            f for f in os.listdir(d) if f.startswith("chunk_")
        )[0]
        os.remove(os.path.join(d, first))
        fresh = make_buf(1000)
        try:
            fresh.load_incremental_from_dir(d)
            raise AssertionError("partial dir must raise")
        except ValueError as e:
            assert "incomplete" in str(e)
        print("test_partial_dir_refuses PASS")
    finally:
        shutil.rmtree(d)


def test_resume_budget_math():
    # runner re-bases: baseline = update - exploration*ratio; the first
    # cycle's budget = exploration*ratio - (update - baseline) = 0 backlog.
    for explore, update, ratio in [
        (770_000, 395_000, 0.5),   # v2-like (update includes 10k pretrain)
        (100_000, 60_000, 0.5),
        (500_000, 250_000, 0.5),
    ]:
        baseline = max(0, int(update - explore * ratio))
        budget = max(0.0, explore * ratio - (update - baseline))
        assert budget == 0.0, (explore, update, budget)
        # after 45k more explore steps, exactly 22.5k updates are earned
        budget2 = max(
            0.0, (explore + 45_000) * ratio - (update - baseline)
        )
        assert abs(budget2 - 45_000 * ratio) < 1e-6
    print("test_resume_budget_math PASS")


if __name__ == "__main__":
    test_roundtrip_no_wrap()
    test_roundtrip_with_wrap()
    test_span_exceeds_capacity_clamp()
    test_legacy_roundtrip_and_counter_seed()
    test_partial_dir_refuses()
    test_resume_budget_math()
    print("ALL INCREMENTAL-BUFFER TESTS PASS")
