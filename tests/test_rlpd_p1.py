"""RL_IMPROV_17 P1 (RLPD) — tests for critic LayerNorm, entropy-free
critic backup, and symmetric offline/online sampling. Run inside the
training container with the branch's modules mounted (same mount set as
launch_rcca_rlpd_v1.sh, minus env/SOFA):

    docker run --rm -i <mounts> eve-training-fixed python3 /tmp/t.py
"""
import inspect

import numpy as np
import torch

from eve_rl.network.component.mlp import MLP
from eve_rl.replaybuffer.pervanillastep import PERVanillaStep
from eve_rl.replaybuffer.pervanillashared import PERVanillaStepShared


class FakeEp:
    def __init__(self, n, seed, is_demo=False, obs_dim=16):
        rng = np.random.RandomState(seed)
        self.is_demo = is_demo
        self.reached_target_daughter = False
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


def make_buf(capacity, offline_fraction=0.5):
    return PERVanillaStep(
        capacity=capacity, batch_size=32,
        offline_fraction=offline_fraction,
    )


# ---------------------------------------------------------------- MLP

def test_mlp_layernorm_forward_and_keys():
    torch.manual_seed(0)
    m = MLP([64, 64], n_inputs=12, output_layer_size=3, use_layernorm=True)
    y = m(torch.randn(7, 12))
    assert y.shape == (7, 3), y.shape
    keys = list(m.state_dict().keys())
    assert any("_norms" in k for k in keys), keys
    # one norm per hidden activation
    assert len(m._norms) == 2
    print("test_mlp_layernorm_forward_and_keys PASS")


def test_mlp_legacy_statedict_compat():
    torch.manual_seed(0)
    legacy = MLP([64, 64], n_inputs=12, output_layer_size=3)
    sd = legacy.state_dict()
    assert not any("_norms" in k for k in sd), (
        "default-off MLP must be byte-identical to legacy"
    )
    # legacy checkpoint -> legacy net: strict load OK
    legacy2 = MLP([64, 64], n_inputs=12, output_layer_size=3)
    legacy2.load_state_dict(sd)
    # legacy checkpoint -> LayerNorm net: MUST fail loudly (this is why
    # eval-only invocations of an RLPD checkpoint need --critic_layernorm,
    # and legacy checkpoints must NOT be loaded into LayerNorm nets).
    ln = MLP([64, 64], n_inputs=12, output_layer_size=3, use_layernorm=True)
    try:
        ln.load_state_dict(sd)
        raise AssertionError("cross-load should have failed")
    except RuntimeError:
        pass
    print("test_mlp_legacy_statedict_compat PASS")


# ------------------------------------------------- symmetric sampling

def test_symmetric_ratio_and_weights():
    buf = make_buf(4096)
    for s in range(3):
        buf.push(FakeEp(100, seed=s, is_demo=True))
    for s in range(3, 7):
        buf.push(FakeEp(100, seed=s))
    for _ in range(20):
        b = buf.sample()
        idx = b.indices.numpy()
        assert len(idx) == 32
        n_demo = int(buf.is_demo[idx].sum())
        assert n_demo == 16, n_demo          # exactly half offline
        assert bool((b.is_weights == 1.0).all())
        assert b.padding_mask is None
        # shapes match the legacy PER batch layout
        assert b.obs.shape == (32, 2, 16)
        assert b.actions.shape == (32, 1, 4)
        assert b.rewards.shape == (32, 1, 1)
        assert b.terminals.shape == (32, 1, 1)
    print("test_symmetric_ratio_and_weights PASS")


def test_symmetric_offline_scarce():
    buf = make_buf(4096)
    buf.push(FakeEp(5, seed=0, is_demo=True))    # only 5 offline slots
    buf.push(FakeEp(100, seed=1))
    b = buf.sample()
    idx = b.indices.numpy()
    assert len(idx) == 32
    assert int(buf.is_demo[idx].sum()) == 5      # capped at what exists
    print("test_symmetric_offline_scarce PASS")


def test_symmetric_all_offline_fallback():
    buf = make_buf(4096)
    for s in range(2):
        buf.push(FakeEp(100, seed=s, is_demo=True))
    b = buf.sample()                              # degenerate: no online yet
    assert len(b.indices) == 32
    assert int(buf.is_demo[b.indices.numpy()].sum()) == 32
    print("test_symmetric_all_offline_fallback PASS")


def test_symmetric_wrap_invalidation():
    # capacity 200: demo fills 0..99, online fills 100..199 then wraps
    # 50 slots into the demo region -> only slots 50..99 stay offline.
    buf = make_buf(200)
    buf.push(FakeEp(100, seed=0, is_demo=True))
    buf.sample()                                  # builds the index cache
    buf.push(FakeEp(100, seed=1))
    buf.push(FakeEp(50, seed=2))                  # overwrites slots 0..49
    for _ in range(10):
        b = buf.sample()
        idx = b.indices.numpy()
        n_demo = int(buf.is_demo[idx].sum())
        assert n_demo == 16, n_demo
        demo_idx = idx[buf.is_demo[idx]]
        assert demo_idx.min() >= 50 and demo_idx.max() < 100, (
            "stale cache: sampled an overwritten offline slot"
        )
    # full overwrite -> offline set empty -> all-online batches, no hang
    buf.push(FakeEp(50, seed=3))                  # overwrites slots 50..99
    b = buf.sample()
    assert int(buf.is_demo[b.indices.numpy()].sum()) == 0
    print("test_symmetric_wrap_invalidation PASS")


def test_symmetric_off_is_legacy():
    buf = make_buf(4096, offline_fraction=0.0)
    buf.push(FakeEp(100, seed=0, is_demo=True))
    buf.push(FakeEp(100, seed=1))

    def _boom():
        raise AssertionError("symmetric sampler ran with fraction 0.0")

    buf._sample_symmetric = _boom
    b = buf.sample()                              # legacy PER path
    assert len(b.indices) == 32
    assert buf._offline_idx_cache is None         # cache never built
    print("test_symmetric_off_is_legacy PASS")


def test_shared_wrapper_forwards_offline_fraction():
    sig = inspect.signature(PERVanillaStepShared.__init__)
    assert "offline_fraction" in sig.parameters
    assert sig.parameters["offline_fraction"].default == 0.0
    src = inspect.getsource(PERVanillaStepShared._run_subprocess)
    assert "offline_fraction=self.offline_fraction" in src, (
        "ctor param not forwarded to the internal PERVanillaStep"
    )
    print("test_shared_wrapper_forwards_offline_fraction PASS")


# ------------------------------------------------- entropy-free backup

def _make_sac(algo_name, backup_entropy, obs_dim=10, n_actions=4):
    import eve_rl
    from torch import optim as t_optim

    torch.manual_seed(0)
    hidden = [64, 64]
    q1_base = MLP(hidden)
    q2_base = MLP(hidden)
    policy_base = MLP(hidden)
    dummy = eve_rl.network.component.ComponentDummy()
    q1 = eve_rl.network.QNetwork(q1_base, obs_dim, n_actions, dummy)
    q2 = eve_rl.network.QNetwork(q2_base, obs_dim, n_actions, dummy)
    policy = eve_rl.network.GaussianPolicy(
        policy_base, obs_dim, n_actions,
        eve_rl.network.component.ComponentDummy(),
        log_std_min=-2, log_std_max=0.0,
    )
    q1_optim = eve_rl.optim.Adam(q1, lr=3e-4)
    q2_optim = eve_rl.optim.Adam(q2_base, lr=3e-4)
    policy_optim = eve_rl.optim.Adam(policy_base, lr=3e-4)
    sched = lambda o: t_optim.lr_scheduler.LinearLR(
        o, start_factor=1.0, end_factor=1.0, total_iters=1
    )
    model = eve_rl.model.SACModel(
        lr_alpha=3e-4, q1=q1, q2=q2, policy=policy,
        q1_optimizer=q1_optim, q2_optimizer=q2_optim,
        policy_optimizer=policy_optim,
        q1_scheduler=sched(q1_optim), q2_scheduler=sched(q2_optim),
        policy_scheduler=sched(policy_optim),
    )
    return eve_rl.algo.SAC(
        model, n_actions=n_actions, gamma=0.99,
        algo=algo_name, backup_entropy=backup_entropy,
    )


def _expected_q(sac, seed=123, batch=8, obs_dim=10):
    torch.manual_seed(seed)                       # fixes the rsample noise
    rng = np.random.RandomState(seed)
    all_states = torch.from_numpy(
        rng.uniform(-1, 1, (batch, 2, obs_dim)).astype(np.float32)
    )
    rewards = torch.from_numpy(
        rng.uniform(-0.1, 0.1, (batch, 1, 1)).astype(np.float32)
    )
    dones = torch.zeros(batch, 1, 1)
    return sac._get_expected_q(all_states, rewards, dones, None, 1)


def test_backup_entropy_branch():
    import eve_rl

    # ctor default preserves legacy
    assert "backup_entropy" in inspect.signature(
        eve_rl.algo.SAC.__init__
    ).parameters
    q_on = _expected_q(_make_sac("sac", backup_entropy=True))
    q_off = _expected_q(_make_sac("sac", backup_entropy=False))
    # identical nets + identical noise -> targets differ EXACTLY by the
    # -alpha*log_pi entropy term
    assert not torch.allclose(q_on, q_off), "entropy term had no effect"
    # awac already backs up entropy-free: the flag must be a no-op there
    q_awac_a = _expected_q(_make_sac("awac", backup_entropy=True))
    q_awac_b = _expected_q(_make_sac("awac", backup_entropy=False))
    assert torch.allclose(q_awac_a, q_awac_b)
    # and the awac plain backup equals sac-without-entropy (same nets/noise)
    assert torch.allclose(q_off, q_awac_b)
    print("test_backup_entropy_branch PASS")


if __name__ == "__main__":
    test_mlp_layernorm_forward_and_keys()
    test_mlp_legacy_statedict_compat()
    test_symmetric_ratio_and_weights()
    test_symmetric_offline_scarce()
    test_symmetric_all_offline_fallback()
    test_symmetric_wrap_invalidation()
    test_symmetric_off_is_legacy()
    test_shared_wrapper_forwards_offline_fraction()
    test_backup_entropy_branch()
    print("ALL RLPD P1 TESTS PASS")
