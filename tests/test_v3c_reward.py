"""RL_IMPROV_18 v3c — machine-1 port tests for the reward pair
(tip-average progress + catheter-slack potential). Critical cases from the
machine-2 handoff §5, incl. the PUMP REGRESSION (§3.3): an on→off(retract
trailing)→on→re-advance closed cycle must net exactly 0. Numpy only."""
import os
import tempfile

import numpy as np

import eve.reward.arclengthprogress as alp
from eve.reward.arclengthprogress import ArcLengthProgress
from eve.util.polyline import point_at_inserted_length

import importlib.util as _ilu

_br_path = "/opt/eve_training/training_scripts/util/buckle_reward.py"
_spec = _ilu.spec_from_file_location("buckle_reward", _br_path)
_br = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_br)
cath_slack_potential = _br.cath_slack_potential

# identity CS transform for synthetic geometry
alp.tracking3d_to_vessel_cs = lambda p, r, c: np.asarray(p, dtype=float)


class FakeProj:
    def __init__(self, s, ct=0.0):
        self.s = s
        self.cross_track_dist = ct


class FakePathContext:
    """Scripted frontier projection + on/off-path flags."""

    def __init__(self, total=200.0):
        self.total_length = total
        self.s = 0.0
        self.on = True
        self.off_arc = 0.0

    def reset(self):
        pass

    def get_projection(self):
        return FakeProj(self.s)

    def is_on_correct_path(self):
        return self.on

    def get_off_path_arc_since_divergence(self):
        return self.off_arc

    def get_local_tolerance(self):
        return 100.0  # kill the lateral penalty in these tests


class FakeFluoro:
    def __init__(self):
        self.image_rot_zx = None
        self.image_center = None
        self.tracking3d = np.zeros((2, 3))


class FakeIntervention:
    """Straight wire along +x from insertion at origin."""

    def __init__(self):
        self.fluoroscopy = FakeFluoro()
        self.device_lengths_inserted = [0.0, 0.0]

    def set_state(self, ins_gw, ins_cath):
        self.device_lengths_inserted = [float(ins_gw), float(ins_cath)]
        lead = max(ins_gw, ins_cath)
        n = max(int(lead // 5) + 2, 2)
        xs = np.linspace(lead, 0.0, n)          # distal-first
        self.fluoroscopy.tracking3d = np.stack(
            [xs, np.zeros(n), np.zeros(n)], axis=1
        )


class FakePathfinder:
    def __init__(self):
        self.path_points_vessel_cs = np.array(
            [[0.0, 0, 0], [200.0, 0, 0]]
        )


def make_avg(w=0.5):
    interv = FakeIntervention()
    pc = FakePathContext()
    r = ArcLengthProgress(
        intervention=interv, pathfinder=FakePathfinder(),
        progress_factor=0.01, lateral_penalty_factor=0.001,
        path_context=pc, tip_mode="avg", avg_gw_weight=w,
    )
    return r, interv, pc


def step_to(r, interv, pc, ins_gw, ins_cath, frontier_s, on=True):
    interv.set_state(ins_gw, ins_cath)
    pc.s = frontier_s
    pc.on = on
    r.step()
    return r.reward


def test_cath_slack_potential_props():
    assert cath_slack_potential(0.0) == 0.0
    assert cath_slack_potential(15.0) == 0.0            # dead-band edge
    assert cath_slack_potential(1e6) == -1.0            # cap
    vals = [cath_slack_potential(x) for x in np.linspace(0, 300, 61)]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))  # monotone
    assert all(-1.0 <= v <= 0.0 for v in vals)
    # closed cycle nets 0 through the delta form
    seq = [0, 40, 120, 200, 120, 40, 0]
    total = sum(
        cath_slack_potential(b) - cath_slack_potential(a)
        for a, b in zip(seq, seq[1:])
    )
    assert abs(total) < 1e-12
    print("test_cath_slack_potential_props PASS")


def test_point_at_inserted_length():
    # distal-first straight wire, tip at x=100, insertion at 0
    poly = np.array([[100.0, 0, 0], [50.0, 0, 0], [0.0, 0, 0]])
    np.testing.assert_allclose(point_at_inserted_length(poly, 0.0), [0, 0, 0])
    np.testing.assert_allclose(point_at_inserted_length(poly, 30.0), [30, 0, 0])
    np.testing.assert_allclose(point_at_inserted_length(poly, 999.0), [100, 0, 0])
    # undeployed zero-length pile at the proximal end is ignored
    pile = np.array(
        [[60.0, 0, 0], [20.0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    )
    np.testing.assert_allclose(point_at_inserted_length(pile, 10.0), [10, 0, 0])
    assert point_at_inserted_length(np.zeros((3, 3)), 5.0) is None
    print("test_point_at_inserted_length PASS")


def test_avg_mode_half_pay_and_roundtrip():
    r, interv, pc = make_avg()
    interv.set_state(50.0, 50.0)
    pc.s = 50.0
    r.reset()
    # (a) cath-solo advance (gw parked at 50): frontier 50 -> 70 pays HALF
    rew = step_to(r, interv, pc, 50.0, 70.0, 70.0)
    assert abs(rew - 0.01 * 0.5 * 20.0) < 1e-9, rew
    # (b) trailing-gw advance (cath frontier parked): gw 50 -> 66 pays HALF
    rew = step_to(r, interv, pc, 66.0, 70.0, 70.0)
    assert abs(rew - 0.01 * 0.5 * 16.0) < 1e-9, rew
    # (c) full round trip back to the start nets EXACTLY zero
    total = rew + 0.01 * 0.5 * (20.0 + 16.0)  # forward legs so far
    total2 = 0.0
    total2 += step_to(r, interv, pc, 50.0, 70.0, 70.0)   # gw back to 50
    total2 += step_to(r, interv, pc, 50.0, 50.0, 50.0)   # cath back to 50
    forward = 0.01 * 0.5 * 20.0 + 0.01 * 0.5 * 16.0
    assert abs(forward + total2) < 1e-9, (forward, total2)
    print("test_avg_mode_half_pay_and_roundtrip PASS")


def test_frontier_mode_byte_identical_legacy():
    interv = FakeIntervention()
    pc = FakePathContext()
    r = ArcLengthProgress(
        intervention=interv, pathfinder=FakePathfinder(),
        progress_factor=0.01, lateral_penalty_factor=0.001,
        path_context=pc, tip_mode="frontier",
    )
    interv.set_state(50.0, 40.0)
    pc.s = 50.0
    r.reset()
    rew = step_to(r, interv, pc, 80.0, 40.0, 80.0)
    assert abs(rew - 0.01 * 30.0) < 1e-12   # full legacy frontier delta
    print("test_frontier_mode_byte_identical_legacy PASS")


def test_pump_regression_off_path_freeze():
    """§3.3 BLOCKER: on -> off(retract trailing) -> on -> re-advance must
    net exactly 0 across the closed cycle. A rebaseline during off-path
    would pay the re-advance leg without having charged the retract leg."""
    r, interv, pc = make_avg()
    pc.off_arc = 0.0            # scripted off-arc channel contributes 0
    interv.set_state(60.0, 80.0)
    pc.s = 80.0
    r.reset()
    total = 0.0
    # off-path: trailing gw retracts 60 -> 20 in two steps (frontier held)
    total += step_to(r, interv, pc, 40.0, 80.0, 80.0, on=False)
    total += step_to(r, interv, pc, 20.0, 80.0, 80.0, on=False)
    # rejoin on-path at the same geometry as the moment of divergence-
    # retraction end state: pays s_eff(rejoin) - s_eff(last on-path)
    total += step_to(r, interv, pc, 20.0, 80.0, 80.0, on=True)
    # re-advance trailing back to the original 60
    total += step_to(r, interv, pc, 60.0, 80.0, 80.0, on=True)
    assert abs(total) < 1e-9, f"pump cycle nets {total} (must be 0)"
    print("test_pump_regression_off_path_freeze PASS")


def test_cache_stamp_roundtrip():
    os.environ["EVE_RL_BUCKLE_COEF"] = "0.5"
    os.environ["EVE_RL_CATH_SLACK_COEF"] = "0.5"
    os.environ["EVE_RL_PROGRESS_TIP_MODE"] = "avg"
    os.environ["EVE_RL_AVG_GW_WEIGHT"] = "0.5"
    try:
        from eve_rl.util.experience_cache import (
            save_episodes_npz, cache_reward_version,
        )
        ep = (
            np.zeros((3, 4), np.float32),
            np.zeros((2, 4), np.float32),
            np.zeros(2, np.float32),
            np.array([0.0, 1.0], np.float32),
        )
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.npz")
            save_episodes_npz(p, [ep])
            v = cache_reward_version(p)
            assert v["buckle_coef"] == 0.5
            assert v["cath_slack_coef"] == 0.5
            assert v["progress_tip_mode"] == "avg"
            assert v["avg_gw_weight"] == 0.5
    finally:
        for k in ("EVE_RL_BUCKLE_COEF", "EVE_RL_CATH_SLACK_COEF",
                  "EVE_RL_PROGRESS_TIP_MODE", "EVE_RL_AVG_GW_WEIGHT"):
            os.environ.pop(k, None)
    print("test_cache_stamp_roundtrip PASS")


if __name__ == "__main__":
    test_cath_slack_potential_props()
    test_point_at_inserted_length()
    test_avg_mode_half_pay_and_roundtrip()
    test_frontier_mode_byte_identical_legacy()
    test_pump_regression_off_path_freeze()
    test_cache_stamp_roundtrip()
    print("ALL V3C REWARD TESTS PASS")
