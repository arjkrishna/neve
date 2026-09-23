"""Per-step trajectory recorder for validation / replay episodes.

Records what the end-of-episode snapshot cannot recover afterwards: the two
device polylines at EVERY step (guidewire + catheter, resampled to N nodes by
arclength, float16), plus the scalars the STEP log already carries, plus the
planned path / target / on-path centerlines once per episode -- one ``.npz``
per episode. ~0.19 MB for a 233-step episode. Rendering is done offline by
``monitoring/traj/traj_render.py``; this module never renders.

Attach as a gym wrapper around the EVAL env (``TrajRecordWrapper(env)``); it
records only while ``TRAJ_RECORD_DIR`` is set in the environment, so an
unwrapped or unset run is untouched. Workers receive a ``deepcopy`` of the
wrapped env, so the wrapper keeps no open handles and writes files named by
seed + mesh fingerprint + pid + episode.

Layout written (see monitoring/traj/ANIM_PIPELINE.md):
    ${TRAJ_RECORD_DIR}/[${TRAJ_RECORD_BLOCK}/]s<seed>_<mesh_fp>_pid<pid>_ep<k>.npz

Never raises into the env: every failure is logged to stderr and skipped.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - the training image ships old gym
    import gym

from eve.util.coordtransform import tracking3d_to_vessel_cs

N_NODES = int(os.environ.get("TRAJ_RECORD_NODES", "64"))
PHYS_CODES = {"bridge": 0, "RCCA": 1, "RVA": 2, "LCCA": 3, "LVA": 4, "other": 5}


def _inner(env):
    """Walk .env / .unwrapped down to the object that owns `intervention`."""
    e = env
    for _ in range(8):
        if hasattr(e, "intervention"):
            return e
        nxt = getattr(e, "env", None) or getattr(e, "unwrapped", None)
        if nxt is None or nxt is e:
            break
        e = nxt
    return e


def resample_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample a (K,3) polyline to n nodes equally spaced in arclength.
    Keeps endpoints; a degenerate (<2 point) input is padded with its point."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or len(pts) == 0:
        return np.full((n, 3), np.nan, dtype=np.float32)
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0).astype(np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(pts[:1], n, axis=0).astype(np.float32)
    t = np.linspace(0.0, s[-1], n)
    out = np.empty((n, 3), dtype=np.float32)
    for d in range(3):
        out[:, d] = np.interp(t, s, pts[:, d])
    return out


class TrajRecordWrapper(gym.Wrapper):
    """Record device geometry + navigation scalars every step of every episode."""

    def __init__(self, env):
        super().__init__(env)
        self._rec: Optional[Dict[str, Any]] = None
        self._ep_index = 0

    # ------------------------------------------------------------ plumbing
    def get_config_dict(self):  # some callers serialise the env
        return self.env.get_config_dict()

    @property
    def _dir(self) -> Optional[str]:
        base = os.environ.get("TRAJ_RECORD_DIR")
        if not base:
            return None
        block = os.environ.get("TRAJ_RECORD_BLOCK", "")
        return os.path.join(base, block) if block else base

    # ------------------------------------------------------------ episodes
    def reset(self, *, seed=None, options=None):
        self._flush(cut=True)             # previous episode never finished
        out = self.env.reset(seed=seed, options=options)
        if self._dir:
            try:
                self._begin(seed, options)
                self._record_step(action=None, reward=0.0, terminated=False,
                                  truncated=False, info={})
            except Exception:
                sys.stderr.write("[traj_record] begin failed:\n%s\n" % traceback.format_exc())
                self._rec = None
        return out

    def step(self, action):
        out = self.env.step(action)
        if self._rec is not None:
            try:
                obs, reward, terminated, truncated, info = out
                self._record_step(action, reward, terminated, truncated, info)
                if terminated or truncated:
                    self._rec["reason"] = self._reason(terminated, truncated, info)
                    self._flush(cut=False)
            except Exception:
                sys.stderr.write("[traj_record] step failed:\n%s\n" % traceback.format_exc())
                self._rec = None
        return out

    def close(self):
        self._flush(cut=True)
        return self.env.close()

    # ------------------------------------------------------------ capture
    def _begin(self, seed, options):
        e = _inner(self.env)
        fl = e.intervention.fluoroscopy
        rot, ctr = fl.image_rot_zx, fl.image_center
        tree = e.intervention.vessel_tree
        rec: Dict[str, Any] = dict(
            seed=-1 if seed is None else int(seed),
            mesh_fp=str(getattr(tree, "mesh_fingerprint", "") or ""),
            anatomy_dir=str(getattr(tree, "anatomy_dir", "") or getattr(tree, "mesh_dir", "") or ""),
            target_branch=str(getattr(e, "_target_branch_short", "") or (options or {}).get("target_branch", "") or ""),
            pid=os.getpid(), ep=self._ep_index, wall_start=time.time(),
            rot=np.asarray(rot, dtype=np.float64), ctr=np.asarray(ctr, dtype=np.float64),
            path=None, path_len=float("nan"), target=None,
            cl_coords=[], cl_names=[], cl_onpath=[],
            reason="", steps=[],
        )
        self._ep_index += 1
        try:
            rec["path"] = np.asarray(e.pathfinder.path_points_vessel_cs, dtype=np.float32)
        except Exception:
            rec["path"] = None
        try:
            rec["path_len"] = float(getattr(e._path_context, "_total_length", float("nan")))
        except Exception:
            pass
        try:
            t3 = np.asarray(e.intervention.target.coordinates3d, dtype=np.float64)
            rec["target"] = np.asarray(tracking3d_to_vessel_cs(t3, rot, ctr), dtype=np.float32)
        except Exception:
            rec["target"] = None
        try:
            onpath = set()
            for br in getattr(e.pathfinder, "path_branch_set", []) or []:
                n = getattr(br, "name", None)
                if n: onpath.add(n)
            for br in tree.branches:
                c = np.asarray(br.coordinates, dtype=np.float32)
                if c.size == 0: continue
                rec["cl_coords"].append(c); rec["cl_names"].append(str(getattr(br, "name", "?")))
                rec["cl_onpath"].append(str(getattr(br, "name", "?")) in onpath)
        except Exception:
            pass
        self._rec = rec

    def _record_step(self, action, reward, terminated, truncated, info):
        e = _inner(self.env)
        fl = e.intervention.fluoroscopy
        rot, ctr = self._rec["rot"], self._rec["ctr"]
        polys = []
        try:
            for tr in fl.device_trackings3d or []:
                if tr is None or len(tr) == 0:
                    polys.append(np.full((N_NODES, 3), np.nan, np.float32)); continue
                v = tracking3d_to_vessel_cs(np.asarray(tr, dtype=np.float64), rot, ctr)
                polys.append(resample_polyline(v, N_NODES))
        except Exception:
            pass
        while len(polys) < 2:
            polys.append(np.full((N_NODES, 3), np.nan, np.float32))
        try:
            tip = np.asarray(tracking3d_to_vessel_cs(np.asarray(fl.tracking3d[0], dtype=np.float64), rot, ctr), dtype=np.float32)
        except Exception:
            tip = np.full(3, np.nan, np.float32)
        try:
            ins = [float(x) for x in e.intervention.device_lengths_inserted][:2]
        except Exception:
            ins = [float("nan"), float("nan")]
        try:
            pr = e._path_context.get_projection(); proj_s = float(pr.s); xt = float(pr.cross_track_dist)
        except Exception:
            proj_s, xt = float("nan"), float("nan")
        try:
            on_path = int(bool(e._path_context.is_on_correct_path()))
        except Exception:
            on_path = -1
        try:
            from eve.util.pathcontext import classify_physical_branch
            phys = PHYS_CODES.get(str(classify_physical_branch(e.intervention)), 5)
        except Exception:
            phys = -1
        try:
            d_tgt = float(np.linalg.norm(np.asarray(fl.tracking3d[0], dtype=np.float64)
                                         - np.asarray(e.intervention.target.coordinates3d, dtype=np.float64)))
        except Exception:
            d_tgt = float("nan")
        a = np.full(4, np.nan, np.float32)
        if action is not None:
            try:
                av = np.asarray(action, dtype=np.float32).flatten(); a[:min(4, len(av))] = av[:4]
            except Exception:
                pass
        self._rec["steps"].append(dict(
            gw=polys[0].astype(np.float16), cath=polys[1].astype(np.float16), tip=tip,
            ins_gw=ins[0], ins_cath=ins[1], proj_s=proj_s, xt=xt, on_path=on_path, phys=phys,
            fold=int(getattr(e, "_fold_stall_count", 0)), off_br=int(getattr(e, "_off_branch_steps", 0)),
            cath_slack=float(getattr(e, "_last_cath_slack_mm", float("nan"))),
            d_tgt=d_tgt, cmd=a, reward=float(reward), term=bool(terminated), trunc=bool(truncated),
            success=bool((info or {}).get("success", False)), t=time.time(),
        ))

    @staticmethod
    def _reason(terminated, truncated, info):
        r = (info or {}).get("reason") or (info or {}).get("termination_reason")
        if r: return str(r)
        if terminated and not truncated: return "success"
        if truncated: return "truncated"
        return "unknown"

    # ------------------------------------------------------------ write
    def _flush(self, cut: bool):
        rec, self._rec = self._rec, None
        if rec is None or not rec["steps"] or not self._dir:
            return
        try:
            d = self._dir; os.makedirs(d, exist_ok=True)
            S = rec["steps"]
            arr = lambda k, dt: np.asarray([s[k] for s in S], dtype=dt)
            cl = rec["cl_coords"]
            offs = np.cumsum([0] + [len(c) for c in cl]).astype(np.int64)
            np.savez_compressed(
                os.path.join(d, "s%s_%s_pid%d_ep%d.npz" % (
                    ("%04d" % rec["seed"]) if rec["seed"] >= 0 else "NA", rec["mesh_fp"] or "unknown", rec["pid"], rec["ep"])),
                gw=np.stack([s["gw"] for s in S]), cath=np.stack([s["cath"] for s in S]),
                tip=arr("tip", np.float32), ins_gw=arr("ins_gw", np.float32), ins_cath=arr("ins_cath", np.float32),
                proj_s=arr("proj_s", np.float32), xt=arr("xt", np.float32), on_path=arr("on_path", np.int8),
                phys=arr("phys", np.int8), fold=arr("fold", np.int16), off_br=arr("off_br", np.int16),
                cath_slack=arr("cath_slack", np.float32), d_tgt=arr("d_tgt", np.float32),
                cmd=np.stack([s["cmd"] for s in S]), reward=arr("reward", np.float32),
                term=arr("term", np.bool_), trunc=arr("trunc", np.bool_), success=arr("success", np.bool_),
                t=arr("t", np.float64),
                path=rec["path"] if rec["path"] is not None else np.zeros((0, 3), np.float32),
                path_len=np.float32(rec["path_len"]),
                target=rec["target"] if rec["target"] is not None else np.full(3, np.nan, np.float32),
                cl_coords=np.concatenate(cl) if cl else np.zeros((0, 3), np.float32), cl_offsets=offs,
                cl_names=np.asarray(rec["cl_names"]), cl_onpath=np.asarray(rec["cl_onpath"], dtype=np.bool_),
                meta=np.asarray([rec["seed"], rec["pid"], rec["ep"]], dtype=np.int64),
                mesh_fp=np.asarray(rec["mesh_fp"]), anatomy_dir=np.asarray(rec["anatomy_dir"]),
                target_branch=np.asarray(rec["target_branch"]), reason=np.asarray(rec["reason"] or ("cut" if cut else "")),
                wall_start=np.float64(rec["wall_start"]), n_nodes=np.int32(N_NODES),
                # tracking3d -> vessel-CS transform used above, so log-only episodes
                # (raw tip3d in the STEP log) can be mapped into this frame offline
                image_rot_zx=np.asarray(rec["rot"], dtype=np.float64),
                image_center=np.asarray(rec["ctr"], dtype=np.float64),
            )
        except Exception:
            sys.stderr.write("[traj_record] write failed:\n%s\n" % traceback.format_exc())
