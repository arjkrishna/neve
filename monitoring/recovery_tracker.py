#!/usr/bin/env python3
"""Soft/hard recovery-rate tracker (read-only). RL_IMPROV_16 E8-companion.

Parses per-worker step logs (diagnostics/logs_subprocesses/worker_*.log)
and reports, per 400-episode window (chronological across workers):
stuck-episode share, soft/hard/grind/unrecovered rates, and retract-depth
split by episode outcome. Run inside the training container:
    docker exec -i <container> python3 - < monitoring/recovery_tracker.py
"""
import glob
import os
import re

# ---- run location (edit GLOB for a different run) ----
BASE = "/opt/eve_training/results/eve_paper/neurovascular/full/mesh_ben"
GLOB = "2026-*_rcca_procedural_v3a"
# ---- detector thresholds (mm / steps; judge TRENDS, not absolutes) ----
STALL_EPS = 0.3     # progress gain below this counts as stalled
PUSH_MIN = 2.0      # commanded gw translation above this = "pushing"
STUCK_STEPS = 12    # consecutive stalled-while-pushing steps => stuck event
RETRACT_MIN = 1.0   # min gw retraction (mm) to count as a retract
SOFT_MAX = 8.0      # retraction <= this = SOFT recovery; above = HARD
PASS_EPS = 1.0      # must exceed pre-stuck max progress by this to recover
WINDOW = 400        # episodes per report window

runs = sorted(glob.glob(os.path.join(BASE, GLOB)))
assert runs, "no run dir matching " + GLOB
run = runs[-1]
logs = sorted(glob.glob(os.path.join(run, "diagnostics/logs_subprocesses/worker_*.log")))
assert logs, "no worker step logs in " + run

F = {k: re.compile(k + r"=([-0-9.a-z\[\],]+)") for k in
     ("ep", "ep_step", "wall_time", "proj_s", "term", "trunc")}
INS = re.compile(r"inserted=\[([-0-9.]+),([-0-9.]+)\]")
CMD = re.compile(r"cmd_action=\[([-0-9.]+),")

episodes = []  # (start_wall, success, events list, had_steps)

def close_episode(st):
    if not st["n"]:
        return
    # close any open stuck event as unrecovered
    if st["stuck"]:
        st["events"].append(("unrecovered", st["retract"]))
    episodes.append((st["t0"], st["success"], st["events"]))

for path in logs:
    st = None
    with open(path, errors="replace") as fh:
        for line in fh:
            if "EPISODE_START" in line:
                if st:
                    close_episode(st)
                st = {"n": 0, "t0": None, "success": False, "events": [],
                      "maxp": -1e9, "stall": 0, "stuck": False,
                      "gw_peak": 0.0, "gw_min": 0.0, "retract": 0.0, "p0": 0.0}
                continue
            if st is None or " STEP | " not in line:
                continue
            m_ins, m_cmd = INS.search(line), CMD.search(line)
            m_prog = F["proj_s"].search(line)
            if not (m_ins and m_cmd and m_prog):
                continue
            try:
                gw = float(m_ins.group(1))
                cmd0 = float(m_cmd.group(1))
                prog = float(m_prog.group(1))
            except ValueError:
                continue
            st["n"] += 1
            if st["t0"] is None:
                m_t = F["wall_time"].search(line)
                st["t0"] = float(m_t.group(1)) if m_t else 0.0
            if st["stuck"]:
                # track deepest retraction, look for pass-through
                st["gw_min"] = min(st["gw_min"], gw)
                st["retract"] = max(st["retract"], st["gw_peak"] - st["gw_min"])
                if prog > st["p0"] + PASS_EPS:
                    r = st["retract"]
                    kind = ("grind" if r < RETRACT_MIN
                            else "soft" if r <= SOFT_MAX else "hard")
                    st["events"].append((kind, r))
                    st["stuck"] = False
                    st["stall"] = 0
            else:
                stalled = (prog < st["maxp"] + STALL_EPS) and (cmd0 > PUSH_MIN)
                # decay instead of hard reset: tolerates brief command
                # sign-flips (noise) inside an otherwise-stuck push phase
                st["stall"] = st["stall"] + 1 if stalled else max(0, st["stall"] - 2)
                if st["stall"] >= STUCK_STEPS:
                    st["stuck"] = True
                    st["p0"] = st["maxp"]
                    st["gw_peak"] = gw
                    st["gw_min"] = gw
                    st["retract"] = 0.0
            st["maxp"] = max(st["maxp"], prog)
            if "term=True" in line and "trunc=False" in line:
                st["success"] = True
    if st:
        close_episode(st)

episodes.sort(key=lambda e: e[0])
n = len(episodes)
print("RECOVERY TRACKER: run=%s episodes=%d (window=%d)" %
      (os.path.basename(run), n, WINDOW))
hdr = ("window       eps  stuck-eps  events  soft%  hard%  grind%  "
       "unrec%  succ%  ret-depth succ/fail (mm)")
print(hdr)
for b in range(0, n, WINDOW):
    w = episodes[b:b + WINDOW]
    ev = [e for (_, _, evs) in w for e in evs]
    stuck_eps = sum(1 for (_, _, evs) in w if evs)
    succ = sum(1 for (_, s, _) in w if s)
    tot = max(1, len(ev))
    cnt = {k: sum(1 for (kk, _) in ev if kk == k)
           for k in ("soft", "hard", "grind", "unrecovered")}
    dep_s = [r for (t, s, evs) in w if s for (_, r) in evs]
    dep_f = [r for (t, s, evs) in w if not s for (_, r) in evs]
    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print("%5d-%-5d %5d  %6d     %5d  %5.1f  %5.1f  %6.1f  %6.1f  %5.1f"
          "   %.1f / %.1f" %
          (b + 1, b + len(w), len(w), stuck_eps, len(ev),
           100.0 * cnt["soft"] / tot, 100.0 * cnt["hard"] / tot,
           100.0 * cnt["grind"] / tot, 100.0 * cnt["unrecovered"] / tot,
           100.0 * succ / max(1, len(w)), avg(dep_s), avg(dep_f)))
print("GOAL: soft% RISING across windows and retract-depth succ < fail; "
      "v2's cycles faded ~86->58->28 across run thirds — that fade "
      "recurring (3+ windows monotone down + flat evals) = ALERT.")
