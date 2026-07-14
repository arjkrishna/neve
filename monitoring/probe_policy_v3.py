"""RL_IMPROV_16 E8 — deterministic-policy probe for v3a runs.

Run INSIDE the training container (CPU-only; safe alongside live training):

    docker exec -i rcca_procedural_v3a python3 - < monitoring/probe_policy_v3.py

Reports, degrading gracefully with data availability:
  1. FREEZE PROBE — start-state deterministic mean|a0| of the newest policy
     snapshot vs the run's OWN earliest (pretrain-era) snapshot. The v1
     freeze = regression toward the pretrain-BC attractor; healthy = ratio
     >= 2x and growing. (v2 reference: 0.032 -> 0.152 = 4.7x.)
  2. RETRACT-WHEN-STUCK PROBE (needs the run's replay_buffer.npz, saved at
     each eval) — deterministic mean a0 and P(retract) per gw_slack bin.
     THE E1/E3 gate: v2's coupling was flat/decaying (tail-bin bump +13pp
     -> +8pp); success = the slack-tail retract bump reappears + steepens.
  3. AUX R^2 (same buffer) — aux-head predictions vs true privileged
     labels. E2 gate: contact R^2 >= 0.55 and rising (v2 peaked 0.554 at
     u~130k then regressed); velocity labels should join after the E2
     repoint. Override label indices via env AUX_LABELS_FLAT.

Env overrides: RUN_GLOB (default 2026-*_rcca_procedural_v3a),
RESULTS_DIR, AUX_LABELS_FLAT (default "97,98,102,103" = the v3a repoint;
v2 used "99,100,102,103").
"""
import glob
import os
import numpy as np
import torch

RESULTS = os.environ.get(
    "RESULTS_DIR",
    "/opt/eve_training/results/eve_paper/neurovascular/full/mesh_ben",
)
RUN_GLOB = os.environ.get("RUN_GLOB", "2026-*_rcca_procedural_v3a")
AUX_FLAT = [
    int(x) for x in os.environ.get("AUX_LABELS_FLAT", "97,98,102,103").split(",")
]
N_POLICY_OBS = 97
SLACK_IDX, CONTACT_IDX = 89, 103
SLACK_BINS = [(-1.0, -0.085), (-0.085, 0.057), (0.057, 0.174), (0.174, 1.0)]

torch.set_num_threads(2)

runs = sorted(glob.glob(os.path.join(RESULTS, RUN_GLOB)))
runs = [r for r in runs if os.path.isdir(r)]
if not runs:
    print(f"PROBE: no run dir matching {RUN_GLOB}")
    raise SystemExit(0)
run = runs[-1]

# ---- snapshots (dedupe by update_step) ----
snaps = sorted(
    glob.glob(os.path.join(run, "diagnostics", "policy_snapshots", "policy_*.pt")),
    key=lambda p: int(p.split("_")[-1][:-3]),
)
uniq, seen = [], set()
for s in snaps:
    cp = torch.load(s, map_location="cpu")
    u = int(cp.get("update_step", -1))
    if u in seen:
        continue
    seen.add(u)
    uniq.append((u, int(cp.get("explore_step", -1)), cp["policy"], os.path.basename(s)))
if not uniq:
    print("PROBE: no policy snapshots yet (first cycle pending)")
    raise SystemExit(0)
uniq.sort(key=lambda t: t[0])


def forward_mu_aux(sd, x):
    lw = [k for k in sd if k.endswith(".weight")]
    w_in = [k for k in lw if sd[k].shape[1] == N_POLICY_OBS][0]
    h = torch.relu(x @ sd[w_in].T + sd[w_in.replace("weight", "bias")])
    used = {w_in}
    while True:
        nxt = [
            k for k in lw
            if k not in used and sd[k].shape[1] == h.shape[1] and sd[k].shape[0] > 8
        ]
        if not nxt:
            break
        k = nxt[0]
        h = torch.relu(h @ sd[k].T + sd[k.replace("weight", "bias")])
        used.add(k)
    heads = sorted(
        k for k in lw
        if k not in used and sd[k].shape[1] == h.shape[1] and sd[k].shape[0] <= 8
    )
    mu = h @ sd[heads[0]].T + sd[heads[0].replace("weight", "bias")]
    aux = None
    if len(heads) >= 3:
        aux = h @ sd[heads[2]].T + sd[heads[2].replace("weight", "bias")]
    return mu, aux


# ---- probe states: prefer the run's own buffer; else any cached probes ----
buf_path = os.path.join(run, "checkpoints", "replay_buffer.npz")
inc_dir = os.path.join(run, "checkpoints", "replay_incremental")
obs = acts = None


def _load_from_chunks(inc_dir, max_rows=200_000):
    """RL_IMPROV_16 — incremental-save fallback: concat the NEWEST chunks
    (highest monotonic start) until max_rows; enough for probe sampling."""
    chunks = sorted(
        glob.glob(os.path.join(inc_dir, "chunk_*.npz")), reverse=True
    )
    if not chunks:
        return None, None
    obs_l, term_l, rows = [], [], 0
    for cp in chunks:
        with np.load(cp, allow_pickle=False) as d:
            obs_l.append(np.asarray(d["obs_pairs"])[:, 0].astype(np.float32))
            term_l.append(np.asarray(d["terminals"]).ravel())
            rows += obs_l[-1].shape[0]
        if rows >= max_rows:
            break
    return np.concatenate(obs_l[::-1]), np.concatenate(term_l[::-1])


if os.path.isfile(buf_path):
    with np.load(buf_path, allow_pickle=False) as d:
        n = d["actions"].shape[0]
        rng = np.random.RandomState(1)
        pick = np.sort(rng.choice(n, size=min(16384, n), replace=False))
        obs = d["obs_pairs"][pick, 0].astype(np.float32)
        terms = np.asarray(d["terminals"]).astype(bool).ravel()
        starts = np.where(terms)[0] + 1
        starts = starts[starts < n]
        spick = np.sort(rng.choice(starts, size=min(512, len(starts)), replace=False))
        start_obs = d["obs_pairs"][spick, 0].astype(np.float32)
elif os.path.isfile(os.path.join(inc_dir, "replay_state.npz")):
    all_obs, all_terms = _load_from_chunks(inc_dir)
    if all_obs is not None:
        n = all_obs.shape[0]
        rng = np.random.RandomState(1)
        pick = np.sort(rng.choice(n, size=min(16384, n), replace=False))
        obs = all_obs[pick]
        terms = all_terms.astype(bool)
        starts = np.where(terms)[0] + 1
        starts = starts[starts < n]
        spick = np.sort(
            rng.choice(starts, size=min(512, len(starts)), replace=False)
        )
        start_obs = all_obs[spick]
        print(f"PROBE: sampled from incremental chunks (newest {n} rows)")
    else:
        start_obs = None
else:
    cached = glob.glob(os.path.join(RESULTS, "probe_start_states*.npz"))
    start_obs = np.load(cached[0])["obs"] if cached else None
    print("PROBE: run buffer not saved yet (pre-eval1);"
          + (" using cached start states" if cached else " start-state probe SKIPPED"))

# ---- 1. freeze probe: earliest vs latest snapshot on start states ----
if start_obs is not None:
    xs = torch.tensor(start_obs[:, :N_POLICY_OBS])
    (u0, e0, sd0, n0), (u1, e1, sd1, n1) = uniq[0], uniq[-1]
    a0_first = torch.tanh(forward_mu_aux(sd0, xs)[0])[:, 0].abs().mean().item()
    a0_last = torch.tanh(forward_mu_aux(sd1, xs)[0])[:, 0].abs().mean().item()
    ratio = a0_last / max(a0_first, 1e-6)
    verdict = "OK" if ratio >= 2.0 else ("WATCH" if ratio >= 1.25 else "FREEZE-ALERT")
    if len(uniq) == 1:
        verdict, ratio = "BASELINE-ONLY", 1.0
    print(f"PROBE freeze: baseline({n0},u={u0})={a0_first:.3f}  "
          f"latest({n1},u={u1})={a0_last:.3f}  ratio={ratio:.1f}x  VERDICT={verdict}"
          f"  (v2 healthy ref: 4.7x; freeze = ratio falling toward 1)")

# ---- 2 + 3 need the buffer ----
if obs is not None:
    x = torch.tensor(obs[:, :N_POLICY_OBS])
    labels = torch.tensor(obs[:, AUX_FLAT])
    for tag, (u, e, sd, name) in (
        [("early", uniq[0]), ("latest", uniq[-1])] if len(uniq) > 1
        else [("latest", uniq[-1])]
    ):
        mu, aux = forward_mu_aux(sd, x)
        a0 = torch.tanh(mu)[:, 0] * 30.0  # mm/s
        slack = obs[:, SLACK_IDX]
        rows = []
        for lo, hi in SLACK_BINS:
            m = (slack > lo) & (slack <= hi)
            if m.sum() < 20:
                rows.append(f"[{lo:+.2f},{hi:+.2f}] n<20")
                continue
            sel = a0[torch.tensor(m)]
            rows.append(
                f"[{lo:+.2f},{hi:+.2f}] meanA0={sel.mean():+.2f} "
                f"P(ret)={float((sel < -1.0).float().mean()):.3f}"
            )
        print(f"PROBE retract-vs-slack ({tag} u={u}): " + " | ".join(rows))
        print(f"  (E1/E3 gate: tail-bin P(ret) minus base-bin should be "
              f"positive and GROW; v2: +13pp->+8pp decaying)")
        if aux is not None:
            r2s = []
            for j in range(labels.shape[1]):
                y, p = labels[:, j], aux[:, j]
                vy = y.var().item()
                if vy < 1e-12:
                    r2s.append("dead")
                    continue
                r2 = 1.0 - ((y - p) ** 2).mean().item() / vy
                r2s.append(f"{r2:.3f}")
            print(f"PROBE aux R^2 ({tag} u={u}) labels{AUX_FLAT}: {r2s}"
                  f"  (E2 gate: contact >= 0.55 rising; velocities joining)")
    # stuck-lane composition check (E3)
    stuck = (obs[:, SLACK_IDX] > 0.174) | (obs[:, CONTACT_IDX] > 0.0026)
    print(f"PROBE buffer stuck-state share: {stuck.mean():.3f} "
          f"(lane draws 15% of batches from this pool when enabled)")
