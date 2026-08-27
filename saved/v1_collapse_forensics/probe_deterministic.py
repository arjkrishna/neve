"""Deterministic-policy freeze detector (run INSIDE the container via
`docker exec -i <c> python3 - < probe_deterministic.py`).

Forwards fixed episode-start probe states through the NEWEST policy
snapshot of the v2 run and prints the deterministic action-mean stats.
This is the signal that would have caught the v1 freeze at eval2 (start-
state mean|tanh(mu0)| fell 0.255 -> 0.167 -> 0.089; healthy > 0.10).

Probe states: 512 episode-start obs extracted once from the v1 replay
buffer (obs are mesh-invariant, so v1 start states remain a valid probe
distribution for v2) and cached beside the v1 checkpoints.
"""
import glob
import os
import numpy as np
import torch

RESULTS = "/opt/eve_training/results/eve_paper/neurovascular/full/mesh_ben"
V1_BUFFER = (
    RESULTS + "/2026-07-12_042431_rcca_procedural_v1/checkpoints/replay_buffer.npz"
)
PROBE_CACHE = RESULTS + "/probe_start_states_v1.npz"
N_POLICY_OBS = 97  # policy consumes the first 97 of the 121 flat dims

# ---- probe states (cached) ----
if os.path.isfile(PROBE_CACHE):
    probes = np.load(PROBE_CACHE)["obs"]
else:
    with np.load(V1_BUFFER, allow_pickle=False) as d:
        terms = np.asarray(d["terminals"]).astype(bool).ravel()
        starts = np.where(terms)[0] + 1  # index AFTER each terminal
        starts = starts[starts < terms.shape[0]]
        rng = np.random.RandomState(1)
        pick = rng.choice(starts, size=min(512, len(starts)), replace=False)
        # obs_pairs is (n, 2, 121); [i, 0] = state
        obs = np.stack([d["obs_pairs"][i, 0] for i in np.sort(pick)])
    np.savez_compressed(PROBE_CACHE, obs=obs)
    probes = obs
x = torch.tensor(probes[:, :N_POLICY_OBS], dtype=torch.float32)

# ---- newest snapshot of the newest v2 run ----
snaps = sorted(
    glob.glob(RESULTS + "/2026-*_rcca_procedural_v2/diagnostics/policy_snapshots/policy_*.pt"),
    key=os.path.getmtime,
)
if not snaps:
    print("PROBE: no v2 policy snapshots yet (first eval cycle pending)")
    raise SystemExit(0)
snap = snaps[-1]
cp = torch.load(snap, map_location="cpu")
sd = cp["policy"] if isinstance(cp, dict) and "policy" in cp else cp
meta = ""
if isinstance(cp, dict):
    meta = f" update={cp.get('update_step','?')} explore={cp.get('explore_step','?')}"

# ---- manual forward: relu(in) -> relu(hidden) -> heads[0]=mu, [1]=log_std ----
keys = list(sd.keys())
lin_w = [k for k in keys if k.endswith(".weight")]


def get(k):
    return sd[k]


# identify layers by shape chain from input width
w_in = [k for k in lin_w if get(k).shape[1] == N_POLICY_OBS]
assert w_in, f"no input layer of width {N_POLICY_OBS}; keys={keys[:8]}"
h = torch.relu(x @ get(w_in[0]).T + sd[w_in[0].replace("weight", "bias")])
used = {w_in[0]}
# hidden layers: square-ish weights consuming h's width, excluding 4-wide heads
while True:
    nxt = [
        k for k in lin_w
        if k not in used and get(k).shape[1] == h.shape[1] and get(k).shape[0] > 8
    ]
    if not nxt:
        break
    k = nxt[0]
    h = torch.relu(h @ get(k).T + sd[k.replace("weight", "bias")])
    used.add(k)
heads = [
    k for k in lin_w
    if k not in used and get(k).shape[1] == h.shape[1] and get(k).shape[0] <= 8
]
heads.sort()  # head 0 = mu, 1 = log_std (module order == name order here)
mu = h @ get(heads[0]).T + sd[heads[0].replace("weight", "bias")]
raw_ls = h @ get(heads[1]).T + sd[heads[1].replace("weight", "bias")]
log_std = -2.0 + 0.5 * (0.0 - (-2.0)) * (torch.tanh(raw_ls) + 1)  # bounds (-2, 0)

a = torch.tanh(mu)
m0 = a[:, 0].abs().mean().item()
print(f"PROBE snapshot={os.path.basename(snap)}{meta}")
print(
    "PROBE start-state deterministic action: "
    f"mean|a0|={m0:.3f} ({m0*30:.1f}mm/s) "
    f"median_a0={a[:, 0].median().item():+.4f} "
    f"frac|a0|<0.05={float((a[:, 0].abs() < 0.05).float().mean()):.2f}"
)
print(
    "PROBE per-dim mean|a|: "
    + " ".join(f"d{i}={a[:, i].abs().mean().item():.3f}" for i in range(a.shape[1]))
)
ceil = float((log_std >= -0.05).float().mean())
floor = float((log_std <= -1.95).float().mean())
print(f"PROBE log_std rails: ceiling={ceil:.2f} floor={floor:.2f} (v1 was ceiling=1.00 all run)")
# Calibration on v1 ground truth: healthy peak 0.255; eval2 (-33%, quality
# still 13%) 0.167; fully-frozen eval3 0.086-0.089. So: OK >= 0.15,
# WATCH 0.10-0.15, FREEZE-ALERT < 0.10.
verdict = "OK" if m0 >= 0.15 else ("WATCH" if m0 >= 0.10 else "FREEZE-ALERT")
print(f"PROBE VERDICT: {verdict} (OK>=0.15, WATCH>=0.10; v1 frozen measured 0.086)")
