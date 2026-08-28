import os, json, sys
sys.path.insert(0,'/opt/eve_training/eve')
sys.path.insert(0,'/opt/eve_training/eve_bench')
import numpy as np
np.set_printoptions(suppress=True, precision=4)

from eve_bench.dualdevicenavtopbrain import find_anatomies, _CENTERLINE_SUBDIR, _RCCA_NAME
from eve_bench.dualdevicenav import DualDeviceNav, load_branches

FLOOR, CEIL, KR, MINTOL, MAXR = 2.0, 12.0, 1.5, 2.0, 12.0
LOOK = 20.0
ADIR = "/opt/eve_training/results_topbrain/anatomies"
EXCL = ["topcow_mr_013","topcow_mr_014","topcow_mr_015"]

def arclen(c):
    d = np.linalg.norm(np.diff(c, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])

def pick(branches, key):
    for b in branches:
        if key in str(b.name).upper() or b.name == key:
            return b
    return None

def bridge_of(branches):
    for b in branches:
        if "(11)" in str(getattr(b, "name", "")):
            return b
    return None

def r_at_s(s_arr, r_arr, q):
    q = np.clip(q, 0.0, s_arr[-1])
    idx = np.argmin(np.abs(s_arr[:, None] - q[None, :]), axis=0)
    return r_arr[idx]

def profile(coords, radii, s_off):
    coords = np.asarray(coords, float); radii = np.asarray(radii, float)
    s = arclen(coords)
    stot = s + s_off
    rc = np.clip(radii, FLOOR, CEIL)
    tol = np.maximum(MINTOL, KR * rc)
    o47 = rc / MAXR
    rah = np.clip(r_at_s(s, radii, s + LOOK), FLOOR, CEIL)
    o48 = rah / MAXR
    return dict(s=s, stot=stot, r=radii, rc=rc, tol=tol, o47=o47, o48=o48,
                route=float(s[-1] + s_off))

def box_of(branches):
    lows = np.min([b.low for b in branches], axis=0)
    highs = np.max([b.high for b in branches], axis=0)
    return lows, highs

out = {}

# ---------------- HOST ----------------
h = DualDeviceNav()
hvt = h.vessel_tree
hb = list(hvt.branches)
hrcca = pick(hb, "RCCA")
hbr = bridge_of(hb)
ostium = np.asarray(hrcca.coordinates[0], float)
def bridge_offset(br, ostium, k=2):
    c = np.asarray(br.coordinates, float)
    if np.linalg.norm(c[0]-ostium) < np.linalg.norm(c[-1]-ostium):
        c = c[::-1]
    k = int(min(max(1,k), len(c)-2))
    s = arclen(c)
    return float(s[-1] - s[k])
soff_host = bridge_offset(hbr, ostium) if hbr is not None else 0.0
H = profile(hrcca.coordinates, hrcca.radii, soff_host)
hlow, hhigh = box_of(hb)
out["host"] = dict(
    n=len(H["r"]), route=H["route"], soff=soff_host,
    box_low=hlow.tolist(), box_high=hhigh.tolist(),
    box_ext=(hhigh-hlow).tolist(),
    r_min=float(H["r"].min()), r_med=float(np.median(H["r"])), r_max=float(H["r"].max()),
    clamp_all=float((H["r"] < FLOOR).mean()),
    clamp_distal=float((H["r"][H["stot"] > 130] < FLOOR).mean()),
    o47=[float(H["o47"].min()), float(np.median(H["o47"])), float(H["o47"].max())],
    o47_distal=[float(H["o47"][H["stot"]>130].min()), float(np.median(H["o47"][H["stot"]>130])), float(H["o47"][H["stot"]>130].max())],
    o47_uniq_distal=int(len(np.unique(np.round(H["o47"][H["stot"]>130],6)))),
    o48=[float(H["o48"].min()), float(np.median(H["o48"])), float(H["o48"].max())],
    o48_distal=[float(H["o48"][H["stot"]>130].min()), float(np.median(H["o48"][H["stot"]>130])), float(H["o48"][H["stot"]>130].max())],
    tol=[float(H["tol"].min()), float(np.median(H["tol"])), float(H["tol"].max())],
    tol_distal=[float(H["tol"][H["stot"]>130].min()), float(np.median(H["tol"][H["stot"]>130])), float(H["tol"][H["stot"]>130].max())],
    frac_tol_at_floor=float((H["tol"] <= MINTOL+1e-9).mean()),
    branch_names=[str(b.name) for b in hb],
)
out["host"]["r_distal_raw"] = [float(H["r"][H["stot"]>130].min()), float(np.median(H["r"][H["stot"]>130])), float(H["r"][H["stot"]>130].max())]

# ---------------- COHORT ----------------
roots, names = find_anatomies(ADIR, exclude=EXCL)
coh = {}
for root, nm in zip(roots, names):
    bl = load_branches(os.path.join(root, _CENTERLINE_SUBDIR))
    rc = pick(bl, "RCCA")
    br = bridge_of(bl)
    ost = np.asarray(rc.coordinates[0], float)
    soff = bridge_offset(br, ost) if br is not None else 0.0
    P = profile(rc.coordinates, rc.radii, soff)
    lo, hi = box_of(bl)
    d = P["stot"] > 130
    coh[nm] = dict(
        n=len(P["r"]), route=P["route"], soff=soff,
        box_low=lo.tolist(), box_high=hi.tolist(), box_ext=(hi-lo).tolist(),
        r_min=float(P["r"].min()), r_med=float(np.median(P["r"])), r_max=float(P["r"].max()),
        r_distal=[float(P["r"][d].min()), float(np.median(P["r"][d])), float(P["r"][d].max())],
        clamp_all=float((P["r"] < FLOOR).mean()),
        clamp_distal=float((P["r"][d] < FLOOR).mean()),
        o47=[float(P["o47"].min()), float(np.median(P["o47"])), float(P["o47"].max())],
        o47_distal=[float(P["o47"][d].min()), float(np.median(P["o47"][d])), float(P["o47"][d].max())],
        o47_uniq_distal=int(len(np.unique(np.round(P["o47"][d],6)))),
        o48=[float(P["o48"].min()), float(np.median(P["o48"])), float(P["o48"].max())],
        o48_distal=[float(P["o48"][d].min()), float(np.median(P["o48"][d])), float(P["o48"][d].max())],
        tol=[float(P["tol"].min()), float(np.median(P["tol"])), float(P["tol"].max())],
        tol_distal=[float(P["tol"][d].min()), float(np.median(P["tol"][d])), float(P["tol"][d].max())],
        frac_tol_at_floor=float((P["tol"] <= MINTOL+1e-9).mean()),
    )
out["cohort"] = coh
print(json.dumps(out, indent=1))
