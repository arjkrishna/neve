"""Is the radius observation informative, or pinned by the 2.0 mm floor clamp?

obs47 = clip(stated_r, MIN_RADIUS_FLOOR_MM=2.0, 12.0) / 12.  A station whose true
radius is below the floor reports the floor, so the feature is constant there and
carries no information about narrowing.
"""
import glob, os, sys
import numpy as np
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches, DualDeviceNav

FLOOR, CEIL = 2.0, 12.0

def prof(cl_dir, label):
    br = [b for b in load_branches(cl_dir) if "RCCA" in str(b.name).upper()][0]
    c = np.asarray(br.coordinates, float); r = np.asarray(br.radii, float)
    s = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))))
    obs47 = np.clip(r, FLOOR, CEIL)/CEIL
    clamped = r < FLOOR
    dist = s >= 130.0
    return dict(label=label, n=len(r), s=s, r=r, obs47=obs47,
                clamp_all=100*clamped.mean(),
                clamp_dist=100*clamped[dist].mean() if dist.any() else float('nan'),
                uniq_dist=len(np.unique(np.round(obs47[dist],6))) if dist.any() else 0,
                r_dist_med=float(np.median(r[dist])) if dist.any() else float('nan'),
                obs47_std_dist=float(np.std(obs47[dist])) if dist.any() else float('nan'))

rows=[prof(os.path.join("/opt/eve_training/eve_bench/data/dualdevicenav","Centrelines_comb"),"HOST")]
for d in sorted(glob.glob("/opt/eve_training/results_topbrain/anatomies/*")):
    n=os.path.basename(d)
    if n in ("topcow_mr_013","topcow_mr_014","topcow_mr_015"): continue
    rows.append(prof(os.path.join(d,"Centrelines_comb"), n))

print(f"{'anatomy':16s} {'stations':>8} {'clamped%':>9} {'clamped% distal':>16} "
      f"{'distinct obs47 distal':>22} {'r_distal_med':>13} {'obs47 sd distal':>16}")
print("-"*112)
for r in rows:
    print(f"{r['label']:16s} {r['n']:8d} {r['clamp_all']:8.1f}% {r['clamp_dist']:15.1f}% "
          f"{r['uniq_dist']:22d} {r['r_dist_med']:13.3f} {r['obs47_std_dist']:16.5f}")

h=rows[0]; co=rows[1:]
print("\nHOST distal (s>=130mm) obs47 trace, every 8th station:")
d=h['s']>=130
print("  s(mm) :", " ".join(f"{v:6.0f}" for v in h['s'][d][::8]))
print("  true r:", " ".join(f"{v:6.2f}" for v in h['r'][d][::8]))
print("  obs47 :", " ".join(f"{v:6.4f}" for v in h['obs47'][d][::8]))
print(f"\ncohort clamped% distal: min {min(c['clamp_dist'] for c in co):.1f}  "
      f"median {np.median([c['clamp_dist'] for c in co]):.1f}  max {max(c['clamp_dist'] for c in co):.1f}")
print(f"cohort obs47 sd distal: min {min(c['obs47_std_dist'] for c in co):.5f}  "
      f"median {np.median([c['obs47_std_dist'] for c in co]):.5f}")
print(f"host   obs47 sd distal: {h['obs47_std_dist']:.5f}")
