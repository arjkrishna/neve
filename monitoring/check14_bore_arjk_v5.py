"""CHECK 14 addendum 4 -- consequence check on the RECA fork, all 216.
RECA is the wrong-branch option; if declared r says it is open where the exact bore
is below the device envelope, obs 47/48/49 advertise a decision the sim cannot honour.
Device envelope: catheter r 0.35 mm + SOFA contactDistance 0.30 mm = 0.65 mm.
Guidewire envelope: 0.18 + 0.30 = 0.48 mm.
"""
import glob, json, os, sys
import numpy as np, pyvista as pv, vtk

sys.path.insert(0, "/opt/eve_training/eve"); sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches


def arclen(p):
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def densify(p, r, step=0.25):
    s = arclen(p)
    t = np.linspace(0.0, s[-1], max(int(np.ceil(s[-1] / step)) + 1, 2))
    return np.stack([np.interp(t, s, p[:, k]) for k in range(3)], axis=1), np.interp(t, s, r), t


def qs(a, d=3):
    a = np.asarray(a, float)
    return "  ".join("%7.*f" % (d, v) for v in np.percentile(a, [5, 25, 50, 75, 95]))


CATH, WIRE = 0.65, 0.48
rows = []
for d in sorted(glob.glob("/opt/eve_training/carotid/anatomies/*")):
    if not os.path.isdir(d):
        continue
    prov = json.load(open(os.path.join(d, "provenance.json")))
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
    f = vtk.vtkImplicitPolyDataDistance(); f.SetInput(m)
    brs = {str(b.name): b for b in load_branches(os.path.join(d, "Centrelines_comb"))}
    rec = {"name": os.path.basename(d), "eca_floor": prov["eca_floored_frac"],
           "eca_mm": prov["eca_mm"]}
    for want, tag in (("RECA", "e"), ("RCCA", "r")):
        k = [x for x in brs if want in x.upper()]
        if not k:
            continue
        b = brs[k[0]]
        p = np.asarray(b.coordinates, float); rr0 = np.asarray(b.radii, float)
        q, rr, t = densify(p, rr0)
        c = np.array([f.EvaluateFunction(pt) for pt in q])
        # skip the first 3 mm of the ECA: it shares lumen with the route at the fork
        keep = t >= 3.0 if want == "RECA" else np.ones_like(t, bool)
        cc, tt = c[keep], t[keep]
        rec[tag + "_len"] = float(t[-1])
        rec[tag + "_minb"] = float(cc.min()) if len(cc) else None
        rec[tag + "_fc"] = float(np.mean(cc < CATH)) if len(cc) else None
        rec[tag + "_fw"] = float(np.mean(cc < WIRE)) if len(cc) else None
        # deepest usable penetration before the bore first drops under the envelope
        if len(cc):
            bad = np.where(cc < CATH)[0]
            rec[tag + "_open"] = float(tt[bad[0]]) if len(bad) else float(tt[-1])
        rec[tag + "_decl_at_block"] = float(np.interp(rec[tag + "_open"], t, rr))
    rows.append(rec)

print("n = %d" % len(rows))
print()
print("=== RECA fork, past the first 3 mm (shared-lumen bifurcation excluded) ===")
print("   length mm              : %s" % qs([r["e_len"] for r in rows]))
print("   min exact bore mm      : %s" % qs([r["e_minb"] for r in rows]))
print("   frac below catheter    : %s" % qs([r["e_fc"] for r in rows]))
print("   frac below guidewire   : %s" % qs([r["e_fw"] for r in rows]))
fc = np.array([r["e_fc"] for r in rows])
mb = np.array([r["e_minb"] for r in rows])
print("   anatomies with ANY RECA point under the catheter envelope: %d / %d"
      % (int((fc > 0).sum()), len(rows)))
print("   anatomies with ANY RECA point under the guidewire envelope: %d / %d"
      % (int((np.array([r['e_fw'] for r in rows]) > 0).sum()), len(rows)))
print("   RECA bore negative (centerline outside its own wall): %d" % int((mb < 0).sum()))
op = np.array([r["e_open"] for r in rows])
dab = np.array([r["e_decl_at_block"] for r in rows])
print("   arclength at which RECA first drops under catheter envelope: %s" % qs(op))
print("   DECLARED radius at that same station                      : %s" % qs(dab))

print()
print("=== RCCA route, same envelope ===")
rfc = np.array([r["r_fc"] for r in rows])
rmb = np.array([r["r_minb"] for r in rows])
rop = np.array([r["r_open"] for r in rows])
rdab = np.array([r["r_decl_at_block"] for r in rows])
print("   min exact bore mm      : %s" % qs(rmb))
print("   frac below catheter    : %s" % qs(rfc))
print("   frac below guidewire   : %s" % qs([r["r_fw"] for r in rows]))
print("   anatomies with ANY RCCA point under the catheter envelope: %d / %d"
      % (int((rfc > 0).sum()), len(rows)))
print("   anatomies with ANY RCCA point under the guidewire envelope: %d / %d"
      % (int((np.array([r['r_fw'] for r in rows]) > 0).sum()), len(rows)))
print("   arclength of first sub-catheter station: %s" % qs(rop))
print("   DECLARED radius there                  : %s" % qs(rdab))
print("   of those, first block at s < 130 (lower): %d ; at s >= 130 (siphon): %d"
      % (int(((rfc > 0) & (rop < 130)).sum()), int(((rfc > 0) & (rop >= 130)).sum())))

print()
print("worst RECA (lowest min bore):")
for r in sorted(rows, key=lambda x: x["e_minb"])[:6]:
    print("   %-46s min bore %+.3f  len %.1f  eca_floored %.2f"
          % (r["name"], r["e_minb"], r["e_len"], r["eca_floor"]))
print("worst RCCA (lowest min bore):")
for r in sorted(rows, key=lambda x: x["r_minb"])[:6]:
    print("   %-46s min bore %+.3f  first block at s=%.1f (declared r %.2f there)"
          % (r["name"], r["r_minb"], r["r_open"], r["r_decl_at_block"]))
