"""v2 (tubes) vs v3 (tubes + real surfaces), set-wide, same centerlines."""
import glob, json, os, numpy as np
def load(root, rep):
    out = {}
    for f in glob.glob(os.path.join(root, "*", rep)):
        out[os.path.basename(os.path.dirname(f))] = json.load(open(f))
    return out
def col(S, path, default=np.nan):
    v = []
    for r in S.values():
        x = r
        for k in path:
            x = x.get(k, {}) if isinstance(x, dict) else {}
        v.append(x if isinstance(x, (int, float)) else default)
    return np.array(v, float)
for tag, r2, r3 in (("A TopBrain", "topbrain_data/anatomies_v2", "topbrain_data/anatomies_v3"), ("B carotid", "carotid_data/anatomies_v2", "carotid_data/anatomies_v3")):
    v2 = load(r2, "mesh_v2.json"); v3 = load(r3, "mesh_v3.json")
    print("%s   v2 n=%d  v3 n=%d" % (tag, len(v2), len(v3)))
    for lab, S in (("v2", v2), ("v3", v3)):
        nav = sum(1 for r in S.values() if r["obj"].get("navigable")); lm = col(S, ("obj", "lumen_min_mm")); df = col(S, ("obj", "median_deficit_mm"))
        comps = col(S, ("obj", "comps")); oe = col(S, ("obj", "open_edges"))
        print("   %s navigable %3d/%d | min lumen median %.2f p10 %.2f | deficit median %.2f | comps max %d | open edges median %d max %d" % (
            lab, nav, len(S), np.nanmedian(lm), np.nanpercentile(lm, 10), np.nanmedian(df), comps.max(), np.nanmedian(oe), oe.max()))
    if v3:
        for kind in ("tube", "siphon", "lower"):
            mm = col(v3, ("shape", kind, "max_over_misr", "median")); ar = col(v3, ("shape", kind, "area_r_over_misr", "median")); mx = col(v3, ("shape", kind, "max_over_min", "median"))
            if np.isfinite(mm).any():
                print("   v3 shape %-7s  max/MISR %.2f (p90 of medians %.2f) | area-r/MISR %.2f | max/min %.2f   [n=%d]" % (
                    kind, np.nanmedian(mm), np.nanpercentile(mm, 90), np.nanmedian(ar), np.nanmedian(mx), np.isfinite(mm).sum()))
        dt = col(v3, ("obj", "deficit_tube", "median")); dr = col(v3, ("obj", "deficit_real", "median"))
        print("   v3 deficit on tube sections %.2f, on real sections %.2f (median of medians)" % (np.nanmedian(dt), np.nanmedian(dr)))
