import json, numpy as np
D = json.load(open(r"D:\Arjun\workspace\neve\monitoring\out_rf\refute_dilate.json"))
COH = [k for k in D if k.startswith("topcow")]
ZONES = [("CCA_s20_60", 20, 60), ("cervICA_s60_103", 60, 103.4),
         ("OVERWRITE_s103_136", 103.4, 136), ("GRAFT_s136+", 136, 1e9)]
def z(o, lo, hi):
    S = np.array(o["S"]); return (S >= lo) & (S < min(hi, S[-1] - 8))
print("=== A. SURFACE FIDELITY TO ITS OWN DECLARED (VMTK MISR) RADII ===")
print("tube_dev = signed dist of ideal-tube points (r=stated_r) to the surface; + = mesh DILATED, - = mesh DEFLATED")
print("%-22s %10s %8s %8s %8s %8s" % ("zone", "surface", "tubedev", "reff/r", "rvert/r", "nseg"))
for zn, lo, hi in ZONES:
    for t in ["HOST_COLLISION", "HOST_VISUAL"]:
        o = D[t]; m = z(o, lo, hi)
        if m.sum() < 3: continue
        R = np.array(o["R"]); td = np.array(o["tube_dev"]); rf = np.array(o["reff"]); rv = np.array(o["rvert"]); ns = np.array(o["nseg"], float)
        print("%-22s %10s %+8.3f %8.3f %8.3f %8.0f" % (zn, t.replace("HOST_", ""), np.nanmedian(td[m]), np.nanmedian(rf[m]/R[m]), np.nanmedian(rv[m]/R[m]), np.nanmedian(ns[m])))
    vals = {k: [] for k in ["td", "rf", "rv", "ns"]}
    for t in COH:
        o = D[t]; m = z(o, lo, hi)
        if m.sum() < 3: continue
        R = np.array(o["R"])
        vals["td"].append(np.nanmedian(np.array(o["tube_dev"])[m]))
        vals["rf"].append(np.nanmedian(np.array(o["reff"])[m]/R[m]))
        vals["rv"].append(np.nanmedian(np.array(o["rvert"])[m]/R[m]))
        vals["ns"].append(np.nanmedian(np.array(o["nseg"], float)[m]))
    print("%-22s %10s %+8.3f %8.3f %8.3f %8.0f   [n=%d, tubedev range %+.3f..%+.3f]" % (
        zn, "COHORT", np.median(vals["td"]), np.median(vals["rf"]), np.median(vals["rv"]),
        np.median(vals["ns"]), len(vals["td"]), min(vals["td"]), max(vals["td"])))
print()
print("=== B. ABSOLUTE CALIBRE, DIAMETER mm (2x) ===")
print("%-22s %26s %26s" % ("zone", "DECLARED MISR diam (med)", "MESHED diam 2*r_eff (med)"))
for zn, lo, hi in ZONES:
    row = []
    for t in ["HOST_COLLISION", "HOST_VISUAL"]:
        o = D[t]; m = z(o, lo, hi); R = np.array(o["R"]); rf = np.array(o["reff"])
        row.append((t.replace("HOST_",""), 2*np.median(R[m]), 2*np.nanmedian(rf[m])))
    cd, cm = [], []
    for t in COH:
        o = D[t]; m = z(o, lo, hi)
        if m.sum() < 3: continue
        cd.append(2*np.median(np.array(o["R"])[m])); cm.append(2*np.nanmedian(np.array(o["reff"])[m]))
    print(" %-21s" % zn)
    for nm, a, b in row: print("      %-16s decl %5.2f   mesh %5.2f" % (nm, a, b))
    print("      %-16s decl %5.2f [%5.2f-%5.2f]   mesh %5.2f [%5.2f-%5.2f]" % ("COHORT(med[rng])", np.median(cd), min(cd), max(cd), np.median(cm), min(cm), max(cm)))
print()
print("=== C. WHAT THE COHORT WOULD BE IF MESHED AS FAITHFULLY AS HOST_VISUAL ===")
for zn, lo, hi in ZONES:
    cm = []
    for t in COH:
        o = D[t]; m = z(o, lo, hi)
        if m.sum() < 3: continue
        cm.append(np.nanmedian(np.array(o["reff"])[m]/np.array(o["R"])[m]))
    o = D["HOST_VISUAL"]; m = z(o, lo, hi)
    hv = np.nanmedian(np.array(o["reff"])[m]/np.array(o["R"])[m])
    k = np.median(cm)/hv
    cd = np.median([2*np.median(np.array(D[t]["R"])[z(D[t], lo, hi)]) for t in COH if z(D[t], lo, hi).sum()>=3])
    hd = 2*np.median(np.array(o["R"])[m])
    print("%-22s cohort mesh factor %.3f vs host-visual %.3f -> k=%.3f ; corrected cohort diam %5.2f vs host-visual meshed diam %5.2f" % (zn, np.median(cm), hv, k, cd*hv, hd*hv))
