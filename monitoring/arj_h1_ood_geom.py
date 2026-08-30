#!/usr/bin/env python3
"""H1 / dissimilarity axis. Pure numpy, read-only, no eve import, no meshing.

Replicates RCCAVariedFromMesh._generate()'s RNG stream exactly (verified
against eve/eve/intervention/vesseltree/rccavariedfrommesh.py) to sample the
PROCEDURAL training family, then places the 22 TopBrain grafts in it.
"""
import json, glob, os, pickle
import numpy as np

ROOT = "D:/Arjun/workspace/neve"
HOSTC = os.path.join(ROOT, "eve_bench", "data", "dualdevicenav", "Centrelines_comb")
ANAT = os.path.join(ROOT, "topbrain_data", "anatomies")
RCCA_F = "Centerline curve - RCCA.mrk.json"
EXCL = {"topcow_mr_013", "topcow_mr_014", "topcow_mr_015"}
SEAM = 133.6
STEP = 1.0
STEN = 3          # +/-3 stations = 3 mm chord for curvature / bend


def read_curve(path):
    d = json.load(open(path, "r", encoding="utf-8"))
    m = d["markups"][0]
    pos = np.array([c["position"] for c in m["controlPoints"]], float)
    rad = None
    for me in m.get("measurements", []):
        if me.get("name") == "Radius" and "controlPointValues" in me:
            rad = np.array(me["controlPointValues"], float)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    return np.stack([y, -z, -x], axis=1), rad


def arclen(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def resample(p, r, step=STEP):
    s = arclen(p)
    n = max(int(round(s[-1] / step)) + 1, 4)
    t = np.linspace(0.0, s[-1], n)
    q = np.stack([np.interp(t, s, p[:, i]) for i in range(3)], axis=1)
    rr = np.interp(t, s, r) if r is not None else None
    return q, rr, t


def menger(p, k=STEN):
    a = p[:-2 * k]; b = p[k:-k]; c = p[2 * k:]
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    den = ab * bc * ca
    return np.where(den > 1e-12, 4.0 * area / np.maximum(den, 1e-12), 0.0)


def bend_deg(p, k=STEN):
    d1 = p[k:-k] - p[:-2 * k]
    d2 = p[2 * k:] - p[k:-k]
    d1 = d1 / np.maximum(np.linalg.norm(d1, axis=1, keepdims=True), 1e-9)
    d2 = d2 / np.maximum(np.linalg.norm(d2, axis=1, keepdims=True), 1e-9)
    return np.degrees(np.arccos(np.clip((d1 * d2).sum(1), -1, 1)))


def feats(p, r, tag, seam=SEAM):
    q, rr, t = resample(p, r)
    L = float(t[-1])
    g = t >= seam
    if g.sum() < 4 * STEN + 2:
        return None
    gq = q[g]; grr = rr[g] if rr is not None else None; gt = t[g]
    kap = menger(gq); ben = bend_deg(gq)
    chord = float(np.linalg.norm(gq[-1] - gq[0]))
    gL = float(gt[-1] - gt[0])
    f = dict(tag=tag, L=L, graft_len=gL, graft_chord=chord,
             tort=gL / max(chord, 1e-9),
             Rc_min=float(1.0 / max(kap.max(), 1e-9)),
             kap_p95=float(np.percentile(kap, 95)),
             kap_mean=float(kap.mean()),
             bend_max=float(ben.max()),
             turn_cum=float(ben.sum()),
             turn_per_mm=float(ben.sum() / max(gL, 1e-9)))
    if grr is not None:
        d = t >= (L - 40.0)
        kd = menger(q[d]); bd = bend_deg(q[d])
        f.update(r_min=float(grr.min()), r_med=float(np.median(grr)),
                 r_mean=float(grr.mean()),
                 r_min_d40=float(rr[d].min()),
                 r_med_d40=float(np.median(rr[d])),
                 Rc_min_d40=float(1.0 / max(kd.max(), 1e-9)),
                 bend_max_d40=float(bd.max()),
                 turn_cum_d40=float(bd.sum()))
    f["_q"] = q; f["_t"] = t; f["_r"] = rr
    return f


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def perp_basis(tan):
    t = tan / max(np.linalg.norm(tan), 1e-9)
    ref = np.array([0., 0., 1.]) if abs(t[2]) < 0.9 else np.array([1., 0., 0.])
    n1 = np.cross(t, ref); n1 = n1 / max(np.linalg.norm(n1), 1e-9)
    n2 = np.cross(t, n1); n2 = n2 / max(np.linalg.norm(n2), 1e-9)
    return n1, n2


def perturb(coords, radii, rng, tort, rscale, anchor_mm, ramp_mm,
            distal_anchor_mm, base_amp_mm, freqs=(0.7, 1.3, 2.1)):
    coords = coords.astype(np.float64).copy()
    radii = radii.astype(np.float64).copy()
    n = len(coords)
    seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    w_up = smoothstep((s - anchor_mm) / max(ramp_mm, 1e-6))
    w_dn = smoothstep(((total - s) - distal_anchor_mm) / max(ramp_mm, 1e-6))
    w = np.minimum(w_up, w_dn)
    n1, n2 = perp_basis(coords[-1] - coords[0])
    u = s / max(total, 1e-6)
    amp = base_amp_mm * float(tort)
    d1 = np.zeros(n); d2 = np.zeros(n)
    for f in freqs:
        a1 = rng.normal(0., 1.); a2 = rng.normal(0., 1.)
        p1 = rng.uniform(0., 2 * np.pi); p2 = rng.uniform(0., 2 * np.pi)
        d1 = d1 + a1 * np.sin(2 * np.pi * f * u + p1)
        d2 = d2 + a2 * np.sin(2 * np.pi * f * u + p2)
    norm = max(len(freqs) ** 0.5, 1.0)
    off = (w * amp / norm)[:, None] * (d1[:, None] * n1[None, :] + d2[:, None] * n2[None, :])
    nc = (coords + off).astype(np.float32).astype(np.float64)
    nr = (radii * (1.0 + (float(rscale) - 1.0) * w)).astype(np.float32).astype(np.float64)
    return nc, nr, w


def sample_procedural(hp, hr, rva_p, rva_r, seeds, gens):
    out = []
    rva_len = float(arclen(rva_p)[-1])
    hs = arclen(hp)
    for sd in seeds:
        rng = np.random.default_rng(int(sd))
        for g in range(gens):
            tort = float(np.clip(rng.normal(1.0, 0.3), 0.4, 1.6))
            rsc = float(max(0.75, rng.normal(1.0, 0.07)))
            c, r, w = perturb(hp, hr, rng, tort, rsc, anchor_mm=3.0, ramp_mm=15.0,
                              distal_anchor_mm=25.0, base_amp_mm=4.0)
            perturb(rva_p, rva_r, rng, 1.0, 1.0, anchor_mm=3.0, ramp_mm=12.0,
                    distal_anchor_mm=max(5.0, rva_len - 35.0), base_amp_mm=3.0)
            f = feats(c, r, "proc_s%dg%d" % (sd, g))
            f["tort_param"] = tort
            f["rscale_param"] = rsc
            f["w_at_seam"] = float(np.interp(SEAM, hs, w))
            out.append(f)
    return out


KEYS = ["graft_len", "tort", "Rc_min", "kap_p95", "kap_mean", "bend_max",
        "turn_cum", "turn_per_mm", "r_min", "r_med", "Rc_min_d40",
        "bend_max_d40", "r_min_d40"]


def main():
    hp, hr = read_curve(os.path.join(HOSTC, RCCA_F))
    rvaf = [f for f in glob.glob(os.path.join(HOSTC, "*.mrk.json"))
            if "RVA" in os.path.basename(f)]
    rp, rr_ = read_curve(rvaf[0])
    host = feats(hp, hr, "HOST")

    seeds = list(range(12345, 12361))
    GENS = 13
    proc = sample_procedural(hp, hr, rp, rr_, seeds, GENS)
    print("procedural draws: %d (seeds %d..%d x %d gens)" % (len(proc), seeds[0], seeds[-1], GENS))

    coh = []
    for d in sorted(glob.glob(os.path.join(ANAT, "topcow_mr_*"))):
        nm = os.path.basename(d)
        if nm in EXCL:
            continue
        p, r = read_curve(os.path.join(d, "Centrelines_comb", RCCA_F))
        coh.append(feats(p, r, nm))
    print("cohort: %d" % len(coh))

    pickle.dump({"host": host, "proc": proc, "coh": coh, "hp": hp, "hr": hr},
                open(os.path.join(ROOT, "monitoring", "_h1_geom.pkl"), "wb"))

    w_prof = perturb(hp, hr, np.random.default_rng(0), 1.0, 1.0, 3.0, 15.0, 25.0, 4.0)[2]
    hs = arclen(hp)
    print()
    print("perturbation envelope w(s) on the HOST RCCA (L=%.1f):" % hs[-1])
    for ss in [100, 120, 133.6, 150, 170, 180, 190, 197.5, 205, 212.5, 220, 230, 237]:
        print("   s=%6.1f  w=%.3f" % (ss, np.interp(ss, hs, w_prof)))

    print()
    print("PROCEDURAL ENVELOPE (graft region s>=%.1f of the perturbed HOST RCCA)" % SEAM)
    print("%-13s %9s %9s %9s %9s %9s" % ("feature", "host", "proc_mean", "proc_sd", "proc_min", "proc_max"))
    for k in KEYS:
        v = np.array([f[k] for f in proc])
        print("%-13s %9.3f %9.3f %9.3f %9.3f %9.3f" % (k, host[k], v.mean(), v.std(ddof=1), v.min(), v.max()))

    mu = {k: float(np.mean([f[k] for f in proc])) for k in KEYS}
    sd = {k: float(np.std([f[k] for f in proc], ddof=1)) for k in KEYS}
    print()
    print("COHORT z-scores vs PROCEDURAL family")
    print(("%-8s" + " %8s" * len(KEYS)) % tuple(["anat"] + [k[:8] for k in KEYS]))
    for f in coh:
        print(("%-8s" + " %8.2f" * len(KEYS)) %
              tuple([f["tag"][-6:]] + [(f[k] - mu[k]) / sd[k] for k in KEYS]))
    print(("%-8s" + " %8.2f" * len(KEYS)) %
          tuple(["HOST"] + [(host[k] - mu[k]) / sd[k] for k in KEYS]))
    print()
    print("RAW cohort values")
    print(("%-8s" + " %8s" * len(KEYS)) % tuple(["anat"] + [k[:8] for k in KEYS]))
    for f in coh:
        print(("%-8s" + " %8.3f" * len(KEYS)) % tuple([f["tag"][-6:]] + [f[k] for k in KEYS]))
    print(("%-8s" + " %8.3f" * len(KEYS)) % tuple(["HOST"] + [host[k] for k in KEYS]))


if __name__ == '__main__':
    main()
