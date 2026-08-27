#!/usr/bin/env python3
"""STAGE C: graft a real TopBrain ICA onto the shipped RCCA branch.

Replaces the distal portion of "Centerline curve - RCCA.mrk.json" with a real
right-ICA centerline from TopBrain, producing one drop-in anatomy folder per
TopBrain patient. Every other branch file is copied unchanged.

WHERE THE CUT GOES.  The shipped RCCA branch is 237.5 mm from the ostium; the
TopBrain ICAs run atlantoaxial -> terminus at a median of 106 mm. Both end at
the terminus, so the corresponding host cut is 237.5 - 106 ~= 130 mm. Each
siphon then contributes its own real length, so total route length varies
across the 25 anatomies (~210-274 mm). That is genuine inter-patient ICA
length variation and is deliberately NOT normalised away.

RADIUS.  The host's own radii over roughly 105-130 mm are anatomically
implausible for a cervical ICA (2.6-3.3 mm where 4-5 mm is expected) and are
non-monotonic distally. Rather than scale the real MISR radii down to meet
that suspect number, we hold the TopBrain radii EXACTLY as measured and ramp
the HOST from its last trustworthy anchor at 100 mm (4.45 mm) up to each
siphon's own proximal calibre at the cut, with a smoothstep so the profile has
no kink at either end. This overwrites ~30 mm of measured host radii with an
interpolation: a deliberate repair of a suspect segment, to be disclosed.

COURSE.  Rigid transform only: translate the siphon's proximal point onto the
host cut point, then rotate. Matching the two tangents alone leaves the roll
about the junction undetermined, and the accidental value it took made the
siphons lean over: mean rise along the superior axis fell to 30 mm against the
59 mm the real vessels climb, and three of them ended up descending. So the
rotation matches a FRAME, not a tangent: tangent onto tangent, and the
superior direction onto the superior direction. Superior is +z in both frames
(nibabel world is RAS; the shipped .mrk.json, despite its LPS tag, runs
superiorly along -x, which json_to_branch carries to branch +z). That restores
the climb to 58 mm with no increase in junction kink.

No roll is introduced as a source of variation: it is pinned by the patient's
own superior axis, so the 25 real siphons remain the only thing that differs.

One ICA (mr_021) opens with a cervical coil that doubles back, putting its
proximal tangent 125 deg from its own chord; anchoring on that would stand the
whole siphon on its head. Its first few millimetres are dropped instead. The
host already supplies 130 mm of cervical vessel, so this costs no anatomy.

NEIGHBOURS.  Moving the terminus by 11-42 mm orphans the two branches welded
to the shipped one, and leaves the siphon free to run through vessels the host
happens to have nearby. So the short unnamed cerebral stubs are dropped, and
any patient whose siphon still overlaps a kept vessel is rejected outright
rather than shipped as fused lumens. Both are reported at the end of the run.

    python graft_siphon.py --centerlines topbrain_data/centerlines \\
                           --host eve_bench/data/dualdevicenav/Centrelines_comb \\
                           --out topbrain_data/anatomies
"""
import argparse
import copy
import glob
import json
import os
import shutil
import sys

import numpy as np

RCCA_FILE = "Centerline curve - RCCA.mrk.json"
GRAFT_MM = 130.0        # host arclength at which the real siphon takes over
BLEND_START_MM = 100.0  # last trustworthy host radius anchor
RESAMPLE_MM = 1.0
DISTAL_TRIM_MM = 4.0    # see graft(): keeps the terminus inside its own cap
UP = np.array([0.0, 0.0, 1.0])   # superior, in the branch and world frames alike
ANCHOR_THR = 70.0       # max angle between the anchor tangent and the chord
ANCHOR_MAX_TRIM = 30.0
CRANIAL_Z = 500.0       # above this, an unnamed branch is a cerebral stub
MIN_CLEARANCE_MM = 0.0  # reject a siphon whose lumen overlaps a neighbour's
RVA_FILE = "Centerline curve - RVA.mrk.json"
REPAIR_TARGET_MM = 0.75  # clearance a repaired RVA has to reach
REPAIR_MAX_TRUNC = 45.0  # never cut back more of the RVA than this
REPAIR_MAX_AMP = 14.0    # never deflect it further than this
REPAIR_BLEND_MM = 50.0   # length the deflection is ramped in over


# --- the loader does points.append((y, -z, -x)) on each json position, so
#     branch = (jy, -jz, -jx)  and therefore  json = (-bz, bx, -by).
def json_to_branch(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.stack([y, -z, -x], axis=-1)


def branch_to_json(p):
    bx, by, bz = p[..., 0], p[..., 1], p[..., 2]
    return np.stack([-bz, bx, -by], axis=-1)


def arclength(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def resample(p, r, step=RESAMPLE_MM):
    s = arclength(p)
    n = max(int(round(s[-1] / step)) + 1, 2)
    t = np.linspace(0.0, s[-1], n)
    out = np.stack([np.interp(t, s, p[:, i]) for i in range(3)], axis=1)
    return out, np.interp(t, s, r)


def prep_siphon(p, r, step=RESAMPLE_MM, thr=40.0, max_trim=10.0):
    """Dedup, resample to a uniform step, and trim the seed-point artifacts.

    vmtkCenterlines was run with AppendEndPoints=1, which tacks the source and
    target seeds onto the line slightly off-path. That shows up as 60-100 deg
    turns in the first and last 1-3 mm and nothing anywhere else (measured:
    every large kink sat at arclength fraction <0.03 or >0.98). Trimming those
    few points brings max kink to 23-48 deg, against 36.9 deg for the host.
    """
    keep = np.concatenate([[True], np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6])
    p, r = p[keep], r[keep]
    p, r = resample(p, r, step)
    lo, hi = 0, len(p) - 1
    while hi - lo > 20:
        k = kink_deg(p[lo:hi + 1])
        if k[:3].max() > thr and (lo + 1) * step <= max_trim:
            lo += 1; continue
        if k[-3:].max() > thr and (len(p) - hi) * step <= max_trim:
            hi -= 1; continue
        break
    return p[lo:hi + 1], r[lo:hi + 1]


def kink_deg(p):
    d = np.diff(p, axis=0)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.maximum(n, 1e-9)
    return np.degrees(np.arccos(np.clip((d[:-1] * d[1:]).sum(axis=1), -1, 1)))


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1.0, 1.0))))


def frame(t, up=UP):
    """Orthonormal rows [tangent, superior-in-plane, binormal]."""
    t = unit(np.asarray(t, float))
    n = up - t * float(up @ t)
    if np.linalg.norm(n) < 1e-6:                 # tangent parallel to up
        n = np.array([1.0, 0.0, 0.0]) - t * float(t[0])
    n = unit(n)
    return np.stack([t, n, np.cross(t, n)])


def frame_rotation(t_siph, t_host):
    """Rotation matching tangent to tangent AND superior to superior."""
    return frame(t_host).T @ frame(t_siph)


def anchor_trim(p, thr=ANCHOR_THR, max_trim=ANCHOR_MAX_TRIM, step=RESAMPLE_MM):
    """Advance the proximal end until its tangent agrees with the chord.

    Only fires on a siphon that opens with a coil doubling back on itself, and
    the host supplies the cervical vessel anyway, so the dropped points cost
    no anatomy. Returns the index to start from.
    """
    i, n = 0, int(max_trim / step)
    while i < n and len(p) - i > 30:
        if angle_deg(tangent_at(p[i:], 0), p[-1] - p[i]) <= thr:
            break
        i += 1
    return i


def tangent_at(p, i, span=5):
    lo, hi = max(0, i - span), min(len(p) - 1, i + span)
    return unit(p[hi] - p[lo])


def read_curve(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    m = d["markups"][0]
    pos = np.array([cp["position"] for cp in m["controlPoints"]], float)
    rad = None
    for meas in m.get("measurements", []):
        if meas.get("name") == "Radius" and "controlPointValues" in meas:
            rad = np.array(meas["controlPointValues"], float)
    return d, json_to_branch(pos), rad


def write_curve(template, path, pts_branch, radii):
    d = copy.deepcopy(template)
    m = d["markups"][0]
    pos = branch_to_json(pts_branch)
    cp0 = m["controlPoints"][0]
    m["controlPoints"] = []
    for i, p in enumerate(pos):
        cp = copy.deepcopy(cp0)
        cp["id"] = str(i + 1)
        cp["label"] = "%d" % (i + 1)
        cp["position"] = [float(p[0]), float(p[1]), float(p[2])]
        m["controlPoints"].append(cp)
    m["lastUsedControlPointNumber"] = len(pos)
    for meas in m.get("measurements", []):
        if meas.get("name") == "Radius":
            meas["controlPointValues"] = [float(r) for r in radii]
            meas["enabled"] = True
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f)


def graft(host_pts, host_rad, siph_pts, siph_rad,
          graft_mm=GRAFT_MM, blend_mm=BLEND_START_MM):
    s = arclength(host_pts)
    if s[-1] <= graft_mm + 5:
        raise ValueError("host only %.1f mm, cut at %.1f mm" % (s[-1], graft_mm))
    k = int(np.searchsorted(s, graft_mm))
    k = min(max(k, 2), len(host_pts) - 2)

    keep_p = host_pts[:k + 1].copy()
    keep_r = host_rad[:k + 1].copy()

    # --- trim first: everything below reads the siphon's proximal point, so
    #     the trim has to settle where that point is before they do.
    j = anchor_trim(siph_pts)
    siph_pts, siph_rad = siph_pts[j:], siph_rad[j:]

    # --- radius repair: ramp host from its 100 mm anchor to the siphon's own
    #     proximal calibre, smoothstep, leaving the real MISR values untouched.
    sk = s[:k + 1]
    a = float(np.interp(blend_mm, sk, keep_r))     # trusted host radius at 100 mm
    b = float(siph_rad[0])                          # this patient's real ICA radius
    w = smoothstep((sk - blend_mm) / max(graft_mm - blend_mm, 1e-6))
    inside = sk >= blend_mm
    keep_r = np.where(inside, a + (b - a) * w, keep_r)

    # --- course: translate, then a frame match so the roll about the junction
    #     is pinned by the superior axis instead of left to chance
    t_host = tangent_at(keep_p, len(keep_p) - 1)
    t_siph = tangent_at(siph_pts, 0)
    R = frame_rotation(t_siph, t_host)
    moved = (siph_pts - siph_pts[0]) @ R.T + keep_p[-1]

    pts = np.vstack([keep_p, moved[1:]])
    rad = np.concatenate([keep_r, siph_rad[1:]])

    # Trim the last couple of millimetres. The mesher marks the centerline
    # into a 0.6/0.9 mm voxel cube and smooths it twice, which pulls the
    # terminal cap inward; on a thin distal ICA that leaves the final
    # centerline point OUTSIDE the lumen it is supposed to sit in, and a
    # target sampled there could never be reached. The shipped RCCA does not
    # show this because its terminus is 4.3 mm across. Verified against the
    # enclosure test in check_anatomies.py.
    pts, rad = resample(pts, rad)
    s_end = arclength(pts)
    keep = s_end <= (s_end[-1] - DISTAL_TRIM_MM)
    if int(keep.sum()) >= 30:
        pts, rad = pts[keep], rad[keep]
    return pts, rad


def is_cranial_stub(fname, pts, z=CRANIAL_Z):
    """Short unnamed branch sitting entirely in the cerebral cluster.

    Two of them (13 and 24) begin exactly at the shipped RCCA terminus, so the
    graft, which moves that terminus by 11-42 mm, would leave them starting in
    mid-air; the rest are what several siphons were found running through. The
    named vessels are kept: RCCA, RVA, LCCA, LVA and the arch. The split is
    unambiguous, every stub sits above z=572 and every other unnamed branch
    tops out at z=437.
    """
    return " - " not in fname and float(pts[:, 2].min()) > z


def min_clearance(p, r, branches, from_mm=GRAFT_MM):
    """Smallest lumen-to-lumen gap between the grafted portion and a neighbour.

    Centre distance minus both radii, so negative means the two lumens
    interpenetrate. Only the grafted portion is measured: below the cut the
    RCCA is shipped geometry meeting its neighbours at real junctions, where
    touching is correct.
    """
    s = arclength(p)
    k = int(np.searchsorted(s, from_mm))
    q, qr = p[k:], r[k:]
    worst, who = float("inf"), ""
    for name, bp, br in branches:
        d = np.linalg.norm(q[:, None, :] - bp[None, :, :], axis=2)
        g = float((d - qr[:, None] - br[None, :]).min())
        if g < worst:
            worst, who = g, name
    return worst, who


def pair_gaps(pa, ra, pb, rb):
    """Lumen gap between every point of one vessel and every point of another."""
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    return d - ra[:, None] - rb[None, :]


def fib_sphere(n=400):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    th = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi),
                     np.cos(phi)], axis=1)


_DIRS = fib_sphere()


def rva_shorten(q, qr, vp, vr, target=REPAIR_TARGET_MM,
                max_trunc=REPAIR_MAX_TRUNC, step=1.0):
    """Cut the RVA back short of the contact.

    Costs no invented geometry: the tip is a free end once the basilar stub is
    dropped, so this only makes the wrong-branch decoy shorter. Worth trying
    before any deflection, and it only works when the contact is near the tip.
    """
    vs = arclength(vp)
    g = pair_gaps(q, qr, vp, vr)
    for L in np.arange(vs[-1], vs[-1] - max_trunc, -step):
        m = vs <= L
        if int(m.sum()) < 30:
            break
        if float(g[:, m].min()) >= target:
            return vp[m].copy(), vr[m].copy(), float(vs[-1] - L)
    return None


def rva_deflect(q, qr, vp, vr, others, target=REPAIR_TARGET_MM,
                max_amp=REPAIR_MAX_AMP, blend=REPAIR_BLEND_MM, step=0.25):
    """Bend the RVA away from this one siphon.

    Zero at the proximal end, smoothstepped to full amplitude before the
    lowest contact and held to the tip, so the bifurcation where wrong-branch
    entry happens never moves and no second bend is introduced. The smallest
    amplitude that works is taken, and the result is checked against the other
    vessels so the RVA is not pushed out of one collision into another.
    """
    vs = arclength(vp)
    prof = pair_gaps(q, qr, vp, vr).min(axis=0)
    contact = vs[prof < 3.0]
    lo = float(contact.min()) if len(contact) else float(vs[int(np.argmin(prof))])
    b1 = max(lo - 10.0, 40.0)
    b0 = max(b1 - blend, 10.0)
    w = smoothstep((vs - b0) / max(b1 - b0, 1e-6))
    live = w > 0.01                       # the only part that actually moves

    # Trim the comparison sets to what could possibly come into contact. The
    # displacement is bounded by max_amp, so anything beyond that plus a
    # margin can never matter, and dropping it keeps the search affordable.
    reach = max_amp + 25.0
    near_q = q[pair_gaps(q, qr, vp[live], vr[live]).min(axis=1) < reach]
    near_qr = qr[pair_gaps(q, qr, vp[live], vr[live]).min(axis=1) < reach]
    trimmed = []
    for name, op, orad in others:
        g0 = pair_gaps(op, orad, vp[live], vr[live]).min(axis=1)
        if (g0 < reach).any():
            trimmed.append((name, op[g0 < reach], orad[g0 < reach]))

    for amp in np.arange(step, max_amp + 1e-9, step):
        for n in _DIRS:
            moved = vp + amp * w[:, None] * n
            if float(pair_gaps(near_q, near_qr, moved, vr).min()) < target:
                continue
            if any(float(pair_gaps(op, orad, moved[live], vr[live]).min()) < target
                   for _, op, orad in trimmed):
                continue
            return moved, vr.copy(), float(amp), n.copy(), (b0, b1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centerlines", default="topbrain_data/centerlines")
    ap.add_argument("--host", default="eve_bench/data/dualdevicenav/Centrelines_comb")
    ap.add_argument("--out", default="topbrain_data/anatomies")
    ap.add_argument("--graft-mm", type=float, default=GRAFT_MM)
    ap.add_argument("--blend-mm", type=float, default=BLEND_START_MM)
    ap.add_argument("--min-clearance", type=float, default=MIN_CLEARANCE_MM,
                    help="reject a siphon whose lumen comes closer than this "
                         "to a neighbouring vessel (mm)")
    ap.add_argument("--no-repair", action="store_true",
                    help="reject overlapping anatomies outright instead of "
                         "shortening or deflecting that anatomy's RVA")
    a = ap.parse_args()

    host_rcca = os.path.join(a.host, RCCA_FILE)
    tmpl, hp, hr = read_curve(host_rcca)
    if hr is None:
        print("host RCCA has no Radius measurement"); return 1
    hp, hr = resample(hp, hr)
    print("host RCCA: %.1f mm, %d pts after resample, diam %.2f -> %.2f mm"
          % (arclength(hp)[-1], len(hp), 2 * hr[0], 2 * hr[-1]))
    print("cut at %.0f mm, radius ramp from %.0f mm\n" % (a.graft_mm, a.blend_mm))

    others, dropped, neighbours = [], [], []
    for f in sorted(os.listdir(a.host)):
        if not f.endswith(".json") or f == RCCA_FILE:
            continue
        _, bp, br = read_curve(os.path.join(a.host, f))
        if is_cranial_stub(f, bp):
            dropped.append(f)
            continue
        others.append(f)
        if br is not None:
            neighbours.append((f.replace("Centerline curve", "cc")
                                .replace(".mrk.json", ""), bp, br))
    print("kept %d branches, dropped %d cranial stubs: %s"
          % (len(others), len(dropped),
             ", ".join(x.replace("Centerline curve ", "")
                       .replace(".mrk.json", "") for x in dropped)))
    print("clearance measured against %d of them\n" % len(neighbours))

    rva_tmpl, vp, vr = read_curve(os.path.join(a.host, RVA_FILE))
    rva_name = RVA_FILE.replace("Centerline curve", "cc").replace(".mrk.json", "")
    os.makedirs(a.out, exist_ok=True)

    ok, rejected, repaired = 0, [], []
    print("%-16s %9s %9s %9s %9s %9s %-10s %s"
          % ("case", "total_mm", "siphon_mm", "d_join", "d_term", "clear_mm",
             "nearest", "repair"))
    for f in sorted(glob.glob(os.path.join(a.centerlines, "*_ica.json"))):
        stem = os.path.basename(f).replace("_ica.json", "")
        with open(f) as fh:
            d = json.load(fh)
        sp = np.array(d["points"], float)
        sr = np.array(d["radii"], float)
        sp, sr = prep_siphon(sp, sr)
        try:
            gp, gr = graft(hp, hr, sp, sr, a.graft_mm, a.blend_mm)
        except Exception as e:                        # noqa: BLE001
            print("  SKIP %-14s %s" % (stem, e)); continue

        # --- repair, per anatomy and only where needed. The 21 that clear on
        #     their own keep the shipped RVA byte for byte; nothing global is
        #     perturbed to accommodate a handful of patients.
        k = int(np.searchsorted(arclength(gp), a.graft_mm))
        sq, sqr = gp[k:], gr[k:]
        rva, note = None, ""
        clear, who = min_clearance(gp, gr, neighbours, a.graft_mm)
        if clear < a.min_clearance and who == rva_name and not a.no_repair:
            rest = [n for n in neighbours if n[0] != rva_name]
            cut = rva_shorten(sq, sqr, vp, vr)
            if cut is not None:
                rva = (cut[0], cut[1])
                note = "shortened %.0f mm" % cut[2]
            else:
                bent = rva_deflect(sq, sqr, vp, vr, rest)
                if bent is not None:
                    rva = (bent[0], bent[1])
                    note = ("deflected %.2f mm, blend %.0f-%.0f mm"
                            % (bent[2], bent[4][0], bent[4][1]))
            if rva is not None:
                patched = [(n[0], rva[0], rva[1]) if n[0] == rva_name else n
                           for n in neighbours]
                clear, who = min_clearance(gp, gr, patched, a.graft_mm)

        total = arclength(gp)[-1]
        ki = int(np.argmin(np.abs(arclength(gp) - a.graft_mm)))
        row = ("%-16s %9.1f %9.1f %9.2f %9.2f %+9.2f %-10s %s"
               % (stem, total, d["length_mm"], 2 * gr[ki], 2 * gr[-1],
                  clear, who, note))
        if clear < a.min_clearance:
            rejected.append((stem, clear, who))
            print(row + "   REJECTED")
            continue

        folder = os.path.join(a.out, stem, "Centrelines_comb")
        os.makedirs(folder, exist_ok=True)
        for o in others:
            if o == RVA_FILE and rva is not None:
                write_curve(rva_tmpl, os.path.join(folder, o), rva[0], rva[1])
            else:
                shutil.copy2(os.path.join(a.host, o), os.path.join(folder, o))
        write_curve(tmpl, os.path.join(folder, RCCA_FILE), gp, gr)
        ok += 1
        if note:
            repaired.append((stem, note, clear))
        print(row)

    print("\nwrote %d anatomy folders to %s" % (ok, a.out))
    if repaired:
        print("repaired the RVA in %d of them (the other %d keep it unchanged):"
              % (len(repaired), ok - len(repaired)))
        for stem, note, clear in repaired:
            print("   %-16s %-34s clearance %+.2f mm" % (stem, note, clear))
    if rejected:
        print("rejected %d for lumen overlap (needed >= %.1f mm):"
              % (len(rejected), a.min_clearance))
        for stem, clear, who in rejected:
            print("   %-16s %+6.2f mm against %s" % (stem, clear, who))
    return 0


if __name__ == "__main__":
    sys.exit(main())
