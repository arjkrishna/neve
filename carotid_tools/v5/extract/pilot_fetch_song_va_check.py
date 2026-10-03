#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_fetch_song_va_check.py (= /sp/v5_va/pilot_fetch_song_va_check.py in the build containers)
"""pilot_fetch_song_va_check.py (pilot_fetch_song, resume pass, 2026-10-01 ~20:15 EDT).

Why: qc_va_axial.png (pilot_fetch_song_va_axial.py) centres its crops on the median of dense bone, which above
~arch+90 mm is the mandible in some cases, so it cannot confirm the VAs in V2-V3. This script tracks the spinal
canal instead and looks at the VA in the canal frame.

Per case:
  1. canal track, slice by slice, arch-10 mm .. foramen magnum: bone = HU >= T_BONE (qc_pass1 levels.bone_threshold_hu),
     closed with a 2 mm disk, holes 60-900 mm2 inside a +-50 mm x window around qc_pass1 spine_mid_x. Seed: the hole
     nearest the spine midline (area 100-450 mm2) between arch+40 and arch+90 mm; followed up and down (<= 4 mm jump
     per slice, a gap of > 25 mm ends the track); gaps interpolated; median-smoothed over 9 mm.
  2. <img>/<case>/qc_va_check.png
       row 1: straightened MIPs in the canal frame (dx = x - canal x, dy = y - canal y; RAS, +dx = patient right,
              +dy = anterior): coronal MIP of the foramen band dy +2..+26 mm (raw window, then bone removed), and the
              sagittal MIP of each side, |dx| 8..32 mm, dy -15..+55 mm (bone removed: CCA anterior, VA posterior).
              Bone removed = HU >= T_BONE, axial holes >= 60 mm2 filled (vertebral bodies; the ~20-40 mm2 transverse
              foramina stay open), dilated ~1 mm.
       rows 2-3: 10 axial crops centred on the canal from arch+10 mm to the canal top, with the ts_screen foramen
              band (|dx| 12-22 mm, dy +7..+19 mm) outlined in cyan.
  3. <img>/<case>/qc_va_check.json: canal track, band continuity per side = fraction of slices from arch+40 mm to the
     canal top whose band (|dx| 10-26 mm, dy +4..+22 mm, bone removed) max >= 0.6 x the aorta median, and the
     longest run below it (mm); plus the side-file check (4).
  4. side-file check: ct_bg-100.nii.gz and ct_negzero-100.nii.gz equal rint(shipped) wherever they are not -100;
     -100 only where the shipped value is 0 (bg: and outside body_mask.nii.gz, every such voxel; negzero: exactly the
     -0.0 voxels); shipped values integer and int16-representable; shape, affine and qform/sform codes as shipped.
usage: pilot_fetch_song_va_check.py CASE [CASE ...]   (neve-build-stagea; read-only on the shipped and side files)
"""
import sys, os, json, time
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage import measure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

IMG = "/repo/carotid_data/va_sources/thin_slice/song"
DER = "/repo/carotid_data/va_sources/derived/thin_slice/song"


def disk(r):
    r = max(1, int(round(r)))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def side_check(case, img, v0):
    out = dict()
    dd = os.path.join(DER, case)
    qc0, sc0 = int(img.header["qform_code"]), int(img.header["sform_code"])
    neg0 = (v0 == 0) & np.signbit(v0)
    zero = v0 == 0
    rv = np.rint(v0)
    out["shipped_integer_valued"] = bool(np.array_equal(rv, v0))
    out["shipped_int16_range"] = bool(v0.min() >= -32768 and v0.max() <= 32767)
    bm = None
    p = os.path.join(dd, "body_mask.nii.gz")
    if os.path.exists(p):
        bi = nib.load(p); bm = np.asanyarray(bi.dataobj).astype(bool)
        out["body_mask"] = dict(shape_ok=bi.shape == img.shape, affine_ok=bool(np.allclose(bi.affine, img.affine, atol=1e-4)),
                                frac=round(float(bm.mean()), 4))
    for name, rule in (("ct_bg-100.nii.gz", "bg"), ("ct_negzero-100.nii.gz", "negzero")):
        p = os.path.join(dd, name)
        if not os.path.exists(p):
            out[name] = "missing"; continue
        si = nib.load(p)
        s = np.asanyarray(si.dataobj)
        r = dict(dtype=str(si.get_data_dtype()), shape_ok=si.shape == img.shape,
                 affine_ok=bool(np.allclose(si.affine, img.affine, atol=1e-4)),
                 codes_ok=int(si.header["qform_code"]) == qc0 and int(si.header["sform_code"]) == sc0)
        m = s == -100
        keep = ~m
        r["equal_elsewhere"] = bool(np.array_equal(s[keep].astype(np.float64), rv[keep]))
        r["minus100_only_on_zero"] = bool(zero[m].all())
        r["n_minus100"] = int(m.sum()); r["frac_minus100"] = round(float(m.mean()), 4)
        if rule == "bg":
            if bm is not None:
                r["minus100_outside_body_only"] = bool(not np.logical_and(m, bm).any())
                r["every_zero_outside_body_set"] = bool(np.array_equal(m, zero & ~bm))
        else:
            r["minus100_equals_negzero"] = bool(np.array_equal(m, neg0))
        r["ok"] = bool(r["shape_ok"] and r["affine_ok"] and r["codes_ok"] and r["equal_elsewhere"] and r["minus100_only_on_zero"]
                       and r.get("minus100_outside_body_only", True) and r.get("every_zero_outside_body_set", True)
                       and r.get("minus100_equals_negzero", True))
        out[name] = r
        del s, m, keep
    out["ok"] = bool(out["shipped_integer_valued"] and out["shipped_int16_range"]
                     and all(isinstance(out.get(n), dict) and out[n]["ok"] for n in ("ct_bg-100.nii.gz", "ct_negzero-100.nii.gz")))
    return out


def main():
    for case in sys.argv[1:]:
        t0 = time.time()
        d = os.path.join(IMG, case)
        q = json.load(open(os.path.join(d, "qc_pass1.json")))
        img = nib.load(os.path.join(d, "ct.nii.gz"))
        v0 = np.asanyarray(img.dataobj).astype(np.float32)
        rec = dict(case=case, date=time.strftime("%Y-%m-%d %H:%M"), script="pilot_fetch_song_va_check.py")
        rec["side_check"] = side_check(case, img, v0)
        ras = nib.as_closest_canonical(nib.Nifti1Image(v0, img.affine))
        A = ras.affine
        assert np.allclose(A[:3, :3], np.diag(np.diag(A[:3, :3]))) and (np.diag(A)[:3] > 0).all(), "not axis-aligned RAS"
        v = np.asanyarray(ras.dataobj)
        sp = np.diag(A)[:3].astype(float); org = A[:3, 3].astype(float)
        nx, ny, nz = v.shape
        zmm = org[2] + np.arange(nz) * sp[2]
        za = float(q["aorta_auto"]["arch_top_z"])
        aoh = float(q["aorta_auto"].get("aorta_hu_med") or 400.0)
        T_BONE = float(q["levels"]["bone_threshold_hu"])
        vmax = float(max(450.0, 1.3 * aoh))
        # instance 3 (20:30): qc_pass1 spine_mid_x is off by ~29 mm on AHMU1218008 (it is the bone median, pulled by the
        # clavicles/shoulder), so the reference x is the median vertebral-body centroid of the L2-L4 candidate levels,
        # the x window is +-60 mm and the seed is the densest canal-sized hole cluster (see below).
        bmx = [c.get("body_mid_x") for n, c in q.get("candidates", {}).items() if n[:2] in ("L2", "L3", "L4")]
        bmx = [b for b in bmx if b is not None]
        xmid = float(np.median(bmx)) if bmx else float(q["spine_mid_x"])
        ixm = int(round((xmid - org[0]) / sp[0]))
        hw = int(round(60 / sp[0]))
        x0, x1 = max(0, ixm - hw), min(nx, ixm + hw + 1)
        se_c = disk(2.0 / sp[0])
        k_lo = int(np.clip(np.searchsorted(zmm, za - 10), 0, nz - 1))
        k_hi = int(np.clip(np.searchsorted(zmm, za + 260), 0, nz - 1))

        cache = {}

        def cands(k):
            if k in cache:
                return cache[k]
            s = v[x0:x1, :, k]
            bone = s >= T_BONE
            bc = ndi.binary_closing(bone, structure=se_c, border_value=0)
            holes = ndi.binary_fill_holes(bc) & ~bc
            lab, n = ndi.label(holes)
            out = []
            if n:
                for r in measure.regionprops(lab):
                    ar = r.area * sp[0] * sp[1]
                    if 60 <= ar <= 900:
                        out.append((org[0] + (x0 + r.centroid[0]) * sp[0], org[1] + r.centroid[1] * sp[1], ar))
            cache[k] = out
            return out

        # seed
        # seed (instance 3): canal-sized holes (100-450 mm2) within 30 mm of xmid between arch+40 and arch+90 mm; the
        # spinal canal is the hole that recurs at the same (x, y) on many slices, so cluster them on an 8 mm grid and
        # take the entry nearest the median of the most populated cell.
        pool = []
        for k in range(nz):
            if not (za + 40 <= zmm[k] <= za + 90) or (k % 3):
                continue
            for (x, y, ar) in cands(k):
                if 100 <= ar <= 450 and abs(x - xmid) < 30:
                    pool.append((k, x, y, ar))
        best = None
        if pool:
            from collections import Counter
            cell = Counter((int(np.floor(p[1] / 8.0)), int(np.floor(p[2] / 8.0))) for p in pool)
            (gx, gy), _n = cell.most_common(1)[0]
            near = [p for p in pool if abs(p[1] - (gx + 0.5) * 8) <= 8 and abs(p[2] - (gy + 0.5) * 8) <= 8]
            mx, my = np.median([p[1] for p in near]), np.median([p[2] for p in near])
            k, x, y, ar = min(near, key=lambda p: np.hypot(p[1] - mx, p[2] - my))
            best = (0.0, k, x, y, ar)
            rec["seed_pool"] = dict(n=len(pool), n_cell=int(_n), xref=round(xmid, 1))
        cx = np.full(nz, np.nan); cy = np.full(nz, np.nan); car = np.full(nz, np.nan)
        rec["canal"] = dict(seed=None)
        if best is None:
            rec["canal"]["error"] = "no canal seed between arch+40 and arch+90 mm"
            json.dump(rec, open(os.path.join(d, "qc_va_check.json"), "w"), indent=1)
            print(case, "NO CANAL SEED", flush=True)
            continue
        _, ks, xs_, ys_, ars = best
        rec["canal"]["seed"] = dict(k=int(ks), z=round(float(zmm[ks]), 1), x=round(xs_, 1), y=round(ys_, 1), area_mm2=round(ars, 0))
        cx[ks], cy[ks], car[ks] = xs_, ys_, ars
        ends = {}
        for step, kend in ((1, k_hi), (-1, k_lo)):
            px, py = xs_, ys_; last = ks; k = ks + step
            while (k <= kend if step > 0 else k >= kend):
                cs = [c for c in cands(k) if np.hypot(c[0] - px, c[1] - py) <= 4.0]
                if cs:
                    c = min(cs, key=lambda c: np.hypot(c[0] - px, c[1] - py))
                    cx[k], cy[k], car[k] = c; px, py = c[0], c[1]; last = k
                elif abs(zmm[k] - zmm[last]) > 25:
                    break
                k += step
            ends[step] = last
        k_top, k_bot = ends[1], ends[-1]
        idx = np.arange(nz); ok = ~np.isnan(cx)
        cxi = np.interp(idx, idx[ok], cx[ok]); cyi = np.interp(idx, idx[ok], cy[ok])
        w = max(3, int(round(9 / sp[2])) | 1)
        cxs = ndi.median_filter(cxi, size=w, mode="nearest"); cys = ndi.median_filter(cyi, size=w, mode="nearest")
        found = int(ok[k_bot:k_top + 1].sum()); span = k_top - k_bot + 1
        rec["canal"].update(k_bottom=int(k_bot), z_bottom=round(float(zmm[k_bot]), 1), k_top=int(k_top),
                            z_top=round(float(zmm[k_top]), 1), top_above_arch_mm=round(float(zmm[k_top] - za), 1),
                            frac_slices_found=round(found / max(1, span), 3),
                            track_every_5mm=[[round(float(zmm[k]), 1), round(float(cxs[k]), 1), round(float(cys[k]), 1),
                                              None if np.isnan(car[k]) else round(float(car[k]), 0)]
                                             for k in range(k_bot, k_top + 1, max(1, int(round(5 / sp[2]))))])
        # straightened MIPs
        kz0 = int(np.clip(np.searchsorted(zmm, za - 10), 0, nz - 1)); kz1 = int(min(nz - 1, k_top + int(round(15 / sp[2]))))
        DXW = 45.0
        ndx = int(round(DXW / sp[0])); dxs = np.arange(-ndx, ndx + 1) * sp[0]
        dy_a, dy_b = int(round(2 / sp[1])), int(round(26 / sp[1]))
        sy_a, sy_b = int(round(-15 / sp[1])), int(round(55 / sp[1]))
        sx_a, sx_b = int(round(8 / sp[0])), int(round(32 / sp[0]))
        min_hole = 60.0 / (sp[0] * sp[1]); dil = disk(1.0 / sp[0])
        cor_raw, cor_ves, sag_r, sag_l, zrows = [], [], [], [], []
        band_r, band_l = [], []
        bx_a, bx_b = int(round(10 / sp[0])), int(round(26 / sp[0]))
        by_a, by_b = int(round(4 / sp[1])), int(round(22 / sp[1]))
        PADX = ndx + sx_b + 2; PADY = max(-sy_a, dy_b, sy_b, by_b) + 2
        for k in range(kz0, kz1 + 1):
            s = v[:, :, k]
            ix = int(round((cxs[k] - org[0]) / sp[0])); iy = int(round((cys[k] - org[1]) / sp[1]))
            xa, xb = ix - PADX, ix + PADX + 1; ya, yb = iy - PADY, iy + PADY + 1
            blk = np.zeros((xb - xa, yb - ya), np.float32)
            sxa, sxb, sya, syb = max(0, xa), min(nx, xb), max(0, ya), min(ny, yb)
            blk[sxa - xa:sxb - xa, sya - ya:syb - ya] = s[sxa:sxb, sya:syb]
            bone = blk >= T_BONE
            f = ndi.binary_fill_holes(bone); holes = f & ~bone
            if holes.any():
                hl, hn = ndi.label(holes)
                hs = ndi.sum(holes, hl, range(1, hn + 1))
                bone = bone | np.isin(hl, 1 + np.nonzero(hs >= min_hole)[0])
            bone = ndi.binary_dilation(bone, structure=dil)
            ves = np.where(bone, 0, blk)
            cxp, cyp = PADX, PADY  # canal centre in block indices
            xsl = slice(cxp - ndx, cxp + ndx + 1)
            cor_raw.append(blk[xsl, cyp + dy_a:cyp + dy_b + 1].max(axis=1))
            cor_ves.append(ves[xsl, cyp + dy_a:cyp + dy_b + 1].max(axis=1))
            sag_r.append(ves[cxp + sx_a:cxp + sx_b + 1, cyp + sy_a:cyp + sy_b + 1].max(axis=0))
            sag_l.append(ves[cxp - sx_b:cxp - sx_a + 1, cyp + sy_a:cyp + sy_b + 1].max(axis=0))
            band_r.append(float(ves[cxp + bx_a:cxp + bx_b + 1, cyp + by_a:cyp + by_b + 1].max()))
            band_l.append(float(ves[cxp - bx_b:cxp - bx_a + 1, cyp + by_a:cyp + by_b + 1].max()))
            zrows.append(zmm[k])
        zrows = np.array(zrows); band_r = np.array(band_r); band_l = np.array(band_l)
        sel = (zrows >= za + 40) & (zrows <= zmm[k_top])
        thr = 0.6 * aoh
        cont = {}
        for side, b in (("R", band_r), ("L", band_l)):
            bb = b[sel] >= thr
            run = 0; worst = 0
            for x in bb:
                run = 0 if x else run + 1; worst = max(worst, run)
            cont[side] = dict(frac_slices_bright=round(float(bb.mean()) if bb.size else 0.0, 3),
                              longest_dark_run_mm=round(worst * sp[2], 1), n_slices=int(bb.size))
        rec["band_continuity"] = dict(threshold_hu=round(thr, 0), z_from=round(za + 40, 1), z_to=round(float(zmm[k_top]), 1),
                                      band="|dx| 10-26 mm, dy +4..+22 mm from the canal centre, bone removed", **cont)
        # figure
        fig = plt.figure(figsize=(26, 25))
        gs = fig.add_gridspec(3, 5, height_ratios=[1.6, 1, 1])
        ext_c = [dxs[0], dxs[-1], zrows[0], zrows[-1]]
        dys = np.arange(sy_a, sy_b + 1) * sp[1]
        ext_s = [dys[0], dys[-1], zrows[0], zrows[-1]]
        panels = [(np.array(cor_raw).clip(0, 1500), "coronal MIP, foramen band dy +2..+26, raw (0..%d HU)" % vmax, ext_c, "c"),
                  (np.array(cor_ves), "coronal MIP, foramen band, bone removed (150..%d HU)" % vmax, ext_c, "c"),
                  (np.array(sag_r), "RIGHT sagittal MIP |dx| 8..32, bone removed", ext_s, "s"),
                  (np.array(sag_l), "LEFT sagittal MIP |dx| 8..32, bone removed", ext_s, "s")]
        for j, (im, ttl, ext, kind) in enumerate(panels):
            a = fig.add_subplot(gs[0, j] if j < 3 else gs[0, 3:5])
            lo = 0 if j == 0 else 150
            a.imshow(im, cmap="gray", origin="lower", vmin=lo, vmax=vmax, extent=ext,
                     aspect=1.0 if kind == "c" else 1.0, interpolation="nearest")
            if kind == "c":
                a.set_xlim(DXW, -DXW)
                for xx in (12, 22, -12, -22):
                    a.axvline(xx, color="c", lw=0.5, ls=":")
                a.set_xlabel("dx from canal (mm; patient R on image left)")
            else:
                for yy in (7, 19):
                    a.axvline(yy, color="c", lw=0.5, ls=":")
                a.set_xlabel("dy from canal (mm; anterior to the right)")
            a.axhline(za, color="orange", lw=0.7, ls="--"); a.axhline(za + 40, color="y", lw=0.5, ls=":")
            a.axhline(zmm[k_top], color="m", lw=0.7, ls="--")
            a.set_title(ttl, fontsize=11); a.set_ylabel("z RAS mm")
        # axial crops
        zl = np.linspace(za + 10, zmm[k_top], 10)
        for j, z in enumerate(zl):
            a = fig.add_subplot(gs[1 + j // 5, j % 5])
            k = int(np.clip(np.argmin(np.abs(zmm - z)), 0, nz - 1))
            cxk, cyk = cxs[k], cys[k]
            s = v[:, :, k]
            ext = [org[0], org[0] + nx * sp[0], org[1], org[1] + ny * sp[1]]
            a.imshow(s.T, cmap="gray", origin="lower", vmin=0, vmax=vmax, extent=ext, interpolation="nearest")
            a.set_xlim(cxk + 45, cxk - 45); a.set_ylim(cyk - 25, cyk + 60)
            a.plot(cxk, cyk, "g+", ms=12)
            for sg in (1, -1):
                a.add_patch(Rectangle((cxk + sg * 12 if sg > 0 else cxk - 22, cyk + 7), 10, 12, fill=False, color="c", lw=0.8, ls="--"))
            a.set_title("z %.0f (arch %+.0f) k %d%s" % (zmm[k], zmm[k] - za, k, "" if not np.isnan(cx[k]) else " (canal interp.)"), fontsize=10)
            a.tick_params(labelsize=7)
        bc = rec["band_continuity"]
        fig.suptitle("%s  VA check in the canal frame (R/L band bright %.2f / %.2f of slices arch+40..canal top; longest dark run "
                     "%.0f / %.0f mm). orange: arch top, yellow: arch+40, magenta: canal top (foramen magnum). Cyan: ts_screen "
                     "foramen band |dx| 12-22, dy +7..+19" % (case, bc["R"]["frac_slices_bright"], bc["L"]["frac_slices_bright"],
                                                              bc["R"]["longest_dark_run_mm"], bc["L"]["longest_dark_run_mm"]),
                     fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.98]); plt.savefig(os.path.join(d, "qc_va_check.png"), dpi=55); plt.close(fig)
        rec["seconds"] = round(time.time() - t0, 1)
        json.dump(rec, open(os.path.join(d, "qc_va_check.json"), "w"), indent=1)
        print(case, "canal top above arch %.0f mm" % rec["canal"]["top_above_arch_mm"], "found %.2f" % rec["canal"]["frac_slices_found"],
              "bandR", cont["R"], "bandL", cont["L"], "side_ok", rec["side_check"]["ok"], "%.0fs" % rec["seconds"], flush=True)
        del v, v0, ras, cache


if __name__ == "__main__":
    main()
