#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_fetch_song_va_check2.py (= /sp/v5_va/pilot_fetch_song_va_check2.py in the build containers)
"""pilot_fetch_song_va_check2.py (pilot_fetch_song, instance 3, 2026-10-01 ~20:27 EDT).

Why: qc_va_axial.png centres its crops on the median of dense bone, which above ~arch+90 mm is the mandible, so it
cannot confirm the VAs in V2-V4. Instance 2's pilot_fetch_song_va_check.py tracked the spinal canal as a hole in the
>= 700 HU bone ring; it never ran, and on AHMU1218008 it finds no canal at all: that patient's vertebrae read mostly
300-700 HU (only cortical rims reach 700) and the canal ring never closes (results/pilot_fetch_song_dbg/). This version
tracks the VERTEBRAL COLUMN instead, by mean shift:

  1. seed: qc_pass1 candidates L3 (arch+60) body_mid_x/y (L2/L4 if missing); 15 mean-shift iterations on that slice.
  2. slice by slice up (to min(arch+240, top-3 mm)) and down (to arch-10): centroid of the voxels >= 200 HU within
     22 mm of the current centre, moved 30% of the way per slice and at most 1.5 mm; no bone (< 40 mm2) keeps the
     centre. Then median-smoothed over 9 mm. The centre sits on the vertebra (body + arch), not on the canal.
  3. <case>/qc_va_check.png:
       row 1: straightened MIPs in the column frame (dx = x - col x, dy = y - col y; RAS: +dx = patient right,
              +dy = anterior): coronal MIP of the band dy -8..+12 mm (raw), the same with bone removed, and the
              sagittal MIP of each side, |dx| 13..32 mm, dy -45..+50 mm, bone removed.
              Bone removed = HU >= qc_pass1 bone threshold, axial holes >= 60 mm2 filled, dilated ~1 mm. On low-density
              bone (008) part of the vertebra stays; read the tube shape, not the brightness.
       rows 2-4: 15 axial crops centred on the column from arch+10 mm to min(track top, arch+215) (window 0..max(450, 1.6 aorta)).
  4. <case>/qc_va_check.json: column track (every 5 mm), indicative band brightness per side (fraction of slices from
     arch+40 to the track top whose band |dx| 12-34, dy -15..+12 mm, bone removed, has a max >= 0.6 x aorta median),
     and the side-file integrity check of pilot_fetch_song_va_check.side_check (instance 2's function, unchanged).
usage: pilot_fetch_song_va_check2.py CASE [CASE ...]   (neve-build-stagea; read-only on shipped and side files)
"""
import sys, os, json, time
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, "/sp/v5_va")
from pilot_fetch_song_va_check import side_check, disk  # noqa: E402

IMG = "/repo/carotid_data/va_sources/thin_slice/song"
T_COL = 200.0
R_COL = 22.0  # rev ~20:29 EDT: the track holds where the disk has > 900 mm2 of bone (skull base)


def main():
    for case in sys.argv[1:]:
        t0 = time.time()
        d = os.path.join(IMG, case)
        q = json.load(open(os.path.join(d, "qc_pass1.json")))
        img = nib.load(os.path.join(d, "ct.nii.gz"))
        v0 = np.asanyarray(img.dataobj).astype(np.float32)
        rec = dict(case=case, date=time.strftime("%Y-%m-%d %H:%M"), script="pilot_fetch_song_va_check2.py")
        rec["side_check"] = side_check(case, img, v0)
        ras = nib.as_closest_canonical(nib.Nifti1Image(v0, img.affine))
        del v0
        A = ras.affine
        assert np.allclose(A[:3, :3], np.diag(np.diag(A[:3, :3]))) and (np.diag(A)[:3] > 0).all(), "not axis-aligned RAS"
        v = np.asanyarray(ras.dataobj)
        sp = np.diag(A)[:3].astype(float); org = A[:3, 3].astype(float)
        nx, ny, nz = v.shape
        xs = org[0] + np.arange(nx) * sp[0]; ys = org[1] + np.arange(ny) * sp[1]
        zmm = org[2] + np.arange(nz) * sp[2]
        za = float(q["aorta_auto"]["arch_top_z"])
        aoh = float(q["aorta_auto"].get("aorta_hu_med") or 400.0)
        T_BONE = float(q["levels"]["bone_threshold_hu"])
        vmax = float(max(450.0, 1.6 * aoh))  # rev ~20:29 EDT: VA lumen (~aorta HU) mid-grey, cortical bone white

        def centroid(k, px, py):
            ia = max(0, int((px - R_COL - org[0]) / sp[0])); ib = min(nx, int((px + R_COL - org[0]) / sp[0]) + 2)
            ja = max(0, int((py - R_COL - org[1]) / sp[1])); jb = min(ny, int((py + R_COL - org[1]) / sp[1]) + 2)
            sub = v[ia:ib, ja:jb, k]
            X, Y = np.meshgrid(xs[ia:ib], ys[ja:jb], indexing="ij")
            m = (sub >= T_COL) & ((X - px) ** 2 + (Y - py) ** 2 <= R_COL ** 2)
            if m.sum() * sp[0] * sp[1] < 40:
                return None
            return float(X[m].mean()), float(Y[m].mean()), float(m.sum() * sp[0] * sp[1])

        cand = q.get("candidates", {})
        seed = None
        for nm in ("L3", "L2", "L4"):
            for n, c in cand.items():
                if n.startswith(nm) and c.get("body_mid_x") is not None:
                    seed = (int(c["k"]), float(c["body_mid_x"]), float(c["body_mid_y"]), n); break
            if seed:
                break
        if seed is None:
            rec["error"] = "no body_mid seed in qc_pass1 candidates"
            json.dump(rec, open(os.path.join(d, "qc_va_check.json"), "w"), indent=1); print(case, "NO SEED", flush=True)
            continue
        # qc_pass1 k is on the RAS grid (the qc scripts work in canonical RAS); check by z
        ks = int(np.argmin(np.abs(zmm - float(cand[seed[3]]["z"]))))
        px, py = seed[1], seed[2]
        for _ in range(15):
            c = centroid(ks, px, py)
            if c is None:
                break
            px, py = c[0], c[1]
        rec["column_seed"] = dict(level=seed[3], z=round(float(zmm[ks]), 1), start=[round(seed[1], 1), round(seed[2], 1)],
                                  converged=[round(px, 1), round(py, 1)])
        k_lo = int(np.clip(np.searchsorted(zmm, za - 10), 0, nz - 1))
        k_hi = int(np.clip(np.searchsorted(zmm, min(za + 240, zmm[-1] - 3)), 0, nz - 1))
        cx = np.full(nz, np.nan); cy = np.full(nz, np.nan); hit = np.zeros(nz, bool)
        cx[ks], cy[ks], hit[ks] = px, py, True
        for step, kend in ((1, k_hi), (-1, k_lo)):
            qx, qy = px, py
            k = ks + step
            while (k <= kend if step > 0 else k >= kend):
                c = centroid(k, qx, qy)
                if c is not None and c[2] <= 900:  # rev ~20:29 EDT: > 900 mm2 of bone in the disk = skull base: hold
                    dx, dy = 0.3 * (c[0] - qx), 0.3 * (c[1] - qy)
                    dd = np.hypot(dx, dy)
                    if dd > 1.5:
                        dx, dy = dx * 1.5 / dd, dy * 1.5 / dd
                    qx, qy = qx + dx, qy + dy; hit[k] = True
                cx[k], cy[k] = qx, qy
                k += step
        k_bot, k_top = k_lo, k_hi
        # rev ~20:31 EDT: at the occiput the disk centroid drifts sideways into the skull (008: +45 mm by arch+200), so
        # above arch+150 mm the centre is clamped to within 8 mm of its arch+150 position.
        k150 = int(np.clip(np.searchsorted(zmm, za + 150), ks, k_top))
        for k in range(k150 + 1, k_top + 1):
            ddx, ddy = cx[k] - cx[k150], cy[k] - cy[k150]
            dd = np.hypot(ddx, ddy)
            if dd > 8.0:
                cx[k], cy[k] = cx[k150] + ddx * 8.0 / dd, cy[k150] + ddy * 8.0 / dd
        w = max(3, int(round(9 / sp[2])) | 1)
        cxs = cx.copy(); cys = cy.copy()
        seg = slice(k_bot, k_top + 1)
        cxs[seg] = ndi.median_filter(cx[seg], size=w, mode="nearest"); cys[seg] = ndi.median_filter(cy[seg], size=w, mode="nearest")
        rec["column"] = dict(z_bottom=round(float(zmm[k_bot]), 1), z_top=round(float(zmm[k_top]), 1),
                             top_above_arch_mm=round(float(zmm[k_top] - za), 1),
                             frac_slices_with_bone=round(float(hit[seg].mean()), 3),
                             image_top_above_arch_mm=round(float(zmm[-1] - za), 1),
                             track_every_5mm=[[round(float(zmm[k]), 1), round(float(cxs[k]), 1), round(float(cys[k]), 1)]
                                              for k in range(k_bot, k_top + 1, max(1, int(round(5 / sp[2]))))])
        # straightened MIPs
        DXW = 50.0
        ndx = int(round(DXW / sp[0])); dxs = np.arange(-ndx, ndx + 1) * sp[0]
        dy_a, dy_b = int(round(-8 / sp[1])), int(round(12 / sp[1]))  # rev ~20:29 EDT: thin band through the foramina
        sy_a, sy_b = int(round(-45 / sp[1])), int(round(50 / sp[1]))
        sx_a, sx_b = int(round(13 / sp[0])), int(round(32 / sp[0]))  # rev ~20:29 EDT: foramen column only
        bx_a, bx_b = int(round(12 / sp[0])), int(round(34 / sp[0]))
        by_a, by_b = int(round(-15 / sp[1])), int(round(12 / sp[1]))
        min_hole = 60.0 / (sp[0] * sp[1]); dil = disk(1.0 / sp[0])
        PADX = max(ndx, sx_b) + 2; PADY = max(-sy_a, -dy_a, sy_b, dy_b) + 2
        cor_raw, cor_ves, sag_r, sag_l, zrows, band_r, band_l = [], [], [], [], [], [], []
        for k in range(k_bot, k_top + 1):
            s = v[:, :, k]
            ix = int(round((cxs[k] - org[0]) / sp[0])); iy = int(round((cys[k] - org[1]) / sp[1]))
            xa, xb = ix - PADX, ix + PADX + 1; ya, yb = iy - PADY, iy + PADY + 1
            blk = np.zeros((xb - xa, yb - ya), np.float32)
            sxa, sxb, sya, syb = max(0, xa), min(nx, xb), max(0, ya), min(ny, yb)
            if sxb > sxa and syb > sya:
                blk[sxa - xa:sxb - xa, sya - ya:syb - ya] = s[sxa:sxb, sya:syb]
            bone = blk >= T_BONE
            f = ndi.binary_fill_holes(bone); holes = f & ~bone
            if holes.any():
                hl, hn = ndi.label(holes)
                hs = ndi.sum(holes, hl, range(1, hn + 1))
                bone = bone | np.isin(hl, 1 + np.nonzero(hs >= min_hole)[0])
            bone = ndi.binary_dilation(bone, structure=dil)
            ves = np.where(bone, 0, blk)
            c0, c1 = PADX, PADY
            xsl = slice(c0 - ndx, c0 + ndx + 1)
            cor_raw.append(blk[xsl, c1 + dy_a:c1 + dy_b + 1].max(axis=1))
            cor_ves.append(ves[xsl, c1 + dy_a:c1 + dy_b + 1].max(axis=1))
            sag_r.append(ves[c0 + sx_a:c0 + sx_b + 1, c1 + sy_a:c1 + sy_b + 1].max(axis=0))
            sag_l.append(ves[c0 - sx_b:c0 - sx_a + 1, c1 + sy_a:c1 + sy_b + 1].max(axis=0))
            band_r.append(float(ves[c0 + bx_a:c0 + bx_b + 1, c1 + by_a:c1 + by_b + 1].max()))
            band_l.append(float(ves[c0 - bx_b:c0 - bx_a + 1, c1 + by_a:c1 + by_b + 1].max()))
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
        rec["band_brightness_indicative"] = dict(
            threshold_hu=round(thr, 0), z_from=round(za + 40, 1), z_to=round(float(zmm[k_top]), 1),
            band="|dx| 12-34 mm, dy -15..+12 mm from the column centre, bone removed; residual low-density bone counts "
                 "as bright, so this is indicative only -- the verdict is by eye", **cont)
        # figure
        fig = plt.figure(figsize=(26, 33))
        gs = fig.add_gridspec(4, 5, height_ratios=[1.7, 1, 1, 1])
        ext_c = [dxs[0], dxs[-1], zrows[0], zrows[-1]]
        dys = np.arange(sy_a, sy_b + 1) * sp[1]
        ext_s = [dys[0], dys[-1], zrows[0], zrows[-1]]
        panels = [(np.array(cor_raw), "coronal MIP, band dy -8..+12, raw (0..%d HU)" % vmax, ext_c, "c", 0),
                  (np.array(cor_ves), "coronal MIP, same band, bone removed (150..%d)" % vmax, ext_c, "c", 150),
                  (np.array(sag_r), "RIGHT sagittal MIP |dx| 13..32, bone removed", ext_s, "s", 150),
                  (np.array(sag_l), "LEFT sagittal MIP |dx| 13..32, bone removed", ext_s, "s", 150)]
        for j, (im, ttl, ext, kind, lo) in enumerate(panels):
            a = fig.add_subplot(gs[0, j] if j < 3 else gs[0, 3:5])
            a.imshow(im, cmap="gray", origin="lower", vmin=lo, vmax=vmax, extent=ext, aspect=1.0, interpolation="nearest")
            if kind == "c":
                a.set_xlim(DXW, -DXW)
                a.set_xlabel("dx from column (mm; patient R on image left)")
            else:
                a.set_xlabel("dy from column (mm; anterior to the right)")
            a.axhline(za, color="orange", lw=0.7, ls="--"); a.axhline(za + 40, color="y", lw=0.5, ls=":")
            a.set_title(ttl, fontsize=11); a.set_ylabel("z RAS mm")
        zl = np.linspace(za + 10, min(float(zmm[k_top]), za + 215), 15)
        ext = [org[0], org[0] + nx * sp[0], org[1], org[1] + ny * sp[1]]
        for j, z in enumerate(zl):
            a = fig.add_subplot(gs[1 + j // 5, j % 5])
            k = int(np.clip(np.argmin(np.abs(zmm - z)), 0, nz - 1))
            cxk, cyk = cxs[k], cys[k]
            a.imshow(v[:, :, k].T, cmap="gray", origin="lower", vmin=0, vmax=vmax, extent=ext, interpolation="nearest")
            a.set_xlim(cxk + 40, cxk - 40); a.set_ylim(cyk - 35, cyk + 40)
            a.plot(cxk, cyk, "g+", ms=12)
            for sg in (1, -1):
                a.add_patch(Rectangle((cxk + 12 if sg > 0 else cxk - 34, cyk - 15), 22, 27, fill=False, color="c", lw=0.6, ls=":"))
            a.set_title("z %.0f (arch %+.0f) k %d" % (zmm[k], zmm[k] - za, k), fontsize=10)
            a.tick_params(labelsize=7)
        bc = rec["band_brightness_indicative"]
        fig.suptitle("%s  VA check in the vertebral-column frame (green +: column centre; cyan: |dx| 12-34, dy -15..+12). "
                     "orange: arch top, yellow: arch+40. Indicative band brightness R/L %.2f / %.2f" %
                     (case, bc["R"]["frac_slices_bright"], bc["L"]["frac_slices_bright"]), fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.985]); plt.savefig(os.path.join(d, "qc_va_check.png"), dpi=60); plt.close(fig)
        rec["seconds"] = round(time.time() - t0, 1)
        json.dump(rec, open(os.path.join(d, "qc_va_check.json"), "w"), indent=1)
        print(case, "column top above arch %.0f mm" % rec["column"]["top_above_arch_mm"],
              "bone frac %.2f" % rec["column"]["frac_slices_with_bone"], "bandR", cont["R"], "bandL", cont["L"],
              "side_ok", rec["side_check"]["ok"], "%.0fs" % rec["seconds"], flush=True)
        del v, ras


if __name__ == "__main__":
    main()
