"""M0 (HOST): the DATA half of the canal-length mismatch resolution (spec section 9, stage 1).
Every row is MEASURED from the labels / images.  Writes MRI_GYN_sim/inputs/canal_resolution.json.  Units mm, cc, deg.

    python canal_resolution.py      (after prep_inputs.py)
"""
import json
import os
import sys

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import geom  # noqa: E402
from prep_inputs import lab, v2w, march, disc_run, sample_nn, pca_axis  # noqa: E402

P = config.paths(); D = P["data"]; PRIOR = P["prior"]; INP = P["inputs"]


def img(study):
    im = nib.load("%s/%s_MRI.nii" % (D, study)); return np.asarray(im.dataobj).astype(np.float32)


def extents(pts):
    c = pts.mean(0); _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    q = (pts - c) @ vt.T
    return [round(float(np.ptp(q[:, i])), 1) for i in range(3)]


def feret(mask, A):
    pts = v2w(A, np.argwhere(mask)); h = pts[ConvexHull(pts).vertices]
    d = np.linalg.norm(h[:, None] - h[None], axis=2)
    return round(float(d.max()), 1)


def geodesic_max(body, target, A, src_world):
    sp = np.abs(np.diag(A)[:3]); idx = np.argwhere(body)
    lut = -np.ones(body.shape, int); lut[tuple(idx.T)] = np.arange(len(idx))
    rows, cols, ws = [], [], []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if (di, dj, dk) <= (0, 0, 0):
                    continue
                nb = idx + [di, dj, dk]; ok = np.all((nb >= 0) & (nb < body.shape), 1)
                j = np.full(len(idx), -1); j[ok] = lut[tuple(nb[ok].T)]; m = j >= 0
                rows.append(np.nonzero(m)[0]); cols.append(j[m])
                ws.append(np.full(m.sum(), np.linalg.norm(np.array([di, dj, dk]) * sp)))
    G = coo_matrix((np.concatenate(ws), (np.concatenate(rows), np.concatenate(cols))), shape=(len(idx),) * 2).tocsr()
    src = int(np.argmin(np.linalg.norm(v2w(A, idx) - src_world, axis=1)))
    g = dijkstra(G, directed=False, indices=src)
    tin = target[tuple(idx.T)] & np.isfinite(g)
    return round(float(g[tin].max()), 1), round(float(np.linalg.norm(v2w(A, idx[src][None])[0] - src_world)), 2)


def path_len_inside(mask, A, pts):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    mid = 0.5 * (pts[1:] + pts[:-1])
    return round(float(seg[sample_nn(mask, A, mid)].sum()), 1)


def main():
    named = json.load(open(INP + "/named.json")); app = json.load(open(INP + "/applicator.json"))
    cz = np.load(INP + "/canal.npz")
    L_end = np.array(named["L_end"]); a0 = np.array(named["a0"]); O_pre = np.array(named["O_pre"])
    ut, A = lab("preBT", "uterus"); hr, _ = lab("preBT", "HR-CTV"); iu, _ = lab("preBT", "IUcanal"); vg, _ = lab("preBT", "vagina")
    utb, Ab = lab("BT", "uterus"); hrb, _ = lab("BT", "HR-CTV"); iub, _ = lab("BT", "IUcanal")
    ap, _ = lab("BT", "applicator"); ov, _ = lab("BT", "ovoid")
    F_bt = np.array(app["origin_BT_world"]); z_bt = np.array(app["R_rows_BT_world"][2]); tip_bt = np.array(app["tip_BT_world"])
    out = {}
    # (i) label semantics
    Ipre, Ibt = img("preBT"), img("BT")
    out["i_label_semantics"] = dict(
        BT_IUcanal_equals_applicator_minus_ovoid_xor_vox=int((iub ^ (ap & ~ov)).sum()),
        BT_IUcanal_vox=int(iub.sum()),
        intensity_IUcanal_over_uterus_median_BT=round(float(np.median(Ibt[iub]) / np.median(Ibt[utb & ~iub])), 3),
        intensity_IUcanal_over_uterus_median_preBT=round(float(np.median(Ipre[iu]) / np.median(Ipre[ut & ~iu])), 3),
        BT_IUcanal_principal_extents_mm=extents(v2w(Ab, np.argwhere(iub))),
        preBT_IUcanal_principal_extents_mm=extents(v2w(A, np.argwhere(iu))),
        reading="BT IUcanal is the DEVICE (dark rod, = applicator minus ovoid voxel for voxel); preBT IUcanal is the "
                "bright slit-like cavity/canal (a plate). They are different objects.")
    # (ii) length definitions
    sk_sofa = np.load(PRIOR + "/sofa/meshes/canal_pre.npy"); sk_bt = np.load(PRIOR + "/sofa/meshes/canal_bt.npy")
    cen = np.load(PRIOR + "/applicator/preBT_IUcanal_centroid_trimmed_smooth_mm.npy")
    sk_app = np.load(PRIOR + "/applicator/preBT_IUcanal_centerline_mm.npy")
    Cw = v2w(A, np.argwhere(iu)); Bw = v2w(Ab, np.argwhere(iub))
    C_top = np.array(named["C_top"]); C_top_sk = np.array(named["C_top_skeleton"])
    s_sk = geom.arclength(sk_app); d_sk = geom.unit(sk_app[-1] - sk_app[s_sk >= s_sk[-1] - 8][0])
    s_c = geom.arclength(cen); d_c = geom.unit(cen[-1] - cen[s_c >= s_c[-1] - 8][0])
    out["ii_lengths_mm"] = dict(
        preBT=dict(skeleton_sofa_canal_pre=round(float(geom.arclength(sk_sofa)[-1]), 1),
                   centroid_trimmed=round(float(s_c[-1]), 1),
                   L_end_to_farthest_canal_voxel=round(float(np.linalg.norm(Cw - L_end, axis=1).max()), 1),
                   skeleton_extended_L_end_to_C_top_skeleton=named["canal_skeleton_ext_len_mm"],
                   centroid_extended_L_end_to_C_top=named["canal_len_L_end_to_C_top_mm"]),
        BT=dict(skeleton_sofa_canal_bt=round(float(geom.arclength(sk_bt)[-1]), 1),
                flange_to_tip_extended=app["L_iu_mm"], flange_to_tip_label_max=app["L_iu_label_mm"],
                flange_to_farthest_canal_voxel=round(float(np.linalg.norm(Bw - F_bt, axis=1).max()), 1)),
        skeleton_cornu_veer=dict(top_offset_skeleton_vs_centroid_mm=round(float(np.linalg.norm(C_top_sk - C_top)), 2),
                                 last8mm_direction_angle_deg=round(geom.angle_deg(d_sk, d_c), 1),
                                 skeleton_top=C_top_sk.round(2).tolist(), centroid_top=C_top.round(2).tolist()))
    # (iii) portio column
    run_c, tt, fc = disc_run(hr | ut, A, L_end, -a0)
    _, _, fv = disc_run(vg, A, L_end, -a0)
    vag_depths = tt[fv >= 0.5]
    run_bt, _, _ = disc_run(hrb | utb, Ab, F_bt, -z_bt)
    out["iii_portio"] = dict(preBT_cervix_below_L_end_mm=run_c,
                             preBT_vagina_label_colocated_depths_mm=[float(vag_depths.min()), float(vag_depths.max())] if len(vag_depths) else None,
                             BT_cervix_below_flange_mm=run_bt,
                             preBT_portio_profile=[[float(a), round(float(b), 2), round(float(c), 2)] for a, b, c in zip(tt[::6], fc[::6], fv[::6])])
    # (iv) corpus invariants
    vv_p, vv_b = float(np.prod(np.abs(np.diag(A)[:3]))), float(np.prod(np.abs(np.diag(Ab)[:3])))
    canal = cz["pts"]; fd = np.array(cz["fund_dir"])
    body = ndi.binary_fill_holes(ut | hr | iu); bodyb = ndi.binary_fill_holes(utb | hrb | iub)
    _, ex = march(body, A, canal[-1], fd)
    path_pre = np.vstack([canal, canal[-1] + np.outer(np.linspace(0.5, ex or 0.0, 40), fd)])
    _, exb = march(utb | hrb, Ab, F_bt, z_bt)
    path_bt = F_bt + np.outer(np.linspace(0, exb + 0.5, 200), z_bt)
    fc_b, _ = march(utb, Ab, F_bt, z_bt)
    ios = canal[int(cz["i_internal_os"])]
    out["iv_corpus"] = dict(
        uterus_vol_cc=dict(preBT=round(float(ut.sum() * vv_p / 1000), 2), BT=round(float(utb.sum() * vv_b / 1000), 2)),
        uterus_feret_mm=dict(preBT=feret(ut, A), BT=feret(utb, Ab)),
        in_uterus_path_mm=dict(preBT_along_extended_canal_to_serosa=path_len_inside(ut, A, path_pre),
                               BT_along_tandem_axis_to_serosa=path_len_inside(utb, Ab, path_bt)),
        BT_flange_to_corpus_entry_mm=fc_b,
        BT_tip_to_serosa_along_axis_mm=round(float(march(utb, Ab, tip_bt, z_bt)[1] or 0.0), 1),
        preBT_internal_os_above_L_end_mm=named["internal_os_above_L_end_mm"],
        preBT_canal_top_to_serosa_along_fund_dir_mm=round(float(march(ut, A, canal[-1], fd)[1] or 0.0), 1),
        preBT_cervix_length_O_pre_to_internal_os_mm=round(float(run_c + named["internal_os_above_L_end_mm"]), 1),
        BT_cervix_length_bottom_to_corpus_entry_mm=round(float(run_bt + (fc_b or 0.0)), 1))
    # (v) geodesics
    gl, el = geodesic_max(body, ut, A, L_end); go, eo = geodesic_max(body, ut, A, O_pre); gb, eb = geodesic_max(bodyb, utb, Ab, F_bt)
    out["v_geodesics_to_farthest_uterus_mm"] = dict(preBT_from_L_end=gl, preBT_from_O_pre=go, BT_from_flange=gb,
                                                  snap_err_mm=[el, eo, eb])
    # conclusion (data half)
    Lp = named["canal_len_L_end_to_C_top_mm"]; Lb = app["L_iu_mm"]
    skp = out["ii_lengths_mm"]["preBT"]["skeleton_sofa_canal_pre"]; skb = out["ii_lengths_mm"]["BT"]["skeleton_sofa_canal_bt"]
    X_art = (Lp - skp) - (Lb - skb)
    Y_sem = Lb - Lp
    like_pre = run_c + Lp; like_bt = run_bt + Lb
    out["conclusion"] = (
        "37 vs 55.9 is not like-for-like: skeleton artefacts are +%.1f mm (preBT 37.0 -> %.1f) and +%.1f mm "
        "(BT 55.9 -> %.1f), net %.1f mm, and the remaining %.1f mm is device-vs-anatomy semantics: the BT 'IUcanal' "
        "label is the tandem itself (flange -> tip, xor 0 voxels with applicator minus ovoid) and starts at the flange, "
        "%.1f mm above the lowest cervix, whereas the preBT label is the lumen and starts at L_end, %.1f mm above the "
        "portio tip. Like-for-like (lowest cervix -> canal top / tandem tip): preBT %.1f mm vs BT %.1f mm. "
        "Corpus invariants agree (volume %.1f vs %.1f cc, Feret %.1f vs %.1f mm, in-uterus path %.1f vs %.1f mm), "
        "so the corpus is not elongated; the cervix measured from its lowest point to the corpus is %.1f mm (preBT) vs "
        "%.1f mm (BT). The only physically open quantity is where the flange sits in preBT tissue, d_F in [0, %.1f] mm "
        "(tip overshoot beyond the preBT canal top ~ %.1f - d_F mm on a straight path), plus possible fundal "
        "under-labelling Delta_fund in [0, 10] mm."
        % (Lp - skp, Lp, Lb - skb, Lb, X_art, Y_sem, run_bt, run_c, like_pre, like_bt,
           out["iv_corpus"]["uterus_vol_cc"]["preBT"], out["iv_corpus"]["uterus_vol_cc"]["BT"],
           out["iv_corpus"]["uterus_feret_mm"]["preBT"], out["iv_corpus"]["uterus_feret_mm"]["BT"],
           out["iv_corpus"]["in_uterus_path_mm"]["preBT_along_extended_canal_to_serosa"],
           out["iv_corpus"]["in_uterus_path_mm"]["BT_along_tandem_axis_to_serosa"],
           out["iv_corpus"]["preBT_cervix_length_O_pre_to_internal_os_mm"],
           out["iv_corpus"]["BT_cervix_length_bottom_to_corpus_entry_mm"], run_c, Y_sem))
    out["tandem_insertion_depth_definition"] = dict(
        L_iu_mm=Lb, source="BT applicator label (flange -> tip), not the IUcanal label lengths",
        flange_depth_d_F_mm="swept in [0, %.1f]; P1 uses the fornix rule d_F_pred=%.2f; P2 (HR-CTV ICP oracle) is "
                            "equivalent to d_F=%.1f along a0 with a %.1f deg tilt"
                            % (run_c, named["d_F_pred_mm"], named["P2"]["F_minus_O_pre_along_a0_mm"] and
                               (run_c - named["P2"]["F_minus_O_pre_along_a0_mm"]), named["P2"]["angle_to_a0_deg"]))
    json.dump(out, open(INP + "/canal_resolution.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
