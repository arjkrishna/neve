"""HOST (python 3.13): the dataset's BT (post-insertion) segmentations in a hybrid run's frame, for the overlay
views of label_views.py (--overlay), plus the numbers that say what is misaligned and why.

    python hybrid/bt_overlay.py prep --tag G32            # -> figs/overlay_prep/<tag>_PELVIS/{<name>.obj, stats.json}

Frame: the evaluator's PELVIS frame (validation/alignment.json, peri-organ MI registration; no organ label enters
its fit), y_pre = R x_BT + t.  So the overlays compare the simulated organs with the real ones relative to the
pelvis, and include the pose rule's device error.  Surfaces: marching cubes on each lightly smoothed label mask,
mapped voxel -> BT world -> preBT world.  Per structure, stats.json records the label volumes before and after
insertion, and three centroids: preBT label, BT label (mapped), and the simulated body at rest and at the end.
Their differences separate "the model moved it the wrong way" from "the organ changed volume or shape, which the
model cannot do".  Patient-derived output: local only."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import geom  # noqa: E402
import eval_hybrid as EH  # noqa: E402
import evaluate as ev  # noqa: E402

DATA = EH.DATA
STRUCT = {                       # overlay name: (model body, BT mask builder, preBT mask builder)
    "cervix": "cervix", "corpus": "corpus", "vagina": "vagina", "bladder": "bladder", "rectum": "rectum",
    "sigmoid": "sigmoid"}


def pre_label(n):
    import nibabel as nib
    im = nib.load("%s/preBT_MRI_label_%s.nii" % (DATA, n))
    return np.asarray(im.dataobj) > 0, im.affine.copy()


def mask_surface(mask, aff, sd=0.8, max_tris=16000):
    """Closed surface of a boolean mask (world mm, via the affine): Gaussian-smoothed marching cubes, decimated."""
    import vtk
    from vtk.util import numpy_support as ns
    m = np.pad(mask.astype(np.float32), 2)
    img = vtk.vtkImageData()
    img.SetDimensions(*m.shape)
    img.GetPointData().SetScalars(ns.numpy_to_vtk(m.ravel(order="F"), deep=1))
    g = vtk.vtkImageGaussianSmooth()
    g.SetInputData(img)
    g.SetStandardDeviations(sd, sd, sd)
    g.Update()
    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputConnection(g.GetOutputPort())
    fe.SetValue(0, 0.5)
    fe.Update()
    pd = fe.GetOutput()
    if pd.GetNumberOfPolys() > max_tris:
        dec = vtk.vtkQuadricDecimation()
        dec.SetInputData(pd)
        dec.SetTargetReduction(1.0 - max_tris / float(pd.GetNumberOfPolys()))
        dec.Update()
        pd = dec.GetOutput()
    V = ns.vtk_to_numpy(pd.GetPoints().GetData()).astype(float) - 2.0
    F = ns.vtk_to_numpy(pd.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    return V @ aff[:3, :3].T + aff[:3, 3], F


def mask_points(mask, aff):
    ijk = np.argwhere(mask)
    return ijk @ aff[:3, :3].T + aff[:3, 3]


def solid_centroid(V, F):
    """Volume centroid of a closed triangulated surface."""
    V = np.asarray(V, float)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    v = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    return ((a + b + c) / 4.0 * v[:, None]).sum(0) / v.sum(), float(abs(v.sum())) / 1000.0


def centreline(P, step=2.0):
    """Slab-centroid polyline of a thin structure's points along their principal axis."""
    c = P.mean(0)
    w, U = np.linalg.eigh(np.cov((P - c).T))
    ax = U[:, -1]
    s = (P - c) @ ax
    out = []
    for lo in np.arange(s.min(), s.max(), step):
        m = (s >= lo) & (s < lo + step)
        if m.sum() >= 3:
            out.append(P[m].mean(0))
    return np.array(out)


def prep(tag):
    cx = EH.Ctx()
    fmap, R, t = cx.pelvis_map()
    to_pre = lambda X: np.asarray(X, float) @ R.T + t           # BT world -> preBT world (y = R x + t)
    bt = cx.bt
    lab = dict(bt.lab)
    lab["IUcanal"] = ev.bt_label("IUcanal")[0]
    for n in EH.OARS:
        lab.setdefault(n, cx.bt.lab[n])
    btm = {"cervix": lab["HR-CTV"] & ~lab["uterus"], "corpus": lab["uterus"], "vagina": lab["vagina"],
           "bladder": lab["bladder"], "rectum": lab["rectum"], "sigmoid": lab["sigmoid"],
           "applicator": lab["applicator"], "ovoid": lab["ovoid"], "IUcanal": lab["IUcanal"]}
    pre = {n: pre_label(n) for n in ("HR-CTV", "uterus", "vagina", "bladder", "rectum", "sigmoid", "IUcanal")}
    pre_aff = pre["uterus"][1]
    prem = {"cervix": pre["HR-CTV"][0] & ~pre["uterus"][0], "corpus": pre["uterus"][0], "vagina": pre["vagina"][0],
            "bladder": pre["bladder"][0], "rectum": pre["rectum"][0], "sigmoid": pre["sigmoid"][0]}
    vv_pre = float(abs(np.linalg.det(pre_aff[:3, :3])))
    out = "%s/overlay_prep/%s_PELVIS" % (EH.FIGS, tag)
    os.makedirs(out, exist_ok=True)
    # ---- surfaces of the BT structures in the preBT (model) frame
    for n, m in btm.items():
        V, F = mask_surface(m, bt.aff, sd=0.6 if n in ("applicator", "IUcanal") else 0.8)
        geom.write_obj("%s/%s.obj" % (out, n), to_pre(V), F, header="BT label %s in preBT world (PELVIS frame); local only" % n)
    cl_bt = centreline(to_pre(mask_points(btm["IUcanal"], bt.aff)))
    cl_pre = centreline(mask_points(pre["IUcanal"][0], pre_aff))
    np.save(out + "/IUcanal_centreline_bt.npy", cl_bt)
    np.save(out + "/IUcanal_centreline_pre.npy", cl_pre)
    # ---- the numbers
    rd = "%s/%s" % (EH.RUNS, tag)
    cfg = json.load(open(rd + "/cfg.json"))
    mdirs = EH.run_mesh_dirs(tag)
    stats = dict(tag=tag, frame="PELVIS (validation/alignment.json frames.%s): y_pre = R x_BT + t" % EH.PELVIS_KEY,
                 axes="preBT world RAS: +x = patient right, +y = anterior, +z = superior", structures={})
    model_pts = {}
    for n, body in STRUCT.items():
        Vr, Fr = geom.read_obj(mdirs[body] + "/surface.obj")
        Vf, Ff = geom.read_obj("%s/final/%s.obj" % (rd, body))
        Vr, Vf = np.asarray(Vr, float), np.asarray(Vf, float)
        if body == "vagina" and cfg.get("vagina_model") == "wall":
            meta = json.load(open(mdirs["vagina"] + "/meta.json"))
            Vr_s, Fr_s = EH.wall_outer_solid(Vr, Fr, meta)
            Vf_s, Ff_s = EH.wall_outer_solid(Vf, Ff, meta)
        else:
            Vr_s, Fr_s, Vf_s, Ff_s = Vr, np.asarray(Fr, int), Vf, np.asarray(Ff, int)
        def vox(V_, F_):
            m_ = ev.voxelize(fmap(V_), np.asarray(F_, int), bt.shape, bt.aff)
            P_ = to_pre(mask_points(m_, bt.aff))
            return P_.mean(0), len(P_) * bt.vv / 1000.0, P_
        c_rest, vol_rest, _ = vox(Vr_s, Fr_s)
        c_fin, vol_fin, Pm = vox(Vf_s, Ff_s)
        model_pts[n] = Pm
        Pb = to_pre(mask_points(btm[n], bt.aff))
        Pp = mask_points(prem[n], pre_aff)
        c_bt, c_pre = Pb.mean(0), Pp.mean(0)
        true_d, model_d = c_bt - c_pre, c_fin - c_rest
        cosang = float(true_d @ model_d / max(1e-9, np.linalg.norm(true_d) * np.linalg.norm(model_d)))
        stats["structures"][n] = dict(
            vol_pre_label_cc=round(len(Pp) * vv_pre / 1000.0, 2), vol_bt_label_cc=round(len(Pb) * bt.vv / 1000.0, 2),
            vol_model_rest_cc=round(vol_rest, 2), vol_model_final_cc=round(vol_fin, 2),
            centroid_pre_label=c_pre.round(2).tolist(), centroid_bt_label=c_bt.round(2).tolist(),
            centroid_model_rest=c_rest.round(2).tolist(), centroid_model_final=c_fin.round(2).tolist(),
            true_displacement_mm=true_d.round(2).tolist(), model_displacement_mm=model_d.round(2).tolist(),
            true_displacement_len=round(float(np.linalg.norm(true_d)), 2),
            model_displacement_len=round(float(np.linalg.norm(model_d)), 2),
            direction_agreement_deg=round(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))), 1),
            residual_final_minus_bt_mm=(c_fin - c_bt).round(2).tolist(),
            residual_len=round(float(np.linalg.norm(c_fin - c_bt)), 2))
    # device: the real applicator / ovoid labels vs the model device
    dj = json.load(open(rd + "/device_final.json"))
    stats["device"] = dict(ovoid_label_centroid=to_pre(mask_points(btm["ovoid"], bt.aff)).mean(0).round(2).tolist(),
                           model_ovoid_centres=np.asarray(dj["ovoid_centres_mm"], float).round(2).tolist(),
                           model_flange=dj["flange_mm"], model_tube_axis=dj["tube_axis"],
                           bt_flange_pre=to_pre(bt.F[None])[0].round(2).tolist(),
                           bt_axis_pre=(bt.a @ R.T).round(5).tolist())
    # ---- the measurements behind the likely causes
    F_m, a_m = np.asarray(dj["flange_mm"], float), geom.unit(dj["tube_axis"])
    Fbt, abt = to_pre(bt.F[None])[0], geom.unit(bt.a @ R.T)
    th = np.radians(float(json.load(open(EH.APPD + "/applicator.json"))["params"]["angle_deg"]["value"]))
    R_ov = np.array([dj["ovoid_x_app"], dj["ovoid_y_app"], dj["ovoid_axis"]], float) if "ovoid_axis" in dj else \
        np.array([dj["x_app"], dj["y_app"], dj["tube_axis"]], float)
    up_v = -(np.array([0.0, np.sin(th), -np.cos(th)]) @ R_ov)
    cz = np.load(EH.INP + "/canal.npz")
    cp, ios = np.asarray(cz["pts"], float), int(cz["i_internal_os"])
    a_pre = geom.unit(cp[ios] - cp[0])

    def lr_split(P, O_, d, vox_cc):
        q = P - O_
        x = (q - np.outer(q @ d, d))[:, 0]
        return round(float((x > 0).sum() * vox_cc), 1), round(float((x <= 0).sum() * vox_cc), 1), float(x.mean())
    Hp = mask_points(prem["cervix"], pre_aff)
    Hb = to_pre(mask_points(btm["cervix"], bt.aff))
    rp, lp, xp = lr_split(Hp, cp[0], a_pre, vv_pre / 1000.0)
    rb, lb, xb = lr_split(Hb, Fbt, abt, bt.vv / 1000.0)
    Xc = np.asarray(__import__("vagina_wall").read_vtk_legacy(mdirs["cervix"] + "/tets.vtk")[0], float) + \
        np.load("%s/final/cervix_u.npy" % rd)
    xm = float(((Xc - F_m) - np.outer((Xc - F_m) @ a_m, a_m))[:, 0].mean())
    Pc = mask_points(pre["IUcanal"][0], pre_aff)
    lab_start = float(((Pc - cp[0]) @ a_pre).min())
    q = cl_bt - Fbt
    bt_canal_dev = float(np.linalg.norm(q - np.outer(q @ abt, abt), axis=1).max())
    widths = {}
    for hh in (-60.0, -40.0, -20.0, 0.0):
        row = {}
        for nm, P_ in (("BT", to_pre(mask_points(btm["vagina"], bt.aff))), ("model", model_pts["vagina"])):
            hv = (P_ - F_m) @ up_v
            m_ = np.abs(hv - hh) < 2.5
            if m_.sum() > 30:
                Q = P_[m_]
                row[nm] = [round(float(np.percentile(Q[:, 0], 97) - np.percentile(Q[:, 0], 3)), 1),
                           round(float(np.percentile(Q[:, 1], 97) - np.percentile(Q[:, 1], 3)), 1)]
        widths["%+.0f" % hh] = row
    stats["measurements"] = dict(
        hrctv_lr_split_cc=dict(pre_right=rp, pre_left=lp, bt_right=rb, bt_left=lb,
                               note="HR-CTV outside the uterus, right / left of the pre-insertion canal line (pre) and of the real tandem (BT)"),
        hrctv_mean_x_offset_mm=dict(pre_vs_canal=round(xp, 1), bt_vs_tandem=round(xb, 1), model_vs_tandem=round(xm, 1),
                                    note="+ = patient right"),
        canal_label_starts_above_os_mm=round(lab_start, 1),
        bt_canal_vs_real_tandem_max_mm=round(bt_canal_dev, 1),
        vagina_width_lr_ap_mm=widths,
        ring_offset_mm=(to_pre(mask_points(btm["ovoid"], bt.aff)).mean(0) - np.asarray(dj["ovoid_centres_mm"], float).mean(0)).round(1).tolist())
    ev_m = "%s/%s/metrics.json" % (EH.EVALD, tag)
    if os.path.exists(ev_m):
        m = json.load(open(ev_m))["frames"]["PELVIS"]
        stats["scores_PELVIS"] = {s: dict(dice=round(v["dice"], 3), msd=round(v["msd"], 2), hd95=round(v["hd95"], 2))
                                  for s, v in m["structures"].items()}
        stats["pose_rule_error"] = m.get("pose_rule_error")
    json.dump(stats, open(out + "/stats.json", "w"), indent=1)
    print("measurements:", json.dumps(stats["measurements"], indent=1))
    print("wrote", out)
    for n, s in stats["structures"].items():
        print("  %-8s vol pre %6.1f -> BT %6.1f cc (x%.2f) | model %6.1f -> %6.1f | true move %5.1f mm, model move %5.1f mm, "
              "angle %5.1f deg | model centroid - BT %5.1f mm %s" % (
                  n, s["vol_pre_label_cc"], s["vol_bt_label_cc"], s["vol_bt_label_cc"] / max(1e-9, s["vol_pre_label_cc"]),
                  s["vol_model_rest_cc"], s["vol_model_final_cc"], s["true_displacement_len"], s["model_displacement_len"],
                  s["direction_agreement_deg"], s["residual_len"], np.round(s["residual_final_minus_bt_mm"], 1).tolist()))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prep"])
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    prep(a.tag)


if __name__ == "__main__":
    main()
