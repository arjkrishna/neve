"""M0 (HOST, python 3.13 with nibabel/scipy/skimage/vtk): preBT surfaces, label ROI, named geometry,
applicator model and poses  ->  MRI_GYN_sim/inputs/.

Frame: preBT world RAS mm (nibabel affine, diagonal -1.125, -1.125, 1.6).  The BT study lives in its
own world frame; BT quantities are converted to the applicator frame (device facts) or mapped by an
explicit rigid transform (P2).  Nothing in the data directory is written.

    python prep_inputs.py            # writes inputs/*, logs/prep_inputs.json
Every named point is cross-checked against the expected MEASURED value from the earlier workflow
(tolerance 1 mm / 2 deg); deviations are logged with an explanation, and the numeric gate result
is written to inputs/named.json["checks"].  The reference values are patient-derived and therefore
live OUTSIDE the repo, in <APPSIM_PRIOR>/expected_checks.json; without that file every check is
recorded as skipped and the run proceeds unchanged.
"""
import json
import os
import sys
import time

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage import measure
import vtk
from vtk.util import numpy_support as ns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import geom  # noqa: E402

P = config.paths()
D = P["data"]
INP = P["inputs"]
PRIOR = P["prior"]
os.makedirs(INP, exist_ok=True); os.makedirs(P["logs"], exist_ok=True)
CHECKS = []
try:                                   # patient-derived reference values: outside the repo (see the docstring)
    with open(PRIOR + "/expected_checks.json") as _fh:
        EXP = json.load(_fh)
except (OSError, ValueError):
    EXP = {}


def check(name, got, exp, tol, unit="mm", note=""):
    got = np.asarray(got, float)
    if exp is None:
        CHECKS.append(dict(name=name, got=np.round(got, 3).tolist(), expected=None, err=None, tol=tol, unit=unit,
                           ok=None, note=(note + "  " if note else "") + "SKIPPED: no reference in expected_checks.json"))
        print("CHECK %-38s SKIP (no reference value)" % name, flush=True)
        return None
    exp = np.asarray(exp, float)
    if unit == "deg":
        err = geom.angle_deg(got, exp)
    else:
        err = float(np.linalg.norm(got - exp))
    ok = bool(err <= tol)
    CHECKS.append(dict(name=name, got=np.round(got, 3).tolist(), expected=exp.tolist(), err=round(err, 3),
                       tol=tol, unit=unit, ok=ok, note=note))
    print("CHECK %-38s %s err=%.3f %s (tol %g) %s" % (name, "OK  " if ok else "FAIL", err, unit, tol, note), flush=True)
    return ok


def check_scalar(name, got, key, tol, unit="mm", note=""):
    """Scalar check against EXP[key] (absolute difference); recorded as skipped when either side is missing."""
    exp = EXP.get(key)
    if got is None or exp is None:
        CHECKS.append(dict(name=name, got=None if got is None else round(float(got), 3), expected=exp, err=None,
                           tol=tol, unit=unit, ok=None,
                           note=(note + "  " if note else "") + ("SKIPPED: no value" if got is None else
                                                                 "SKIPPED: no reference in expected_checks.json")))
        print("CHECK %-38s SKIP" % name, flush=True)
        return None
    err = abs(float(got) - float(exp))
    ok = bool(err <= tol)
    CHECKS.append(dict(name=name, got=round(float(got), 3), expected=exp, err=round(err, 3), tol=tol, unit=unit,
                       ok=ok, note=note))
    print("CHECK %-38s %s err=%.3f %s (tol %g) %s" % (name, "OK  " if ok else "FAIL", err, unit, tol, note), flush=True)
    return ok


def lab(study, name):
    im = nib.load("%s/%s_MRI_label_%s.nii" % (D, study, name))
    return np.asarray(im.dataobj) > 0, im.affine.copy()


def v2w(A, ijk):
    return np.atleast_2d(ijk).astype(float) @ A[:3, :3].T + A[:3, 3]


def w2v(A, xyz):
    return (np.atleast_2d(xyz) - A[:3, 3]) @ np.linalg.inv(A[:3, :3]).T


def sample_nn(mask, A, X):
    ijk = np.rint(w2v(A, X)).astype(int)
    ok = np.all((ijk >= 0) & (ijk < mask.shape), 1)
    v = np.zeros(len(ijk), bool); v[ok] = mask[tuple(ijk[ok].T)]
    return v


def sample_lin(mask, A, X):
    v = w2v(A, X)
    return ndi.map_coordinates(mask.astype(np.float32), v.T, order=1, mode="constant") >= 0.5


def largest(m):
    labm, k = ndi.label(m, np.ones((3, 3, 3)))
    if k <= 1:
        return m
    s = ndi.sum(m, labm, range(1, k + 1))
    return labm == (int(np.argmax(s)) + 1)


def pca_axis(pts):
    c = pts.mean(0); _, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt[0], s / np.sqrt(len(pts))


# ----------------------------------------------------------------------------- surfaces
def surface(mask, aff, n_tris=4000):
    """marching cubes (world mm) -> windowed-sinc smoothing -> quadric decimation; as sofa/build_meshes.py."""
    sp = np.sqrt((aff[:3, :3] ** 2).sum(0))
    m = np.pad(mask, 2).astype(np.uint8)
    v, f, _, _ = measure.marching_cubes(m, 0.5, spacing=tuple(sp))
    idx = v / sp - 2.0
    w = v2w(aff, idx)
    if np.linalg.det(aff[:3, :3]) < 0:
        f = f[:, ::-1]
    pts = vtk.vtkPoints(); pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(w), deep=1))
    cells = vtk.vtkCellArray()
    cells.SetCells(len(f), ns.numpy_to_vtkIdTypeArray(np.c_[np.full(len(f), 3), f].astype(np.int64).ravel(), deep=1))
    poly = vtk.vtkPolyData(); poly.SetPoints(pts); poly.SetPolys(cells)
    cl = vtk.vtkCleanPolyData(); cl.SetInputData(poly); cl.Update()
    sm = vtk.vtkWindowedSincPolyDataFilter(); sm.SetInputData(cl.GetOutput())
    sm.SetNumberOfIterations(20); sm.SetPassBand(0.1); sm.NonManifoldSmoothingOn(); sm.NormalizeCoordinatesOn(); sm.Update()
    n0 = sm.GetOutput().GetNumberOfCells()
    dec = vtk.vtkQuadricDecimation(); dec.SetInputData(sm.GetOutput())
    dec.SetTargetReduction(max(0.0, 1.0 - n_tris / n0)); dec.VolumePreservationOn(); dec.Update()
    tri = vtk.vtkTriangleFilter(); tri.SetInputData(dec.GetOutput()); tri.Update()
    cl2 = vtk.vtkCleanPolyData(); cl2.SetInputData(tri.GetOutput()); cl2.Update()
    nrm = vtk.vtkPolyDataNormals(); nrm.SetInputData(cl2.GetOutput()); nrm.AutoOrientNormalsOn()
    nrm.ConsistencyOn(); nrm.SplittingOff(); nrm.ComputeCellNormalsOn(); nrm.Update()
    out = nrm.GetOutput()
    Pt = ns.vtk_to_numpy(out.GetPoints().GetData()).astype(float)
    F = ns.vtk_to_numpy(out.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    fe = vtk.vtkFeatureEdges(); fe.SetInputData(out); fe.BoundaryEdgesOn(); fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff(); fe.NonManifoldEdgesOn(); fe.Update()
    mp = vtk.vtkMassProperties(); mp.SetInputData(out); mp.Update()
    vol_mesh = mp.GetVolume() / 1000.0; vol_lab = mask.sum() * np.prod(sp) / 1000.0
    st = dict(tris=int(len(F)), verts=int(len(Pt)), open_or_nonmanifold_edges=int(fe.GetOutput().GetNumberOfCells()),
              mesh_vol_cc=round(vol_mesh, 3), label_vol_cc=round(vol_lab, 3),
              vol_err_pct=round(100 * (vol_mesh - vol_lab) / vol_lab, 2),
              signed_vol_cc=round(geom.mesh_volume(Pt, F) / 1000.0, 3))
    st["gate_ok"] = bool(st["open_or_nonmanifold_edges"] == 0 and abs(st["vol_err_pct"]) < 2.0 and st["signed_vol_cc"] > 0)
    return Pt, F, st


def march(mask, A, p, d, step=0.1, maxd=80):
    """distance from p along d: (first inside, last of the first contiguous inside run)."""
    t = np.arange(0, maxd, step); v = sample_nn(mask, A, p + t[:, None] * d)
    inside = np.nonzero(v)[0]
    if len(inside) == 0:
        return None, None
    j = inside[0]
    while j + 1 < len(v) and v[j + 1]:
        j += 1
    return float(t[inside[0]]), float(t[j])


def extend_ends(R, mask, A, win=8.0):
    """Extend a polyline's two ends along their local directions to the mask boundary (s05 method)."""
    s = geom.arclength(R)
    dtip = geom.unit(R[-1] - R[s >= s[-1] - win][0]); dbot = geom.unit(R[0] - R[s <= win][-1])
    _, e_tip = march(mask, A, R[-1], dtip); _, e_bot = march(mask, A, R[0], dbot)
    e_tip = e_tip or 0.0; e_bot = e_bot or 0.0
    return R[0] + e_bot * dbot, R[-1] + e_tip * dtip, e_bot, e_tip


def disc_run(mask, A, p0, d, depth=45.0, r=3.0):
    """How far (mm) mask continues from p0 along d (7-ray 3-mm disc, frac>=0.5), as check_os.py."""
    t = np.arange(0, depth, 0.5)
    e1 = geom.unit(np.cross(d, [1, 0, 0])); e2 = np.cross(d, e1)
    offs = [np.zeros(3)] + [r * (np.cos(a) * e1 + np.sin(a) * e2) for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
    frac = np.mean([sample_lin(mask, A, p0[None] + np.outer(t, d) + o) for o in offs], 0)
    run = 0.0
    for tt, fr in zip(t, frac):
        if fr >= 0.5:
            run = float(tt)
        else:
            break
    return run, t, frac


def main():
    t0 = time.time()
    log = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"))
    # ------------------------------------------------------------------ preBT labels
    ut, A = lab("preBT", "uterus"); hr, _ = lab("preBT", "HR-CTV"); iu, _ = lab("preBT", "IUcanal"); vg, _ = lab("preBT", "vagina")
    sp = np.abs(np.diag(A)[:3]); vv = float(np.prod(sp))
    log["preBT_label_vol_cc"] = {n: round(float(m.sum() * vv / 1000), 2) for n, m in
                                 (("uterus", ut), ("HR-CTV", hr), ("IUcanal", iu), ("vagina", vg))}
    # 3.1 surfaces
    body = largest(ndi.binary_fill_holes(ut | hr | iu))
    surfs = {}
    for name, m, ntri in (("pre_body", body, 4000), ("pre_uterus", largest(ut), 4000),
                          ("pre_hrctv", largest(hr), 4000), ("pre_vagina", largest(vg), 3000)):
        V, F, st = surface(m, A, ntri)
        geom.write_obj(os.path.join(INP, name + ".obj"), V, F,
                       header="%s: preBT world RAS mm; derived from labels (local only)" % name)
        surfs[name] = st
        print(name, json.dumps(st), flush=True)
    log["surfaces"] = surfs
    # 3.2 ROI
    region = np.zeros(ut.shape, np.uint8)
    region[vg] = 4; region[ut] = 1; region[hr & ~ut] = 2; region[iu] = 3
    nz = np.argwhere(region > 0); mg = np.ceil(15.0 / sp).astype(int)
    lo = np.maximum(nz.min(0) - mg, 0); hi = np.minimum(nz.max(0) + mg + 1, region.shape)
    roi = region[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    Aroi = A.copy(); Aroi[:3, 3] = v2w(A, lo[None])[0]
    np.savez_compressed(os.path.join(INP, "pre_labels_roi.npz"), region=roi, affine=Aroi,
                        codes=np.array(["0 background", "1 uterus", "2 HR-CTV-not-uterus", "3 IUcanal", "4 vagina"]))
    log["roi"] = dict(shape=list(roi.shape), lo_ijk=lo.tolist())

    # ------------------------------------------------------------------ 3.3 named geometry
    sk = np.load(PRIOR + "/applicator/preBT_IUcanal_centerline_mm.npy")        # skeleton (44.5 mm)
    L_end, C_top_sk, e_bot, e_tip = extend_ends(sk, iu, A)
    check("L_end", L_end, EXP.get("L_end"), 1.0)
    check("C_top(skeleton-extended)", C_top_sk, EXP.get("C_top"), 1.0)
    L_sk_ext = float(geom.arclength(sk)[-1] + e_bot + e_tip)
    Cw = v2w(A, np.argwhere(iu))
    ch = geom.unit(C_top_sk - L_end)
    lo15 = Cw[(Cw - L_end) @ ch < 15.0]
    _, a0, _ = pca_axis(lo15); a0 = a0 * np.sign(a0 @ ch)
    check("a0 (lower-15 mm chord slab PCA)", a0, EXP.get("a0"), 2.0, "deg")
    near15 = Cw[np.linalg.norm(Cw - L_end, axis=1) < 15.0]
    _, a0_ball, _ = pca_axis(near15); a0_ball = a0_ball * np.sign(a0_ball @ ch)
    variants = dict(a0_ball15=a0_ball.round(4).tolist(),
                    a0_ball15_angle_to_a0_deg=round(geom.angle_deg(a0_ball, a0), 2))
    for key, name_, ang_ in (("a0_design_contact", "design_contact_lower_canal", "design_contact_angle_to_a0_deg"),
                             ("a0_other_variant", "other_variant", "other_variant_angle_to_a0_deg")):
        if key in EXP:                                   # the earlier workflow's axis variants (reference file only)
            variants[name_] = list(EXP[key])
            variants[ang_] = round(geom.angle_deg(EXP[key], a0), 2)
    # canal polyline: centroid line (39.0 mm) extended to the label ends
    cen = np.load(PRIOR + "/applicator/preBT_IUcanal_centroid_trimmed_smooth_mm.npy")
    if np.linalg.norm(cen[0] - L_end) > np.linalg.norm(cen[-1] - L_end):
        cen = cen[::-1]
    c_bot, c_top, ce_bot, ce_tip = extend_ends(cen, iu, A)
    L_cen = float(geom.arclength(cen)[-1])
    if ce_tip == 0.0:
        # The trimmed per-slice centroid line ends OUTSIDE the curved/plate-like label (a slice centroid of a
        # V-shaped cavity section need not be inside it), so the mask march finds nothing.  "Extend to the
        # label end" is then done slice-wise: along the last-8 mm direction up to the label's top axial slice.
        s_c0 = geom.arclength(cen); d_end = geom.unit(cen[-1] - cen[s_c0 >= s_c0[-1] - 8][0])
        z_top = v2w(A, [[0, 0, np.argwhere(iu)[:, 2].max()]])[0][2]
        ce_tip = float(max(0.0, (z_top - cen[-1][2]) / d_end[2]))
        c_top = cen[-1] + ce_tip * d_end
    canal_lab = np.vstack([L_end[None], cen, c_top[None]]) if np.linalg.norm(cen[0] - L_end) > 0.3 else np.vstack([cen, c_top[None]])
    L_canal = float(geom.arclength(canal_lab)[-1])
    check("C_top (centroid line extended)", c_top, EXP.get("C_top"), 1.0,
          note="spec C_top comes from the skeleton extension; centroid-line end extended along its own direction")
    check_scalar("canal length L_end->C_top (centroid, extended)", L_canal, "canal_length_mm", 1.0)
    # O_pre: walk from L_end along -a0 until (HR-CTV|uterus) ends
    cx = hr | ut
    run, tt, frac = disc_run(cx, A, L_end, -a0)
    O_pre = L_end - run * a0
    check_scalar("O_pre depth below L_end along -a0", run, "portio_depth_mm", 1.5,
                 note="check_os.json measured it from the skeleton start along the skeleton lower direction")
    sk0 = np.load(PRIOR + "/sofa/meshes/canal_pre.npy")
    s_sk0 = geom.arclength(sk0); q = sk0[s_sk0 <= 12]; _, dpre, _ = pca_axis(q); dpre = dpre * np.sign(dpre @ (q[-1] - q[0]))
    run_sk, _, _ = disc_run(cx, A, sk0[0], -dpre)
    O_sk = sk0[0] - run_sk * dpre
    print("O_pre from L_end: depth %.1f mm -> %s ; replicate check_os (skeleton start): depth %.1f -> %s ; |dO| %.2f mm"
          % (run, np.round(O_pre, 2), run_sk, np.round(O_sk, 2), np.linalg.norm(O_pre - O_sk)), flush=True)
    # extended canal: O_pre -> L_end straight, then canal_lab; 1.0 mm resampling
    ext = np.vstack([O_pre[None], canal_lab])
    canal = geom.resample(ext, 1.0)
    s0 = geom.arclength(canal)
    ins_ut = sample_nn(ut, A, canal)
    i_Lend = int(np.argmin(np.linalg.norm(canal - L_end, axis=1)))
    cand = np.nonzero(ins_ut & (np.arange(len(canal)) >= i_Lend))[0]
    i_ios = int(cand[0]) if len(cand) else -1
    ios_above_Lend = float(s0[i_ios] - s0[i_Lend]) if i_ios >= 0 else None
    check_scalar("internal os above L_end", ios_above_Lend, "internal_os_above_L_end_mm", 1.0)
    # fundal extension direction (last 10 mm) and the max allowed extension (>= 3 mm wall remaining)
    s_c = geom.arclength(canal)
    d_f = geom.unit(canal[-1] - canal[s_c >= s_c[-1] - 10][0])
    _, body_exit = march(body, A, canal[-1], d_f)
    delta_max = max(0.0, (body_exit or 0.0) - 3.0)
    # h_apex
    Vw = v2w(A, np.argwhere(vg)); r_ = Vw - L_end; h_ = r_ @ a0
    rad = np.linalg.norm(r_ - np.outer(h_, a0), axis=1)
    h_apex = float(h_[rad < 20.0].max())
    check_scalar("h_apex", h_apex, "h_apex_mm", 1.0)
    np.savez(os.path.join(INP, "canal.npz"), pts=canal, s0=s0, i_L_end=i_Lend, i_internal_os=i_ios,
             fund_dir=d_f, delta_fund_max_mm=delta_max, a0=a0, L_end=L_end, O_pre=O_pre)

    # ------------------------------------------------------------------ 3.4 applicator (BT)
    ap, Ab = lab("BT", "applicator"); ov, _ = lab("BT", "ovoid"); iub, _ = lab("BT", "IUcanal"); utb, _ = lab("BT", "uterus")
    osj = json.load(open(PRIOR + "/proto/design_fast/ovoid_spheres.json"))
    # BT flange and tandem axis: re-derived from the BT IUcanal label here; when the reference file carries the
    # earlier workflow's values those are what the model uses, and the re-derivation is checked against them
    skb = np.load(PRIOR + "/applicator/BT_IUcanal_centerline_mm.npy")
    Fb2, Tb2, _, _ = extend_ends(skb, iub, Ab)
    Bw = v2w(Ab, np.argwhere(iub)); _, zb2, _ = pca_axis(Bw); zb2 = zb2 * np.sign(zb2 @ (Tb2 - Fb2))
    F_bt = np.asarray(EXP["F_bt"], float) if "F_bt" in EXP else np.asarray(Fb2, float)
    z_bt = geom.unit(EXP["z_bt"]) if "z_bt" in EXP else geom.unit(zb2)
    R_app = geom.frame_from(z_bt, [1, 0, 0])          # rows x, y, z (BT world)
    check("frame x vs ovoid_spheres.json", R_app[0], osj["frame"]["x"], 1.0, "deg")
    check("frame y vs ovoid_spheres.json", R_app[1], osj["frame"]["y"], 1.0, "deg")
    check("flange re-derived (BT IUcanal inferior end)", Fb2, EXP.get("F_bt"), 0.5)
    check("axis re-derived (BT IUcanal PCA)", zb2, EXP.get("z_bt"), 1.0, "deg")
    # tandem length and tip from the APPLICATOR label (rod = applicator minus ovoid)
    Aw = v2w(Ab, np.argwhere(ap & ~ov)); ra = Aw - F_bt; za = ra @ z_bt
    rada = np.linalg.norm(ra - np.outer(za, z_bt), axis=1)
    rod = rada < 4.0
    L_iu_label = float(za[rod].max())
    tip_bt = F_bt + L_iu_label * z_bt
    check_scalar("L_iu from applicator label (flange->tip)", L_iu_label, "L_iu_label_mm", 1.0)
    # label semantics: BT IUcanal vs applicator minus ovoid (voxel-for-voxel)
    sem = dict(IUcanal_vox=int(iub.sum()), app_minus_ovoid_vox=int((ap & ~ov).sum()),
               xor_vox=int((iub ^ (ap & ~ov)).sum()))
    Ow = v2w(Ab, np.argwhere(ov)); zo = (Ow - F_bt) @ z_bt
    h_ovtop = float(np.percentile(zo, 99))
    check_scalar("h_ovtop (p99 z_app of ovoid voxels)", h_ovtop, "h_ovtop_mm", 1.0)
    applicator = dict(
        units="mm", frame="applicator: origin = BT tandem flange, z = tandem axis flange->tip, x = BT world x "
                      "orthogonalised, y = z cross x",
        origin_BT_world=F_bt.tolist(), R_rows_BT_world=R_app.tolist(),
        L_iu_mm=61.8, L_iu_label_mm=round(L_iu_label, 2), r_tandem_mm=2.18, shaft_len_mm=26.0,
        tip_BT_world=tip_bt.round(3).tolist(), h_ovtop_mm=round(h_ovtop, 3),
        spheres=osj["spheres"], sphere_coverage_frac=osj["coverage_frac"],
        source=dict(L_iu="MEASURED BT applicator label (final_applicator_metrics 61.79)", r="MEASURED FWHM 4.36 mm / 2",
                    spheres="MEASURED greedy EDT pack of the BT ovoid label (proto/design_fast/ovoid_spheres.json)"),
        label_semantics=sem)
    json.dump(applicator, open(os.path.join(INP, "applicator.json"), "w"), indent=1)

    # ------------------------------------------------------------------ 3.5 poses
    def pose_json(Fp, a, x):
        return dict(F=np.round(Fp, 4).tolist(), a=np.round(geom.unit(a), 6).tolist(), x=np.round(geom.ortho(x, a), 6).tolist())
    xw = geom.ortho([1, 0, 0], a0)
    d_pred = float(np.clip(h_ovtop - h_apex, 0.0, run))
    poses = dict(units="mm; preBT world RAS", L_end=L_end.tolist(), a0=a0.tolist(), O_pre=O_pre.tolist(),
                 d_F_pred_mm=round(d_pred, 3), d_F_rule="clamp(h_ovtop - h_apex, 0, portio depth)",
                 P1=dict(pose_json(L_end - d_pred * a0, a0, xw), d_F_mm=round(d_pred, 3)))
    # spec sweep {0,7,14,21,27.5}; 27.5 was the portio depth measured from the skeleton start.  From L_end the
    # MEASURED portio depth is `run` (22.5 mm), so the deepest sweep point is clamped to it (flange at O_pre).
    sweep = sorted(set([0.0, 7.0, 14.0, 21.0, min(27.5, run)]))
    poses["SWEEP"] = {("%g" % d): dict(pose_json(L_end - d * a0, a0, xw), d_F_mm=d) for d in sweep}
    poses["portio_depth_mm"] = run
    icp = json.load(open(PRIOR + "/proto/design_contact/check_hrctv_icp.json"))["HRCTV_only_icp"]
    R2 = np.array(icp["R"]); t2 = np.array(icp["t"])
    F2 = R2 @ F_bt + t2; a2 = R2 @ z_bt; x2 = R2 @ R_app[0]
    poses["P2"] = dict(pose_json(F2, a2, x2), R_BT_to_pre=R2.tolist(), t_BT_to_pre=t2.tolist(),
                       convention="y_pre = R x_BT + t (HR-CTV-only ICP, check_hrctv_icp.json)")
    zu = np.load(PRIOR + "/registration/out/rigid_label_uterus+HRCTV.npz")
    Ru, tu = zu["R_dice"], zu["t_dice"]
    poses["P2u"] = dict(pose_json(Ru @ F_bt + tu, Ru @ z_bt, Ru @ R_app[0]), R_BT_to_pre=Ru.tolist(), t_BT_to_pre=tu.tolist(),
                        note="LEAKS the uterus: sensitivity only, never validation")
    off = L_end - F2
    ax2, lat2 = float(off @ a2), float(np.linalg.norm(off - (off @ a2) * a2))
    off_sk = sk0[0] - F2
    ax2s, lat2s = float(off_sk @ a2), float(np.linalg.norm(off_sk - (off_sk @ a2) * a2))
    tip2 = F2 + 61.8 * a2
    sdu = np.where(ut, -ndi.distance_transform_edt(ut, sampling=sp), ndi.distance_transform_edt(~ut, sampling=sp))
    sd_tip2 = float(ndi.map_coordinates(sdu, w2v(A, tip2[None]).T, order=1)[0])
    p2 = dict(L_end_minus_F_axial_lateral_mm=[round(ax2, 2), round(lat2, 2)],
              skeleton_start_minus_F_axial_lateral_mm=[round(ax2s, 2), round(lat2s, 2)],
              angle_to_a0_deg=round(geom.angle_deg(a2, a0), 2),
              angle_to_skeleton_lower12_deg=round(geom.angle_deg(a2, dpre), 2),
              tip_signed_dist_to_uterus_mm=round(sd_tip2, 2),
              F_minus_O_pre_along_a0_mm=round(float((F2 - O_pre) @ a0), 2))
    check("P2 flange axial/lateral below skeleton start (as check_hrctv_icp)", [ax2s, lat2s],
          EXP.get("P2_flange_axial_lateral_mm"), 1.0,
          note="check_hrctv_icp measured from canal_pre.npy[0] (skeleton start), not from L_end")
    check_scalar("P2 tip signed dist to uterus", sd_tip2, "P2_tip_signed_dist_mm", 1.0)
    check_scalar("P2 axis vs skeleton lower-12 mm canal", p2["angle_to_skeleton_lower12_deg"],
                 "P2_axis_vs_skeleton_deg", 2.0, unit="deg")
    poses["P2"]["checks"] = p2
    json.dump(poses, open(os.path.join(INP, "poses.json"), "w"), indent=1)

    named = dict(units="mm, preBT world RAS", L_end=L_end.round(3).tolist(), C_top_skeleton=C_top_sk.round(3).tolist(),
                 C_top=c_top.round(3).tolist(), a0=a0.round(5).tolist(), a0_variants=variants,
                 O_pre=O_pre.round(3).tolist(), portio_depth_below_L_end_mm=run,
                 O_pre_check_os_replica=O_sk.round(3).tolist(), portio_depth_check_os_replica_mm=run_sk,
                 portio_profile_depth_frac=[[float(a), round(float(b), 2)] for a, b in zip(tt[::6], frac[::6])],
                 canal_len_L_end_to_C_top_mm=round(L_canal, 2), canal_centroid_len_mm=round(L_cen, 2),
                 canal_skeleton_ext_len_mm=round(L_sk_ext, 2),
                 canal_centroid_ext_mm=dict(bottom=round(ce_bot, 2), top=round(ce_tip, 2)),
                 n_canal_pts=int(len(canal)), extended_canal_len_mm=round(float(s0[-1]), 2),
                 i_L_end=i_Lend, i_internal_os=i_ios, internal_os_above_L_end_mm=ios_above_Lend,
                 h_apex_mm=round(h_apex, 3), h_ovtop_mm=round(h_ovtop, 3), d_F_pred_mm=round(d_pred, 3),
                 fund_dir=d_f.round(4).tolist(), delta_fund_max_mm=round(delta_max, 2),
                 P2=p2, surfaces=surfs, checks=CHECKS, all_checks_ok=bool(all(c["ok"] for c in CHECKS)))
    json.dump(named, open(os.path.join(INP, "named.json"), "w"), indent=1)
    log.update(named=named, wall_s=round(time.time() - t0, 1))
    json.dump(log, open(os.path.join(P["logs"], "prep_inputs.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in named.items() if k not in ("checks", "surfaces", "portio_profile_depth_frac")}, indent=1))
    failed = [c["name"] for c in CHECKS if not c["ok"]]
    print("CHECKS failed:", failed)
    print("wall %.1f s" % (time.time() - t0))
    # spec 3.3: a named-geometry check outside tolerance must stop the pipeline, unless the deviation was reviewed and
    # accepted with a reason:  python prep_inputs.py --accept-deviations runs_def/accepted_deviations.json
    acc = {}
    if "--accept-deviations" in sys.argv:
        acc = json.load(open(sys.argv[sys.argv.index("--accept-deviations") + 1]))["accepted"]
    open_fail = [n for n in failed if n not in acc]
    named["accepted_deviations"] = {n: acc[n] for n in failed if n in acc}
    json.dump(named, open(os.path.join(INP, "named.json"), "w"), indent=1)
    if open_fail:
        print("STOP: named-geometry checks failed without an accepted deviation: %s" % open_fail)
        sys.exit(2)


if __name__ == "__main__":
    main()
