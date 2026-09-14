"""HOST evaluation (spec section 8).  Units mm, cc, deg.

    python evaluate.py --e0                 # evaluator gate E0 -> eval/E0.json (must pass before any sim is scored)
    python evaluate.py --tag R1_P2_S0       # -> eval/<tag>/visible.json and eval/<tag>/sealed_heldout.json

Alignment E_app: rigid map sim (preBT world) -> BT world that puts the simulated applicator onto the real one
(Kabsch on applicator-frame landmarks transformed by each pose).  Surfaces are mapped by E_app and voxelised
on the BT grid (320x288x104, 1.125x1.125x1.6 mm) with vtkPolyDataToImageStencil; dice()/surf_dists() are copied
verbatim from the registration stage.  B0 (undeformed preBT surfaces at the same pose) goes through the
identical route.  SEALED metrics (uterus, tip/canal landing) are written only to sealed_heldout.json and are
never printed; calibrate.py reads visible.json only.
"""
import argparse
import json
import os
import sys

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
import vtk
from vtk.util import numpy_support as ns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import geom  # noqa: E402

P = config.paths(); D = P["data"]; INP = P["inputs"]; EV = P["eval"]


# --------------------------------------------------------------------- copied VERBATIM from registration/common.py L45-65
def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def surf_dists(a, b, spacing):
    """Symmetric surface distances (mm) between binary masks a, b on the same grid."""
    if a.sum() == 0 or b.sum() == 0:
        return dict(msd=float("nan"), hd95=float("nan"), hd=float("nan"))
    st = ndi.generate_binary_structure(3, 1)
    sa = a & ~ndi.binary_erosion(a, st)
    sb = b & ~ndi.binary_erosion(b, st)
    # crop to joint bbox + margin for speed
    idx = np.argwhere(sa | sb)
    lo = np.maximum(idx.min(0) - 5, 0); hi = idx.max(0) + 6
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    sa, sb = sa[sl], sb[sl]
    da = ndi.distance_transform_edt(~sa, sampling=spacing)
    db = ndi.distance_transform_edt(~sb, sampling=spacing)
    d = np.concatenate([db[sa], da[sb]])
    return dict(msd=float(d.mean()), hd95=float(np.percentile(d, 95)), hd=float(d.max()))
# ---------------------------------------------------------------------------------------------------------------------


def bt_label(n):
    im = nib.load("%s/BT_MRI_label_%s.nii" % (D, n)); return np.asarray(im.dataobj) > 0, im.affine.copy()


def _poly(V, F):
    pts = vtk.vtkPoints(); pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(V, float), deep=1))
    ca = vtk.vtkCellArray()
    ca.SetCells(len(F), ns.numpy_to_vtkIdTypeArray(np.c_[np.full(len(F), 3), F].astype(np.int64).ravel(), deep=1))
    p = vtk.vtkPolyData(); p.SetPoints(pts); p.SetPolys(ca)
    return p


def voxelize(V, F, shape, aff):
    """Closed surface (world mm) -> boolean mask on the grid (shape, aff). Done in continuous index space, where
    voxel centres are integer points (handles the negative-spacing affine)."""
    ijk = (np.asarray(V, float) - aff[:3, 3]) @ np.linalg.inv(aff[:3, :3]).T
    st = vtk.vtkPolyDataToImageStencil(); st.SetInputData(_poly(ijk, F))
    st.SetOutputOrigin(0, 0, 0); st.SetOutputSpacing(1, 1, 1)
    st.SetOutputWholeExtent(0, shape[0] - 1, 0, shape[1] - 1, 0, shape[2] - 1); st.SetTolerance(0.0); st.Update()
    im = vtk.vtkImageData(); im.SetExtent(0, shape[0] - 1, 0, shape[1] - 1, 0, shape[2] - 1)
    im.SetOrigin(0, 0, 0); im.SetSpacing(1, 1, 1); im.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    im.GetPointData().GetScalars().Fill(0)
    s2 = vtk.vtkImageStencil(); s2.SetInputData(im); s2.SetStencilConnection(st.GetOutputPort())
    s2.ReverseStencilOn(); s2.SetBackgroundValue(1); s2.Update()
    arr = ns.vtk_to_numpy(s2.GetOutput().GetPointData().GetScalars()).reshape(shape[2], shape[1], shape[0])
    m = arr.transpose(2, 1, 0) > 0
    if m.sum() == 0:   # fallback: vtkSelectEnclosedPoints on the bbox
        m = _voxelize_enclosed(ijk, F, shape)
    return m


def _voxelize_enclosed(ijk, F, shape):
    lo = np.maximum(np.floor(ijk.min(0)).astype(int) - 1, 0); hi = np.minimum(np.ceil(ijk.max(0)).astype(int) + 2, shape)
    g = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i]) for i in range(3)], indexing="ij"), -1).reshape(-1, 3).astype(float)
    pts = vtk.vtkPoints(); pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(g), deep=1))
    pd = vtk.vtkPolyData(); pd.SetPoints(pts)
    sel = vtk.vtkSelectEnclosedPoints(); sel.SetInputData(pd); sel.SetSurfaceData(_poly(ijk, F)); sel.SetTolerance(1e-6); sel.Update()
    ins = ns.vtk_to_numpy(sel.GetOutput().GetPointData().GetArray("SelectedPoints")).astype(bool)
    m = np.zeros(shape, bool); gi = g[ins].astype(int); m[tuple(gi.T)] = True
    return m


def metrics(sim, ref, sp):
    r = dict(dice=dice(ref, sim)); r.update(surf_dists(ref, sim, sp))
    return {k: round(v, 4) for k, v in r.items()}


def transform(V, R, t):
    return np.asarray(V) @ np.asarray(R).T + np.asarray(t)


def app_landmarks(F, R_rows, sph):
    """Applicator-frame landmarks -> world, for a pose (F, frame rows)."""
    p_app = [[0, 0, s] for s in range(0, 61, 5)] + [s["center_app_mm"] for s in sph] + [[10, 0, 0], [0, 10, 0]]
    return np.asarray(F) + np.asarray(p_app, float) @ np.asarray(R_rows)


def march_mask(mask, aff, p, d, step=0.1, maxd=100):
    ijk = lambda X: np.rint((X - aff[:3, 3]) @ np.linalg.inv(aff[:3, :3]).T).astype(int)  # noqa: E731
    t = np.arange(0, maxd, step); X = p + t[:, None] * d; I = ijk(X)
    ok = np.all((I >= 0) & (I < mask.shape), 1); v = np.zeros(len(t), bool); v[ok] = mask[tuple(I[ok].T)]
    return t, v


def exit_dist(mask, aff, p, d):
    """Distance along d from p to the exit of mask (p inside); negative = p outside (distance back to mask)."""
    t, v = march_mask(mask, aff, p, d)
    if v[0]:
        out = np.nonzero(~v)[0]
        return float(t[out[0]]) if len(out) else float(t[-1])
    tb, vb = march_mask(mask, aff, p, -d)
    ins = np.nonzero(vb)[0]
    return -float(tb[ins[0]]) if len(ins) else float("nan")


def entry_dist(mask, aff, p, d):
    t, v = march_mask(mask, aff, p, d)
    ins = np.nonzero(v)[0]
    return float(t[ins[0]]) if len(ins) else float("nan")


def below_flange_extent(mask, aff, F, a, radius=25.0, onaxis=3.0):
    X = np.argwhere(mask) @ aff[:3, :3].T + aff[:3, 3]
    r = X - F; h = r @ a; rad = np.linalg.norm(r - np.outer(h, a), axis=1)
    b = -h
    return (float(b[rad < radius].max()) if (rad < radius).any() else float("nan"),
            float(b[rad < onaxis].max()) if (rad < onaxis).any() else float("nan"))


def enclosed(points, V, F):
    pts = vtk.vtkPoints(); pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(points, float), deep=1))
    pd = vtk.vtkPolyData(); pd.SetPoints(pts)
    sel = vtk.vtkSelectEnclosedPoints(); sel.SetInputData(pd); sel.SetSurfaceData(_poly(V, F)); sel.SetTolerance(1e-6); sel.Update()
    return ns.vtk_to_numpy(sel.GetOutput().GetPointData().GetArray("SelectedPoints")).astype(bool)


class BT:
    def __init__(self):
        self.lab = {n: bt_label(n)[0] for n in ("uterus", "HR-CTV", "vagina", "applicator", "ovoid")}
        self.aff = bt_label("uterus")[1]; self.shape = self.lab["uterus"].shape
        self.sp = np.abs(np.diag(self.aff)[:3]); self.vv = float(np.prod(self.sp))
        app = json.load(open(INP + "/applicator.json"))
        self.F = np.array(app["origin_BT_world"]); self.R = np.array(app["R_rows_BT_world"]); self.a = self.R[2]
        self.L = float(app["L_iu_mm"]); self.tip = self.F + self.L * self.a; self.sph = app["spheres"]


def score_surface(bt, V, F, name):
    m = voxelize(V, F, bt.shape, bt.aff)
    return m, metrics(m, bt.lab[name], bt.sp)


def e0():
    bt = BT(); poses = json.load(open(INP + "/poses.json"))
    res = {}
    exp = {"B_HR": {"uterus": (0.816, 2.07, 4.66), "HR-CTV": (0.785, 2.74, 9.22)}, "B1": {"uterus": (0.883, 1.38, 3.38)}}
    for key, pk in (("B_HR", "P2"), ("B1", "P2u")):
        R = np.array(poses[pk]["R_BT_to_pre"]); t = np.array(poses[pk]["t_BT_to_pre"])
        res[key] = {}
        for nm, obj in (("uterus", "pre_uterus"), ("HR-CTV", "pre_hrctv")):
            if nm not in exp[key]:
                continue
            V, Fc = geom.read_obj(INP + "/%s.obj" % obj)
            Vb = (V - t) @ R                                             # x_BT = R^T (y - t)
            _, mt = score_surface(bt, Vb, Fc, nm)
            e = exp[key][nm]
            ok = abs(mt["dice"] - e[0]) <= 0.02 and abs(mt["msd"] - e[1]) <= 0.2 and abs(mt["hd95"] - e[2]) <= 0.4
            res[key][nm] = dict(got=mt, expected=dict(dice=e[0], msd=e[1], hd95=e[2]), ok=bool(ok))
            print("E0 %-5s %-7s dice %.3f msd %.2f hd95 %.2f  (exp %.3f / %.2f / %.2f)  %s"
                  % (key, nm, mt["dice"], mt["msd"], mt["hd95"], e[0], e[1], e[2], "OK" if ok else "FAIL"), flush=True)
    res["pass"] = bool(all(v["ok"] for k in ("B_HR", "B1") for v in res[k].values()))
    json.dump(res, open(EV + "/E0.json", "w"), indent=1)
    return res


def evaluate_run(tag, runs_dir=None):
    runs_dir = runs_dir or P["runs"]
    rd = os.path.join(runs_dir, tag); z = np.load(rd + "/final.npz"); summ = json.load(open(rd + "/summary.json"))
    bt = BT(); poses = json.load(open(INP + "/poses.json")); named = json.load(open(INP + "/named.json"))
    Fs, a_s, xs = z["F"], geom.unit(z["a"]), z["x"]
    Rs = geom.pose_frame(dict(F=Fs, a=a_s, x=xs))
    Ps = app_landmarks(Fs, Rs, bt.sph); Pb = app_landmarks(bt.F, bt.R, bt.sph)
    Re, te = geom.kabsch(Ps, Pb)                                          # x_BT = Re y + te
    resid = float(np.linalg.norm(transform(Ps, Re, te) - Pb, axis=1).max())
    out_v = dict(tag=tag, E_app=dict(R=Re.tolist(), t=te.tolist(), landmark_resid_max_mm=resid))
    pose_name = summ["cfg"]["pose"]
    if pose_name == "P2":
        R2 = np.array(poses["P2"]["R_BT_to_pre"]); t2 = np.array(poses["P2"]["t_BT_to_pre"])
        dR = geom.rot_angle_deg(Re @ R2); dt = float(np.linalg.norm(te + R2.T @ t2))
        out_v["E_app_vs_T_HR_inverse"] = dict(rot_deg=dR, trans_mm=dt, ok=bool(dR < 0.01 and dt < 1e-3))
    # E_pelvic diagnostic
    zp = np.load(P["prior"] + "/registration/out/rigid_mi_global_labelcentroid.npz") if P["prior"] else None
    if zp is not None:
        Rp, tp = zp["R"], zp["t"]
        Fb_p = Rp.T @ (Fs - tp); ab_p = Rp.T @ a_s
        out_v["E_pelvic"] = dict(flange_offset_mm=float(np.linalg.norm(Fb_p - bt.F)), axis_angle_deg=geom.angle_deg(ab_p, bt.a))
    sealed = dict(tag=tag)
    masks = {}
    for nm, obj in (("uterus", "uterus"), ("HR-CTV", "hrctv")):
        Vd, Fc = geom.read_obj(rd + "/surf_%s.obj" % obj)
        V0, _ = geom.read_obj(INP + "/pre_%s.obj" % obj)
        m_sim, mt = score_surface(bt, transform(Vd, Re, te), Fc, nm)
        m_b0, mb = score_surface(bt, transform(V0, Re, te), Fc, nm)
        masks[nm] = m_sim
        vol_ratio = float(bt.lab[nm].sum() / max(1, m_sim.sum()))
        rec = dict(sim=mt, B0=mb, vol_sim_cc=round(float(m_sim.sum() * bt.vv / 1000), 3),
                   vol_B0_cc=round(float(m_b0.sum() * bt.vv / 1000), 3), vol_BT_cc=round(float(bt.lab[nm].sum() * bt.vv / 1000), 3),
                   vol_ratio_BT_over_sim=round(vol_ratio, 4),
                   mesh_vol_ratio_sim_over_rest=round(geom.mesh_volume(Vd, Fc) / geom.mesh_volume(V0, Fc), 4))
        (sealed if nm == "uterus" else out_v)[nm] = rec
    # device landmarks in BT frame
    tip_b = transform(Fs + ctrl_L(z) * a_s, Re, te); F_b = transform(Fs, Re, te); a_b = Re @ a_s
    ut_sim = masks["uterus"]
    ref = dict(tip_to_serosa_mm=exit_dist(bt.lab["uterus"], bt.aff, bt.tip, bt.a),
               flange_to_corpus_mm=entry_dist(bt.lab["uterus"], bt.aff, bt.F, bt.a),
               farthest_uterus_from_F_mm=float(np.linalg.norm(np.argwhere(bt.lab["uterus"]) @ bt.aff[:3, :3].T + bt.aff[:3, 3] - bt.F, axis=1).max()))
    canal_f = z["canal_final"]; i_ios = int(named["i_internal_os"]); i_le = int(named["i_L_end"])
    sealed.update(
        tip_to_serosa_mm=exit_dist(ut_sim, bt.aff, tip_b, a_b),
        flange_to_corpus_mm=entry_dist(ut_sim, bt.aff, F_b, a_b),
        internal_os_landing_s_mm=float((canal_f[i_ios] - Fs) @ a_s),
        farthest_uterus_from_F_mm=float(np.linalg.norm(np.argwhere(ut_sim) @ bt.aff[:3, :3].T + bt.aff[:3, 3] - F_b, axis=1).max()),
        canal_chord_angle_to_rod_deg=geom.angle_deg(canal_f[-1] - canal_f[i_le], a_s),
        BT_reference=ref)
    hc = masks["HR-CTV"] | ut_sim
    bf_sim = below_flange_extent(hc, bt.aff, F_b, a_b); bf_bt = below_flange_extent(bt.lab["HR-CTV"] | bt.lab["uterus"], bt.aff, bt.F, bt.a)
    out_v["below_flange_extent_mm"] = dict(sim=bf_sim[0], sim_onaxis=bf_sim[1], BT=bf_bt[0], BT_onaxis=bf_bt[1])
    Vb, Fb = geom.read_obj(rd + "/surf_body.obj")
    rod = Fs + np.outer(np.arange(0.0, ctrl_L(z) + 1e-9, 1.0), a_s)
    ins = enclosed(rod, Vb, Fb)
    first_in = int(np.argmax(ins)) if ins.any() else len(ins)
    out_v["piercing"] = dict(frac_rod_outside_body_beyond_entry=float((~ins[first_in:]).mean()) if first_in < len(ins) else 1.0,
                             frac_rod_outside_body_all=float((~ins).mean()), n_samples=int(len(ins)),
                             rod_entry_s_mm=float(first_in))
    fin = summ.get("final", {})
    out_v["numerics"] = dict(status=summ["status"], converged=summ["converged"], phaseA_control_ok=summ["phaseA_control_ok"],
                             min_vol_ratio=summ["min_vol_ratio_run"], max_stretch=summ["max_stretch_run"],
                             admissibility_flag=summ["admissibility_flag"], tie_err_mean=fin.get("tie_err_mean"),
                             tie_err_max=fin.get("tie_err_max"), pen_max=fin.get("pen_max"),
                             F_axial_mN=fin.get("F_axial_mN"), F_lat_sum_mN=fin.get("F_lat_sum_mN"), M_F_mNmm=fin.get("M_F_mNmm"),
                             forces_note=summ.get("forces_note"))
    od = os.path.join(EV, tag); os.makedirs(od, exist_ok=True)
    json.dump(out_v, open(od + "/visible.json", "w"), indent=1)
    json.dump(sealed, open(od + "/sealed_heldout.json", "w"), indent=1)
    return out_v


def ctrl_L(z):
    return float(z["L_iu"])


# ===================================================================================== stage 2: VALIDATION
# python evaluate.py --freeze TAG [TAG ...]   -> validation/freeze.json (sha256 of code, run configs and results)
# python evaluate.py --validate                -> validation/metrics_M1.json (refuses without a matching freeze)
# Held-out rule of this stage: NO parameter is calibrated on any BT organ (all runs are S0).  Organ metrics are
# computed once, after the freeze, for the rigid-only baseline and the simulation through the identical route:
# preBT (rest or deformed) surface -> rigid map into BT world -> vtkPolyDataToImageStencil on the BT grid ->
# dice() / surf_dists() (verbatim registration/common.py; validated by the E0 gate).
VAL = P["out"] + "/validation"
FRAME_OF_POSE = {"PB": "BONE", "PBg": "GLOBAL", "P2": "HR", "P2u": "UH"}
STRUCTS = (("uterus", "uterus"), ("HR-CTV", "hrctv"), ("vagina", "vagina"))
CODE_FILES = ("config.py", "geom.py", "scene.py", "controller.py", "ties.py", "run_insertion.py", "align.py",
              "evaluate.py", "prep_inputs.py", "prep_m2.py")


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _run_files(tag):
    rd = os.path.join(P["runs"], tag)
    return [os.path.join(rd, f) for f in ("cfg.json", "summary.json", "final.npz")]


def freeze(tags):
    os.makedirs(VAL, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    rec = dict(created=__import__("time").strftime("%Y-%m-%d %H:%M:%S"),
               statement="Written BEFORE any organ metric of these runs was computed or read in stage 2. All runs are "
                         "S0 (no parameter calibrated on BT anatomy). Any change after this point is post hoc.",
               runs=list(tags), code={f: sha256_file(os.path.join(here, f)) for f in CODE_FILES},
               run_files={t: {os.path.basename(p): sha256_file(p) for p in _run_files(t)} for t in tags},
               inputs={f: sha256_file(os.path.join(INP, f)) for f in ("poses.json", "poses_align.json", "applicator.json",
                                                                       "pre_uterus.obj", "pre_hrctv.obj", "pre_vagina.obj")},
               alignment=sha256_file(VAL + "/alignment.json"))
    json.dump(rec, open(VAL + "/freeze.json", "w"), indent=1)
    print("froze %d runs -> %s" % (len(tags), VAL + "/freeze.json"))


def check_freeze():
    fz = json.load(open(VAL + "/freeze.json"))
    bad = [(t, os.path.basename(p)) for t in fz["runs"] for p in _run_files(t)
           if sha256_file(p) != fz["run_files"][t][os.path.basename(p)]]
    if bad:
        sys.exit("REFUSED: run files changed after the freeze: %s" % bad)
    return fz


def mask_world(m, aff):
    return np.argwhere(m) @ aff[:3, :3].T + aff[:3, 3]


def spheres_mask(centres, radii, shape, aff):
    """Union of balls (world mm) on the grid; the ovoid/cap sphere pack of the device."""
    m = np.zeros(shape, bool); inv = np.linalg.inv(aff); sp = np.abs(np.diag(aff)[:3])
    for c, r in zip(np.atleast_2d(centres), radii):
        ijk = inv[:3, :3] @ c + inv[:3, 3]; h = np.ceil(r / sp).astype(int) + 1
        lo = np.maximum(np.floor(ijk).astype(int) - h, 0); hi = np.minimum(np.ceil(ijk).astype(int) + h + 1, shape)
        g = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i]) for i in range(3)], indexing="ij"), -1)
        X = g.reshape(-1, 3) @ aff[:3, :3].T + aff[:3, 3]
        ins = np.linalg.norm(X - c, axis=1) <= r
        m[tuple(g.reshape(-1, 3)[ins].T)] = True
    return m


def pca_major(m, aff, ref):
    X = mask_world(m, aff); c = X.mean(0); _, s, vt = np.linalg.svd(X - c, full_matrices=False)
    v = vt[0] * (1.0 if vt[0] @ ref >= 0 else -1.0)
    return c, v, float(s[0] / s[1])


def signed_depth(m, aff, p):
    """+ = p inside m (distance to the outside), - = outside (distance to m); mm, voxel-exact EDT, nearest voxel."""
    sp = np.abs(np.diag(aff)[:3])
    ijk = np.rint((np.asarray(p) - aff[:3, 3]) @ np.linalg.inv(aff[:3, :3]).T).astype(int)
    if not np.all((ijk >= 0) & (ijk < m.shape)):
        return float("nan")
    if m[tuple(ijk)]:
        return float(ndi.distance_transform_edt(m, sampling=sp)[tuple(ijk)])
    return -float(ndi.distance_transform_edt(~m, sampling=sp)[tuple(ijk)])


def uterus_landmarks(ut, bt, cerv_axis, ios):
    """Device-relative uterus geometry on the BT grid, all measured from the REAL BT tandem (flange F, axis a, tip).
    Flexion angle = angle between the cervical canal axis and the corpus axis, the corpus axis being the line from
    the internal os to the corpus (uterus-label) centroid.  (The PCA major axis is also reported, but the BT corpus
    is nearly isotropic in its two largest principal directions -- l1/l2 = 1.12 -- so a PCA flexion is ill-posed.)
    cervical axis / internal os: BT = tandem axis and tandem entry into the corpus; baseline = preBT a0 and the
    preBT internal-os canal point; sim = its deformed canal (L_end -> internal os) and deformed internal-os point."""
    X = mask_world(ut, bt.aff)
    c, v, ratio = pca_major(ut, bt.aff, bt.a)
    ax = geom.unit(c - np.asarray(ios, float))
    return dict(
        tip_to_serosa_along_axis_mm=round(exit_dist(ut, bt.aff, bt.tip, bt.a), 2),
        tip_depth_signed_mm=round(signed_depth(ut, bt.aff, bt.tip), 2),
        fundal_height_above_flange_mm=round(float(((X - bt.F) @ bt.a).max()), 2),
        flange_to_corpus_entry_mm=round(entry_dist(ut, bt.aff, bt.F, bt.a), 2),
        corpus_centroid_BT=c.round(2).tolist(), internal_os_BT=np.round(ios, 2).tolist(),
        corpus_axis_BT=ax.round(4).tolist(), cervical_axis_BT=np.round(cerv_axis, 4).tolist(),
        flexion_angle_deg=round(geom.angle_deg(cerv_axis, ax), 2),
        corpus_axis_to_tandem_deg=round(geom.angle_deg(bt.a, ax), 2),
        corpus_pca_axis_BT=v.round(4).tolist(), corpus_pca_ratio_l1_l2=round(ratio, 2),
        corpus_pca_axis_to_tandem_deg=round(geom.angle_deg(bt.a, v), 2))


def seg_dist(Pts, F, a, L):
    s = np.clip((Pts - F) @ a, 0.0, L)
    return np.linalg.norm(Pts - (F + s[:, None] * a), axis=1)


def canal_metrics(C, bt, i_le, i_ios):
    """C: canal points already in the BT frame. Intra-uterine canal = internal os -> canal end."""
    d_iu = seg_dist(C[i_ios:], bt.F, bt.a, bt.L)
    return dict(intrauterine_canal_to_BT_tandem_mean_mm=round(float(d_iu.mean()), 2),
                intrauterine_canal_to_BT_tandem_max_mm=round(float(d_iu.max()), 2),
                canal_end_to_BT_tip_mm=round(float(np.linalg.norm(C[-1] - bt.tip)), 2),
                canal_end_axial_minus_BT_tip_mm=round(float((C[-1] - bt.F) @ bt.a - bt.L), 2),
                internal_os_along_BT_axis_mm=round(float((C[i_ios] - bt.F) @ bt.a), 2),
                cervical_canal_axis_to_BT_tandem_deg=round(geom.angle_deg(C[i_ios] - C[i_le], bt.a), 2))


def seg_metrics(a_mask, ref, sp, vv):
    r = metrics(a_mask, ref, sp); r["vol_cc"] = round(float(a_mask.sum() * vv / 1000.0), 3)
    return r


def compare(sim, base):
    d = {k: round(sim[k] - base[k], 4) for k in ("dice", "msd", "hd95")}
    b = dict(dice=sim["dice"] > base["dice"], msd=sim["msd"] < base["msd"], hd95=sim["hd95"] < base["hd95"])
    b["all3"] = bool(all(b.values())); b["none"] = bool(not any(v for k, v in b.items() if k != "all3"))
    return dict(delta_sim_minus_baseline=d, sim_better=b)


def validate_run(tag, frame, bt, al, named, cache):
    rd = os.path.join(P["runs"], tag); z = np.load(rd + "/final.npz"); s = json.load(open(rd + "/summary.json"))
    if frame == "E_APP":        # device-to-device Kabsch (P1 / SWEEP poses): hides pose error by construction
        Rs_ = geom.pose_frame(dict(F=z["F"], a=z["a"], x=z["x"]))
        Re, te = geom.kabsch(app_landmarks(z["F"], Rs_, bt.sph), app_landmarks(bt.F, bt.R, bt.sph))
        R, t = Re.T, -Re.T @ te
        fr = dict(role="E_app (device onto device; secondary, hides pose error)", uses_labels=[],
                  held_out=["uterus", "HR-CTV", "vagina"])
    else:
        fr = al["frames"][frame]; R = np.array(fr["R_BT_to_pre"]); t = np.array(fr["t_BT_to_pre"])
    to_bt = lambda Y: (np.asarray(Y) - t) @ R  # noqa: E731
    fkey = "E_APP:" + tag if frame == "E_APP" else frame
    # consistency: the simulated device mapped into BT must land on the real device (E_app == T^-1) for own-frame runs
    Fs, a_s = z["F"], geom.unit(z["a"])
    dev = dict(flange_err_mm=round(float(np.linalg.norm(to_bt(Fs) - bt.F)), 3), axis_err_deg=round(geom.angle_deg(R.T @ a_s, bt.a), 3))
    i_le, i_ios = int(named["i_L_end"]), int(named["i_internal_os"])
    rec = dict(tag=tag, frame=frame, frame_role=fr["role"], frame_uses_labels=fr["uses_labels"], held_out=fr["held_out"],
               pose=s["cfg"]["pose"], m2=bool(s["cfg"].get("m2")), m2_mode=s["cfg"].get("m2_mode", "inflate") if s["cfg"].get("m2") else None,
               body_mesh=s["cfg"].get("body_mesh", "pre_body"), cell_mm=s["cfg"]["cell_mm"],
               supports=dict(kappa_f_mN_per_mm3=s["cfg"]["kappa_f_mN_per_mm3"], kappa_lig_mN_per_mm3=s["cfg"]["kappa_lig_mN_per_mm3"]),
               device_landing=dev,
               numerics=dict(status=s["status"], converged=s["converged"], admissibility_flag=s["admissibility_flag"],
                             max_stretch=s["max_stretch_run"], min_vol_ratio=s["min_vol_ratio_run"],
                             tie_err_max=s["final"].get("tie_err_max"), pen_max=s["final"].get("pen_max"),
                             F_axial_mN=s["final"].get("F_axial_mN"), F_ovoid_axial_mN=s["final"].get("F_ovoid_axial_mN"),
                             final_umax_mm=s.get("final_umax_mm"), ms_per_step_median=s.get("ms_per_step_median")),
               structures={})
    # piercing (rod samples outside the deformed body beyond the entry point) and the pre-registered numeric gates
    Vb_, Fb_ = geom.read_obj(rd + "/surf_body.obj")
    ins = enclosed(Fs + np.outer(np.arange(0.0, float(z["L_iu"]) + 1e-9, 1.0), a_s), Vb_, Fb_)
    first = int(np.argmax(ins)) if ins.any() else len(ins)
    pierce = float((~ins[first:]).mean()) if first < len(ins) else 1.0
    import gates as gates_mod                 # the single gate definition (gates.py)
    gg = gates_mod.numeric_gates(s, pierce=pierce)
    rec["numerics"].update(piercing_frac_rod_outside_body=pierce, numeric_gates=gg["preregistered"],
                           numeric_gates_pass=gg["pass_preregistered"], numeric_gates_added_v2=gg["added_v2"])
    ov = cache.setdefault("ovoid_spheres", spheres_mask(bt.F + np.array([q["center_app_mm"] for q in bt.sph]) @ bt.R,
                                                        [q["r_mm"] for q in bt.sph], bt.shape, bt.aff))
    masks = {}
    for nm, obj in STRUCTS:
        V0, Fc = geom.read_obj(INP + "/pre_%s.obj" % obj)
        key = ("base", fkey, nm)
        if key not in cache:
            cache[key] = voxelize(to_bt(V0), Fc, bt.shape, bt.aff)
        mb = cache[key]
        sp_ = rd + "/surf_%s.obj" % obj
        simulated = os.path.exists(sp_)
        if simulated:
            Vd, Fd = geom.read_obj(sp_); ms = voxelize(to_bt(Vd), Fd, bt.shape, bt.aff)
            vol_change = geom.mesh_volume(Vd, Fd) / geom.mesh_volume(V0, Fc) - 1.0
        else:
            ms = mb; vol_change = 0.0
        masks[nm] = (mb, ms)
        ref = bt.lab[nm]
        rb = seg_metrics(mb, ref, bt.sp, bt.vv); rs = seg_metrics(ms, ref, bt.sp, bt.vv)
        rec["structures"][nm] = dict(simulated=simulated, BT_vol_cc=round(float(ref.sum() * bt.vv / 1000.0), 3),
                                     baseline=rb, sim=rs, sim_mesh_volume_change_frac=round(float(vol_change), 4), **compare(rs, rb))
    # vagina like-for-like: the BT vagina label CONTAINS the ovoids (100 % of ovoid voxels) and the packing
    # (BT 99.9 cc vs preBT 10.8 cc).  'vagina+device' adds the ovoid sphere pack (placed at the real device) to both
    # model masks; the slab variant scores only 0-18 mm below the BT vaginal apex along the tandem (ovoid zone; deeper
    # is packing, not modelled).
    ref_v = bt.lab["vagina"]
    zb = (mask_world(ref_v, bt.aff) - bt.F) @ bt.a; apex = float(zb.max())
    key = "slab"
    if key not in cache:
        g = np.indices(bt.shape).reshape(3, -1).T @ bt.aff[:3, :3].T + bt.aff[:3, 3]
        zz = ((g - bt.F) @ bt.a).reshape(bt.shape); cache[key] = (zz <= apex) & (zz >= apex - 18.0)
    slab = cache[key]
    mb, ms = masks["vagina"]
    for nm_, (b_, s_, r_) in {"vagina+device": (mb | ov, ms | ov, ref_v),
                              "vagina_ovoid_slab": (mb & slab, ms & slab, ref_v & slab),
                              "vagina+device_ovoid_slab": ((mb | ov) & slab, (ms | ov) & slab, ref_v & slab)}.items():
        rb = seg_metrics(b_, r_, bt.sp, bt.vv); rs = seg_metrics(s_, r_, bt.sp, bt.vv)
        rec["structures"][nm_] = dict(simulated=rec["structures"]["vagina"]["simulated"],
                                      BT_vol_cc=round(float(r_.sum() * bt.vv / 1000.0), 3), baseline=rb, sim=rs, **compare(rs, rb))
    rec["structures"]["vagina_ovoid_slab"]["slab_mm_below_BT_apex"] = [0.0, 18.0]
    # canal / tandem, tip-to-fundus, flexion (the BT reference is computed by the same functions on the BT labels)
    Cb, Cs = to_bt(z["canal_rest"]), to_bt(z["canal_final"])
    a0b = R.T @ np.array(named["a0"])
    rec["canal"] = dict(baseline=canal_metrics(Cb, bt, i_le, i_ios), sim=canal_metrics(Cs, bt, i_le, i_ios),
                        note="sim canal is tied to the simulated rod, which lands on the BT tandem by construction: the "
                             "canal-on-tandem numbers are a consistency check, not a prediction")
    cerv_sim = geom.unit(Cs[i_ios] - Cs[i_le])
    lm_b = uterus_landmarks(masks["uterus"][0], bt, a0b, Cb[i_ios])
    lm_s = uterus_landmarks(masks["uterus"][1], bt, cerv_sim, Cs[i_ios])
    if "lm_BT" not in cache:
        cache["lm_BT"] = uterus_landmarks(bt.lab["uterus"], bt, bt.a,
                                          bt.F + entry_dist(bt.lab["uterus"], bt.aff, bt.F, bt.a) * bt.a)
    lm_ref = cache["lm_BT"]
    rec["uterus_geometry"] = dict(BT=lm_ref, baseline=lm_b, sim=lm_s)
    err = {}
    for k, lm in (("baseline", lm_b), ("sim", lm_s)):
        err[k] = dict(corpus_centroid_err_mm=round(float(np.linalg.norm(np.subtract(lm["corpus_centroid_BT"], lm_ref["corpus_centroid_BT"]))), 2),
                      corpus_axis_err_deg=round(geom.angle_deg(lm["corpus_axis_BT"], lm_ref["corpus_axis_BT"]), 2),
                      flexion_err_deg=round(lm["flexion_angle_deg"] - lm_ref["flexion_angle_deg"], 2),
                      tip_to_serosa_err_mm=round(lm["tip_to_serosa_along_axis_mm"] - lm_ref["tip_to_serosa_along_axis_mm"], 2),
                      fundal_height_err_mm=round(lm["fundal_height_above_flange_mm"] - lm_ref["fundal_height_above_flange_mm"], 2),
                      flange_to_corpus_err_mm=round(lm["flange_to_corpus_entry_mm"] - lm_ref["flange_to_corpus_entry_mm"], 2))
    rec["uterus_geometry_errors_vs_BT"] = err
    return rec


def validate():
    fz = check_freeze(); al = json.load(open(VAL + "/alignment.json"))
    named = json.load(open(INP + "/named.json")); bt = BT(); cache = {}
    out = dict(units="mm, cc, deg, mN", freeze_sha256=sha256_file(VAL + "/freeze.json"), freeze_created=fz["created"],
               method=dict(
                   voxelisation="deformed (sim) or rest (baseline) preBT surface mesh -> rigid map x_BT = R^T (y - t) -> "
                                "vtkPolyDataToImageStencil in continuous index space of the BT grid (320x288x104, "
                                "1.125x1.125x1.6 mm; voxel centre inside the closed surface); fallback vtkSelectEnclosedPoints. "
                                "Validated by gate E0 (reproduces the label-resampled rigid baselines to 0.001 Dice).",
                   metrics="dice(); surf_dists() copied verbatim from registration/common.py: MSD = symmetric mean surface "
                           "distance, HD95 = 95th percentile of the pooled symmetric distances (6-connected surface voxels).",
                   baseline="the same preBT surfaces, undeformed, through the identical rigid map (rigid-only)",
                   device="the BT tandem (flange, axis, L_iu 61.8 mm) and the 14-sphere ovoid pack are the real device in "
                          "BT coordinates; the simulation's device pose was generated from the same rigid frame, so it "
                          "lands on the real device (device_landing)",
                   held_out="no parameter of any run was fitted to any BT organ (S0). BONE frame uses no organ label; "
                            "HR frame uses the HR-CTV (so HR-CTV is not held out there)."),
               alignment=dict((k, dict(role=v["role"], source=v["source"], uses_labels=v["uses_labels"],
                                       rotation_deg=v["rotation_deg"], R_BT_to_pre=v["R_BT_to_pre"], t_BT_to_pre=v["t_BT_to_pre"]))
                              for k, v in al["frames"].items()),
               alignment_agreement=al["agreement"], runs={})
    for tag in fz["runs"]:
        s = json.load(open(os.path.join(P["runs"], tag, "summary.json")))
        frames_ = [FRAME_OF_POSE.get(s["cfg"]["pose"], "E_APP")]
        if frames_[0] == "BONE":
            frames_.append("GLOBAL")                         # frame sensitivity (whole-image MI)
        for frame in [f for f in frames_ if f]:
            r = validate_run(tag, frame, bt, al, named, cache)
            out["runs"]["%s@%s" % (tag, frame)] = r
            su = r["structures"]
            print("%-22s %-6s %-18s | uterus b %.3f/%.2f/%.2f s %.3f/%.2f/%.2f | HR-CTV b %.3f/%.2f s %.3f/%.2f | vag+dev slab b %.3f s %.3f"
                  % (tag, frame, r["numerics"]["status"], su["uterus"]["baseline"]["dice"], su["uterus"]["baseline"]["msd"],
                     su["uterus"]["baseline"]["hd95"], su["uterus"]["sim"]["dice"], su["uterus"]["sim"]["msd"], su["uterus"]["sim"]["hd95"],
                     su["HR-CTV"]["baseline"]["dice"], su["HR-CTV"]["baseline"]["msd"], su["HR-CTV"]["sim"]["dice"], su["HR-CTV"]["sim"]["msd"],
                     su["vagina+device_ovoid_slab"]["baseline"]["dice"], su["vagina+device_ovoid_slab"]["sim"]["dice"]), flush=True)
    # pre-registered M1 validation (README, copied from the design spec before any unsealing): P2-S0 tandem only
    k = "R1_P2_S0@HR"
    if k in out["runs"]:
        r = out["runs"][k]; u = r["structures"]["uterus"]["sim"]
        crit = dict(uterus_msd_lt_2p07=u["msd"] < 2.07, uterus_hd95_lt_4p66=u["hd95"] < 4.66, uterus_dice_gt_0p816=u["dice"] > 0.816,
                    tip_to_serosa_gt_0=r["uterus_geometry"]["sim"]["tip_to_serosa_along_axis_mm"] > 0,
                    uterus_volume_change_lt_3pct=abs(r["structures"]["uterus"]["sim_mesh_volume_change_frac"]) < 0.03)
        crit["PASS"] = bool(all(crit.values()))
        out["preregistered_M1_validation_R1_P2_S0"] = crit
        sp = os.path.join(EV, "R1_P2_S0", "sealed_heldout.json")        # cross-check against the stage-1 sealed file
        if os.path.exists(sp):
            sealed = json.load(open(sp))["uterus"]["sim"]
            out["preregistered_M1_validation_R1_P2_S0"]["stage1_sealed_file_agrees"] = bool(
                all(abs(sealed[q] - u[q]) < 1e-3 for q in ("dice", "msd", "hd95")))
    json.dump(out, open(VAL + "/metrics_M1.json", "w"), indent=1)
    print("wrote", VAL + "/metrics_M1.json")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--e0", action="store_true"); ap.add_argument("--tag", nargs="*")
    ap.add_argument("--freeze", nargs="*"); ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    os.makedirs(EV, exist_ok=True)
    if a.freeze:
        freeze(a.freeze)
    if a.validate:
        validate()
    if a.e0:
        e0()
    for t in a.tag or []:
        e0p = json.load(open(EV + "/E0.json")) if os.path.exists(EV + "/E0.json") else {}
        if not e0p.get("pass"):
            print("WARNING: evaluator gate E0 has not passed; scoring anyway, flagged", flush=True)
        v = evaluate_run(t)
        print(json.dumps(v, indent=1))
        print("(sealed held-out metrics written to eval/%s/sealed_heldout.json; not printed)" % t)


if __name__ == "__main__":
    main()
