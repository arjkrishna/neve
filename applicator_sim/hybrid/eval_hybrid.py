"""HOST (python 3.13): evaluation of the hybrid pelvis simulator (CONTRACT section 6).  Units mm, cc, deg.

    python hybrid/eval_hybrid.py selftest                    # build runs/SELFTEST_K0 and score/compare/field it
    python hybrid/eval_hybrid.py comparators                 # NONE + K0(every Delta)  -> eval/comparators/metrics.json
    python hybrid/eval_hybrid.py score   --tag TAG           # -> eval/<tag>/metrics.json
    python hybrid/eval_hybrid.py compare --tag TAG           # -> eval/<tag>/compare.json + figs
    python hybrid/eval_hybrid.py field   --tag TAG [--ref-dvf f.nii.gz]   # -> eval/<tag>/field/

Everything derived lives under MRI_GYN_sim/hybrid/; the data folder is read-only; no patient-derived coordinate is
hard-coded (every number is read from inputs/, hybrid/meshes/, hybrid/applicator/ or the labels).

FRAMES (both are rigid preBT-world -> BT-world maps; structures are voxelised on the BT grid)
  E_app   the MODEL's own final tube pose superimposed on the REAL BT tandem (Kabsch on applicator-frame landmarks).
          Scores the tissue around the device independently of the pose rule: the device lands on the real one by
          construction, so the pose-rule error is removed from the organ metrics.
  PELVIS  the peri-organ MI frame ("BONE" in align.py / validation/alignment.json; no organ label enters its fit).
          Scores the pose rule and the OARs: x_BT = R^T (y_pre - t) with the convention y_pre = R x_BT + t.

STRUCTURES (BT reference on the left, model mask on the right)
  cervix        BT HR-CTV minus BT uterus          <- cervix body (preBT HR-CTV minus uterus)
  vagina        BT vagina label                    <- vagina body                     (tissue only, no device)
  vagina+device BT vagina label | BT applicator    <- vagina body | model device      (CONTRACT 6: the BT vagina
                label contains the device, so the device is added on BOTH sides)
  bladder / rectum / sigmoid   their BT labels     <- their bodies
  corpus        BT uterus label                    <- corpus body                     (REPORTED, not a target)

METRICS  Dice; MSD = symmetric mean surface distance; HD95 = 95th percentile of the POOLED symmetric surface
distances (the registration-stage definition, evaluate.surf_dists verbatim); HD95max = max of the two directed 95th
percentiles; vol_cc.  Computed by final_eval.metrics_ext, i.e. the identical route used by the stage-2/3 evaluation:
surface -> rigid map into BT world -> vtkPolyDataToImageStencil on the BT grid -> dice()/surf_dists().
Plus: device-to-OAR minimum distances (rectum, bladder, sigmoid; BT reference = the BT applicator label vs the BT OAR
labels, which reproduces the contract's 2.76 / 6.27 mm), the tip-to-serosa margin (sub-voxel, final_eval.exit_sub;
BT reference 5.44 mm) and the pose-rule error (predicted tube vs real BT tandem: axis angle and flange offset).

COMPARATORS (pre-declared, CONTRACT 6)
  NONE  preBT bodies unmoved.
  K0    corpus + cervix moved rigidly by the pose rule's corpus transform (the cervix rides with the corpus), every
        other body unmoved; one per Delta in pose.json corpus.by_flange_shift_mm, plus the default.
  SIM   the hybrid run.
Every SIM-minus-comparator difference is reported with the frame-perturbation noise floor of final_eval
(N_BOOT=30 draws, rotation sd 2 deg/axis about the BT uterus centroid, translation sd 2.34/sqrt(3) mm/axis) and the
declared +/-0.5 mm equivalence margin on MSD.
"""
import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True                      # never leave __pycache__ in the repo
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import vtk
from vtk.util import numpy_support as ns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))           # applicator_sim (parent package)
import config                                        # noqa: E402
import geom                                          # noqa: E402
import evaluate as ev                                # noqa: E402
import final_eval as fe                              # noqa: E402

VERSION = "eval_hybrid/1.0"

P = config.paths()
DATA = P["data"]; INP = P["inputs"]
HYB = P["out"] + "/hybrid"
MESHES = HYB + "/meshes"; APPD = HYB + "/applicator"; RUNS = HYB + "/runs"
EVALD = HYB + "/eval"; FIGS = HYB + "/figs"; LOGS = HYB + "/logs"
VALD = P["out"] + "/validation"

BODIES = ("corpus", "cervix", "vagina", "bladder", "rectum", "sigmoid")   # mesh_bodies priority order
DEVICE_PARTS = ("tube", "shaft", "ovoid_L", "ovoid_R")
K0_BODIES = ("corpus", "cervix")                     # what the K0 rigid motion moves
SCORED = ("cervix", "vagina", "vagina+device", "bladder", "rectum", "sigmoid", "corpus")
OARS = ("rectum", "bladder", "sigmoid")
FRAMES = ("E_app", "PELVIS")
PELVIS_KEY = "BONE"                                  # align.py's historical name for the peri-organ MI frame

BAND_MM = 25.0                                       # extrapolation band outside the bodies (field)
IDW_K = 12                                           # neighbours for the inverse-distance extrapolation
INV_ITERS = 0                                        # fixed-point polish sweeps after the scatter/IDW inverse
                                                     # (0 = none: the IDW seed is exact for rigid motion and the
                                                     #  extrapolated band folds, so polishing there is harmful)

COLORS = dict(corpus="#d62728", cervix="#ff7f0e", vagina="#9467bd", bladder="#1f77b4",
              rectum="#2ca02c", sigmoid="#8c564b", device="#e7e7e7")


# --------------------------------------------------------------------------------------------- small helpers
def _j(o):
    """JSON-safe (numpy scalars/arrays -> python)."""
    if isinstance(o, dict):
        return {k: _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_j(v) for v in o]
    if isinstance(o, np.ndarray):
        return _j(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def wjson(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_j(obj), fh, indent=1)
    return path


def read_ug(path):
    """Legacy VTK unstructured grid -> (points (n,3), tets (m,4))."""
    r = vtk.vtkUnstructuredGridReader(); r.SetFileName(path)
    r.ReadAllScalarsOn(); r.ReadAllVectorsOn(); r.Update()
    ug = r.GetOutput()
    pts = ns.vtk_to_numpy(ug.GetPoints().GetData()).astype(float)
    return ug, pts


def run_mesh_dirs(tag):
    """Per-body mesh directory FOR ONE RUN.  Every body is `meshes/<body>/`, except the vagina of a Stage-2a wall
    run (`cfg.vagina_model == "wall"`), whose mesh is `meshes/<cfg.vagina_wall_dir>/` -- the scene loads it through
    the shadow root `meshes/_scene_<dir>/vagina/`, which holds the same three files.

    `final/<body>_u.npy` is written in the node order of the mesh the RUN used, so the field command must resolve
    the mesh per run: against the Stage-1 solid vagina it exits with
    "vagina: tets.vtk has 1503 points, vagina_u.npy has 1120"."""
    d = {b: "%s/%s" % (MESHES, b) for b in BODIES}
    p = "%s/%s/cfg.json" % (RUNS, tag)
    cfg = json.load(open(p)) if os.path.exists(p) else {}
    if cfg.get("vagina_model") == "wall":
        wd = cfg.get("vagina_wall_dir", "vagina_wall")
        for cand in ("%s/_scene_%s/vagina" % (MESHES, wd), "%s/%s" % (MESHES, wd)):
            if os.path.exists(cand + "/tets.vtk"):
                d["vagina"] = cand
                break
        else:
            raise SystemExit("run %s has vagina_model='wall' but no mesh for vagina_wall_dir=%r under %s"
                             % (tag, wd, MESHES))
    return d


def app_landmarks(F, R_rows, L):
    """Applicator-frame landmarks -> world.  p_world = F + p_app @ R_rows (rows = x, y, z axes in world)."""
    zs = list(np.arange(0.0, float(L) + 1e-9, 5.0))
    p = [[0.0, 0.0, z] for z in zs]
    p += [[10, 0, 0], [0, 10, 0], [-10, 0, 0], [0, -10, 0], [10, 0, float(L)], [0, 10, float(L)]]
    return np.asarray(F, float) + np.asarray(p, float) @ np.asarray(R_rows, float)


def mask_min_dist(a, b, sp):
    """Minimum distance (mm) between two binary masks on the same grid; 0.0 if they overlap, nan if either empty."""
    if not a.any() or not b.any():
        return float("nan")
    if (a & b).any():
        return 0.0
    return float(ndi.distance_transform_edt(~b, sampling=sp)[a].min())


def rot_frame_from_axis(a, xref=(1.0, 0.0, 0.0)):
    """Applicator frame rows (x, y, z) from the tube axis, world-x orthogonalised: the convention used by both
    prep_inputs (BT device) and applicator_venezia (predicted device), so E_app carries no spurious roll."""
    return geom.frame_from(np.asarray(a, float), xref)


# --------------------------------------------------------------------------------------------- context
class Ctx:
    """BT reference grid + labels, the preBT grid, the body meshes, the device meshes and the pose rule."""

    def __init__(self):
        self.bt = ev.BT()                                     # uterus, HR-CTV, vagina, applicator, ovoid + device pose
        for n in OARS:
            self.bt.lab[n] = ev.bt_label(n)[0]
        self.al = json.load(open(VALD + "/alignment.json"))
        self.pose = json.load(open(APPD + "/pose.json"))
        self.appj = json.load(open(APPD + "/applicator.json"))
        self.bodies = json.load(open(MESHES + "/bodies.json"))
        self.meta = {b: json.load(open("%s/%s/meta.json" % (MESHES, b))) for b in BODIES}
        self.rest = {b: geom.read_obj("%s/%s/surface.obj" % (MESHES, b)) for b in BODIES}
        self.devsurf = {p: geom.read_obj("%s/%s.obj" % (APPD, p)) for p in DEVICE_PARTS}
        im = nib.load(DATA + "/preBT_MRI.nii")
        self.pre_shape = tuple(int(s) for s in im.shape); self.pre_aff = im.affine.copy()
        self.pre_sp = np.sqrt((self.pre_aff[:3, :3] ** 2).sum(0))
        bt = self.bt
        self.ref = {
            "cervix": bt.lab["HR-CTV"] & ~bt.lab["uterus"],
            "vagina": bt.lab["vagina"],
            "vagina+device": bt.lab["vagina"] | bt.lab["applicator"],
            "bladder": bt.lab["bladder"], "rectum": bt.lab["rectum"], "sigmoid": bt.lab["sigmoid"],
            "corpus": bt.lab["uterus"],
        }
        self.bt_dev = bt.lab["applicator"]                    # BT device reference (rod + ovoid signal void)
        # BT reference landing / device-to-OAR numbers (computed, not hard-coded)
        self.bt_tip_to_serosa = float(fe.exit_sub(bt.lab["uterus"], bt.aff, bt.tip, bt.a))
        self.bt_flange_to_corpus = float(fe.entry_sub(bt.lab["uterus"], bt.aff, bt.F, bt.a))
        self.bt_dev_oar = {o: mask_min_dist(self.bt_dev, bt.lab[o], bt.sp) for o in OARS}

    # ---- frames
    def pelvis_map(self):
        f = self.al["frames"][PELVIS_KEY]
        R = np.array(f["R_BT_to_pre"], float); t = np.array(f["t_BT_to_pre"], float)
        return (lambda Y: (np.asarray(Y, float) - t) @ R), R, t

    def eapp_map(self, dev):
        """Kabsch putting the model's device onto the real BT tandem.  Returns (map, residual_mm)."""
        Ps = app_landmarks(dev["flange"], dev["R_rows"], self.bt.L)
        Pb = app_landmarks(self.bt.F, self.bt.R, self.bt.L)
        Re, te = geom.kabsch(Ps, Pb)
        resid = float(np.linalg.norm(Ps @ Re.T + te - Pb, axis=1).max())
        return (lambda Y: np.asarray(Y, float) @ Re.T + te), resid, Re, te

    # ---- the pose rule
    def deltas(self):
        d = self.pose["corpus"]["by_flange_shift_mm"]
        return sorted(d.keys(), key=lambda s: float(s))

    def default_delta(self):
        return str(self.pose["default_flange_shift_mm"])

    def delta_key(self, delta):
        """Nearest stored Delta key for a requested value (exact match preferred)."""
        keys = self.deltas()
        if delta is None:
            return self.default_delta()
        s = ("%g" % float(delta))
        if s in keys:
            return s
        f = float(delta)
        return min(keys, key=lambda k: abs(float(k) - f))

    def rule_device(self, delta_key):
        """Predicted device pose in preBT world for a Delta (the tube axis is the same for every Delta; the flange
        slides along the shaft axis)."""
        d = self.pose["device_final"]
        e = self.pose["corpus"]["by_flange_shift_mm"][delta_key]
        a = geom.unit(np.array(e["tip"], float) - np.array(e["flange"], float))
        R = rot_frame_from_axis(a)
        return dict(flange=np.array(e["flange"], float), axis=a, R_rows=R,
                    tip=np.array(e["tip"], float), shaft_axis=np.array(d["shaft_axis"], float),
                    source="pose.json corpus.by_flange_shift_mm['%s'] (rule v2)" % delta_key)

    def corpus_T(self, delta_key):
        return np.array(self.pose["corpus"]["by_flange_shift_mm"][delta_key]["T_preBT_to_target"], float)


# --------------------------------------------------------------------------------------------- models
class Model:
    """A set of body surfaces in preBT world + a device pose (or None)."""

    def __init__(self, name, surfs, dev, meta=None):
        self.name = name; self.surfs = surfs; self.dev = dev; self.meta = meta or {}


def model_none(cx):
    """NONE: preBT bodies unmoved.  It has no device of its own; the rule's device at the DEFAULT Delta is attached
    so that (a) the E_app frame exists for it and (b) `vagina+device` stays like-for-like with SIM and K0.  The plain
    `vagina` structure is device-free on every model, so the tissue-only number is always available too."""
    dk = cx.default_delta()
    return Model("NONE", {b: cx.rest[b] for b in BODIES}, cx.rule_device(dk),
                 meta=dict(definition="preBT bodies unmoved (no simulation, no insertion)",
                           device=("rule v2 device at the default Delta = %s mm; NONE has no device of its own, it is "
                                   "attached only for the E_app frame and for vagina+device" % dk),
                           flange_shift_mm=float(dk)))


def model_k0(cx, delta_key):
    """K0: corpus + cervix moved rigidly by the rule's corpus transform (the cervix rides with the corpus)."""
    T = cx.corpus_T(delta_key); R = T[:3, :3]; t = T[:3, 3]
    surfs = {}
    for b in BODIES:
        V, F = cx.rest[b]
        surfs[b] = ((V @ R.T + t, F) if b in K0_BODIES else (V, F))
    return Model("K0@%s" % delta_key, surfs, cx.rule_device(delta_key),
                 meta=dict(definition="corpus + cervix moved rigidly by the pose rule's corpus transform "
                                      "(cervix rides with the corpus); all other bodies unmoved",
                           flange_shift_mm=float(delta_key), T_preBT_to_target=T.tolist(),
                           rotation_deg=float(geom.rot_angle_deg(R))))


def read_device_final(path, L_iu):
    """Tolerant reader for runs/<tag>/device_final.json (the scene module owns the exact key names)."""
    d = json.load(open(path))
    g = lambda *k: next((d[x] for x in k if x in d), None)  # noqa: E731
    F = np.array(g("flange", "F", "flange_mm"), float)
    a = g("tube_axis", "axis", "a")
    tip = g("tip", "tip_mm")
    if a is None and tip is not None:
        a = geom.unit(np.array(tip, float) - F)
    a = geom.unit(np.array(a, float))
    Rr = g("R_rows", "R", "frame_rows")
    xa = g("x_app", "x")
    if Rr is not None:
        R = np.array(Rr, float)
    elif xa is not None:                       # build the frame from the run's own roll reference
        x = geom.ortho(np.array(xa, float), a)
        R = np.stack([x, np.cross(a, x), a])
    else:
        R = rot_frame_from_axis(a)
    if tip is None:
        tip = F + float(L_iu) * a
    return dict(flange=F, axis=a, R_rows=R, tip=np.array(tip, float),
                shaft_axis=np.array(g("shaft_axis") if g("shaft_axis") is not None else a, float),
                source=os.path.relpath(path, HYB).replace("\\", "/"), raw=d)


def model_sim(cx, tag):
    rd = "%s/%s" % (RUNS, tag)
    dev = read_device_final(rd + "/device_final.json", cx.bt.L)
    surfs = {}
    missing = []
    for b in BODIES:
        p = "%s/final/%s.obj" % (rd, b)
        if os.path.exists(p):
            surfs[b] = geom.read_obj(p)
        else:
            surfs[b] = cx.rest[b]; missing.append(b)
    cfg = json.load(open(rd + "/cfg.json")) if os.path.exists(rd + "/cfg.json") else {}
    summ = json.load(open(rd + "/summary.json")) if os.path.exists(rd + "/summary.json") else {}
    return Model("SIM", surfs, dev, meta=dict(tag=tag, cfg=cfg, bodies_missing_final_obj=missing,
                                              flange_shift_mm=run_delta(cfg, summ, dev, cx),
                                              run_status=run_status(summ)))


def run_delta(cfg, summ, dev, cx):
    """The run's scenario Delta: the device file and the summary are more reliable than cfg (which may leave
    flange_shift_mm null and resolve the default inside the scene module)."""
    for src in (dev.get("raw", {}), summ, summ.get("device_final", {}), cfg):
        for k in ("flange_shift_mm", "delta_mm", "Delta", "flange_shift", "delta"):
            if isinstance(src, dict) and src.get(k) is not None:
                return float(src[k])
    return float(cx.default_delta())


def run_status(summ):
    """Whether the run actually completed: a SIM that aborted or never reached the final pose must not be read as
    a prediction, so the verdict travels with the metrics."""
    if not summ:
        return dict(available=False, note="no summary.json in the run directory")
    g = summ.get("gates", {}) or {}
    conv = summ.get("convergence", {}) or {}
    fin = summ.get("final", {}) or {}
    ok = bool(summ.get("converged")) and not g.get("inverted_tets") and not g.get("nan") \
        and bool(g.get("reached_final_pose", True))
    return dict(available=True, status=summ.get("status"), converged=bool(summ.get("converged")),
                usable_as_prediction=ok, n_steps=summ.get("n_steps"),
                reached_final_pose=g.get("reached_final_pose"), inverted_tets=g.get("inverted_tets"),
                min_vol_ratio_run=g.get("min_vol_ratio_run"), final_u=fin.get("u"), final_phase=fin.get("phase"),
                dx_max_final_mm=conv.get("dx_max_final_mm"),
                projected_remaining_drift_mm=conv.get("projected_remaining_drift_mm"),
                displacement=summ.get("displacement"),
                warning=None if ok else "RUN DID NOT COMPLETE: its metrics describe the aborted state, not a "
                                        "prediction of the post-insertion anatomy")


# --------------------------------------------------------------------------------------------- masks + scoring
def device_mask(cx, dev, fmap):
    """Union of the four device parts, placed at `dev` in preBT world, mapped by fmap, voxelised on the BT grid.
    The parts are voxelised SEPARATELY and OR-ed: they overlap (the tube runs through the caps) and the even-odd
    stencil rule would cancel the overlap if they were merged into one polydata."""
    if dev is None:
        return None
    m = np.zeros(cx.bt.shape, bool)
    for _, (V, F) in cx.devsurf.items():
        Vw = np.asarray(dev["flange"], float) + V @ np.asarray(dev["R_rows"], float)
        m |= ev.voxelize(fmap(Vw), F, cx.bt.shape, cx.bt.aff)
    return m


def model_masks(cx, model, fmap):
    m = {b: ev.voxelize(fmap(V), F, cx.bt.shape, cx.bt.aff) for b, (V, F) in model.surfs.items()}
    dev = device_mask(cx, model.dev, fmap)
    m["device"] = dev
    m["vagina+device"] = (m["vagina"] | dev) if dev is not None else m["vagina"]
    return m


def pose_rule_error(cx, dev, fmap):
    """Predicted tube vs the real BT tandem, in the frame given by fmap (used with PELVIS)."""
    bt = cx.bt
    F_b = fmap(np.asarray(dev["flange"], float)[None])[0]
    tip_b = fmap(np.asarray(dev["tip"], float)[None])[0]
    a_b = geom.unit(tip_b - F_b)
    off = F_b - bt.F; along = float(off @ bt.a); lat = float(np.linalg.norm(off - along * bt.a))
    toff = tip_b - bt.tip
    # sagittal / coronal decomposition of the axis error, in the BT applicator frame (x = patient right, y = anterior)
    xb, yb = bt.R[0], bt.R[1]
    sag = float(np.degrees(np.arctan2(a_b @ yb, a_b @ bt.a)))
    cor = float(np.degrees(np.arctan2(a_b @ xb, a_b @ bt.a)))
    return dict(axis_angle_deg=round(geom.angle_deg(a_b, bt.a), 3),
                axis_error_components=dict(sagittal_deg=round(sag, 3), coronal_deg=round(cor, 3)),
                flange_offset_mm=dict(total=round(float(np.linalg.norm(off)), 3), along_BT_axis=round(along, 3),
                                      lateral=round(lat, 3), vector_BT=np.round(off, 3).tolist()),
                tip_offset_mm=dict(total=round(float(np.linalg.norm(toff)), 3),
                                   along_BT_axis=round(float(toff @ bt.a), 3)),
                predicted_flange_BT=np.round(F_b, 3).tolist(), predicted_axis_BT=np.round(a_b, 5).tolist(),
                real_flange_BT=np.round(bt.F, 3).tolist(), real_axis_BT=np.round(bt.a, 5).tolist(),
                sign_convention="along < 0 = predicted flange inferior to the real one along the BT tandem; "
                                "sagittal > 0 = predicted axis more anterior; coronal > 0 = more to the patient's right")


def score_model(cx, model, frame):
    """All metrics of one model in one frame."""
    if frame == "PELVIS":
        fmap, R, t = cx.pelvis_map()
        finfo = dict(name="PELVIS", source="validation/alignment.json frames.%s (peri-organ MI; no organ label in "
                                           "its fit)" % PELVIS_KEY, R_BT_to_pre=R.tolist(), t_BT_to_pre=t.tolist(),
                     convention="y_pre = R x_BT + t ; x_BT = R^T (y_pre - t)")
    else:
        fmap, resid, Re, te = cx.eapp_map(model.dev)
        finfo = dict(name="E_app", source="Kabsch of the model's applicator-frame landmarks onto the real BT tandem",
                     R=Re.tolist(), t=te.tolist(), landmark_resid_max_mm=round(resid, 9),
                     convention="x_BT = R y_pre + t")
    m = model_masks(cx, model, fmap)
    bt = cx.bt
    out = dict(frame=finfo, structures={})
    for s in SCORED:
        out["structures"][s] = fe.metrics_ext(m[s], cx.ref[s], bt.sp, bt.vv)
        out["structures"][s]["BT_vol_cc"] = round(float(cx.ref[s].sum() * bt.vv / 1000.0), 3)
    # device-derived quantities
    F_b = fmap(np.asarray(model.dev["flange"], float)[None])[0]
    tip_b = fmap(np.asarray(model.dev["tip"], float)[None])[0]
    a_b = geom.unit(tip_b - F_b)
    dev = m["device"]
    dmod = {o: round(mask_min_dist(dev, m[o], bt.sp), 3) for o in OARS}
    out["device_to_OAR_min_mm"] = dict(
        model=dmod,
        BT_reference={o: round(cx.bt_dev_oar[o], 3) for o in OARS},
        error={o: round(dmod[o] - cx.bt_dev_oar[o], 3) for o in OARS},
        overlap_cc={o: round(float((dev & m[o]).sum() * bt.vv / 1000.0), 3) for o in OARS}
        if dev is not None else None,
        definition="minimum distance between the device mask and the OAR mask on the BT grid. A distance of 0 means "
                   "the two masks intersect, so overlap_cc (the intersection volume) says how deeply: the kinematic "
                   "comparators do not push the OARs aside, so their device interpenetrates them. "
                   "BT reference = BT applicator label vs the BT OAR labels.")
    t2s = float(fe.exit_sub(m["corpus"], bt.aff, tip_b, a_b))
    f2c = float(fe.entry_sub(m["corpus"], bt.aff, F_b, a_b))
    out["landing"] = dict(tip_to_serosa_mm=round(t2s, 3), BT_tip_to_serosa_mm=round(cx.bt_tip_to_serosa, 3),
                          tip_to_serosa_err_mm=round(t2s - cx.bt_tip_to_serosa, 3),
                          flange_to_corpus_mm=round(f2c, 3), BT_flange_to_corpus_mm=round(cx.bt_flange_to_corpus, 3),
                          definition="sub-voxel first crossing of the trilinearly interpolated occupancy through 0.5 "
                                     "along the model tube axis (final_eval.exit_sub/entry_sub); corpus = uterine body")
    out["pose_rule_error"] = pose_rule_error(cx, model.dev, fmap)
    out["device_vol_cc"] = round(float(dev.sum() * bt.vv / 1000.0), 3) if dev is not None else None
    return out, m


def score_all_frames(cx, model):
    return {f: score_model(cx, model, f)[0] for f in FRAMES}


def _fmt(m):
    return "%.3f / %5.2f / %5.2f" % (m["dice"], m["msd"], m["hd95"])


def print_table(title, rows):
    print("\n%s" % title)
    print("  %-14s | %-22s | %-22s" % ("structure", "E_app  dice/msd/hd95", "PELVIS dice/msd/hd95"))
    print("  " + "-" * 64)
    for s, a, b in rows:
        print("  %-14s | %-22s | %-22s" % (s, _fmt(a), _fmt(b)), flush=True)


# --------------------------------------------------------------------------------------------- cmd: score
def cmd_score(tag, cx=None, quiet=False):
    cx = cx or Ctx()
    model = model_sim(cx, tag)
    res = score_all_frames(cx, model)
    out = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), version=VERSION, tag=tag,
               units="mm; volumes cc; angles deg; Dice [-]",
               frames_doc=dict(E_app="the run's final tube pose superimposed on the real BT tandem",
                               PELVIS="peri-organ MI frame (align.py '%s')" % PELVIS_KEY),
               definitions=dict(
                   cervix="BT HR-CTV minus BT uterus vs the cervix body",
                   vagina="BT vagina label vs the vagina body (tissue only, no device on either side)",
                   **{"vagina+device": "BT vagina | BT applicator vs vagina body | model device (CONTRACT 6)"},
                   corpus="BT uterus vs the corpus body (REPORTED, not a target)",
                   metrics="Dice; MSD; HD95 = 95th pct of the pooled symmetric surface distances; HD95max = max of "
                           "the two directed 95th pcts (final_eval.metrics_ext / evaluate.surf_dists)"),
               run=dict(cfg=model.meta.get("cfg", {}), flange_shift_mm=model.meta.get("flange_shift_mm"),
                        status=model.meta.get("run_status", {}),
                        bodies_missing_final_obj=model.meta.get("bodies_missing_final_obj", []),
                        device_final={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                      for k, v in model.dev.items() if k != "raw"}),
               BT_reference=dict(tip_to_serosa_mm=round(cx.bt_tip_to_serosa, 3),
                                 flange_to_corpus_mm=round(cx.bt_flange_to_corpus, 3),
                                 device_to_OAR_min_mm={o: round(cx.bt_dev_oar[o], 3) for o in OARS},
                                 grid=dict(shape=list(cx.bt.shape), spacing_mm=cx.bt.sp.tolist())),
               frames=res)
    p = wjson("%s/%s/metrics.json" % (EVALD, tag), out)
    if not quiet:
        st = model.meta.get("run_status", {})
        if st.get("available") and not st.get("usable_as_prediction"):
            print("\n*** WARNING %s: status=%s converged=%s reached_final_pose=%s inverted_tets=%s (u=%s, %d steps)"
                  % (tag, st.get("status"), st.get("converged"), st.get("reached_final_pose"),
                     st.get("inverted_tets"), st.get("final_u"), st.get("n_steps") or 0))
            print("    %s" % st.get("warning"), flush=True)
        print_table("SIM %s" % tag, [(s, res["E_app"]["structures"][s], res["PELVIS"]["structures"][s]) for s in SCORED])
        pr = res["PELVIS"]["pose_rule_error"]
        print("  pose-rule error (PELVIS): axis %.2f deg | flange %.2f mm (along %.2f, lateral %.2f)"
              % (pr["axis_angle_deg"], pr["flange_offset_mm"]["total"], pr["flange_offset_mm"]["along_BT_axis"],
                 pr["flange_offset_mm"]["lateral"]))
        for f in FRAMES:
            d = res[f]["device_to_OAR_min_mm"]
            print("  %-6s device-to-OAR mm: %s  (BT %s)" % (f, d["model"], d["BT_reference"]))
            print("  %-6s tip-to-serosa %.2f mm (BT %.2f)" % (f, res[f]["landing"]["tip_to_serosa_mm"],
                                                              cx.bt_tip_to_serosa))
        print("wrote", p, flush=True)
    return out


# --------------------------------------------------------------------------------------------- cmd: comparators
def cmd_comparators(cx=None, deltas=None):
    cx = cx or Ctx()
    dks = deltas or cx.deltas()
    models = [("NONE", model_none(cx))] + [("K0@%s" % d, model_k0(cx, d)) for d in dks]
    out = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), version=VERSION,
               units="mm; volumes cc; angles deg; Dice [-]",
               comparators=dict(
                   NONE="preBT bodies unmoved (no insertion). It has no device of its own: the rule's device at the "
                        "default Delta is attached only to define the E_app frame and to keep vagina+device "
                        "like-for-like; the plain `vagina` structure is device-free on every model.",
                   K0="corpus + cervix moved rigidly by the pose rule's corpus transform (the cervix rides with the "
                      "corpus), all other bodies unmoved; one entry per Delta in pose.json"),
               default_flange_shift_mm=float(cx.default_delta()),
               deltas_mm=[float(d) for d in dks],
               BT_reference=dict(tip_to_serosa_mm=round(cx.bt_tip_to_serosa, 3),
                                 flange_to_corpus_mm=round(cx.bt_flange_to_corpus, 3),
                                 device_to_OAR_min_mm={o: round(cx.bt_dev_oar[o], 3) for o in OARS}),
               models={})
    for name, mdl in models:
        res = score_all_frames(cx, mdl)
        out["models"][name] = dict(meta=mdl.meta, frames=res)
        print_table(name, [(s, res["E_app"]["structures"][s], res["PELVIS"]["structures"][s]) for s in SCORED])
        pr = res["PELVIS"]["pose_rule_error"]
        print("  pose-rule error (PELVIS): axis %.2f deg | flange %.2f mm (along %.2f, lat %.2f) | corpus Dice %.3f"
              % (pr["axis_angle_deg"], pr["flange_offset_mm"]["total"], pr["flange_offset_mm"]["along_BT_axis"],
                 pr["flange_offset_mm"]["lateral"], res["PELVIS"]["structures"]["corpus"]["dice"]), flush=True)
    p = wjson("%s/comparators/metrics.json" % EVALD, out)
    print("wrote", p, flush=True)
    return out


# --------------------------------------------------------------------------------------------- cmd: compare
def struct_mask(m, s):
    return m[s]


def _fm_key(mdl, frame):
    """Hashable identity of a model's frame map: PELVIS is shared by every model; E_app depends only on the
    model's device pose, so models with the same device share it."""
    if frame == "PELVIS":
        return "PELVIS"
    d = mdl.dev
    return ("E_app", tuple(np.round(np.asarray(d["flange"], float), 6)),
            tuple(np.round(np.asarray(d["R_rows"], float).ravel(), 6)))


def boot_noise_floor(cx, models, n=None):
    """final_eval's frame-perturbation noise floor: the SAME random rigid perturbation (rotation sd 2 deg/axis about
    the BT uterus centroid, translation sd 2.34/sqrt(3) mm/axis) is applied to every model; the spread of the
    sim-minus-comparator deltas over the draws is the frame-choice noise floor of that comparison.

    Masks and metrics are memoised per draw: the models share the bodies they do not move (and, when their device
    poses agree, the device mask and the whole E_app map), so an identical mask is voxelised and scored once."""
    n = n or fe.N_BOOT
    c = ev.mask_world(cx.bt.lab["uterus"], cx.bt.aff).mean(0)
    fmaps = {k: {"PELVIS": cx.pelvis_map()[0], "E_app": cx.eapp_map(m.dev)[0]} for k, m in models.items()}
    fkeys = {k: {f: _fm_key(m, f) for f in FRAMES} for k, m in models.items()}
    rng = np.random.default_rng(0)
    draws = {f: {k: [] for k in models} for f in FRAMES}
    t0 = time.time()
    for b in range(n):
        pt = fe._perturb(rng, c)
        mcache, scache = {}, {}
        for f in FRAMES:
            for k, mdl in models.items():
                fm = fmaps[k][f]; fk = fkeys[k][f]
                warp = lambda Y, _fm=fm: pt(_fm(Y))                          # noqa: E731
                mk = {}
                for bd, (V, F) in mdl.surfs.items():
                    ck = (fk, id(V))
                    if ck not in mcache:
                        mcache[ck] = ev.voxelize(warp(V), F, cx.bt.shape, cx.bt.aff)
                    mk[bd] = mcache[ck]
                dk = (fk, "device", tuple(np.round(np.asarray(mdl.dev["flange"], float), 6)))
                if dk not in mcache:
                    mcache[dk] = device_mask(cx, mdl.dev, warp)
                mk["device"] = mcache[dk]
                vk = (fk, "vag+dev", id(mdl.surfs["vagina"][0]), dk)
                if vk not in mcache:
                    mcache[vk] = mk["vagina"] | mk["device"]
                mk["vagina+device"] = mcache[vk]
                rec = {}
                for s in SCORED:
                    sk = (id(mk[s]), s)
                    if sk not in scache:
                        scache[sk] = ev.metrics(mk[s], cx.ref[s], cx.bt.sp)
                    rec[s] = scache[sk]
                draws[f][k].append(rec)
        if b == 0:
            print("  [noise floor] draw 1/%d in %.1f s (%d masks, %d scored)"
                  % (n, time.time() - t0, len(mcache), len(scache)), flush=True)
    return draws, n


def deltas_with_floor(draws, n, sim_key, comp_key):
    res = {}
    for f in FRAMES:
        res[f] = {}
        for s in SCORED:
            rec = {}
            for q in ("dice", "msd", "hd95"):
                d = np.array([draws[f][sim_key][i][s][q] - draws[f][comp_key][i][s][q] for i in range(n)])
                lo, med, hi = np.percentile(d, [5, 50, 95])
                rec[q] = dict(median=round(float(med), 4), p5=round(float(lo), 4), p95=round(float(hi), 4))
            d = rec["msd"]
            better = ("sim better" if d["p95"] < 0 else
                      ("comparator better" if d["p5"] > 0 else "interval covers 0"))
            equiv = bool(d["p5"] > -fe.EQUIV_MM and d["p95"] < fe.EQUIV_MM)
            rec["msd_verdict"] = better + ("; within +/-%.1f mm equivalence" % fe.EQUIV_MM if equiv else "")
            rec["msd_equivalent"] = equiv
            res[f][s] = rec
    return res


def cmd_compare(tag, cx=None, n_boot=None):
    cx = cx or Ctx()
    sim = model_sim(cx, tag)
    dk = cx.delta_key(sim.meta.get("flange_shift_mm"))
    none_m = model_none(cx); k0_m = model_k0(cx, dk)
    models = {"SIM": sim, "NONE": none_m, "K0": k0_m}
    point = {k: score_all_frames(cx, m) for k, m in models.items()}
    t0 = time.time()
    draws, n = boot_noise_floor(cx, models, n_boot)
    print("noise floor: %d draws in %.0f s" % (n, time.time() - t0), flush=True)
    out = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), version=VERSION, tag=tag,
               units="mm; Dice [-]", run_flange_shift_mm=float(dk),
               comparators=dict(NONE=none_m.meta, K0=k0_m.meta),
               noise_floor=dict(n_draws=n, rot_sd_deg=fe.ROT_SD_DEG, trans_sd_mm_per_axis=round(fe.TR_SD_MM, 4),
                                equivalence_margin_msd_mm=fe.EQUIV_MM,
                                definition="the same random rigid perturbation is applied to SIM and to the "
                                           "comparator; the 5-50-95 percentiles of the sim-minus-comparator delta "
                                           "over the draws are the frame-choice noise floor (final_eval.cmd_boot)"),
               point_estimates={k: {f: {s: point[k][f]["structures"][s] for s in SCORED} for f in FRAMES}
                                for k in models},
               delta_sim_minus={})
    for comp in ("NONE", "K0"):
        out["delta_sim_minus"][comp] = dict(
            point={f: {s: {q: round(point["SIM"][f]["structures"][s][q] - point[comp][f]["structures"][s][q], 4)
                           for q in ("dice", "msd", "hd95", "hd95max")} for s in SCORED} for f in FRAMES},
            noise_floor=deltas_with_floor(draws, n, "SIM", comp))
    p = wjson("%s/%s/compare.json" % (EVALD, tag), out)
    for comp in ("NONE", "K0"):
        print("\nSIM minus %s" % comp)
        for f in FRAMES:
            for s in SCORED:
                d = out["delta_sim_minus"][comp]["noise_floor"][f][s]["msd"]
                v = out["delta_sim_minus"][comp]["noise_floor"][f][s]["msd_verdict"]
                print("  %-6s %-14s dMSD %+6.2f [%+6.2f, %+6.2f]  %s"
                      % (f, s, d["median"], d["p5"], d["p95"], v), flush=True)
    fig_summary(cx, tag, out)
    fig_overlays(cx, tag, models)
    print("wrote", p, flush=True)
    return out


# --------------------------------------------------------------------------------------------- figures
def cmd_figs(tag, cx=None):
    """Redraw both figures from an existing eval/<tag>/compare.json, without repeating the noise floor."""
    cx = cx or Ctx()
    cj = json.load(open("%s/%s/compare.json" % (EVALD, tag)))
    dk = cx.delta_key(cj.get("run_flange_shift_mm"))
    models = {"SIM": model_sim(cx, tag), "NONE": model_none(cx), "K0": model_k0(cx, dk)}
    fig_summary(cx, tag, cj)
    fig_overlays(cx, tag, models)


def fig_summary(cx, tag, cmp_out):
    ys = np.arange(len(SCORED))[::-1]
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 7.6))
    for r, f in enumerate(FRAMES):
        a = ax[r][0]
        for k, mk, col in (("SIM", "o", "#d62728"), ("NONE", "s", "#7f7f7f"), ("K0", "^", "#1f77b4")):
            v = [cmp_out["point_estimates"][k][f][s]["dice"] for s in SCORED]
            a.plot(v, ys, mk, color=col, label=k, ms=7, mfc=(col if k == "SIM" else "none"))
        a.set_yticks(ys); a.set_yticklabels(SCORED); a.set_xlim(0, 1); a.grid(alpha=.3)
        a.set_xlabel("Dice"); a.set_title("%s - Dice" % f)
        if r == 0:
            a.legend(loc="lower right", fontsize=8)
        for c, comp in enumerate(("NONE", "K0")):
            a = ax[r][c + 1]
            nf = cmp_out["delta_sim_minus"][comp]["noise_floor"][f]
            med = [nf[s]["msd"]["median"] for s in SCORED]
            lo = [nf[s]["msd"]["median"] - nf[s]["msd"]["p5"] for s in SCORED]
            hi = [nf[s]["msd"]["p95"] - nf[s]["msd"]["median"] for s in SCORED]
            a.axvspan(-fe.EQUIV_MM, fe.EQUIV_MM, color="#c7e9c0", alpha=.55, zorder=0,
                      label="+/-%.1f mm equivalence" % fe.EQUIV_MM)
            a.axvline(0, color="k", lw=.8, zorder=1)
            a.errorbar(med, ys, xerr=[lo, hi], fmt="o", color="#d62728", ms=6, capsize=3, zorder=3)
            a.set_yticks(ys); a.set_yticklabels([]); a.grid(alpha=.3)
            a.set_xlabel("MSD(SIM) - MSD(%s)   [mm]" % comp)
            a.set_title("%s - SIM minus %s" % (f, comp))
            if r == 0 and c == 0:
                a.legend(loc="lower right", fontsize=8)
    fig.suptitle("hybrid eval %s   |   left: absolute Dice   right: MSD difference with the %d-draw frame-perturbation "
                 "noise floor (5-95%%)" % (tag, cmp_out["noise_floor"]["n_draws"]), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .95))
    os.makedirs(FIGS, exist_ok=True)
    p = "%s/eval_%s_summary.png" % (FIGS, tag)
    fig.savefig(p, dpi=125); plt.close(fig)
    print("wrote", p, flush=True)
    return p


def _contour(ax, m2d, color, ls, lw=1.3):
    if m2d.any():
        ax.contour(m2d.astype(float), levels=[0.5], colors=[color], linestyles=[ls], linewidths=lw)


def fig_overlays(cx, tag, models):
    """BT sagittal / coronal / axial planes through the tandem; BT contours solid, models dashed."""
    bt = cx.bt
    img = np.asarray(nib.load(DATA + "/BT_MRI.nii").dataobj).astype(float)
    inv = np.linalg.inv(bt.aff[:3, :3])
    mid = bt.F + 0.5 * bt.L * bt.a
    ijk = np.rint((mid - bt.aff[:3, 3]) @ inv.T).astype(int)
    ijk = np.clip(ijk, 0, np.array(bt.shape) - 1)
    styles = {"SIM": ("dashed", 1.6), "NONE": ("dotted", 1.2), "K0": ("dashdot", 1.3)}
    order = ["SIM", "NONE", "K0"]
    for frame in FRAMES:
        masks = {}
        for k, mdl in models.items():
            fmap = cx.pelvis_map()[0] if frame == "PELVIS" else cx.eapp_map(mdl.dev)[0]
            masks[k] = model_masks(cx, mdl, fmap)
        # crop every panel to the structures of interest (the full FOV is mostly empty pelvis)
        roi = cx.bt_dev.copy()
        for s in ("corpus", "cervix", "vagina", "bladder", "rectum", "sigmoid"):
            roi |= cx.ref[s]
            for k in order:
                roi |= masks[k][s]
        for k in order:
            if masks[k]["device"] is not None:
                roi |= masks[k]["device"]
        ri = np.argwhere(roi); mg = np.array([18, 18, 12])
        rlo = np.maximum(ri.min(0) - mg, 0); rhi = np.minimum(ri.max(0) + mg, np.array(bt.shape) - 1)
        fig, ax = plt.subplots(1, 3, figsize=(16.5, 6.2))
        # slices are arr[idx].T, so rows/cols follow the remaining two index axes; the affine is
        # x = -1.125 i + c, y = -1.125 j + c, z = 1.6 k + c  (so i+ = patient left, j+ = posterior, k+ = superior)
        planes = (("sagittal  (x = %.1f mm)" % mid[0], 0, ijk[0], 1.6 / 1.125, "lower",
                   "anterior  <-->  posterior", "inferior  <-->  superior"),
                  ("coronal  (y = %.1f mm)" % mid[1], 1, ijk[1], 1.6 / 1.125, "lower",
                   "patient R  <-->  L", "inferior  <-->  superior"),
                  ("axial  (z = %.1f mm)" % mid[2], 2, ijk[2], 1.0, "upper",
                   "patient R  <-->  L", "anterior  <-->  posterior"))
        for a, (ttl, axis, idx, asp, orig, xl, yl) in zip(ax, planes):
            sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
            bg = img[sl].T
            a.imshow(bg, cmap="gray", origin=orig, aspect=asp,
                     vmin=np.percentile(bg, 1), vmax=np.percentile(bg, 99.5))
            for s in ("corpus", "cervix", "vagina", "bladder", "rectum", "sigmoid"):
                _contour(a, cx.ref[s][sl].T, COLORS[s], "-", 1.7)
                for k in order:
                    _contour(a, masks[k][s][sl].T, COLORS[s], styles[k][0], styles[k][1])
            _contour(a, cx.bt_dev[sl].T, "w", "-", 1.7)
            for k in order:
                if masks[k]["device"] is not None:
                    _contour(a, masks[k]["device"][sl].T, "w", styles[k][0], styles[k][1])
            ca, ra = ([1, 2] if axis == 0 else ([0, 2] if axis == 1 else [0, 1]))   # (col, row) index axes
            a.set_xlim(rlo[ca] - .5, rhi[ca] + .5)
            a.set_ylim((rhi[ra] + .5, rlo[ra] - .5) if orig == "upper" else (rlo[ra] - .5, rhi[ra] + .5))
            a.set_title(ttl, fontsize=9); a.set_xlabel(xl, fontsize=8); a.set_ylabel(yl, fontsize=8)
            a.set_xticks([]); a.set_yticks([])
        h = [plt.Line2D([], [], color=COLORS[s], ls="-", label=s) for s in
             ("corpus", "cervix", "vagina", "bladder", "rectum", "sigmoid")]
        h += [plt.Line2D([], [], color="#888888", ls="-", label="device (white on the image)")]
        h += [plt.Line2D([], [], color="k", ls="-", label="BT label (solid)")]
        h += [plt.Line2D([], [], color="k", ls=styles[k][0], label="%s" % k) for k in order]
        fig.legend(handles=h, loc="lower center", ncol=11, fontsize=8, frameon=False)
        fig.suptitle("%s  -  BT planes through the tandem, %s frame   (BT labels solid, models dashed)"
                     % (tag, frame), fontsize=11)
        fig.tight_layout(rect=(0, .05, 1, .95))
        os.makedirs(FIGS, exist_ok=True)
        p = "%s/eval_%s_overlay_%s.png" % (FIGS, tag, frame)
        fig.savefig(p, dpi=125); plt.close(fig)
        print("wrote", p, flush=True)


# --------------------------------------------------------------------------------------------- cmd: field
def probe_body(cx, body, u, shape, aff, mdir=None):
    """Barycentric interpolation of the nodal displacement inside one body's tets, on the preBT grid.
    Returns (u_grid (n,3), valid (n,), sel_index_tuple) restricted to the body's index bbox.
    `mdir` is the body's mesh directory for this run (run_mesh_dirs); it defaults to meshes/<body>."""
    ug, pts = read_ug("%s/tets.vtk" % (mdir or "%s/%s" % (MESHES, body)))
    if len(pts) != len(u):
        raise SystemExit("%s: tets.vtk has %d points, %s_u.npy has %d" % (body, len(pts), body, len(u)))
    arr = ns.numpy_to_vtk(np.ascontiguousarray(u, float), deep=1); arr.SetName("u")
    ug.GetPointData().AddArray(arr)
    inv = np.linalg.inv(aff[:3, :3])
    idx = (pts - aff[:3, 3]) @ inv.T
    lo = np.maximum(np.floor(idx.min(0)).astype(int) - 2, 0)
    hi = np.minimum(np.ceil(idx.max(0)).astype(int) + 3, np.array(shape))
    g = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i]) for i in range(3)], indexing="ij"), -1)
    X = g.reshape(-1, 3) @ aff[:3, :3].T + aff[:3, 3]
    vp = vtk.vtkPoints(); vp.SetData(ns.numpy_to_vtk(np.ascontiguousarray(X, float), deep=1))
    pd = vtk.vtkPolyData(); pd.SetPoints(vp)
    pr = vtk.vtkProbeFilter(); pr.SetInputData(pd); pr.SetSourceData(ug); pr.Update()
    o = pr.GetOutput()
    uu = ns.vtk_to_numpy(o.GetPointData().GetArray("u")).reshape(-1, 3)
    ok = ns.vtk_to_numpy(o.GetPointData().GetArray("vtkValidPointMask")).astype(bool)
    return uu, ok, g.reshape(-1, 3)


def surface_cloud(cx, u_by_body, rest=None, meta=None):
    """Dense point cloud of every body's surface (vertices + triangle centroids) with its displacement, for the
    outside extrapolation.  Vertices are mapped to tet nodes by meta['surface_obj_vertex_to_tet_node'].
    `rest`/`meta` are the RUN's own meshes (run_mesh_dirs); they default to the Stage-1 ones."""
    rest = rest if rest is not None else cx.rest
    meta = meta if meta is not None else cx.meta
    Ps, Us = [], []
    for b in BODIES:
        V, F = rest[b]
        m = np.asarray(meta[b]["surface_obj_vertex_to_tet_node"], int)
        uv = u_by_body[b][m]
        Ps.append(V); Us.append(uv)
        Ps.append(V[F].mean(1)); Us.append(uv[F].mean(1))
    return np.vstack(Ps), np.vstack(Us)


def sample_field(U, aff, X):
    inv = np.linalg.inv(aff)
    ijk = (np.asarray(X, float) @ inv[:3, :3].T + inv[:3, 3]).T
    return np.stack([ndi.map_coordinates(U[..., c], ijk, order=1, mode="nearest") for c in range(3)], -1)


def backward_map(U, aff, shape, src_mask, prio=None, iters=INV_ITERS, tol_vox=1.6, k=8):
    """Backward map v(y) such that x = y - v(y) satisfies x + u(x) = y, over the sources selected by `src_mask`.

    Two things defeat a naive fixed point v <- u(y - v) started at v = 0:
      * the corpus moves ~30 mm, so the first sample u(y) is taken well outside the body (where the field is tapered
        to zero) and the iteration never walks back to the source;
      * the map is NOT injective for the kinematic comparators -- K0 moves the corpus rigidly while every other body
        stays put, so the corpus sweeps through stationary material and two sources share one target.
    So the map is seeded by SCATTER (every source voxel x is pushed forward to x + u(x); the nearest pushed-forward
    point of a target gives v = u(x)), the branch is chosen by `prio` (lower wins: the meshing priority
    corpus > cervix > vagina > bladder > rectum > sigmoid, then the extrapolated band), and the fixed-point sweeps
    that polish the seed are STEP-LIMITED to one voxel so they cannot hop to the other branch.
    Targets with no pushed-forward point within tol_vox voxels keep v = 0."""
    sp = np.sqrt((aff[:3, :3] ** 2).sum(0))
    tol = float(tol_vox * sp.max()); lim = float(sp.max())
    V = np.zeros(tuple(shape) + (3,), np.float32)
    covered = np.zeros(shape, bool)
    si = np.argwhere(src_mask)
    if not len(si):
        return V, covered, dict(n_support=0, n_covered=0, max_residual_mm=0.0)
    Xs = si @ aff[:3, :3].T + aff[:3, 3]
    Us = U[src_mask].astype(np.float64)
    Ps = Xs + Us                                        # deformed (pushed-forward) positions
    tree = cKDTree(Ps)
    pr = (prio[src_mask].astype(np.float64) if prio is not None else np.zeros(len(si)))
    inv = np.linalg.inv(aff)
    pidx = Ps @ inv[:3, :3].T + inv[:3, 3]
    lo = np.maximum(np.floor(pidx.min(0)).astype(int) - 2, 0)
    hi = np.minimum(np.ceil(pidx.max(0)).astype(int) + 3, np.array(shape))
    g = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i]) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    kk = int(min(k, len(Ps))); resb = []; namb = 0; nbod = 0
    nbody = len(BODIES)
    for s in range(0, len(g), 300000):
        ch = g[s:s + 300000]
        Y = ch @ aff[:3, :3].T + aff[:3, 3]
        d, i = tree.query(Y, k=kk, workers=-1)
        if kk == 1:
            d = d[:, None]; i = i[:, None]
        good = d <= tol
        pri = pr[i]
        rank = pri * 1e6 + d                            # priority first, then distance
        rank[~good] = np.inf
        j = np.argmin(rank, 1); ar = np.arange(len(ch))
        ok = good[ar, j]
        best = pri[ar, j]
        # inverse-distance average over the neighbours of the WINNING branch only.  For a rigid (affine) motion this
        # is exact: sum_w u(x_i) = u(sum_w x_i) and the weighted centroid is the preimage of y.
        same = good & (pri == best[:, None])
        w = np.where(same, 1.0 / np.maximum(d, 1e-6) ** 2, 0.0)
        sw = w.sum(1)
        v = np.zeros_like(Y)
        if ok.any():
            v[ok] = np.einsum("nk,nkc->nc", w[ok], Us[i[ok]]) / sw[ok][:, None]
            for _ in range(iters):                      # optional polish (off by default: the band folds)
                stp = sample_field(U, aff, Y - v) - v
                nn = np.linalg.norm(stp, axis=1)
                v = v + stp * np.minimum(1.0, lim / np.maximum(nn, 1e-9))[:, None]
                v[~ok] = 0.0
            r = np.linalg.norm(sample_field(U, aff, Y - v) - v, axis=1)[ok]
            isb = best[ok] < nbody
            if isb.any():
                resb.append(r[isb].astype(np.float32)); nbod += int(isb.sum())
            # genuine branch ambiguity: a within-tol neighbour whose displacement differs from the chosen one by
            # more than a voxel (adjacent lattice neighbours always sit within tol, so counting them means nothing)
            spread = np.linalg.norm(Us[i] - v[:, None, :], axis=2)
            namb += int(np.sum(((spread > 2.0) & good).any(1) & ok))
        V[tuple(ch.T)] = v
        covered[tuple(ch[ok].T)] = True
    rb = np.concatenate(resb) if resb else np.zeros(1, np.float32)
    return V, covered, dict(
        residual_in_bodies_mm=dict(median=round(float(np.median(rb)), 6), p95=round(float(np.percentile(rb, 95)), 6),
                                   max=round(float(rb.max()), 6), n=int(nbod)),
        n_covered=int(covered.sum()), n_support=int(src_mask.sum()),
        n_targets_branch_ambiguous=int(namb), tol_mm=round(tol, 4), polish_sweeps=iters,
        note="residual = |u(y - v) - v| at the covered targets whose source is a BODY (0 = exact inverse); the "
             "median/p95 are the meaningful numbers. The max is dominated by the interface between a MOVING body's "
             "extrapolated band and a STATIC body's interior, where the field is genuinely discontinuous (the band "
             "carries the moving body's displacement right up to the static body's wall), so trilinear sampling "
             "there blends two very different values. Over the %g mm band the taper also drives |grad u| > 1, so the "
             "map folds and no single-valued inverse exists; n_targets_branch_ambiguous counts targets with a "
             "within-tol source whose displacement differs by more than 2 mm from the chosen branch, which the "
             "priority order resolves." % BAND_MM)


def warp_with_backward(vol, aff, V, order):
    """warped(y) = vol(y - V(y)), sampled in index space.  Chunked along i to bound the transient memory."""
    shape = vol.shape
    src = vol.astype(np.float32) if order else vol
    out = np.zeros(shape, src.dtype)
    inv = np.linalg.inv(aff)
    jk = np.stack(np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing="ij"), -1).reshape(-1, 2)
    for i0 in range(0, shape[0], 16):
        i1 = min(i0 + 16, shape[0])
        g = np.concatenate([np.repeat(np.arange(i0, i1), len(jk))[:, None],
                            np.tile(jk, (i1 - i0, 1))], 1).astype(float)
        Y = g @ aff[:3, :3].T + aff[:3, 3]
        X = Y - V[i0:i1].reshape(-1, 3)
        ijk = (X @ inv[:3, :3].T + inv[:3, 3]).T
        out[i0:i1] = ndi.map_coordinates(src, ijk, order=order, mode="constant").reshape(i1 - i0, shape[1], shape[2])
    return out


def cmd_field(tag, cx=None, ref_dvf=None, band=BAND_MM):
    cx = cx or Ctx()
    rd = "%s/%s" % (RUNS, tag)
    od = "%s/%s/field" % (EVALD, tag)
    os.makedirs(od, exist_ok=True)
    shape = cx.pre_shape; aff = cx.pre_aff; sp = cx.pre_sp
    # Per-body mesh root for THIS run (a Stage-2a wall run's vagina is not meshes/vagina): final/<body>_u.npy is in
    # the node order of the mesh the run used, so the rest surfaces and metas are re-read from there.
    md = run_mesh_dirs(tag)
    rest = {b: geom.read_obj(md[b] + "/surface.obj") for b in BODIES}
    rmeta = {b: json.load(open(md[b] + "/meta.json")) for b in BODIES}
    u_by_body = {}
    for b in BODIES:
        p = "%s/final/%s_u.npy" % (rd, b)
        if not os.path.exists(p):
            raise SystemExit("missing %s (the scene module writes final/<body>_u.npy)" % p)
        u_by_body[b] = np.load(p).astype(float).reshape(-1, 3)
    # ---------------- inside the bodies: barycentric in the tets, priority corpus > cervix > ... on overlaps
    U = np.zeros(tuple(shape) + (3,), np.float32)
    inside = np.zeros(shape, bool)
    owner = np.zeros(shape, np.uint8)                 # 0 = none, 1..6 = body rank+1 (meshing priority)
    per_body = {}
    for rank, b in enumerate(BODIES):
        uu, ok, gi = probe_body(cx, b, u_by_body[b], shape, aff, md[b])
        sel = gi[ok]
        cur = inside[tuple(sel.T)]
        new = ~cur                                    # earlier (higher priority) body wins
        tgt = sel[new]
        U[tuple(tgt.T)] = uu[ok][new]
        inside[tuple(tgt.T)] = True
        owner[tuple(tgt.T)] = rank + 1
        per_body[b] = dict(voxels_probed=int(ok.sum()), voxels_assigned=int(new.sum()),
                           u_max_mm=round(float(np.linalg.norm(u_by_body[b], axis=1).max()), 4),
                           u_mean_mm=round(float(np.linalg.norm(u_by_body[b], axis=1).mean()), 4),
                           n_nodes=int(len(u_by_body[b])))
    # ---------------- outside: inverse-distance from the nearest body-surface points, tapered to 0 at `band`
    d_out = ndi.distance_transform_edt(~inside, sampling=sp)
    cand = (~inside) & (d_out <= band)
    Pc, Uc = surface_cloud(cx, u_by_body, rest, rmeta)
    tree = cKDTree(Pc)
    ci = np.argwhere(cand)
    extrap = np.zeros(shape, bool)
    kk = min(IDW_K, len(Pc))
    for lo in range(0, len(ci), 200000):                 # chunked: the k-neighbour gather is the memory peak
        ch = ci[lo:lo + 200000]
        Xc = ch @ aff[:3, :3].T + aff[:3, 3]
        dd, ii = tree.query(Xc, k=kk, workers=-1)
        w = 1.0 / np.maximum(dd, 1e-6) ** 2
        uex = np.einsum("nk,nkc->nc", w, Uc[ii]) / w.sum(1)[:, None]
        taper = np.clip(1.0 - dd[:, 0] / band, 0.0, 1.0) ** 2
        uex *= taper[:, None]
        U[tuple(ch.T)] = uex
        extrap[tuple(ch[taper > 0].T)] = True
    mask = np.zeros(shape, np.uint8); mask[extrap] = 1; mask[inside] = 2
    # ---------------- write the field + mask
    nib.save(nib.Nifti1Image(U, aff), od + "/u_preBT.nii.gz")
    nib.save(nib.Nifti1Image(mask, aff), od + "/mask.nii.gz")
    # ---------------- inverse (backward) map, restricted to the active box
    prio = np.where(owner > 0, owner.astype(np.int16) - 1, len(BODIES)).astype(np.int16)  # bodies 0..5, band 6
    supp = mask > 0
    Vb, cov, invstat = backward_map(U, aff, shape, supp, prio=prio)
    # ---------------- warped image and warped labels
    # The deformation maps the support S onto `cov`.  A target is: in cov -> pull back through the map; outside cov
    # and outside S -> material that never moved, keep it; outside cov but inside S -> the material left and nothing
    # arrived (a vacuum), set to background.  Without this the moving bodies would be duplicated at their rest
    # position, because v = 0 there makes the pull-back the identity.
    place = lambda w, src: np.where(cov, w, np.where(supp, 0, src))    # noqa: E731
    pre_img = np.asarray(nib.load(DATA + "/preBT_MRI.nii").dataobj).astype(np.float32)
    nib.save(nib.Nifti1Image(place(warp_with_backward(pre_img, aff, Vb, 1), pre_img).astype(np.float32), aff),
             od + "/warped_preBT_MRI.nii.gz")
    lab_names = ("uterus", "HR-CTV", "vagina", "bladder", "rectum", "sigmoid")
    lab = np.zeros(shape, np.uint8)
    for i, n in enumerate(lab_names, 1):
        m = np.asarray(nib.load("%s/preBT_MRI_label_%s.nii" % (DATA, n)).dataobj) > 0
        lab[m & (lab == 0)] = i
    nib.save(nib.Nifti1Image(place(warp_with_backward(lab, aff, Vb, 0), lab).astype(np.uint8), aff),
             od + "/warped_preBT_labels.nii.gz")
    # ---------------- sanity check: warping the REST body masks must reproduce final/<body>.obj
    # Each body is warped with its OWN backward map (sources = the voxels the field owns for that body).  A single
    # global map cannot serve this check when bodies interpenetrate in the deformed configuration (K0 moves the
    # corpus through the static rectum), because the global map necessarily picks one branch per target.
    check = {}
    for rank, b in enumerate(BODIES):
        V0, F0 = rest[b]
        m0 = ev.voxelize(V0, F0, shape, aff)
        fp = "%s/final/%s.obj" % (rd, b)
        if not os.path.exists(fp):
            continue
        Vd, Fd = geom.read_obj(fp)
        mt = ev.voxelize(Vd, Fd, shape, aff)
        Vbb, covb, _ = backward_map(U, aff, shape, owner == rank + 1)
        mw = (warp_with_backward(m0.astype(np.uint8), aff, Vbb, 0) > 0) & covb
        r = dict(dice=ev.dice(mt, mw)); r.update(ev.surf_dists(mt, mw, sp))
        r["vol_warped_cc"] = round(float(mw.sum() * np.prod(sp) / 1000.0), 3)
        r["vol_target_cc"] = round(float(mt.sum() * np.prod(sp) / 1000.0), 3)
        check[b] = {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in r.items()}
    nrm = np.linalg.norm(U, axis=-1)
    stats = dict(
        written=time.strftime("%Y-%m-%d %H:%M:%S"), version=VERSION, tag=tag,
        units="mm", frame="preBT world RAS mm (nibabel affine of preBT_MRI.nii; x=R, y=A, z=S)",
        file="u_preBT.nii.gz",
        format=dict(shape=list(U.shape), dtype="float32", affine=aff.tolist(),
                    ordering="4-D NIfTI (i, j, k, c); c = 0,1,2 = displacement along world +x (R), +y (A), +z (S) "
                             "in millimetres",
                    convention="FORWARD (Lagrangian) displacement defined at preBT material positions: "
                               "x_deformed = x_preBT + u(x_preBT)",
                    mask="mask.nii.gz: 0 = zero field (beyond the band), 1 = extrapolated, 2 = inside a body"),
        method=dict(inside="barycentric interpolation of the nodal displacement inside each body's tetrahedra "
                           "(vtkProbeFilter on meshes/<body>/tets.vtk); overlaps resolved by the meshing priority "
                           "corpus > cervix > vagina > bladder > rectum > sigmoid",
                    outside="inverse-distance (k=%d, 1/d^2) from the nearest body-surface points (surface vertices "
                            "+ triangle centroids, displacement taken from the tet nodes), multiplied by the taper "
                            "(1 - d/%g)^2 so the field decays continuously to zero at the band edge" % (IDW_K, band),
                    band_mm=band,
                    inverse="backward map v(y) with y - v(y) = x and x + u(x) = y, seeded by scatter (every "
                            "supported voxel x is pushed forward to x + u(x); the nearest pushed-forward point of a "
                            "target gives v = u(x)), the branch chosen by the meshing priority (bodies before the "
                            "extrapolated band), then polished by %d fixed-point sweeps v <- u(y - v) whose step is "
                            "limited to one voxel so they stay on the seeded branch. A plain fixed point from v = 0 "
                            "does not converge at this displacement (~30 mm), and the map is not injective wherever "
                            "a body is moved through material that is not (see inverse.n_targets_multi_source). "
                            "warped(y) = preBT(y - v(y)), trilinear for the image and nearest for the labels"
                            % INV_ITERS),
        per_body=per_body,
        mesh_dirs={b: os.path.relpath(md[b], HYB).replace("\\", "/") for b in BODIES},
        voxels=dict(inside=int(inside.sum()), extrapolated=int(extrap.sum()),
                    zero=int(mask.size - inside.sum() - extrap.sum()), total=int(mask.size)),
        magnitude_mm=dict(max=round(float(nrm.max()), 4), mean_inside=round(float(nrm[inside].mean()), 4),
                          p95_inside=round(float(np.percentile(nrm[inside], 95)), 4) if inside.any() else None,
                          mean_extrapolated=round(float(nrm[extrap].mean()), 4) if extrap.any() else 0.0),
        inverse=invstat,
        warped_label_check=dict(
            definition="voxelise meshes/<body>/surface.obj on the preBT grid, warp it with THAT BODY'S OWN backward "
                       "map (sources = the voxels the field owns for that body) and compare with the voxelisation "
                       "of runs/<tag>/final/<body>.obj on the same grid. The per-body map is used because the "
                       "global map must pick a single branch per target where bodies interpenetrate.",
            per_body=check,
            pass_all=bool(check and all(v["dice"] > 0.95 and v["msd"] < 0.5 for v in check.values()))),
        outputs=["u_preBT.nii.gz", "mask.nii.gz", "warped_preBT_MRI.nii.gz", "warped_preBT_labels.nii.gz",
                 "field_stats.json"],
        warped_label_codes={str(i): n for i, n in enumerate(lab_names, 1)})
    if ref_dvf:
        stats["ref_dvf"] = compare_ref_dvf(cx, ref_dvf, U, aff, shape, inside)
    else:
        stats["ref_dvf"] = ref_dvf_hook()
    p = wjson(od + "/field_stats.json", stats)
    ri = invstat["residual_in_bodies_mm"]
    print("field %s: inside %d, extrapolated %d voxels; |u| max %.2f mm; inverse residual in bodies "
          "median %.3f / p95 %.3f / max %.2f mm (%d covered, %d branch-ambiguous)"
          % (tag, inside.sum(), extrap.sum(), nrm.max(), ri["median"], ri["p95"], ri["max"],
             invstat["n_covered"], invstat["n_targets_branch_ambiguous"]))
    for b, v in check.items():
        print("  warped-label check %-8s dice %.4f  msd %.3f  hd95 %.3f" % (b, v["dice"], v["msd"], v["hd95"]))
    print("  warped-label check pass_all =", stats["warped_label_check"]["pass_all"])
    print("wrote", p, flush=True)
    return stats


def ref_dvf_hook():
    msg = dict(
        status="no reference field supplied (--ref-dvf absent)",
        hook="python hybrid/eval_hybrid.py field --tag <tag> --ref-dvf <path>",
        accepted_now=dict(
            format="NIfTI vector field (.nii / .nii.gz), 4-D (i, j, k, 3) or 5-D (i, j, k, 1, 3)",
            units="millimetres",
            convention="FORWARD displacement at preBT material positions, components along world +x (R), +y (A), "
                       "+z (S). A field on a different grid is resampled onto the preBT grid (trilinear) using its "
                       "own affine; a field with a different component convention must be converted first.",
            reported="per-structure statistics of |u_pred - u_ref| (mean, median, p95, max) over each body's rest "
                     "mask, plus the same over the inside-body region and the per-component signed bias"),
        to_be_added=dict(
            format="DICOM REG (spatial registration / deformable registration object) exported by RayStation",
            blocked_on="the exact RayStation export flavour is not yet known: DICOM Spatial Registration (rigid) vs "
                       "Deformable Spatial Registration (grid), the grid frame of reference, and whether the vectors "
                       "are stored pre- or post-image-orientation. Supply one export and the reader is a small "
                       "addition (pydicom -> DeformableRegistrationSequence -> DeformableRegistrationGridSequence "
                       "-> VectorGridData, then the same comparison code)."))
    print("\n[--ref-dvf] " + msg["status"])
    print("  hook:        " + msg["hook"])
    print("  accepted:    NIfTI vector field, 4-D (i,j,k,3) or 5-D (i,j,k,1,3), mm, forward displacement, RAS "
          "components (resampled onto the preBT grid via its own affine)")
    print("  to be added: DICOM REG (RayStation) - blocked on knowing the export format "
          "(rigid SRO vs deformable grid; VectorGridData layout)", flush=True)
    return msg


def compare_ref_dvf(cx, path, U, aff, shape, inside):
    im = nib.load(path)
    A = np.asarray(im.dataobj).astype(np.float32)
    A = np.squeeze(A)
    if A.ndim != 4 or A.shape[-1] != 3:
        raise SystemExit("--ref-dvf: expected a 4-D vector field with 3 components, got shape %s" % (A.shape,))
    same = (A.shape[:3] == tuple(shape)) and np.allclose(im.affine, aff, atol=1e-4)
    if same:
        Ur = A
    else:                                    # resample the reference onto the preBT grid
        g = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3)
        X = g @ aff[:3, :3].T + aff[:3, 3]
        inv = np.linalg.inv(im.affine)
        ijk = (X @ inv[:3, :3].T + inv[:3, 3]).T
        Ur = np.stack([ndi.map_coordinates(A[..., c], ijk, order=1, mode="constant") for c in range(3)],
                      -1).reshape(tuple(shape) + (3,))
    D = U - Ur
    n = np.linalg.norm(D, axis=-1)
    out = dict(status="compared", ref=path, ref_shape=list(A.shape), ref_affine=im.affine.tolist(),
               resampled=bool(not same),
               convention_assumed="forward displacement in mm, components along world +x (R), +y (A), +z (S)",
               overall=_err_stats(n, D, inside), per_structure={})
    for b in BODIES:
        V, F = cx.rest[b]
        m = ev.voxelize(V, F, shape, aff)
        out["per_structure"][b] = _err_stats(n, D, m)
    return out


def _err_stats(n, D, m):
    if not m.any():
        return dict(n_voxels=0)
    v = n[m]
    return dict(n_voxels=int(m.sum()), mean_mm=round(float(v.mean()), 4), median_mm=round(float(np.median(v)), 4),
                p95_mm=round(float(np.percentile(v, 95)), 4), max_mm=round(float(v.max()), 4),
                signed_bias_mm=[round(float(D[..., c][m].mean()), 4) for c in range(3)])


# --------------------------------------------------------------------------------------------- cmd: selftest
def cmd_selftest(cx=None, delta=None, n_boot=None):
    """Build runs/SELFTEST_K0 by applying the K0 rigid motion to the corpus and cervix nodes (u = rigid
    displacement) with zero displacement elsewhere, then run score / compare / field end to end.  SIM must then
    reproduce K0 at the same Delta to machine precision."""
    cx = cx or Ctx()
    dk = cx.delta_key(delta)
    T = cx.corpus_T(dk); R = T[:3, :3]; t = T[:3, 3]
    tag = "SELFTEST_K0"
    rd = "%s/%s" % (RUNS, tag)
    os.makedirs(rd + "/final", exist_ok=True)
    for b in BODIES:
        _, pts = read_ug("%s/%s/tets.vtk" % (MESHES, b))
        u = (pts @ R.T + t - pts) if b in K0_BODIES else np.zeros_like(pts)
        np.save("%s/final/%s_u.npy" % (rd, b), u.astype(np.float64))
        V, F = cx.rest[b]
        m = np.asarray(cx.meta[b]["surface_obj_vertex_to_tet_node"], int)
        geom.write_obj("%s/final/%s.obj" % (rd, b), V + u[m], F,
                       header="SELFTEST_K0 %s: preBT world RAS mm; rest surface + u[surface_obj_vertex_to_tet_node]" % b)
    dev = cx.rule_device(dk)
    wjson(rd + "/device_final.json",
          dict(units="mm", frame="preBT world RAS mm", flange=dev["flange"], tube_axis=dev["axis"],
               shaft_axis=dev["shaft_axis"], x_app=dev["R_rows"][0], y_app=dev["R_rows"][1],
               R_rows=dev["R_rows"], tip=dev["tip"], L_iu_mm=float(cx.bt.L), flange_shift_mm=float(dk),
               source="derived from hybrid/applicator/pose.json by eval_hybrid.py selftest",
               convention="p_world = flange + p_app @ R_rows (rows = applicator x, y, z axes in preBT world)"))
    wjson(rd + "/cfg.json",
          dict(tag=tag, synthetic=True, flange_shift_mm=float(dk),
               note="synthetic run: the K0 rigid motion applied to the corpus and cervix nodes, zero elsewhere. "
                    "Not a SOFA run; it exists to test eval_hybrid end to end.",
               generator="%s selftest" % VERSION))
    print("built %s (Delta = %s mm)\n" % (rd, dk), flush=True)
    s = cmd_score(tag, cx)
    c = cmd_compare(tag, cx, n_boot=n_boot)
    f = cmd_field(tag, cx)
    # ---- verdicts
    ver = {}
    k0 = None
    if os.path.exists("%s/comparators/metrics.json" % EVALD):
        cj = json.load(open("%s/comparators/metrics.json" % EVALD))
        k0 = cj["models"].get("K0@%s" % dk)
    worst = 0.0
    if k0:
        for fr in FRAMES:
            for st in SCORED:
                for q in ("dice", "msd", "hd95"):
                    worst = max(worst, abs(s["frames"][fr]["structures"][st][q]
                                           - k0["frames"][fr]["structures"][st][q]))
        ver["SIM_equals_K0_max_abs_metric_diff"] = round(worst, 6)
        ver["SIM_equals_K0"] = bool(worst < 2e-3)     # the metrics are stored rounded to 4 dp; sub-voxel agreement
    dmax = 0.0
    for comp in ("K0",):
        for fr in FRAMES:
            for st in SCORED:
                dmax = max(dmax, abs(c["delta_sim_minus"][comp]["point"][fr][st]["msd"]))
    ver["delta_SIM_minus_K0_max_abs_msd"] = round(dmax, 6)
    ver["warped_label_check"] = f["warped_label_check"]["per_body"]
    ver["warped_label_pass_all"] = f["warped_label_check"]["pass_all"]
    ver["inverse_residual_in_bodies_mm"] = f["inverse"]["residual_in_bodies_mm"]
    ver["PASS"] = bool(ver.get("SIM_equals_K0", True) and dmax < 2e-3 and f["warped_label_check"]["pass_all"])
    p = wjson("%s/%s/selftest.json" % (EVALD, tag), dict(written=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                         version=VERSION, delta_mm=float(dk), verdict=ver))
    print("\n=================== SELF-TEST")
    print(json.dumps(_j(ver), indent=1))
    print("wrote", p, flush=True)
    return ver


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="hybrid pelvis simulator evaluation (CONTRACT section 6)")
    ap.add_argument("cmd", choices=["score", "comparators", "compare", "field", "selftest", "figs"])
    ap.add_argument("--tag")
    ap.add_argument("--ref-dvf", dest="ref_dvf")
    ap.add_argument("--deltas", nargs="*", help="restrict the comparator Delta list")
    ap.add_argument("--delta", type=float, help="selftest: Delta to use (default = pose.json default)")
    ap.add_argument("--n-boot", type=int, help="override the number of noise-floor draws (default %d)" % fe.N_BOOT)
    ap.add_argument("--band-mm", type=float, default=BAND_MM)
    a = ap.parse_args()
    os.makedirs(EVALD, exist_ok=True); os.makedirs(FIGS, exist_ok=True); os.makedirs(LOGS, exist_ok=True)
    cx = Ctx()
    if a.cmd == "comparators":
        cmd_comparators(cx, deltas=a.deltas)
    elif a.cmd == "selftest":
        cmd_selftest(cx, delta=a.delta, n_boot=a.n_boot)
    else:
        if not a.tag:
            sys.exit("--tag is required for %s" % a.cmd)
        if a.cmd == "score":
            cmd_score(a.tag, cx)
        elif a.cmd == "compare":
            cmd_compare(a.tag, cx, n_boot=a.n_boot)
        elif a.cmd == "field":
            cmd_field(a.tag, cx, ref_dvf=a.ref_dvf, band=a.band_mm)
        elif a.cmd == "figs":
            cmd_figs(a.tag, cx)


if __name__ == "__main__":
    main()
