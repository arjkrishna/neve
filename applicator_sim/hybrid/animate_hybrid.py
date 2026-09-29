"""HOST renderer of a hybrid run exported with cfg `frame_every` > 0 (`run_hybrid.write_frame`).  CONTRACT.md
section 4 outputs -> a video of the insertion.  Units mm; frame preBT world RAS (x = R, y = A, z = S).  Nothing here
touches the simulation or the patient data (the MRI is read read-only); every output goes to hybrid/figs/anim/.

    python   hybrid/animate_hybrid.py prep   --tag HA1            # py3.13 (nibabel): the MRI backdrop plane, optional
    py -3.11 hybrid/animate_hybrid.py render --tag HA1            # pyvista OFF-SCREEN: one PNG per frame + the GIF
    python   hybrid/animate_hybrid.py mp4    --tag HA1            # py3.13 + imageio(-ffmpeg from --pk): the MP4

    py -3.11 hybrid/animate_hybrid.py render --tag HA1 --zoom --bodies vagina,cervix --name HA1_vagina
The two interpreters carry different packages (py3.13: nibabel/scipy/imageio; py3.11: pyvista), hence the modes.

Each frame is two panels:
  LEFT   a sagittal-like view along the patient LR axis (camera on the patient's left looking +x, so ANTERIOR is on
         the LEFT and SUPERIOR is up), parallel projection, with the anatomy CUT at the device's own sagittal plane
         and the near (patient-left) half removed, so the device inside the tissue is visible (--no-clip disables).
         A stretch of the tube / shaft the cut removes all or nearly all of (it lies in front of the section:
         TF0c's curved vaginal shaft; the tube early in a tandem-first run, since the plane is the LAST frame's) is
         drawn whole, see-through, with a dark silhouette, and the caption says so (front_cells).
  RIGHT  an oblique 3-D view from the patient's left-anterior-superior, nothing cut, bladder/sigmoid translucent.
Both carry the six organs in the scene's own colours (meshes/bodies.json), the device in grey, and the REST state as
a faint wireframe, so the motion is visible.  The cameras are fixed for the whole run.

Device parts are the run's own (run_extra_parts): the tandem body (tube + shaft) always; the ovoid body (caps, and the
rods / packing when the run loaded them) only when the run had one.  A tandem-only run (applicator.json params
tandem_only, e.g. applicator_v4 for TF0; or cfg device_ovoids false / ovoid_mode "none" / device_parts) has none,
and every ovoid-dependent actor, legend entry and camera point is simply left out.  cfg device_part_files (G32: the
26 mm caps) is honoured as the scene honours it.  Render through the run's own tree: hybrid/render_tree.py.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True                      # never leave __pycache__ in the repo
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config  # noqa: E402
import geom  # noqa: E402

BODIES = ["corpus", "cervix", "vagina", "bladder", "rectum", "sigmoid"]
TANDEM = ["tube", "shaft"]
OVOIDS = ["ovoid_L", "ovoid_R"]
COL_TANDEM = (0.16, 0.16, 0.19)                     # the device is grey: near-black tube/shaft ...
COL_OVOID = (0.62, 0.62, 0.66)                      # ... and light grey ovoids, so the two parts stay separable
# per-panel opacity: the bladder is 251 cc and sits in front of everything, so it is the one that must stay see-through
OPACITY_CUT = dict(corpus=0.97, cervix=0.97, vagina=1.0, bladder=0.55, rectum=0.75, sigmoid=0.55)
OPACITY_3D = dict(corpus=0.92, cervix=0.95, vagina=1.0, bladder=0.20, rectum=0.50, sigmoid=0.28)
# the ovoid assembly is 39 mm across and sits exactly where the tissue of interest is, so an opaque one hides the
# vagina/portio it is seating against -- the one thing this animation exists to show.  Translucent in the cut view.
OVOID_OPACITY = (0.50, 0.85)                        # (cut panel, oblique panel)
# The sagittal cut keeps x >= x_cut (the plane through the mid-point of the tube).  A tandem-body stretch whose whole
# cross-section lies IN FRONT of that plane (camera side) would vanish with the removed half: TF0c's curved vaginal
# shaft lies 0-8 mm in front at step 192 (0.00 of its vertices kept) and the 22 mm of the applicator_v4 tube below the
# flange up to 6 mm (the tube leans in x), so the tandem rod disappeared from the section.  A stretch the cut removes
# all or nearly all of -- each 1 mm cross-section station keeping less than FRONT_KEEP_FRAC of its vertices, i.e.
# the tandem axis more than 0.59 r in front of the plane (a cylinder keeps acos(d / r) / pi) and the clipped band
# narrower than 81 % of its diameter -- is drawn WHOLE, see-through, with a dark silhouette, over the clipped part
# (which is drawn as before).  Per station, not per part: a cut through a part keeps about half of its vertices
# wherever it cuts it (G32 step 180: tube 0.51, straight rod 0.45 of the whole part), so a per-part "less than half
# kept" test would redraw G32's rod.  MEASURED on G32, every station of every frame: at least 0.42 kept in the
# labelled still's cut, 0.37 in the video's (its plane is the last frame's): nothing is redrawn, its renders are
# unchanged.  TF0c: such a stretch in all 102 frames of the still's cut (the whole shaft at steps 192 / 213; the
# tube below the flange or the shaft's far end, outside the body, elsewhere) and in 71 of the video's.  The ovoid
# body is left out: its far cap and rod lie in the removed half BY DESIGN (G32: left cap 13 mm, left rod 8 mm in
# front; the labels say so).
FRONT_PARTS = ("tube", "shaft")
FRONT_KEEP_FRAC = 0.30
FRONT_OPACITY = 0.55
COL_FRONT_EDGE = (0.0, 0.0, 0.0)
STATION_MM = 1.0                                    # cross-section bins along a part's own axis


def part_stations(part, V_app, app=None):
    """Station (1 mm bin of the arclength along the part's own axis) of every vertex of a tandem-body part, in the
    applicator frame: along the swept centreline for the applicator_v4 shaft (landmarks.shaft_centreline, from the
    flange), else along the principal axis of its vertices (a straight tube or rod)."""
    V = np.asarray(V_app, float)
    lm = (app or {}).get("landmarks", {})
    if part == "shaft" and "shaft_centreline" in lm:
        C = np.vstack([[0.0, 0.0, 0.0], np.asarray(lm["shaft_centreline"], float)])
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
        ss = np.arange(0.0, s[-1] + 1e-9, 0.25 * STATION_MM)
        Cs = np.array([[np.interp(v, s, C[:, k]) for k in range(3)] for v in ss])
        t = ss[np.argmin(np.linalg.norm(V[:, None, :] - Cs[None, :, :], axis=2), axis=1)]
    else:
        c = V.mean(0)
        t = (V - c) @ np.linalg.svd(V - c, full_matrices=False)[2][0]
    return np.floor((t - t.min()) / STATION_MM).astype(int)


def front_cells(stations, Vw, F, x_cut):
    """The faces of a part (world vertices Vw, faces F) on a stretch the cut (x >= x_cut kept) removes all or nearly
    all of: stations keeping less than FRONT_KEEP_FRAC of their vertices; and how far in front of the plane the part
    reaches there (mm; 0 when none).  Returns (face mask, mm)."""
    x = np.asarray(Vw, float)[:, 0]
    n = np.bincount(stations)
    kept = np.bincount(stations, weights=(x >= x_cut).astype(float), minlength=len(n)) / np.maximum(n, 1)
    vf = kept[stations] < FRONT_KEEP_FRAC
    fm = vf[np.asarray(F, int)].any(1)
    return fm, (float(x_cut - x[vf].min()) if vf.any() else 0.0)


def paths():
    P = config.paths()
    hyb = P["out"] + "/hybrid"
    return dict(data=P["data"], out=P["out"], hybrid=hyb, meshes=hyb + "/meshes", applicator=hyb + "/applicator",
                runs=hyb + "/runs", figs=hyb + "/figs", logs=hyb + "/logs", anim=hyb + "/figs/anim")


P = paths()


def load_json(p):
    with open(p) as fh:
        return json.load(fh)


def run_dir(tag):
    return P["runs"] + "/" + tag


PART_FILES = {}                                     # cfg device_part_files of the run being drawn (run_extra_parts)
OVOID_BODY_ALL = ("ovoid_L", "ovoid_R", "rod_L", "rod_R", "packing")


def run_cfg(tag):
    fn = "%s/%s/cfg.json" % (P["runs"], tag)
    return load_json(fn) if os.path.exists(fn) else {}


def driven_wall_start(tag):
    """cfg wall_drive "device" (S7b): the v5 wall's collapsed START surface (meta wall.v5.start_surface_file, same vertex
    order as surface.obj), drawn as the vagina's reference wireframe: its surface.obj is the SEATED stress-free shape,
    which would read as a pre-opened vagina.  None for every other run."""
    if run_cfg(tag).get("wall_drive") != "device":
        return None
    m = load_json(P["meshes"] + "/vagina/meta.json")
    fn = ((m.get("wall") or {}).get("v5") or {}).get("start_surface_file")
    fp = "%s/vagina/%s" % (P["meshes"], fn) if fn else None
    return geom.read_obj(fp) if fp and os.path.exists(fp) else None


def app_json():
    return load_json(P["applicator"] + "/applicator.json")


def part_obj(p):
    """The OBJ a device part was loaded from in the run: cfg device_part_files overrides the file name (G32 loaded
    ovoid_L_d26.obj as ovoid_L), exactly as scene_hybrid._pobj and eval_hybrid.run_devsurf read it.  A tree whose
    applicator junction points at a variant dir that holds the override under the plain name (the README's v4out:
    applicator_v3_d26/ovoid_L.obj IS ovoid_L_d26.obj, byte-identical) falls back to <part>.obj, with a warning."""
    fn = "%s/%s.obj" % (P["applicator"], PART_FILES.get(p, p))
    if p in PART_FILES and not os.path.exists(fn) and os.path.exists("%s/%s.obj" % (P["applicator"], p)):
        print("[parts] %s: %s not in %s; using %s.obj (build the tree with render_tree.py to be sure)"
              % (p, os.path.basename(fn), P["applicator"], p))
        PART_FILES.pop(p)
        fn = "%s/%s.obj" % (P["applicator"], p)
    return fn


def ovoid_body_on(cfg, app=None):
    """Whether the run carried an ovoid body.  False for a tandem-only applicator (applicator.json params
    tandem_only, applicator_v4: tube + shaft only, no ring, rods or packing), cfg device_ovoids false, ovoid_mode
    "none", or a cfg device_parts list without ovoid parts.  ovoid_mode "off" is NOT tandem-only: the scene still
    loads and moves the ovoid body there, it only drops its contacts (scene_hybrid, dgrp = [])."""
    app = app if app is not None else app_json()
    if cfg.get("ring_phases"):                          # S7f: the two ring halves, each its own body (ring_halves)
        return True
    if bool((app.get("params", {}).get("tandem_only") or {}).get("value")):
        return False
    if cfg.get("device_ovoids") is False or cfg.get("ovoid_mode") == "none":
        return False
    if cfg.get("device_parts"):
        return any(p in OVOID_BODY_ALL for p in cfg["device_parts"])
    return True


def run_extra_parts(tag):
    """The device parts of THIS run, set in the shared TANDEM / OVOIDS lists in place (every renderer draws
    TANDEM + OVOIDS) and in PART_FILES.  Stage 3: the cap rods and the packing ride with the ovoids body only when
    the run's cfg.json asked for them (device_rods / device_packing) and the applicator dir has the files.  A
    tandem-only run (ovoid_body_on False) empties OVOIDS; a part whose OBJ is missing is dropped rather than crash
    the render.  Returns the rods / packing parts added (the pre-S6c return value)."""
    cfg = run_cfg(tag)
    PART_FILES.clear()
    PART_FILES.update(cfg.get("device_part_files") or {})
    want = (["rod_L", "rod_R"] if cfg.get("device_rods") else []) + (["packing"] if cfg.get("device_packing") else [])
    for p in ("rod_L", "rod_R", "packing"):
        if p in OVOIDS and p not in want:
            OVOIDS.remove(p)
        if p in want and p not in OVOIDS and os.path.exists(part_obj(p)):
            OVOIDS.append(p)
    for p in ("ovoid_L", "ovoid_R"):                    # restore the caps (a previous call may have removed them)
        if p not in OVOIDS:
            OVOIDS.insert(("ovoid_L", "ovoid_R").index(p), p)
    if not ovoid_body_on(cfg):
        del OVOIDS[:]
    if cfg.get("device_parts"):
        keep = set(cfg["device_parts"])
        OVOIDS[:] = [p for p in OVOIDS if p in keep]
        TANDEM[:] = [p for p in ("tube", "shaft") if p in keep]
    OVOIDS[:] = [p for p in OVOIDS if os.path.exists(part_obj(p))]
    TANDEM[:] = [p for p in TANDEM if os.path.exists(part_obj(p))]
    return [p for p in OVOIDS if p in want]


def ovoid_centres(dev):
    """The frame's cap centres as an (n, 3) array; (0, 3) for a run without caps (tandem-only: cap_centres = [])."""
    return np.asarray(dev.get("ovoid_centres_mm") or [], float).reshape(-1, 3)


def ovoid_origin(dev):
    """The ovoid body's origin of one frame; the flange when the frame has none (tandem-only)."""
    return dev["ovoid_origin_mm"] if dev.get("ovoid_origin_mm") is not None else dev["flange_mm"]


_PHASE_NAMES = None


def phase_names():
    """run_hybrid.PHASE_NAMES, read from its SOURCE (ast), not imported: run_hybrid imports the scene lazily but is a
    container module, and the tandem-first phases (V, C, L, ...) are added there by S5.  A frame written before a
    phase had a name carries phase_name == phase; the renderers then look the name up here."""
    global _PHASE_NAMES
    if _PHASE_NAMES is None:
        import ast
        _PHASE_NAMES = {}
        try:
            with open(HERE + "/run_hybrid.py") as fh:
                tree = ast.parse(fh.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PHASE_NAMES"
                                                        for t in node.targets):
                    v = node.value
                    if isinstance(v, ast.Call):                         # PHASE_NAMES = dict(B="...", ...)
                        _PHASE_NAMES.update({k.arg: ast.literal_eval(k.value) for k in v.keywords if k.arg})
                    else:                                               # PHASE_NAMES = {"B": "...", ...}
                        _PHASE_NAMES.update(ast.literal_eval(v))
        except Exception as e:                                          # never let a name lookup stop a render
            print("[phase_names] could not read run_hybrid.PHASE_NAMES: %s" % e)
    return _PHASE_NAMES


def phase_label(dev):
    """'V (vaginal travel)': the frame's own phase_name, else run_hybrid.PHASE_NAMES, else the letter alone."""
    ph = str(dev.get("phase", "?"))
    nm = dev.get("phase_name")
    if not nm or nm == ph:
        nm = phase_names().get(ph, ph)
    return ph, nm


TF_PHASES = ("V", "C", "L", "R_L", "R_R", "R_S", "K1", "K2")   # fix plan S5 / S6c: phases of the tandem-first schedules


def is_tandem_first(tag, idx=None):
    """cfg insertion_path 'tandem_first', or any exported frame in a tandem-first phase."""
    if str(run_cfg(tag).get("insertion_path", "")).startswith("tandem_first"):
        return True
    idx = idx if idx is not None else (load_json(frames_dir(tag) + "/index.json")
                                       if os.path.exists(frames_dir(tag) + "/index.json") else {"frames": []})
    return any(f.get("phase") in TF_PHASES for f in idx["frames"])


_TF = {}


def tf_metrics(tag):
    """eval/<tag>/tf_metrics.json (S0, tf_metrics.py) by step, or {} when the run has not been scored."""
    if tag not in _TF:
        fn = "%s/eval/%s/tf_metrics.json" % (P["hybrid"], tag)
        _TF[tag] = {}
        if os.path.exists(fn):
            try:
                _TF[tag] = {int(f["step"]): f for f in load_json(fn).get("frames", [])}
            except Exception as e:
                print("[tf_metrics] %s unreadable: %s" % (fn, e))
    return _TF[tag]


def frame_depth(tag, dev, fr=None):
    """Tandem depth d of one frame, mm past the external os (O_true, tandem_path.npz), with its source:
    the frame's planned depth (device json tf_d_mm / tip_s_mm / tip_s, written by the tandem-first schedule),
    else S0's d_used_mm for this step (tf_metrics.json), else MEASURED here = (tip - O_true carried) . tube axis,
    O_true carried by inverse-distance interpolation of the frame's cervix SURFACE displacement (an approximation
    of S0's harmonic carry; say so on the figure).  Returns (d, source) or (None, None)."""
    for k in ("tf_d_mm", "tip_s_mm", "tip_s"):
        if dev.get(k) is not None:
            return float(dev[k]), "planned (%s)" % k
    tm = tf_metrics(tag).get(int(dev["step"]))
    if tm and (tm.get("insertion") or {}).get("d_used_mm") is not None:
        return float(tm["insertion"]["d_used_mm"]), "S0 tf_metrics (%s)" % tm["insertion"].get("d_source", "")
    fn = P["out"] + "/inputs/tandem_path.npz"
    if fr is None or not os.path.exists(fn) or "cervix" not in fr.get("surfaces", {}):
        return None, None
    O = np.asarray(np.load(fn)["O_true"], float)
    V0, _ = geom.read_obj("%s/cervix/surface.obj" % P["meshes"])
    V1, _ = geom.read_obj("%s/%s" % (frames_dir(tag), fr["surfaces"]["cervix"]))
    d = np.linalg.norm(np.asarray(V0, float) - O, axis=1)
    nn = np.argsort(d)[:12]
    w = 1.0 / np.maximum(d[nn], 1e-6) ** 2
    Oc = O + (w[:, None] * (np.asarray(V1, float)[nn] - np.asarray(V0, float)[nn])).sum(0) / w.sum()
    return float((np.asarray(dev["tip_mm"], float) - Oc) @ geom.unit(dev["tube_axis"])), "measured (surface IDW)"


def frames_dir(tag):
    return run_dir(tag) + "/frames"


def prep_dir(tag):
    return P["anim"] + "/prep_" + tag


def load_index(tag):
    fi = frames_dir(tag) + "/index.json"
    if not os.path.exists(fi):
        raise SystemExit("no frames for run %s (%s).  Re-run it with cfg frame_every > 0." % (tag, fi))
    return load_json(fi)


def frame_R(dev, R_default, part=None):
    """The applicator frame of ONE frame: rows x_app, y_app, tube_axis from its device json (Stage 3, the device
    rotates during the insertion), else the run-constant R_rows.  For a part of the OVOIDS body (caps, rods,
    packing): the body's own frame ovoid_x_app / ovoid_y_app / ovoid_axis when the frame carries one (ovoid_mode
    "rods": the vaginal part travels separately from the tube)."""
    if part is not None and part in OVOIDS and all(k in dev for k in ("ovoid_x_app", "ovoid_y_app", "ovoid_axis")):
        return np.array([dev["ovoid_x_app"], dev["ovoid_y_app"], dev["ovoid_axis"]], float)
    if all(k in dev for k in ("x_app", "y_app", "tube_axis")):
        return np.array([dev["x_app"], dev["y_app"], dev["tube_axis"]], float)
    return np.asarray(R_default, float)


def part_pose(dev, p, R_default):
    """(origin, R_rows) of device part p in one frame: a ring half of cfg ring_phases (S7f) from the frame's
    ring_halves (each half its own pose), else the ovoid body's origin / frame for OVOIDS parts and the flange / the
    tandem frame otherwise (as every renderer placed parts before)."""
    rh = {h.get("part"): h for h in (dev.get("ring_halves") or {}).values()}
    if p in rh:
        return np.asarray(rh[p]["origin_mm"], float), np.asarray(rh[p]["R_rows"], float)
    return (ovoid_origin(dev) if p in OVOIDS else dev["flange_mm"]), frame_R(dev, R_default, p)


def device_world(V_app, origin, R_rows):
    """Device surfaces from the applicator frame into preBT world: p = origin + p_app @ R_rows (index.json)."""
    return np.asarray(origin, float) + np.asarray(V_app, float) @ np.asarray(R_rows, float)


# ================================================================================= prep: the MRI backdrop (py3.13)
def prep(tag, half_width_mm=95.0, margin_mm=45.0):
    """Sagittal plane of the preBT MRI through the mid-point of the device's FINAL intrauterine tube, cropped to the
    scene and saved as an 8-bit image + its world coordinates.  The tube leaves an exact sagittal plane by
    |a_x| * L_iu / 2 (reported), so one image column is an honest 'slice through the axis'."""
    import nibabel as nib
    from PIL import Image

    idx = load_index(tag)
    fd = frames_dir(tag)
    dev = load_json(fd + "/" + idx["frames"][-1]["device"])
    dev0 = load_json(fd + "/" + idx["frames"][0]["device"])
    od = prep_dir(tag)
    os.makedirs(od, exist_ok=True)
    Fs = np.asarray(dev["flange_mm"], float)
    a = geom.unit(dev["tube_axis"])
    L = float(dev["L_iu_mm"])
    # ---- scene bounds: rest + final organ surfaces, and the device from its start pose to its final pose
    lo, hi = [], []
    for b in idx["bodies"]:
        for p in ("%s/%s/surface.obj" % (P["meshes"], b), "%s/%s" % (fd, idx["frames"][-1]["surfaces"][b])):
            V, _ = geom.read_obj(p)
            lo.append(V.min(0))
            hi.append(V.max(0))
    for d in (dev0, dev):
        pts = np.array([d["flange_mm"], d["tip_mm"], ovoid_origin(d)], float)
        lo.append(pts.min(0) - 25.0)
        hi.append(pts.max(0) + 25.0)
    lo = np.min(lo, 0) - margin_mm
    hi = np.max(hi, 0) + margin_mm
    # ---- the slice
    im = nib.load(P["data"] + "/preBT_MRI.nii")
    A = im.affine
    if not np.allclose(A[:3, :3], np.diag(np.diag(A[:3, :3]))):
        raise RuntimeError("preBT affine is not axis-aligned; this slice extraction assumes diagonal voxels")
    vol = np.asarray(im.dataobj).astype(np.float32)
    x0 = float(Fs[0] + 0.5 * L * a[0])
    inv = np.linalg.inv(A)
    i0 = int(min(max(np.rint(inv[0, 0] * x0 + inv[0, 3]), 0), vol.shape[0] - 1))
    x_plane = float(A[0, 0] * i0 + A[0, 3])
    sl = vol[i0]                                                   # (ny, nz), indexed [j, k]
    y = A[1, 1] * np.arange(sl.shape[0]) + A[1, 3]
    z = A[2, 2] * np.arange(sl.shape[1]) + A[2, 3]
    if y[1] < y[0]:
        sl, y = sl[::-1], y[::-1]
    if z[1] < z[0]:
        sl, z = sl[:, ::-1], z[::-1]
    dy, dz = float(abs(A[1, 1])), float(abs(A[2, 2]))
    win = float(np.percentile(sl[sl > 0], 99.5)) if (sl > 0).any() else 1.0
    yc = 0.5 * (lo[1] + hi[1])
    yy = np.arange(yc - half_width_mm, yc + half_width_mm + dy, dy)
    z_top = hi[2] + 0.25 * (hi[2] - lo[2])      # cover the overlay headroom the renderer reserves (Scene._headroom),
    zz = np.arange(lo[2], z_top + dz, dz)       # otherwise the backdrop ends in a black band inside the viewport
    img = np.zeros((len(yy), len(zz)), np.float32)                 # zero (black) outside the acquired slice
    jj = np.rint((yy - y[0]) / dy).astype(int)
    kk = np.rint((zz - z[0]) / dz).astype(int)
    okj = (jj >= 0) & (jj < len(y))
    okk = (kk >= 0) & (kk < len(z))
    img[np.ix_(okj, okk)] = sl[np.ix_(jj[okj], kk[okk])]
    img8 = (255 * np.clip(img / win, 0, 1) ** 0.9).astype(np.uint8)
    np.savez(od + "/backdrop.npz", img=img8, y=yy.astype(np.float32), z=zz.astype(np.float32), x=np.float32(x_plane),
             dy=np.float32(dy), dz=np.float32(dz))
    Image.fromarray(img8.T[::-1, ::-1]).save(od + "/backdrop_preview.png")     # rows S up, columns A left
    rec = dict(tag=tag, units="mm; preBT world RAS", plane="sagittal x = %.3f (voxel column %d)" % (x_plane, i0),
               x_plane=x_plane, x_axis_mid=round(x0, 3), i0=i0, window_hi=round(win, 2),
               axis_out_of_plane_mm=round(float(abs(a[0]) * L / 2.0), 3),
               bounds=dict(lo=lo.round(2).tolist(), hi=hi.round(2).tolist()),
               source="%s/preBT_MRI.nii (READ-ONLY)" % P["data"])
    with open(od + "/prep.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    print("prep %s: sagittal plane x = %.2f (tube mid x = %.2f, i %d), tube leaves the plane by %.2f mm, window %.0f"
          % (tag, x_plane, x0, i0, rec["axis_out_of_plane_mm"], win))
    print("        backdrop %d x %d px (%.0f x %.0f mm), %s" % (img8.shape[0], img8.shape[1],
                                                                yy[-1] - yy[0], zz[-1] - zz[0], od + "/backdrop.npz"))
    return rec


# ================================================================================= render (py -3.11, pyvista)
def _pv():
    import pyvista as pv
    pv.OFF_SCREEN = True
    return pv


def poly(pv, V, F):
    F = np.asarray(F, int)
    return pv.PolyData(np.asarray(V, float), np.c_[np.full(len(F), 3), F].ravel())


class Scene:
    """The two fixed viewports and every static actor (rest wireframes, backdrop, legend, axes)."""

    def __init__(self, tag, bodies, size=(1600, 800), backdrop=False, zoom=False, clip=True):
        pv = self.pv = _pv()
        self.tag, self.bodies, self.size, self.clip = tag, bodies, size, clip
        self.idx = load_index(tag)
        self.fd = frames_dir(tag)
        self.R_rows = np.asarray(self.idx["R_rows"], float)
        self.col = {b: tuple(load_json(P["meshes"] + "/bodies.json")["bodies"][b]["color"]) for b in BODIES}
        self.frames = self.idx["frames"]
        self.dev_first = load_json(self.fd + "/" + self.frames[0]["device"])
        self.dev_last = load_json(self.fd + "/" + self.frames[-1]["device"])
        run_extra_parts(tag)                    # the run's own parts: rods / packing, or no ovoid body at all
        self.dev_app = {p: geom.read_obj(part_obj(p)) for p in TANDEM + OVOIDS}
        self.rest = {b: geom.read_obj("%s/%s/surface.obj" % (P["meshes"], b)) for b in bodies}
        ws = driven_wall_start(tag) if "vagina" in bodies else None
        if ws is not None:                              # S7b: the driven v5 wall's reference = its collapsed preBT
            self.rest["vagina"] = ws                    # START, not the seated stress-free shape
        self.last = {b: geom.read_obj("%s/%s" % (self.fd, self.frames[-1]["surfaces"][b])) for b in bodies}
        # ---- the cut plane: the device's own sagittal plane (mid-point of the final tube)
        Fs = np.asarray(self.dev_last["flange_mm"], float)
        a = geom.unit(self.dev_last["tube_axis"])
        self.x_cut = float(Fs[0] + 0.5 * float(self.dev_last["L_iu_mm"]) * a[0])
        lo, hi = self._box(zoom)
        self.lo, self.hi = lo, hi
        self.c = 0.5 * (lo + hi)
        # ---- plotter
        self.pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=list(size), col_weights=[0.44, 0.56],
                             border=True)
        self.bd = None
        if backdrop:
            pd = prep_dir(tag) + "/backdrop.npz"
            if os.path.exists(pd):
                self.bd = np.load(pd)
            else:
                print("[render] --backdrop: %s missing; run `python animate_hybrid.py prep --tag %s` first "
                      "(continuing without it)" % (pd, tag))
        for k in (0, 1):
            self.pl.subplot(0, k)
            self.pl.set_background("white")
            try:
                self.pl.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)
            except Exception:
                pass
            if self.bd is not None:
                x_bd = float(hi[0] + 40.0) if k == 0 else float(self.bd["x"])
                g = pv.ImageData(dimensions=(1, len(self.bd["y"]), len(self.bd["z"])),
                                 spacing=(1.0, float(self.bd["dy"]), float(self.bd["dz"])),
                                 origin=(x_bd, float(self.bd["y"][0]), float(self.bd["z"][0])))
                g.point_data["mri"] = self.bd["img"].ravel(order="F")
                self.pl.add_mesh(g, scalars="mri", cmap="gray", clim=(0, 255), show_scalar_bar=False, lighting=False,
                                 opacity=1.0 if k == 0 else 0.55, name="backdrop", interpolate_before_map=True)
            for b in bodies:                                  # the REST state, faint wireframe, in the body's colour
                m = poly(pv, *self.rest[b])
                if k == 0 and clip:
                    m = m.clip(normal="x", origin=(self.x_cut, 0, 0), invert=False)
                self.pl.add_mesh(m, style="wireframe", color=self.col[b], opacity=0.30, line_width=1,
                                 name="rest_" + b)
        self.pl.subplot(0, 1)
        self.pl.add_axes(xlabel="R", ylabel="A", zlabel="S", line_width=3)
        dev_keys = [("tube / shaft", COL_TANDEM)] + ([("ovoids", COL_OVOID)] if OVOIDS else []) \
            + [("rest state", (0.6, 0.6, 0.6))]          # no ovoid entry for a tandem-only run
        self.pl.add_legend(labels=[(b, self.col[b]) for b in bodies] + dev_keys,
                           bcolor="white", face="rectangle", size=(0.17, 0.04 + 0.035 * (len(bodies) + len(dev_keys))),
                           loc="lower right")            # upper right collides with the per-frame info text
        # ---- cameras, fixed for every frame
        w0 = 0.44 * size[0]
        self.pscale = max(0.5 * (hi[2] - lo[2]), 0.5 * (hi[1] - lo[1]) / (w0 / size[1]))
        self.cam0 = [(self.c[0] - 600.0, self.c[1], self.c[2]), tuple(self.c), (0.0, 0.0, 1.0)]
        d = geom.unit([-0.62, 0.62, 0.48])
        self.cam1 = [tuple(self.c + d * (0.52 * max(hi - lo) / np.tan(np.radians(15.0)))), tuple(self.c),
                     (0.0, 0.0, 1.0)]
        self.apply_cameras()
        self._shown = False
        self._sil = {}                          # (panel, part) -> silhouette of a stretch drawn in front of the cut
        self.front = {}                         # this frame's parts with a stretch in front of the cut: part -> mm
        app = app_json()
        self.stations = {p: part_stations(p, self.dev_app[p][0], app) for p in TANDEM if p in FRONT_PARTS}

    def _box(self, zoom):
        """The camera box.  Default: everything the run touches.  --zoom: the vagina and the seated ovoids."""
        pts = []
        if zoom:
            for b in ("vagina", "cervix"):
                if b in self.bodies:
                    pts += [self.rest[b][0], self.last[b][0]]
            oc = ovoid_centres(self.dev_last)               # (0, 3) for a tandem-only run: drops out of the box
            pts += [oc - 24.0, oc + 24.0, np.atleast_2d(self.dev_last["flange_mm"])]
            if pts:
                V = np.vstack(pts)
                return self._headroom(V.min(0) - 8.0, V.max(0) + 8.0)
        for b in self.bodies:
            pts += [self.rest[b][0], self.last[b][0]]
        for d in (self.dev_first, self.dev_last):
            oc = ovoid_centres(d)
            pts += [np.atleast_2d(d["flange_mm"]), np.atleast_2d(d["tip_mm"]), oc - 22.0, oc + 22.0]
        V = np.vstack(pts)
        return self._headroom(V.min(0) - 4.0, V.max(0) + 4.0)

    @staticmethod
    def _headroom(lo, hi, frac=0.17):
        """Reserve empty space at the TOP of both viewports for the three overlay lines: without it the third line
        ('max |u| ...') is drawn over the corpus and is unreadable in the cut view."""
        hi = hi.copy()
        hi[2] += frac * (hi[2] - lo[2])
        return lo, hi

    def apply_cameras(self):
        pl = self.pl
        pl.subplot(0, 0)
        pl.camera_position = self.cam0
        pl.camera.parallel_projection = True
        pl.camera.parallel_scale = self.pscale
        pl.renderer.reset_camera_clipping_range()
        pl.subplot(0, 1)
        pl.camera_position = self.cam1
        pl.camera.parallel_projection = False
        pl.camera.view_angle = 30.0
        pl.renderer.reset_camera_clipping_range()

    def text(self, k, txt, name, position="upper_left", font_size=10):
        over_mri = (k == 0 and self.bd is not None)
        self.pl.subplot(0, k)
        self.pl.add_text(txt, position=position, font_size=font_size, name=name,
                         color="white" if over_mri else "black", shadow=over_mri)

    def draw_frame(self, fr):
        """Replace the moving actors with this frame's surfaces and device."""
        pv = self.pv
        dev = load_json(self.fd + "/" + fr["device"])
        self.front = {}
        meshes = []
        for b in self.bodies:
            V, F = geom.read_obj("%s/%s" % (self.fd, fr["surfaces"][b]))
            meshes.append((b, poly(pv, V, F)))
        parts = []
        for p in TANDEM + OVOIDS:
            V, F = self.dev_app[p]
            o, Rp = part_pose(dev, p, self.R_rows)
            parts.append((p, poly(pv, device_world(V, o, Rp), F)))
        for k in (0, 1):
            self.pl.subplot(0, k)
            op = OPACITY_CUT if k == 0 else OPACITY_3D
            for b, m in meshes:
                mm = m.clip(normal="x", origin=(self.x_cut, 0, 0), invert=False) if (k == 0 and self.clip) else m
                self.pl.add_mesh(mm, color=self.col[b], opacity=op[b], smooth_shading=True, specular=0.15,
                                 name="body_" + b)
            for p, m in parts:
                cut = k == 0 and self.clip
                mm = m.clip(normal="x", origin=(self.x_cut, 0, 0), invert=False) if cut else m
                if mm.n_points == 0:
                    self.pl.remove_actor("dev_" + p)
                else:
                    self.pl.add_mesh(mm, color=COL_OVOID if p in OVOIDS else COL_TANDEM,
                                     opacity=(0.12 if p == "packing" else OVOID_OPACITY[k]) if p in OVOIDS else 1.0,
                                     smooth_shading=True, name="dev_" + p)
                if p not in self.stations:
                    continue
                old = self._sil.pop((k, p), None)             # last frame's silhouette of this part, if any
                if old is not None:
                    self.pl.renderer.remove_actor(old, render=False)
                fm, fmm = front_cells(self.stations[p], m.points, self.dev_app[p][1], self.x_cut) if cut \
                    else (None, 0.0)
                if fm is None or not fm.any():                # THIS panel's renderer only (Plotter.remove_actor by
                    self.pl.renderer.remove_actor("front_" + p, render=False)   # name clears it from both panels)
                    continue
                F = np.asarray(self.dev_app[p][1], int)[fm]     # the stretch in front of the section: whole,
                sub = pv.PolyData(np.asarray(m.points), np.c_[np.full(len(F), 3), F].ravel())   # see-through, outlined
                self.pl.add_mesh(sub, color=COL_TANDEM, opacity=FRONT_OPACITY, smooth_shading=True, name="front_" + p)
                self._sil[(k, p)] = self.pl.add_silhouette(sub, color=COL_FRONT_EDGE, line_width=1.6)
                self.front[p] = fmm
        return dev

    def shot(self, path):
        self.apply_cameras()
        if not self._shown:
            self.pl.show(auto_close=False)
            self._shown = True
        self.pl.render()
        self.pl.screenshot(path)

    def close(self):
        self.pl.close()


_TF_RUN = {}


def depth_line(tag, dev, fr=None):
    """Second overlay line.  Stage 1-3 runs (G32): the schedule's travel and the tip past O_pre, as always.  A
    tandem-first run: the tandem depth d past the physician's external os (O_true), which is what its phases are
    defined by (fix plan S5), with the source of the number (frame_depth)."""
    if tag not in _TF_RUN:
        _TF_RUN[tag] = is_tandem_first(tag)
    if not _TF_RUN[tag]:
        return ("inserted depth %.1f / %.1f mm along the path (u = %.2f)   tip %+.1f mm past the preBT external os"
                % (dev["advance_mm"], dev["advance_mm"] + dev["remaining_mm"], dev["u"], dev["tip_beyond_O_pre_mm"]))
    d, src = frame_depth(tag, dev, fr)
    s = ("tandem depth d %.1f mm past the external os (O_true; %s)" % (d, src)) if d is not None else \
        "tandem depth: not available"
    return s + "   u = %.2f" % float(dev.get("u", float("nan")))


def overlay(tag, dev, bodies, n_last, clip, zoom, fr=None):
    """step / phase / inserted depth / max displacement -- the four numbers the animation must state."""
    d = dev["disp"]
    order = [b for b in BODIES if b in bodies]
    umax = max(d[b]["umax_mm"] for b in order) if order else float("nan")
    per = "  ".join("%s %.1f" % (b[:3], d[b]["umax_mm"]) for b in order)
    ph, ph_name = phase_label(dev)
    txt = ("run %s   step %d/%d   phase %s (%s)%s\n" % (tag, dev["step"], n_last, ph, ph_name,
                                                        ("  settle %d" % dev["settle_step"])
                                                        if dev.get("settle_step") else "")
           + depth_line(tag, dev, fr) + "\n"
           + "max |u| %.1f mm  (%s)   contacts %d   dx %.3f mm/step"
           % (umax, per, dev["n_contacts"] or 0, dev["dx_max_mm"] or 0.0))
    cap = ("sagittal-like view: camera on the patient's left, ANTERIOR left, SUPERIOR up"
           + ("; anatomy cut at the device's sagittal plane, near half removed" if clip else "; nothing cut")
           + ("   [ZOOM: vagina / ovoid region]" if zoom else ""))
    return txt, cap


def save_gif(pngs, out, fps, hold_last_s, max_mb, width0=1000):
    """One shared palette (no flicker), the last frame held.  Shrinks / drops frames until it is under max_mb."""
    from PIL import Image
    attempts = [(width0, 1), (int(0.8 * width0), 1), (int(0.8 * width0), 2), (int(0.62 * width0), 2),
                (int(0.62 * width0), 3)]
    for width, step in attempts:
        sel = pngs[::step]
        if sel[-1] != pngs[-1]:
            sel.append(pngs[-1])
        w0, h0 = Image.open(pngs[0]).size
        size = (width, int(round(width * h0 / w0)))
        ims = [Image.open(p).convert("RGB").resize(size, Image.LANCZOS) for p in sel]
        pal = ims[-1].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        q = [im.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG) for im in ims]
        dur = [int(round(1000.0 / (fps / step)))] * len(q)
        dur[-1] = int(hold_last_s * 1000)
        q[0].save(out, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=True, disposal=1)
        mb = os.path.getsize(out) / 1e6
        print("[gif] %s  %d frames  %dx%d  %.1f MB%s" % (os.path.basename(out), len(q), size[0], size[1], mb,
                                                         "" if mb <= max_mb else "  > cap %.0f MB, retrying" % max_mb))
        if mb <= max_mb:
            return dict(path=out, n=len(q), size=list(size), mb=round(mb, 2), every=step, fps=round(fps / step, 2))
    return dict(path=out, n=len(q), size=list(size), mb=round(mb, 2), every=step, fps=round(fps / step, 2),
                over_cap=True)


def render(tag, name=None, bodies=None, every=1, limit=None, fps=9.0, hold_last_s=1.5, size=(1600, 800),
           backdrop=False, zoom=False, clip=True, gif_max_mb=12.0):
    idx = load_index(tag)
    have = list(idx["bodies"])
    bodies = [b for b in (bodies or BODIES) if b in have] or have
    missing = [b for b in (bodies or []) if b not in have]
    if idx.get("skipped_bodies"):
        print("[render] the run did not export: %s" % idx["skipped_bodies"])
    if missing:
        print("[render] not exported by this run, dropped: %s" % missing)
    name = name or (tag + ("_zoom" if zoom else "") + ("" if len(bodies) == len(have) else "_" + "-".join(bodies)))
    fr = idx["frames"]
    if every > 1:
        fr = fr[::every] + ([fr[-1]] if (len(fr) - 1) % every else [])
    if limit:
        fr = fr[:limit]
    od = P["anim"] + "/frames_" + name
    os.makedirs(od, exist_ok=True)
    sc = Scene(tag, bodies, size=size, backdrop=backdrop, zoom=zoom, clip=clip)
    n_last = idx["frames"][-1]["step"]
    sc.text(1, "oblique 3-D view from the patient's left-anterior-superior (nothing cut; bladder and sigmoid "
               "translucent)\nrest state = faint wireframe%s   device = grey   Delta = %g mm%s"
            % (" (vagina: its collapsed preBT start)" if driven_wall_start(tag) is not None else "",
               idx["flange_shift_mm"], "" if OVOIDS else "   tandem only (no ring, rods or packing)"),
            "cap1", position="lower_left", font_size=8)
    pngs, fronts = [], {}
    for i, f in enumerate(fr):
        dev = sc.draw_frame(f)
        if sc.front:
            fronts[str(dev["step"])] = {p: round(v, 2) for p, v in sc.front.items()}
        txt, cap = overlay(tag, dev, bodies, n_last, clip, zoom, fr=f)
        if sc.front:                                  # tandem stretches the cut would remove (front_cells)
            cap += ("\nsee-through, outlined: the %s where %s IN FRONT of the cut plane (up to %.0f mm)"
                    % (" and ".join(dict(tube="tube", shaft="vaginal shaft")[p] for p in sc.front),
                       "it lies" if len(sc.front) == 1 else "they lie", max(sc.front.values())))
        sc.text(0, txt, "info0", font_size=10)
        sc.text(0, cap, "cap0", position="lower_left", font_size=8)
        sc.text(1, txt, "info1", font_size=10)
        p = od + "/step_%04d.png" % dev["step"]
        sc.shot(p)
        pngs.append(p)
        print("frame %d/%d  step %d %s u=%.2f" % (i + 1, len(fr), dev["step"], dev["phase"], dev["u"]), flush=True)
    sc.close()
    gif = save_gif(pngs, P["anim"] + "/insertion_%s.gif" % name, fps, hold_last_s, gif_max_mb)
    meta = dict(tag=tag, name=name, bodies=bodies, zoom=bool(zoom), clip=bool(clip), backdrop=bool(sc.bd is not None),
                pngs=[os.path.basename(p) for p in pngs], fps=fps, hold_last_s=hold_last_s, size=list(size),
                png_size_mb=round(sum(os.path.getsize(p) for p in pngs) / 1e6, 1), gif=gif,
                camera=dict(lo=sc.lo.round(2).tolist(), hi=sc.hi.round(2).tolist(), x_cut=round(sc.x_cut, 3)),
                device_parts=dict(tandem=list(TANDEM), ovoid_body=list(OVOIDS),
                                  files={p: os.path.basename(part_obj(p)) for p in TANDEM + OVOIDS},
                                  applicator=P["applicator"]))
    if fronts:                                          # steps with a tube / shaft stretch drawn in front of the cut
        meta["drawn_in_front_of_cut_mm"] = fronts
    with open(od + "/frames.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print("[render] %d PNG in %s (%.1f MB), gif %s" % (len(pngs), od, meta["png_size_mb"], gif["path"]))
    return meta


# ================================================================================= mp4 (py3.13 + imageio-ffmpeg)
def mp4(tag, name=None, fps=9.0, hold_last_s=1.5, pk=None):
    pk = pk or os.environ.get("APPSIM_PK")
    if pk and os.path.isdir(pk):
        sys.path.insert(0, pk)
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401
    except Exception as e:
        print("mp4 skipped: %s\n  pip install --target <scratch>/pk imageio-ffmpeg, then --pk <scratch>/pk" % e)
        return None
    name = name or tag
    fd = P["anim"] + "/frames_" + name
    meta = load_json(fd + "/frames.json")
    out = P["anim"] + "/insertion_%s.mp4" % name
    w = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=16)
    n_hold = max(1, int(round(hold_last_s * fps)))
    for i, p in enumerate(meta["pngs"]):
        im = imageio.imread(fd + "/" + p)[:, :, :3]
        for _ in range(n_hold if i == len(meta["pngs"]) - 1 else 1):
            w.append_data(im)
    w.close()
    n = len(meta["pngs"]) + n_hold - 1
    print("wrote %s (%.1f MB, %d frames at %g fps = %.1f s, last frame held %.1f s)"
          % (out, os.path.getsize(out) / 1e6, n, fps, n / fps, hold_last_s))
    meta["mp4"] = dict(path=out, mb=round(os.path.getsize(out) / 1e6, 2), fps=fps, n_frames=n,
                       duration_s=round(n / fps, 2), hold_last_s=hold_last_s)
    with open(fd + "/frames.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["prep", "render", "mp4"])
    ap.add_argument("--tag", required=True, help="run tag: hybrid/runs/<tag>/frames must exist")
    ap.add_argument("--name", default=None, help="output basename (default: <tag>[_zoom][_<bodies>])")
    ap.add_argument("--bodies", default=None, help="comma-separated subset, e.g. vagina,cervix (default: all)")
    ap.add_argument("--zoom", action="store_true", help="frame the vagina / ovoid region instead of the whole pelvis")
    ap.add_argument("--backdrop", action="store_true", help="draw the preBT MRI sagittal plane (needs `prep` first)")
    ap.add_argument("--no-clip", dest="clip", action="store_false",
                    help="do not cut the anatomy at the device's sagittal plane in the left panel")
    ap.add_argument("--every", type=int, default=1, help="render every N-th exported frame")
    ap.add_argument("--limit", type=int, default=None, help="render only the first N frames (tests)")
    ap.add_argument("--fps", type=float, default=9.0)
    ap.add_argument("--hold", type=float, default=1.5, help="seconds the final frame is held")
    ap.add_argument("--size", default="1600x800", help="PNG size, WxH (multiples of 16 keep libx264 happy)")
    ap.add_argument("--gif-max-mb", type=float, default=12.0)
    ap.add_argument("--pk", default=None, help="pip --target dir holding imageio-ffmpeg (mp4 mode)")
    a = ap.parse_args()
    os.makedirs(P["anim"], exist_ok=True)
    bodies = [b.strip() for b in a.bodies.split(",")] if a.bodies else None
    if a.mode == "prep":
        prep(a.tag)
    elif a.mode == "render":
        w, h = (int(v) for v in a.size.lower().split("x"))
        render(a.tag, name=a.name, bodies=bodies, every=a.every, limit=a.limit, fps=a.fps, hold_last_s=a.hold,
               size=(w, h), backdrop=a.backdrop, zoom=a.zoom, clip=a.clip, gif_max_mb=a.gif_max_mb)
    else:
        name = a.name or (a.tag + ("_zoom" if a.zoom else "") + ("_" + "-".join(bodies) if bodies else ""))
        mp4(a.tag, name=name, fps=a.fps, hold_last_s=a.hold, pk=a.pk)


if __name__ == "__main__":
    main()
