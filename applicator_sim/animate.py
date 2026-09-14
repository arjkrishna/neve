"""HOST renderer of an insertion run written with cfg frame_every > 0 (run_insertion.write_frame).  Units mm; preBT
world RAS (x = R, y = A, z = S).  Nothing here touches the simulation; every output goes to $APPSIM_OUT/figs/anim/.

    python animate.py prep --tag ANIM_F1a       # base python (nibabel, scipy, skimage): the backdrop (preBT MRI sagittal
                                                #   slice through the final tandem axis), the real BT uterus / HR-CTV /
                                                #   applicator surfaces mapped into the sim frame by E_app (evaluate.py's
                                                #   device-onto-device Kabsch)      -> figs/anim/prep_<tag>/
    py -3.11 animate.py render --tag ANIM_F1a   # pyvista OFF-SCREEN: one 1600x800 PNG per frame (sagittal-like view +
                                                #   oblique 3-D view)               -> figs/anim/frames_<tag>/*.png
                                                #   and insertion_<tag>.gif (PIL, ~9 fps, last frame held 1.5 s)
    py -3.11 animate.py compare --tag ANIM_F1a  # final simulated uterus + HR-CTV (solid) vs the real BT ones (wireframe)
                                                #   in the same frame, both devices  -> figs/anim/final_vs_BT_<tag>.png
    python animate.py mp4 --tag ANIM_F1a        # base python + imageio (+ imageio-ffmpeg from $APPSIM_PK, optional)
                                                #                                    -> figs/anim/insertion_<tag>.mp4
The two interpreters carry different packages (base: nibabel/scipy/skimage/imageio, 3.11: pyvista), hence the modes.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import geom  # noqa: E402

P = config.paths()
ANIM = P["figs"] + "/anim"

# colours (named, so the legend text matches)
COL = dict(uterus="burlywood", hrctv="crimson", rest="dimgray", tandem="gold", shaft="dimgray", flange="silver",
           ovoid="mediumorchid", canal="royalblue", bt_uterus="dodgerblue", bt_hrctv="darkorange", bt_tandem="limegreen",
           bt_app="limegreen")
SHAFT_BELOW_FLANGE_MM = 26.0        # MEASURED straight shaft below the flange (config shaft_len_mm; drawn only)


def run_dir(tag):
    return P["runs"] + "/" + tag


def prep_dir(tag):
    return ANIM + "/prep_" + tag


def load_json(p):
    with open(p) as fh:
        return json.load(fh)


def obj_bounds(*paths):
    lo, hi = [], []
    for p in paths:
        V, _ = geom.read_obj(p); lo.append(V.min(0)); hi.append(V.max(0))
    return np.min(lo, 0), np.max(hi, 0)


# ======================================================================================== prep (base python)
def prep(tag):
    import nibabel as nib
    from scipy import ndimage as ndi
    from skimage import measure
    from evaluate import app_landmarks                       # the very landmarks evaluate.py aligns the devices with

    rd = run_dir(tag); od = prep_dir(tag); os.makedirs(od, exist_ok=True)
    z = np.load(rd + "/final.npz"); cfg = load_json(rd + "/cfg.json")
    Fs, a_s, xs, L = np.asarray(z["F"], float), geom.unit(z["a"]), np.asarray(z["x"], float), float(z["L_iu"])
    inp = P["inputs"]; po = load_json(inp + "/poses.json"); app = load_json(inp + "/applicator.json")
    a0 = geom.unit(po["a0"]); O = np.asarray(po["O_pre"], float)
    # ---- scene bounds: rest + final organ surfaces and the device from its approach start to its final pose
    lo, hi = obj_bounds(inp + "/pre_uterus.obj", inp + "/pre_hrctv.obj", rd + "/surf_uterus.obj", rd + "/surf_hrctv.obj")
    F_A = O - (float(cfg["phaseA_start_mm"]) + L) * a0                    # flange at the first approach step
    dev = np.array([F_A - 28.0 * a0, F_A + L * a0, Fs - 28.0 * a_s, Fs + L * a_s])   # 28 mm: shaft + ovoid pack below F
    olo, ohi = lo.copy(), hi.copy()                                      # organ (rest + final) bounds
    lo = np.minimum(lo, dev.min(0) - np.array([22.0, 22.0, 0.0])); hi = np.maximum(hi, dev.max(0) + np.array([22.0, 22.0, 0.0]))
    bounds = dict(lo=(lo - 5).round(2).tolist(), hi=(hi + 5).round(2).tolist())      # everything (backdrop crop)
    # camera framing: the organs, the final device and 40 mm below the rest organ bottom (the ovoid pack is in view for
    # the second half of the insertion; during the approach the device enters from below the frame)
    flo = np.array([min(olo[0], Fs[0]) - 10.0, olo[1] - 10.0, olo[2] - 40.0])
    fhi = np.array([max(ohi[0], Fs[0]) + 10.0, ohi[1] + 10.0, max(ohi[2], (Fs + L * a_s)[2]) + 6.0])
    frame = dict(lo=flo.round(2).tolist(), hi=fhi.round(2).tolist())
    # ---- backdrop: preBT MRI sagittal plane through the final tandem axis (its mid point; the axis leaves an exact
    # sagittal plane by |a_x| L / 2 < 1 mm over the rod, so one image column is an honest "slice through the axis")
    im = nib.load(P["data"] + "/preBT_MRI.nii"); A = im.affine
    if not np.allclose(A[:3, :3], np.diag(np.diag(A[:3, :3]))):
        raise RuntimeError("preBT affine is not diagonal; the slice extraction assumes axis-aligned voxels")
    vol = np.asarray(im.dataobj).astype(np.float32)
    x0 = float(Fs[0] + 0.5 * L * a_s[0]); inv = np.linalg.inv(A)
    i0 = int(np.rint(inv[0, 0] * x0 + inv[0, 3])); i0 = min(max(i0, 0), vol.shape[0] - 1)
    x_plane = float(A[0, 0] * i0 + A[0, 3])
    sl = vol[i0]                                                             # (ny, nz) indexed [j, k]
    y = A[1, 1] * np.arange(sl.shape[0]) + A[1, 3]; zc = A[2, 2] * np.arange(sl.shape[1]) + A[2, 3]
    if y[1] < y[0]:
        sl, y = sl[::-1], y[::-1]
    if zc[1] < zc[0]:
        sl, zc = sl[:, ::-1], zc[::-1]
    dy, dz = float(abs(A[1, 1])), float(abs(A[2, 2]))
    win_hi = float(np.percentile(sl[sl > 0], 99.5)) if (sl > 0).any() else 1.0
    # crop to the view (the sagittal viewport shows ~150 mm width around the scene centre) and pad with black outside
    yc = 0.5 * (lo[1] + hi[1]); y_lo, y_hi = yc - 85.0, yc + 85.0
    z_lo, z_hi = float(bounds["lo"][2]) - 5.0, float(bounds["hi"][2]) + 5.0
    yy = np.arange(y_lo, y_hi + dy, dy); zz = np.arange(z_lo, z_hi + dz, dz)
    img = np.zeros((len(yy), len(zz)), np.float32)
    jj = np.rint((yy - y[0]) / dy).astype(int); kk = np.rint((zz - zc[0]) / dz).astype(int)
    okj = (jj >= 0) & (jj < len(y)); okk = (kk >= 0) & (kk < len(zc))
    img[np.ix_(okj, okk)] = sl[np.ix_(jj[okj], kk[okk])]
    img8 = np.clip(img / win_hi, 0, 1); img8 = (255 * img8 ** 0.9).astype(np.uint8)
    np.savez(od + "/backdrop.npz", img=img8, y=yy.astype(np.float32), z=zz.astype(np.float32), x=x_plane, x_axis_mid=x0,
             dy=dy, dz=dz, window=np.array([0.0, win_hi]))
    from PIL import Image
    Image.fromarray(img8.T[::-1, ::-1]).save(od + "/backdrop_preview.png")     # rows: S at top; cols: A at left
    # ---- E_app: rigid map sim (preBT world) -> BT world putting the simulated device onto the real one (evaluate.py)
    Rs = geom.pose_frame(dict(F=Fs, a=a_s, x=xs))
    Fb = np.asarray(app["origin_BT_world"], float); Rr = np.asarray(app["R_rows_BT_world"], float); sph = app["spheres"]
    Ps = app_landmarks(Fs, Rs, sph); Pb = app_landmarks(Fb, Rr, sph)
    Re, te = geom.kabsch(Ps, Pb)                                             # x_BT = Re y + te
    resid = float(np.linalg.norm(Ps @ Re.T + te - Pb, axis=1).max())
    to_pre = lambda X: (np.atleast_2d(np.asarray(X, float)) - te) @ Re      # y = Re^T (x - te)   (BT -> preBT)  # noqa
    pa = load_json(inp + "/poses_align.json").get("PB")
    chk = None
    if pa is not None:                                                       # the PB pose was generated from the BONE frame
        Rb, tb = np.asarray(pa["R_BT_to_pre"], float), np.asarray(pa["t_BT_to_pre"], float)
        chk = dict(rot_diff_deg=round(geom.rot_angle_deg(Re.T @ Rb.T), 4),
                   flange_diff_mm=round(float(np.linalg.norm(to_pre(Fb)[0] - (Rb @ Fb + tb))), 4))
    # ---- real BT anatomy and device, mapped into the sim frame
    surf = {}
    for lab, nm in (("uterus", "bt_uterus"), ("HR-CTV", "bt_hrctv"), ("applicator", "bt_applicator")):
        li = nib.load("%s/BT_MRI_label_%s.nii" % (P["data"], lab)); Al = li.affine
        m = (np.asarray(li.dataobj) > 0).astype(np.float32)
        sm = ndi.gaussian_filter(m, sigma=(0.8, 0.8, 0.6))                  # sub-voxel smoothing of the binary label
        V, Fc, _, _ = measure.marching_cubes(sm, level=0.5, spacing=(1.0, 1.0, 1.0))
        Vw = V @ Al[:3, :3].T + Al[:3, 3]                                    # index -> BT world mm
        Vp = to_pre(Vw)
        geom.write_obj(od + "/%s.obj" % nm, Vp, Fc, header="BT %s label (marching cubes), mapped into preBT world by "
                                                           "E_app^-1 for run %s (local only)" % (lab, tag))
        surf[nm] = dict(n_verts=int(len(Vp)), n_faces=int(len(Fc)), vox=int(m.sum()),
                        mesh_cc=round(abs(geom.mesh_volume(Vp, Fc)) / 1000.0, 2))
    Fb_pre = to_pre(Fb)[0]; ab_pre = geom.unit(Re.T @ Rr[2]); tip_pre = Fb_pre + L * ab_pre
    sph_pre = to_pre(Fb + np.asarray([s["center_app_mm"] for s in sph], float) @ Rr)
    comp = dict(tag=tag, units="mm, deg; preBT world RAS", E_app=dict(R=Re.tolist(), t=te.tolist(), landmark_resid_max_mm=resid,
                convention="x_BT = R y_pre + t; BT -> preBT by y = R^T (x - t)"), E_app_vs_BONE_frame=chk,
                sim_device=dict(flange=Fs.round(4).tolist(), axis=a_s.round(6).tolist(), tip=(Fs + L * a_s).round(4).tolist(), L_iu_mm=L,
                                r_tandem_mm=float(z["r_rod"]), sphere_centers=np.asarray(z["sphere_centers"]).round(4).tolist(),
                                sphere_r=np.asarray(z["sphere_r"]).round(4).tolist()),
                bt_device_in_pre=dict(flange=Fb_pre.round(4).tolist(), axis=ab_pre.round(6).tolist(), tip=tip_pre.round(4).tolist(),
                                      flange_err_mm=round(float(np.linalg.norm(Fb_pre - Fs)), 4), axis_err_deg=round(geom.angle_deg(ab_pre, a_s), 4),
                                      sphere_centers=sph_pre.round(4).tolist(), sphere_r=[s["r_mm"] for s in sph],
                                      note="MEASURED BT pack; the run's spheres carry sph_r_scale / sph_dz_mm"),
                bt_surfaces=surf, bounds=bounds, frame=frame,
                backdrop=dict(x_plane=x_plane, x_axis_mid=x0, i0=i0, window_hi=win_hi))
    with open(od + "/prep.json", "w") as fh:
        json.dump(comp, fh, indent=1)
    print("prep %s: backdrop plane x=%.2f (axis mid x=%.2f), E_app resid %.3f mm, vs BONE %s, BT device landing err %.3f mm / %.3f deg"
          % (tag, x_plane, x0, resid, chk, comp["bt_device_in_pre"]["flange_err_mm"], comp["bt_device_in_pre"]["axis_err_deg"]))
    print("bounds", bounds, "surfaces", surf)


# ======================================================================================== rendering (py -3.11, pyvista)
def _pv():
    import pyvista as pv
    pv.OFF_SCREEN = True
    return pv


def poly(pv, V, F):
    F = np.asarray(F, int)
    return pv.PolyData(np.asarray(V, float), np.c_[np.full(len(F), 3), F].ravel())


def backdrop_grid(pv, bd, x):
    img = bd["img"]; y = bd["y"]; z = bd["z"]
    g = pv.ImageData(dimensions=(1, len(y), len(z)), spacing=(1.0, float(bd["dy"]), float(bd["dz"])),
                     origin=(float(x), float(y[0]), float(z[0])))
    g.point_data["mri"] = img.ravel(order="F")            # x fastest, then y, then z: index j + ny k
    return g


class Scene:
    """Two fixed viewports.  Left: orthographic view along +x (patient's L-R axis) with anterior (+y) on the LEFT and
    superior (+z) up, the MRI backdrop pushed behind everything (parallel projection: its image does not move).
    Right: oblique 3-D view from the patient's left-anterior-superior, the same slice at its true position, translucent."""

    def __init__(self, tag, size=(1600, 800)):
        pv = self.pv = _pv()
        pd = prep_dir(tag); self.info = load_json(pd + "/prep.json"); bd = np.load(pd + "/backdrop.npz")
        blo = np.array(self.info["bounds"]["lo"]); bhi = np.array(self.info["bounds"]["hi"])      # backdrop / scene
        lo = np.array(self.info["frame"]["lo"]); hi = np.array(self.info["frame"]["hi"]); c = 0.5 * (lo + hi)   # camera
        self.lo, self.hi, self.c = lo, hi, c
        self.pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=list(size), col_weights=[0.42, 0.58], border=True)
        self.size = size
        rest, _ = self.load_obj(P["inputs"] + "/pre_uterus.obj")
        self.rest = rest.decimate(0.7)                     # a lighter wireframe than the 4000-triangle export
        for k in (0, 1):
            self.pl.subplot(0, k); self.pl.set_background("white")
            try:
                self.pl.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)
            except Exception:
                pass
            x_bd = bhi[0] + 15.0 if k == 0 else float(bd["x"])
            self.pl.add_mesh(backdrop_grid(pv, bd, x_bd), scalars="mri", cmap="gray", clim=(0, 255), show_scalar_bar=False,
                             lighting=False, opacity=1.0 if k == 0 else 0.6, name="backdrop", interpolate_before_map=True)
            self.pl.add_mesh(self.rest, style="wireframe", color=COL["rest"], opacity=0.4, line_width=1, name="rest")
        # cameras (fixed for every frame)
        self.pl.subplot(0, 0)
        w0 = 0.42 * size[0]; aspect = w0 / size[1]
        half_h = 0.5 * (hi[2] - lo[2]); half_w = 0.5 * (hi[1] - lo[1])
        self.pscale = max(half_h, half_w / aspect)
        self.cam0 = [(c[0] - 600.0, c[1], c[2]), (c[0], c[1], c[2]), (0.0, 0.0, 1.0)]
        self.pl.subplot(0, 1)
        d = geom.unit([-0.62, 0.62, 0.48])
        self.cam1 = [tuple(c + d * (0.62 * (hi[2] - lo[2]) / np.tan(np.radians(15.0)))), tuple(c), (0.0, 0.0, 1.0)]
        self.pl.add_axes(xlabel="R", ylabel="A", zlabel="S", line_width=3)
        self.apply_cameras()
        self.pl.subplot(0, 0)
        self.pl.add_text("A", position="left_edge", font_size=12, color="white", shadow=True, name="lblA")
        self.pl.add_text("P", position="right_edge", font_size=12, color="white", shadow=True, name="lblP")

    def apply_cameras(self):
        pl = self.pl
        pl.subplot(0, 0); pl.camera_position = self.cam0; pl.camera.parallel_projection = True; pl.camera.parallel_scale = self.pscale
        pl.subplot(0, 1); pl.camera_position = self.cam1; pl.camera.parallel_projection = False; pl.camera.view_angle = 30.0

    def load_obj(self, p):
        V, F = geom.read_obj(p)
        return poly(self.pv, V, F), V

    def both(self, fn):
        for k in (0, 1):
            self.pl.subplot(0, k); fn(k)

    def add_device(self, F, a, L, r, spheres, prefix="sim", col_tandem=None, col_ov=None, shaft=True, ov_opacity=0.85,
                   ov_style="surface", tandem_opacity=1.0):
        pv = self.pv; F = np.asarray(F, float); a = geom.unit(a); tip = F + L * a
        rod = pv.Cylinder(center=F + 0.5 * L * a, direction=a, radius=r, height=L, resolution=32).merge(pv.Sphere(radius=r, center=tip))
        parts = [(rod, col_tandem or COL["tandem"], tandem_opacity, "surface")]
        if shaft:
            sh = pv.Cylinder(center=F - 0.5 * SHAFT_BELOW_FLANGE_MM * a, direction=a, radius=r, height=SHAFT_BELOW_FLANGE_MM, resolution=24)
            parts.append((sh, COL["shaft"], tandem_opacity, "surface"))
            parts.append((pv.Cylinder(center=F, direction=a, radius=2.0 * r, height=1.5, resolution=32), COL["flange"], tandem_opacity, "surface"))
        if spheres:
            ov = pv.merge([pv.Sphere(radius=float(s[1]), center=np.asarray(s[0], float), theta_resolution=24, phi_resolution=24)
                           for s in spheres])
            parts.append((ov, col_ov or COL["ovoid"], ov_opacity, ov_style))
        else:
            for k in (0, 1):
                self.pl.subplot(0, k); self.pl.remove_actor(prefix + "_ovoid")
        names = [prefix + "_rod", prefix + "_shaft", prefix + "_flange", prefix + "_ovoid"]
        if not shaft:
            names = [prefix + "_rod", prefix + "_ovoid"]

        def add(k):
            for (m, col, op, st), nm in zip(parts, names):
                kw = dict(color=col, opacity=op, name=nm, smooth_shading=True)
                if st == "wireframe":
                    kw.update(style="wireframe", line_width=1, smooth_shading=False)
                self.pl.add_mesh(m, **kw)
        self.both(add)

    def add_surface(self, mesh, name, color, opacity, style="surface", line_width=1):
        def add(k):
            if style == "wireframe":
                self.pl.add_mesh(mesh, style="wireframe", color=color, opacity=opacity, line_width=line_width, name=name)
            else:
                self.pl.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True, name=name, specular=0.2)
        self.both(add)

    def add_polyline(self, pts, name, color, radius=0.5):
        pv = self.pv; pts = np.asarray(pts, float)
        tube = pv.lines_from_points(pts).tube(radius=radius, n_sides=10)
        self.both(lambda k: self.pl.add_mesh(tube, color=color, name=name, smooth_shading=True))

    def text(self, k, txt, name, position="upper_left", font_size=10, color=None):
        # the sagittal view is drawn over the MRI: white text with a shadow; the oblique view has a white background
        col = color or ("white" if k == 0 else "black")
        self.pl.subplot(0, k); self.pl.add_text(txt, position=position, font_size=font_size, color=col, name=name, shadow=(k == 0))

    def shot(self, path):
        self.apply_cameras()
        if not getattr(self, "_shown", False):
            self.pl.show(auto_close=False); self._shown = True
        self.pl.render(); self.pl.screenshot(path)

    def close(self):
        self.pl.close()


LEGEND = ("uterus (sim, tan)   HR-CTV (sim, red)   preBT rest uterus (grey wire)   tandem (gold) + shaft   "
          "ovoids (purple)   canal (blue)   backdrop: preBT MRI sagittal slice through the final tandem axis")


def render(tag, every=1, limit=None, fps=9.0, hold_last_s=1.5):
    from PIL import Image
    rd = run_dir(tag); fd = rd + "/frames"; idx = load_json(fd + "/index.json")["frames"]
    if every > 1:
        idx = idx[::every] + ([idx[-1]] if (len(idx) - 1) % every else [])
    if limit:
        idx = idx[:limit]
    log = {}
    if os.path.exists(rd + "/log.jsonl"):
        for ln in open(rd + "/log.jsonl"):
            r = json.loads(ln); log[int(r["step"])] = r
    summ = load_json(rd + "/run.json") if os.path.exists(rd + "/run.json") else {}
    od = ANIM + "/frames_" + tag; os.makedirs(od, exist_ok=True)
    sc = Scene(tag); pngs = []
    sc.text(0, LEGEND, "legend", position="lower_left", font_size=8)
    sc.text(1, "oblique 3-D view (from the patient's left, anterior, superior); slice at its true position, translucent",
            "legend1", position="lower_left", font_size=8, color="dimgray")
    n_total = summ.get("n_steps")
    for i, fr in enumerate(idx):
        dev = load_json(fd + "/" + fr["device"])
        k = int(dev["step"]); row = log.get(k, {})
        ut, _ = sc.load_obj(fd + "/" + fr["surfaces"]["uterus"]); hr, _ = sc.load_obj(fd + "/" + fr["surfaces"]["hrctv"])
        sc.add_surface(ut, "uterus", COL["uterus"], 0.55); sc.add_surface(hr, "hrctv", COL["hrctv"], 0.45)
        sc.add_polyline(dev["canal_mm"], "canal", COL["canal"], radius=0.45)
        sph = [(s["center_mm"], s["r_mm"]) for s in dev.get("spheres", [])]
        sc.add_device(dev["flange_mm"], dev["axis"], float(dev["rod_length_mm"]), float(dev["tandem_r_mm"]), sph)
        umax = row.get("umax", dev.get("umax_mm")); ph = dev.get("phase_name", dev["phase"])
        hold = "  (hold step %d)" % dev["hold_step"] if dev["phase"] == "H" else ""
        txt = ("step %d%s   phase %s (%s)%s\n" % (k, ("/%d" % (n_total - 1)) if n_total else "", dev["phase"], ph, hold)
               + "inserted depth %.1f / %.1f mm (u = %.2f)   tip above the preBT os (rest) %.1f mm\n"
               % (dev["depth_mm"], dev["rod_length_mm"], dev["u"], dev["tip_beyond_O_pre_mm"])
               + "max displacement %.1f mm   mean %.1f mm   ties %d   contacts %d"
               % (umax if umax is not None else float("nan"), dev.get("umean_mm", float("nan")), dev["n_engaged"], dev["n_contacts"]))
        if dev["phase"] == "H":
            txt += "   dx %.3f mm/step" % dev["dx_max_mm"]
        sc.text(0, "sagittal view (A left, S up)   run %s\n" % tag + txt, "info0", font_size=10)
        sc.text(1, txt, "info1", font_size=10)
        pth = od + "/step_%04d.png" % k
        sc.shot(pth); pngs.append(pth)
        print("frame %d/%d step %d %s" % (i + 1, len(idx), k, dev["phase"]), flush=True)
    sc.close()
    # ---- GIF (PIL): 1200x600, one shared palette (no flicker), ~fps, last frame held
    frames = [Image.open(p).convert("RGB").resize((1200, 600), Image.LANCZOS) for p in pngs]
    pal = frames[-1].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG) for f in frames]
    dur = [int(round(1000.0 / fps))] * len(q); dur[-1] = int(hold_last_s * 1000)
    gif = ANIM + "/insertion_%s.gif" % tag
    q[0].save(gif, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=False, disposal=1)
    print("wrote %s (%d frames, %.1f MB)" % (gif, len(q), os.path.getsize(gif) / 1e6))
    with open(ANIM + "/frames_%s/frames.json" % tag, "w") as fh:
        json.dump(dict(tag=tag, pngs=[os.path.basename(p) for p in pngs], fps=fps, hold_last_s=hold_last_s), fh, indent=1)


def surface_distances(pv, A, B):
    """Symmetric mesh-based surface distance (mm): vertices of A to surface B and vice versa (vtkImplicitPolyDataDistance)."""
    import vtk
    out = []
    for src, tgt in ((A, B), (B, A)):
        f = vtk.vtkImplicitPolyDataDistance(); f.SetInput(tgt)
        out.append(np.abs([f.EvaluateFunction(p) for p in src.points]))
    d = np.concatenate(out)
    return dict(msd=float(d.mean()), hd95=float(np.percentile(d, 95)), n=int(len(d)))


def compare(tag):
    rd = run_dir(tag); pd = prep_dir(tag); info = load_json(pd + "/prep.json")
    sc = Scene(tag); pv = sc.pv
    ut, _ = sc.load_obj(rd + "/surf_uterus.obj"); hr, _ = sc.load_obj(rd + "/surf_hrctv.obj")
    but, _ = sc.load_obj(pd + "/bt_uterus.obj"); bhr, _ = sc.load_obj(pd + "/bt_hrctv.obj"); bapp, _ = sc.load_obj(pd + "/bt_applicator.obj")
    sc.add_surface(ut, "uterus", COL["uterus"], 0.7); sc.add_surface(hr, "hrctv", COL["hrctv"], 0.55)
    sc.add_surface(but.decimate(0.8), "bt_uterus", COL["bt_uterus"], 0.9, style="wireframe", line_width=1)
    sc.add_surface(bhr.decimate(0.8), "bt_hrctv", COL["bt_hrctv"], 0.9, style="wireframe", line_width=1)
    sc.add_surface(bapp, "bt_app", COL["bt_app"], 0.25)
    sd = info["sim_device"]; bd = info["bt_device_in_pre"]
    sc.add_device(sd["flange"], sd["axis"], sd["L_iu_mm"], sd["r_tandem_mm"], list(zip(sd["sphere_centers"], sd["sphere_r"])), prefix="sim",
                  ov_opacity=0.5)
    # the real BT tandem as a green wireframe cage around the (coincident) gold simulated rod
    Fb, ab = np.asarray(bd["flange"], float), geom.unit(bd["axis"]); Lb = float(sd["L_iu_mm"])
    cage = pv.Cylinder(center=Fb + 0.5 * Lb * ab, direction=ab, radius=3.4, height=Lb, resolution=20)
    sc.add_surface(cage, "bt_rod", COL["bt_tandem"], 1.0, style="wireframe", line_width=2)
    z = np.load(rd + "/final.npz")
    sc.add_polyline(z["canal_final"], "canal", COL["canal"], radius=0.45)
    du = surface_distances(pv, ut, but); dh = surface_distances(pv, hr, bhr)
    summ = load_json(rd + "/run.json") if os.path.exists(rd + "/run.json") else {}
    txt = ("final state of run %s (%s, %d steps) vs the real post-insertion (BT) anatomy\n" % (tag, summ.get("status", "?"), summ.get("n_steps", 0))
           + "BT labels mapped into the sim frame by E_app (simulated tandem onto the real tandem)\n"
           + "sim: uterus tan, HR-CTV red, tandem gold, ovoids purple\n"
           + "BT: uterus blue wire, HR-CTV orange wire, applicator label green, tandem green cage\n"
           + "mesh-based symmetric surface distance: uterus MSD %.2f mm (HD95 %.1f), HR-CTV %.2f mm (HD95 %.1f)\n" % (du["msd"], du["hd95"], dh["msd"], dh["hd95"])
           + "device landing (BT tandem in the sim frame): flange %.2f mm, axis %.2f deg; E_app resid %.2f mm"
           % (bd["flange_err_mm"], bd["axis_err_deg"], info["E_app"]["landmark_resid_max_mm"]))
    sc.text(0, "sagittal view (A left, S up): final simulated anatomy (solid) vs real BT anatomy (wireframe)", "info0", font_size=10)
    sc.text(1, "oblique 3-D view\n" + txt, "info1", font_size=10)
    out = ANIM + "/final_vs_BT_%s.png" % tag
    sc.shot(out); sc.close()
    rec = dict(tag=tag, mesh_distances=dict(uterus=du, hrctv=dh), note="marching-cubes BT surfaces vs the deformed sim surfaces, "
               "vertex-to-surface both ways; NOT the voxel metrics of final_eval.py")
    with open(ANIM + "/final_vs_BT_%s.json" % tag, "w") as fh:
        json.dump(rec, fh, indent=1)
    print("wrote", out, json.dumps(rec["mesh_distances"]))


def mp4(tag, fps=9.0, hold_last_s=1.5):
    pk = os.environ.get("APPSIM_PK")
    if pk and os.path.isdir(pk):
        sys.path.insert(0, pk)
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401
    except Exception as e:
        print("mp4 skipped: %s" % e); return None
    meta = load_json(ANIM + "/frames_%s/frames.json" % tag); fd = ANIM + "/frames_" + tag
    out = ANIM + "/insertion_%s.mp4" % tag
    w = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=16)
    n_hold = int(round(hold_last_s * fps))
    for i, p in enumerate(meta["pngs"]):
        im = imageio.imread(fd + "/" + p)[:, :, :3]
        for _ in range(n_hold if i == len(meta["pngs"]) - 1 else 1):
            w.append_data(im)
    w.close()
    print("wrote %s (%.1f MB)" % (out, os.path.getsize(out) / 1e6))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "render", "compare", "mp4"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--every", type=int, default=1, help="render every N-th exported frame")
    ap.add_argument("--limit", type=int, default=None, help="render only the first N frames (tests)")
    ap.add_argument("--fps", type=float, default=9.0)
    a = ap.parse_args()
    os.makedirs(ANIM, exist_ok=True)
    if a.mode == "prep":
        prep(a.tag)
    elif a.mode == "render":
        render(a.tag, every=a.every, limit=a.limit, fps=a.fps)
    elif a.mode == "compare":
        compare(a.tag)
    else:
        mp4(a.tag, fps=a.fps)


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
