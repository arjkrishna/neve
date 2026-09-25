"""Labelled stills of one state of a hybrid run, in the two orientations of the full-view MP4: the sagittal section
through the applicator seen from the patient's left, and the oblique 3-D view from the left-anterior-superior.
Every organ, the device parts and the anatomical landmarks are called out with leader lines: introitus, vaginal
lumen, fornices / vault, the vagina-HR-CTV junction, external and internal os, the intrauterine canal (the
pre-insertion canal polyline carried through the simulated deformation), fundus, tandem tip, flange -- and, in
red, where the model's contact fails (wall nodes inside the cervix, by signed distance).

    APPSIM_OUT=<render tree> py -3.11 hybrid/label_views.py --tag G32 [--step 180] [--size 1400]

The render tree must be the one whose meshes/ junction matches the run's wall (README, "Videos"), and it needs an
inputs/ junction (canal.npz) and, for the tip-to-serosa figure, hybrid/eval/.  Every number printed on the figure
is read or computed from the run's files; nothing is hard-coded.  Writes figs/labeled/<tag>_step<k>_{sagittal,
oblique}.png -- patient-derived, local only, never committed."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import geom  # noqa: E402
import vagina_wall as VW  # noqa: E402
import animate_hybrid as AH  # noqa: E402

P = AH.P
BODIES = AH.BODIES
OVOID_BODY = ("ovoid_L", "ovoid_R", "rod_L", "rod_R", "packing")
COL_CANAL = (0.0, 0.70, 0.85)
COL_ROD = (0.45, 0.45, 0.50)
COL_PACK = (0.93, 0.88, 0.72)
COL_DEFECT = (0.85, 0.05, 0.05)
TXT_DEVICE = (0.20, 0.20, 0.24)
TXT_LANDMARK = (0.0, 0.0, 0.0)
TXT_DEFECT = (0.80, 0.05, 0.05)
# per-view opacity: the sagittal matches the MP4's cut panel; the oblique makes the wall / uterus / cervix
# see-through so that the applicator inside them can be labelled
OPA = dict(sagittal=dict(corpus=0.97, cervix=0.97, vagina=1.0, bladder=0.55, rectum=0.75, sigmoid=0.55),
           oblique=dict(corpus=0.33, cervix=0.35, vagina=0.18, bladder=0.10, rectum=0.40, sigmoid=0.22))
MIN_SEP_PX = 46.0                      # anchors of different labels are kept at least this far apart where possible


# ------------------------------------------------------------------------------------------------ state
def idw(X0, u, Q, k=6):
    """Inverse-distance interpolation of the node displacements u (at X0) onto the points Q."""
    d = np.linalg.norm(X0[None, :, :] - Q[:, None, :], axis=2)
    nn = np.argsort(d, axis=1)[:, :k]
    w = 1.0 / np.maximum(np.take_along_axis(d, nn, 1), 1e-6) ** 2
    return (w[:, :, None] * u[nn]).sum(1) / w.sum(1)[:, None]


def implicit_distance(V, F):
    """Signed distance function of a closed surface (negative inside)."""
    import vtk
    from vtk.util import numpy_support as ns
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(V, float), deep=1))
    ca = vtk.vtkCellArray()
    ca.SetCells(len(F), ns.numpy_to_vtkIdTypeArray(np.c_[np.full(len(F), 3), F].astype(np.int64).ravel(), deep=1))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(ca)
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(pd)
    return lambda X: np.array([imp.EvaluateFunction(*map(float, x)) for x in np.atleast_2d(X)])


class State:
    """Bodies (surfaces + tet nodes), device parts and landmarks of one frame of a run, in preBT world mm."""

    def __init__(self, tag, step=None):
        self.tag = tag
        rd = AH.run_dir(tag)
        frs = AH.load_index(tag)["frames"]
        fr = frs[-1] if step is None else [f for f in frs if int(f["step"]) == int(step)][0]
        self.step, self.is_last = int(fr["step"]), fr is frs[-1]
        fd = AH.frames_dir(tag)
        self.dev = AH.load_json(fd + "/" + fr["device"])
        self.cfg = AH.load_json(rd + "/cfg.json")
        self.surf = {b: tuple(np.asarray(x) for x in geom.read_obj("%s/%s" % (fd, fr["surfaces"][b]))) for b in BODIES}
        self.col = {b: tuple(AH.load_json(P["meshes"] + "/bodies.json")["bodies"][b]["color"]) for b in BODIES}
        self.meta = {b: AH.load_json("%s/%s/meta.json" % (P["meshes"], b)) for b in BODIES}
        self.X0 = {b: np.asarray(VW.read_vtk_legacy("%s/%s/tets.vtk" % (P["meshes"], b))[0], float) for b in BODIES}
        self.X = {}
        for b in BODIES:                       # node positions: exact for the last frame, else from the surface
            fn = "%s/final/%s_u.npy" % (rd, b)
            if self.is_last and os.path.exists(fn):
                self.X[b] = self.X0[b] + np.load(fn)
            else:
                s2n = np.asarray(self.meta[b]["surface_obj_vertex_to_tet_node"], int)
                self.X[b] = self.X0[b] + idw(self.X0[b][s2n], self.surf[b][0] - self.X0[b][s2n], self.X0[b])
        d = self.dev
        self.F, self.a, self.tip = (np.asarray(d["flange_mm"], float), geom.unit(d["tube_axis"]),
                                    np.asarray(d["tip_mm"], float))
        self.R_t = np.array([d["x_app"], d["y_app"], d["tube_axis"]], float)
        self.R_o = np.array([d["ovoid_x_app"], d["ovoid_y_app"], d["ovoid_axis"]], float) \
            if "ovoid_axis" in d else self.R_t
        self.Fo = np.asarray(d["ovoid_origin_mm"], float)
        self.app = AH.load_json(P["applicator"] + "/applicator.json")["params"]
        th = np.radians(float(self.app["angle_deg"]["value"]))
        self.d_rod_app, self.y_rod_app = np.array([0.0, np.sin(th), -np.cos(th)]), np.array([0.0, np.cos(th), np.sin(th)])
        parts = ["tube", "shaft", "ovoid_L", "ovoid_R"] + (["rod_L", "rod_R"] if self.cfg.get("device_rods") else []) \
            + (["packing"] if self.cfg.get("device_packing") else [])
        self.parts = {}
        for p in parts:
            V, Fc = geom.read_obj("%s/%s.obj" % (P["applicator"], p))
            org, R = (self.Fo, self.R_o) if p in OVOID_BODY else (self.F, self.R_t)
            self.parts[p] = (org + np.asarray(V, float) @ R, np.asarray(Fc, int))
        # the wall's rings
        w = self.meta["vagina"]["wall"]
        self.gi = np.asarray(w["grid_index"], int)
        self.n_ax, self.n_rd = int(w["n_axial"]), int(w["n_radial"])
        self.a_v = geom.unit(self.X0["vagina"][self.inner(self.n_ax - 1)].mean(0) - self.X0["vagina"][self.inner(0)].mean(0))
        y = np.array([0.0, 1.0, 0.0])
        self.e_ant = geom.unit(y - (y @ self.a_v) * self.a_v)      # anterior, in the wall's ring planes
        # the pre-insertion canal polyline carried by the tissue: each point moves with its own body
        cz = np.load(P["out"] + "/inputs/canal.npz")
        pts, self.i_ios = np.asarray(cz["pts"], float), int(cz["i_internal_os"])
        dc = np.linalg.norm(pts[:, None] - self.X0["cervix"][None], axis=2).min(1)
        dk = np.linalg.norm(pts[:, None] - self.X0["corpus"][None], axis=2).min(1)
        uc = idw(self.X0["cervix"], self.X["cervix"] - self.X0["cervix"], pts)
        uk = idw(self.X0["corpus"], self.X["corpus"] - self.X0["corpus"], pts)
        self.canal = pts + np.where((dk < dc)[:, None], uk, uc)
        self.canal_rest = pts

    def inner(self, k):
        return np.nonzero((self.gi[:, 0] == k) & (self.gi[:, 1] == 0))[0]

    def tube_dist(self, Q):
        q = np.atleast_2d(Q) - self.F
        return np.linalg.norm(q - np.outer(q @ self.a, self.a), axis=1)

    def facts(self):
        """Numbers printed on the figure, all computed from the run's files."""
        f = {}
        Xw, X0w = self.X["vagina"], self.X0["vagina"]
        top = self.inner(self.n_ax - 1)
        f["vault_lift"] = float((Xw[top].mean(0) - X0w[top].mean(0)) @ self.a_v)
        f["os_below_flange"] = float(-(self.canal[0] - self.F) @ self.a)
        f["os_lift"] = float((self.canal[0] - self.canal_rest[0]) @ self.a_v)
        f["canal_cervix_max"] = float(self.tube_dist(self.canal[: self.i_ios + 1]).max())
        f["canal_cavity_max"] = float(self.tube_dist(self.canal[self.i_ios:]).max())
        f["angle"] = float(self.app["angle_deg"]["value"])
        f["corpus_rot"] = float(AH.load_json(P["applicator"] + "/pose.json")["corpus"]["rotation_deg"])
        if "packing" in self.parts:
            q = self.parts["packing"][0] - self.Fo
            dr = self.d_rod_app @ self.R_o
            f["pack_r"] = float(np.percentile(np.linalg.norm(q - np.outer(q @ dr, dr), axis=1), 90))
        ev = "%s/eval/%s/metrics.json" % (P["hybrid"], self.tag)
        if os.path.exists(ev):
            f["tip_to_serosa"] = float(AH.load_json(ev)["frames"]["E_app"]["landing"]["tip_to_serosa_mm"])
        # the junction by SIGNED distance of the lumen nodes to the cervix (< 0 = inside the cervix)
        sdf = implicit_distance(*self.surf["cervix"])
        inn = np.nonzero(self.gi[:, 1] == 0)[0]
        sd = sdf(Xw[inn])
        self.junction = inn[(sd >= 0.0) & (sd < 1.5)]
        self.pen, self.pen_depth = inn[sd < 0.0], -sd[sd < 0.0]
        f["rim_contact_pct"] = float(100.0 * np.mean(sdf(Xw[top]) < 1.5))
        f["pen_n"] = int(len(self.pen))
        f["pen_max"] = float(self.pen_depth.max()) if len(self.pen) else 0.0
        f["umax"] = {b: float(v["umax_mm"]) for b, v in self.dev.get("disp", {}).items()}
        return f


# ------------------------------------------------------------------------------------------------ anchors
def ring_cut_points(st, k, x_cut):
    """Anterior and posterior points of wall ring k on the kept side of the sagittal cut (x >= x_cut)."""
    Q = st.X["vagina"][st.inner(k)]
    c = Q.mean(0)
    out = {}
    for side, sgn in (("ant", 1.0), ("post", -1.0)):
        m = (Q[:, 0] >= x_cut) & (sgn * ((Q - c) @ st.e_ant) > 0)
        if m.any():
            out[side] = Q[m][np.argmin(Q[m][:, 0] - x_cut)]
    return out


def rim_point(pm, x_cut):
    """A point on the cut outline of a clipped surface, nearest the outline's median."""
    R = np.asarray(pm.points)
    Rc = R[np.abs(R[:, 0] - x_cut) < 1e-3]
    R = Rc if len(Rc) else R
    ref = np.median(R, 0)
    return R[np.argmin(np.linalg.norm(R[:, 1:] - ref[1:], axis=1))]


def front_point(V, cam, target):
    """The surface point nearest the camera along the line of sight to `target`."""
    V = np.asarray(V, float)
    r = geom.unit(np.asarray(target, float) - cam)
    q = V - cam
    t = q @ r
    perp = np.linalg.norm(q - np.outer(t, r), axis=1)
    for tol in (1.5, 3.0, 6.0, 12.0, 1e9):
        m = perp < tol
        if m.any():
            return V[m][np.argmin(t[m])]


def pick_clear(C, proj, taken, prefer):
    """From candidate points C, the most preferred one (smallest `prefer`) whose projection keeps MIN_SEP_PX from
    every anchor already placed; if none does, the one farthest from them."""
    C = np.asarray(C, float)
    pc = np.array([proj(q) for q in C])
    if taken:
        dmin = np.min(np.linalg.norm(pc[:, None, :] - np.array(taken)[None, :, :], axis=2), axis=1)
    else:
        dmin = np.full(len(C), np.inf)
    key = np.array([prefer(q, p) for q, p in zip(C, pc)])
    ok = dmin >= MIN_SEP_PX
    return C[np.nonzero(ok)[0][np.argmin(key[ok])]] if ok.any() else C[np.argmax(dmin)]


def hull2d(P):
    """Convex hull of 2-D points (Andrew's monotone chain), counter-clockwise, closed."""
    P = sorted(set(map(tuple, np.round(np.asarray(P, float), 2))))
    if len(P) < 3:
        return np.array(P)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for q in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    h = lo[:-1] + up[:-1]
    return np.array(h + [h[0]])


# ------------------------------------------------------------------------------------------------ render
def oblique_camera(st, Sw, Sh):
    """Left-anterior-superior view direction of the MP4, framed on the column from the introitus to the fundus."""
    d = geom.unit([-0.62, 0.62, 0.48])
    v = -d
    up = np.array([0.0, 0.0, 1.0])
    up_s = geom.unit(up - (up @ v) * v)
    intro = st.X["vagina"][st.inner(0)].mean(0)
    Vk = st.surf["corpus"][0]
    top = Vk[np.argmax((Vk - st.F) @ st.a)]
    focal = 0.5 * (intro + top)
    chord = abs((top - intro) @ up_s) + 2 * 22.0
    D = 0.5 * chord / np.tan(np.radians(15.0))
    return focal + d * D, focal


def render(st, view, Sw, Sh, add_actors=None, after=None, labels=True):
    """Render one view.  Hooks (used by overlay_views.py): add_actors(pl, pv, dict(x_cut, cut)) adds actors before
    the render; after(proj, info) runs with the live projection before the plotter closes, its result is returned."""
    pv = AH._pv()
    cut = view == "sagittal"
    x_cut = float(st.F[0] + 0.5 * float(st.dev["L_iu_mm"]) * st.a[0])          # the MP4's cut plane
    Vall = np.vstack([st.surf[b][0] for b in BODIES] + [V for V, _ in st.parts.values()])
    lo, hi = Vall.min(0) - 4.0, Vall.max(0) + 4.0
    c = 0.5 * (lo + hi)
    pl = pv.Plotter(off_screen=True, window_size=[Sw, Sh])
    pl.set_background("white")
    try:
        pl.enable_depth_peeling(number_of_peels=12, occlusion_ratio=0.0)
    except Exception:
        pass
    op = OPA[view]
    clipped = {}
    for b in BODIES:
        m = AH.poly(pv, *st.surf[b])
        if cut:
            m = m.clip(normal="x", origin=(x_cut, 0, 0), invert=False)
        clipped[b] = m
        sil = dict(color=tuple(0.55 * np.asarray(st.col[b])), line_width=1.3) \
            if (not cut and b in ("vagina", "cervix", "corpus")) else None
        pl.add_mesh(m, color=st.col[b], opacity=op[b], smooth_shading=True, specular=0.15, silhouette=sil)
    for p, (V, Fc) in st.parts.items():
        m = AH.poly(pv, V, Fc)
        if cut:
            m = m.clip(normal="x", origin=(x_cut, 0, 0), invert=False)
        if m.n_points == 0:
            continue
        colr = AH.COL_TANDEM if p in ("tube", "shaft") else (COL_PACK if p == "packing" else
                                                              (COL_ROD if p.startswith("rod") else AH.COL_OVOID))
        opac = 1.0 if p in ("tube", "shaft") or p.startswith("rod") else \
            (0.10 if p == "packing" else (0.55 if cut else 0.95))
        sil = None if (cut or p in ("tube", "shaft")) else dict(color=(0.25, 0.25, 0.28), line_width=1.2)
        pl.add_mesh(m, color=colr, opacity=opac, smooth_shading=True, silhouette=sil)
        if p == "packing" and cut:                             # its cut outline, so the faint cylinder reads
            e = m.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False,
                                        non_manifold_edges=False)
            if e.n_points:
                pl.add_mesh(e, color=(0.62, 0.50, 0.30), line_width=3)
    pl.add_mesh(pv.Spline(st.canal, 300).tube(radius=0.9), color=COL_CANAL, smooth_shading=True)
    if not cut and len(st.pen):                               # the contact defect: wall nodes inside the cervix
        pl.add_points(st.X["vagina"][st.pen], color=COL_DEFECT, point_size=11, render_points_as_spheres=True)
    if add_actors is not None:
        add_actors(pl, pv, dict(x_cut=x_cut, cut=cut))
    if cut:
        pscale = 0.5 * max(hi[2] - lo[2], hi[1] - lo[1]) * 1.04
        cam = np.array([c[0] - 600.0, c[1], c[2]])
        pl.camera_position = [tuple(cam), tuple(c), (0.0, 0.0, 1.0)]
        pl.camera.parallel_projection = True
        pl.camera.parallel_scale = pscale
    else:
        cam, focal = oblique_camera(st, Sw, Sh)
        pl.camera_position = [tuple(cam), tuple(focal), (0.0, 0.0, 1.0)]
        pl.camera.parallel_projection = False
        pl.camera.view_angle = 30.0
        pl.add_axes(xlabel="R", ylabel="A", zlabel="S", line_width=3, viewport=(0.78, 0.0, 1.0, 0.16))
    pl.renderer.reset_camera_clipping_range()
    pl.show(auto_close=False)
    pl.render()
    img = pl.screenshot(return_img=True)
    ren = pl.renderer
    ww, wh = pl.ren_win.GetSize()

    def proj(p):
        ren.SetWorldPoint(float(p[0]), float(p[1]), float(p[2]), 1.0)
        ren.WorldToDisplay()
        x, y, _ = ren.GetDisplayPoint()
        return np.array([x * img.shape[1] / ww, img.shape[0] - y * img.shape[0] / wh])

    outlines = []
    if not cut:                                              # the caps sit behind tissue: outline them
        for pname in ("ovoid_L", "ovoid_R"):
            outlines.append(hull2d([proj(q) for q in st.parts[pname][0]]))
    info = dict(x_cut=x_cut, cam=cam, clipped=clipped, proj=proj, outlines=outlines,
                px_per_mm=(img.shape[0] / (2 * pscale)) if cut else None)
    lab = labels_for(st, view, info) if labels else []
    for L in lab:
        L["px"] = [proj(q) for q in L["pts"]]
    extra = after(proj, info) if after is not None else None
    pl.close()
    return img, lab, info, extra


def labels_for(st, view, info):
    """The labels of one view: name, descriptor lines, category, colour and their 3-D anchor points.  Fixed
    anchors are placed first; the flexible ones (vault rim, junction, caps, defect) then keep clear of them."""
    f = st.fx
    cut = view == "sagittal"
    x_cut, cam, proj = info["x_cut"], info["cam"], info["proj"]
    L, taken = [], []

    def add(name, desc, cat, pts, color=None, side=None):
        pts = [np.asarray(q, float) for q in pts if q is not None]
        if pts:
            L.append(dict(name=name, desc=desc, cat=cat, pts=pts, color=color, side=side))
            taken.extend(proj(q) for q in pts)

    um = f["umax"]
    # ---- fixed: device axis points and canal landmarks
    Lt = float(st.dev["L_iu_mm"])
    add("tandem tip", ("%.1f mm from the fundal serosa" % f["tip_to_serosa"]) if "tip_to_serosa" in f else "",
        "device", [st.tip])
    add("tandem (intrauterine tube)", "in the uterine cavity", "device", [st.F + 0.55 * Lt * st.a])
    add("flange", "tandem bends %.1f° here; ring plane" % f["angle"], "device", [st.F], side=None if cut else "L")
    add("external os", "%.0f mm below the flange" % f["os_below_flange"], "landmark", [st.canal[0]],
        side=None if cut else "R")
    add("internal os", "top of the cervical canal", "landmark", [st.canal[st.i_ios]])
    add("intrauterine canal", "pre-insertion canal, moved with the tissue:\n≤ %.1f mm from the tandem in the cervix,\n"
        "curving %.0f mm off it near the fundus (rigid uterus)" % (f["canal_cervix_max"], f["canal_cavity_max"]),
        "landmark", [st.canal[(st.i_ios + len(st.canal)) // 2]], color=COL_CANAL)
    Vk = st.surf["corpus"][0]
    q = Vk - st.F
    h = q @ st.a
    lat = np.linalg.norm(q - np.outer(h, st.a), axis=1)
    m = (lat < 15.0) & ((Vk[:, 0] >= x_cut) if cut else True)
    add("uterine fundus", "", "landmark", [Vk[m][np.argmax(h[m])] if m.any() else None])
    d_rod_t = st.d_rod_app @ st.R_t
    add("tandem rod", "straight, along the vagina", "device", [st.F + 45.0 * d_rod_t])
    if "rod_L" in st.parts:
        offs = st.app["ovoid_rod_offsets_mm"]["value"]
        rq = [st.Fo + (ox * np.array([1.0, 0, 0]) + oy * st.y_rod_app + 40.0 * st.d_rod_app) @ st.R_o for ox, oy in offs]
        if cut:
            rq = [p_ for p_ in rq if p_[0] >= x_cut]
        add("ovoid rod" + ("" if cut else "s"), "%.0f mm anterior of the tandem rod" % offs[0][1], "device", rq)
    # ---- organs
    organ_desc = dict(corpus="rigid; rotated %.0f° with the tandem" % f["corpus_rot"],
                      cervix="drawn up %.0f mm onto the tandem" % f["os_lift"],
                      bladder="displaced up to %.0f mm" % um.get("bladder", float("nan")),
                      rectum="displaced up to %.0f mm" % um.get("rectum", float("nan")),
                      sigmoid="displaced up to %.0f mm" % um.get("sigmoid", float("nan")))
    names = dict(corpus="uterus (corpus)", cervix="HR-CTV / cervix", bladder="bladder", rectum="rectum",
                 sigmoid="sigmoid colon")
    for b in ("corpus", "cervix", "bladder", "rectum", "sigmoid"):
        V = st.surf[b][0]
        if cut:
            qb = rim_point(info["clipped"][b], x_cut)
        elif b in ("corpus", "cervix"):                     # on the anterior surface, clear of the tandem
            qb = front_point(V, cam, V.mean(0) + 12.0 * st.e_ant)
        elif b == "rectum":                                 # its lower part, clear of the ring
            hh = (V - st.F) @ st.a_v
            mm = (hh < -15.0) & (hh > -45.0)
            qb = front_point(V[mm], cam, V[mm].mean(0)) if mm.any() else front_point(V, cam, V.mean(0))
        else:
            qb = front_point(V, cam, V.mean(0))
        add(names[b], organ_desc[b], "organ", [qb], color=st.col[b])
    Qw = st.X["vagina"][st.inner(st.n_ax // 3)]
    qw = ring_cut_points(st, st.n_ax // 3, x_cut).get("post") if cut else Qw[np.argmin(np.linalg.norm(Qw - cam, axis=1))]
    add("vaginal wall", "1.2 mm wall; vault lifted %.0f mm" % f["vault_lift"], "organ", [qw], color=st.col["vagina"])
    if "packing" in st.parts:
        pr = f.get("pack_r", 17.0)
        qp = st.Fo + (30.0 * st.d_rod_app - pr * st.y_rod_app) @ st.R_o if cut else \
            front_point(st.parts["packing"][0], cam, st.Fo + (30.0 * st.d_rod_app) @ st.R_o)
        add("packing (model)", "%.0f mm cylinder around the rods" % pr, "device", [qp])
    # ---- introitus, lumen, vault, junction, caps, defect
    if cut:
        r0 = ring_cut_points(st, 0, x_cut)
        add("introitus", "vaginal opening", "landmark", [r0.get("ant"), r0.get("post")])
        add("vaginal lumen", "held open by the applicator", "landmark",
            [st.X["vagina"][st.inner(st.n_ax // 2)].mean(0)])
        rt = ring_cut_points(st, st.n_ax - 1, x_cut)
        add("vagina–HR-CTV junction", "anterior fornix; the only contact\nin this plane (the rest is lateral)",
            "landmark", [rt.get("ant")])
        add("posterior fornix", "vault rim, open in this plane", "landmark", [rt.get("post")])
        oc = [np.asarray(q_, float) for q_ in st.dev["ovoid_centres_mm"]]
        add("ovoid (ring cap)", "right cap; the left one is in the removed half", "device",
            [q_ for q_ in oc if q_[0] >= x_cut])
    else:
        Q0 = st.X["vagina"][st.inner(0)]
        add("introitus", "vaginal opening", "landmark", [Q0[np.argmin(np.linalg.norm(Q0 - cam, axis=1))]])
        if len(st.pen):
            Pp = st.X["vagina"][st.pen]
            dp = pick_clear(Pp, proj, taken, lambda q_, p_: -float(st.pen_depth[np.argmin(np.linalg.norm(Pp - q_, axis=1))]))
            add("contact defect", "cervix crosses the wall by up to %.1f mm\n(red: %d wall nodes inside the cervix)"
                % (f["pen_max"], f["pen_n"]), "defect", [dp])
        J = st.X["vagina"][st.junction]
        if len(J):
            j1 = pick_clear(J, proj, taken, lambda q_, p_: p_[0])
            j2 = pick_clear(J, proj, taken + [proj(j1)], lambda q_, p_: -p_[0])
            add("vagina–HR-CTV junction", "wall on the cervix over %.0f %% of the vault rim" % f["rim_contact_pct"],
                "landmark", [j1, j2])
        Qt = st.X["vagina"][st.inner(st.n_ax - 1)]
        add("vaginal vault (rim)", "lifted %.0f mm with the cervix" % f["vault_lift"], "landmark",
            [pick_clear(Qt, proj, taken, lambda q_, p_: p_[1])])
        caps = []
        for pname in ("ovoid_L", "ovoid_R"):
            Vc = st.parts[pname][0]
            caps.append(pick_clear(Vc, proj, taken + [proj(c_) for c_ in caps],
                                   lambda q_, p_: float(np.linalg.norm(q_ - cam))))
        add("ovoids (ring caps)", "left and right, perpendicular to the tandem", "device", caps)
    return L


# ------------------------------------------------------------------------------------------------ compose
def place(ys, heights, top, bottom, gap=8.0):
    """1-D label placement: keep each label near its target y without overlaps, inside [top, bottom]."""
    n = len(ys)
    y = np.array(ys, float)
    for _ in range(60):
        for i in range(1, n):
            need = 0.5 * (heights[i - 1] + heights[i]) + gap
            if y[i] - y[i - 1] < need:
                mid = 0.5 * (y[i] + y[i - 1])
                y[i - 1], y[i] = mid - 0.5 * need, mid + 0.5 * need
        if n:
            y[0] = max(y[0], top + 0.5 * heights[0])
            y[-1] = min(y[-1], bottom - 0.5 * heights[-1])
        for i in range(n - 2, -1, -1):
            y[i] = min(y[i], y[i + 1] - (0.5 * (heights[i] + heights[i + 1]) + gap))
        for i in range(1, n):
            y[i] = max(y[i], y[i - 1] + (0.5 * (heights[i - 1] + heights[i]) + gap))
    return y


def compose(img, labels, info, view, st, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    Sh, Sw = img.shape[:2]
    mL = mR = 560
    top, bot = 120, 150
    W, H = mL + Sw + mR, top + Sh + bot
    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.imshow(img, extent=(mL, mL + Sw, top + Sh, top), aspect="auto", zorder=0)
    halo = [pe.withStroke(linewidth=3.0, foreground="white")]
    mk = dict(organ="o", device="s", landmark="D", defect="X")
    for h2 in info.get("outlines", []):
        if len(h2) >= 3:
            H2 = h2 + np.array([mL, top])
            ax.fill(H2[:, 0], H2[:, 1], color=(0.80, 0.80, 0.84), alpha=0.45, zorder=1, lw=0)
            ax.plot(H2[:, 0], H2[:, 1], color=(0.15, 0.15, 0.18), lw=1.7, ls=(0, (6, 3)), zorder=2)
    split = []
    for Lb in labels:                  # anchors on both halves of the image: one label per side, no crossing leader
        xs = np.array([q[0] for q in Lb["px"]])
        if Lb["side"] is None and len(xs) > 1 and xs.min() < 0.5 * Sw < xs.max() and np.ptp(xs) > 0.18 * Sw:
            left = [q for q, x_ in zip(Lb["px"], xs) if x_ < 0.5 * Sw]
            right = [q for q, x_ in zip(Lb["px"], xs) if x_ >= 0.5 * Sw]
            split.append(dict(Lb, px=left, side="L"))
            split.append(dict(Lb, px=right, side="R", desc=""))
        else:
            split.append(Lb)
    labels = split
    for Lb in labels:
        P2 = np.array(Lb["px"]) + np.array([mL, top])
        Lb["P2"] = P2
        Lb["side"] = Lb["side"] or ("L" if P2[:, 0].mean() < mL + 0.5 * Sw else "R")
        Lb["lines"] = [s for s in Lb["desc"].split("\n") if s] if Lb["desc"] else []
        Lb["h"] = 24.0 + 17.0 * len(Lb["lines"])
        Lb["tcol"] = (tuple(0.72 * np.asarray(Lb["color"])) if Lb["cat"] == "organ" else
                      (TXT_DEVICE if Lb["cat"] == "device" else (TXT_DEFECT if Lb["cat"] == "defect" else
                       (tuple(0.8 * np.asarray(Lb["color"])) if Lb["color"] else TXT_LANDMARK))))
    for side in ("L", "R"):
        grp = sorted([Lb for Lb in labels if Lb["side"] == side], key=lambda Lb: Lb["P2"][:, 1].mean())
        ys = place([Lb["P2"][:, 1].mean() for Lb in grp], [Lb["h"] for Lb in grp], top + 6, top + Sh - 6)
        for Lb, y in zip(grp, ys):
            xt = mL - 22 if side == "L" else mL + Sw + 22
            ha = "right" if side == "L" else "left"
            yn = y - 0.5 * Lb["h"] + 12
            ax.text(xt, yn, Lb["name"], ha=ha, va="center", fontsize=14, fontweight="bold", color=Lb["tcol"], zorder=5)
            for i, s in enumerate(Lb["lines"]):
                ax.text(xt, yn + 21 + 17 * i, s, ha=ha, va="center", fontsize=11, color=(0.33, 0.33, 0.36), zorder=5)
            x0 = mL - 14 if side == "L" else mL + Sw + 14
            for q in Lb["P2"]:
                ax.plot([x0, q[0]], [yn, q[1]], color=Lb["tcol"], lw=1.2, zorder=6, path_effects=halo,
                        solid_capstyle="round")
                ax.plot([q[0]], [q[1]], marker=mk[Lb["cat"]], ms=8.0, mfc=Lb["tcol"], mec="white", mew=1.4, zorder=7)
    ttl = dict(sagittal="Seated applicator: sagittal section through the device",
               oblique="Seated applicator: oblique 3-D view")[view]
    sub = dict(sagittal="Cut through the applicator's sagittal plane and viewed from the patient's left; the near "
                        "(left) half is removed.  Anterior is to the left, superior is up.",
               oblique="Viewed from the patient's left, anterior and superior; nothing cut.  All tissue is drawn "
                       "see-through to show the applicator inside.")[view]
    ax.text(W / 2, 40, ttl, ha="center", va="center", fontsize=21, fontweight="bold")
    ax.text(W / 2, 80, "run %s, step %d (end of insertion, after settling).  %s" % (st.tag, st.step, sub),
            ha="center", va="center", fontsize=12.5, color=(0.25, 0.25, 0.28))
    if view == "sagittal":
        ax.text(mL + 16, top + 22, "◀ anterior", ha="left", va="center", fontsize=12, color=(0.3, 0.3, 0.3))
        ax.text(mL + Sw - 16, top + 22, "posterior ▶", ha="right", va="center", fontsize=12, color=(0.3, 0.3, 0.3))
        ax.text(mL + Sw / 2, top + 22, "▲ superior", ha="center", va="center", fontsize=12, color=(0.3, 0.3, 0.3))
        ppm = info["px_per_mm"]
        x1, yb = mL + Sw - 40, top + Sh - 30
        ax.plot([x1 - 20 * ppm, x1], [yb, yb], color="black", lw=3)
        ax.text(x1 - 10 * ppm, yb - 14, "20 mm", ha="center", va="center", fontsize=11.5)
    ky = top + Sh + 55
    items = [("o", (0.55, 0.55, 0.58), "organ (text in its colour)"), ("s", TXT_DEVICE, "applicator part"),
             ("D", TXT_LANDMARK, "anatomical landmark")]
    if any(Lb["cat"] == "defect" for Lb in labels):
        items.append(("X", TXT_DEFECT, "model defect"))
    x = W / 2 - 150 * (len(items) + 2)
    for m_, c_, t_ in items:
        ax.plot([x], [ky], marker=m_, ms=10, mfc=c_, mec="white", mew=1.4)
        ax.text(x + 16, ky, t_, ha="left", va="center", fontsize=12)
        x += 300
    ax.plot([x, x + 34], [ky, ky], color=COL_CANAL, lw=5, solid_capstyle="round")
    ax.text(x + 44, ky, "intrauterine canal (pre-insertion, moved with the tissue)", ha="left", va="center", fontsize=12)
    if info.get("outlines"):
        x += 560
        ax.plot([x, x + 34], [ky, ky], color=(0.15, 0.15, 0.18), lw=1.7, ls=(0, (6, 3)))
        ax.text(x + 44, ky, "ovoid caps seen through tissue", ha="left", va="center", fontsize=12)
    ax.text(W / 2, ky + 48, "Tandem and tandem rod black; ovoid caps light grey; ovoid rods mid grey; packing pale.  "
                            "Patient-derived figure: keep local.", ha="center", va="center", fontsize=11,
            color=(0.35, 0.35, 0.38))
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--step", type=int, default=None, help="frame step (default: the last frame)")
    ap.add_argument("--size", type=int, default=1400, help="render height in px")
    ap.add_argument("--views", default="sagittal,oblique")
    a = ap.parse_args()
    AH.run_extra_parts(a.tag)
    st = State(a.tag, a.step)
    st.fx = st.facts()
    od = P["figs"] + "/labeled"
    os.makedirs(od, exist_ok=True)
    for view in a.views.split(","):
        Sh = a.size if view == "sagittal" else int(a.size * 1.15)
        Sw = a.size if view == "sagittal" else int(a.size * 0.93)
        img, lab, info, _ = render(st, view, Sw, Sh)
        out = "%s/%s_step%04d_%s.png" % (od, a.tag, st.step, view)
        compose(img, lab, info, view, st, out)
        print("wrote", out, "(%d labels)" % len(lab))
    print("facts:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in st.fx.items() if k != "umax"})


if __name__ == "__main__":
    main()
