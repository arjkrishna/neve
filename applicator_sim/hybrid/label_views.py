"""Labelled stills of one state of a hybrid run, in the two orientations of the full-view MP4: the sagittal section
through the applicator seen from the patient's left, and the oblique 3-D view from the left-anterior-superior.
Every organ, the device parts and the anatomical landmarks are called out with leader lines: introitus, vaginal
lumen, fornices / vault, the vagina-HR-CTV junction, external and internal os, the intrauterine canal (the
labelled canal carried through the simulated deformation), fundus, tandem tip, flange -- and, in red, where the
model's contact fails (wall nodes inside the cervix, by signed distance).

    APPSIM_OUT=<render tree> py -3.11 hybrid/label_views.py --tag G32 [--step 180] [--size 1400]
    APPSIM_OUT=<render tree> py -3.11 hybrid/label_views.py --tag TF0 --steps end:V,d:10,d:20,d:35,end:C,final

--steps takes exported step numbers and/or selectors: "final" (the last frame), "end:<phase>" (the last frame of a
phase, e.g. end:V, end:C) and "d:<mm>" (the first frame whose tandem depth past the external os reaches mm;
animate_hybrid.frame_depth says where the depth comes from).  A step that was not exported (cfg frame_every) is an
error that names the nearest exported one.

Canal and os (fix plan S3, "landmark display"): by default the canal is the PHYSICIAN'S labelled canal, the part
s >= 0 of inputs/tandem_path.npz (from the external os O_true = the labelled canal's vaginal end, to the fundal
end), and the external / internal os are its O_true and i_internal_os.  Each point is carried by the tissue it
lies in at rest: barycentrically in the cervix tets (the frame's nodal displacement), rigidly with the corpus
(Kabsch fit of the frame's corpus surface: the corpus is rigid; an elastic corpus, cfg corpus_model "fem", is carried
barycentrically in its own tets when its nodes are exact).  The nodal displacement is EXACT wherever the run
saved it -- final/<body>_u.npy for the last frame, else the frame's own u_npy (index.json; TF0c saves the cervix's
every frame) -- if it reproduces the frame's surface to U_MATCH_MM; otherwise the interior is filled from the surface
by inverse distance (an approximation: TF0c step 192 lower canal in tube 0.76 that way, 0.619 exact, the value
tf_metrics reports).  State.X_src says which, per body.  The part more than 20 mm above the os is drawn in
orange: the upper canal the tandem does not follow (S0 upper_canal_residual; S6 reports it, the physician decides
Q2).  --canal legacy draws the pre-S3 canal.npz polyline (inverse-distance carried, os = its first point) and
reproduces the earlier figures pixel for pixel.

Device: the run's own parts (animate_hybrid.run_extra_parts).  A tandem-only run (applicator_v4, TF0) has no caps,
ovoid rods or packing: their labels and outlines are left out, and the "tandem rod" label sits on the curved
vaginal shaft (applicator.json landmarks.shaft_centreline) instead of the straight rod of params.angle_deg, anchored
on the shaft as drawn (shaft_anchor).  The sagittal cut keeps x >= x_cut; a stretch of the tube / shaft the cut
removes all or nearly all of, i.e. that lies in front of the plane (TF0c step 192: the curved shaft, 0-8 mm in
front, and the tube below the flange), is drawn whole, see-through, outlined, and its label says so
(animate_hybrid.front_cells; G32's cut passes through its tube and rod everywhere, so its figures are unchanged).
Tandem-first runs word the phase-dependent numbers accordingly: the cervix's largest displacement and the os's move
along the vagina, the vault lift signed, a move under 0.5 mm left out rather than printed as "0 mm".

The render tree must be the one built for the run (hybrid/render_tree.py: meshes = the run's scene root,
applicator = its applicator_dir, plus inputs/ and hybrid/eval/).  Every number printed on the figure is read or
computed from the run's files; nothing is hard-coded.  Writes figs/labeled/<tag>_step<k>_{sagittal,oblique}.png
(legacy canal: <...>_canalnpz.png) -- patient-derived, local only, never committed."""
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
import tandem_path as TP  # noqa: E402

P = AH.P
BODIES = AH.BODIES
OVOID_BODY = ("ovoid_L", "ovoid_R", "rod_L", "rod_R", "packing")
COL_CANAL = (0.0, 0.70, 0.85)
COL_RESID = (0.95, 0.45, 0.05)         # the labelled canal above LOWER_CANAL_MM: not followed by the tandem
COL_ROD = (0.45, 0.45, 0.50)
COL_PACK = (0.93, 0.88, 0.72)
COL_DEFECT = (0.85, 0.05, 0.05)
TXT_DEVICE = (0.20, 0.20, 0.24)
TXT_LANDMARK = (0.0, 0.0, 0.0)
TXT_DEFECT = (0.80, 0.05, 0.05)
LOWER_CANAL_MM = 20.0                  # fix plan S0: the lower canal the tandem must follow (canal_s 0..20 mm)
TUBE_TOL_MM = 1.0                      # S0: "in the tube" = within r_tube + 1 mm of the tube segment
# per-view opacity: the sagittal matches the MP4's cut panel; the oblique makes the wall / uterus / cervix
# see-through so that the applicator inside them can be labelled
OPA = dict(sagittal=dict(corpus=0.97, cervix=0.97, vagina=1.0, bladder=0.55, rectum=0.75, sigmoid=0.55),
           oblique=dict(corpus=0.33, cervix=0.35, vagina=0.18, bladder=0.10, rectum=0.40, sigmoid=0.22))
MIN_SEP_PX = 46.0                      # anchors of different labels are kept at least this far apart where possible
U_MATCH_MM = 0.005                     # a saved nodal displacement is used only if it reproduces the frame's surface to
                                       # this (tf_metrics.U_MATCH_MM: OBJs are written at 1 um, u_npy is float32)


# ------------------------------------------------------------------------------------------------ state
def idw(X0, u, Q, k=6):
    """Inverse-distance interpolation of the node displacements u (at X0) onto the points Q."""
    d = np.linalg.norm(X0[None, :, :] - Q[:, None, :], axis=2)
    nn = np.argsort(d, axis=1)[:, :k]
    w = 1.0 / np.maximum(np.take_along_axis(d, nn, 1), 1e-6) ** 2
    return (w[:, :, None] * u[nn]).sum(1) / w.sum(1)[:, None]


def tet_locate(X0, T, Q, tol=1e-9):
    """For each point of Q: the index of a rest tet containing it (-1 if none) and its 4 barycentric weights."""
    A = X0[T]                                                   # (m, 4, 3)
    M = np.transpose(A[:, 1:] - A[:, :1], (0, 2, 1))            # columns = edges from vertex 0
    ok = np.abs(np.linalg.det(M)) > 1e-12
    Mi = np.zeros_like(M)
    Mi[ok] = np.linalg.inv(M[ok])
    idx, W = np.full(len(Q), -1, int), np.zeros((len(Q), 4))
    for i, q in enumerate(np.atleast_2d(Q)):
        lam = np.einsum("mij,mj->mi", Mi, q - A[:, 0])
        w = np.c_[1.0 - lam.sum(1), lam]
        hit = np.nonzero(ok & (w >= -tol).all(1))[0]
        if len(hit):
            idx[i], W[i] = hit[0], w[hit[0]]
    return idx, W


def kabsch(A, B):
    """R, t with B ~ A @ R.T + t (least squares, proper rotation)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - ca @ R.T


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


def signed_mm(v, pos, neg, fmt="%.0f mm %s"):
    """'6 mm below the flange' / '3 mm above the flange': a signed number never printed as a negative distance."""
    return fmt % (abs(v), pos if v >= 0 else neg)


class State:
    """Bodies (surfaces + tet nodes), device parts and landmarks of one frame of a run, in preBT world mm.
    canal: "path" (tandem_path.npz, the physician's labelled canal; default) | "legacy" (canal.npz, pre-S3)."""

    def __init__(self, tag, step=None, canal="path"):
        self.tag = tag
        AH.run_extra_parts(tag)                 # the run's own parts (idempotent): rods / packing, or tandem only
        rd = AH.run_dir(tag)
        frs = AH.load_index(tag)["frames"]
        fr = frs[-1] if step is None else [f for f in frs if int(f["step"]) == int(step)][0]
        self.step, self.is_last, self.fr = int(fr["step"]), fr is frs[-1], fr
        fd = AH.frames_dir(tag)
        self.dev = AH.load_json(fd + "/" + fr["device"])
        self.cfg = AH.load_json(rd + "/cfg.json")
        self.tf = AH.is_tandem_first(tag)
        self.phase, self.phase_name = AH.phase_label(self.dev)
        self.surf = {b: tuple(np.asarray(x) for x in geom.read_obj("%s/%s" % (fd, fr["surfaces"][b]))) for b in BODIES}
        self.col = {b: tuple(AH.load_json(P["meshes"] + "/bodies.json")["bodies"][b]["color"]) for b in BODIES}
        self.meta = {b: AH.load_json("%s/%s/meta.json" % (P["meshes"], b)) for b in BODIES}
        self.tets = {b: VW.read_vtk_legacy("%s/%s/tets.vtk" % (P["meshes"], b)) for b in BODIES}
        self.X0 = {b: np.asarray(self.tets[b][0], float) for b in BODIES}
        self.X, self.X_src = {}, {}
        u_fr = fr.get("u_npy") or {}           # index.json: the frame's saved nodal displacements (frame_u_bodies)
        for b in BODIES:                       # node positions: EXACT where the run saved them, else from the surface
            s2n = np.asarray(self.meta[b]["surface_obj_vertex_to_tet_node"], int)
            cand = []
            if self.is_last:
                cand.append(("exact (final/%s_u.npy)" % b, "%s/final/%s_u.npy" % (rd, b)))
            if b in u_fr:
                cand.append(("exact (frame u_npy)", "%s/%s" % (fd, u_fr[b])))
            for src, fn in cand:
                if not os.path.exists(fn):
                    continue
                U = np.asarray(np.load(fn), float)
                if U.shape != self.X0[b].shape:
                    print("[State] %s: %s has shape %s, the tree's tets %s; not used"
                          % (b, fn, U.shape, self.X0[b].shape))
                    continue
                err = float(np.abs(self.X0[b][s2n] + U[s2n] - self.surf[b][0]).max())
                if err > U_MATCH_MM:           # a displacement that does not reproduce the frame's surface is not it
                    print("[State] %s: %s misses the frame surface by %.4f mm; not used" % (b, fn, err))
                    continue
                self.X[b], self.X_src[b] = self.X0[b] + U, src
                break
            else:
                self.X[b] = self.X0[b] + idw(self.X0[b][s2n], self.surf[b][0] - self.X0[b][s2n], self.X0[b])
                self.X_src[b] = "surface IDW"
        # the corpus's rigid motion: Kabsch of its nodes when they are exact, else of its SURFACE (exact for a rigid
        # body; the IDW-filled interior is not rigid), as tf_metrics fits it.  S9 (cfg corpus_model "fem"): the corpus
        # is elastic, so this is only its gross motion; carry() then moves corpus points barycentrically when the
        # corpus's nodes are exact (final/corpus_u.npy, or frame_u_bodies "corpus")
        self.corpus_fem = self.cfg.get("corpus_model", "rigid") == "fem"
        if self.X_src["corpus"].startswith("exact"):
            self.Rk, self.tk = kabsch(self.X0["corpus"], self.X["corpus"])
        else:
            s2k = np.asarray(self.meta["corpus"]["surface_obj_vertex_to_tet_node"], int)
            self.Rk, self.tk = kabsch(self.X0["corpus"][s2k], self.surf["corpus"][0])
        d = self.dev
        self.F, self.a, self.tip = (np.asarray(d["flange_mm"], float), geom.unit(d["tube_axis"]),
                                    np.asarray(d["tip_mm"], float))
        self.R_t = np.array([d["x_app"], d["y_app"], d["tube_axis"]], float)
        self.R_o = np.array([d["ovoid_x_app"], d["ovoid_y_app"], d["ovoid_axis"]], float) \
            if "ovoid_axis" in d else self.R_t
        self.Fo = np.asarray(AH.ovoid_origin(d), float)
        self.appj = AH.app_json()
        self.app = self.appj["params"]
        th = np.radians(float(self.app["angle_deg"]["value"]))
        self.d_rod_app, self.y_rod_app = np.array([0.0, np.sin(th), -np.cos(th)]), np.array([0.0, np.cos(th), np.sin(th)])
        lm = self.appj.get("landmarks", {})
        # applicator_v4: the vaginal tandem is a sweep along the BT label, not a straight rod at angle_deg (which is a
        # pose-rule parameter only there).  Its centreline from the flange, in the applicator frame.
        self.shaft_cl_app = np.vstack([[0.0, 0.0, 0.0], np.asarray(lm["shaft_centreline"], float)]) \
            if "shaft_centreline" in lm else None
        self.r_tube = float(d.get("r_tandem_mm") or self.app.get("r_tandem_mm", {}).get("value") or 2.18)
        self.parts, self.stations = {}, {}
        for p in list(AH.TANDEM) + list(AH.OVOIDS):
            V, Fc = geom.read_obj(AH.part_obj(p))
            org, R = (self.Fo, self.R_o) if p in OVOID_BODY else (self.F, self.R_t)
            self.parts[p] = (org + np.asarray(V, float) @ R, np.asarray(Fc, int))
            if p in AH.FRONT_PARTS:            # cross-section stations: the cut view draws a stretch in front whole
                self.stations[p] = AH.part_stations(p, V, self.appj)
        self.has_ring = "ovoid_L" in self.parts or "ovoid_R" in self.parts
        # the wall's rings
        w = self.meta["vagina"]["wall"]
        self.gi = np.asarray(w["grid_index"], int)
        self.n_ax, self.n_rd = int(w["n_axial"]), int(w["n_radial"])
        self.a_v = geom.unit(self.X0["vagina"][self.inner(self.n_ax - 1)].mean(0) - self.X0["vagina"][self.inner(0)].mean(0))
        y = np.array([0.0, 1.0, 0.0])
        self.e_ant = geom.unit(y - (y @ self.a_v) * self.a_v)      # anterior, in the wall's ring planes
        self.canal_src = canal
        if canal == "legacy":
            # the pre-insertion canal polyline carried by the tissue: each point moves with its own body
            cz = np.load(P["out"] + "/inputs/canal.npz")
            pts, self.i_ios = np.asarray(cz["pts"], float), int(cz["i_internal_os"])
            dc = np.linalg.norm(pts[:, None] - self.X0["cervix"][None], axis=2).min(1)
            dk = np.linalg.norm(pts[:, None] - self.X0["corpus"][None], axis=2).min(1)
            uc = idw(self.X0["cervix"], self.X["cervix"] - self.X0["cervix"], pts)
            uk = idw(self.X0["corpus"], self.X["corpus"] - self.X0["corpus"], pts)
            self.canal = pts + np.where((dk < dc)[:, None], uk, uc)
            self.canal_rest = pts
            self.canal_s, self.canal_by = None, None
        else:
            T = TP.load(P["out"] + "/inputs")
            i0 = T["i_os"]
            self.path = T
            self.canal_rest = np.asarray(T["pts"][i0:], float)
            self.canal_s = np.asarray(T["s"][i0:], float)
            self.i_ios = int(T["i_internal_os"]) - i0
            self.canal, self.canal_by = self.carry(self.canal_rest)
        self.depth, self.depth_src = AH.frame_depth(tag, d, fr)
        if self.canal_s is not None and (self.depth_src or "").startswith("measured"):
            # the same measured depth on THIS carry (O_true in the cervix tets), not the surface-IDW approximation
            self.depth = float((self.tip - self.canal[0]) @ self.a)
            self.depth_src = "measured (tip - carried O_true, along the tube)"

    def carry(self, Q):
        """Rest points Q moved with the tissue they lie in: inside a corpus tet -> the corpus's rigid motion (Kabsch of
        its nodes; the corpus is rigid); inside a cervix tet -> barycentric interpolation of the cervix nodal
        displacement; outside both -> the nearer body (corpus rigidly, cervix by inverse distance).  Returns the
        carried points and, per point, 'corpus' | 'cervix' | 'cervix~' (outside the tets).  An ELASTIC corpus (cfg
        corpus_model "fem", S9) with exact nodes: inside a corpus tet barycentrically ('corpus'), outside both but
        nearer the corpus by inverse distance ('corpus~'); without exact nodes it falls back to the Kabsch motion."""
        Q = np.atleast_2d(np.asarray(Q, float))
        Rk, tk = self.Rk, self.tk
        ik, Wk = tet_locate(self.X0["corpus"], self.tets["corpus"][1], Q)
        ic, Wc = tet_locate(self.X0["cervix"], self.tets["cervix"][1], Q)
        Uc = self.X["cervix"] - self.X0["cervix"]
        fem_k = bool(getattr(self, "corpus_fem", False)) and self.X_src["corpus"].startswith("exact")
        Uk = self.X["corpus"] - self.X0["corpus"]
        out, by = np.zeros_like(Q), []
        for i, q in enumerate(Q):
            if fem_k and ik[i] >= 0:
                out[i], b = q + Wk[i] @ Uk[self.tets["corpus"][1][ik[i]]], "corpus"
            elif fem_k and ic[i] < 0 and np.linalg.norm(self.X0["corpus"] - q, axis=1).min() \
                    < np.linalg.norm(self.X0["cervix"] - q, axis=1).min():
                out[i], b = q + idw(self.X0["corpus"], Uk, q[None])[0], "corpus~"
            elif ik[i] >= 0 or (ic[i] < 0 and np.linalg.norm(self.X0["corpus"] - q, axis=1).min()
                                < np.linalg.norm(self.X0["cervix"] - q, axis=1).min()):
                out[i], b = q @ Rk.T + tk, "corpus"
            elif ic[i] >= 0:
                out[i], b = q + Wc[i] @ Uc[self.tets["cervix"][1][ic[i]]], "cervix"
            else:
                out[i], b = q + idw(self.X0["cervix"], Uc, q[None])[0], "cervix~"
            by.append(b)
        return out, by

    def inner(self, k):
        return np.nonzero((self.gi[:, 0] == k) & (self.gi[:, 1] == 0))[0]

    def tube_dist(self, Q):
        q = np.atleast_2d(Q) - self.F
        return np.linalg.norm(q - np.outer(q @ self.a, self.a), axis=1)

    def seg_dist(self, Q):
        """Distance to the tube SEGMENT flange .. tip (S0 tf_metrics.seg_dist) and the axial coordinate."""
        L = float(self.dev["L_iu_mm"])
        q = np.atleast_2d(Q) - self.F
        h = q @ self.a
        return np.linalg.norm(q - np.outer(np.clip(h, 0.0, L), self.a), axis=1), h

    def shaft_samples(self, step_mm=1.0):
        """(arclength from the flange, world point) every step_mm along the applicator_v4 swept shaft centreline."""
        C = self.F + self.shaft_cl_app @ self.R_t
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
        ss = np.arange(0.0, s[-1] + 1e-9, step_mm)
        return ss, np.array([[np.interp(v, s, C[:, k]) for k in range(3)] for v in ss])

    def shaft_point(self, arc_mm):
        """The point of the tandem body's vaginal part at arclength arc_mm from the flange: on the straight rod
        (params.angle_deg) for v1-v3, on the swept centreline for applicator_v4."""
        if self.shaft_cl_app is None:
            return self.F + arc_mm * (self.d_rod_app @ self.R_t)
        C = self.F + self.shaft_cl_app @ self.R_t
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
        return np.array([np.interp(min(arc_mm, s[-1]), s, C[:, k]) for k in range(3)])

    def canal_metrics(self):
        """S0's canal-against-the-tube numbers on THIS carry (tf_metrics.py insertion block definitions): lower
        canal in tube (0 <= canal_s <= min(d, 20)), traversed in-span fraction, upper-canal residual (canal_s > 20).
        Path canal only."""
        if self.canal_s is None:
            return {}
        cs = self.canal_s
        dseg, h = self.seg_dist(self.canal)
        within = dseg <= self.r_tube + TUBE_TOL_MM
        d = self.depth
        L = float(self.dev["L_iu_mm"])
        lower = (cs >= 0) & (cs <= min(d if d is not None else LOWER_CANAL_MM, LOWER_CANAL_MM))
        inspan = (h >= 0) & (h <= L)
        up = cs > LOWER_CANAL_MM
        m = dict(dseg=dseg, h=h, within=within, n_lower=int(lower.sum()),
                 lower_in_tube_frac=float(within[lower].mean()) if lower.any() and (d is None or d > 0) else None,
                 lower_max_mm=float(dseg[(cs >= 0) & (cs <= LOWER_CANAL_MM)].max()),
                 traversed_in_tube_frac=float(within[inspan].mean()) if inspan.any() else None,
                 upper_residual_mm=float(dseg[up].max()) if up.any() else None,
                 upper_residual_s=float(cs[up][np.argmax(dseg[up])]) if up.any() else None)
        return m

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
        if self.shaft_cl_app is not None:                  # v4: the swept shaft's overall lean off the tube line
            ch = self.shaft_cl_app[-1] - self.shaft_cl_app[1]
            f["shaft_lean"] = float(np.degrees(np.arccos(np.clip(geom.unit(ch) @ np.array([0.0, 0.0, -1.0]), -1, 1))))
        if self.is_last and not self.corpus_fem:
            f["corpus_rot"] = float(AH.load_json(P["applicator"] + "/pose.json")["corpus"]["rotation_deg"])
        else:                                              # this frame's corpus rotation (it turns during insertion; an
            f["corpus_rot"] = float(np.degrees(np.arccos(np.clip(0.5 * (np.trace(self.Rk) - 1.0), -1.0, 1.0))))
        if self.corpus_fem:                                # elastic corpus: its gross rotation, the Kabsch fit)
            e = np.linalg.norm(self.X0["corpus"] @ self.Rk.T + self.tk - self.X["corpus"], axis=1)
            f["corpus_nonrigid"] = float(e.max())          # how far it is from rigid (nodes; approximate if not exact)
        if "packing" in self.parts:
            q = self.parts["packing"][0] - self.Fo
            dr = self.d_rod_app @ self.R_o
            f["pack_r"] = float(np.percentile(np.linalg.norm(q - np.outer(q @ dr, dr), axis=1), 90))
        ev = "%s/eval/%s/metrics.json" % (P["hybrid"], self.tag)
        if os.path.exists(ev) and self.is_last:            # the evaluator scores the final state only
            f["tip_to_serosa"] = float(AH.load_json(ev)["frames"]["E_app"]["landing"]["tip_to_serosa_mm"])
        # the junction by SIGNED distance of the lumen nodes to the cervix (< 0 = inside the cervix)
        sdf = implicit_distance(*self.surf["cervix"])
        self.sdf_cervix = sdf
        inn = np.nonzero(self.gi[:, 1] == 0)[0]
        sd = sdf(Xw[inn])
        self.junction = inn[(sd >= 0.0) & (sd < 1.5)]
        self.pen, self.pen_depth = inn[sd < 0.0], -sd[sd < 0.0]
        f["rim_contact_pct"] = float(100.0 * np.mean(sdf(Xw[top]) < 1.5))
        f["pen_n"] = int(len(self.pen))
        f["pen_max"] = float(self.pen_depth.max()) if len(self.pen) else 0.0
        f["umax"] = {b: float(v["umax_mm"]) for b, v in self.dev.get("disp", {}).items()}
        # where the tip is, by signed distance (the tube's label must not say "in the uterine cavity" during V)
        sdk = implicit_distance(*self.surf["corpus"])(self.tip[None])[0]
        f["tip_in"] = "corpus" if sdk < 0 else ("cervix" if sdf(self.tip[None])[0] < 0 else "vagina")
        cm = self.canal_metrics()
        for k in ("lower_in_tube_frac", "lower_max_mm", "traversed_in_tube_frac", "upper_residual_mm",
                  "upper_residual_s"):
            if cm.get(k) is not None:
                f[k] = cm[k]
        if self.depth is not None:
            f["depth"] = float(self.depth)
        return f


def state_words(st):
    """'end of insertion, after settling' (the seated state of a Stage 1-3 run, as before) or the frame's phase and
    tandem depth."""
    if st.is_last and not st.tf:
        return "end of insertion, after settling"
    w = "phase %s, %s" % (st.phase, st.phase_name)
    if st.is_last:
        w = "final state, " + w
    if st.depth is not None:
        w += "; tandem depth d = %.1f mm past the external os" % st.depth
    return w


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


def shaft_anchor(st, proj, taken):
    """Anchor of the 'tandem rod' label on the applicator_v4 swept shaft (below its start, where it leaves the tube):
    a centreline point inside the vagina (between the introitus and vault ring centres of this frame) at the middle
    of that stretch, clear of the anchors already placed (pick_clear); 10 mm below the shaft's start when none of it
    is inside (early V).  Every centreline point is
    on the shaft as drawn: where the cut keeps part of a cross-section, that part's band covers its centre; where it
    keeps (nearly) none, the stretch is drawn whole (render: AH.front_cells).  Before this, the anchor was the point
    45 mm down the centreline, which at TF0c step 192 lies below the introitus, in empty space once the cut had
    removed the shaft."""
    s, Q = st.shaft_samples()
    s0 = 0.0                             # where the shaft leaves the tube (landmarks.shaft_start; v4: 21 mm below the
    lm = st.appj.get("landmarks", {})    # flange, the tube's own lower end overlapping the centreline above it)
    if "shaft_start" in lm:
        q0 = st.F + np.asarray(lm["shaft_start"], float) @ st.R_t
        s0 = float(s[np.argmin(np.linalg.norm(Q - q0, axis=1))])
    keep = s >= s0 + 3.0                                       # clear of the flange's / the tube's anchors
    if keep.any():
        s, Q = s[keep], Q[keep]
    c0 = st.X["vagina"][st.inner(0)].mean(0)
    c1 = st.X["vagina"][st.inner(st.n_ax - 1)].mean(0)
    ax = geom.unit(c1 - c0)
    h = (Q - c0) @ ax
    m = (h > 0.0) & (h < float((c1 - c0) @ ax))
    if m.any():
        C, sc = Q[m], s[m]
        mid = 0.5 * (sc.min() + sc.max())
    else:                                # the shaft is all outside (early V): just below its start, near the introitus
        C, sc = Q, s
        mid = sc.min() + min(10.0, 0.5 * (sc.max() - sc.min()))
    return pick_clear(C, proj, taken, lambda q_, p_: abs(float(sc[np.argmin(np.linalg.norm(C - q_, axis=1))]) - mid))


def signed_word(v, fmt="%+.0f"):
    """'+3' / '−7' (a true minus sign)."""
    return (fmt % v).replace("-", "−")


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


def add_canal(pl, pv, st):
    """The carried canal as a tube: one colour (legacy), or the followed lower canal (canal_s <= 20 mm) in cyan and
    the upper canal in orange (path)."""
    if st.canal_s is None:
        pl.add_mesh(pv.Spline(st.canal, 300).tube(radius=0.9), color=COL_CANAL, smooth_shading=True)
        return
    lo = st.canal_s <= LOWER_CANAL_MM
    k = int(np.nonzero(lo)[0].max())
    for sel, col in ((slice(0, k + 1), COL_CANAL), (slice(k, None), COL_RESID)):
        C = st.canal[sel]
        if len(C) >= 2:
            pl.add_mesh(pv.Spline(C, 8 * len(C)).tube(radius=0.9), color=col, smooth_shading=True)


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
    front, front_all = {}, {}                   # tandem stretches drawn whole in front of the section: part -> mm
    for p, (V, Fc) in st.parts.items():
        if cut and p in st.stations:            # a stretch the cut removes all or nearly all of
            fm, fmm = AH.front_cells(st.stations[p], V, Fc, x_cut)
            if fm.any():
                front[p], front_all[p] = fmm, bool(fm.all())
                pl.add_mesh(AH.poly(pv, V, Fc[fm]), color=AH.COL_TANDEM, opacity=AH.FRONT_OPACITY,
                            smooth_shading=True, silhouette=dict(color=AH.COL_FRONT_EDGE, line_width=1.6))
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
    add_canal(pl, pv, st)
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
            if pname in st.parts:                            # (a tandem-only run has none)
                outlines.append(hull2d([proj(q) for q in st.parts[pname][0]]))
    info = dict(x_cut=x_cut, cam=cam, clipped=clipped, proj=proj, outlines=outlines, front=front, front_all=front_all,
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
    path = st.canal_s is not None

    def add(name, desc, cat, pts, color=None, side=None):
        pts = [np.asarray(q, float) for q in pts if q is not None]
        if pts:
            L.append(dict(name=name, desc=desc, cat=cat, pts=pts, color=color, side=side))
            taken.extend(proj(q) for q in pts)

    um = f["umax"]
    front = info.get("front") or {}

    def front_note(p):
        """A stretch the section would remove is drawn whole in front of it (render: AH.front_cells)."""
        if p not in front:
            return ""
        return ("\nlies in front of the section plane (up to %.0f mm):\ndrawn whole, see-through" % front[p]) \
            if (info.get("front_all") or {}).get(p) else \
            ("\nsee-through where it lies in front of the\nsection plane (up to %.0f mm)" % front[p])
    # ---- fixed: device axis points and canal landmarks
    Lt = float(st.dev["L_iu_mm"])
    add("tandem tip", ("%.1f mm from the fundal serosa" % f["tip_to_serosa"]) if "tip_to_serosa" in f else "",
        "device", [st.tip])
    add("tandem (intrauterine tube)", dict(corpus="in the uterine cavity", cervix="tip in the cervical canal",
                                           vagina="tip not yet in the cervix")[f["tip_in"]] + front_note("tube"),
        "device", [st.F + 0.55 * Lt * st.a])
    if st.has_ring:
        fl = "tandem bends %.1f° here; ring plane" % f["angle"]
    elif "shaft_lean" in f:
        fl = "no ring in this run; the vaginal tandem\nleans %.0f° off the tube line below it" % f["shaft_lean"]
    else:
        fl = "no ring in this run"
    add("flange", fl, "device", [st.F], side=None if cut else "L")
    if path:
        add("external os", "labelled canal's vaginal end (O_true);\n%s"
            % signed_mm(f["os_below_flange"], "below the flange", "above the flange"), "landmark", [st.canal[0]],
            side=None if cut else "R")
        add("internal os", "top of the cervical canal,\n%.0f mm above the external os" % st.canal_s[st.i_ios],
            "landmark", [st.canal[st.i_ios]])
        cm_lo = f.get("lower_max_mm", float("nan"))
        up = f.get("upper_residual_mm")
        if st.tf:                        # tandem-first: the S0 numbers of this frame, the residual named as such
            lines = ["physician's labelled canal, moved with the tissue",
                     "lower %.0f mm (cyan, followed): up to %.1f mm off the tube" % (LOWER_CANAL_MM, cm_lo)]
            if f.get("lower_in_tube_frac") is not None:
                lines.append("%.0f %% of s 0–%.0f mm within %.1f mm of the tube" % (
                    100.0 * f["lower_in_tube_frac"], min(f.get("depth", LOWER_CANAL_MM), LOWER_CANAL_MM),
                    st.r_tube + TUBE_TOL_MM))
            if up is not None:
                lines += ["upper-canal residual (orange, s > %.0f mm," % LOWER_CANAL_MM,
                          "not followed): %.1f mm off the tube at s = %.0f mm" % (up, f["upper_residual_s"])]
            cdesc = "\n".join(lines)
        else:
            cdesc = "physician's labelled canal, moved with the tissue:\n≤ %.1f mm from the tube in its lower " \
                    "%.0f mm;\n%s" % (cm_lo, LOWER_CANAL_MM, ("orange above: up to %.0f mm off the tube\n(upper canal, "
                                                             "not followed)" % up) if up is not None else "")
        add("intrauterine canal", cdesc, "landmark", [st.canal[(st.i_ios + len(st.canal)) // 2]], color=COL_CANAL)
    else:
        add("external os", signed_mm(f["os_below_flange"], "below the flange", "above the flange"), "landmark",
            [st.canal[0]], side=None if cut else "R")
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
    if st.shaft_cl_app is None:          # v1-v3: the straight rod, 45 mm from the flange (on the kept side in G32)
        q_rod = st.shaft_point(45.0)
    else:                                # v4: on the swept shaft as drawn, clear of the lumen label's anchor
        q_rod = shaft_anchor(st, proj, taken + ([proj(st.X["vagina"][st.inner(st.n_ax // 2)].mean(0))] if cut else []))
    add("tandem rod", ("straight, along the vagina" if st.shaft_cl_app is None else
                       "curved: swept along the BT label's\nvaginal tandem") + front_note("shaft"), "device", [q_rod])
    if "rod_L" in st.parts:
        offs = st.app["ovoid_rod_offsets_mm"]["value"]
        rq = [st.Fo + (ox * np.array([1.0, 0, 0]) + oy * st.y_rod_app + 40.0 * st.d_rod_app) @ st.R_o for ox, oy in offs]
        if cut:
            rq = [p_ for p_ in rq if p_[0] >= x_cut]
        add("ovoid rod" + ("" if cut else "s"), "%.0f mm anterior of the tandem rod" % offs[0][1], "device", rq)
    # ---- organs
    # the cervix: how far the os moved along the vagina (a rounded "0 mm" is not a claim worth printing), and, in a
    # tandem-first run (where the os stays put through V and C and is drawn up in L), its largest displacement
    ol = f["os_lift"]
    moved = abs(ol) >= 0.5
    if st.tf:
        cx = (["displaced up to %.0f mm" % um["cervix"]] if "cervix" in um else []) + \
            ([("os drawn up %.0f mm along the vagina" % ol) if ol > 0 else
              ("os pushed %.0f mm down the vagina" % -ol)] if moved else [])
        cx = ";\n".join(cx)
    elif moved:
        cx = ("drawn up %.0f mm onto the tandem" % ol) if ol >= 0 else ("os pushed %.0f mm down the vagina" % -ol)
    else:
        cx = "displaced up to %.0f mm" % um.get("cervix", float("nan"))
    organ_desc = dict(corpus=("rigid; rotated %.0f° with the tandem" % f["corpus_rot"]) if not st.corpus_fem else
                      ("elastic; rotated %.0f° with the tandem,\n%.1f mm off rigid at most" % (f["corpus_rot"],
                                                                                          f.get("corpus_nonrigid", 0.0))),
                      cervix=cx,
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
    vl = f["vault_lift"]                 # the vault rim's move along the vagina's rest axis (+ = up, toward the uterus)
    if st.tf:                            # the wall's own thickness range (meta), the lift signed, omitted when < 0.5 mm
        ts = (st.meta["vagina"]["wall"].get("thickness_stats") or {})
        wd = ("wall %.1f–%.1f mm thick" % (ts["min"], ts["max"])) if "min" in ts and "max" in ts else "vaginal wall"
        if abs(vl) >= 0.5:
            wd += ";\nvault lift %s mm (%s)" % (signed_word(vl), "up, toward the uterus" if vl > 0 else "pushed down")
    else:
        wd = "1.2 mm wall" + (("; vault %s %.0f mm" % ("lifted" if vl >= 0 else "lowered", abs(vl)))
                              if abs(vl) >= 0.5 else "")
    add("vaginal wall", wd, "organ", [qw], color=st.col["vagina"])
    if "packing" in st.parts:
        pr = f.get("pack_r", 17.0)
        qp = st.Fo + (30.0 * st.d_rod_app - pr * st.y_rod_app) @ st.R_o if cut else \
            front_point(st.parts["packing"][0], cam, st.Fo + (30.0 * st.d_rod_app) @ st.R_o)
        add("packing (model)", "%.0f mm cylinder around the rods" % pr, "device", [qp])
    # ---- introitus, lumen, vault, junction, caps, defect
    if cut:
        r0 = ring_cut_points(st, 0, x_cut)
        add("introitus", "vaginal opening", "landmark", [r0.get("ant"), r0.get("post")])
        add("vaginal lumen", "held open by the applicator" if st.has_ring else "around the vaginal tandem",
            "landmark", [st.X["vagina"][st.inner(st.n_ax // 2)].mean(0)])
        rt = ring_cut_points(st, st.n_ax - 1, x_cut)
        if st.canal_src == "legacy":                     # the pre-S6c figure, word for word (a visual reading)
            add("vagina–HR-CTV junction", "anterior fornix; the only contact\nin this plane (the rest is lateral)",
                "landmark", [rt.get("ant")])
            add("posterior fornix", "vault rim, open in this plane", "landmark", [rt.get("post")])
        else:
            # the vault rim's gap to the cervix IN THIS PLANE, measured (signed distance of the cut lumen node), not
            # read off the picture.  MEASURED on G32 step 180: anterior 1.7 mm (the cervix is on the lumen side, the
            # outer node is 3.2 mm off), posterior 15.1 mm -- the earlier "the only contact" was 0.2 mm over the
            # 1.5 mm rule the junction nodes use.  Within the contact alarm distance (cfg alarm_mm) = "junction".
            alarm = float(st.cfg.get("alarm_mm") or 3.0)
            gap = {k: float(st.sdf_cervix(rt[k][None])[0]) for k in ("ant", "post") if rt.get(k) is not None}
            word = lambda g: ("%.1f mm inside the cervix" % -g) if g < 0 else ("%.1f mm from the cervix" % g)  # noqa
            if "ant" in gap:
                add("vagina–HR-CTV junction" if gap["ant"] < alarm else "anterior fornix",
                    "anterior fornix, %s\nin this plane" % word(gap["ant"]), "landmark", [rt.get("ant")])
            if "post" in gap:
                add("posterior fornix", "vault rim, %s\nin this plane" % word(gap["post"]), "landmark",
                    [rt.get("post")])
        if st.has_ring:
            oc = [np.asarray(q_, float) for q_ in AH.ovoid_centres(st.dev)]
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
        if abs(vl) < 0.5:
            vd = ""
        elif st.tf:
            vd = "lift %s mm along the vagina" % signed_word(vl)
        else:
            vd = "%s %.0f mm with the cervix" % ("lifted" if vl >= 0 else "lowered", abs(vl))
        add("vaginal vault (rim)", vd, "landmark", [pick_clear(Qt, proj, taken, lambda q_, p_: p_[1])])
        caps = []
        for pname in ("ovoid_L", "ovoid_R"):
            if pname not in st.parts:
                continue
            Vc = st.parts[pname][0]
            caps.append(pick_clear(Vc, proj, taken + [proj(c_) for c_ in caps],
                                   lambda q_, p_: float(np.linalg.norm(q_ - cam))))
        if caps:
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
    what = "Seated applicator" if (st.is_last and st.has_ring and not st.tf) else \
        ("Tandem only" if not st.has_ring else "Insertion state")
    ttl = dict(sagittal="%s: sagittal section through the device" % what,
               oblique="%s: oblique 3-D view" % what)[view]
    sub = dict(sagittal="Cut through the applicator's sagittal plane and viewed from the patient's left; the near "
                        "(left) half is removed.  Anterior is to the left, superior is up.",
               oblique="Viewed from the patient's left, anterior and superior; nothing cut.  All tissue is drawn "
                       "see-through to show the applicator inside.")[view]
    ax.text(W / 2, 40, ttl, ha="center", va="center", fontsize=21, fontweight="bold")
    ax.text(W / 2, 80, "run %s, step %d (%s).  %s" % (st.tag, st.step, state_words(st), sub),
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
    if st.canal_s is None:
        ax.plot([x, x + 34], [ky, ky], color=COL_CANAL, lw=5, solid_capstyle="round")
        ax.text(x + 44, ky, "intrauterine canal (pre-insertion, moved with the tissue)", ha="left", va="center",
                fontsize=12)
    else:
        ax.plot([x, x + 20], [ky, ky], color=COL_CANAL, lw=5, solid_capstyle="round")
        ax.plot([x + 22, x + 42], [ky, ky], color=COL_RESID, lw=5, solid_capstyle="round")
        ax.text(x + 52, ky, "labelled canal, moved with the tissue (orange: > %.0f mm above the os)" % LOWER_CANAL_MM,
                ha="left", va="center", fontsize=12)
        x += 90
    if info.get("outlines"):
        x += 560
        ax.plot([x, x + 34], [ky, ky], color=(0.15, 0.15, 0.18), lw=1.7, ls=(0, (6, 3)))
        ax.text(x + 44, ky, "ovoid caps seen through tissue", ha="left", va="center", fontsize=12)
    foot = "Tandem and tandem rod black; ovoid caps light grey; ovoid rods mid grey; packing pale." if st.has_ring else \
        "Tandem (intrauterine tube and vaginal shaft) black; no ring, ovoid rods or packing in this run."
    if info.get("front"):                # tandem stretches the cut would remove (render: AH.front_cells)
        foot += "  Where the %s %s in front of the section plane, %s drawn whole, see-through, outlined." % (
            " and ".join(dict(tube="tube", shaft="vaginal shaft")[p] for p in info["front"]),
            *(("lies", "it is") if len(info["front"]) == 1 else ("lie", "they are")))
    ax.text(W / 2, ky + 48, foot + "  Patient-derived figure: keep local.", ha="center", va="center", fontsize=11,
            color=(0.35, 0.35, 0.38))
    fig.savefig(out, dpi=100)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ steps
def select_steps(tag, spec):
    """Resolve --steps: exported step numbers and the selectors final | end:<phase> | d:<mm> (module docstring)."""
    frs = AH.load_index(tag)["frames"]
    steps = [int(f["step"]) for f in frs]
    out = []
    for tok in [t.strip() for t in str(spec).split(",") if t.strip()]:
        if tok in ("final", "last"):
            out.append(steps[-1])
        elif tok.startswith("end:"):
            cand = [int(f["step"]) for f in frs if f["phase"] == tok[4:]]
            if not cand:
                raise SystemExit("--steps %s: no exported frame in phase %s (phases: %s)"
                                 % (tok, tok[4:], sorted({f["phase"] for f in frs})))
            out.append(cand[-1])
        elif tok.startswith("d:"):
            want, hit = float(tok[2:]), None
            for f in frs:
                if f["phase"] in ("B", "P"):
                    continue
                dev = AH.load_json(AH.frames_dir(tag) + "/" + f["device"])
                d, src = AH.frame_depth(tag, dev, f)
                if d is not None and d >= want:
                    hit = int(f["step"])
                    print("--steps %s -> step %d (d = %.1f mm, %s)" % (tok, hit, d, src))
                    break
            if hit is None:
                raise SystemExit("--steps %s: no exported frame reaches d = %g mm" % (tok, want))
            out.append(hit)
        else:
            k = int(tok)
            if k not in steps:
                near = min(steps, key=lambda s_: abs(s_ - k))
                raise SystemExit("--steps: step %d was not exported (frames every %s steps); nearest exported: %d"
                                 % (k, AH.load_index(tag).get("frame_every"), near))
            out.append(k)
    return list(dict.fromkeys(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--step", type=int, default=None, help="frame step (default: the last frame)")
    ap.add_argument("--steps", default=None, help="several frames: step numbers and/or final, end:<phase>, d:<mm>")
    ap.add_argument("--size", type=int, default=1400, help="render height in px")
    ap.add_argument("--views", default="sagittal,oblique")
    ap.add_argument("--canal", choices=["path", "legacy"], default="path",
                    help="path: the physician's labelled canal (tandem_path.npz, default); legacy: canal.npz")
    a = ap.parse_args()
    AH.run_extra_parts(a.tag)
    steps = select_steps(a.tag, a.steps) if a.steps else [a.step]
    od = P["figs"] + "/labeled"
    os.makedirs(od, exist_ok=True)
    for step in steps:
        st = State(a.tag, step, canal=a.canal)
        st.fx = st.facts()
        for view in a.views.split(","):
            Sh = a.size if view == "sagittal" else int(a.size * 1.15)
            Sw = a.size if view == "sagittal" else int(a.size * 0.93)
            img, lab, info, _ = render(st, view, Sw, Sh)
            out = "%s/%s_step%04d_%s%s.png" % (od, a.tag, st.step, view, "_canalnpz" if a.canal == "legacy" else "")
            compose(img, lab, info, view, st, out)
            print("wrote", out, "(%d labels)" % len(lab))
        print("facts:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in st.fx.items() if k != "umax"})


if __name__ == "__main__":
    main()
