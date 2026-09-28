"""S0 of the G32 fix plan (HOST, python 3.13): the measurements every later acceptance test uses, for one hybrid run,
per saved frame and in the final state.  Units mm, cc, deg.

    python -P hybrid/tf_metrics.py score --tag G32 [--every k] [--steps 60 180] [--no-final]
        -> eval/<tag>/tf_metrics.json ; for G32 it also checks the S0 baselines (printed and stored under "baseline")
    python -P hybrid/tf_metrics.py score --tag X --run-dir <folder> --out <json>     # a run folder elsewhere

Runtime: G32 (61 saved frames + the final tables) ~30 s; --every k scores every k-th saved frame and the last.

WHY.  eval_hybrid scores the final state against BT.  It cannot see the insertion order, the tip's path through the
canal, or bodies passing through each other (lost contact).  The fix plan's audit measured all three for G32 with ~20
one-off scripts (scratchpad wf_verify_*/, crit_clin/); two implementations of the same quantity differed by up to
2.7 mm (fornix depth) and 637 vs 463 mm^2 (contact area), so the definitions are fixed here, in code, once.  The
module must score G32 now and the tandem-first runs next (TF0: applicator_v4 with tube + shaft only, no ovoid body,
phases B, V, C, L, H).  Nothing is simulated: it reads runs/<tag>/{cfg.json, device_final.json, log.jsonl, frames/,
final/}, the run's applicator dir, the meshes the run used, inputs/tandem_path.npz (S3) and the labels.

FRAMES.  Model = preBT world RAS mm (x = patient right, y = anterior, z = superior).  PELVIS = the BT labels mapped into
the model frame with validation/alignment.json frames.BONE (y_pre = R x_BT + t).  Own-flange frame = rows (x_app,
y_app, tube_axis) about the run's flange.  BT label frame = the updated applicator label's tandem: the line through
the 2 mm slab centroids of (updated applicator minus ovoid) over s 15-55 mm above the json flange (voxels within 6 mm
of the json axis), origin = the json flange projected onto it, x = the json x orthogonalised.  Every block states its
frame.

INSIDE TESTS.  Signed distance (vtkImplicitPolyDataDistance on consistently oriented normals, < 0 inside) is the
primary test; every count is cross-checked by vtkSelectEnclosedPoints and the disagreements are reported.  A wall
sheet is closed by a fan disc over each of its boundary loops (the audit's vvm/v9_sections.close_sheet), which
reproduces the G32 step-180 organ counts 59 / 89 (outer) and 44 / 70 (lumen).  CONTAINMENT is the exception: it is
measured against the OPEN inner sheet (the lumen wall's own triangles, no fan discs; normals made consistent and
oriented so that the lumen-ring centres read negative), because the discs are not tissue: a shaft vertex just above
the introitus plane lies a few um outside the introitus disc (TF0c: "clearance" -0.008 mm to the closed lumen, 1.489
mm to the wall).  The closed-lumen reading is reported beside it (containment.closed_sheet).

CARRIED CANAL.  The labelled canal (tandem_path.npz pts with s >= 0) and the planned path tau are carried with the
tissue per frame: points inside a rest cervix tet by barycentric interpolation of the frame's cervix nodal
displacement.  That displacement is EXACT whenever the run saved it: frames/step_<k>_cervix_u.npy (index.json u_npy,
cfg frame_u_bodies), or final/cervix_u.npy for the last saved frame; either is used only if it reproduces the frame's
cervix surface to U_MATCH_MM.  Otherwise (runs whose frames hold surfaces only, e.g. G32) the interior is the harmonic
extension (edge weights 1/length) of the frame's surface displacement (the audit's vcm/v4_frames.py; its error against
final/cervix_u.npy is reported under carry_validation).  Every frame records its source under "carry".  Points in the
corpus (and canal points outside both meshes) by the Kabsch fit of the frame's corpus surface (the corpus is rigid);
vaginal-side points outside the tissue stay at rest.  ELASTIC CORPUS (cfg corpus_model "fem", S9): points inside a rest
corpus tet are carried barycentrically ("k") by the frame's corpus nodal displacement -- frames/step_<k>_corpus_u.npy
(cfg frame_u_bodies), else final/corpus_u.npy for the last saved frame, each only if it reproduces the frame's corpus
surface to U_MATCH_MM, else the Kabsch fit of the frame's corpus surface flagged approximate (carry.corpus); points
outside both meshes still take the Kabsch fit, and tip_in_corpus_rest the Kabsch fit of the serosa (surface minus the
cervix interface).  A rigid run is carried exactly as before.  Every frame and the final state carry a "corpus" block
(pose vs the schedule's target, non-rigid part, volume, span along the tube, carried canal by band, best-line max).

TANDEM-FIRST PLAN (tip_to_plan).  p(d) = carried tau(d) + w_r(d) delta.  tau and delta come from the run's
<applicator_dir>/pose.json "insertion_path_tandem_first" (S5); tau defaults to the S3 path: the labelled canal for
TAU_FOLLOW_MM, then straight along a_lc.  The frame's planned depth d and weight w_r are read from the frame's device
json (keys tip_s / w_r), else from the record's rows by step, else tip_to_plan is null.  S5 keys the rows by 'i' (no
step): the scene runs them after the B and P phases, so step = i + n_balloon + n_presettle; that offset is checked
against the log's canal_tie.tip_s (the scene logs the row's tip_s) and re-fitted or refused when it disagrees (see
Run._tf_rows).  G32 has no record: null.

Patient-derived output: local only."""
import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True                      # never leave __pycache__ in the repo
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402
import scipy.sparse as spa  # noqa: E402
import scipy.sparse.linalg as spl  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
import vtk  # noqa: E402
from vtk.util import numpy_support as ns  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import geom  # noqa: E402
import evaluate as ev  # noqa: E402
import eval_hybrid as EH  # noqa: E402
import tandem_path as TP  # noqa: E402
from vagina_wall import read_vtk_legacy  # noqa: E402

VERSION = "tf_metrics/1.1"      # 1.1: exact per-frame carry (u_npy), containment on the open lumen sheet, rows by 'i'
HYB, RUNS, EVALD, DATA, INP = EH.HYB, EH.RUNS, EH.EVALD, EH.DATA, EH.INP
# the UPDATED applicator label (2026-09-17) sits beside the read-only data folder, not in it
APP_LABEL_NEW = os.environ.get("APPSIM_APP_LABEL", os.path.dirname(DATA) + "/BT_MRI_label_applicator.nii")
NEW_GEOM_JSON = HYB + "/logs/applicator_label_new_geometry.json"   # updated-label tandem axis + tip (audit v15)

ORGANS = ("bladder", "rectum", "sigmoid")
OVOID_BODY = EH.OVOID_BODY                          # parts placed by the ovoid body's own origin / frame
HALVES = dict(L=("ovoid_L", "rod_L"), R=("ovoid_R", "rod_R"))
# scene_hybrid.CFG defaults for the keys read here (a run's cfg.json normally holds every key; these cover the rest)
SCENE_DEFAULTS = dict(apex_attach="follow", apex_lift_fix=True, apex_lift_profile="top", vagina_inferior="fixed",
                      canal_tie_set="canal", n_balloon=0, n_presettle=4)
TUBE_TOL_MM = 1.0             # canal-in-tube: within r_tube + 1 mm of the tube (plan S0)
LOWER_CANAL_MM = 20.0         # lower_canal_in_tube_frac: canal_s 0 .. min(d, 20)
LOWER_GATE_MM = 5.0           # the S5 / S6 lower-canal gate reads the C frames AFTER d = 5 mm (d > 5, strictly)
U_MATCH_MM = 0.005            # a saved nodal displacement is used only if it reproduces the frame's cervix surface to
#                               this (frame OBJs are written at 1 um: rounding <= 0.0005 mm; u_npy is float32)
ROW_TIP_S_TOL_MM = 0.01       # record rows vs the log's canal_tie.tip_s (logged at 1e-3 mm; the next row is ~1 mm off)
TAU_FOLLOW_MM = 20.0          # tau follows the labelled canal this far, then a_lc (plan S5)
DEEP_MM = 0.5                 # the S2 probes' gate: vertices more than 0.5 mm inside
REMOTE_MM = 5.0               # OAR vertices farther than this from every device part and from the wall's outer sheet
CLEAR_MM = 1.0                # containment clearance the S5 host gate asks for
LABEL_LINE = dict(s=(15.0, 55.0), slab=2.0, lat=6.0)     # BT label frame (see FRAMES)
VAG_GRID = dict(h=(-80.0, 20.0), dh=1.0, uv=36.0, duv=0.75)
# vagina regions by height h above the run's flange along the shaft axis.  The ring band is the 22 mm slab that holds
# 93 % of the BT ovoid label (crit_clin/c1: z_app -21.5..+0.5); the lower third is below h -50.
VAG_REGIONS = (("above_ring", 0.5, 20.0), ("ring", -21.5, 0.5), ("mid", -50.0, -21.5), ("lower_third", -80.0, -50.0))
RECTUM_BANDS = (("z<-30", -1e9, -30.0), ("-30..0", -30.0, 0.0), ("z>0", 0.0, 1e9))
# fixed references the audit measured once (not recomputed here; quoted with their source so a table can carry them)
REFERENCE = dict(
    cervix_bounds=dict(volume_ceiling=0.914, best_rigid_min=0.754, canal_on_tandem_rigid=[0.59, 0.62],
                       canal_on_tandem_flipped_roll=[0.634, 0.642],
                       source="fix plan R6 / section 3 (vjm/v04_dice.py, v10_dice_roll.py)"),
    oracle_ceilings=dict(rectum_slice_shift=0.656, bladder_rigid=0.915, sigmoid_rigid=0.05,
                         source="fix plan S0 (vom/v6_oars.py)"))


# ============================================================================================ small helpers
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rnd(x, n=3):
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return [rnd(v, n) for v in x]
    x = float(x)
    return round(x, n) if np.isfinite(x) else None


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(float(unit(a) @ unit(b)), -1.0, 1.0))))


def first(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


def seg_dist(P, F, a, L):
    """Distance of P to the tube segment F .. F + L a, and the axial coordinate of each point."""
    q = np.atleast_2d(P) - F
    h = q @ a
    return np.linalg.norm(q - np.outer(np.clip(h, 0.0, L), a), axis=1), h


def frac(m):
    m = np.asarray(m, bool)
    return round(float(m.mean()), 4) if m.size else None


def polydata(V, F):
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(V, float), deep=1))
    F = np.asarray(F, np.int64)
    ca = vtk.vtkCellArray()
    ca.SetData(ns.numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(F) + 1, 3, dtype=np.int64), deep=1),
               ns.numpy_to_vtkIdTypeArray(np.ascontiguousarray(F.ravel()), deep=1))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(ca)
    return pd


class Surf:
    """A closed triangulated surface: signed distance (primary) and the enclosed-points test (cross-check).
    Points farther than `margin` outside the bounding box are not evaluated (sd = +inf, enclosed = False).

    closed=False: an OPEN sheet.  Its normals are made consistent but not auto-oriented (vtkPolyDataNormals
    AutoOrientNormals needs a closed surface); orient(P) then fixes the sign so that the reference points P read
    negative.  The sign is meaningful only near the sheet, away from its open boundary (vtkImplicitPolyDataDistance
    takes it from the normal at the closest point); there is no enclosed-points test."""

    def __init__(self, V, F, name="", closed=True):
        self.V = np.asarray(V, float)
        self.F = np.asarray(F, int)
        self.name = name
        self.closed = bool(closed)
        self.sign = 1.0
        self.orient_check = None
        self.lo, self.hi = self.V.min(0), self.V.max(0)
        self._pd = self._imp = None

    @property
    def pd(self):
        if self._pd is None:
            self._pd = polydata(self.V, self.F)
        return self._pd

    def near(self, P, margin=0.0):
        P = np.atleast_2d(P)
        return np.all((P >= self.lo - margin) & (P <= self.hi + margin), axis=1)

    def sd(self, P, margin=0.0):
        P = np.atleast_2d(np.asarray(P, float))
        out = np.full(len(P), np.inf)
        m = self.near(P, margin) if margin is not None else np.ones(len(P), bool)
        if m.any():
            if self._imp is None:
                nf = vtk.vtkPolyDataNormals()
                nf.SetInputData(self.pd)
                nf.ComputeCellNormalsOn(); nf.ComputePointNormalsOn(); nf.ConsistencyOn()
                if self.closed:
                    nf.AutoOrientNormalsOn()
                nf.SplittingOff(); nf.Update()
                self._imp = vtk.vtkImplicitPolyDataDistance()
                self._imp.SetInput(nf.GetOutput())
            arr = vtk.vtkDoubleArray()
            self._imp.FunctionValue(ns.numpy_to_vtk(np.ascontiguousarray(P[m]), deep=1), arr)
            out[m] = ns.vtk_to_numpy(arr) if self.sign > 0 else -ns.vtk_to_numpy(arr)
        return out

    def orient(self, P_inside):
        """Open sheet: fix the sign so that the reference points (the median of their raw signed distances) read
        negative; orient_check = the fraction of them negative afterwards (1.0 = every reference agrees)."""
        self.sign = 1.0
        s = self.sd(P_inside, None)
        if np.median(s) > 0:
            self.sign = -1.0
            s = -s
        self.orient_check = float((s < 0).mean())
        return self

    def enclosed(self, P, margin=0.0):
        if not self.closed:
            raise ValueError("enclosed(): %s is an open sheet" % self.name)
        P = np.atleast_2d(np.asarray(P, float))
        out = np.zeros(len(P), bool)
        m = self.near(P, margin)
        if m.any():
            pts = vtk.vtkPoints()
            pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(P[m]), deep=1))
            q = vtk.vtkPolyData()
            q.SetPoints(pts)
            sel = vtk.vtkSelectEnclosedPoints()
            sel.SetInputData(q); sel.SetSurfaceData(self.pd); sel.SetTolerance(1e-6); sel.Update()
            out[m] = ns.vtk_to_numpy(sel.GetOutput().GetPointData().GetArray("SelectedPoints")).astype(bool)
        return out


def pen(surf, P):
    """P's points inside `surf`: count by signed distance, the enclosed-points count, their disagreement, the
    deepest penetration and the count deeper than DEEP_MM."""
    P = np.atleast_2d(P)
    if not len(P):
        return dict(n=0, n_enclosed=0, n_disagree=0, max_depth_mm=0.0, n_deeper_0p5=0, n_tested=0), np.zeros(0)
    sd = surf.sd(P, 0.0)
    enc = surf.enclosed(P, 0.0)
    ins = sd < 0
    return dict(n=int(ins.sum()), n_enclosed=int(enc.sum()), n_disagree=int((ins != enc).sum()),
                max_depth_mm=rnd(-sd[ins].min()) if ins.any() else 0.0, n_deeper_0p5=int((sd < -DEEP_MM).sum()),
                n_tested=int(len(P))), sd


def close_sheet(V, F, keep_v):
    """One wall sheet as a closed surface: the faces whose three vertices are in `keep_v`, plus a fan disc over each
    boundary loop, oriented against the loop's edge direction (vvm/v9_sections.close_sheet, the construction that
    reproduced the G32 organ-in-wall counts)."""
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    Fk = F[keep_v[F].all(1)]
    E = np.vstack([Fk[:, [0, 1]], Fk[:, [1, 2]], Fk[:, [2, 0]]])
    n = len(V) + 1
    fwd = E[:, 0].astype(np.int64) * n + E[:, 1]
    rev = E[:, 1].astype(np.int64) * n + E[:, 0]
    bnd = E[~np.isin(fwd, rev)]
    nxt = {int(a): int(b) for a, b in bnd}
    loops, seen = [], set()
    for u0 in nxt:
        if u0 in seen:
            continue
        loop = [u0]
        seen.add(u0)
        u = nxt[u0]
        while u != u0 and u not in seen and u in nxt:
            loop.append(u)
            seen.add(u)
            u = nxt[u]
        loops.append(loop)
    V2 = [V]
    F2 = [Fk]
    c = len(V)
    for lp in loops:
        V2.append(V[lp].mean(0)[None])
        F2.append(np.array([[lp[(i + 1) % len(lp)], lp[i], c] for i in range(len(lp))], int))
        c += 1
    return np.vstack(V2), np.vstack(F2), len(loops)


def grid_mask(surf_list, aff, shape):
    """OR of the voxelisations (vtkPolyDataToImageStencil, voxel centres) of closed surfaces on any affine grid.
    Parts are voxelised separately: overlapping parts would cancel under the even-odd rule if merged."""
    m = np.zeros(shape, bool)
    for V, F in surf_list:
        m |= ev.voxelize(np.asarray(V, float), np.asarray(F, int), shape, aff)
    return m


def box_grid(lo, hi, step=1.0):
    """World-aligned grid over [lo, hi] (voxel centres lo + step/2 + i step): (affine, shape)."""
    lo = np.floor(np.asarray(lo, float)) - 1.0
    hi = np.ceil(np.asarray(hi, float)) + 1.0
    shape = tuple(int(v) for v in np.ceil((hi - lo) / step))
    aff = np.eye(4)
    aff[:3, :3] *= step
    aff[:3, 3] = lo + 0.5 * step
    return aff, shape


def mask_pts(m, aff):
    return np.argwhere(m) @ aff[:3, :3].T + aff[:3, 3]


def sample_nn(mask, aff, X):
    """Nearest-voxel value of `mask` at world points X (False outside the grid)."""
    I = np.rint((np.asarray(X, float) - aff[:3, 3]) @ np.linalg.inv(aff[:3, :3]).T).astype(int)
    ok = np.all((I >= 0) & (I < np.array(mask.shape)), 1)
    out = np.zeros(len(I), bool)
    out[ok] = mask[tuple(I[ok].T)]
    return out


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return round(2.0 * float((a & b).sum()) / s, 4) if s else None


def line_fit(P):
    c = P.mean(0)
    _, U = np.linalg.eigh(np.cov((P - c).T))
    return c, U[:, -1]


# ============================================================================================ the vaginal wall
class Wall:
    """The Stage-2a wall mesh of a run: sheets, stations and the kinematically fixed nodes."""

    def __init__(self, meta, cfg):
        self.meta = meta
        w = meta["wall"]
        self.s2n = np.asarray(meta["surface_obj_vertex_to_tet_node"], int)
        nn = int(self.s2n.max()) + 1
        gi = np.asarray(w["grid_index"], int)
        ns_ = meta["node_sets"]

        def vmask(key):
            m = np.zeros(nn, bool)
            m[np.asarray(ns_.get(key, []), int)] = True
            return m[self.s2n]
        self.inner_v, self.outer_v = vmask("inner_surface"), vmask("outer_surface")
        self.apex_v, self.inf_v = vmask("apex"), vmask("fixed_inferior")
        self.station_v = gi[self.s2n, 0]
        self.n_axial = int(w["n_axial"])
        self.rings = [np.nonzero((self.station_v == k) & self.inner_v)[0] for k in range(self.n_axial)]
        self.cfg = cfg

    def fixed_v(self, phase):
        """Nodes held by a FixedConstraint in this phase: the apex lift's nodes ("lift" + apex_lift_fix; the
        controller switches them on at the first non-B step; profile "linear" prescribes every node) and the
        introitus ring when vagina_inferior is "fixed" (springs otherwise)."""
        g = lambda k: self.cfg.get(k, SCENE_DEFAULTS[k])  # noqa: E731
        m = np.zeros(len(self.s2n), bool)
        if g("apex_attach") == "lift" and g("apex_lift_fix") and phase != "B":
            m |= self.apex_v if g("apex_lift_profile") == "top" else np.ones(len(m), bool)
        if g("vagina_inferior") == "fixed":
            m |= self.inf_v
        return m

    def sheets(self, V, F):
        """The lumen and outer sheets, each CLOSED by fan discs (organ-in-wall counts)."""
        Vi, Fi, _ = close_sheet(V, F, self.inner_v)
        Vo, Fo, _ = close_sheet(V, F, self.outer_v)
        return Surf(Vi, Fi, "wall_lumen"), Surf(Vo, Fo, "wall_outer")

    def lumen_sheet(self, V, F, st=None):
        """The inner (lumen) sheet as it is: OPEN, the wall's own triangles, no fan discs (containment).  Signed
        distance < 0 inside the lumen: the sign is oriented on the lumen-ring centres away from the two ends
        (stations 2 .. n-3)."""
        F = np.asarray(F, int)
        sh = Surf(V, F[self.inner_v[F].all(1)], "wall_lumen_open", closed=False)
        c = (st or self.stations(V))["c"]
        return sh.orient(c[2:-2] if len(c) > 6 else c)

    def stations(self, V):
        """Lumen-ring centres, tangents and mean radii (about the centre, normal to the local tangent)."""
        V = np.asarray(V, float)
        C = np.array([V[r].mean(0) for r in self.rings])
        T = np.gradient(C, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        R = []
        for k, r in enumerate(self.rings):
            q = V[r] - C[k]
            R.append(float(np.linalg.norm(q - np.outer(q @ T[k], T[k]), axis=1).mean()))
        return dict(c=C, t=T, r=np.array(R))


# ============================================================================================ carrying the canal
class Carrier:
    """Moves rest-frame points with the tissue of one frame (see CARRIED CANAL in the module docstring).  corpus =
    (Pk, Tk), the corpus tets, only for an ELASTIC corpus: points inside them are then located as ("k", tet, bary) and
    carried barycentrically; a rigid corpus's run leaves it None, so its points stay "u" (Kabsch) exactly as before."""

    def __init__(self, Pc, Tc, s2n_c, corpus=None):
        self.Pc, self.Tc = Pc, Tc
        self.s2n = s2n_c
        self.Pk = self.Tk = None
        if corpus is not None:
            self.Pk, self.Tk = np.asarray(corpus[0], float), np.asarray(corpus[1], int)
            self.kd_k = cKDTree(self.Pk[self.Tk].mean(1))
            Ak = self.Pk[self.Tk[:, 1:]] - self.Pk[self.Tk[:, :1]]
            self.Minv_k = np.linalg.inv(np.transpose(Ak, (0, 2, 1)))
        self.Vc0 = Pc[s2n_c]
        self.kd = cKDTree(Pc[Tc].mean(1))
        A = Pc[Tc[:, 1:]] - Pc[Tc[:, :1]]
        self.Minv = np.linalg.inv(np.transpose(A, (0, 2, 1)))
        E = set()
        for t in Tc:
            for i in range(4):
                for j in range(i + 1, 4):
                    E.add((min(t[i], t[j]), max(t[i], t[j])))
        E = np.array(sorted(E))
        n = len(Pc)
        w = 1.0 / np.linalg.norm(Pc[E[:, 0]] - Pc[E[:, 1]], axis=1)
        W = spa.coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
        Lap = spa.diags(np.asarray(W.sum(1)).ravel()) - W
        self.bnd = np.zeros(n, bool)
        self.bnd[s2n_c] = True
        inn = ~self.bnd
        self.inn = inn
        self.Lib = Lap[inn][:, self.bnd]
        self.solve = spl.factorized(Lap[inn][:, inn].tocsc())

    def cervix_U(self, Vs):
        """Nodal displacement: the surface nodes exact, the interior the harmonic extension (the FALLBACK: see
        frame_U)."""
        U = np.zeros((len(self.Pc), 3))
        U[self.s2n] = np.asarray(Vs, float) - self.Vc0
        for k in range(3):
            U[self.inn, k] = self.solve(-(self.Lib @ U[self.bnd, k]))
        return U

    def match_U(self, U, Vs):
        """A saved nodal displacement (every tet node) against the frame's surface: (U as float, max surface
        mismatch mm), U None when its shape is wrong or it does not reproduce the surface to U_MATCH_MM."""
        U = np.asarray(U, float)
        if U.shape != self.Pc.shape:
            return None, None
        err = float(np.abs(self.Vc0 + U[self.s2n] - np.asarray(Vs, float)).max())
        return (U if err <= U_MATCH_MM else None), err

    def frame_U(self, Vs, exact=()):
        """The frame's cervix nodal displacement and its source.  `exact`: (name, loader) candidates in order of
        preference (the frame's u_npy, the final state's); the first that matches the surface (match_U) is used,
        else the harmonic extension of the surface."""
        tried = []
        for name, load in exact:
            try:
                U, err = self.match_U(load(), Vs)
            except (OSError, ValueError) as e:
                tried.append(dict(source=name, rejected=str(e)))
                continue
            if U is not None:
                return U, dict(source=name, exact=True, surface_mismatch_mm=rnd(err, 5),
                               **({"rejected": tried} if tried else {}))
            tried.append(dict(source=name, rejected="shape" if err is None else
                              "surface mismatch %.4f mm > %g" % (err, U_MATCH_MM)))
        return self.cervix_U(Vs), dict(source="harmonic extension of the frame surface", exact=False,
                                       **({"rejected": tried} if tried else {}))

    def locate(self, X, vaginal=None, k=40):
        """Per point: ("c", tet, bary) inside a rest cervix tet; ("k", tet, bary) inside a rest corpus tet (elastic
        corpus only); ("u",) corpus-carried; ("v",) vaginal side outside the cervix (stays at rest).  `vaginal` marks
        points of the vaginal side (path s < 0)."""
        X = np.atleast_2d(np.asarray(X, float))
        vaginal = np.zeros(len(X), bool) if vaginal is None else np.asarray(vaginal, bool)
        out = []
        for i, p in enumerate(X):
            hit = None
            for c in np.atleast_1d(self.kd.query(p, k=min(k, len(self.Tc)))[1]):
                l = self.Minv[c] @ (p - self.Pc[self.Tc[c, 0]])
                b = np.r_[1.0 - l.sum(), l]
                if b.min() >= -1e-9:
                    hit = ("c", int(c), b)
                    break
            if hit is None and self.Pk is not None:
                for c in np.atleast_1d(self.kd_k.query(p, k=min(k, len(self.Tk)))[1]):
                    l = self.Minv_k[c] @ (p - self.Pk[self.Tk[c, 0]])
                    b = np.r_[1.0 - l.sum(), l]
                    if b.min() >= -1e-9:
                        hit = ("k", int(c), b)
                        break
            out.append(hit if hit else (("v",) if vaginal[i] else ("u",)))
        return out

    def carry(self, loc, X, U, R, t, Uk=None):
        """Carried points: "c" by the cervix displacement U, "k" by the corpus displacement Uk (barycentric; without Uk
        the corpus's Kabsch fit R, t), "u" by R, t, "v" at rest."""
        X = np.atleast_2d(np.asarray(X, float))
        Y = np.empty_like(X)
        for i, L in enumerate(loc):
            if L[0] == "c":
                Y[i] = L[2] @ (self.Pc[self.Tc[L[1]]] + U[self.Tc[L[1]]])
            elif L[0] == "k" and Uk is not None:
                Y[i] = L[2] @ (self.Pk[self.Tk[L[1]]] + Uk[self.Tk[L[1]]])
            elif L[0] in ("u", "k"):
                Y[i] = R @ X[i] + t
            else:
                Y[i] = X[i]
        return Y


# ============================================================================================ the run
def mesh_dirs(cfg):
    """eval_hybrid.run_mesh_dirs for a cfg (not a tag): every body is meshes/<body>/ except a wall run's vagina,
    which the scene loads from the shadow root meshes/_scene_<vagina_wall_dir>/vagina/ (same three files)."""
    d = {b: "%s/%s" % (EH.MESHES, b) for b in EH.BODIES}
    if cfg.get("vagina_model") == "wall":
        wd = cfg.get("vagina_wall_dir", "vagina_wall")
        for cand in ("%s/_scene_%s/vagina" % (EH.MESHES, wd), "%s/%s" % (EH.MESHES, wd)):
            if os.path.exists(cand + "/tets.vtk"):
                d["vagina"] = cand
                break
        else:
            raise SystemExit("vagina_model='wall' but no mesh for vagina_wall_dir=%r under %s" % (wd, EH.MESHES))
    return d


class Run:
    """Everything one run's scoring needs, read once."""

    def __init__(self, tag, rd=None):
        self.tag = tag
        self.rd = (rd or "%s/%s" % (RUNS, tag)).replace("\\", "/")
        def rj(n, d=None):
            p = "%s/%s" % (self.rd, n)
            return json.load(open(p)) if os.path.exists(p) else d
        self.cfg = rj("cfg.json", {})
        self.summary = rj("summary.json", {})
        self.devfin = rj("device_final.json")
        self.md = mesh_dirs(self.cfg)
        self.rest = {b: geom.read_obj(self.md[b] + "/surface.obj") for b in EH.BODIES}
        self.meta = {b: json.load(open(self.md[b] + "/meta.json")) for b in EH.BODIES}
        self.wall = Wall(self.meta["vagina"], self.cfg) if self.cfg.get("vagina_model") == "wall" else None
        # device: the run's applicator variant, cap files, rods, packing (eval_hybrid.run_devsurf; cx is unused).  A
        # tandem-only run (TF0) whose applicator dir also holds ovoid files names its parts in cfg device_parts, or
        # switches the ovoid body off with device_ovoids false / ovoid_mode "none"
        self.devsurf, self.appd = EH.run_devsurf(None, self.cfg)
        if self.cfg.get("device_parts"):
            self.devsurf = {p: v for p, v in self.devsurf.items() if p in set(self.cfg["device_parts"])}
        elif self.cfg.get("device_ovoids") is False or self.cfg.get("ovoid_mode") in ("none", "off"):
            self.devsurf = {p: v for p, v in self.devsurf.items() if p not in OVOID_BODY}
        self.tandem_parts = [p for p in self.devsurf if p not in OVOID_BODY]
        self.ovoid_parts = [p for p in self.devsurf if p in OVOID_BODY]
        self.appj = json.load(open(self.appd + "/applicator.json"))
        self.pose = json.load(open(self.appd + "/pose.json")) if os.path.exists(self.appd + "/pose.json") else {}
        prm = self.appj.get("params", {})
        pv = lambda *k: next((float(prm[x]["value"]) for x in k if x in prm), None)  # noqa: E731
        self.L_iu = pv("L_iu_mm") or 61.8
        self.r_tube_app = pv("r_tube_mm", "r_tandem_mm")
        # S3 path + cervix / corpus meshes the run used
        self.T = TP.load()
        s = self.T["s"]
        self.ic = np.nonzero(s >= 0)[0]
        self.canal_s = s[self.ic]
        Pc, Tc = read_vtk_legacy(self.md["cervix"] + "/tets.vtk")
        self.Pc, self.Tc = Pc, Tc
        # S9: the corpus as the run modelled it (cfg corpus_model; "rigid" for every run before it) and its tets
        self.corpus_fem = self.cfg.get("corpus_model", "rigid") == "fem"
        Pk, Tk = read_vtk_legacy(self.md["corpus"] + "/tets.vtk")
        self.Pk, self.Tk = np.asarray(Pk, float), np.asarray(Tk, int)
        self.s2n_k = np.asarray(self.meta["corpus"]["surface_obj_vertex_to_tet_node"], int)
        nsk = self.meta["corpus"]["node_sets"]
        self.serosa_nodes = np.setdiff1d(np.asarray(nsk["surface_nodes"], int), np.asarray(nsk["interface_cervix"], int))
        self.serosa_v = np.isin(self.s2n_k, self.serosa_nodes)       # the frame surface's vertices on the serosa
        pk = self.rd + "/final/corpus_u.npy"
        self.final_uk_path = pk if os.path.exists(pk) else None
        self.carrier = Carrier(Pc, Tc, np.asarray(self.meta["cervix"]["surface_obj_vertex_to_tet_node"], int),
                               corpus=(self.Pk, self.Tk) if self.corpus_fem else None)
        self.loc_path = self.carrier.locate(self.T["pts"], vaginal=s < 0)
        self.path_kinds = "".join(L[0] for L in self.loc_path)
        # canal ties: the set the run used and the depth-engagement arclengths (S3 canal_path_s)
        nsets = self.meta["cervix"]["node_sets"]
        self.tie_set = self.cfg.get("canal_tie_set", SCENE_DEFAULTS["canal_tie_set"])
        self.tie_nodes = np.asarray(nsets.get(self.tie_set, []), int)
        self.tie_s = np.asarray(nsets.get(self.tie_set + "_s", []), float)
        self.cp_nodes = np.asarray(nsets.get("canal_path", []), int)
        self.cp_s = np.asarray(nsets.get("canal_path_s", []), float)
        # frames + log (before the plan record: its rows are checked against the log)
        idx = "%s/frames/index.json" % self.rd
        self.frames = json.load(open(idx))["frames"] if os.path.exists(idx) else []
        self.log = {}
        if os.path.exists(self.rd + "/log.jsonl"):
            for ln in open(self.rd + "/log.jsonl"):
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if "step" in r:
                    self.log[int(r["step"])] = r
        # the final state's exact cervix displacement (the last saved frame's carry when it is the final state)
        pf = self.rd + "/final/cervix_u.npy"
        self.final_u_path = pf if os.path.exists(pf) else None
        # tandem-first plan record (S5) and tau: the record's, else the S3 default (always carried per frame, so
        # tip_to_tau exists for every run, G32 included)
        self.tf = self._tf_record()
        self.tau, self.tau_s, self.tau_src = self._tau(self.pose.get("insertion_path_tandem_first") or {})
        self.loc_tau = self.carrier.locate(self.tau, vaginal=self.tau_s < 0)
        self.rest_st = self.wall.stations(self.rest["vagina"][0]) if self.wall else None
        self.rest_solid = {o: Surf(*self.rest[o], o) for o in ("bladder", "rectum")}

    # ---- device placement: p_world = origin + p_app @ R_rows (ovoid body: its own origin and frame)
    def place(self, part, dev):
        V, F = self.devsurf[part]
        if part in OVOID_BODY and dev.get("ovoid_origin_mm") is not None:
            org = np.asarray(dev["ovoid_origin_mm"], float)
            R = np.array([dev["ovoid_x_app"], dev["ovoid_y_app"], dev["ovoid_axis"]], float)
        else:
            org = np.asarray(first(dev, ("flange_mm", "flange")), float)
            R = np.array([dev["x_app"], dev["y_app"], dev["tube_axis"]], float)
        return org + np.asarray(V, float) @ R, F

    def r_tube(self, dev):
        return float(first(dev, ("r_tube_mm", "r_tandem_mm"), self.r_tube_app or 2.18))

    # ---- the frame's cervix nodal displacement (see CARRIED CANAL)
    def frame_U(self, fr, Vs):
        """(U, carry record) of one saved frame: its u_npy, else final/cervix_u.npy when it is the last saved frame,
        each only if it reproduces the frame's cervix surface; else the harmonic extension."""
        fd = self.rd + "/frames"
        cand = []
        un = (fr.get("u_npy") or {}).get("cervix")
        if un:
            cand.append(("frames/%s" % un, lambda: np.load("%s/%s" % (fd, un))))
        if self.final_u_path and self.frames and int(fr["step"]) == int(self.frames[-1]["step"]):
            cand.append(("final/cervix_u.npy (last saved frame)", lambda: np.load(self.final_u_path)))
        return self.carrier.frame_U(Vs, cand)

    def frame_Uk(self, fr, Vk):
        """S9: (Uk, carry record) of one saved frame's CORPUS: its u_npy (cfg frame_u_bodies), else final/corpus_u.npy
        when it is the last saved frame, each only if it reproduces the frame's corpus surface to U_MATCH_MM; else the
        Kabsch fit of the frame's corpus surface (exact for a rigid corpus, an approximation for an elastic one)."""
        fd = self.rd + "/frames"
        cand = []
        un = (fr.get("u_npy") or {}).get("corpus")
        if un:
            cand.append(("frames/%s" % un, lambda: np.load("%s/%s" % (fd, un))))
        if self.final_uk_path and self.frames and int(fr["step"]) == int(self.frames[-1]["step"]):
            cand.append(("final/corpus_u.npy (last saved frame)", lambda: np.load(self.final_uk_path)))
        V0 = self.Pk[self.s2n_k]
        Vk = np.asarray(Vk, float)
        tried = []
        for name, ld in cand:
            try:
                U = np.asarray(ld(), float)
            except (OSError, ValueError) as e:
                tried.append(dict(source=name, rejected=str(e)))
                continue
            if U.shape != self.Pk.shape:
                tried.append(dict(source=name, rejected="shape"))
                continue
            err = float(np.abs(V0 + U[self.s2n_k] - Vk).max())
            if err <= U_MATCH_MM:
                return U, dict(source=name, exact=True, surface_mismatch_mm=rnd(err, 5),
                               **({"rejected": tried} if tried else {}))
            tried.append(dict(source=name, rejected="surface mismatch %.4f mm > %g" % (err, U_MATCH_MM)))
        R, t = geom.kabsch(V0, Vk)
        e = float(np.abs(V0 @ R.T + t - Vk).max())
        return self.Pk @ R.T + t - self.Pk, dict(source="Kabsch fit of the frame's corpus surface", exact=False,
                                                 approx=bool(self.corpus_fem), surface_residual_mm=rnd(e, 4),
                                                 **({"rejected": tried} if tried else {}))

    # ---- the tandem-first plan
    def _tau(self, rec):
        """tau(d) in the rest frame: the record's tau_pts / tau_s, else the plan's S5 definition on the S3 path
        (the path itself for d <= TAU_FOLLOW_MM, including the vaginal part for d < 0, then straight along a_lc)."""
        tau = first(rec, ("tau_pts", "tau"))
        tau_s = first(rec, ("tau_s", "tau_d_mm"))
        if tau is not None and tau_s is not None:
            return np.asarray(tau, float), np.asarray(tau_s, float), "pose.json insertion_path_tandem_first tau"
        T = self.T
        s = T["s"]
        sf = min(float(first(rec, ("tau_follow_mm",), TAU_FOLLOW_MM)), float(s[-1]))
        d = np.arange(float(s[0]), float(s[-1]) + 30.0, 1.0)
        tau = TP.interp_path(T["pts"], s, np.minimum(d, sf)) + np.outer(np.maximum(d - sf, 0.0), unit(T["a_lc"]))
        return tau, d, "S3 path: labelled canal to %.0f mm, then a_lc (plan S5 default)" % sf

    def _tf_record(self):
        rec = self.pose.get("insertion_path_tandem_first")
        if not rec:
            return None
        tau, tau_s, src = self._tau(rec)
        rows, rows_step = self._tf_rows(rec)
        return dict(tau=tau, tau_s=tau_s, delta=np.asarray(first(rec, ("delta_mm", "delta"), [0.0, 0.0, 0.0]), float),
                    rows=rows, rows_step=rows_step, w_table=rec.get("w_r_table"), source=src)

    def _tf_rows(self, rec):
        """The record's rows by simulation step, and how the step was found.  A row that carries 'step' keeps it.
        S5 (applicator_venezia) writes them keyed 'i' (0-based, no step); scene_hybrid.build_schedule_tandem_first
        runs row i after n_balloon B steps and n_presettle P steps, so step = i + n_balloon + n_presettle (a row
        without 'i' takes its list position).  That offset is checked against the log: the scene logs the row's
        tip_s as canal_tie.tip_s, so every row whose step is logged must agree to ROW_TIP_S_TOL_MM.  If the cfg offset
        disagrees, the offset that agrees is searched over the logged steps; if none does, the rows are not used."""
        rows = [r for r in (rec.get("rows") or []) if isinstance(r, dict)]
        if not rows:
            return {}, None
        cf = lambda k: int(self.cfg.get(k, SCENE_DEFAULTS[k]) or 0)  # noqa: E731
        off0 = cf("n_balloon") + cf("n_presettle")
        fixed = {int(r["step"]): r for r in rows if r.get("step") is not None}
        free = [(int(r["i"]) if r.get("i") is not None else j, r) for j, r in enumerate(rows) if r.get("step") is None]
        logged = {k: float(ts) for k, ts in ((k, _tie_log(r)[1]) for k, r in self.log.items()) if ts is not None}

        def check(off):
            e = [abs(float(r["tip_s"]) - logged[i + off]) for i, r in free
                 if r.get("tip_s") is not None and (i + off) in logged]
            return (max(e) if e else None), len(e)
        info = dict(n_rows=len(rows), n_keyed_step=len(fixed), n_keyed_i=len(free), cfg_offset=off0)
        if not free:
            info.update(offset=None, check="every row carries its step")
            return fixed, info
        err, n = check(off0)
        off = off0
        if n == 0:
            info.update(check="unverified: the log has no canal_tie.tip_s at the rows' steps")
        elif err <= ROW_TIP_S_TOL_MM:
            info.update(check="verified: %d rows vs the log's canal_tie.tip_s, max |diff| %.4f mm" % (n, err))
        else:
            best = None
            for o in range(min(logged) - max(i for i, _ in free), max(logged) + 1):
                e, m = check(o)
                if m and e <= ROW_TIP_S_TOL_MM and (best is None or (m, -e) > (best[2], -best[1])):
                    best = (o, e, m)
            if best is None:
                info.update(offset=None, check="REFUSED: cfg offset %d disagrees with the log's canal_tie.tip_s by up "
                                               "to %.3f mm and no offset agrees; rows not used" % (off0, err))
                return fixed, info
            off = best[0]
            info.update(check="RE-FITTED: cfg offset %d disagrees with the log (max |diff| %.3f mm); offset %d agrees "
                              "on %d rows (max |diff| %.4f mm)" % (off0, err, best[0], best[2], best[1]))
        info["offset"] = off
        out = {i + off: r for i, r in free}
        out.update(fixed)
        return out, info

    def plan_at(self, dev, step):
        """(d, w_r, source) of the frame's planned tip, or None.  d = the schedule row's tip_s (S1b: the tip's
        arclength along the tandem path from O_true, = the plan's depth d), read from the frame's device json, else
        the record's rows by step (Run._tf_rows), else the step's log row (canal_tie.tip_s)."""
        if self.tf is None:
            return None
        d = first(dev, ("tip_s", "tf_d_mm", "d_mm"))
        w = first(dev, ("w_r", "tf_w_r"))
        src = "device json"
        row = self.tf["rows"].get(int(step))
        if d is None and row is not None:
            off = (self.tf.get("rows_step") or {}).get("offset")
            d, src = first(row, ("tip_s", "d_mm", "d", "tf_d_mm")), ("pose.json rows" if off is None else
                                                                   "pose.json rows (step = i + %d)" % off)
        if d is None and int(step) in self.log:
            d, src = _tie_log(self.log[int(step)])[1], "log canal_tie.tip_s"
        if w is None and row is not None:
            w = first(row, ("w_r", "tf_w_r"))
        if w is None and d is not None and self.tf["w_table"]:
            t = self.tf["w_table"]
            w = float(np.interp(float(d), t["d_mm"], t["w_r"]))
        if d is None:
            return None
        return dict(d=float(d), w_r=float(w if w is not None else 0.0), source=src,
                    w_r_known=w is not None)


# ============================================================================================ per-frame blocks
def insertion_block(run, dev, step, path_c, tau_c, U, Rk, tk, sol, parts, st, lumen, outer, lumen_open=None,
                    Uk=None, Rr=None, tr=None):
    """Tip vs the carried canal / plan, canal-in-tube, ties, containment, the ovoid body (model frame).  lumen /
    outer: the closed sheets; lumen_open: the open inner sheet (containment; the closed lumen if None).  S9 (elastic
    corpus): Uk carries the plan's corpus points; Rr, tr (the serosa's Kabsch fit) map the tip into the corpus's rest
    frame (default Rk, tk: the corpus surface's, exact for a rigid corpus)."""
    Rr, tr = (Rk, tk) if Rr is None else (Rr, tr)
    T = run.T
    F = np.asarray(first(dev, ("flange_mm", "flange")), float)
    a = unit(dev["tube_axis"])
    L = float(first(dev, ("L_iu_mm",), run.L_iu))
    tip = np.asarray(first(dev, ("tip_mm", "tip"), F + L * a), float)
    rt = run.r_tube(dev)
    canal = path_c[run.ic]
    cs = run.canal_s
    d_lab, s_lab, _ = TP.project(tip[None], canal, cs)
    O = path_c[T["i_os"]]
    d_meas = float((tip - O) @ a)
    d_path, s_path, _ = TP.project(tip[None], path_c, T["s"])
    d_tau, s_tau, _ = TP.project(tip[None], tau_c, run.tau_s)
    out = dict(tip_mm=rnd(tip, 2), r_tube_mm=rnd(rt),
               tip_to_labelled_canal_mm=rnd(d_lab[0]), tip_canal_s_mm=rnd(s_lab[0], 1),
               tip_to_path_mm=rnd(d_path[0]), tip_path_s_mm=rnd(s_path[0], 1),
               tip_to_tau_mm=rnd(d_tau[0]), tip_tau_d_mm=rnd(s_tau[0], 1),
               d_measured_mm=rnd(d_meas, 2), O_true_carried=rnd(O, 2),
               # the tip and the flange in the corpus's REST frame: constant while tandem and corpus move as one body
               tip_in_corpus_rest=rnd(Rr.T @ (tip - tr), 3), flange_in_corpus_rest=rnd(Rr.T @ (F - tr), 3))
    # ---- the plan
    plan = run.plan_at(dev, step)
    if plan is not None:
        tf = run.tf
        tau = TP.interp_path(tf["tau"], tf["tau_s"], plan["d"])
        loc = run.carrier.locate(tau, vaginal=np.array([plan["d"] < 0]))
        p = run.carrier.carry(loc, tau, U, Rk, tk, Uk)[0] + plan["w_r"] * tf["delta"]
        out["plan"] = dict(d_mm=rnd(plan["d"], 2), w_r=rnd(plan["w_r"], 4), w_r_known=plan["w_r_known"],
                           source=plan["source"], tau_carried_by=loc[0][0], p_mm=rnd(p, 2))
        out["tip_to_plan_mm"] = rnd(np.linalg.norm(tip - p))
    else:
        out["tip_to_plan_mm"] = None
    d = plan["d"] if plan is not None else d_meas
    out["d_used_mm"], out["d_source"] = rnd(d, 2), ("plan" if plan is not None else "measured")
    # ---- canal against the tube (segment F .. tip)
    dseg, h = seg_dist(canal, F, a, L)
    within = dseg <= rt + TUBE_TOL_MM
    inspan = (h >= 0) & (h <= L)
    lower = (cs >= 0) & (cs <= min(d, LOWER_CANAL_MM))
    trav = cs <= d
    up = cs > LOWER_CANAL_MM
    out.update(
        lower_canal_in_tube_frac=frac(within[lower]) if d > 0 else None, n_lower=int(lower.sum()) if d > 0 else 0,
        # read by the S5 / S6 gate: "after d = 5 mm", on the unrounded d
        lower_canal_gated=bool(d > LOWER_GATE_MM),
        lower_canal_worst_mm=rnd(dseg[lower].max()) if d > 0 and lower.any() else None,
        traversed_canal_in_tube_frac=frac(within[inspan]) if inspan.any() else None, n_traversed=int(inspan.sum()),
        traversed_by_depth_frac=frac(within[trav]) if d > 0 else None,
        upper_canal_residual_mm=rnd(dseg[up].max()) if up.any() else None,
        # the nearest in-span canal point: a fraction of 0 next to a distance of r_tube + 1.01 mm is a threshold
        # effect (G32 step 132: 3.19 mm against 3.18)
        min_inspan_canal_dist_to_tube_mm=rnd(dseg[inspan].min()) if inspan.any() else None,
        canal_dist_to_tube_mm=dict(s=[int(round(v)) for v in cs[::4]], d=rnd(dseg[::4], 2)))
    # ---- ties: engaged / active counts from the log; the nodes the tip has passed (canal_s <= d) for comparison
    lr = run.log.get(int(step), {})
    ne, ts = _tie_log(lr) if lr else (None, None)
    out["ties"] = dict(n_engaged=ne, n_active=first(dev, ("n_canal_ties",), lr.get("n_canal_ties")), set=run.tie_set,
                       tip_s_logged=rnd(ts, 3), n_set=int(len(run.tie_nodes)),
                       n_set_passed=int((run.tie_s <= d).sum()) if len(run.tie_s) else None,
                       n_canal_path_passed=int((run.cp_s <= d).sum()) if len(run.cp_s) else None,
                       n_canal_path=int(len(run.cp_s)))
    # ---- containment (restricted): tandem-body vertices below the wall's current top, above the introitus and
    # outside the cervix and corpus, tested inside the lumen by signed distance to the OPEN inner sheet (the lumen
    # wall itself; the fan discs that close it are not tissue: see INSIDE TESTS); the closed-lumen reading beside it
    X = np.vstack([parts[p].V for p in run.tandem_parts]) if run.tandem_parts else np.zeros((0, 3))
    if st is not None and len(X):
        c, t = st["c"], st["t"]
        h_top = (X - c[-1]) @ t[-1]
        h_bot = (X - c[0]) @ t[0]
        in_tis = (sol["cervix"].sd(X, 1.0) <= 0) | (sol["corpus"].sd(X, 1.0) <= 0)
        cons = (h_top <= 0) & (h_bot >= 0) & ~in_tis
        shw = lumen_open if lumen_open is not None else lumen
        blk = dict(n_vertices=int(len(X)), n_in_tissue=int(in_tis.sum()), n_above_top=int((h_top > 0).sum()),
                   n_below_introitus=int((h_bot < 0).sum()), n_considered=int(cons.sum()),
                   sheet="open inner sheet (no fan discs)" if lumen_open is not None else "closed lumen (fan discs)")
        if lumen_open is not None:
            blk["sheet_orientation_check"] = rnd(lumen_open.orient_check, 3)
        if cons.any():
            Xc = X[cons]
            sl = shw.sd(Xc, None)
            so = outer.sd(Xc, None)
            ok = sl < 0
            j = int(np.argmax(sl))
            # n_in_wall / n_outside_wall split the vertices OUTSIDE the lumen (inside / outside the closed outer sheet)
            blk.update(frac=frac(ok), n_outside_lumen=int((~ok).sum()),
                       n_in_wall=int((~ok & (so < 0)).sum()), n_outside_wall=int((~ok & (so >= 0)).sum()),
                       min_clearance_mm=rnd(-sl.max()), n_clearance_lt_1mm=int((sl > -CLEAR_MM).sum()),
                       max_outside_mm=rnd(sl.max()) if (~ok).any() else 0.0,
                       worst_h_above_introitus_mm=rnd(h_bot[cons][j], 2))
            if lumen_open is not None:
                sc = lumen.sd(Xc, None)
                blk["closed_sheet"] = dict(frac=frac(sc < 0), n_outside_lumen=int((sc >= 0).sum()),
                                           min_clearance_mm=rnd(-sc.max()),
                                           worst_h_above_introitus_mm=rnd(h_bot[cons][int(np.argmax(sc))], 2),
                                           note="S0 1.0 reading: the lumen closed by fan discs (not used)")
        else:
            blk.update(frac=None)
        out["containment"] = blk
    # ---- ovoid body: inside the introitus, pose relative to the tandem, device-device clearance
    if run.ovoid_parts:
        Xo = np.vstack([parts[p].V for p in run.ovoid_parts])
        ob = {}
        if st is not None:
            hmax = float(((Xo - st["c"][0]) @ st["t"][0]).max())
            ob.update(inside_introitus=bool(hmax > 0), max_height_above_introitus_mm=rnd(hmax, 2),
                      tip_height_above_introitus_mm=rnd((tip - st["c"][0]) @ st["t"][0], 2))
        org = np.asarray(first(dev, ("ovoid_origin_mm",), F), float)
        q = org - F
        ob.update(origin_below_flange_along_tube_mm=rnd(-(q @ a), 2),
                  origin_lateral_mm=rnd(np.linalg.norm(q - (q @ a) * a), 2),
                  axis_angle_to_tube_deg=rnd(ang(first(dev, ("ovoid_axis",), a), a), 2))
        clr = {}
        if len(X):
            for p in run.ovoid_parts:
                sd = parts[p].sd(X, 5.0)
                clr[p] = dict(min_sd_mm=rnd(sd.min()) if np.isfinite(sd.min()) else ">5",
                              n_tandem_vertices_inside=int((sd < 0).sum()))
        ob["tandem_vs_ovoid_parts"] = clr
        # per ovoid half (cap + its rod; S4b fuses them into one closed body per half)
        ob["tandem_vs_ovoid_halves"] = {
            h: min((clr[p]["min_sd_mm"] for p in ps if p in clr and clr[p]["min_sd_mm"] != ">5"), default=">5")
            for h, ps in HALVES.items() if any(p in clr for p in ps)}
        out["ovoid_body"] = ob
    return out


def penetration_block(run, S, sol, parts, lumen, outer, phase):
    """Signed-distance penetration counts (model frame)."""
    out = {}
    if lumen is not None:
        out["organs_in_wall"] = {o: dict(outer=pen(outer, S[o][0])[0], lumen=pen(lumen, S[o][0])[0])
                                 for o in ORGANS}
        W = S["vagina"][0]
        r, sd = pen(sol["cervix"], W)
        ins = sd < 0
        fx = run.wall.fixed_v(phase)
        grp = {}
        for nm, m in (("lumen", run.wall.inner_v), ("outer", run.wall.outer_v)):
            for fn, f in (("fixed", fx), ("free", ~fx)):
                k = ins & m & f
                grp["%s_%s" % (nm, fn)] = dict(n=int(k.sum()), max_depth_mm=rnd(-sd[k].min()) if k.any() else 0.0,
                                               n_deeper_0p5=int((k & (sd < -DEEP_MM)).sum()))
        for nm, m in (("lumen", run.wall.inner_v), ("outer", run.wall.outer_v), ("free", ~fx), ("fixed", fx)):
            k = ins & m
            grp[nm] = dict(n=int(k.sum()), max_depth_mm=rnd(-sd[k].min()) if k.any() else 0.0,
                           n_deeper_0p5=int((k & (sd < -DEEP_MM)).sum()))
        out["wall_in_cervix"] = dict(all=r, **grp)
    Vc = S["cervix"][0]
    out["cervix_in_parts"] = {p: pen(parts[p], Vc)[0] for p in parts}
    out["parts_in_organs"] = {p: {o: pen(sol[o], parts[p].V)[0] for o in ORGANS} for p in parts}
    return out


def wall_block(run, step, st, sol):
    """Axial stretch and necking per station (log + geometry), centreline shape, vault-to-portio gap."""
    if st is None:
        return None
    lr = run.log.get(int(step), {})
    cont = ((lr.get("wall") or {}).get("containment") or {})
    ax = np.asarray([v for v in cont.get("axial_stretch", []) if v is not None], float)
    rin = np.asarray([v for v in cont.get("r_in_mm", []) if v is not None], float)
    c, c0 = st["c"], run.rest_st["c"]
    geo = np.linalg.norm(np.diff(c, axis=0), axis=1) / np.linalg.norm(np.diff(c0, axis=0), axis=1)
    neck = st["r"] / run.rest_st["r"]
    n = len(c)
    k3 = max(2, n // 3)
    ctop, ttop = c[-1], st["t"][-1]
    ts = np.arange(0.0, 40.01, 0.25)
    sdr = sol["cervix"].sd(ctop + np.outer(ts, ttop), None)
    gap = float(sdr[0]) if sdr[0] < 0 else (float(ts[np.argmax(sdr <= 0)]) if (sdr <= 0).any() else None)
    return dict(
        axial_stretch_log=dict(mean=rnd(ax.mean()) if ax.size else None, max=rnd(ax.max()) if ax.size else None),
        r_in_log_mm=dict(min=rnd(rin.min()) if rin.size else None, mean=rnd(rin.mean()) if rin.size else None),
        axial_stretch_geom=dict(mean=rnd(geo.mean()), max=rnd(geo.max()), min=rnd(geo.min())),
        necking_r_over_rest=dict(min=rnd(neck.min()), mean=rnd(neck.mean()), station_min=int(np.argmin(neck))),
        lumen_r_mm=dict(min=rnd(st["r"].min()), mean=rnd(st["r"].mean())),
        centreline_bow_mm=rnd(geom.max_dev_from_chord(c)),
        lower_upper_third_angle_deg=rnd(ang(c[k3] - c[0], c[-1] - c[-1 - k3]), 2),
        length_mm=rnd(np.linalg.norm(np.diff(c, axis=0), axis=1).sum(), 2),
        vault_to_portio_gap_mm=rnd(gap),
        per_station=dict(stretch_geom=rnd(geo, 3), necking=rnd(neck, 3)))


def oar_block(run, S, parts, outer):
    """OAR displacement relative to the device: umax, the tandem-alone intrusion into the preBT OAR at this device
    pose, and the displacement of OAR vertices far (> REMOTE_MM) from every device part and from the wall's outer
    sheet (motion not caused by local contact)."""
    out = {}
    Xt = np.vstack([parts[p].V for p in run.tandem_parts]) if run.tandem_parts else np.zeros((0, 3))
    for o in ORGANS:
        X = S[o][0]
        u = np.linalg.norm(X - run.rest[o][0], axis=1)
        dd = np.full(len(X), np.inf)
        for p in parts.values():
            dd = np.minimum(dd, np.abs(p.sd(X, REMOTE_MM)))
        dsh = np.abs(outer.sd(X, REMOTE_MM)) if outer is not None else np.full(len(X), np.inf)
        far = (dd > REMOTE_MM) & (dsh > REMOTE_MM)
        rec = dict(umax_surface_mm=rnd(u.max()), n_remote=int(far.sum()),
                   remote_umax_mm=rnd(u[far].max()) if far.any() else None,
                   remote_u_p95_mm=rnd(np.percentile(u[far], 95)) if far.any() else None,
                   remote_u_mean_mm=rnd(u[far].mean()) if far.any() else None)
        if o in run.rest_solid and len(Xt):
            sd = run.rest_solid[o].sd(Xt, 1.0)
            rec["tandem_alone_intrusion_into_preBT"] = dict(n_vertices=int((sd < 0).sum()),
                                                            max_depth_mm=rnd(-sd.min()) if (sd < 0).any() else 0.0)
        out[o] = rec
    return out


CORPUS_BAND_EDGES = (0.0, 20.0, 30.0, 35.0, 40.0)   # lower edges of the canal_s bands (S9 design); see corpus_bands
SPAN_R_MM = 5.0               # corpus span along the tube: tet nodes within this of the tube LINE (the audit's 44.1 / 50.1)


def _tet_vol(X, T):
    A = X[T]
    return np.einsum("ij,ij->i", np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]), A[:, 3] - A[:, 0]) / 6.0


def corpus_bands(canal_s_max):
    """The canal_s bands of corpus_block's canal_to_tube_by_band_mm: CORPUS_BAND_EDGES below the canal's end, the last
    band ending at the labelled canal's own end (its largest canal_s) rounded UP to a whole mm, so it holds the canal's
    last point.  Derived from the run's canal, not a constant (the key of the last band therefore names that end)."""
    top = float(np.ceil(float(canal_s_max)))
    e = [x for x in CORPUS_BAND_EDGES if x < top] + [top]
    return tuple(zip(e[:-1], e[1:]))


def corpus_block(run, dev, Xk, path_c, exact):
    """S9: the corpus of one state (model frame).  Xk = its nodes (exact: from a saved nodal displacement, or rigid);
    dev = the state's device json (flange_mm, tube_axis, and T_corpus = the schedule's rigid target when the corpus is
    elastic).  pose_vs_target: every node vs X0 @ T_corpus; nonrigid_serosa: the serosa nodes vs their best rigid fit to
    rest (0 for a rigid corpus); volume: total and per-tet ratios; span_along_tube: h (above the flange along the tube)
    of the tet nodes within SPAN_R_MM of the tube line, [min, max, max - min] (max = the fundus top on the tube);
    canal_to_tube_by_band: the carried labelled canal's max distance to the tube segment per canal_s band;
    canal_best_line_max: the carried canal's max distance from its own best-fit line (straightness)."""
    Pk, Tk = run.Pk, run.Tk
    Xk = np.asarray(Xk, float)
    F = np.asarray(first(dev, ("flange_mm", "flange")), float)
    a = unit(dev["tube_axis"])
    out = dict(model="fem" if run.corpus_fem else "rigid", nodes_exact=bool(exact))
    T = dev.get("T_corpus") or dev.get("corpus_T_preBT_to_final")
    if T is not None:
        T = np.asarray(T, float)
        e = np.linalg.norm(Pk @ T[:3, :3].T + T[:3, 3] - Xk, axis=1)
        out["pose_vs_target_mm"] = dict(rms=rnd(np.sqrt((e ** 2).mean()), 4), max=rnd(e.max(), 4))
    ser = run.serosa_nodes
    R, t = geom.kabsch(Pk[ser], Xk[ser])
    e = np.linalg.norm(Pk[ser] @ R.T + t - Xk[ser], axis=1)
    out["nonrigid_serosa_mm"] = dict(rms=rnd(np.sqrt((e ** 2).mean()), 4), p95=rnd(np.percentile(e, 95), 4),
                                     max=rnd(e.max(), 4))
    v0, v = _tet_vol(Pk, Tk), _tet_vol(Xk, Tk)
    r = v / v0
    out["volume"] = dict(ratio_total=rnd(v.sum() / v0.sum(), 5), min_tet_ratio=rnd(r.min(), 4),
                         n_tets_lt_0p6=int((r < 0.6).sum()), n_inverted=int((r <= 0).sum()))
    q = Xk - F
    h = q @ a
    near = np.linalg.norm(q - np.outer(h, a), axis=1) <= SPAN_R_MM
    out["span_along_tube_mm"] = [rnd(h[near].min(), 2), rnd(h[near].max(), 2), rnd(h[near].max() - h[near].min(), 2)] \
        if near.any() else None
    canal, cs = path_c[run.ic], run.canal_s
    dseg, _ = seg_dist(canal, F, a, run.L_iu)
    out["canal_to_tube_by_band_mm"] = {"%g-%g" % (lo, hi): (rnd(dseg[(cs >= lo) & (cs <= hi)].max(), 3)
                                                            if ((cs >= lo) & (cs <= hi)).any() else None)
                                       for lo, hi in corpus_bands(cs.max())}
    c, ax = line_fit(canal)
    w = canal - c
    out["canal_best_line_max_mm"] = rnd(np.linalg.norm(w - np.outer(w @ ax, ax), axis=1).max(), 3)
    return out


def score_frame(run, fr):
    fd = run.rd + "/frames"
    step = int(fr["step"])
    dev = json.load(open("%s/%s" % (fd, fr["device"])))
    S = {}
    for b in EH.BODIES:
        fn = (fr.get("surfaces") or {}).get(b)
        S[b] = geom.read_obj("%s/%s" % (fd, fn)) if fn and os.path.exists("%s/%s" % (fd, fn)) else run.rest[b]
    sol = {b: Surf(*S[b], b) for b in ("cervix", "corpus") + ORGANS}
    parts = {p: Surf(*run.place(p, dev), p) for p in run.devsurf}
    lumen = outer = st = lumen_open = None
    if run.wall is not None:
        lumen, outer = run.wall.sheets(*S["vagina"])
        st = run.wall.stations(S["vagina"][0])
        lumen_open = run.wall.lumen_sheet(*S["vagina"], st=st)
    U, carry = run.frame_U(fr, S["cervix"][0])
    Rk, tk = geom.kabsch(run.rest["corpus"][0], S["corpus"][0])
    Uk = Rr = tr = None
    if run.corpus_fem:                                  # S9: the elastic corpus's own nodal displacement
        Uk, ck = run.frame_Uk(fr, S["corpus"][0])
        carry = dict(carry, corpus=ck)
        sv = run.serosa_v
        Rr, tr = geom.kabsch(run.rest["corpus"][0][sv], S["corpus"][0][sv])
        Xk, xk_exact = run.Pk + Uk, bool(ck.get("exact"))
    else:                                               # rigid: the surface's Kabsch fit IS the corpus's motion
        Xk, xk_exact = run.Pk @ Rk.T + tk, True
    path_c = run.carrier.carry(run.loc_path, run.T["pts"], U, Rk, tk, Uk)
    tau_c = run.carrier.carry(run.loc_tau, run.tau, U, Rk, tk, Uk)
    phase = dev.get("phase", fr.get("phase"))
    return dict(step=step, phase=phase, u=rnd(dev.get("u"), 4),
                disp_umax_mm={b: (dev.get("disp") or {}).get(b, {}).get("umax_mm") for b in EH.BODIES},
                n_contacts=dev.get("n_contacts"), carry=carry,
                insertion=insertion_block(run, dev, step, path_c, tau_c, U, Rk, tk, sol, parts, st, lumen, outer,
                                          lumen_open, Uk=Uk, Rr=Rr, tr=tr),
                penetration=penetration_block(run, S, sol, parts, lumen, outer, phase),
                wall=wall_block(run, step, st, sol),
                oar=oar_block(run, S, parts, outer),
                corpus=corpus_block(run, dev, Xk, path_c, xk_exact))


# ============================================================================================ final-state blocks
def bt_label_frame(cx, app_new):
    """The updated applicator label's tandem frame (see FRAMES)."""
    bt = cx.bt
    Fj, zj = bt.F, unit(bt.a)
    W = mask_pts(app_new & ~bt.lab["ovoid"], bt.aff)
    q = W - Fj
    s = q @ zj
    lat = np.linalg.norm(q - np.outer(s, zj), axis=1)
    cen = []
    lo, hi = LABEL_LINE["s"]
    for z in np.arange(lo, hi, LABEL_LINE["slab"]):
        m = (s >= z) & (s < z + LABEL_LINE["slab"]) & (lat < LABEL_LINE["lat"])
        if m.sum() > 3:
            cen.append(W[m].mean(0))
    cen = np.array(cen)
    c, ax = line_fit(cen)
    ax = ax * np.sign(ax @ zj)
    F = c + ((Fj - c) @ ax) * ax
    x = geom.ortho(bt.R[0], ax)
    qq = cen - c
    rms = float(np.sqrt((np.linalg.norm(qq - np.outer(qq @ ax, ax), axis=1) ** 2).mean()))
    return dict(F=F, R=np.array([x, np.cross(ax, x), ax]), n_slabs=int(len(cen)), rms_mm=rms,
                angle_to_json_axis_deg=ang(ax, zj))


def portio_table(L):
    """Lowest cervix height (p1 of h, mm) by quadrant (R, A, L, P about +x = patient right, +y = anterior) and 5 mm
    radial band 0-25 mm about the tube (vjm/v08_seating.table)."""
    r = np.hypot(L[:, 0], L[:, 1])
    th = np.degrees(np.arctan2(L[:, 1], L[:, 0]))
    q = dict(R=np.abs(th) < 45, A=(th >= 45) & (th < 135), L=np.abs(th) >= 135, P=(th <= -45) & (th > -135))
    out = {}
    for k, m in q.items():
        out[k] = [rnd(np.percentile(L[m & (r >= r0) & (r < r0 + 5), 2], 1), 2)
                  if (m & (r >= r0) & (r < r0 + 5)).sum() > 5 else None for r0 in range(0, 25, 5)]
    return out


def fornix(Lv, Lc, rmin=5.0, rmax=30.0):
    """Fornix depth per 45 deg sector: vault top (p99.5 of the vagina's h) minus portio bottom (p0.5 of the cervix's
    h) over r 5-30 mm (vjm/v06_g32_vault.fornix)."""
    out = {}
    for k, nm in enumerate(("R", "AR", "A", "AL", "L", "PL", "P", "PR")):
        c = np.radians(45 * k)
        res = []
        for L_ in (Lv, Lc):
            r = np.hypot(L_[:, 0], L_[:, 1])
            dth = np.angle(np.exp(1j * (np.arctan2(L_[:, 1], L_[:, 0]) - c)))
            res.append(L_[(np.abs(dth) < np.radians(22.5)) & (r >= rmin) & (r <= rmax), 2])
        if len(res[0]) and len(res[1]):
            top, bot = np.percentile(res[0], 99.5), np.percentile(res[1], 0.5)
            out[nm] = dict(depth=rnd(top - bot, 2), vault_top=rnd(top, 2), portio_bottom=rnd(bot, 2))
    return out


def junction_tables(Lc, Lv, vcc):
    """Own-flange (or BT label) frame junction measures of a cervix and a (filled) vagina point cloud."""
    i = int(np.argmin(Lc[:, 2]))
    slabs = []
    for h0 in np.arange(-30.0, 15.0, 3.0):
        m = (Lv[:, 2] >= h0) & (Lv[:, 2] < h0 + 3)
        if m.sum() >= 20:
            P = Lv[m]
            slabs.append(dict(h=[h0, h0 + 3], n=int(m.sum()), LR=rnd(np.ptp(P[:, 0]), 1), AP=rnd(np.ptp(P[:, 1]), 1),
                              LR_p3_97=rnd(np.percentile(P[:, 0], 97) - np.percentile(P[:, 0], 3), 1),
                              AP_p3_97=rnd(np.percentile(P[:, 1], 97) - np.percentile(P[:, 1], 3), 1)))
    return dict(portio_p1_h_by_quadrant_radial_band=portio_table(Lc),
                radial_bands_mm=[[r0, r0 + 5] for r0 in range(0, 25, 5)],
                cervix_vol_below_flange_cc=rnd((Lc[:, 2] < 0).sum() * vcc, 2),
                lowest_cervix=dict(h=rnd(Lc[i, 2], 2), x=rnd(Lc[i, 0], 1), y=rnd(Lc[i, 1], 1),
                                   p1_h=rnd(np.percentile(Lc[:, 2], 1), 2)),
                vault_top=dict(p99_5=rnd(np.percentile(Lv[:, 2], 99.5), 2), max=rnd(Lv[:, 2].max(), 2)),
                slabs_3mm=slabs, fornix_depth=fornix(Lv, Lc))


def final_junction(run, cx, lab):
    """Own-flange frame (model) and BT label frame (BT); 1 mm grid voxelisation (voxel centres)."""
    df = run.devfin
    F = np.asarray(first(df, ("flange_mm", "flange")), float)
    R = np.array([df["x_app"], df["y_app"], df["tube_axis"]], float)
    Vc, Fc = geom.read_obj(run.rd + "/final/cervix.obj")
    Vv, Fv = geom.read_obj(run.rd + "/final/vagina.obj")
    if run.wall is not None:
        Vv, Fv, _ = close_sheet(Vv, Fv, run.wall.outer_v)
    lo = np.minimum(Vc.min(0), Vv.min(0))
    hi = np.maximum(Vc.max(0), Vv.max(0))
    aff, shape = box_grid(lo, hi, 1.0)
    mc = grid_mask([(Vc, Fc)], aff, shape)
    mv = grid_mask([(Vv, Fv)], aff, shape)
    Lc = (mask_pts(mc, aff) - F) @ R.T
    Lv = (mask_pts(mv, aff) - F) @ R.T
    model = junction_tables(Lc, Lv, 1e-3)
    model["cervix_and_filled_vagina_cc"] = rnd((mc & mv).sum() * 1e-3, 2)
    dev = {}
    for p in run.devsurf:
        dm = grid_mask([run.place(p, df)], aff, shape)
        dev[p] = rnd((mc & dm).sum() * 1e-3, 2)
    model["cervix_and_device_cc"] = dev
    model["frame"] = "own-flange: rows (x_app, y_app, tube_axis) of device_final about flange_mm; 1 mm grid"
    ring = [p for p in run.ovoid_parts if p.startswith("ovoid")]
    if ring:
        zr = np.vstack([np.asarray(run.devsurf[p][0], float) for p in ring])[:, 2]
        model["ring_top_z_app_mm"] = rnd(zr.max(), 2)
    # BT in the BT label frame
    bt = cx.bt
    fr = lab["frame"]
    to = lambda m: (mask_pts(m, bt.aff) - fr["F"]) @ fr["R"].T  # noqa: E731
    cerv = bt.lab["HR-CTV"] & ~bt.lab["uterus"]
    btj = junction_tables(to(cerv), to(bt.lab["vagina"]), bt.vv / 1000.0)
    btj["cervix_and_vagina_cc"] = rnd((cerv & bt.lab["vagina"]).sum() * bt.vv / 1000.0, 2)
    btj["cervix_and_device_cc"] = rnd((cerv & (lab["app_new"] | bt.lab["ovoid"])).sum() * bt.vv / 1000.0, 2)
    btj["ovoid_label_top_p99_5_h"] = rnd(np.percentile(to(bt.lab["ovoid"])[:, 2], 99.5), 2)
    btj["frame"] = ("BT label frame (updated applicator label's tandem, s 15-55 slab line); NOTE the BT vagina above "
                    "F-12 is the ovoid label (fix plan section 1), so vault widths there are not a vaginal wall")
    return dict(model=model, BT=btj, tag="PREDICTED given a CALIBRATED device pose (Delta in-sample)")


def pelvis_grid(run):
    """The (h, u, v) grid of the vagina slabs: h along the shaft axis from the run's flange, u = patient right,
    v = anterior (normal to the axis), as an affine (i -> h, j -> u, k -> v)."""
    df = run.devfin
    F0 = np.asarray(first(df, ("flange_mm", "flange")), float)
    a = unit(first(run.pose.get("device_final", {}), ("shaft_axis",), first(df, ("shaft_axis",), df["tube_axis"])))
    eu = geom.ortho(np.array([1.0, 0.0, 0.0]), a)
    evv = np.cross(a, eu)
    evv = evv if evv[1] >= 0 else -evv
    h0, h1 = VAG_GRID["h"]
    dh, du, U = VAG_GRID["dh"], VAG_GRID["duv"], VAG_GRID["uv"]
    shape = (int(round((h1 - h0) / dh)) + 1, int(round(2 * U / du)) + 1, int(round(2 * U / du)) + 1)
    aff = np.eye(4)
    aff[:3, 0], aff[:3, 1], aff[:3, 2] = a * dh, eu * du, evv * du
    aff[:3, 3] = F0 + h0 * a - U * eu - U * evv
    hs = h0 + dh * np.arange(shape[0])
    us = -U + du * np.arange(shape[1])
    return aff, shape, hs, us, dict(F0=F0, a=a, eu=eu, ev=evv)


def slab_stats(m, hs, us, du, h0, w=5.0):
    sel = (hs >= h0) & (hs < h0 + w)
    mm = m[sel]
    if not mm.any():
        return None
    idx = np.argwhere(mm)
    uu, vv = us[idx[:, 1]], us[idx[:, 2]]
    return dict(area_mm2=rnd(mm.sum() * du * du / sel.sum(), 0), cu=rnd(uu.mean(), 1), cv=rnd(vv.mean(), 1),
                LR=rnd(np.percentile(uu, 99) - np.percentile(uu, 1), 1),
                AP=rnd(np.percentile(vv, 99) - np.percentile(vv, 1), 1))


def final_vagina(run, cx, lab, masks):
    """Pelvis frame: 5 mm slabs by region, and filled Dice / volume against both BT references."""
    out = dict(tag="PREDICTED")
    bt = cx.bt
    if "vagina_filled" in masks:
        mf = masks["vagina_filled"]
        old = cx.ref["vagina+device"]
        new = bt.lab["vagina"] | lab["app_new"] | bt.lab["ovoid"]
        out["filled_dice_BT_grid"] = dict(
            old_reference=dice(mf, old), updated_reference=dice(mf, new),
            model_cc=rnd(mf.sum() * bt.vv / 1000.0, 2), old_reference_cc=rnd(old.sum() * bt.vv / 1000.0, 2),
            updated_reference_cc=rnd(new.sum() * bt.vv / 1000.0, 2),
            definition="BT grid, PELVIS frame: the solid inside the wall's outer sheet (eval_hybrid.wall_outer_solid) "
                       "vs "
                       "BT vagina | old applicator (old) and vagina | updated applicator | ovoid (updated)")
    if run.wall is None:
        return out
    aff, shape, hs, us, gfr = pelvis_grid(run)
    Vv, Fv = geom.read_obj(run.rd + "/final/vagina.obj")
    Vo, Fo, _ = close_sheet(Vv, Fv, run.wall.outer_v)
    Vi, Fi, _ = close_sheet(Vv, Fv, run.wall.inner_v)
    mo = grid_mask([(Vo, Fo)], aff, shape)
    mi = grid_mask([(Vi, Fi)], aff, shape)
    fmap = cx.pelvis_map()[0]
    Xg = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).reshape(-1, 3) @ aff[:3, :3].T \
        + aff[:3, 3]
    Xb = fmap(Xg)
    mb = sample_nn(bt.lab["vagina"], bt.aff, Xb).reshape(shape)
    mbd = sample_nn(bt.lab["vagina"] | lab["app_new"] | bt.lab["ovoid"], bt.aff, Xb).reshape(shape)
    du = VAG_GRID["duv"]
    cell = VAG_GRID["dh"] * du * du / 1000.0
    rows = []
    for h0 in np.arange(VAG_GRID["h"][0], VAG_GRID["h"][1], 5.0):
        reg = next((n for n, a_, b_ in VAG_REGIONS if a_ <= h0 + 2.5 < b_), None)
        rows.append(dict(h=[h0, h0 + 5], region=reg, model_filled=slab_stats(mo, hs, us, du, h0),
                         model_lumen=slab_stats(mi, hs, us, du, h0), BT_vagina=slab_stats(mb, hs, us, du, h0),
                         BT_vagina_device=slab_stats(mbd, hs, us, du, h0)))
    out["slabs_5mm"] = rows
    out["grid_dice"] = dict(filled_vs_BT_vagina_device=dice(mo, mbd), filled_vs_BT_vagina=dice(mo, mb),
                            model_filled_cc=rnd(mo.sum() * cell, 1), model_lumen_cc=rnd(mi.sum() * cell, 1),
                            BT_vagina_cc=rnd(mb.sum() * cell, 1), BT_vagina_device_cc=rnd(mbd.sum() * cell, 1))
    out["frame"] = ("PELVIS: h along the run's shaft axis (pose.json device_final.shaft_axis) from the final flange, "
                    "u = patient right, v = anterior; grid %.2f mm in-plane, %.1f mm along h; regions %s"
                    % (du, VAG_GRID["dh"], [list(r) for r in VAG_REGIONS]))
    out["grid_axes"] = dict(origin_flange=rnd(gfr["F0"], 3), h_axis=rnd(gfr["a"], 5), u_axis=rnd(gfr["eu"], 5),
                            v_axis=rnd(gfr["ev"], 5))
    return out


def lr_side(P, O, a, vcc, w=None):
    """Lateral x-offset (+ = patient right) of P about the line O + s a; mean, right / left cc (weighted by w cc if
    given, else vcc per point)."""
    a = unit(a)
    q = P - O
    s = q @ a
    x = (q - np.outer(s, a))[:, 0]
    w = np.full(len(P), vcc) if w is None else w
    return x, s, dict(mean_x_mm=rnd((x * w).sum() / w.sum(), 2), right_cc=rnd(w[x > 0].sum(), 2),
                      left_cc=rnd(w[x <= 0].sum(), 2))


def final_hrctv(run, cx, lab, masks):
    """Cervix (HR-CTV minus uterus) left/right split about the model tube (tet-volume weighted final cervix), per
    level, its bottom, the BT and preBT references (both BT tandems; the physician path, S3), and the Dice triple."""
    df = run.devfin
    F = np.asarray(first(df, ("flange_mm", "flange")), float)
    a = unit(df["tube_axis"])
    X = run.Pc + np.load(run.rd + "/final/cervix_u.npy")
    Tc = run.Tc
    cen = X[Tc].mean(1)
    vol = np.abs(np.einsum("ij,ij->i", X[Tc[:, 1]] - X[Tc[:, 0]],
                           np.cross(X[Tc[:, 2]] - X[Tc[:, 0]], X[Tc[:, 3]] - X[Tc[:, 0]]))) / 6000.0   # cc
    x, s, tot = lr_side(cen, F, a, None, vol)
    lev = []
    for s0 in np.arange(-30.0, 45.0, 5.0):
        m = (s >= s0) & (s < s0 + 5)
        if vol[m].sum() > 0.1:
            lev.append(dict(h=[s0, s0 + 5], cc=rnd(vol[m].sum(), 2),
                            mean_x_mm=rnd((x * vol)[m].sum() / vol[m].sum(), 2)))
    m10 = (s >= 0) & (s < 10)
    Vc = geom.read_obj(run.rd + "/final/cervix.obj")[0]
    R = np.array([df["x_app"], df["y_app"], df["tube_axis"]], float)
    hc = (Vc - F) @ R[2]
    out = dict(tag="PREDICTED (BT references: REFERENCE)",
               model_vs_model_tube=dict(tot, definition="final cervix tets (meshes/cervix + final/cervix_u.npy), "
                                                        "tet-volume weighted, x-component of the offset normal to "
                                                        "the model tube (device_final)"),
               model_per_level=lev,
               model_lowest_10mm_above_flange_mean_x_mm=rnd((x * vol)[m10].sum() / vol[m10].sum(), 2)
               if vol[m10].sum() > 0 else None,
               model_cervix_bottom_h_mm=dict(min=rnd(hc.min(), 2), p1=rnd(np.percentile(hc, 1), 2),
                                             note="surface vertices, height along the tube above the flange"))
    ring = [p for p in run.ovoid_parts if p.startswith("ovoid")]
    if ring:
        zr = float(np.vstack([np.asarray(run.devsurf[p][0], float) for p in ring])[:, 2].max())
        out["model_cervix_bottom_minus_ring_top_mm"] = rnd(hc.min() - zr, 2)
    # BT references
    bt = cx.bt
    cerv = bt.lab["HR-CTV"] & ~bt.lab["uterus"]
    Pb = mask_pts(cerv, bt.aff)
    ng = json.load(open(NEW_GEOM_JSON))
    xb, sb, r_json = lr_side(Pb, bt.F, bt.a, bt.vv / 1000.0)
    _, _, r_new = lr_side(Pb, np.asarray(ng["tandem"]["tip"], float), np.asarray(ng["tandem"]["axis"], float),
                          bt.vv / 1000.0)
    bl = []
    for s0 in np.arange(-10.0, 45.0, 5.0):
        m = (sb >= s0) & (sb < s0 + 5)
        if m.sum() > 20:
            bl.append(dict(h=[s0, s0 + 5], cc=rnd(m.sum() * bt.vv / 1000.0, 2), mean_x_mm=rnd(xb[m].mean(), 2)))
    out["BT_reference"] = dict(vs_json_tandem=r_json, vs_updated_label_tandem=r_new, per_level_json_tandem=bl,
                               note="BT frame: + = BT world x (patient right)")
    # preBT references (the model's rest state): the physician path (S3) and canal.npz
    hr = nib.load(DATA + "/preBT_MRI_label_HR-CTV.nii")
    A = hr.affine
    ut = np.asarray(nib.load(DATA + "/preBT_MRI_label_uterus.nii").dataobj) > 0
    Pp = mask_pts((np.asarray(hr.dataobj) > 0) & ~ut, A)
    vcc = float(abs(np.linalg.det(A[:3, :3]))) / 1000.0
    pts = run.T["pts"]
    o = np.argsort(pts[:, 2])
    xz = Pp[:, 0] - np.interp(Pp[:, 2], pts[o, 2], pts[o, 0])
    _, _, cp = TP.project(Pp, pts, run.T["s"])
    xl = Pp[:, 0] - cp[:, 0]
    cz = np.load(INP + "/canal.npz")
    O_pre = np.asarray(cz["O_pre"], float)
    _, _, r_npz = lr_side(Pp, O_pre, np.asarray(cz["pts"], float)[int(cz["i_internal_os"])] - O_pre, vcc)
    side = lambda v: dict(mean_x_mm=rnd(v.mean(), 2), right_cc=rnd((v > 0).sum() * vcc, 2),  # noqa: E731
                          left_cc=rnd((v <= 0).sum() * vcc, 2))
    out["preBT_reference"] = dict(vs_physician_path_xz=side(xz), vs_physician_path_local=side(xl),
                                  vs_canal_npz_line=r_npz,
                                  note="path_xz: voxel x minus the path's x at the voxel's z (S3 / vcf v6); local: "
                                       "x of the offset to the closest path point; canal.npz line: O_pre -> internal "
                                       "os (the old reference, not to be quoted)")
    # Dice triple (PELVIS frame, BT grid)
    mc = masks["cervix"]
    A_ = lab["app_new"] | bt.lab["ovoid"]
    refs = dict(HRCTV_minus_U=cerv, HRCTV_minus_U_minus_V=cerv & ~bt.lab["vagina"],
                HRCTV_minus_U_minus_V_or_A=cerv & ~(bt.lab["vagina"] | A_))
    out["cervix_dice_PELVIS"] = {k: dict(dice=dice(mc, r), ref_cc=rnd(r.sum() * bt.vv / 1000.0, 2))
                                 for k, r in refs.items()}
    out["cervix_dice_PELVIS"]["model_cc"] = rnd(mc.sum() * bt.vv / 1000.0, 2)
    out["cervix_bounds"] = REFERENCE["cervix_bounds"]
    return out


def final_organs(run, cx, masks):
    """Rectum on the preBT grid (vom/v6: BT label resampled nearest-voxel through PELVIS) on its common z-extent and in
    bands; rectosigmoid union; bladder and sigmoid on the BT grid; device-to-OAR (eval_hybrid definition)."""
    bt = cx.bt
    pre_aff, pre_shape = cx.pre_aff, cx.pre_shape
    fmap, Rp, tp = cx.pelvis_map()
    lab = {}
    for n in ("rectum", "sigmoid"):
        lab[n] = np.asarray(nib.load("%s/preBT_MRI_label_%s.nii" % (DATA, n)).dataobj) > 0.5
    mod = {n: ev.voxelize(*run.rest_final[n], pre_shape, pre_aff) for n in ("rectum", "sigmoid")}
    # the box that holds every mask involved
    inv = np.linalg.inv(pre_aff)
    boxes = []
    for n in ("rectum", "sigmoid"):
        for m in (lab[n], mod[n]):
            if m.any():
                i = np.argwhere(m)
                boxes += [i.min(0), i.max(0)]
        Pb = mask_pts(bt.lab[n], bt.aff) @ Rp.T + tp
        I = Pb @ inv[:3, :3].T + inv[:3, 3]
        boxes += [np.floor(I.min(0)), np.ceil(I.max(0))]
    B = np.array(boxes)
    lo = np.maximum(B.min(0).astype(int) - 2, 0)
    hi = np.minimum(B.max(0).astype(int) + 3, np.array(pre_shape))
    g = np.stack(np.meshgrid(*[np.arange(lo[k], hi[k]) for k in range(3)], indexing="ij"), -1)
    sub = tuple(slice(lo[k], hi[k]) for k in range(3))
    Xw = g.reshape(-1, 3) @ pre_aff[:3, :3].T + pre_aff[:3, 3]
    Xb = fmap(Xw)
    btm = {n: sample_nn(bt.lab[n], bt.aff, Xb).reshape(g.shape[:3]) for n in ("rectum", "sigmoid")}
    Z = Xw[:, 2].reshape(g.shape[:3])
    pre = {n: lab[n][sub] for n in lab}
    mm = {n: mod[n][sub] for n in mod}
    vv = float(abs(np.linalg.det(pre_aff[:3, :3]))) / 1000.0
    r = dict(whole=dict(model=dice(mm["rectum"], btm["rectum"]), no_motion=dice(pre["rectum"], btm["rectum"])))
    for nm, z0, z1 in RECTUM_BANDS:
        b = (Z >= z0) & (Z < z1)
        r[nm] = dict(model=dice(mm["rectum"] & b, btm["rectum"] & b),
                     no_motion=dice(pre["rectum"] & b, btm["rectum"] & b),
                     model_cc=rnd((mm["rectum"] & b).sum() * vv, 2), BT_cc=rnd((btm["rectum"] & b).sum() * vv, 2))
    zp, zb = Z[pre["rectum"]], Z[btm["rectum"]]
    zlo, zhi = max(zp.min(), zb.min()), min(zp.max(), zb.max())
    b = (Z >= zlo) & (Z <= zhi)
    r["common_z_extent"] = dict(z=[rnd(zlo, 1), rnd(zhi, 1)], model=dice(mm["rectum"] & b, btm["rectum"] & b),
                                no_motion=dice(pre["rectum"] & b, btm["rectum"] & b))
    rs = dict(model=dice(mm["rectum"] | mm["sigmoid"], btm["rectum"] | btm["sigmoid"]),
              no_motion=dice(pre["rectum"] | pre["sigmoid"], btm["rectum"] | btm["sigmoid"]))
    dev = masks.get("device")
    d2o = {o: rnd(EH.mask_min_dist(dev, masks[o], bt.sp), 3) for o in ORGANS} if dev is not None else None
    return dict(tag="PREDICTED (sigmoid: NOT PREDICTED)",
                rectum_preBT_grid=r, rectosigmoid_preBT_grid=rs,
                bladder_dice_BT_grid=dice(masks["bladder"], cx.ref["bladder"]),
                sigmoid_BT_grid=dict(dice=dice(masks["sigmoid"], cx.ref["sigmoid"]),
                                     flag="not predicted; model motion is an artefact of the corpus schedule"),
                device_to_OAR_min_mm=dict(model=d2o, BT_reference={o: rnd(cx.bt_dev_oar[o], 3) for o in ORGANS},
                                          definition="eval_hybrid: device mask vs OAR mask on the BT grid, PELVIS"),
                oracle_ceilings=REFERENCE["oracle_ceilings"])


def final_device(run, cx, lab):
    """Device against the updated label at the BT pose (applicator.json BT_pose, BT grid) and the model tube against
    both BT tandem references (PELVIS)."""
    bt = cx.bt
    out = dict(tag="CALIBRATED (in-sample geometry fit: a consistency ceiling, not validation)")
    bp = run.appj.get("BT_pose")
    if bp:
        Or = np.asarray(bp["origin_BT_world"], float)
        Rb = np.asarray(bp["R_rows_BT_world"], float)
        parts = [p for p in run.devsurf if p != "packing"]
        m = grid_mask([(Or + np.asarray(run.devsurf[p][0], float) @ Rb, run.devsurf[p][1]) for p in parts],
                      bt.aff, bt.shape)
        new = lab["app_new"] | bt.lab["ovoid"]
        out.update(dice_vs_updated_applicator_or_ovoid=dice(m, new),
                   dice_vs_old_applicator_or_ovoid=dice(m, bt.lab["applicator"] | bt.lab["ovoid"]),
                   device_cc=rnd(m.sum() * bt.vv / 1000.0, 2), parts=parts,
                   note="parts minus packing at applicator.json BT_pose; BT voxel centres (stencil)")
    lc = run.appj.get("label_check")
    out["label_check"] = lc if lc is not None else "absent (S4c writes applicator.json['label_check'])"
    df = run.devfin
    fmap = cx.pelvis_map()[0]
    F = np.asarray(first(df, ("flange_mm", "flange")), float)
    a = unit(df["tube_axis"])
    tip = np.asarray(first(df, ("tip_mm", "tip"), F + run.L_iu * a), float)
    Fb, tb = fmap(F[None])[0], fmap(tip[None])[0]
    ab = unit(tb - Fb)
    ng = json.load(open(NEW_GEOM_JSON))
    fr = lab["frame"]
    out["tube_vs_BT"] = dict(
        angle_to_json_axis_deg=rnd(ang(ab, bt.a), 2), angle_to_label_line_deg=rnd(ang(ab, fr["R"][2]), 2),
        angle_to_label_geometry_axis_deg=rnd(ang(ab, ng["tandem"]["axis"]), 2),
        tip_error_vs_label_tip_mm=rnd(np.linalg.norm(tb - np.asarray(ng["tandem"]["tip"], float)), 2),
        tip_error_vs_json_tip_mm=rnd(np.linalg.norm(tb - bt.tip), 2),
        label_line=dict(n_slabs=fr["n_slabs"], rms_mm=rnd(fr["rms_mm"], 2),
                        angle_to_json_axis_deg=rnd(fr["angle_to_json_axis_deg"], 2),
                        definition="2 mm slab centroids of (updated applicator minus ovoid), s 15-55 mm above the json "
                                   "flange, voxels within 6 mm of the json axis; PCA line"),
        frame="PELVIS (model tube mapped x_BT = R^T (y - t))")
    return out


def run_status(run):
    """Per-step motion trend over the last 10 logged steps: a run is not labelled final while any organ or the
    cervix still moves faster step on step."""
    steps = sorted(run.log)[-10:]
    out = {}
    for b in ("cervix", "bladder", "rectum", "sigmoid", "vagina") + (("corpus",) if run.corpus_fem else ()):
        dx = np.array([((run.log[k].get("disp") or {}).get(b) or {}).get("dx", np.nan) for k in steps], float)
        ok = np.isfinite(dx)
        if ok.sum() >= 3:
            slope = float(np.polyfit(np.arange(len(dx))[ok], dx[ok], 1)[0])
            out[b] = dict(dx_last_mm=rnd(dx[ok][-1], 4), slope_mm_per_step=rnd(slope, 5),
                          rising=bool(slope > 0 and dx[ok][-1] > float(run.cfg.get("conv_dx_mm", 0.02))))
    tissue = [b for b in ("cervix", "bladder", "rectum", "sigmoid") + (("corpus",) if run.corpus_fem else ())
              if out.get(b, {}).get("rising")]
    # cost and element health per phase (S6: <= 12 s/step, cervix min volume ratio >= 0.6, no inverted tets)
    per_phase = {}
    for k in sorted(run.log):
        r = run.log[k]
        ph = per_phase.setdefault(r.get("phase", "?"), dict(n=0, s=[], contacts=[], mvr={}))
        ph["n"] += 1
        if r.get("wall_ms") is not None:
            ph["s"].append(float(r["wall_ms"]) / 1000.0)
        if r.get("n_contacts") is not None:
            ph["contacts"].append(int(r["n_contacts"]))
        for b, d in (r.get("disp") or {}).items():
            if isinstance(d, dict) and d.get("min_vol_ratio") is not None:
                ph["mvr"][b] = min(ph["mvr"].get(b, 9.0), float(d["min_vol_ratio"]))
    perf = {p: dict(steps=v["n"], s_per_step_mean=rnd(np.mean(v["s"]), 2) if v["s"] else None,
                    s_per_step_max=rnd(np.max(v["s"]), 2) if v["s"] else None,
                    contacts_mean=rnd(np.mean(v["contacts"]), 0) if v["contacts"] else None,
                    min_vol_ratio=v["mvr"]) for p, v in per_phase.items()}
    mvr = {}
    for v in per_phase.values():
        for b, x in v["mvr"].items():
            mvr[b] = min(mvr.get(b, 9.0), x)
    gates = run.summary.get("gates") or {}
    return dict(per_body=out, rising_tissue=tissue, label="final" if not tissue else "NOT final (motion rising)",
                wall_rising=bool(out.get("vagina", {}).get("rising")),
                summary_status=run.summary.get("status"), converged=run.summary.get("converged"),
                inverted_tets=gates.get("inverted_tets"), nan=gates.get("nan"),
                min_vol_ratio_run=mvr, per_phase=perf)


def carry_validation(run):
    """Harmonic extension of final/cervix.obj vs the exact final/cervix_u.npy: interior nodes and canal points."""
    p = run.rd + "/final/cervix_u.npy"
    if not os.path.exists(p):
        return None
    Uf = np.load(p)
    Uh = run.carrier.cervix_U(geom.read_obj(run.rd + "/final/cervix.obj")[0])
    e = np.linalg.norm(Uh - Uf, axis=1)
    ei = e[run.carrier.inn]
    Rk, tk = geom.kabsch(run.rest["corpus"][0], geom.read_obj(run.rd + "/final/corpus.obj")[0])
    X = run.T["pts"]
    Uk = np.load(run.final_uk_path) if (run.corpus_fem and run.final_uk_path) else None     # S9: the same in both carries
    ep = np.linalg.norm(run.carrier.carry(run.loc_path, X, Uh, Rk, tk, Uk) -
                        run.carrier.carry(run.loc_path, X, Uf, Rk, tk, Uk), axis=1)
    cerv = np.array([L[0] == "c" for L in run.loc_path])
    st = lambda v: dict(median=rnd(np.median(v)), max=rnd(v.max()), n=int(len(v))) if len(v) else None  # noqa: E731
    ties = run.tie_nodes
    return dict(interior_nodes_mm=st(ei),
                path_points_in_cervix_mm=st(ep[cerv]), canal_points_in_cervix_mm=st(ep[cerv & (run.T["s"] >= 0)]),
                tie_set_nodes_mm=st(e[ties]) if len(ties) else None,
                canal_path_nodes_mm=st(e[run.cp_nodes]) if len(run.cp_nodes) else None,
                path_carried_by=run.path_kinds,
                note="harmonic extension of final/cervix.obj vs the exact final/cervix_u.npy; corpus-carried points "
                     "are exact (rigid; 'k' points of an elastic corpus: final/corpus_u.npy barycentrically); 'v' points "
                     "(vaginal side, outside the cervix) stay at rest",
                audit_reference="vcm/v4_frames.py: median 0.56, max 0.94 mm")


def sim_model(run, cx):
    """eval_hybrid.model_sim for this run, but with the run's own (filtered) device parts, so the masks carry
    exactly the parts the run used."""
    dev = EH.read_device_final(run.rd + "/device_final.json", cx.bt.L)
    if run.ovoid_parts and run.devfin.get("ovoid_origin_mm") is not None:
        dev["ov_origin"] = run.devfin["ovoid_origin_mm"]
        if "ovoid_axis" in run.devfin:
            dev["ov_R_rows"] = [run.devfin["ovoid_x_app"], run.devfin["ovoid_y_app"], run.devfin["ovoid_axis"]]
    surfs = {b: run.rest_final.get(b, run.rest[b]) for b in EH.BODIES}
    return EH.Model("SIM", surfs, dev, meta=dict(devsurf=run.devsurf,
                                                 wall_meta=run.meta["vagina"] if run.wall is not None else None))


def _pose_vs_target(run, F, a):
    """The run's final tube against the validated target (<applicator_dir>/pose.json device_final)."""
    t = run.pose.get("device_final") or {}
    if not t:
        return None
    Ft = np.asarray(first(t, ("flange", "flange_mm")), float)
    return dict(flange_mm=rnd(np.linalg.norm(F - Ft)), axis_deg=rnd(ang(a, first(t, ("tube_axis", "axis"))), 3),
                target="%s/pose.json device_final" % os.path.relpath(run.appd, HYB).replace("\\", "/"))


def score_final(run, cx):
    df = run.devfin
    run.rest_final = {b: geom.read_obj(run.rd + "/final/%s.obj" % b) for b in EH.BODIES
                      if os.path.exists(run.rd + "/final/%s.obj" % b)}
    masks = EH.model_masks(cx, sim_model(run, cx), cx.pelvis_map()[0])
    im = nib.load(APP_LABEL_NEW)
    if im.shape != cx.bt.shape or not np.allclose(im.affine, cx.bt.aff, atol=1e-3):
        raise SystemExit("updated applicator label %s is not on the BT grid" % APP_LABEL_NEW)
    lab = dict(app_new=np.asarray(im.dataobj) > 0.5)
    lab["frame"] = bt_label_frame(cx, lab["app_new"])
    out = dict(junction=final_junction(run, cx, lab), vagina=final_vagina(run, cx, lab, masks),
               hrctv=final_hrctv(run, cx, lab, masks), organs=final_organs(run, cx, masks),
               device_vs_label=final_device(run, cx, lab), carry_validation=carry_validation(run))
    # the exact final carried canal (final/cervix_u.npy) beside the per-frame harmonic one
    Uf = np.load(run.rd + "/final/cervix_u.npy")
    Rk, tk = geom.kabsch(run.rest["corpus"][0], run.rest_final["corpus"][0])
    Ukf = np.load(run.final_uk_path) if run.final_uk_path else None           # every run writes final/corpus_u.npy
    pc = run.carrier.carry(run.loc_path, run.T["pts"], Uf, Rk, tk, Ukf if run.corpus_fem else None)
    F = np.asarray(first(df, ("flange_mm", "flange")), float)
    a = unit(df["tube_axis"])
    tip = np.asarray(first(df, ("tip_mm", "tip"), F + run.L_iu * a), float)
    canal = pc[run.ic]
    dseg, h = seg_dist(canal, F, a, run.L_iu)
    rt = run.r_tube_app or 2.18
    sF = float(run.T["s_F"])
    seat = (run.T["s"] >= sF) & (run.T["s"] <= 20.0)
    dP, _ = seg_dist(pc[seat], F, a, run.L_iu)
    out["canal_final_exact"] = dict(
        tip_to_labelled_canal_mm=rnd(TP.project(tip[None], canal, run.canal_s)[0][0]),
        traversed_canal_in_tube_frac=frac((dseg <= rt + TUBE_TOL_MM)[(h >= 0) & (h <= run.L_iu)]),
        upper_canal_residual_mm=rnd(dseg[run.canal_s > LOWER_CANAL_MM].max()),
        path_sF_to_Otrue_plus20_max_dist_to_tube_mm=rnd(dP.max()),
        O_true_height_above_flange_mm=rnd((pc[run.T["i_os"]] - F) @ a, 2),
        uterus_dice_PELVIS=dice(masks["corpus"], cx.ref["corpus"]),
        tube_pose_vs_target=_pose_vs_target(run, F, a),
        definition="final/cervix_u.npy (exact) instead of the harmonic extension; S6 final gate: the path from s_F "
                   "to O_true + 20 mm within 3.2 mm of the tube")
    # S9: the final corpus (final/corpus_u.npy: exact for both models) against the schedule's target, its non-rigid part,
    # volume, span along the tube, the carried canal by band and its straightness (see corpus_block)
    Xk = run.Pk + Ukf if Ukf is not None else run.Pk @ Rk.T + tk
    out["corpus"] = corpus_block(run, df, Xk, pc, Ukf is not None)
    out["corpus"]["uterus_dice_PELVIS"] = out["canal_final_exact"]["uterus_dice_PELVIS"]
    return out


# ============================================================================================ summaries
def summarize(run, rows):
    """Run-level aggregates of the per-frame series (the numbers the S0 / S5 / S6 gates read)."""
    if not rows:
        return {}
    g = lambda r, *k: _dig(r, k)  # noqa: E731
    s = {}
    t = [(r["step"], r["u"], g(r, "insertion", "tip_to_labelled_canal_mm")) for r in rows]
    tv = [x for x in t if x[2] is not None]
    if tv:
        i = int(np.argmin([x[2] for x in tv]))
        s["tip_to_labelled_canal"] = dict(min_mm=tv[i][2], at_step=tv[i][0], at_u=tv[i][1], final_mm=tv[-1][2])
    tr = [(r["step"], r["u"], g(r, "insertion", "traversed_canal_in_tube_frac")) for r in rows]
    nz = [x for x in tr if x[2] is not None and x[2] > 0]
    s["traversed_canal_in_tube"] = dict(
        final=tr[-1][2], first_nonzero_step=nz[0][0] if nz else None, first_nonzero_u=nz[0][1] if nz else None,
        zero_through_u=max([x[1] for x in tr if x[1] is not None and (not nz or x[0] < nz[0][0])] or [None]))
    tp = [(r["step"], r["phase"], g(r, "insertion", "tip_to_plan_mm")) for r in rows]
    tpc = [x[2] for x in tp if x[1] == "C" and x[2] is not None]
    s["tip_to_plan"] = dict(max_C_mm=max(tpc) if tpc else None, n_C_frames=len(tpc),
                            max_all_mm=max([x[2] for x in tp if x[2] is not None] or [None]))
    tt = [(r["phase"], g(r, "insertion", "tip_to_tau_mm")) for r in rows]
    ttc = [x[1] for x in tt if x[0] == "C" and x[1] is not None]
    s["tip_to_tau"] = dict(max_C_mm=max(ttc) if ttc else None, final_mm=tt[-1][1], source=run.tau_src,
                           note="distance to the carried tau curve (no delta, no planned depth): a lower bound of "
                                "tip_to_plan")
    # phase L (TF0 lift): tandem and corpus move as one body -> the tip and flange are fixed in the corpus rest frame
    Lr = [r for r in rows if r["phase"] == "L"]
    if Lr:
        tc = np.array([g(r, "insertion", "tip_in_corpus_rest") for r in Lr], float)
        fc = np.array([g(r, "insertion", "flange_in_corpus_rest") for r in Lr], float)
        tl = [g(r, "insertion", "tip_to_labelled_canal_mm") for r in Lr]
        s["lift_L"] = dict(n_frames=len(Lr), tip_in_corpus_drift_mm=rnd(np.linalg.norm(tc - tc[0], axis=1).max()),
                           flange_in_corpus_drift_mm=rnd(np.linalg.norm(fc - fc[0], axis=1).max()),
                           tip_to_labelled_canal_spread_mm=rnd(max(tl) - min(tl)))
    per = {}
    for r in rows:
        p = per.setdefault(r["phase"], dict(steps=[], tip_canal=[], contain=[], cross=[], wic=[]))
        p["steps"].append(r["step"])
        p["tip_canal"].append(g(r, "insertion", "tip_to_labelled_canal_mm"))
        c = g(r, "insertion", "containment", "frac")
        if c is not None:
            p["contain"].append(c)
        o = g(r, "penetration", "organs_in_wall") or {}
        p["cross"].append(sum((o.get(k, {}).get("outer") or {}).get("n_deeper_0p5", 0) for k in ORGANS))
        p["wic"].append(g(r, "penetration", "wall_in_cervix", "free", "n_deeper_0p5") or 0)
    s["per_phase"] = {k: dict(steps=[min(v["steps"]), max(v["steps"])], n_frames=len(v["steps"]),
                              tip_to_labelled_canal_min_mm=min(v["tip_canal"]),
                              containment_min=min(v["contain"]) if v["contain"] else None,
                              organ_sheet_crossings_deeper_0p5_max=max(v["cross"]),
                              free_wall_in_cervix_deeper_0p5_max=max(v["wic"])) for k, v in per.items()}
    # the S5 / S6 lower-canal gate: C frames AFTER d = 5 mm (d > 5 strictly, on the unrounded d), each >= 0.9
    lc5 = [(r["step"], g(r, "insertion", "d_used_mm"), g(r, "insertion", "lower_canal_in_tube_frac"),
            g(r, "insertion", "lower_canal_worst_mm")) for r in rows
           if r["phase"] == "C" and g(r, "insertion", "lower_canal_gated")
           and g(r, "insertion", "lower_canal_in_tube_frac") is not None]
    lo9 = [dict(step=k, d_mm=d, frac=v, worst_mm=w) for k, d, v, w in lc5 if v < 0.9]
    s["lower_canal_in_tube_C_after_5mm"] = dict(
        min=min(x[2] for x in lc5) if lc5 else None, n_frames=len(lc5), n_below_0p9=len(lo9), below_0p9=lo9,
        gate="C frames with d > %g mm (strictly after d = %g); S5 / S6 ask >= 0.9 on each" % (LOWER_GATE_MM,
                                                                                            LOWER_GATE_MM))
    cr = [r.get("carry") or {} for r in rows]
    src = {}
    for r, c in zip(rows, cr):
        k = "frames/step_<k>_cervix_u.npy" if str(c.get("source", "")).startswith("frames/") else c.get("source")
        src.setdefault(k, []).append(r["step"])
    s["carry"] = dict(n_exact=sum(1 for c in cr if c.get("exact")), n_harmonic=sum(1 for c in cr if not c.get("exact")),
                      by_source={k: dict(n=len(v), steps=v if len(v) <= 5 else [v[0], "...", v[-1]])
                                 for k, v in src.items()},
                      max_surface_mismatch_mm=max([c.get("surface_mismatch_mm") or 0.0 for c in cr if c.get("exact")]
                                                  or [None]),
                      n_rejected=sum(1 for c in cr if c.get("rejected")))
    cf = [(r["step"], g(r, "insertion", "containment", "frac"), g(r, "insertion", "containment", "min_clearance_mm"),
           g(r, "insertion", "containment", "n_outside_lumen"),
           g(r, "insertion", "containment", "closed_sheet", "min_clearance_mm")) for r in rows]
    cfv = [x for x in cf if x[1] is not None]
    s["containment"] = dict(min_frac=min(x[1] for x in cfv) if cfv else None,
                            worst_step=min(cfv, key=lambda x: x[1])[0] if cfv else None,
                            min_clearance_mm=min(x[2] for x in cfv if x[2] is not None) if cfv else None,
                            min_clearance_step=min((x for x in cfv if x[2] is not None), key=lambda x: x[2])[0]
                            if cfv else None,
                            n_frames=len(cfv), n_frames_outside=sum(1 for x in cfv if x[3]),
                            max_n_outside_lumen=max(x[3] or 0 for x in cfv) if cfv else None,
                            sheet=(g(rows[0], "insertion", "containment", "sheet") if cfv else None),
                            closed_sheet_min_clearance_mm=min([x[4] for x in cfv if x[4] is not None] or [None]))
    ov = [(r["step"], r["u"], g(r, "insertion", "ovoid_body", "inside_introitus")) for r in rows]
    ins = [x for x in ov if x[2]]
    s["ovoid_body"] = dict(present=bool(run.ovoid_parts), first_inside_introitus_step=ins[0][0] if ins else None,
                           first_inside_introitus_u=ins[0][1] if ins else None,
                           last_outside_step=max([x[0] for x in ov if x[2] is False and (not ins or x[0] < ins[0][0])]
                                                 or [None]))
    ties = sorted((k, _tie_log(r)[0] or 0, r.get("u")) for k, r in run.log.items())
    eng = [x for x in ties if x[1] > 0]
    s["ties"] = dict(first_engaged_step=eng[0][0] if eng else None, first_engaged_u=eng[0][2] if eng else None,
                     final_n=ties[-1][1] if ties else None, set=run.tie_set, per_node=_tie_lag(run))
    s["upper_canal_residual_final_mm"] = g(rows[-1], "insertion", "upper_canal_residual_mm")
    cb = [(r["step"], r.get("corpus") or {}) for r in rows]
    if any(c for _, c in cb):                           # S9: the corpus over the run (every run; "fem" is the new model)
        nr = [(k, (c.get("nonrigid_serosa_mm") or {}).get("max")) for k, c in cb]
        nr = [x for x in nr if x[1] is not None]
        pv = [(k, (c.get("pose_vs_target_mm") or {}).get("max")) for k, c in cb]
        pv = [x for x in pv if x[1] is not None]
        mv = [(k, (c.get("volume") or {}).get("min_tet_ratio")) for k, c in cb]
        mv = [x for x in mv if x[1] is not None]
        s["corpus"] = dict(model=cb[-1][1].get("model"),
                           n_frames_nodes_exact=sum(1 for _, c in cb if c.get("nodes_exact")),
                           nonrigid_serosa_max_mm=dict(max=max(x[1] for x in nr), at_step=max(nr, key=lambda x: x[1])[0])
                           if nr else None,
                           pose_vs_target_max_mm=dict(max=max(x[1] for x in pv), at_step=max(pv, key=lambda x: x[1])[0])
                           if pv else None,
                           min_tet_ratio=dict(min=min(x[1] for x in mv), at_step=min(mv, key=lambda x: x[1])[0])
                           if mv else None,
                           final_frame=cb[-1][1])
    last = rows[-1]
    s["final_frame"] = dict(step=last["step"], organs_in_wall=g(last, "penetration", "organs_in_wall"),
                            wall_in_cervix=g(last, "penetration", "wall_in_cervix"),
                            wall=dict((k, v) for k, v in (last.get("wall") or {}).items() if k != "per_station"))
    # worst per frame, organ-sheet crossings deeper than 0.5 mm (the S2 probes' gate)
    wx = []
    for r in rows:
        o = g(r, "penetration", "organs_in_wall") or {}
        wx.append((r["step"], sum((o.get(k, {}).get("outer") or {}).get("n_deeper_0p5", 0) for k in ORGANS)))
    s["organ_sheet_crossings_deeper_0p5"] = dict(max=max(x[1] for x in wx), at_step=max(wx, key=lambda x: x[1])[0])
    return s


def _dig(r, keys):
    for k in keys:
        if not isinstance(r, dict):
            return None
        r = r.get(k)
    return r


def _tie_log(r):
    """(n_engaged, tip_s) of one log row.  S1b logs row["canal_tie"] = {n_engaged, n_nodes, tip_s} (n_engaged =
    nodes engaged so far; they stay engaged); older runs only n_canal_ties (active ties) and no tip_s."""
    ct = r.get("canal_tie") or {}
    n = ct.get("n_engaged", r.get("n_canal_ties"))
    return (int(n) if n is not None else None), first(ct, ("tip_s",), first(r, ("tip_s", "tip_path_s_mm", "tf_d_mm")))


def _tie_lag(run):
    """Per-node depth-engagement lag from the log (S6: every tie node engages within 3 steps of the tip passing its
    canal_s, never before).  Depth engagement (scene_hybrid canal_engage "depth") engages node i at the first step
    whose row has tip_s >= canal_s[i], so nodes engage in canal_s order and the per-node lag follows from the logged
    counts: node j (j-th smallest canal_s) is PASSED at the first step where #(canal_s <= running max tip_s) > j and
    ENGAGED at the first step where n_engaged > j."""
    if not len(run.tie_s):
        return "the tie set %r has no per-node arclength (%r_s)" % (run.tie_set, run.tie_set)
    rows = [run.log[k] for k in sorted(run.log)]
    recs = [(int(r["step"]),) + _tie_log(r) for r in rows]
    if not any(x[2] is not None for x in recs) or not any(x[1] is not None for x in recs):
        return "not logged (needs canal_tie.n_engaged and canal_tie.tip_s per log row: scene canal_engage 'depth')"
    cs = np.sort(run.tie_s)
    n = len(cs)
    passed = np.full(n, -1)
    engaged = np.full(n, -1)
    run_max = -np.inf
    early = 0
    for k, ne, ts in recs:
        if ts is not None:
            run_max = max(run_max, float(ts))
        exp = int((cs <= run_max).sum())
        for j in range(exp):
            if passed[j] < 0:
                passed[j] = k
        for j in range(min(ne or 0, n)):
            if engaged[j] < 0:
                engaged[j] = k
        early += int((ne or 0) > exp)
    both = (passed >= 0) & (engaged >= 0)
    lag = engaged[both] - passed[both]
    return dict(n_nodes=n, n_passed=int((passed >= 0).sum()), n_engaged=int((engaged >= 0).sum()),
                n_passed_not_engaged=int(((passed >= 0) & (engaged < 0)).sum()),
                max_lag_steps=int(lag.max()) if lag.size else None, min_lag_steps=int(lag.min()) if lag.size else None,
                n_lag_gt_3=int((lag > 3).sum()), n_steps_engaged_exceeds_passed=early,
                gate="S6: every tie node engages within 3 steps of the tip passing its canal_s, never before")


# ============================================================================================ G32 baseline (S0 accept)
def baseline_g32(res):
    """The S0 acceptance list: re-scoring G32 must reproduce the audit's verified baselines.  Tolerances: the plan's
    +/-0.5 mm, +/-0.1 cc, +/-0.01 Dice; +/-10 % on vertex counts where the plan says so (organ-in-wall) and for the
    other counts; +/-0.01 on fractions and stretch."""
    S, Fz = res.get("summary", {}), res.get("final", {})
    fr = _frame(res, 180) or (res["frames"][-1] if res.get("frames") else {})
    f60 = _frame(res, 60) or {}
    rows = []

    def chk(name, measured, lo, hi, note=""):
        ok = None if measured is None else bool(lo <= measured <= hi)
        rows.append(dict(test=name, measured=measured, accept=[rnd(lo), rnd(hi)], **{"pass": ok}, note=note))
    g = _dig
    chk("vagina_filled Dice, old reference (BT grid, PELVIS)",
        g(Fz, ("vagina", "filled_dice_BT_grid", "old_reference")), 0.617, 0.642, "0.627-0.632 +/- 0.01")
    t = S.get("tip_to_labelled_canal", {})
    chk("tip-to-carried-labelled-canal minimum (mm)", t.get("min_mm"), 4.05, 5.2, "4.55-4.7 +/- 0.5 at u ~0.81")
    chk("  ... at u", t.get("at_u"), 0.76, 0.86, "u ~0.81")
    chk("tip-to-carried-labelled-canal final (mm)", t.get("final_mm"), 12.5, 13.9, "13.0-13.4 +/- 0.5")
    tr = S.get("traversed_canal_in_tube", {})
    early = [r["insertion"]["traversed_canal_in_tube_frac"] or 0.0 for r in res.get("frames", [])
             if r["u"] is not None and r["u"] <= 0.8115]
    f132 = _frame(res, 132) or {}
    chk("traversed-canal-in-tube: max over frames with u <= 0.811", max(early) if early else None, 0.0, 0.0,
        "0.00 through u 0.811; first non-zero here at u %s (the audit's canal line: u 0.847, 3 of 36 points; "
        "here the nearest in-span point at step 132 is %s mm against r_tube + 1 = %s mm: a threshold effect)"
        % (tr.get("first_nonzero_u"), g(f132, ("insertion", "min_inspan_canal_dist_to_tube_mm")),
           rnd((g(f132, ("insertion", "r_tube_mm")) or 0) + TUBE_TOL_MM)))
    chk("traversed-canal-in-tube final", tr.get("final"), 0.53, 0.57, "0.54-0.56 +/- 0.01")
    chk("ovoid body inside the introitus, first step", S.get("ovoid_body", {}).get("first_inside_introitus_step"),
        72, 76, "72-76")
    ow = g(fr, ("penetration", "organs_in_wall")) or {}
    for o, n_out, n_lum in (("bladder", 59, 44), ("rectum", 89, 70)):
        chk("step 180 %s vertices inside the outer sheet" % o, g(ow, (o, "outer", "n")), 0.9 * n_out, 1.1 * n_out,
            "%d +/- 10 %%" % n_out)
        chk("step 180 %s vertices in the lumen" % o, g(ow, (o, "lumen", "n")), 0.9 * n_lum, 1.1 * n_lum,
            "%d +/- 10 %%" % n_lum)
    o60 = g(f60, ("penetration", "organs_in_wall")) or {}
    chk("step 60 bladder vertices inside the outer sheet", g(o60, ("bladder", "outer", "n")), 45, 55, "about 50")
    chk("step 60 rectum vertices inside the outer sheet", g(o60, ("rectum", "outer", "n")), 86, 112, "96-102")
    wc = g(fr, ("penetration", "wall_in_cervix")) or {}
    chk("wall lumen nodes inside the cervix", g(wc, ("lumen", "n")), 38, 46, "42 +/- 10 %")
    chk("wall outer nodes inside the cervix", g(wc, ("outer", "n")), 28, 34, "31 +/- 10 %")
    chk("wall-in-cervix max depth (mm)", g(wc, ("lumen", "max_depth_mm")), 4.42, 5.42, "4.92 +/- 0.5")
    chk("free wall nodes inside the cervix", g(wc, ("free", "n")), 31, 39, "35 (R5)")
    chk("free wall-in-cervix max depth (mm)", g(wc, ("free", "max_depth_mm")), 2.16, 3.16, "2.66 +/- 0.5 (R5)")
    cp = g(fr, ("penetration", "cervix_in_parts")) or {}
    for p, n, d in (("ovoid_L", 23, 5.08), ("ovoid_R", 23, 4.53), ("packing", 94, 11.61)):
        chk("cervix vertices inside %s" % p, g(cp, (p, "n")), 0.9 * n, 1.1 * n, "%d" % n)
        chk("  ... max depth (mm)", g(cp, (p, "max_depth_mm")), d - 0.5, d + 0.5, "%.2f +/- 0.5" % d)
    h = g(Fz, ("hrctv", "model_vs_model_tube")) or {}
    chk("HR-CTV\\U mean x-offset about the model tube (mm)", h.get("mean_x_mm"), 4.33, 5.33, "+4.83 +/- 0.5")
    chk("  ... right cc", h.get("right_cc"), 26.2, 26.4, "26.3 +/- 0.1")
    chk("  ... left cc", h.get("left_cc"), 19.4, 19.6, "19.5 +/- 0.1")
    chk("device Dice vs updated applicator | ovoid", g(Fz, ("device_vs_label", "dice_vs_updated_applicator_or_ovoid")),
        0.30, 0.32, "0.31 +/- 0.01")
    chk("rectum Dice z < -30 (preBT grid)", g(Fz, ("organs", "rectum_preBT_grid", "z<-30", "model")), 0.003, 0.023,
        "0.013 +/- 0.01")
    d2o = g(Fz, ("organs", "device_to_OAR_min_mm", "model")) or {}
    chk("device-to-rectum (mm)", d2o.get("rectum"), 0.0, 0.5, "0.0 +/- 0.5")
    chk("device-to-bladder (mm)", d2o.get("bladder"), 0.0, 0.5, "0.0 +/- 0.5")
    w = g(fr, ("wall", "axial_stretch_log")) or {}
    chk("wall axial stretch mean at step 180", w.get("mean"), 1.29, 1.31, "1.30 +/- 0.01")
    chk("wall axial stretch max at step 180", w.get("max"), 1.368, 1.388, "1.378 +/- 0.01")
    chk("HR-CTV\\U preBT offset vs the physician path (mm)",
        g(Fz, ("hrctv", "preBT_reference", "vs_physician_path_xz", "mean_x_mm")), 2.7, 3.9, "+3.2-3.4 +/- 0.5")
    chk("BT HR-CTV\\U vs the json tandem (mm)", g(Fz, ("hrctv", "BT_reference", "vs_json_tandem", "mean_x_mm")),
        -5.18, -4.18, "-4.68 +/- 0.5")
    chk("BT HR-CTV\\U vs the updated-label tandem (mm)",
        g(Fz, ("hrctv", "BT_reference", "vs_updated_label_tandem", "mean_x_mm")), -3.12, -2.12, "-2.62 +/- 0.5")
    chk("model tube vs json axis (deg)", g(Fz, ("device_vs_label", "tube_vs_BT", "angle_to_json_axis_deg")),
        1.39, 1.59, "1.49")
    chk("model tube vs label s15-55 line (deg)", g(Fz, ("device_vs_label", "tube_vs_BT", "angle_to_label_line_deg")),
        2.34, 2.54, "2.44")
    chk("tip error vs label tip (mm)", g(Fz, ("device_vs_label", "tube_vs_BT", "tip_error_vs_label_tip_mm")),
        1.71, 2.71, "2.21 +/- 0.5")
    chk("cervix & filled vagina (cc)", g(Fz, ("junction", "model", "cervix_and_filled_vagina_cc")), 9.88, 10.08,
        "9.98 +/- 0.1")
    return rows


def _frame(res, step):
    return next((r for r in res.get("frames", []) if r["step"] == step), None)


# ============================================================================================ command
def cmd_score(tag, every=1, steps=None, final=True, rd=None, out=None):
    t0 = time.time()
    run = Run(tag, rd)
    frs = run.frames
    if steps:
        want = set(int(s) for s in steps)
        frs = [f for f in frs if int(f["step"]) in want]
    elif every > 1 and frs:
        frs = frs[::every] + ([frs[-1]] if (len(frs) - 1) % every else [])
    print("[tf_metrics] %s: %d of %d saved frames; parts %s (tandem body %s, ovoid body %s); wall %s; plan %s"
          % (tag, len(frs), len(run.frames), sorted(run.devsurf), run.tandem_parts, run.ovoid_parts,
             run.cfg.get("vagina_wall_dir") if run.wall else "none",
             run.tf["source"] if run.tf else "none (no insertion_path_tandem_first)"), flush=True)
    if run.tf and run.tf.get("rows_step"):
        print("[tf_metrics] plan rows by step: offset %s -- %s" % (run.tf["rows_step"].get("offset"),
                                                                   run.tf["rows_step"].get("check")), flush=True)
    rows = []
    for fr in frs:
        t1 = time.time()
        r = score_frame(run, fr)
        rows.append(r)
        ins = r["insertion"]
        ow = (r["penetration"].get("organs_in_wall") or {})
        print("  step %3d %s u %.3f | carry %s | tip-canal %6.2f plan %s | trav %s lower %s | contain %s clear %s | "
              "organs in outer %s lumen %s | %.1f s"
              % (r["step"], r["phase"], r["u"] or 0, "exact" if r["carry"].get("exact") else "harm.",
                 ins["tip_to_labelled_canal_mm"], ins["tip_to_plan_mm"],
                 ins["traversed_canal_in_tube_frac"], ins["lower_canal_in_tube_frac"],
                 (ins.get("containment") or {}).get("frac"), (ins.get("containment") or {}).get("min_clearance_mm"),
                 [ow.get(o, {}).get("outer", {}).get("n") for o in ORGANS],
                 [ow.get(o, {}).get("lumen", {}).get("n") for o in ORGANS], time.time() - t1), flush=True)
    res = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), version=VERSION, tag=tag,
               units="mm; volumes cc; angles deg; Dice [-]",
               run=dict(applicator_dir=os.path.relpath(run.appd, HYB).replace("\\", "/"), parts=sorted(run.devsurf),
                        tandem_body=run.tandem_parts, ovoid_body=run.ovoid_parts,
                        vagina_wall_dir=run.cfg.get("vagina_wall_dir") if run.wall else None,
                        insertion_path=run.cfg.get("insertion_path"), tie_set=run.tie_set,
                        n_frames_saved=len(run.frames), n_frames_scored=len(rows),
                        plan_record=run.tf["source"] if run.tf else None,
                        plan_rows_by_step=run.tf["rows_step"] if run.tf else None,
                        final_cervix_u=bool(run.final_u_path),
                        path_carried_by=run.path_kinds,
                        corpus_model=("fem" if run.corpus_fem else "rigid")),
               status=run_status(run),
               definitions=DEFINITIONS, frames=rows, summary=summarize(run, rows))
    if final:
        t1 = time.time()
        cx = EH.Ctx()
        res["final"] = score_final(run, cx)
        print("[tf_metrics] final-state tables in %.0f s" % (time.time() - t1), flush=True)
    if tag == "G32" and final and not steps and rd is None:
        res["baseline"] = baseline_g32(res)
        np_ = sum(1 for r in res["baseline"] if r["pass"] is True)
        nf = [r for r in res["baseline"] if r["pass"] is False]
        print("\nS0 baseline check (G32): %d pass, %d fail, %d n/a" % (np_, len(nf), sum(1 for r in res["baseline"]
                                                                                          if r["pass"] is None)))
        for r in res["baseline"]:
            print("  %-4s %-58s %-10s accept %s  (%s)" % ({True: "ok", False: "FAIL", None: "n/a"}[r["pass"]],
                                                         r["test"], r["measured"], r["accept"], r["note"]))
    p = EH.wjson(out or "%s/%s/tf_metrics.json" % (EVALD, tag), res)
    print("wrote %s (%.0f s)" % (p, time.time() - t0), flush=True)
    return res


DEFINITIONS = dict(
    frames="model = preBT world RAS mm; per-frame blocks from runs/<tag>/frames (surfaces + device json)",
    tip_to_labelled_canal="distance from the tip (device json tip_mm) to the carried labelled canal polyline "
                          "(tandem_path.npz pts with s >= 0, moved with the tissue: see CARRIED CANAL)",
    tip_to_plan="|tip - (carried tau(d) + w_r(d) delta)|; null without an insertion_path_tandem_first record",
    tip_to_tau="distance from the tip to the carried tau polyline (the record's tau, else the S3 default); needs "
               "no planned depth, ignores delta: a lower bound of tip_to_plan, available for every run",
    tip_in_corpus_rest="the tip and flange mapped into the corpus's rest frame (inverse Kabsch of the frame's corpus "
                       "surface): constant while tandem and corpus move as one body (S6 phase L)",
    d_used="planned depth d of the frame when the plan gives it, else the measured depth = (tip - carried O_true) . "
           "tube axis",
    carry="per frame: the source of the cervix nodal displacement that carries the canal / tau (frames/"
          "step_<k>_cervix_u.npy or final/cervix_u.npy = exact, each only if it reproduces the frame's cervix surface "
          "to U_MATCH_MM = 0.005 mm; else the harmonic extension of the surface) and the surface mismatch; an elastic "
          "corpus (cfg corpus_model fem) adds carry.corpus, the same for the corpus (frames/step_<k>_corpus_u.npy, "
          "final/corpus_u.npy, else its surface's Kabsch fit, flagged approx)",
    corpus="S9, per frame and final: model (rigid / fem); nodes_exact; pose_vs_target_mm = every corpus node vs X0 @ "
           "T_corpus (the schedule's rigid target: device json T_corpus, device_final corpus_T_preBT_to_final); "
           "nonrigid_serosa_mm = the serosa nodes vs their best rigid (Kabsch) fit to rest; volume = total / minimum tet "
           "ratio, tets below 0.6, inverted; span_along_tube_mm = [min, max, max - min] of h above the flange along the "
           "tube of the tet nodes within 5 mm of the tube line (max = the fundus top on the tube); "
           "canal_to_tube_by_band_mm = max distance of the carried labelled canal to the tube segment per canal_s band "
           "0-20 / 20-30 / 30-35 / 35-40 / 40-<end> (end = the labelled canal's largest canal_s rounded up to a whole mm, "
           "corpus_bands); canal_best_line_max_mm = max distance of the carried canal from its "
           "own best-fit line (straightness; S9 accept <= 2 mm)",
    lower_canal_in_tube_frac="fraction of carried labelled-canal points with 0 <= canal_s <= min(d, 20) within "
                             "r_tube + 1 mm of the tube segment flange..tip (gated in S5 / S6 on the C frames with "
                             "d > 5 mm: lower_canal_gated; lower_canal_worst_mm = the farthest of those points)",
    traversed_canal_in_tube_frac="the same over the carried canal points whose axial coordinate lies within the "
                                 "tube's span (0 .. L_iu above the flange): the audit's vcm/v4 definition",
    traversed_by_depth_frac="the same over canal_s <= d",
    upper_canal_residual="max distance to the tube segment of the carried canal points with canal_s > 20",
    ties="n_engaged: log / device json n_canal_ties; n_canal_path_passed: canal_path nodes with canal_s <= d",
    containment="tandem-body vertices below the wall's top ring plane, above the introitus ring plane and outside "
                "the cervix and corpus (signed distance, 1 mm bbox margin); contained = signed distance < 0 to the "
                "OPEN inner sheet (the lumen wall's own triangles, no fan discs, sign oriented on the lumen-ring "
                "centres); clearance = -signed distance; n_in_wall / n_outside_wall split the vertices outside the "
                "lumen by the closed outer sheet; closed_sheet = the same against the lumen closed by fan discs "
                "(tf_metrics 1.0; the introitus disc gives false 'outside' readings just above the introitus plane)",
    ovoid_body="inside_introitus: any ovoid-body vertex above the introitus ring plane (wall station 0, normal "
               "along the wall's centreline); tandem_vs_ovoid_parts: min signed distance of tandem-body vertices to "
               "each ovoid-body part (> 5 mm reported as '>5')",
    penetration="organs_in_wall: organ surface vertices inside the closed outer sheet / closed lumen sheet; "
                "wall_in_cervix: wall nodes inside the cervix surface, split lumen / outer sheet and fixed / free "
                "(FixedConstraint nodes of the phase); cervix_in_parts: cervix surface vertices inside each device "
                "part; parts_in_organs: device-part vertices inside each OAR.  n by signed distance, n_enclosed by "
                "vtkSelectEnclosedPoints, n_disagree between them; n_deeper_0p5 = more than 0.5 mm inside",
    wall="axial_stretch_log / r_in_log: log.jsonl wall.containment (the scene's own measure, per station); "
         "axial_stretch_geom: lumen-ring centre spacing over rest; necking: mean lumen-ring radius over rest; "
         "bow: max distance of the ring-centre line from its chord; vault_to_portio_gap: from the top ring centre "
         "along the local axis to the cervix surface (negative = the centre is inside the cervix)",
    oar="umax of surface vertices vs rest; remote = vertices > 5 mm from every device part and from the outer "
        "sheet; tandem_alone_intrusion_into_preBT: tandem-body vertices at the frame's pose inside the REST OAR",
    final="junction (own-flange / BT label frame), vagina (PELVIS), HR-CTV, organs, device vs the updated label; "
          "each table carries a PREDICTED / CALIBRATED / SCENARIO / REFERENCE tag")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["score"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--every", type=int, default=1, help="score every k-th saved frame (the last one always)")
    ap.add_argument("--steps", nargs="*", help="score only these saved steps (no baseline check)")
    ap.add_argument("--no-final", dest="final", action="store_false", help="skip the final-state tables")
    ap.add_argument("--run-dir", dest="rd", help="score a run folder other than runs/<tag> (no baseline check)")
    ap.add_argument("--out", help="output json (default eval/<tag>/tf_metrics.json)")
    a = ap.parse_args()
    cmd_score(a.tag, every=max(1, a.every), steps=a.steps, final=a.final, rd=a.rd, out=a.out)


if __name__ == "__main__":
    main()
