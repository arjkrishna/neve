"""S3 of the G32 fix plan (HOST, python 3.13 with nibabel/scipy/skimage/vtk): the physician's tandem path.

The physician drew the preBT IUcanal label from inside the uterus through the HR-CTV down to the vagina as ONE
connected tube and called it the tandem's travel path; the tandem goes in first and must follow it.  canal.npz
(prep_inputs.py) does not: below L_end it runs 22.5 mm straight along a0 through tumour to O_pre, which is ~23 mm
from the nearest IUcanal voxel and ~4 mm outside the vagina label (fix plan R4).  canal.npz and O_pre stay what the
validated pose rule was built on; O_pre is the "extrapolated portio bottom along a0", not the os.  This module builds
the path from the labels alone:

  vaginal part   per-slice centroids of the FULL raw preBT vagina label, including the ~5.3 cc that lies inside the
                 HR-CTV label (the collapsed upper slit), 5-slice centred running mean whose window is TRUNCATED at
                 the two ends (min_periods=1: the end vertices are means of 3 slices).  MEASURED: a window that
                 shrinks symmetrically instead leaves the top vertex on the top slice's centroid, which sits ~1.9 mm
                 posterior of the canal's end at the same height, and folds the path by 120 / 97 deg at the
                 junction; the truncated window ends the vaginal part one slice lower and joins O_true with 61 / 42
                 deg turns (and is the running mean the audit's reference path used).
  canal part     the IUcanal geodesic centreline: 26-connected voxel graph (edge = voxel-centre distance in mm),
                 Dijkstra from the IUcanal voxels 26-adjacent to the vagina label, 2 mm geodesic bins, one centroid
                 per bin, ONE binomial [1/4 1/2 1/4] pass over the centroids (the wf_canal geodesic_centreline
                 construction behind the audit's ~46.5 mm), and an end cap at each end: the end centroid extended
                 along the direction of the last two bins to the extreme voxel-centre projection.  MEASURED: the raw
                 bin centroids zigzag across the lumen (1 mm bins give a 70.7 mm "canal"); without the pass the
                 capped 2 mm line is 47.8 mm, with it 46.5 mm, the pass moving no centroid by more than 0.7 mm.
  O_true         the canal part's vaginal end cap = the IUcanal's vaginal end (the label's bottom slice, which it
                 shares with the vagina label's top slice) = the physician's external os.  The S5 rotation pivot and
                 the "external os" of the landmark views.
  junction       the vaginal top vertex is joined to O_true by one straight segment (the junction gap).
  internal os    first 1 mm path sample above O_true INSIDE the uterus label by signed distance
                 (vtkImplicitPolyDataDistance to the unsmoothed label surface; rule "path_surface_sdf").  The plan's
                 wording, "first 2 mm-bin centroid whose nearest voxel is uterus" (rule "bins_nn"), is kept as a cfg
                 option and always reported: it quantises the crossing to the bin spacing -- the first bin centroid
                 past the signed-distance crossing still has an outside nearest voxel (the path runs obliquely
                 through the 1.6 mm slices), so the rule lands on the next bin, ~2 mm above every other construction
                 (MEASURED 12.9 against 10.2-11.3 mm).  All constructions are re-measured and recorded.
  a_lc           lower-canal direction: the normalised mean of the unit chords from O_true to the path points 15,
                 16, ... 20 mm above it.  NOT the local 6 mm tangent (two implementations disagreed on it by 26 deg).
  s              arclength along the path from O_true: negative on the vaginal side, 0 at O_true, positive toward
                 the fundus.  pts are sampled at every integer s (1 mm) plus the two exact end points.
  s_F = -d_F     d_F = the pose-rule constant (applicator_v3/pose.json corpus.d_F_mm, 17.5 mm): the path point the
                 validated final state carries to the flange.

Commands
    python -P hybrid/tandem_path.py build      # -> inputs/tandem_path.npz + inputs/tandem_path.json; prints the S3 checks
    python -P hybrid/tandem_path.py nodesets   # adds canal_path / canal_path_s (and the alias apex_pair_g32) to
                                               # meshes/cervix/meta.json and to its copy in meshes/_scene_<wall>/cervix/
    python -P hybrid/tandem_path.py corpus_nodesets   # S9: adds canal_path / canal_path_s to meshes/corpus/meta.json
                                               # (the elastic corpus's canal ties) and to meshes/_scene_<wall>/corpus/
Nothing is re-meshed; the existing node sets (in particular "canal", which G32 and the apex lift use) are untouched.
Frame: model = preBT world RAS mm (x=R, y=A, z=S).  Signed distances are negative inside.  The module imports only
numpy at load time (py3.8-safe: the container may import load() / project()); nibabel, scipy, skimage and vtk are
imported inside the host-only builders.  No patient-derived coordinate is written into this file: every point is
measured from the labels at run time, and the acceptance limits are the plan's.
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.dont_write_bytecode = True                  # never leave __pycache__ in the repo
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
import config  # noqa: E402

FRAME = "preBT world RAS mm (nibabel affine of preBT_MRI_label_*.nii; x=R, y=A, z=S) = model frame"
CFG = dict(
    bin_mm=2.0,                       # geodesic bin of the canal centreline (plan S3)
    bin_mm_alt=(3.0,),                # reported robustness bins (plan: "also report the 3 mm-bin length")
    bin_smooth_passes=1,              # binomial [1/4 1/2 1/4] passes over the bin centroids (0 = raw centroids)
    vag_window=5,                     # slices in the vaginal running mean (centred, truncated at the ends)
    step_mm=1.0,                      # output spacing (integer s)
    internal_os_rule="path_surface_sdf",               # | "bins_nn" (the plan's wording; see the module docstring)
    a_lc_chords_mm=(15.0, 16.0, 17.0, 18.0, 19.0, 20.0),
    tangent_mm=6.0,                   # the local tangent the plan says NOT to use (reported only)
    pose_json="hybrid/applicator_v3/pose.json",       # d_F source (relative to APPSIM_OUT); fallback d_F_default_mm
    tf_pose_json="hybrid/applicator_v4/pose.json",    # S9a: the tandem-first record (rows with tip_s) the corpus
                                                      # canal set must cover (corpus_nodesets --pose overrides it)
    d_F_default_mm=17.5,
    set_radius_mm=3.0, set_radius_fallback_mm=4.0,    # canal_path: cervix nodes within 3 mm (4 mm if < set_min_nodes)
    set_min_nodes=15,
    set_top_above_ios_mm=3.0,                         # the set runs from s_F up to the internal os + 3 mm
    wall_dir="vagina_wall_tet26v4",                   # the scene root whose cervix copy is kept in step
)
# S3 "Accept" limits (fix plan section 2, S3)
ACCEPT = dict(O_true_to_iu_voxel_mm=1.5, O_true_to_L_end_mm=1.5, junction_gap_mm=2.0, vag_centroid_dev_mm=1.5,
              canal_len_mm=(45.0, 47.0), total_len_mm=(121.0, 127.0), internal_os_above_O_mm=(7.5, 11.5),
              set_min_nodes=15, set_frac_within_3mm=0.90,
              corpus_set_reach_tol_mm=0.5)        # S9a: the corpus set's largest canal_s must reach the deepest tip_s of
                                                  # the tandem-first schedule (tip_s_reach) to within this


def paths():
    P = config.paths()
    hyb = P["out"] + "/hybrid"
    return dict(data=P["data"], out=P["out"], inputs=P["inputs"], hybrid=hyb, meshes=hyb + "/meshes",
                logs=hyb + "/logs")


# ============================================================================ polyline helpers (numpy only)
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(float(unit(a) @ unit(b)), -1.0, 1.0))))


def arclength(P):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def interp_path(V, sv, s):
    """Points at arclengths s on the polyline V whose vertex arclengths are sv."""
    s = np.atleast_1d(np.asarray(s, float))
    return np.stack([np.interp(s, sv, V[:, k]) for k in range(3)], 1)


def project(X, pts, s):
    """Closest point of each X on the polyline pts (vertex arclengths s): returns (distance mm, arclength s of the
    closest point, closest point).  Segment by segment, so it is exact for the stored 1 mm path."""
    X = np.atleast_2d(np.asarray(X, float))
    best = np.full(len(X), np.inf)
    sb = np.zeros(len(X))
    cp = np.zeros_like(X)
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ab = b - a
        l2 = float(ab @ ab)
        t = np.clip(((X - a) @ ab) / l2, 0.0, 1.0) if l2 > 0 else np.zeros(len(X))
        q = a + t[:, None] * ab
        d = np.linalg.norm(X - q, axis=1)
        m = d < best
        best[m] = d[m]
        sb[m] = s[i] + t[m] * (s[i + 1] - s[i])
        cp[m] = q[m]
    return best, sb, cp


def load(inputs=None):
    """inputs/tandem_path.npz as a dict of arrays (python ints for the index keys).  Keys: pts, s, i_os,
    i_internal_os, i_fundus, O_true, a_lc, vag_end, s_internal_os, s_vag_top, d_F, s_F, canal_bins,
    vag_centroids_raw, vag_centroids_smooth."""
    inputs = inputs or paths()["inputs"]
    z = np.load(inputs + "/tandem_path.npz")
    out = {k: np.asarray(z[k]) for k in z.files}
    for k in ("i_os", "i_internal_os", "i_fundus", "vag_end"):
        out[k] = int(out[k])
    return out


# ============================================================================ label helpers (host)
def v2w(A, ijk):
    return np.atleast_2d(ijk).astype(float) @ A[:3, :3].T + A[:3, 3]


def w2v(A, xyz):
    return (np.atleast_2d(xyz) - A[:3, 3]) @ np.linalg.inv(A[:3, :3]).T


def load_labels(data):
    import nibabel as nib
    out, A = {}, None
    for key, name in (("iu", "IUcanal"), ("vg", "vagina"), ("hr", "HR-CTV"), ("ut", "uterus")):
        im = nib.load("%s/preBT_MRI_label_%s.nii" % (data, name))
        out[key] = np.asarray(im.dataobj) > 0
        if A is None:
            A = im.affine.copy()
        elif not np.allclose(A, im.affine):
            raise SystemExit("preBT labels do not share one affine (%s)" % name)
    return out, A


def sample_nn(mask, A, X):
    """Label of the voxel nearest to each point (False outside the volume)."""
    ijk = np.rint(w2v(A, X)).astype(int)
    ok = np.all((ijk >= 0) & (ijk < mask.shape), 1)
    v = np.zeros(len(ijk), bool)
    v[ok] = mask[tuple(ijk[ok].T)]
    return v


def voxel_sdf(mask, A):
    """Signed distance volume (mm, negative inside) whose zero level lies on the voxel faces: distance to the
    nearest inside voxel centre minus distance to the nearest outside voxel centre (sample it trilinearly)."""
    from scipy import ndimage as ndi
    sp = np.sqrt((A[:3, :3] ** 2).sum(0))
    return (ndi.distance_transform_edt(~mask, sampling=sp) - ndi.distance_transform_edt(mask, sampling=sp)).astype(np.float32)


def sample_lin(vol, A, X):
    from scipy import ndimage as ndi
    return ndi.map_coordinates(vol, w2v(A, X).T, order=1, mode="nearest")


class LabelSurface:
    """vtkImplicitPolyDataDistance to the UNSMOOTHED marching-cubes surface of a label (level 0.5, i.e. through the
    voxel faces).  The sign is verified at the label's deepest voxel and flipped if the triangle order is inward."""

    def __init__(self, mask, A):
        import vtk
        from skimage import measure
        from scipy import ndimage as ndi
        from vtk.util import numpy_support as ns
        m = np.pad(mask, 1).astype(np.uint8)
        v, f, _, _ = measure.marching_cubes(m, 0.5)
        w = v2w(A, v - 1.0)
        pts = vtk.vtkPoints()
        pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(w), deep=1))
        cells = vtk.vtkCellArray()
        conn = ns.numpy_to_vtkIdTypeArray(np.ascontiguousarray(f.astype(np.int64).ravel()), deep=1)
        offs = ns.numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(f) + 1, 3, dtype=np.int64), deep=1)
        cells.SetData(offs, conn)
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetPolys(cells)
        nrm = vtk.vtkPolyDataNormals()
        nrm.SetInputData(poly)
        nrm.ConsistencyOn()
        nrm.AutoOrientNormalsOn()
        nrm.SplittingOff()
        nrm.ComputeCellNormalsOn()
        nrm.Update()
        self.poly = nrm.GetOutput()
        self.imp = vtk.vtkImplicitPolyDataDistance()
        self.imp.SetInput(self.poly)
        self.sign = 1.0
        sp = np.sqrt((A[:3, :3] ** 2).sum(0))
        dt = ndi.distance_transform_edt(mask, sampling=sp)
        deep = v2w(A, np.unravel_index(int(np.argmax(dt)), mask.shape))
        if self(deep)[0] > 0:
            self.sign = -1.0

    def __call__(self, X):
        return self.sign * np.array([self.imp.EvaluateFunction(p.tolist()) for p in np.atleast_2d(X)])


# ============================================================================ the two parts of the path
def canal_centreline(iu, vg, A, bin_mm=2.0, passes=1):
    """IUcanal geodesic centreline from the vaginal end (plan S3).  Returns dict(bins (k,3) smoothed bin centroids
    from the vaginal end, bins_raw, O_true (vaginal end cap), fundus (fundal end cap), g (geodesic mm per voxel), W
    (voxel centres), n_vox, n_comp26, n_touch, touch_centroid, geo_max, caps_mm, smooth_shift_max)."""
    from scipy import ndimage as ndi
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    idx = np.argwhere(iu)
    n = len(idx)
    W = v2w(A, idx)
    lut = -np.ones(iu.shape, np.int64)
    lut[tuple(idx.T)] = np.arange(n)
    rows, cols, wts = [], [], []
    for o in [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]:
        nb = idx + np.array(o)
        ok = np.all((nb >= 0) & (nb < iu.shape), 1)
        jj = -np.ones(n, np.int64)
        jj[ok] = lut[tuple(nb[ok].T)]
        good = jj >= 0
        rows.append(np.nonzero(good)[0])
        cols.append(jj[good])
        wts.append(np.full(int(good.sum()), float(np.linalg.norm(A[:3, :3] @ np.array(o, float)))))
    G = csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    ncomp = int(connected_components(G, directed=False)[0])
    touch = iu & ndi.binary_dilation(vg, np.ones((3, 3, 3), bool))
    src = lut[tuple(np.argwhere(touch).T)]
    if len(src) == 0:
        raise SystemExit("the IUcanal label does not touch the vagina label: no vaginal end")
    g = dijkstra(G, indices=src, min_only=True)
    if not np.all(np.isfinite(g)):
        raise SystemExit("%d IUcanal voxels are not connected to the vaginal end" % int((~np.isfinite(g)).sum()))
    return dict(g=g, W=W, n_vox=n, n_comp26=ncomp, n_touch=int(len(src)), touch_centroid=W[src].mean(0),
                geo_max=float(g.max()), **bin_line(W, g, bin_mm, passes))


def bin_line(W, g, bin_mm, passes=1):
    """Centroid per geodesic bin, `passes` binomial [1/4 1/2 1/4] passes (end centroids kept), then an end cap at
    each end (the end centroid extended along the direction of the last two bins to the extreme voxel-centre
    projection)."""
    cl, nv = [], []
    for lo in np.arange(0.0, float(g.max()) + bin_mm, bin_mm):
        m = (g >= lo) & (g < lo + bin_mm)
        if m.any():
            cl.append(W[m].mean(0))
            nv.append(int(m.sum()))
    raw = np.array(cl)
    cl = raw.copy()
    for _ in range(int(passes)):
        if len(cl) > 4:
            cs = cl.copy()
            cs[1:-1] = 0.25 * cl[:-2] + 0.5 * cl[1:-1] + 0.25 * cl[2:]
            cl = cs
    d_lo = unit(cl[0] - cl[min(2, len(cl) - 1)])
    d_hi = unit(cl[-1] - cl[max(-3, -len(cl))])
    e_lo = max(0.0, float(((W - cl[0]) @ d_lo).max()))
    e_hi = max(0.0, float(((W - cl[-1]) @ d_hi).max()))
    return dict(bins=cl, bins_raw=raw, bin_nvox=nv, O_true=cl[0] + e_lo * d_lo, fundus=cl[-1] + e_hi * d_hi,
                caps_mm=(e_lo, e_hi), smooth_shift_max=float(np.linalg.norm(cl - raw, axis=1).max()))


def vaginal_centreline(vg, A, window=5):
    """Per-slice centroids of the raw vagina label (bottom -> top) and their centred running mean over `window`
    slices, the window truncated at the two ends."""
    if np.abs(A[:3, :3] - np.diag(np.diag(A[:3, :3]))).max() > 1e-6:
        raise SystemExit("the label affine is not diagonal: 'slice' would not be axial")
    idx = np.argwhere(vg)
    ks = np.unique(idx[:, 2])
    if len(ks) != ks.max() - ks.min() + 1:
        raise SystemExit("the vagina label has empty slices between %d and %d" % (ks.min(), ks.max()))
    raw, nvox = [], []
    for k in ks:
        W = v2w(A, idx[idx[:, 2] == k])
        raw.append(W.mean(0))
        nvox.append(len(W))
    raw = np.array(raw)
    order = np.argsort(raw[:, 2])
    raw, nvox = raw[order], np.array(nvox)[order]
    h = window // 2
    sm = np.array([raw[max(0, i - h): i + h + 1].mean(0) for i in range(len(raw))])
    return raw, sm, nvox


def assemble(vag, canal_bins, O_true, fundus, step=1.0):
    """One polyline introitus -> vaginal top -> O_true -> canal bins -> fundus; s = 0 at O_true.  Returns
    (pts, s, dict(verts, s_verts, n_vag, i_os, vag_end, s_vag_top, gap))."""
    V = np.vstack([vag, O_true[None], canal_bins, fundus[None]])
    sv = arclength(V)
    n_vag = len(vag)
    sv = sv - sv[n_vag]
    s = np.arange(np.ceil(sv[0] / step) * step, np.floor(sv[-1] / step) * step + 0.5 * step, step)
    if s[0] - sv[0] > 1e-6:
        s = np.r_[sv[0], s]
    if sv[-1] - s[-1] > 1e-6:
        s = np.r_[s, sv[-1]]
    pts = interp_path(V, sv, s)
    i_os = int(np.argmin(np.abs(s)))
    s_vag_top = float(sv[n_vag - 1])
    vag_end = int(np.nonzero(s <= s_vag_top + 1e-9)[0][-1])
    return pts, s, dict(verts=V, s_verts=sv, n_vag=n_vag, i_os=i_os, vag_end=vag_end, s_vag_top=s_vag_top,
                        gap=float(np.linalg.norm(O_true - vag[-1])))


def turn_deg(V, i):
    """Turning angle of polyline V at vertex i."""
    return angle_deg(V[i] - V[i - 1], V[i + 1] - V[i])


def read_d_F(PT):
    p = PT["out"] + "/" + CFG["pose_json"]
    try:
        v = float(json.load(open(p))["corpus"]["d_F_mm"])
        return v, "%s corpus.d_F_mm" % CFG["pose_json"]
    except (OSError, KeyError, ValueError, TypeError):
        return float(CFG["d_F_default_mm"]), "default %g mm (%s not readable)" % (CFG["d_F_default_mm"], CFG["pose_json"])


# ============================================================================ build
def _chk(rows, name, value, ok, limit, unit_="mm", note=""):
    rows.append(dict(name=name, value=value, limit=limit, unit=unit_, ok=(None if ok is None else bool(ok)), note=note))
    flag = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    print("  [%s] %-58s %s %s   (limit %s)%s" % (flag, name, json.dumps(value), unit_, limit, ("  " + note) if note else ""),
          flush=True)


def r3(x, n=3):
    return np.round(np.asarray(x, float), n).tolist()


def first_s(s, mask):
    k = np.nonzero(mask)[0]
    return (round(float(s[k[0]]), 2), int(k[0])) if len(k) else (None, None)


def build(args):
    PT = paths()
    t0 = time.time()
    L, A = load_labels(PT["data"])
    iu, vg, hr, ut = L["iu"], L["vg"], L["hr"], L["ut"]
    vv = float(abs(np.linalg.det(A[:3, :3])))
    print("[tandem_path] labels from %s (voxel %s mm)" % (PT["data"], r3(np.sqrt((A[:3, :3] ** 2).sum(0)))), flush=True)
    # ---------------------------------------------------------------- the two parts
    C = canal_centreline(iu, vg, A, CFG["bin_mm"], CFG["bin_smooth_passes"])
    raw, vag, _ = vaginal_centreline(vg, A, CFG["vag_window"])
    pts, s, J = assemble(vag, C["bins"], C["O_true"], C["fundus"], CFG["step_mm"])
    O_true = C["O_true"]
    V, sv, n_vag = J["verts"], J["s_verts"], J["n_vag"]
    i_os = J["i_os"]
    if abs(s[i_os]) > 1e-9:
        raise SystemExit("O_true is not a sample (s %.3g)" % s[i_os])
    L_canal = float(arclength(np.vstack([O_true[None], C["bins"], C["fundus"][None]]))[-1])
    alt_len = dict(bins_only_2mm=round(float(arclength(C["bins"])[-1]), 2),
                   raw_bins_with_caps_2mm=None, raw_bins_only_2mm=round(float(arclength(C["bins_raw"])[-1]), 2))
    Cr = bin_line(C["W"], C["g"], CFG["bin_mm"], 0)
    alt_len["raw_bins_with_caps_2mm"] = round(float(arclength(np.vstack([Cr["O_true"][None], Cr["bins"], Cr["fundus"][None]]))[-1]), 2)
    for bw in CFG["bin_mm_alt"]:
        for ps in (CFG["bin_smooth_passes"], 0):
            Cb = bin_line(C["W"], C["g"], bw, ps)
            alt_len["%gmm_%s" % (bw, "smoothed" if ps else "raw")] = dict(
                with_caps=round(float(arclength(np.vstack([Cb["O_true"][None], Cb["bins"], Cb["fundus"][None]]))[-1]), 2),
                bins_only=round(float(arclength(Cb["bins"])[-1]), 2),
                O_true_shift_mm=round(float(np.linalg.norm(Cb["O_true"] - O_true)), 2))
    L_vag = float(arclength(vag)[-1])
    # ---------------------------------------------------------------- signed distances
    sdv = {k: voxel_sdf(m, A) for k, m in (("vg", vg), ("hr", hr), ("ut", ut), ("iu", iu))}
    srf = {k: LabelSurface(m, A) for k, m in (("vg", vg), ("hr", hr), ("ut", ut), ("iu", iu))}
    from scipy.spatial import cKDTree
    kd_iu = cKDTree(C["W"])
    # ---------------------------------------------------------------- internal os (every construction)
    up = s >= 0
    iup = np.nonzero(up)[0]
    bin_s = sv[n_vag + 1: n_vag + 1 + len(C["bins"])]
    cons = {}
    sv_ = first_s(bin_s, sample_nn(ut, A, C["bins"]))[0]
    cons["bins_nn"] = dict(s=sv_, how="first %g mm-bin centroid (from the vaginal end) whose nearest voxel is labelled "
                                      "uterus (the fix plan's wording)" % CFG["bin_mm"])
    su = srf["ut"](pts[up])
    sv_, k_ = first_s(s[up], su < 0)
    cons["path_surface_sdf"] = dict(s=sv_, how="first 1 mm path sample (s >= 0) inside the uterus label by signed "
                                               "distance (vtkImplicitPolyDataDistance, unsmoothed marching cubes)")
    if k_ is not None and k_ > 0:              # the zero crossing between the last outside and the first inside sample
        a_, b_ = su[k_ - 1], su[k_]
        cons["path_surface_sdf"]["crossing_s"] = round(float(s[up][k_ - 1] + a_ / (a_ - b_) * (s[up][k_] - s[up][k_ - 1])), 2)
    cons["bins_voxel_sdf"] = dict(s=first_s(bin_s, sample_lin(sdv["ut"], A, C["bins"]) < 0)[0],
                                  how="first bin centroid with face-level voxel signed distance to the uterus < 0 (the "
                                      "verifier's construction, vcm/v1_labels.py: 8.48 from its unsmoothed bin 0)")
    cons["path_nn"] = dict(s=first_s(s[up], sample_nn(ut, A, pts[up]))[0],
                           how="first 1 mm path sample (s >= 0) whose nearest voxel is uterus (the investigator's "
                               "construction, wf_canal/s1: 10.0 from its own corner-biased vaginal end)")
    ccl = PT["hybrid"] + "/figs/overlay_prep/G32_PELVIS/IUcanal_centreline_pre.npy"
    if os.path.exists(ccl):
        cl3 = np.load(ccl)
        j = first_s(arclength(cl3), sample_nn(ut, A, cl3))[0]
        cons["ccl_c3_overlay_centreline"] = dict(
            s=j, how="first vertex of figs/overlay_prep/G32_PELVIS/IUcanal_centreline_pre.npy in the uterus (nearest "
                     "voxel), arclength from that centreline's own first vertex (ccl/c3_tie_set.py: 11.2)")
    rule = CFG["internal_os_rule"]
    if cons.get(rule, {}).get("s") is None:
        raise SystemExit("internal os rule %s found no crossing" % rule)
    i_ios = int(iup[np.argmin(np.abs(s[up] - cons[rule]["s"]))])
    s_ios = float(s[i_ios])
    cvals = [v["s"] for v in cons.values() if v["s"] is not None]
    # ---------------------------------------------------------------- lower-canal direction
    chords = {("%g" % Lc): unit(interp_path(pts, s, [Lc])[0] - O_true) for Lc in CFG["a_lc_chords_mm"]}
    a_lc = unit(np.mean(list(chords.values()), 0))
    tan = unit(interp_path(pts, s, [CFG["tangent_mm"]])[0] - O_true)
    q20 = pts[(s >= 0) & (s <= max(CFG["a_lc_chords_mm"]) + 1e-9)]
    a_pca = np.linalg.svd(q20 - q20.mean(0), full_matrices=False)[2][0]
    a_pca = a_pca * np.sign(a_pca @ a_lc)
    # ---------------------------------------------------------------- d_F / s_F
    d_F, d_F_src = read_d_F(PT)
    s_F = -d_F
    # ---------------------------------------------------------------- reference inputs (canal.npz)
    cz = np.load(PT["inputs"] + "/canal.npz")
    L_end, O_pre, a0 = np.asarray(cz["L_end"], float), np.asarray(cz["O_pre"], float), np.asarray(cz["a0"], float)
    dO, sO, _ = project(O_pre[None], pts, s)
    # ---------------------------------------------------------------- acceptance (plan S3)
    rows = []
    print("[tandem_path] S3 acceptance", flush=True)
    d_iu = float(kd_iu.query(O_true)[0])
    _chk(rows, "O_true: distance to the nearest IUcanal voxel centre", round(d_iu, 2), d_iu <= ACCEPT["O_true_to_iu_voxel_mm"],
         "<= %g" % ACCEPT["O_true_to_iu_voxel_mm"], note="signed distance to the IUcanal surface %.2f" % srf["iu"](O_true[None])[0])
    sdv_O = float(srf["vg"](O_true[None])[0])
    sdv_Ov = float(sample_lin(sdv["vg"], A, O_true[None])[0])
    _chk(rows, "O_true: signed distance to the vagina label (surface)", round(sdv_O, 2), sdv_O <= 0.0, "<= 0",
         note="voxel-face SDF %.2f mm; nearest voxel vagina: %s" % (sdv_Ov, bool(sample_nn(vg, A, O_true[None])[0])))
    sdh_O = float(srf["hr"](O_true[None])[0])
    sdh_Ov = float(sample_lin(sdv["hr"], A, O_true[None])[0])
    _chk(rows, "O_true: signed distance to the HR-CTV label (surface)", round(sdh_O, 2), sdh_O < 0.0, "< 0 (inside)",
         note="voxel-face SDF %.2f mm; nearest voxel HR-CTV: %s" % (sdh_Ov, bool(sample_nn(hr, A, O_true[None])[0])))
    dLv = O_true - L_end
    dL = float(np.linalg.norm(dLv))
    dL_ax = float(dLv @ a_lc)
    sd_Lend_iu = float(srf["iu"](L_end[None])[0])
    _chk(rows, "|O_true - L_end|", round(dL, 2), dL <= ACCEPT["O_true_to_L_end_mm"], "<= %g" % ACCEPT["O_true_to_L_end_mm"],
         note="along a_lc %.2f, lateral %.2f; L_end's signed distance to the IUcanal surface %.2f (audit: 1.14 to a "
              "corner-biased end)" % (dL_ax, float(np.linalg.norm(dLv - dL_ax * a_lc)), sd_Lend_iu))
    ds = np.diff(s)
    one = bool(np.all(ds > 0) and ds.max() <= CFG["step_mm"] + 1e-9 and np.all(np.isfinite(pts)))
    _chk(rows, "one polyline introitus -> fundus (s strictly increasing, spacing <= 1 mm)", dict(n=int(len(pts)),
         s_first=round(float(s[0]), 2), s_last=round(float(s[-1]), 2), z_first=round(float(pts[0, 2]), 1),
         z_last=round(float(pts[-1, 2]), 1)), one, "monotone", "")
    _chk(rows, "junction gap (vaginal top vertex -> O_true)", round(J["gap"], 2), J["gap"] <= ACCEPT["junction_gap_mm"],
         "<= %g" % ACCEPT["junction_gap_mm"],
         note="turn at the vaginal top %.1f deg, at O_true %.1f deg" % (turn_deg(V, n_vag - 1), turn_deg(V, n_vag)))
    # path vs the raw per-slice vagina centroids over the slices where the vagina label overlaps the HR-CTV
    kk = np.unique(np.argwhere(vg & hr)[:, 2])
    z_ov = v2w(A, np.c_[np.zeros((len(kk), 2)), kk])[:, 2]
    sel = (raw[:, 2] >= z_ov.min() - 0.05) & (raw[:, 2] <= z_ov.max() + 0.05)
    dev, dev_s, _ = project(raw[sel], pts, s)
    jmax = int(np.argmax(dev))
    _chk(rows, "path vs vagina per-slice centroids, z %.1f..%.1f (3-D max)" % (z_ov.min(), z_ov.max()),
         round(float(dev.max()), 2), dev.max() <= ACCEPT["vag_centroid_dev_mm"], "<= %g" % ACCEPT["vag_centroid_dev_mm"],
         note="worst slice z %.1f (s %.1f); mean %.2f; max over the other slices %.2f; plan z range -28.3..-1.1"
              % (raw[sel][jmax, 2], dev_s[jmax], dev.mean(), np.delete(dev, jmax).max()))
    _chk(rows, "canal part length (2 mm bins, with end caps)", round(L_canal, 2),
         ACCEPT["canal_len_mm"][0] <= L_canal <= ACCEPT["canal_len_mm"][1], "%g-%g" % ACCEPT["canal_len_mm"],
         note="caps %s; bins only %.2f; raw (unsmoothed) centroids with caps %.2f; 3 mm bins %s"
              % (r3(C["caps_mm"], 2), alt_len["bins_only_2mm"], alt_len["raw_bins_with_caps_2mm"], alt_len.get("3mm_smoothed")))
    L_tot = float(s[-1] - s[0])
    _chk(rows, "total length introitus -> fundus", round(L_tot, 2),
         ACCEPT["total_len_mm"][0] <= L_tot <= ACCEPT["total_len_mm"][1], "124 +/- 3",
         note="vaginal %.2f + junction %.2f + canal %.2f" % (L_vag, J["gap"], L_canal))
    _chk(rows, "internal os above O_true (rule %s)" % rule, round(s_ios, 2),
         ACCEPT["internal_os_above_O_mm"][0] <= s_ios <= ACCEPT["internal_os_above_O_mm"][1],
         "%g-%g" % ACCEPT["internal_os_above_O_mm"], note="all constructions %s" % {k: v["s"] for k, v in cons.items()})
    _chk(rows, "a_lc spread (max angle of the 15..20 mm chords to a_lc)",
         round(max(angle_deg(c, a_lc) for c in chords.values()), 2), None, "reported", "deg",
         note="a_lc vs a0 %.1f deg, vs PCA of s 0..20 %.1f deg, vs the 6 mm tangent %.1f deg"
              % (angle_deg(a_lc, a0), angle_deg(a_lc, a_pca), angle_deg(a_lc, tan)))
    _chk(rows, "O_pre: distance to the path / its s", [round(float(dO[0]), 2), round(float(sO[0]), 1)], None, "reported")
    # plan S3 "Effect": HR-CTV minus uterus, voxel x minus the reference line's x at the voxel's z (+ = patient right),
    # against this path and against canal.npz (the audit's wf_verify_canal_feasibility/v6_hrctv_side.py definition)
    Xh = v2w(A, np.argwhere(hr & ~ut))
    vcc = vv / 1000.0

    def x_of_z(P, z):
        o = np.argsort(P[:, 2])
        return np.interp(z, P[o, 2], P[o, 0])

    cpts = np.asarray(cz["pts"], float)
    q = Xh - O_pre
    a_c = unit(cpts[int(cz["i_internal_os"])] - O_pre)
    side = dict(path_xz=Xh[:, 0] - x_of_z(pts, Xh[:, 2]), canal_npz_xz=Xh[:, 0] - x_of_z(cpts, Xh[:, 2]),
                canal_npz_line=(q - np.outer(q @ a_c, a_c))[:, 0])
    side = {k: dict(mean_x_mm=round(float(v.mean()), 2), right_cc=round(float((v > 0).sum() * vcc), 2),
                    left_cc=round(float((v < 0).sum() * vcc), 2)) for k, v in side.items()}
    _chk(rows, "HR-CTV\\U mean x-offset: path x(z) / canal.npz line", [side["path_xz"]["mean_x_mm"],
         side["canal_npz_line"]["mean_x_mm"]], None, "reported", note="R/L cc vs path %.1f / %.1f (plan: +3.2-3.4, "
         "24.1-24.4 / 21.4-21.7 vs the path; +6.0 vs canal.npz)" % (side["path_xz"]["right_cc"], side["path_xz"]["left_cc"]))
    # ---------------------------------------------------------------- membership along the path (1 mm)
    mem = dict(s=r3(s, 2), sd_vagina=r3(sample_lin(sdv["vg"], A, pts), 2), sd_hrctv=r3(sample_lin(sdv["hr"], A, pts), 2),
               sd_uterus=r3(sample_lin(sdv["ut"], A, pts), 2), sd_iucanal=r3(sample_lin(sdv["iu"], A, pts), 2),
               note="face-level voxel signed distances (mm, negative inside), trilinear")
    below = s < 0
    in_hr_vag = below & (sample_lin(sdv["hr"], A, pts) < 0)
    in_vg_vag = below & (sample_lin(sdv["vg"], A, pts) < 0)
    # ---------------------------------------------------------------- write
    out_npz = PT["inputs"] + "/tandem_path.npz"
    np.savez(out_npz, pts=pts, s=s, i_os=i_os, i_internal_os=i_ios, i_fundus=len(pts) - 1, O_true=O_true, a_lc=a_lc,
             vag_end=J["vag_end"], s_internal_os=s_ios, s_vag_top=J["s_vag_top"], d_F=d_F, s_F=s_F,
             canal_bins=C["bins"], vag_centroids_raw=raw, vag_centroids_smooth=vag)
    info = dict(
        what="S3 physician tandem path (fix plan G32 audit rev 2): vaginal slit centroid line + IUcanal geodesic "
             "centreline, one polyline introitus -> fundus",
        frame=FRAME, units="mm; deg", written=time.strftime("%Y-%m-%d %H:%M:%S"), code="hybrid/tandem_path.py build",
        labels=dict(data=PT["data"], used=["preBT_MRI_label_IUcanal", "preBT_MRI_label_vagina", "preBT_MRI_label_HR-CTV",
                                           "preBT_MRI_label_uterus"],
                    vol_cc={k: round(float(m.sum() * vv / 1000.0), 3) for k, m in L.items()},
                    vagina_in_hrctv_cc=round(float((vg & hr).sum() * vv / 1000.0), 3)),
        definitions=dict(
            pts="(N,3) path samples, model frame; every integer s (1 mm) plus the two exact end points",
            s="arclength along the path from O_true (negative on the vaginal side, 0 at O_true, + toward the fundus)",
            i_os="index of O_true (s = 0)",
            i_internal_os="index of the internal os sample (rule %s: %s)" % (rule, cons[rule]["how"]),
            i_fundus="index of the fundal end (= N-1)",
            vag_end="last index on the vaginal part (s <= s_vag_top); the samples vag_end+1 .. i_os lie on the straight "
                    "junction segment",
            O_true="IUcanal vaginal end = vaginal end cap of the %g mm-bin geodesic centreline (Dijkstra from the IUcanal "
                   "voxels 26-adjacent to the vagina label); the physician's external os; S5 rotation pivot" % CFG["bin_mm"],
            vaginal_part="per-slice centroids of the full raw preBT vagina label (bottom -> top), %d-slice centred running "
                         "mean, window truncated at the ends" % CFG["vag_window"],
            canal_part="O_true, the %g mm geodesic-bin centroids (26-connected voxel graph, edge = voxel-centre distance) "
                       "after %d binomial [1/4 1/2 1/4] pass(es), fundal end cap (end centroid extended along the last "
                       "two bins to the extreme voxel projection)" % (CFG["bin_mm"], CFG["bin_smooth_passes"]),
            a_lc="normalised mean of the unit chords O_true -> path(s) for s in %s mm (not the local %g mm tangent)"
                 % (list(CFG["a_lc_chords_mm"]), CFG["tangent_mm"]),
            s_F="-d_F: the path point the validated final state carries to the flange (d_F from %s)" % d_F_src,
            O_pre="canal.npz O_pre is the EXTRAPOLATED PORTIO BOTTOM ALONG a0 (L_end - 22.5 a0 through HR-CTV|uterus), "
                  "kept only as an input of the validated pose rule; it is not the os"),
        O_true=r3(O_true), L_end=r3(L_end), O_pre=r3(O_pre), a_lc=r3(a_lc, 5), a0=r3(a0, 5),
        a_lc_chords=dict((k, dict(dir=r3(v, 5), angle_to_a_lc_deg=round(angle_deg(v, a_lc), 2))) for k, v in chords.items()),
        a_lc_vs=dict(a0_deg=round(angle_deg(a_lc, a0), 2), pca_s0_20_deg=round(angle_deg(a_lc, a_pca), 2),
                     tangent_6mm_deg=round(angle_deg(a_lc, tan), 2), tangent_6mm=r3(tan, 5), pca_s0_20=r3(a_pca, 5)),
        d_F_mm=round(d_F, 3), s_F=round(s_F, 3), d_F_source=d_F_src,
        n_pts=int(len(pts)), i_os=i_os, i_internal_os=i_ios, i_fundus=int(len(pts) - 1), vag_end=J["vag_end"],
        s_internal_os=round(s_ios, 2), s_vag_top=round(J["s_vag_top"], 2), internal_os_pt=r3(pts[i_ios]),
        internal_os_rule=rule, internal_os_constructions=cons,
        internal_os_range_mm=[round(min(cvals), 2), round(max(cvals), 2)],
        internal_os_audit_quoted="7.5-8.5 (verifier), 10.0 (investigator), 11.2 (ccl/c3_tie_set.py); each measured from "
                                 "its own reference point, not from this O_true",
        lengths=dict(vaginal_part=round(L_vag, 2), junction_gap=round(J["gap"], 2), canal_part_2mm_with_caps=round(L_canal, 2),
                     canal_caps_mm=r3(C["caps_mm"], 2), canal_smooth_shift_max_mm=round(C["smooth_shift_max"], 2),
                     canal_variants=alt_len, total=round(L_tot, 2), geodesic_max_from_vaginal_end=round(C["geo_max"], 2)),
        iucanal=dict(n_vox=C["n_vox"], components_26=C["n_comp26"], n_vox_touching_vagina=C["n_touch"],
                     touch_centroid=r3(C["touch_centroid"]), n_bins=int(len(C["bins"])), bin_nvox=C["bin_nvox"]),
        vagina=dict(n_slices=int(len(raw)), z_range=[round(float(raw[0, 2]), 2), round(float(raw[-1, 2]), 2)],
                    top_vertex=r3(vag[-1]), hrctv_overlap_z=[round(float(z_ov.min()), 2), round(float(z_ov.max()), 2)],
                    centroid_dev_3d=dict(max=round(float(dev.max()), 2), mean=round(float(dev.mean()), 2),
                                         per_slice_z_dev_s=[[round(float(z), 1), round(float(d), 2), round(float(q), 1)]
                                                            for z, d, q in zip(raw[sel, 2], dev, dev_s)]),
                    path_mm_below_O_true_inside_hrctv=int(in_hr_vag.sum()),
                    path_mm_below_O_true_inside_vagina=int(in_vg_vag.sum())),
        junction=dict(vag_top=r3(V[n_vag - 1]), O_true=r3(O_true), gap=round(J["gap"], 2),
                      turn_at_vag_top_deg=round(turn_deg(V, n_vag - 1), 1), turn_at_O_true_deg=round(turn_deg(V, n_vag), 1)),
        O_true_checks=dict(dist_to_iu_voxel=round(d_iu, 2), sd_iucanal_surface=round(float(srf["iu"](O_true[None])[0]), 2),
                           sd_vagina_surface=round(sdv_O, 2), sd_vagina_voxel=round(sdv_Ov, 2),
                           sd_hrctv_surface=round(sdh_O, 2), sd_hrctv_voxel=round(sdh_Ov, 2),
                           sd_uterus_surface=round(float(srf["ut"](O_true[None])[0]), 2), dist_to_L_end=round(dL, 2),
                           L_end_minus_O_true_along_a_lc=round(-dL_ax, 2), L_end_sd_iucanal_surface=round(sd_Lend_iu, 2)),
        O_pre_on_path=dict(dist=round(float(dO[0]), 2), s=round(float(sO[0]), 2),
                           dist_to_O_true=round(float(np.linalg.norm(O_pre - O_true)), 2)),
        L_end_on_path=dict(zip(("dist", "s"), [round(float(v[0]), 2) for v in project(L_end[None], pts, s)[:2]])),
        hrctv_minus_uterus_side=dict(side, definition="voxel x minus the reference x at the voxel's z (path_xz, "
                                                      "canal_npz_xz) or the lateral x-component from the line O_pre -> "
                                                      "canal.npz internal os (canal_npz_line); + = patient right"),
        membership_1mm=mem,
        accept=rows, all_accept_ok=bool(all(r["ok"] for r in rows if r["ok"] is not None)),
        cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()}, accept_limits=ACCEPT,
        wall_s=round(time.time() - t0, 1))
    with open(PT["inputs"] + "/tandem_path.json", "w") as fh:
        json.dump(info, fh, indent=1)
    print("[tandem_path] wrote %s (+ tandem_path.json): %d pts, s %.2f..%.2f, i_os %d, i_internal_os %d (s %.2f), "
          "vag_end %d, a_lc %s; S3 path checks: %d pass, %d fail  (%.1f s)"
          % (out_npz, len(pts), s[0], s[-1], i_os, i_ios, s_ios, J["vag_end"], r3(a_lc, 4),
             sum(1 for r in rows if r["ok"] is True), sum(1 for r in rows if r["ok"] is False), time.time() - t0), flush=True)
    return info


# ============================================================================ node sets (cervix meta.json)
def _read_vtk_points(path):
    """Points of a legacy ASCII VTK unstructured grid (the tets.vtk that mesh_bodies.py writes)."""
    txt = open(path).read().split()
    i = txt.index("POINTS")
    n = int(txt[i + 1])
    return np.asarray(txt[i + 3: i + 3 + 3 * n], float).reshape(n, 3)


def _write_json_atomic(obj, path):
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w") as fh:                     # text mode, like mesh_bodies.py: the byte layout is unchanged
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def nodesets(args):
    PT = paths()
    T = load()
    pts, s = T["pts"], T["s"]
    s_F = float(T["s_F"])
    s_top = float(T["s_internal_os"]) + CFG["set_top_above_ios_mm"]
    src_dir = PT["meshes"] + "/cervix"
    X = _read_vtk_points(src_dir + "/tets.vtk")
    d, sp, _ = project(X, pts, s)
    rad = CFG["set_radius_mm"]
    sel = np.nonzero((d <= rad) & (sp >= s_F) & (sp <= s_top))[0]
    if len(sel) < CFG["set_min_nodes"]:
        rad = CFG["set_radius_fallback_mm"]
        sel = np.nonzero((d <= rad) & (sp >= s_F) & (sp <= s_top))[0]
    sel = sel[np.argsort(sp[sel], kind="stable")]              # ordered from s_F upward
    cs = np.round(sp[sel], 3)
    # the "within r of the sub-path s_F..s_top" reading (ccl/c3), for comparison: the set above keeps a node only if its
    # nearest point on the FULL path lies in the range, so the two can differ at the two ends
    sub = (s > s_F) & (s < s_top)
    ps = np.vstack([interp_path(pts, s, [s_F]), pts[sub], interp_path(pts, s, [s_top])])
    d_sub = project(X, ps, arclength(ps))[0]
    L, A = load_labels(PT["data"])
    in_vg = sample_nn(L["vg"], A, X[sel])
    meta_src = src_dir + "/meta.json"
    m = json.load(open(meta_src))
    canal_old = list(m["node_sets"]["canal"])
    rows = []
    print("[tandem_path] canal_path node set (cervix, %d nodes in the mesh)" % len(X), flush=True)
    _chk(rows, "canal_path nodes", int(len(sel)), len(sel) >= ACCEPT["set_min_nodes"], ">= %d" % ACCEPT["set_min_nodes"], "",
         note="radius %g mm; s_F %.2f .. internal os + %g = %.2f" % (rad, s_F, CFG["set_top_above_ios_mm"], s_top))
    f3 = float(np.mean(d[sel] <= 3.0)) if len(sel) else 0.0
    _chk(rows, "fraction within 3 mm of the path", round(f3, 3), f3 >= ACCEPT["set_frac_within_3mm"],
         ">= %g" % ACCEPT["set_frac_within_3mm"], "", note="max distance %.2f mm" % (d[sel].max() if len(sel) else np.nan))
    nb = int((cs < s_F - 1e-6).sum())
    _chk(rows, "nodes below s_F", nb, nb == 0, "0", "", note="canal_s range %.2f..%.2f" % (cs.min(), cs.max()))
    _chk(rows, "tract part (s < 0) / canal part (s >= 0)", [int((cs < 0).sum()), int((cs >= 0).sum())], None, "reported", "",
         note="tract nodes inside the vagina label (nearest voxel) %d of %d" % (int((in_vg & (cs < 0)).sum()), int((cs < 0).sum())))
    _chk(rows, "nodes within %g mm of the sub-path s_F..s_top (ccl/c3 reading)" % rad, int((d_sub <= rad).sum()), None,
         "reported", "", note="overlap with the old 'canal' set %d of %d" % (len(set(sel.tolist()) & set(canal_old)), len(canal_old)))
    ok = all(r["ok"] for r in rows if r["ok"] is not None)
    if not ok and not args.force:
        raise SystemExit("canal_path acceptance failed: meta.json NOT written (use --force to write anyway)")
    ns, defs = m["node_sets"], m["node_set_defs"]
    ns["canal_path"] = [int(i) for i in sel]
    ns["canal_path_s"] = [float(v) for v in cs]
    ns["apex_pair_g32"] = [int(i) for i in canal_old]
    defs["canal_path"] = ("cervix nodes within %g mm (%g mm if fewer than %d) of the physician's tandem path "
                          "(inputs/tandem_path.npz) whose nearest path point has s_F <= s <= internal os + %g mm; s_F = -d_F "
                          "(the path point the validated final state carries to the flange); ordered by canal_s"
                          % (CFG["set_radius_mm"], CFG["set_radius_fallback_mm"], CFG["set_min_nodes"], CFG["set_top_above_ios_mm"]))
    defs["canal_path_s"] = ("per canal_path node (same order): canal_s = arclength (mm) of the nearest path point from O_true, "
                            "positive toward the fundus, negative on the vaginal side (NOT node indices)")
    defs["apex_pair_g32"] = ("identical copy of 'canal' (the O_pre-based set G32 used) under the name the fix plan gives the "
                             "apex-lift pairing (S3 / S6: apex_lift_pair = apex_pair_g32); 'canal' itself is unchanged")
    m["node_set_sizes"].update(canal_path=len(sel), canal_path_s=len(sel), apex_pair_g32=len(canal_old))
    m.setdefault("node_set_extra", {})["canal_path"] = dict(
        source="inputs/tandem_path.npz (hybrid/tandem_path.py build)", radius_mm=rad, s_F=round(s_F, 3),
        d_F_mm=round(float(T["d_F"]), 3), s_top=round(s_top, 3), s_internal_os=round(float(T["s_internal_os"]), 3),
        O_true_mm=r3(T["O_true"]), n=int(len(sel)), n_tract_s_lt_0=int((cs < 0).sum()), n_canal_s_ge_0=int((cs >= 0).sum()),
        dist_to_path_max_mm=round(float(d[sel].max()), 3), tract_nodes_in_vagina_label=int((in_vg & (cs < 0)).sum()),
        written=time.strftime("%Y-%m-%d %H:%M:%S"))
    _write_json_atomic(m, meta_src)
    print("[tandem_path] wrote %s: canal_path %d, canal_path_s %d, apex_pair_g32 %d ('canal' unchanged, %d)"
          % (meta_src, len(sel), len(cs), len(canal_old), len(m["node_sets"]["canal"])), flush=True)
    # the scene root's copy: only when its tets.vtk is the same mesh; mtime > the source so the refresh keeps it
    for wd in args.wall:
        dst_dir = "%s/_scene_%s/cervix" % (PT["meshes"], wd)
        if not os.path.exists(dst_dir + "/meta.json"):
            print("[tandem_path] no scene copy at %s (created from meshes/cervix on the next run)" % dst_dir, flush=True)
            continue
        Xd = _read_vtk_points(dst_dir + "/tets.vtk")
        if Xd.shape != X.shape or np.abs(Xd - X).max() > 1e-6:
            print("[tandem_path] SKIP %s: its tets.vtk is not the source mesh" % dst_dir, flush=True)
            continue
        tmp = "%s/meta.json.tmp.%d" % (dst_dir, os.getpid())
        shutil.copy2(meta_src, tmp)
        os.replace(tmp, dst_dir + "/meta.json")
        st = os.stat(meta_src)
        os.utime(dst_dir + "/meta.json", (st.st_atime, st.st_mtime + 1.0))
        same = open(meta_src, "rb").read() == open(dst_dir + "/meta.json", "rb").read()
        print("[tandem_path] updated %s/meta.json (identical %s; mtime source + 1 s)" % (dst_dir, same), flush=True)
    return rows


# ============================================================================ node sets (corpus meta.json, S9)
def _surface_depth(V, F, X):
    """Depth of the points X below the closed surface (V, F): minus the signed distance (vtkImplicitPolyDataDistance on
    consistently auto-oriented normals; > 0 inside)."""
    import vtk
    from vtk.util import numpy_support as ns
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(V, float), deep=1))
    cells = vtk.vtkCellArray()
    F = np.asarray(F, np.int64)
    cells.SetData(ns.numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(F) + 1, 3, dtype=np.int64), deep=1),
                  ns.numpy_to_vtkIdTypeArray(np.ascontiguousarray(F.ravel()), deep=1))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cells)
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputData(poly)
    nrm.ConsistencyOn()
    nrm.AutoOrientNormalsOn()
    nrm.SplittingOff()
    nrm.ComputeCellNormalsOn()
    nrm.Update()
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(nrm.GetOutput())
    return -np.array([imp.EvaluateFunction([float(v) for v in p]) for p in np.atleast_2d(X)])


def tip_s_reach(pose_json=None):
    """The deepest canal arclength the tandem tip reaches in the tandem-first schedule: the largest tip_s of the
    pose.json insertion_path_tandem_first rows (the rows scene_hybrid.build_schedule regenerates; written by
    hybrid/tf_gate.py search --write), default CFG tf_pose_json.  Returns (tip_s_max or None, source text)."""
    p = pose_json or (paths()["out"] + "/" + CFG["tf_pose_json"])
    try:
        rows = json.load(open(p))["insertion_path_tandem_first"]["rows"]
        ts = [float(r["tip_s"]) for r in rows if r.get("tip_s") is not None]
    except (OSError, KeyError, TypeError, ValueError) as e:
        return None, "%s: no tandem-first rows (%s)" % (p, type(e).__name__)
    if not ts:
        return None, "%s: no tip_s in the tandem-first rows" % p
    return max(ts), "%s insertion_path_tandem_first rows (largest tip_s)" % p


def corpus_nodesets(args):
    """S9a of the G32 fix plan: the corpus's canal node set for the elastic corpus's canal ties (scene_hybrid
    corpus_canal_tie_set "canal_path").  The corpus nodes within set_radius_mm (3 mm; 4 mm if fewer than set_min_nodes)
    of the LABELLED canal (the path's s >= 0 part, O_true -> fundal end), ordered by canal_s = the arclength of their
    nearest canal point; canal_path_s holds it (mm from O_true, NOT node indices).  Additive: surface_nodes and
    interface_cervix are untouched and nothing is re-meshed.  Recorded per node in node_set_extra.canal_path: the
    distance to the canal and the depth below the corpus surface (meshes/corpus/surface.obj, signed distance).  Accept:
    >= set_min_nodes nodes, >= 90 % within 3 mm, canal_s max >= the deepest tip_s of the tandem-first schedule (the
    canal depth the tandem tip reaches, tip_s_reach) - corpus_set_reach_tol_mm; without a tandem-first record the reach
    is only reported."""
    import geom
    PT = paths()
    T = load()
    pts, s = T["pts"], T["s"]
    up = s >= 0
    canal, cs_all = pts[up], s[up]
    src_dir = PT["meshes"] + "/corpus"
    X = _read_vtk_points(src_dir + "/tets.vtk")
    d, sp, _ = project(X, canal, cs_all)
    rad = CFG["set_radius_mm"]
    sel = np.nonzero(d <= rad)[0]
    if len(sel) < CFG["set_min_nodes"]:
        rad = CFG["set_radius_fallback_mm"]
        sel = np.nonzero(d <= rad)[0]
    sel = sel[np.argsort(sp[sel], kind="stable")]              # ordered from the os upward
    cs = np.round(sp[sel], 3)
    V, F = geom.read_obj(src_dir + "/surface.obj")
    depth = _surface_depth(V, F, X[sel])
    meta_src = src_dir + "/meta.json"
    m = json.load(open(meta_src))
    ns = m["node_sets"]
    on_surf = np.isin(sel, np.asarray(ns["surface_nodes"], int))
    on_ifc = np.isin(sel, np.asarray(ns["interface_cervix"], int))
    rows = []
    print("[tandem_path] corpus canal_path node set (corpus, %d nodes in the mesh)" % len(X), flush=True)
    _chk(rows, "corpus canal_path nodes", int(len(sel)), len(sel) >= ACCEPT["set_min_nodes"],
         ">= %d" % ACCEPT["set_min_nodes"], "", note="radius %g mm of the labelled canal (s >= 0)" % rad)
    f3 = float(np.mean(d[sel] <= 3.0)) if len(sel) else 0.0
    _chk(rows, "fraction within 3 mm of the canal", round(f3, 3), f3 >= ACCEPT["set_frac_within_3mm"],
         ">= %g" % ACCEPT["set_frac_within_3mm"], "", note="max distance %.2f mm" % (d[sel].max() if len(sel) else np.nan))
    smax = float(cs.max()) if len(cs) else float("nan")
    reach, reach_src = tip_s_reach(getattr(args, "pose", None))
    note = "canal_s range %.2f..%.2f; canal end s %.2f; tip reach from %s" % (float(cs.min()), smax,
                                                                             float(cs_all.max()), reach_src)
    if reach is None:
        _chk(rows, "canal_s max", round(smax, 2), None, "reported (no tandem-first tip_s)", note=note)
    else:
        need = reach - ACCEPT["corpus_set_reach_tol_mm"]
        _chk(rows, "canal_s max", round(smax, 2), smax >= need, ">= %.2f (largest tip_s %.2f - %g)"
             % (need, reach, ACCEPT["corpus_set_reach_tol_mm"]), note=note)
    _chk(rows, "on the surface / on interface_cervix", [int(on_surf.sum()), int(on_ifc.sum())], None, "reported", "",
         note="depth below the corpus surface: min %.2f, median %.2f mm; depth of the 5 highest-s nodes %s"
         % (float(depth.min()), float(np.median(depth)), np.round(depth[-5:], 2).tolist()))
    ok = all(r["ok"] for r in rows if r["ok"] is not None)
    if not ok and not args.force:
        raise SystemExit("corpus canal_path acceptance failed: meta.json NOT written (use --force to write anyway)")
    defs = m.setdefault("node_set_defs", {})
    ns["canal_path"] = [int(i) for i in sel]
    ns["canal_path_s"] = [float(v) for v in cs]
    defs["canal_path"] = ("corpus nodes within %g mm (%g mm if fewer than %d) of the physician's LABELLED canal "
                          "(inputs/tandem_path.npz, s >= 0: O_true -> fundal end); ordered by canal_s.  The elastic "
                          "corpus's canal ties (scene_hybrid corpus_canal_tie_set, S9)"
                          % (CFG["set_radius_mm"], CFG["set_radius_fallback_mm"], CFG["set_min_nodes"]))
    defs["canal_path_s"] = ("per canal_path node (same order): canal_s = arclength (mm) of the nearest labelled-canal point "
                            "from O_true, positive toward the fundus (NOT node indices)")
    m.setdefault("node_set_sizes", {}).update(canal_path=len(sel), canal_path_s=len(sel))
    m.setdefault("node_set_extra", {})["canal_path"] = dict(
        source="inputs/tandem_path.npz (hybrid/tandem_path.py build), s >= 0", radius_mm=rad,
        O_true_mm=r3(T["O_true"]), n=int(len(sel)), canal_s_range=[round(float(cs.min()), 3), round(smax, 3)],
        n_on_surface=int(on_surf.sum()), n_on_interface_cervix=int(on_ifc.sum()),
        dist_to_canal_mm=np.round(d[sel], 3).tolist(), depth_below_surface_mm=np.round(depth, 3).tolist(),
        written=time.strftime("%Y-%m-%d %H:%M:%S"), code="hybrid/tandem_path.py corpus_nodesets")
    _write_json_atomic(m, meta_src)
    print("[tandem_path] wrote %s: canal_path %d, canal_path_s %d (surface_nodes %d, interface_cervix %d unchanged)"
          % (meta_src, len(sel), len(cs), len(ns["surface_nodes"]), len(ns["interface_cervix"])), flush=True)
    for wd in args.wall:                             # the scene root's copy, as `nodesets` keeps the cervix's
        dst_dir = "%s/_scene_%s/corpus" % (PT["meshes"], wd)
        if not os.path.exists(dst_dir + "/meta.json"):
            print("[tandem_path] no scene copy at %s (created from meshes/corpus on the next run)" % dst_dir, flush=True)
            continue
        Xd = _read_vtk_points(dst_dir + "/tets.vtk")
        if Xd.shape != X.shape or np.abs(Xd - X).max() > 1e-6:
            print("[tandem_path] SKIP %s: its tets.vtk is not the source mesh" % dst_dir, flush=True)
            continue
        tmp = "%s/meta.json.tmp.%d" % (dst_dir, os.getpid())
        shutil.copy2(meta_src, tmp)
        os.replace(tmp, dst_dir + "/meta.json")
        st = os.stat(meta_src)
        os.utime(dst_dir + "/meta.json", (st.st_atime, st.st_mtime + 1.0))
        same = open(meta_src, "rb").read() == open(dst_dir + "/meta.json", "rb").read()
        print("[tandem_path] updated %s/meta.json (identical %s; mtime source + 1 s)" % (dst_dir, same), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="inputs/tandem_path.npz + .json, S3 path checks")
    p = sub.add_parser("nodesets", help="canal_path / canal_path_s / apex_pair_g32 in meshes/cervix/meta.json")
    p.add_argument("--wall", nargs="*", default=[CFG["wall_dir"]], help="scene roots whose cervix copy is updated too")
    p.add_argument("--force", action="store_true", help="write even if a node-set check fails")
    p = sub.add_parser("corpus_nodesets", help="S9: canal_path / canal_path_s in meshes/corpus/meta.json")
    p.add_argument("--wall", nargs="*", default=[CFG["wall_dir"]], help="scene roots whose corpus copy is updated too")
    p.add_argument("--force", action="store_true", help="write even if a node-set check fails")
    p.add_argument("--pose", help="pose.json holding insertion_path_tandem_first (default APPSIM_OUT/%s)"
                   % CFG["tf_pose_json"])
    args = ap.parse_args()
    if args.cmd == "build":
        build(args)
    elif args.cmd == "corpus_nodesets":
        corpus_nodesets(args)
    else:
        nodesets(args)


if __name__ == "__main__":
    main()
