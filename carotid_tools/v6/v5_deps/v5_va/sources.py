# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\sources.py (= /sp/v5_va/sources.py in the build containers)
"""Loaders for the two ready-made cervical-VA sources of the v5 work.

    import sys; sys.path.insert(0, "/sp/v5_va")
    from sources import load_vmr0248, load_spl
    vmr = load_vmr0248()                 # paths/walls/contours, RAS mm
    vmr = load_vmr0248(with_ct=True)     # + the 0.4x0.4x0.7 mm CTA (234 MB int16)
    spl = load_spl()                     # label-derived surfaces + VA centrelines, RAS mm
    spl = load_spl(with_image=True, with_labels=True, model_surfaces=True)

Everything returned is in ONE frame: patient RAS, millimetres (x+ = patient
right, y+ = anterior, z+ = superior).  Surfaces are pyvista.PolyData.
Each source keeps its own origin (no registration between sources or to
the host; that is the grafting step's job).

Frames (verified, see notes/data.md):
  * VMR 0248: SimVascular project stored in CENTIMETRES.  Source axes are
    already RAS (det +1, no mirror): left subclavian at smaller x, airway
    anterior (+y) of carotids anterior of VAs, brachiocephalic trunk on the
    side named right.  Loader multiplies by 10, nothing else.
  * SPL atlas: NRRDs are declared LPS with negative x/y directions; the .vtk
    models are stored in RAS (Slicer convention; all 21 checked models sit on
    their label voxels to < 1 mm in RAS, 16-153 mm off in LPS).

Python 3.8+, numpy, scipy, vtk, pyvista (skimage only for on-the-fly SPL
surfaces when derived files are missing).
Licences: VMR -- research/development use, keep LICENSE + acknowledgement
(Wilson et al. 2013, NLM R01LM013120, NHLBI HHSN268201100035C);
SPL atlas -- 3D Slicer licence (research, keep notice).
"""
import gzip
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

CM = 10.0
_CANDIDATE_ROOTS = (
    os.environ.get("NEVE_VA_SOURCES", ""),
    "/repo/carotid_data/va_sources",
    "/opt/eve_training/carotid_data/va_sources",
    "D:/Arjun/neve/carotid_data/va_sources",
    r"D:\Arjun\neve\carotid_data\va_sources",
)


def va_root(root=None):
    if root:
        return root
    for r in _CANDIDATE_ROOTS:
        if r and os.path.isdir(r):
            return r
    raise FileNotFoundError("va_sources root not found; set NEVE_VA_SOURCES")


# ----------------------------------------------------------------- helpers
def arclen(p):
    p = np.asarray(p, float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def read_nrrd(path):
    """Minimal NRRD reader (gzip/raw).  Returns (arr[i,j,k], dirs 3x3 rows =
    per-index-axis step vector in the file's space, origin, header dict)."""
    raw = open(path, "rb").read()
    hdr_end = raw.index(b"\n\n")
    h = {}
    for ln in raw[:hdr_end].decode("latin1").splitlines()[1:]:
        if ln.startswith("#") or ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        h[k.strip()] = v.strip()
    sizes = [int(s) for s in h["sizes"].split()]
    dirs = np.array([[float(x) for x in t.split(",")] for t in re.findall(r"\(([^)]*)\)", h["space directions"])])
    org = np.array([float(x) for x in re.findall(r"\(([^)]*)\)", h["space origin"])[0].split(",")])
    data = raw[hdr_end + 2:]
    if h.get("encoding") == "gzip":
        data = gzip.decompress(data)
    dt = {"short": "<i2", "unsigned char": "u1", "uchar": "u1", "ushort": "<u2", "unsigned short": "<u2",
          "float": "<f4", "int": "<i4"}[h["type"]]
    arr = np.frombuffer(data, dtype=dt).reshape(sizes[::-1]).transpose(2, 1, 0)
    return arr, dirs, org, h


def nrrd_affine_ras(dirs, org, space):
    """4x4 ijk -> RAS mm for a NRRD with the given space string."""
    flip = np.diag([-1.0, -1.0, 1.0]) if space.lower().startswith("left-posterior") else np.eye(3)
    A = np.eye(4)
    A[:3, :3] = flip @ np.asarray(dirs, float).T
    A[:3, 3] = flip @ np.asarray(org, float)
    return A


def _pv():
    import pyvista as pv
    return pv


def _read_poly(path, scale=1.0):
    pv = _pv()
    m = pv.read(path)
    if not isinstance(m, pv.PolyData):
        m = m.extract_surface()
    if scale != 1.0:
        m = m.copy()
        m.points = np.asarray(m.points, float) * scale
    return m


def _polyline(points, radius=None):
    pv = _pv()
    pts = np.asarray(points, float)
    lines = np.r_[len(pts), np.arange(len(pts))]
    m = pv.PolyData(pts, lines=lines)
    if radius is not None:
        m.point_data["radius_mm"] = np.asarray(radius, float)
    return m


# ----------------------------------------------------------------- VMR 0248
def parse_pth(fn):
    """SimVascular .pth -> dict(points mm, tangent, rotation, control mm) in RAS."""
    root = ET.parse(fn).getroot()
    pe = root.find("./timestep/path_element")
    ctrl = np.array([[float(p.get(a)) for a in "xyz"] for p in pe.find("control_points")], float) * CM
    pos, tan, rot = [], [], []
    for p in pe.find("path_points"):
        pos.append([float(p.find("pos").get(a)) for a in "xyz"])
        tan.append([float(p.find("tangent").get(a)) for a in "xyz"])
        rot.append([float(p.find("rotation").get(a)) for a in "xyz"])
    return {"points": np.array(pos) * CM, "tangent": np.array(tan), "rotation": np.array(rot), "control": ctrl}


def parse_ctgr(fn):
    """SimVascular .ctgr -> list of contours (RAS mm): centre, tangent, polygon,
    r_eq (sqrt(area/pi)), r_min/r_max about the centroid, circularity."""
    root = ET.parse(fn).getroot()
    out = []
    for c in root.find("timestep").findall("contour"):
        pp = c.find("path_point")
        ctr = np.array([float(pp.find("pos").get(a)) for a in "xyz"]) * CM
        t = np.array([float(pp.find("tangent").get(a)) for a in "xyz"])
        t /= np.linalg.norm(t)
        pts = np.array([[float(p.get(a)) for a in "xyz"] for p in c.find("contour_points")], float) * CM
        u = np.cross(t, [1, 0, 0] if abs(t[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u)
        v = np.cross(t, u)
        q = pts - pts.mean(0)
        x, y = q @ u, q @ v
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        per = float(np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1).sum())
        rr = np.hypot(x, y)
        out.append({"type": c.get("type"), "path_point_id": int(pp.get("id")), "center": ctr, "tangent": t,
                    "points": pts, "area_mm2": float(area), "r_eq": float(np.sqrt(area / np.pi)),
                    "r_min": float(rr.min()), "r_max": float(rr.max()),
                    "circularity": float(4 * np.pi * area / per ** 2) if per > 0 else 0.0})
    return out


def _radius_along(path, contours):
    """Contour r_eq interpolated over the path arc (flat beyond the end contours)."""
    s = arclen(path)
    cp = np.array([s[min(c["path_point_id"], len(path) - 1)] for c in contours])
    cr = np.array([c["r_eq"] for c in contours])
    o = np.argsort(cp)
    return np.interp(s, cp[o], cr[o]), cp[o]


def _leave_index(V, S, s_contours):
    """First VA path index outside the parent (subclavian) lumen."""
    rS = np.array([c["r_eq"] for c in s_contours]); cS = np.array([c["center"] for c in s_contours])
    for i, x in enumerate(V):
        ds = np.linalg.norm(S - x, axis=1); k = int(np.argmin(ds))
        kc = int(np.argmin(np.linalg.norm(cS - S[k], axis=1)))
        if ds[k] >= rS[kc]:
            return i
    return len(V) - 1


def load_vmr0248(root=None, with_ct=False, with_model=True, with_walls=True, all_contours=False):
    """VMR 0248_H_AOCERE_CAS, RAS mm.

    Returns dict:
      paths[name]      -> {"points","tangent","control"}   all 15 SimVascular paths (aorta mostly BELOW the CT)
      contours[name]   -> list of contour dicts (the 15 base groups; + 16 carotid-stenosis variants if all_contours)
      walls[name]      -> pyvista wall face (open) from Simulations/40Right85Left/mesh-complete/mesh-surfaces
      model            -> closed union surface Models/40Left85Right.vtp (ModelFaceID cell array)
      vessels[...]     -> load-ready VA pieces:
         right_vertebral / left_vertebral: "points" (origin = where the path leaves the
             subclavian lumen -> confluence/end), "radius_mm" (contour r_eq), "polyline",
             "full_points" (the raw SimVascular path incl. the stretch inside the subclavian and,
             for the left one, the basilar), "origin_index", "end_index", "wall"
         basilar: tail of the left_vertebral path from the right-VA confluence to its end
         left_subclavian / right_subclavian: path + radius + wall
      ct (with_ct)     -> {"array"[k,j,i] int16 HU, "spacing_mm"(x,y,z), "origin_mm", "affine_ijk_to_ras", "sample"(fn)}
    """
    base = os.path.join(va_root(root), "vmr0248", "0248_H_AOCERE_CAS_reference")
    out = {"name": "vmr0248", "frame": "RAS", "units": "mm", "base": base, "paths": {}, "contours": {},
           "walls": {}, "vessels": {}}
    for fn in sorted(os.listdir(os.path.join(base, "Paths"))):
        out["paths"][fn[:-4]] = parse_pth(os.path.join(base, "Paths", fn))
    used = {"left_vertebral", "right_vertebral", "left_subclavian", "right_subclavian", "aorta", "right_posterior",
            "left_communicating", "right_communicating", "anterior_connector", "left_anterior", "right_anterior",
            "left_carotid", "right_carotid", "left_carotid_external", "right_carotid_external"}
    for fn in sorted(os.listdir(os.path.join(base, "Segmentations"))):
        nm = fn[:-5]
        if all_contours or nm in used:
            out["contours"][nm] = parse_ctgr(os.path.join(base, "Segmentations", fn))
    if with_walls:
        wd = os.path.join(base, "Simulations/40Right85Left/mesh-complete/mesh-surfaces")
        for fn in sorted(os.listdir(wd)):
            if fn.startswith("wall_") and fn.endswith(".vtp"):
                out["walls"][fn[5:-4]] = _read_poly(os.path.join(wd, fn), CM)
    if with_model:
        out["model"] = _read_poly(os.path.join(base, "Models/40Left85Right.vtp"), CM)

    P = {k: v["points"] for k, v in out["paths"].items()}
    C = out["contours"]
    L, R = P["left_vertebral"], P["right_vertebral"]
    rL, cposL = _radius_along(L, C["left_vertebral"])
    rR, _ = _radius_along(R, C["right_vertebral"])
    # The modelled basilar is the stretch where the two VA tubes overlap: the right path runs inside
    # the left one for its last ~17 mm. Merge = first right-path point (walking back from its end)
    # whose centreline gap to the left path exceeds rL + rR.
    iR = len(R) - 1
    while iR > 0:
        d = np.linalg.norm(L - R[iR - 1], axis=1); j = int(np.argmin(d))
        if d[j] > rL[j] + rR[iR - 1]:
            break
        iR -= 1
    jL = int(np.argmin(np.linalg.norm(L - R[iR], axis=1)))          # merge on the left path
    jtip = int(np.argmin(np.linalg.norm(L - P["right_posterior"][0], axis=1)))  # right PCA origin = basilar tip
    jend = int(np.searchsorted(arclen(L), cposL[-1]))              # last left contour: lofted wall ends
    for side, V, r_full, i1 in (("left", L, rL, jL), ("right", R, rR, iR)):
        nm = side + "_vertebral"
        i0 = _leave_index(V, P[side + "_subclavian"], C[side + "_subclavian"])
        pts, rad = V[i0:i1 + 1], r_full[i0:i1 + 1]
        out["vessels"][nm] = {"points": pts, "radius_mm": rad, "polyline": _polyline(pts, rad),
                              "full_points": V, "full_radius_mm": r_full, "origin_index": i0, "end_index": i1,
                              "origin_arc_on_full_mm": float(arclen(V)[i0]), "end_arc_on_full_mm": float(arclen(V)[i1]),
                              "length_mm": float(arclen(pts)[-1]), "wall": out["walls"].get(nm),
                              "note": ("points: origin (path leaves subclavian lumen) -> VA merge. "
                                       + ("full path continues as basilar (to the right-PCA origin) then left P1/P2; "
                                          "wall_left_vertebral covers all of it up to the last contour"
                                          if side == "left" else
                                          "full path continues ~17 mm inside the left path (the shared basilar)"))}
    bas = L[jL:jtip + 1]
    out["vessels"]["basilar"] = {"points": bas, "radius_mm": rL[jL:jtip + 1], "polyline": _polyline(bas, rL[jL:jtip + 1]),
                                 "length_mm": float(arclen(bas)[-1]), "merge_index_on_left_path": jL,
                                 "tip_index_on_left_path": jtip, "merge_index_on_right_path": iR,
                                 "note": "segment of SimVascular path left_vertebral (merge -> right_posterior origin)"}
    lp = L[jtip:jend + 1]
    out["vessels"]["left_pca"] = {"points": lp, "radius_mm": rL[jtip:jend + 1], "polyline": _polyline(lp, rL[jtip:jend + 1]),
                                  "length_mm": float(arclen(lp)[-1]),
                                  "note": "left_vertebral path past the basilar tip, walled to its last contour; "
                                          "left_communicating (PCom) leaves ~10 mm in; path runs a further "
                                          f"{arclen(L)[-1] - cposL[-1]:.1f} mm unwalled"}
    for side in ("left", "right"):
        nm = side + "_subclavian"
        r, _ = _radius_along(P[nm], C[nm])
        out["vessels"][nm] = {"points": P[nm], "radius_mm": r, "polyline": _polyline(P[nm], r),
                              "wall": out["walls"].get(nm)}

    if with_ct:
        out["ct"] = load_vmr0248_ct(root)
    out["licence"] = os.path.join(base, "LICENSE")
    out["citation"] = ("Wilson NM, Ortiz AK, Johnson AB. The Vascular Model Repository. J Med Devices 7(4):040923, 2013; "
                       "NLM R01LM013120; NHLBI HHSN268201100035C; www.vascularmodel.com (0248_H_AOCERE_CAS)")
    return out


def load_vmr0248_ct(root=None):
    """CT of VMR 0248 as int16 HU [k,j,i]; spacing/origin in mm (x,y,z), RAS axes."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    base = os.path.join(va_root(root), "vmr0248", "0248_H_AOCERE_CAS_reference")
    rd = vtk.vtkXMLImageDataReader(); rd.SetFileName(os.path.join(base, "Images/p12full.vti")); rd.Update()
    im = rd.GetOutput(); dims = im.GetDimensions()
    sp = np.array(im.GetSpacing()) * CM; org = np.array(im.GetOrigin()) * CM
    arr = vtk_to_numpy(im.GetPointData().GetScalars()).reshape(dims[2], dims[1], dims[0])
    A = np.eye(4); A[:3, :3] = np.diag(sp); A[:3, 3] = org

    def sample(P, order=1, cval=-3000):
        from scipy.ndimage import map_coordinates
        ijk = (np.asarray(P, float) - org) / sp
        return map_coordinates(arr, [ijk[:, 2], ijk[:, 1], ijk[:, 0]], order=order, mode="constant", cval=cval)

    return {"array": arr, "array_order": "kji", "spacing_mm": sp, "origin_mm": org, "dims_ijk": dims,
            "affine_ijk_to_ras": A, "bounds_mm": [org, org + sp * (np.array(dims) - 1)], "sample": sample}


# ----------------------------------------------------------------- SPL atlas
SPL_VESSELS = {101: "right_common_carotid", 102: "left_common_carotid", 103: "right_subclavian",
               104: "left_subclavian", 105: "right_vertebral", 106: "left_vertebral",
               107: "right_internal_carotid", 108: "left_internal_carotid", 111: "right_external_carotid",
               112: "left_external_carotid", 115: "arch_of_aorta", 116: "brachiocephalic_trunk"}
SPL_BONES = {10: "skull", 11: "C1", 12: "C2", 13: "C3", 14: "C4", 15: "C5", 16: "C6", 17: "C7", 21: "T1"}


def _spl_label_surface(lab, A, value, sigma_mm=0.6):
    """Closed marching-cubes surface of one label (Gaussian sigma_mm pre-smoothing), RAS mm."""
    from scipy import ndimage
    from skimage import measure
    pv = _pv()
    m = (lab == value)
    ii = np.argwhere(m); lo = np.maximum(ii.min(0) - 4, 0); hi = np.minimum(ii.max(0) + 5, m.shape)
    sub = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    sp = np.linalg.norm(A[:3, :3], axis=0)
    if sigma_mm > 0:
        sub = ndimage.gaussian_filter(sub, sigma=sigma_mm / sp)
    sub = np.pad(sub, 1)
    v, f, _, _ = measure.marching_cubes(sub, 0.5)
    v = v - 1 + lo
    ras = v @ A[:3, :3].T + A[:3, 3]
    faces = np.c_[np.full(len(f), 3), f].ravel()
    s = pv.PolyData(ras, faces).clean().connectivity("largest").extract_surface().triangulate()
    return s.compute_normals(auto_orient_normals=True, consistent_normals=True, inplace=False)


def load_spl(root=None, with_image=False, with_labels=False, model_surfaces=False, label_surfaces=True):
    """SPL Head-and-Neck atlas (2016-09), RAS mm.

    Returns dict:
      vessels[name] -> {"label": int, "surface": label-derived closed surface (from derived/ or
                        computed), "model": original .vtk model (fragmented, open; only if
                        model_surfaces), "centerline"/"radius_mm"/"polyline" for the two VAs}
      bones[name]   -> original .vtk model (RAS) when model_surfaces
      image/labels  -> {"array"[i,j,k], "affine_ijk_to_ras"} when requested
    """
    base = os.path.join(va_root(root), "spl_atlas", "head-neck-2016-09")
    der = os.path.join(va_root(root), "spl_atlas", "derived")
    out = {"name": "spl_atlas", "frame": "RAS", "units": "mm", "base": base, "vessels": {}, "bones": {}}
    lab = A = None
    if with_labels or label_surfaces:
        lab, dirs, org, h = read_nrrd(os.path.join(base, "labels/HN-Atlas-labels.nrrd"))
        A = nrrd_affine_ras(dirs, org, h["space"])
        if with_labels:
            out["labels"] = {"array": lab, "affine_ijk_to_ras": A, "lut": dict(SPL_VESSELS, **SPL_BONES)}
    if with_image:
        img, dirs, org, h = read_nrrd(os.path.join(base, "grayscale/Osirix-Manix-255-res.nrrd"))
        out["image"] = {"array": img, "affine_ijk_to_ras": nrrd_affine_ras(dirs, org, h["space"])}
    models = {int(f.split("_")[1]): f for f in os.listdir(os.path.join(base, "models")) if f.endswith(".vtk")}
    for L, nm in SPL_VESSELS.items():
        rec = {"label": L}
        if label_surfaces:
            fp = os.path.join(der, nm + "_label_surface_ras.vtp")
            rec["surface"] = _read_poly(fp) if os.path.exists(fp) else _spl_label_surface(lab, A, L)
        if model_surfaces:
            rec["model"] = _read_poly(os.path.join(base, "models", models[L]))  # already RAS
        out["vessels"][nm] = rec
    for side in ("right", "left"):
        fp = os.path.join(der, side + "_vertebral_centerline_ras.json")
        if os.path.exists(fp):
            d = json.load(open(fp))
            p, r = np.array(d["points"]), np.array(d["radius_mm"])
            out["vessels"][side + "_vertebral"].update({"centerline": p, "radius_mm": r, "polyline": _polyline(p, r),
                                                        "length_mm": float(arclen(p)[-1])})
    if model_surfaces:
        for L, nm in SPL_BONES.items():
            out["bones"][nm] = _read_poly(os.path.join(base, "models", models[L]))
    out["licence"] = os.path.join(base, "LICENSE.md")
    out["citation"] = ("SPL/NAC Head and Neck Atlas (head-neck-2016-09), Surgical Planning Laboratory, Brigham and "
                       "Women's Hospital, https://www.openanatomy.org/atlases/nac/head-neck-2016-09.zip; "
                       "3D Slicer licence. Grayscale: OsiriX 'MANIX' CTA sample, resampled to 255x255")
    return out
