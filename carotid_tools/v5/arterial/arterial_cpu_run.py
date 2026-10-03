#!/usr/bin/env python
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\arterial_cpu_run.py (= /sp/v5_va/arterial_cpu_run.py in the build containers)
"""CPU runner for the Arterial pipeline (neve-arterial image, conda base env: py3.10 + vmtk 1.5.0).

Wraps arterial.run.processor.ArterialProcessor so that it runs sensibly on a CPU-only box and
leaves machine-readable records behind:

  * input preparation  -> <case>/cta.nii.gz (+ <case>/prep.json)
      - reoriented to LAS (Arterial's code accepts only RAS/LAS/LPS and assumes axis 2 = I->S),
      - oblique rotation stripped if present (Arterial derives spacing from the affine diagonal);
        prep.json keeps the 4x4 that maps Arterial world coords back to the input's world coords,
      - optional z crop (--crop-k k0:k1 in INPUT voxel index of the S axis, or --crop-mm z0:z1
        in input world S mm) so a long CTA fits Arterial's 2.5e6..3.5e7 mm^3 vessel-bbox gate.
  * head/neck split (default hybrid mode only): Arterial runs TotalSegmentator's craniofacial
      model at 0.5 mm on the upper half of the CT to find the skull. On a 1.5 mm CT that is a
      27x upsampling (tens of minutes on CPU, ~10 GB). --skull-mask <nii> (e.g. TotalSegmentator's
      own skull.nii.gz on the same grid as --input) replaces that step; --slicer laplacian uses
      Arterial's built-in LoG fallback; --slicer ts keeps the original model.
  * CPU predictor settings -> nnU-Net test-time mirroring OFF by default (8x fewer forward passes);
      --tta turns it back on. Torch / nnU-Net thread count from --threads.
  * per-stage wall time, peak RAM (cgroup memory.peak / sampled memory.current and ru_maxrss)
      -> <case>/run_report.json, plus the list of files written.

Stages: seg, cl (centerlines+branching+postprocessing), lm (landmarks), vl (vessel labelling),
        fe (feature extraction), ap (access prediction).  --stages core = seg,cl,vl (default);
        --stages full = seg,cl,lm,vl,fe,ap.

Licences: Arterial code PolyForm-Noncommercial-1.0.0, weights CC BY-NC 4.0 (Zenodo 22694951),
TotalSegmentator craniofacial model Apache-2.0. Research use only.
"""
import argparse
import json
import os
import resource
import sys
import threading
import time
import traceback

import numpy as np


# ----------------------------------------------------------------------------- memory monitor
def _read_int(path):
    try:
        with open(path) as f:
            v = f.read().strip()
        return None if v == "max" else int(v)
    except Exception:
        return None


class MemMonitor(threading.Thread):
    """Samples the container cgroup memory.current every 0.5 s (includes child processes)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.stage_peak = 0
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            v = _read_int("/sys/fs/cgroup/memory.current")
            if v is None:
                v = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
            if v is not None:
                self.peak = max(self.peak, v)
                self.stage_peak = max(self.stage_peak, v)
            time.sleep(0.5)

    def stop(self):
        self._halt.set()


# ----------------------------------------------------------------------------- input preparation
def _to_arterial_grid(data, A0, crop_k):
    """Crops along the source S axis, reorients to LAS, strips obliquity. Returns (data, A1, A2, info)."""
    from nibabel.orientations import (axcodes2ornt, io_orientation, ornt_transform, inv_ornt_aff,
                                      apply_orientation)
    info = {}
    ornt0 = io_orientation(A0)
    s_axis = int(np.where(ornt0[:, 0] == 2)[0][0])
    if crop_k is not None:
        k0, k1 = max(0, crop_k[0]), min(data.shape[s_axis], crop_k[1])
        sl = [slice(None)] * 3
        sl[s_axis] = slice(k0, k1)
        data = data[tuple(sl)]
        shift = np.zeros(4); shift[s_axis] = k0; shift[3] = 1
        A0 = A0.copy(); A0[:3, 3] = (A0 @ shift)[:3]
        info["crop_k_source"] = [int(k0), int(k1)]
    xform = ornt_transform(io_orientation(A0), axcodes2ornt(("L", "A", "S")))
    data = apply_orientation(data, xform)
    A1 = A0 @ inv_ornt_aff(xform, data.shape)       # world coords preserved exactly
    R = A1[:3, :3]
    sp = np.sqrt((R ** 2).sum(axis=0))
    D = np.diag(np.sign(np.diag(R)) * sp)
    A2 = np.eye(4); A2[:3, :3] = D; A2[:3, 3] = A1[:3, 3]
    info["oblique_max_offdiag"] = float(np.abs(R - D).max())
    info["spacing_mm"] = sp.tolist()
    return np.ascontiguousarray(data), A1, A2, info


def _s_axis_crop_from_mm(A0, shape, crop_mm):
    from nibabel.orientations import io_orientation
    s_axis = int(np.where(io_orientation(A0)[:, 0] == 2)[0][0])
    centre = np.array([(d - 1) / 2.0 for d in shape] + [1.0])
    zs = []
    for k in range(shape[s_axis]):
        c = centre.copy(); c[s_axis] = k
        zs.append((A0 @ c)[2])
    zs = np.array(zs)
    ks = np.where((zs >= min(crop_mm)) & (zs <= max(crop_mm)))[0]
    return int(ks.min()), int(ks.max()) + 1


def prepare_input(src, case_dir, crop_k=None, crop_mm=None, skull_mask=None, fov_mm=None, fov_center=None, resample_mm=None):
    import nibabel as nib

    img = nib.load(src)
    A0 = img.affine.copy()
    data = np.asanyarray(img.dataobj)
    info = {"source": os.path.abspath(src), "source_shape": list(data.shape),
            "source_axcodes": "".join(nib.aff2axcodes(A0)), "source_affine": A0.tolist(),
            "source_dtype": str(data.dtype)}
    if crop_mm is not None:
        crop_k = _s_axis_crop_from_mm(A0, data.shape, crop_mm)
    data, A1, A2, gi = _to_arterial_grid(data, A0, crop_k)
    info.update(gi)
    ij_win = None
    if fov_mm is not None:
        # in-plane crop to fov_mm x fov_mm (LAS grid: axis0 = R->L, axis1 = P->A) around fov_center (world x,y)
        if fov_center is None:
            c_ij = np.array([(data.shape[0] - 1) / 2.0, (data.shape[1] - 1) / 2.0])
        else:
            c_ij = (np.linalg.inv(A2) @ np.array([fov_center[0], fov_center[1], A2[2, 3], 1.0]))[:2]
        half = [fov_mm / 2.0 / abs(A2[0, 0]), fov_mm / 2.0 / abs(A2[1, 1])]
        lo = [max(0, int(round(c_ij[d] - half[d]))) for d in range(2)]
        hi = [min(data.shape[d], int(round(c_ij[d] + half[d]))) for d in range(2)]
        ij_win = (lo, hi)
        data = np.ascontiguousarray(data[lo[0]:hi[0], lo[1]:hi[1], :])
        T = np.eye(4); T[0, 3] = lo[0]; T[1, 3] = lo[1]
        A1 = A1 @ T; A2 = A2 @ T
        info["fov_crop_ij_las"] = [lo, hi]
    scale = None
    if resample_mm is not None:
        # isotropic resampling (trilinear) on an exact sub-grid: new voxel i' sits at old index i'*s/sp
        from scipy import ndimage
        sp_now = np.abs(np.diag(A2)[:3])
        scale = resample_mm / sp_now
        new_shape = tuple(int(np.floor((n - 1) / f)) + 1 for n, f in zip(data.shape, scale))
        data = ndimage.affine_transform(data.astype(np.float32), scale, output_shape=new_shape, order=1,
                                        mode="nearest")
        S = np.diag(list(scale) + [1.0])
        A1 = A1 @ S; A2 = A2 @ S
        info["resampled_to_mm"] = resample_mm
        info["source_spacing_mm"] = info["spacing_mm"]; info["spacing_mm"] = [float(resample_mm)] * 3
    to_input_world = A1 @ np.linalg.inv(A2)
    info.update({"arterial_axcodes": "".join(nib.aff2axcodes(A2)), "arterial_shape": list(data.shape),
                 "arterial_affine": A2.tolist(), "arterial_to_input_world": to_input_world.tolist(),
                 "note": "Arterial outputs live in the world frame of arterial_affine; map points p (mm) with "
                         "arterial_to_input_world @ [p,1] to get the source CT's world frame (identity unless "
                         "the source was oblique)."})
    if (np.issubdtype(data.dtype, np.integer) or scale is not None) and data.min() >= -32768 and data.max() <= 32767:
        out = np.rint(data).astype(np.int16)
    else:
        out = data.astype(np.float32)
    info["hu_p1_p50_p999"] = [float(np.percentile(out, q)) for q in (1, 50, 99.9)]
    os.makedirs(case_dir, exist_ok=True)

    def _save(arr, path):
        n = nib.Nifti1Image(arr, A2)
        n.set_qform(A2, code=1); n.set_sform(A2, code=1)
        nib.save(n, path)

    _save(out, os.path.join(case_dir, "cta.nii.gz"))
    if skull_mask:
        mi = nib.load(skull_mask)
        if mi.shape != tuple(info["source_shape"]) or not np.allclose(mi.affine, A0, atol=1e-3):
            raise ValueError("skull mask must be on the same grid as the input CT")
        md, _, _, _ = _to_arterial_grid(np.asanyarray(mi.dataobj) > 0, A0, crop_k)
        if ij_win is not None:
            (lo, hi) = ij_win
            md = np.ascontiguousarray(md[lo[0]:hi[0], lo[1]:hi[1], :])
        if scale is not None:
            from scipy import ndimage
            md = ndimage.affine_transform(md.astype(np.uint8), scale, output_shape=data.shape, order=0) > 0
        _save(md.astype(np.uint8), os.path.join(case_dir, "skull_mask.nii.gz"))
        info["skull_mask"] = os.path.abspath(skull_mask)
        info["skull_mask_voxels"] = int(md.sum())
    with open(os.path.join(case_dir, "prep.json"), "w") as f:
        json.dump(info, f, indent=1)
    print("prepared", json.dumps({k: info.get(k) for k in ("source_axcodes", "arterial_axcodes", "arterial_shape",
                                                             "spacing_mm", "oblique_max_offdiag", "hu_p1_p50_p999",
                                                             "crop_k_source", "fov_crop_ij_las", "resampled_to_mm", "skull_mask_voxels")}), flush=True)
    return info


# ----------------------------------------------------------------------------- CPU patches
def patch_for_cpu(tta, threads, tile_step):
    import torch
    torch.set_num_threads(threads)
    os.environ["nnUNet_def_n_proc"] = str(threads)
    import nnunetv2.configuration as cfg
    cfg.default_num_processes = threads
    import nnunetv2.inference.predict_from_raw_data as p
    p.default_num_processes = threads
    orig = p.nnUNetPredictor.__init__

    cuda = torch.cuda.is_available()

    def init(self, *a, **k):
        # tta: None = auto (upstream mirroring on GPU, off on CPU); True/False forces it
        if (tta is False) or (tta is None and not cuda):
            k["use_mirroring"] = False
        if tile_step is not None:
            k["tile_step_size"] = tile_step
        if not cuda:
            k["perform_everything_on_device"] = False
        k["allow_tqdm"] = False
        orig(self, *a, **k)
        print(f"[cpu-patch] nnUNetPredictor device={'cuda' if cuda else 'cpu'} use_mirroring={self.use_mirroring} "
              f"tile_step={self.tile_step_size} threads={torch.get_num_threads()}", flush=True)

    p.nnUNetPredictor.__init__ = init


def patch_slicer(slicer, case_dir):
    """Replaces the TotalSegmentator cranium step of the head/neck split."""
    import arterial.segmentation.utils as su
    if slicer == "ts":
        return
    if slicer == "laplacian":
        def _raise(*a, **k):
            raise RuntimeError("TotalSegmentator cranium step disabled (--slicer laplacian)")
        su.run_cranium_segmentation_totalsegmentator = _raise
        return
    if slicer == "mask":
        import nibabel as nib
        path = os.path.join(case_dir, "skull_mask.nii.gz")

        def _from_mask(cta_array, cta_affine):
            m = np.asanyarray(nib.load(path).dataobj) > 0
            if m.shape != cta_array.shape:
                raise RuntimeError(f"skull mask shape {m.shape} != CTA {cta_array.shape}")
            print(f"[slicer] using supplied skull mask ({int(m.sum())} voxels)", flush=True)
            return su.get_largest_connected_component(m)
        su.run_cranium_segmentation_totalsegmentator = _from_mask


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CTA NIfTI (HU). Copied/reoriented into <case>/cta.nii.gz")
    ap.add_argument("--case", required=True, help="output case directory")
    ap.add_argument("--stages", default="core", help="core | full | comma list of seg,cl,lm,vl,fe,ap")
    ap.add_argument("--fast", action="store_true", help="Arterial fast mode: 3d_lowres on the whole volume")
    ap.add_argument("--tta", dest="tta", action="store_true", default=None,
                    help="force nnU-Net mirroring TTA on (default: on with a GPU, off on CPU where it costs ~5x)")
    ap.add_argument("--no-tta", dest="tta", action="store_false", help="force mirroring TTA off")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("ARTERIAL_THREADS", "12")))
    ap.add_argument("--tile-step", type=float, default=None, help="override nnU-Net tile_step_size (Arterial uses 0.75)")
    ap.add_argument("--crop-k", default=None, help="k0:k1 slab along the source S axis (voxels)")
    ap.add_argument("--crop-mm", default=None, help="z0:z1 slab in source world S (mm)")
    ap.add_argument("--fov-mm", type=float, default=None,
                    help="crop in-plane to this square FOV (Arterial's arch/seed heuristics assume ~220 mm, i.e. "
                         "512 px at 0.43 mm, centred on the neck)")
    ap.add_argument("--resample-mm", type=float, default=None,
                    help="resample the CT (trilinear) to this isotropic spacing before Arterial; the masks and "
                         "surfaces are then produced on this grid (Arterial was trained on ~0.43x0.43x0.4 mm)")
    ap.add_argument("--fov-center", default=None, help="x,y world (RAS mm) centre of the --fov-mm crop (default: image centre)")
    ap.add_argument("--skull-mask", default=None, help="skull mask on the --input grid (implies --slicer mask)")
    ap.add_argument("--slicer", default=None, choices=["ts", "laplacian", "mask"],
                    help="head/neck split method (default: mask if --skull-mask else ts)")
    ap.add_argument("--no-prep", action="store_true", help="use <case>/cta.nii.gz as is")
    a = ap.parse_args()

    stages = {"core": ["seg", "cl", "vl"], "full": ["seg", "cl", "lm", "vl", "fe", "ap"]}.get(a.stages)
    if stages is None:
        stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    slicer = a.slicer or ("mask" if a.skull_mask else "ts")

    mon = MemMonitor(); mon.start()
    report = {"input": a.input, "case": a.case, "stages": stages, "fast": a.fast, "tta": a.tta,
              "threads": a.threads, "slicer": slicer if not a.fast else "n/a (fast mode)",
              "times_s": {}, "stage_peak_ram_gb": {}, "errors": {}}
    t_all = time.time()
    if not a.no_prep:
        ck = tuple(int(x) for x in a.crop_k.split(":")) if a.crop_k else None
        cm = tuple(float(x) for x in a.crop_mm.split(":")) if a.crop_mm else None
        t0 = time.time()
        fc = tuple(float(x) for x in a.fov_center.split(",")) if a.fov_center else None
        report["prep"] = prepare_input(a.input, a.case, ck, cm, a.skull_mask, a.fov_mm, fc, a.resample_mm)
        report["times_s"]["prep"] = round(time.time() - t0, 1)

    patch_for_cpu(a.tta, a.threads, a.tile_step)
    patch_slicer(slicer, a.case)
    from arterial.run.processor import ArterialProcessor
    ns = argparse.Namespace(case_dir=a.case, cta_nifti_path=os.path.join(a.case, "cta.nii.gz"),
                            mode="extracranial_vessels", sampling_distance_mm=2, fast_segmentation=a.fast,
                            skip_segmentation=False, skip_centerline_extraction=False, skip_branching=False,
                            skip_clipping=False, skip_vessel_labelling=False, skip_feature_extraction=False,
                            skip_access_prediction=False, skip_landmark_detection=False, cl_dice_nnunet=False,
                            no_slicing=False, set_threshold_099=False)
    P = ArterialProcessor(ns)
    calls = {"seg": P.perform_segmentation, "cl": P.perform_centerline_extraction,
             "lm": P.perform_landmark_detection, "vl": P.perform_vessel_labelling,
             "fe": P.perform_feature_extraction, "ap": P.perform_access_prediction}
    ok = True
    for s in ["seg", "cl", "lm", "vl", "fe", "ap"]:
        if s not in stages:
            continue
        mon.stage_peak = 0
        t0 = time.time()
        print(f"\n===== stage {s} =====", flush=True)
        failed = False
        try:
            calls[s]()
        except Exception as e:  # keep going so later independent stages still report
            report["errors"][s] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            ok = False
            failed = True
        report["times_s"][s] = round(time.time() - t0, 1)
        report["stage_peak_ram_gb"][s] = round(mon.stage_peak / 2**30, 2)
        print(f"===== stage {s}: {report['times_s'][s]} s, peak {report['stage_peak_ram_gb'][s]} GB", flush=True)
        with open(os.path.join(a.case, "run_report.json"), "w") as f:
            json.dump(report, f, indent=1)
        if failed and s in ("seg", "cl"):
            break
    report["times_s"]["total"] = round(time.time() - t_all, 1)
    mon.stop()
    peak = _read_int("/sys/fs/cgroup/memory.peak")
    report["peak_ram_gb_cgroup_memory.peak"] = None if peak is None else round(peak / 2**30, 2)
    report["peak_ram_gb_sampled"] = round(mon.peak / 2**30, 2)
    report["maxrss_gb_self"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    report["maxrss_gb_children"] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20, 2)
    files = []
    for root, _, fs in os.walk(a.case):
        for fn in fs:
            p = os.path.join(root, fn)
            files.append([os.path.relpath(p, a.case), os.path.getsize(p)])
    report["files"] = sorted(files)
    report["ok"] = ok
    with open(os.path.join(a.case, "run_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ("times_s", "stage_peak_ram_gb", "peak_ram_gb_cgroup_memory.peak",
                                             "peak_ram_gb_sampled", "errors", "ok")}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
