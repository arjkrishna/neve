"""HOST (python 3.13): rigid preBT <-> BT alignments used by the stage-2 validation.  Units mm, deg.

    python align.py      # -> MRI_GYN_sim/validation/alignment.json and MRI_GYN_sim/inputs/poses_align.json

Convention (as the registration stage, registration/common.py): a rigid transform (R, t) maps BT (fixed) world
points to preBT (moving) world points,  y_pre = R x_BT + t.  preBT anatomy / simulation results are carried into
the BT frame by the inverse,  x_BT = R^T (y_pre - t).  The BT applicator is carried into preBT by the forward map,
which defines the final applicator pose of the simulation in that frame, so the simulated device lands exactly on
the real one when the result is mapped back (E_app == T^-1, asserted in evaluate.py).

Frames (choice and justification)
- BONE (PRIMARY; the key name is historical -- it is a PERI-ORGAN MI frame, not a bone registration): Mattes-MI rigid
  registration of the two MR images restricted to the mask  box(all organs + 60 mm) & body & (distance to the BT
  organ labels + applicator > 10 mm)  (registration/s2c_rigid_mi_v2.py -> out/rigid_mi2_bonyPelvis.npz).  The mask
  holds all non-organ pelvic tissue (bone, fat, muscle, bowel), not only bone, and the BT organ/applicator labels are
  used to build it (exclusion only).  No organ label enters the similarity metric, so uterus, HR-CTV and vagina are
  held out of the fit; it was NOT verified against bony landmarks.  Verified upstream: 6 starts (rotations -20..+20 deg
  about L-R and a label-based start) converge to the same optimum; whole-image MI (rigid_mi2_global) and an
  independent NMI re-implementation (verify_registration/v2) agree to < 0.6 deg / < 0.5 mm; bladder Dice 0.79.
  Organs at the applicator are displaced ~24 mm relative to bone between the sessions, so in this frame the
  simulation must reproduce a large motion, which is exactly what a predictive model has to do.
- HR (SECONDARY; the M1 design frame, pose P2): HR-CTV-only surface ICP (proto/design_contact/check_hrctv_icp.json,
  mean residual 2.34 mm).  It uses the HR-CTV, so in this frame the HR-CTV is NOT held out (its rigid baseline is
  the ICP optimum for that very structure); uterus and vagina are held out.
- GLOBAL (sensitivity of the bone frame): whole-image MI (rigid_mi2_global.npz).
- UH (leaky ceiling "B1", never a validation frame): uterus+HR-CTV label rigid (rigid_label_uterus+HRCTV.npz).
Bladder/rectum/sigmoid ICP frames were rejected: filling changes and packing move those organs (bladder+rectum ICP
residual 5.4 mm; rectum Dice 0.39).
"""
import json
import os
import sys

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import geom  # noqa: E402

P = config.paths()
REG = P["prior"] + "/registration/out"
VAL = P["out"] + "/validation"


def load_mask(study, name):
    im = nib.load("%s/%s_MRI_label_%s.nii" % (P["data"], study, name))
    return np.asarray(im.dataobj) > 0, im.affine.copy()


def world(mask, aff):
    return np.argwhere(mask) @ aff[:3, :3].T + aff[:3, 3]


def signed_dist_fn(mask, aff):
    sp = np.abs(np.diag(aff)[:3])
    sd = np.where(mask, -ndi.distance_transform_edt(mask, sampling=sp), ndi.distance_transform_edt(~mask, sampling=sp))
    inv = np.linalg.inv(aff)

    def f(X):
        ijk = (np.atleast_2d(X) @ inv[:3, :3].T + inv[:3, 3]).T
        return ndi.map_coordinates(sd, ijk, order=1, mode="nearest")
    return f


def frames():
    icp = json.load(open(P["prior"] + "/proto/design_contact/check_hrctv_icp.json"))["HRCTV_only_icp"]
    zb = np.load(REG + "/rigid_mi2_bonyPelvis.npz"); zg = np.load(REG + "/rigid_mi2_global.npz")
    zu = np.load(REG + "/rigid_label_uterus+HRCTV.npz")
    return {
        "BONE": dict(R=zb["R"], t=zb["t"], role="PRIMARY", pose="PB", uses_labels=[],
                     source="registration/out/rigid_mi2_bonyPelvis.npz (MI, bony-pelvis mask)",
                     held_out=["uterus", "HR-CTV", "vagina"]),
        "HR": dict(R=np.array(icp["R"]), t=np.array(icp["t"]), role="SECONDARY (M1 design frame)", pose="P2",
                   uses_labels=["HR-CTV"], source="proto/design_contact/check_hrctv_icp.json (HR-CTV-only ICP)",
                   held_out=["uterus", "vagina"], icp_mean_resid_mm=icp["icp_mean_resid_mm"]),
        "GLOBAL": dict(R=zg["R"], t=zg["t"], role="SENSITIVITY (bone frame)", pose="PBg", uses_labels=[],
                       source="registration/out/rigid_mi2_global.npz (MI, whole image)",
                       held_out=["uterus", "HR-CTV", "vagina"]),
        "UH": dict(R=zu["R_dice"], t=zu["t_dice"], role="LEAKY CEILING (B1), never validation", pose="P2u",
                   uses_labels=["uterus", "HR-CTV"], source="registration/out/rigid_label_uterus+HRCTV.npz",
                   held_out=["vagina"]),
    }


def main():
    os.makedirs(VAL, exist_ok=True)
    app = json.load(open(P["inputs"] + "/applicator.json")); po = json.load(open(P["inputs"] + "/poses.json"))
    Fb = np.array(app["origin_BT_world"]); Rr = np.array(app["R_rows_BT_world"]); L = float(app["L_iu_mm"])
    sph_app = np.array([s["center_app_mm"] for s in app["spheres"]]); sph_r = np.array([s["r_mm"] for s in app["spheres"]])
    L_end = np.array(po["L_end"]); a0 = np.array(po["a0"]); O = np.array(po["O_pre"])
    ut, A = load_mask("preBT", "uterus"); hr, _ = load_mask("preBT", "HR-CTV"); iu, _ = load_mask("preBT", "IUcanal")
    sd_ut = signed_dist_fn(ut, A); sd_body = signed_dist_fn(ut | hr | iu, A)
    utb, Ab = load_mask("BT", "uterus"); hrb, _ = load_mask("BT", "HR-CTV")
    c_ut_bt = world(utb, Ab).mean(0); c_hr_bt = world(hrb, Ab).mean(0)
    c_ut_pre = world(ut, A).mean(0); c_hr_pre = world(hr, A).mean(0)
    fr = frames(); out = dict(units="mm, deg", convention="y_pre = R x_BT + t ; x_BT = R^T (y_pre - t)", frames={})
    poses_align = dict(units="mm; preBT world RAS", note="BT applicator mapped into preBT by each frame (align.py)")
    for name, f in fr.items():
        R, t = np.asarray(f["R"], float), np.asarray(f["t"], float)
        F = R @ Fb + t; a = geom.unit(R @ Rr[2]); x = geom.ortho(R @ Rr[0], a); tip = F + L * a
        spc = (Fb + sph_app @ Rr) @ R.T + t
        off = F - L_end; ax = float(off @ a0); lat = float(np.linalg.norm(off - ax * a0))
        inv = lambda y: (np.asarray(y) - t) @ R  # noqa: E731   preBT -> BT
        rec = dict(role=f["role"], source=f["source"], uses_labels=f["uses_labels"], held_out=f["held_out"],
                   R_BT_to_pre=R.tolist(), t_BT_to_pre=t.tolist(), rotation_deg=round(geom.rot_angle_deg(R), 3),
                   pose_name=f["pose"],
                   BT_device_in_preBT=dict(
                       flange=F.round(3).tolist(), axis=a.round(5).tolist(), x_app=x.round(5).tolist(), tip=tip.round(3).tolist(),
                       flange_above_L_end_along_a0_mm=round(ax, 2), flange_lateral_from_a0_line_mm=round(lat, 2),
                       flange_above_O_pre_along_a0_mm=round(float((F - O) @ a0), 2),
                       d_F_equiv_mm=round(float((L_end - F) @ a0), 2),
                       axis_angle_to_a0_deg=round(geom.angle_deg(a, a0), 2),
                       tip_signed_dist_to_preBT_uterus_mm=round(float(sd_ut(tip)[0]), 2),
                       flange_signed_dist_to_preBT_body_mm=round(float(sd_body(F)[0]), 2),
                       sphere_centres=spc.round(3).tolist(), sphere_r_mm=sph_r.tolist()),
                   preBT_centroid_to_BT_centroid_mm=dict(
                       uterus=round(float(np.linalg.norm(inv(c_ut_pre) - c_ut_bt)), 2),
                       HR_CTV=round(float(np.linalg.norm(inv(c_hr_pre) - c_hr_bt)), 2),
                       uterus_vec_BT_frame=(c_ut_bt - inv(c_ut_pre)).round(2).tolist()))
        if name in ("BONE", "GLOBAL", "UH"):
            if f["pose"] not in po:
                poses_align[f["pose"]] = dict(F=F.round(4).tolist(), a=a.round(6).tolist(), x=x.round(6).tolist(),
                                              R_BT_to_pre=R.tolist(), t_BT_to_pre=t.tolist(), frame=name,
                                              convention="y_pre = R x_BT + t", d_F_mm=round(float((L_end - F) @ a0), 3))
        out["frames"][name] = rec
        print("%-6s rot %5.1f deg  flange %s  above L_end %.1f (lat %.1f)  above O_pre %.1f  axis-a0 %.1f deg  "
              "tip sd(uterus) %+.1f  uterus centroid err %.1f" % (name, rec["rotation_deg"], F.round(1), ax, lat,
                                                                  rec["BT_device_in_preBT"]["flange_above_O_pre_along_a0_mm"],
                                                                  rec["BT_device_in_preBT"]["axis_angle_to_a0_deg"],
                                                                  rec["BT_device_in_preBT"]["tip_signed_dist_to_preBT_uterus_mm"],
                                                                  rec["preBT_centroid_to_BT_centroid_mm"]["uterus"]), flush=True)
    # agreement between frames, at the BT flange and the BT HR-CTV centroid (displacement of a BT point's preimage)
    agree = {}
    for n1, n2 in (("BONE", "GLOBAL"), ("BONE", "HR"), ("HR", "UH")):
        R1, t1 = np.asarray(fr[n1]["R"], float), np.asarray(fr[n1]["t"], float)
        R2, t2 = np.asarray(fr[n2]["R"], float), np.asarray(fr[n2]["t"], float)
        agree["%s_vs_%s" % (n1, n2)] = dict(rot_diff_deg=round(geom.rot_angle_deg(R1 @ R2.T), 3),
                                            at_BT_flange_mm=round(float(np.linalg.norm((R1 - R2) @ Fb + t1 - t2)), 2),
                                            at_BT_HRCTV_centroid_mm=round(float(np.linalg.norm((R1 - R2) @ c_hr_bt + t1 - t2)), 2))
    out["agreement"] = agree
    out["bone_frame_verification"] = dict(
        upstream="registration/out/s2c_log.txt: 6 starts -> identical optimum (sd of HR-CTV displacement 0.0 mm); "
                 "verify_registration/v2_log.txt: independent NMI re-run, 5 starts, rotation 11.3-12.3 deg vs 11.7, "
                 "HR-CTV displacement within 0.5 mm",
        bladder_dice_upstream=0.79, relative_organ_motion_vs_bone_mm=24.0,
        note="motion of the organs at the applicator relative to bone is real between-session motion (applicator, "
             "packing, filling), not registration error: bone-frame MI variants agree to < 0.6 deg")
    json.dump(out, open(VAL + "/alignment.json", "w"), indent=1)
    json.dump(poses_align, open(P["inputs"] + "/poses_align.json", "w"), indent=1)
    print(json.dumps(agree, indent=1))
    print("wrote", VAL + "/alignment.json", P["inputs"] + "/poses_align.json")


if __name__ == "__main__":
    main()
