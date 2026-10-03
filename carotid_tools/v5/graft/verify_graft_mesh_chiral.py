#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\verify_graft_mesh_chiral.py (= /sp/v5_va/verify_graft_mesh_chiral.py in the build containers)
"""verify_graft_mesh: chirality of the host frame vs VMR 0248's frame, from anatomy alone (eve-training-fixed).

sign = ((right - left) x (anterior - posterior)) . (superior)  -> +1 for an anatomically true (right-handed RAS-like)
geometry in a right-handed frame, -1 for a mirror image. A proper rotation keeps the sign, a mirror flips it, so a
"native" donor keeps the host's chirality only if both signs agree.
Also prints the radius-step locations from gates_proto.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/sp/v5_va")


def read_curve(path):
    d = json.load(open(path, encoding="utf-8"))
    pos = np.array([cp["position"] for cp in d["markups"][0]["controlPoints"]], float)
    return np.stack([pos[:, 1], -pos[:, 2], -pos[:, 0]], 1)


def at_height(p, z, axis):
    h = p @ axis
    i = int(np.argmin(np.abs(h - z)))
    return p[i]


def sign_of(right_va, left_va, right_cca, up):
    up = up / np.linalg.norm(up)
    # heights: 40 mm above the right VA origin along `up`
    z0 = right_va[0] @ up + 40.0
    rv, lv, rc = at_height(right_va, z0, up), at_height(left_va, z0, up), at_height(right_cca, z0, up)
    right = rv - lv
    ant = rc - rv
    ant = ant - (ant @ up) * up
    right = right - (right @ up) * up
    return float(np.sign(np.cross(right, ant) @ up)), {"right": right.tolist(), "anterior": ant.tolist(), "up": up.tolist()}


def main():
    res = {}
    for name in sys.argv[1:]:
        c = "/repo/carotid_data/anatomies_v4/%s/Centrelines_comb" % name
        RVA = read_curve(os.path.join(c, "Centerline curve - RVA.mrk.json"))
        LVA = read_curve(os.path.join(c, "Centerline curve - LVA.mrk.json"))
        RCCA = read_curve(os.path.join(c, "Centerline curve - RCCA.mrk.json"))
        sub17 = read_curve(os.path.join(c, "Centerline curve (17).mrk.json"))
        io = int(np.argmin(np.linalg.norm(RVA - sub17[0], axis=1)))
        sg, info = sign_of(RVA[io:], LVA, RCCA, np.array([0.0, 0.0, 1.0]))
        res["host:" + name] = sg
        print("host", name, "sign", sg, json.dumps(info), "RCCA x mean %.1f LCCA? RVA x %.1f LVA x %.1f" % (
            RCCA[:, 0].mean(), RVA[io:, 0].mean(), LVA[:, 0].mean()), flush=True)
    from sources import load_vmr0248
    v = load_vmr0248(with_ct=False, with_model=False, with_walls=False)
    P = {k: np.asarray(x["points"], float) for k, x in v["paths"].items()}
    print("VMR paths:", sorted(P), flush=True)
    rv = np.asarray(v["vessels"]["right_vertebral"]["points"], float)
    lv = np.asarray(v["vessels"]["left_vertebral"]["points"], float)
    up = rv[min(150, len(rv) - 1)] - rv[0]
    sg, info = sign_of(rv, lv, P["right_carotid"], up)
    res["vmr_native"] = sg
    print("VMR native sign", sg, json.dumps(info), flush=True)
    M = np.array([-1.0, 1.0, 1.0])
    # follow mode: the RVA takes VMR's LEFT VA mirrored, the "right carotid" is VMR's left carotid mirrored
    sg2, info2 = sign_of(lv * M, rv * M, P["left_carotid"] * M, up * M)
    res["vmr_mirrored_follow"] = sg2
    print("VMR mirrored (follow) sign", sg2, json.dumps(info2), flush=True)
    # radius steps
    g = json.load(open("/sp/v5_va/verify_graft_mesh_out/gates_proto.json"))
    for n, o in g.items():
        for sd in ("right", "left"):
            x = o["sides"][sd]
            print("dr", n[:24], sd, "max step %.2f at %.1f mm past O_h; stub %.1f mm; cut %.1f" % (
                x["r_max_step"], x["r_max_step_at_mm"], x["recorded"]["stub_mm"], x["recorded"]["cut_at_mm"]))
    json.dump(res, open("/sp/v5_va/verify_graft_mesh_out/chirality.json", "w"), indent=1)


if __name__ == "__main__":
    main()
