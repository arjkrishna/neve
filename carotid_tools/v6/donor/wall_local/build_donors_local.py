#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_local\build_donors_local.py (= /sp/v6/wall_local/build_donors_local.py in the build containers)
"""build_donors_local.py (s2_harden_wall_bake, D15): the donor records (donors_v5 schema) of the LOCAL walls.

= v6_wall's build_donors6.py (imported unchanged: v5's builder graft_v5_reg_donor.build_unit with the wall files
swapped, landmarks remapped onto the new line, v5's edge / measure copied, VMR 0248 through v5's build_vmr) with the
variant "local" read from /v6w/seg/local and written to /v6w/donors_local (= carotid_data/v6_work/wall/donors_local;
every /v6w path rewritten to /repo/carotid_data/v6_work/wall/...). Each side record's "v6" block gains "d15": the zone
runs, the zone share, the splice record and the surface identity check (local_qc.json).
    python3 build_donors_local.py [unit,...]       (eve-training-fixed, as build_donors6)
"""
import json
import os
import sys

sys.argv = [sys.argv[0], "local"] + sys.argv[1:]
sys.path.insert(0, "/sp/v6/wall")
import build_donors6 as BD  # noqa: E402

BD.SEG = "/v6w/seg/local"
BD.OUT = "/v6w/donors_local"
VERSION = "build_donors_local 1.0 (2026-10-04, s2_harden_wall_bake)"


def add_d15():
    for unit in sorted(os.listdir(BD.OUT)):
        qf = os.path.join(BD.SEG, unit, "local_qc.json")
        if not os.path.exists(qf):
            continue
        q = json.load(open(qf))
        for sd in ("R", "L"):
            f = os.path.join(BD.OUT, unit, "%s.json" % sd)
            if not os.path.exists(f) or sd not in q["sides"]:
                continue
            r = json.load(open(f))
            v = "rva" if sd == "R" else "lva"
            vj = json.load(open(os.path.join(BD.SEG, unit, "%s_stageA_vmtk.json" % v)))
            qs = q["sides"][sd]
            r.setdefault("v6", {})["d15"] = {
                "builder": VERSION, "rule": "D15: v5's ct_seg4 wall outside the artefact zones, stage 1's plateau inside "
                                            "(2 mm blend); surface vertices outside the zones = v5's; line = v5's outside",
                "zones": qs["zones"], "surface_check": {k: qs["surface"][k] for k in (
                    "v5_recompute_equals_v5_file", "vertices", "vertices_outside_zones", "outside_vertices_equal_v5_file",
                    "triangles_outside", "triangles_outside_equal_v5", "spill_undone_max_mm", "topology", "topology_v5")},
                "voxels": qs["voxels"], "splice": vj.get("splice")}
            json.dump(BD.rewrite(r), open(f, "w"), indent=1)
    json.dump({"what": "v6 donor VA records, D15 LOCAL wall (s2_harden_wall_bake): v5's ct_seg4 wall outside the "
                       "artefact zones (bone contact + floored stretches, 2 mm blend), stage 1's plateau rule inside",
               "variant": "local", "donor_root": "/repo/carotid_data/v6_work/wall/donors_local",
               "use": "graft_v5_reg-style grafters: --donor-root /repo/carotid_data/v6_work/wall/donors_local (env "
                      "V6VA_DONOR_ROOT for graft_v6_va); same schema as donors_v5 / donors/plateau",
               "subclavian_junctions": "/repo/carotid_data/v6_work/wall/subclavian_local/<donor_id>/ (same file names as "
                                       "wall/subclavian/<donor_id>/); VA-tool junction records: "
                                       "/repo/carotid_data/v6_work/wall/donors_local/_junction_records/<donor>/<R|L>/junction.json "
                                       "(env V6VA_SEG)",
               "builder": VERSION, "notes": "scratchpad/v6/notes/s2_harden_wall_bake.md"},
              open(os.path.join(BD.OUT, "CHOSEN.json"), "w"), indent=1)


if __name__ == "__main__":
    BD.main()
    add_d15()
