"""HOST (python 3.13): extra M2b input -- closed surface of uterus|HR-CTV|IUcanal|vagina (one connected body) for
the vagina-in-body variant (config body_mesh='pre_bodyvag').  Same marching-cubes / smoothing / decimation route
and the same gates as prep_inputs.surface (0 open or non-manifold edges, |mesh - label volume| < 2 %).  Units mm, cc.

    python prep_m2.py        # -> MRI_GYN_sim/inputs/pre_bodyvag.obj, logs/prep_m2.json
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import geom  # noqa: E402
import prep_inputs as pi  # noqa: E402   (module import only builds paths; main() is not run)

P = config.paths()


def main():
    ut, A = pi.lab("preBT", "uterus"); hr, _ = pi.lab("preBT", "HR-CTV"); iu, _ = pi.lab("preBT", "IUcanal"); vg, _ = pi.lab("preBT", "vagina")
    m = ut | hr | iu | vg
    labm, k = ndi.label(m, np.ones((3, 3, 3)))
    body = pi.largest(ndi.binary_fill_holes(m))
    V, F, st = pi.surface(body, A, 5000)
    st.update(n_components_union=int(k), fill_holes_added_vox=int(ndi.binary_fill_holes(m).sum() - m.sum()))
    geom.write_obj(os.path.join(P["inputs"], "pre_bodyvag.obj"), V, F,
                   header="pre_bodyvag: uterus|HR-CTV|IUcanal|vagina, preBT world RAS mm; derived from labels (local only)")
    json.dump(dict(pre_bodyvag=st), open(os.path.join(P["logs"], "prep_m2.json"), "w"), indent=1)
    print(json.dumps(st))
    if not st["gate_ok"]:
        sys.exit("pre_bodyvag failed the surface gate")


if __name__ == "__main__":
    main()
