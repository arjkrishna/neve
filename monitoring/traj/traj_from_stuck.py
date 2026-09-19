#!/usr/bin/env python
"""traj_from_stuck.py -- convert saved/stuck *episode*-format records
(proj[] gw[] cmd[] cs[] fold[] ...) into the same feature schema as
traj_extract.py, for runs whose worker logs are not on this machine.

usage: python traj_from_stuck.py <in.jsonl[.gz]> <out_prefix> --tag T [--eval]
Fields the stuck records lack (catheter insertion, on_path, phys, tip, ...)
come out NaN / -1; `cmd` in these records is already |cmd|, so the signed
detector variant (evS_*) is not computable.
"""
import sys, gzip, json, argparse
import numpy as np
from traj_extract import core_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out_prefix"); ap.add_argument("--tag", required=True)
    ap.add_argument("--eval", action="store_true", help="force is_eval=True for every record")
    a = ap.parse_args()
    op = gzip.open if a.inp.endswith(".gz") else open
    fo = open(a.out_prefix + ".features.jsonl", "w")
    S = {k: [] for k in ("proj", "gw", "fold", "cmd0")}; offs = [0]; keys = []; n = 0
    with op(a.inp, "rt") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            proj = r.get("proj") or []
            if len(proj) < 2: continue
            gw = r["gw"]; cmd = r["cmd"]; fold = r.get("fold") or [0] * len(proj)
            cs = r.get("cs")
            f, ev = core_features(proj, gw, None, cmd, fold, r.get("pl"))
            seed = r.get("seed")
            row = dict(tag=a.tag, pid=str(r.get("tag") or ""), ep=i, is_eval=bool(a.eval or seed is not None),
                       seed=seed, anatomy=None, mesh_fp=None, target=None, target_branch="RCCA",
                       wt_start=r.get("wt"), gsteps_start=-1, worker_file=a.inp.split("/")[-1],
                       complete=r.get("reason") is not None, reason=r.get("reason"), final_branch=None,
                       success=bool(r.get("succ")), grader_success=None, ret=None, out_steps=None,
                       pl=r.get("pl"), steps=len(proj), events=ev)
            row.update(f)
            nan = float("nan")
            row.update(dict(cmd_retract_frac=nan, evS_n=-1, evS_unrec=-1, cmd_cath_retract_frac=nan,
                            gw_rot_abs_mean=nan, cath_rot_abs_mean=nan, off_steps=-1, off_excursions=-1,
                            off_final=-1, off_last100_frac=nan, off_run_max=-1, off_br_max=-1,
                            phys_bridge=-1, phys_RCCA=-1, phys_RVA=-1, phys_LCCA=-1, phys_LVA=-1,
                            phys_other=-1, phys_rva_first_frac=-1.0, phys_final=-1, xt_max=nan, xt_mean=nan,
                            local_r_min=nan, dtgt_final=nan, dtgt_min=nan, tip_path_mm=nan,
                            tip_tortuosity=nan, tip_backsteps=-1, bphi_min=nan, bphi_mean=nan,
                            herr_abs_mean=nan, rew_sum=nan, rew_neg_steps=-1, rew_min=nan,
                            daughters_max=-1, entries_max=-1, overshoot_any=-1, wall_dur_s=nan))
            if cs:
                CS = np.asarray(cs, float)
                row.update(cath_slack_max=float(CS.max()), cath_slack_final=float(CS[-1]),
                           cath_slack_ge50=int((CS > 50).sum()))
            else:
                row.update(cath_slack_max=nan, cath_slack_final=nan, cath_slack_ge50=-1)
            fo.write(json.dumps(row) + "\n")
            S["proj"].append(np.clip(np.round(np.asarray(proj) * 10), -32000, 32000).astype(np.int16))
            S["gw"].append(np.clip(np.round(np.asarray(gw) * 10), -32000, 32000).astype(np.int16))
            S["fold"].append(np.clip(np.asarray(fold), 0, 32000).astype(np.int16))
            S["cmd0"].append(np.clip(np.round(np.asarray(cmd) * 100), -32000, 32000).astype(np.int16))
            offs.append(offs[-1] + len(proj)); keys.append("%s|%s|%d" % (a.tag, row["pid"], i)); n += 1
    fo.close()
    arrs = {k: np.concatenate(v) for k, v in S.items()}
    arrs["offsets"] = np.asarray(offs, np.int64); arrs["keys"] = np.asarray(keys)
    np.savez_compressed(a.out_prefix + ".series.npz", **arrs)
    sys.stderr.write("[%s] episodes=%d\n" % (a.tag, n))


if __name__ == "__main__":
    main()
