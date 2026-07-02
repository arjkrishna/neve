"""Build a curated per-daughter AWAC seed npz from multi-target heatup chunks.

Plan v13 — generalization of the (scratchpad) builder that produced
``lcca_awac_seed_v1.npz`` (the D1+D2 recipe), now parameterized per daughter
and pointed at any harvest chunk dir (e.g. ``saved/heatup_z345_obsv3``).

Reads ONLY the Version-A chunks (``heatup_targetpart_<D>_w*_p*_c*.npz``) —
the complete partition of every harvested episode by grading target — so no
episode is duplicated (Version-B ``endedin`` files are a subset view).

Curation recipe per daughter D (mirrors the v1 LCCA seed):
  KEEP ALL   is_clean episodes with NOT overshoot  (the clean lane, D1)
  KEEP ALL   grader_success (even unclean)
  KEEP ALL   near-misses: received_correct_daughter OR ever-at-ostium
             (guidance ``is_at_ostium`` flag, flat_obs column 74 — unchanged
             by obs-v3 since new dims were appended after index 75)
  CAP        wrong-branch committers (received_wrong, no correct contact)
             at --cap_wrong (reservoir-thinned, deterministic rng)
  CAP        the remainder (mostly trunk/vessel_end) at --cap_trunk
  KEEP ALL   recovery episodes regardless of caps: received_wrong AND
             ended clean — the retraction demos (fable.md concern #1).

Output npz uses the exact experience_cache schema (same keys as the chunks),
loadable via --heatup_cache_file.

Usage (host, no SOFA needed):
  python "training _scripts/build_daughter_seed.py" \
      --chunks_dir saved/heatup_z345_obsv3 --daughter LCCA \
      --out saved/lcca_awac_seed_obsv3.npz --cap_wrong 3000 --cap_trunk 1500
"""

import argparse
import glob
import os
import sys

import numpy as np

OSTIUM_COL = 74  # flat_obs index of guidance is_at_ostium (46 + 28)

META_KEYS = [
    "meta_return", "meta_reached", "meta_is_demo", "meta_is_clean",
    "meta_target_branch_idx", "meta_target_branch_short",
    "meta_final_branch_short", "meta_received_correct_daughter",
    "meta_received_wrong_daughter", "meta_grader_success",
    "meta_grader_failure_timeout", "meta_overshoot",
]


def iter_chunk_episodes(path):
    """Yield per-episode dicts from one chunk npz."""
    d = np.load(path, allow_pickle=True)
    n = int(d["n_episodes"])
    lengths = d["lengths"].astype(int)
    flat_obs = d["flat_obs"]
    actions = d["actions"]
    rewards = d["rewards"]
    terminals = d["terminals"]
    meta = {k: d[k] for k in META_KEYS if k in d.files}
    o0 = a0 = 0
    for i in range(n):
        L = int(lengths[i])
        ep = {
            "obs": flat_obs[o0:o0 + L + 1],
            "actions": actions[a0:a0 + L],
            "rewards": rewards[a0:a0 + L],
            "terminals": terminals[a0:a0 + L],
            "length": L,
        }
        for k, arr in meta.items():
            ep[k] = arr[i]
        o0 += L + 1
        a0 += L
        yield ep


def classify(ep):
    """Return the curation class for an episode."""
    clean = bool(ep.get("meta_is_clean", False))
    overshoot = bool(ep.get("meta_overshoot", False))
    success = bool(ep.get("meta_grader_success", False))
    rc = bool(ep.get("meta_received_correct_daughter", False))
    rw = bool(ep.get("meta_received_wrong_daughter", False))
    ever_ostium = bool(np.any(ep["obs"][:, OSTIUM_COL] > 0.5))
    if rw and (clean or success):
        return "recovery"           # retraction demo — keep unconditionally
    if clean and not overshoot:
        return "clean"
    if success:
        return "success_unclean"
    if rc or ever_ostium:
        return "near_miss"
    if rw:
        return "wrong_branch"
    return "trunk_other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks_dir", required=True)
    p.add_argument("--daughter", required=True,
                   choices=["RCCA", "LCCA", "RVA", "LVA"])
    p.add_argument("--out", required=True)
    p.add_argument("--cap_wrong", type=int, default=3000)
    p.add_argument("--cap_trunk", type=int, default=1500)
    p.add_argument("--cap_near_miss", type=int, default=3000,
                   help="cap the (broad) near-miss class: received_correct "
                        "OR ever-at-ostium episodes that didn't succeed")
    p.add_argument("--rng_seed", type=int, default=7)
    args = p.parse_args()

    pattern = os.path.join(
        args.chunks_dir, f"heatup_targetpart_{args.daughter}_w*_p*_c*.npz"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"NO CHUNKS match {pattern}", file=sys.stderr)
        sys.exit(2)
    print(f"{len(files)} Version-A chunks for {args.daughter}")

    keep = {"clean": [], "success_unclean": [], "near_miss": [],
            "recovery": []}
    pool = {"wrong_branch": [], "trunk_other": []}
    obs_dim = None
    bad = 0
    for fi, f in enumerate(files):
        try:
            for ep in iter_chunk_episodes(f):
                if obs_dim is None:
                    obs_dim = ep["obs"].shape[1]
                elif ep["obs"].shape[1] != obs_dim:
                    bad += 1
                    continue
                c = classify(ep)
                (keep if c in keep else pool)[c].append(ep)
        except Exception as e:  # corrupt partial chunk (mid-write) — skip
            print(f"  skip {os.path.basename(f)}: {e}")
        if (fi + 1) % 200 == 0:
            print(f"  ... {fi+1}/{len(files)} files")

    rng = np.random.default_rng(args.rng_seed)

    def thin(lst, cap):
        if len(lst) <= cap:
            return lst
        idx = rng.choice(len(lst), size=cap, replace=False)
        return [lst[i] for i in sorted(idx)]

    wrong = thin(pool["wrong_branch"], args.cap_wrong)
    trunk = thin(pool["trunk_other"], args.cap_trunk)
    near = thin(keep["near_miss"], args.cap_near_miss)

    all_eps = (keep["clean"] + keep["recovery"] + keep["success_unclean"]
               + near + wrong + trunk)
    counts = {k: len(v) for k, v in keep.items()}
    counts["near_miss_kept"] = len(near)
    counts["wrong_branch_kept"] = len(wrong)
    counts["wrong_branch_total"] = len(pool["wrong_branch"])
    counts["trunk_other_kept"] = len(trunk)
    counts["trunk_other_total"] = len(pool["trunk_other"])
    n_trans = sum(e["length"] for e in all_eps)
    print(f"composition: {counts}")
    print(f"TOTAL: {len(all_eps)} episodes / {n_trans} transitions "
          f"/ obs_dim={obs_dim} / skipped_dim_mismatch={bad}")

    # ---- write out in the experience_cache schema ----
    lengths = np.array([e["length"] for e in all_eps], dtype=np.int32)
    flat_obs = np.concatenate([e["obs"] for e in all_eps]).astype(np.float32)
    actions = np.concatenate([e["actions"] for e in all_eps])
    rewards = np.concatenate([e["rewards"] for e in all_eps])
    terminals = np.concatenate([e["terminals"] for e in all_eps])
    out = {
        "n_episodes": np.int32(len(all_eps)),
        "lengths": lengths,
        "flat_obs": flat_obs,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "position": np.int32(0),
    }
    for k in META_KEYS:
        vals = [e.get(k) for e in all_eps]
        out[k] = np.array(vals)
    tmp = args.out + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, args.out)
    sz = os.path.getsize(args.out) / 1e6
    print(f"WROTE {args.out} ({sz:.0f} MB)")
    n_clean_lane = counts["clean"] + counts["recovery"]
    print(f"clean-lane episodes (is_clean & !overshoot + recovery): "
          f"{n_clean_lane} ({100.0*n_clean_lane/max(len(all_eps),1):.1f}%)")


if __name__ == "__main__":
    main()
