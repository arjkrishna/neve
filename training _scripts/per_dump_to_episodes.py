"""Convert a runner-saved PER buffer dump (checkpoints/replay_buffer.npz)
into the episode-cache schema loadable by --heatup_cache_file.

Plan v13 — built after the lcca_awac_obsv3 resume incident: the per-eval
replay_buffer.npz is the PER buffer's transition-level state
(obs_pairs/tree/...), NOT the experience_cache episode schema, so pointing
--heatup_cache_file at it wedges the loader. This tool makes buffer-level
resume possible: PER dump -> episode npz -> seed the restarted run.

Lossless where it matters: obs (from obs_pairs), actions, rewards,
terminals, and the per-transition is_clean / is_demo / episode_returns
companion arrays (so the balanced-lane flags survive). Ring order is
insertion order when the buffer never wrapped (n < capacity, the common
case); a wrapped buffer is unrolled from `position`. A trailing partial
episode (no terminal) is dropped.

Usage:
  python "training _scripts/per_dump_to_episodes.py" \
      --dump <run>/checkpoints/replay_buffer.npz \
      --out saved/<name>_resume_buffer.npz
"""

import argparse
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    d = np.load(args.dump, allow_pickle=True)
    n = int(d["n"])
    cap = int(d["capacity"])
    pos = int(d["position"])

    order = np.arange(n)
    if n >= cap:  # wrapped ring: oldest entry is at `position`
        order = np.concatenate([np.arange(pos, cap), np.arange(0, pos)])

    op = d["obs_pairs"][:cap][order]
    ac = d["actions"][:cap][order]
    rw = d["rewards"][:cap][order]
    tm = d["terminals"][:cap][order].astype(bool)
    ic = d["is_clean"][:cap][order] if "is_clean" in d.files else np.zeros(n, bool)
    dm = d["is_demo"][:cap][order] if "is_demo" in d.files else np.zeros(n, bool)
    er = (d["episode_returns"][:cap][order]
          if "episode_returns" in d.files else np.zeros(n))

    ends = np.flatnonzero(tm)
    if len(ends) == 0:
        raise SystemExit("no terminal transitions — nothing to convert")
    dropped_tail = n - 1 - int(ends[-1])

    lengths, obs_parts, meta = [], [], {
        "meta_is_clean": [], "meta_is_demo": [], "meta_return": [],
        "meta_reached": [],
    }
    start = 0
    for e in ends:
        e = int(e)
        L = e - start + 1
        lengths.append(L)
        # flat_obs: s_0..s_{L-1} from obs_pairs[:,0], plus terminal s_L
        obs_parts.append(np.vstack([op[start:e + 1, 0], op[e, 1][None]]))
        meta["meta_is_clean"].append(bool(ic[e]))
        meta["meta_is_demo"].append(bool(dm[e]))
        ret = float(er[e]) if er[e] != 0.0 else float(rw[start:e + 1].sum())
        meta["meta_return"].append(ret)
        # reached proxy for loaders that key the clean lane off `reached`
        meta["meta_reached"].append(bool(ic[e]))
        start = e + 1

    n_eps = len(lengths)
    out = {
        "n_episodes": np.int32(n_eps),
        "lengths": np.array(lengths, dtype=np.int32),
        "flat_obs": np.concatenate(obs_parts).astype(np.float32),
        "actions": ac[: start],
        "rewards": rw[: start],
        "terminals": tm[: start],
        "position": np.int32(0),
    }
    for k, v in meta.items():
        out[k] = np.array(v)

    assert out["flat_obs"].shape[0] == sum(lengths) + n_eps
    assert out["actions"].shape[0] == sum(lengths)

    tmp = args.out + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, args.out)
    print(f"episodes={n_eps}  transitions={sum(lengths)}  "
          f"clean={int(np.sum(out['meta_is_clean']))}  "
          f"demo={int(np.sum(out['meta_is_demo']))}  "
          f"dropped_tail_transitions={dropped_tail}  "
          f"obs_dim={out['flat_obs'].shape[1]}")
    print(f"WROTE {args.out} ({os.path.getsize(args.out)//1048576} MB)")


if __name__ == "__main__":
    main()
