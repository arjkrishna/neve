"""Serialization utilities for replay buffer episode data.

Format: compressed numpy archive (.npz) with flat layout + length index.
Variable-length episodes are concatenated along axis 0 with a lengths array
for reconstruction. This avoids object arrays which break np.savez_compressed.

File contents:
    n_episodes: int scalar
    lengths: int array (n_episodes,) — number of actions (steps) per episode
    flat_obs: float array (sum(lengths+1), obs_dim) — concatenated observations
    actions: float array (sum(lengths), action_dim) — concatenated actions
    rewards: float array (sum(lengths),) — concatenated rewards
    terminals: bool array (sum(lengths),) — concatenated terminal flags
    position: int scalar — ring buffer write position (for replay buffer restore)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def save_episodes_npz(path, episodes, position=0, metadata=None):
    """Save a list of episode numpy tuples to a compressed .npz file.

    Args:
        path: Output file path (should end in .npz).
        episodes: List of (flat_obs, actions, rewards, terminals) numpy tuples.
            flat_obs shape: (T+1, obs_dim)
            actions shape: (T, action_dim)
            rewards shape: (T,)
            terminals shape: (T,)
        position: Ring buffer write position (for replay buffer state).
        metadata: Plan v8 + Plan v12 — optional list (one dict per episode)
            of episode-quality flags. Supported keys (all optional,
            absent keys default to safe values):
              - "episode_return": float (Plan v8 — return-prioritized replay)
              - "reached_target_daughter": bool (Plan v8 — clean-thread flag
                used by the balanced sampler / outcome priority lane)
              - "is_demo": bool (Plan v8 — DQfD demo lane; Plan v12 keeps
                this False for all heatup-harvested Episodes)
              - "is_clean": bool (Plan v12 R17 — R17-safe definition:
                MUST be derived by the caller from `infos[-1]` strictly
                via `info['final_branch_idx'] == info['target_daughter_branch_idx']
                AND NOT truncated_as_failure`, NOT from any runtime latch
                like `env5._reached_target_daughter` which never resets
                after the wire touches the target daughter)
              - "target_branch_idx": int (Plan v12 — which virtual env's
                target this Episode was recorded against; -1 = single-target
                single-Episode-per-step legacy)
            None → flags omitted.
    """
    if not episodes:
        logger.warning("save_episodes_npz: empty episode list, nothing to save")
        return

    lengths = np.array([ep[1].shape[0] for ep in episodes], dtype=np.int32)

    all_obs = np.concatenate([ep[0] for ep in episodes], axis=0)
    all_actions = np.concatenate([ep[1] for ep in episodes], axis=0)
    all_rewards = np.concatenate([ep[2] for ep in episodes], axis=0)
    all_terminals = np.concatenate([ep[3] for ep in episodes], axis=0)

    save_kwargs = dict(
        n_episodes=np.int32(len(episodes)),
        lengths=lengths,
        flat_obs=all_obs,
        actions=all_actions,
        rewards=all_rewards,
        terminals=all_terminals,
        position=np.int32(position),
    )
    if metadata is not None:
        save_kwargs["meta_return"] = np.array(
            [m.get("episode_return", 0.0) for m in metadata], dtype=np.float64
        )
        save_kwargs["meta_reached"] = np.array(
            [m.get("reached_target_daughter", False) for m in metadata], dtype=bool
        )
        save_kwargs["meta_is_demo"] = np.array(
            [m.get("is_demo", False) for m in metadata], dtype=bool
        )
        # Plan v12 R17 — explicit is_clean field (R17-safe definition
        # caller-computed from infos[-1] strictly; see docstring above).
        # Default False when omitted preserves backward-compat: Stage 3
        # PER balanced_fraction lane treats absent is_clean as "not clean"
        # rather than crashing on missing column.
        save_kwargs["meta_is_clean"] = np.array(
            [m.get("is_clean", False) for m in metadata], dtype=bool
        )
        # Plan v12 — per-virtual-env target identity. -1 sentinel for
        # legacy single-target episodes (preserves backward-compat with
        # caches written before Plan v12).
        save_kwargs["meta_target_branch_idx"] = np.array(
            [m.get("target_branch_idx", -1) for m in metadata], dtype=np.int32
        )
        # Plan v12 — traceability fields per user requirement:
        # "make sure it is traceable which daughter each episode ended up
        # in since we would need to filter with daughters (success/
        # failures); success → target in daughter; reached daughter and
        # got daughter bonus reward 1.0".
        # Persisted as fixed-length numpy string arrays so load can return
        # exactly the same Python str values without object-array round-
        # tripping. 6-char string is enough for "RCCA"/"LCCA"/"RVA"/"LVA"/
        # "other"/"unknown".
        save_kwargs["meta_target_branch_short"] = np.array(
            [str(m.get("target_branch_short", "unknown"))[:8] for m in metadata],
            dtype="U8",
        )
        save_kwargs["meta_final_branch_short"] = np.array(
            [str(m.get("final_branch_short", "unknown"))[:8] for m in metadata],
            dtype="U8",
        )
        save_kwargs["meta_received_correct_daughter"] = np.array(
            [m.get("received_correct_daughter", False) for m in metadata],
            dtype=bool,
        )
        save_kwargs["meta_received_wrong_daughter"] = np.array(
            [m.get("received_wrong_daughter", False) for m in metadata],
            dtype=bool,
        )
        # Plan v12 — Challenge-2 deep-reach (per-grader) + failure-timeout
        # reason, for separate filtering of threaded vs deep-reached vs timed-out.
        save_kwargs["meta_grader_success"] = np.array(
            [m.get("grader_success", False) for m in metadata], dtype=bool
        )
        save_kwargs["meta_grader_failure_timeout"] = np.array(
            [m.get("grader_failure_timeout", False) for m in metadata], dtype=bool
        )
        # RL_IMPROV_8 OST — overshoot inside the correct daughter (soft -1, folds
        # into is_clean). Persisted for downstream OST-vs-clean filtering.
        save_kwargs["meta_overshoot"] = np.array(
            [m.get("overshoot", False) for m in metadata], dtype=bool
        )
    np.savez_compressed(path, **save_kwargs)
    total_steps = int(lengths.sum())
    logger.info(f"Saved {len(episodes)} episodes ({total_steps} steps) to {path}")


# Plan v12 Stage 2 — daughter short-names for Version-B ("snapshot version")
# bucketing. A heatup Episode belongs to daughter D's Version-B file iff the
# wire physically ENDED in D and D was this grader's target, i.e.
# final_branch_short == target_branch_short == D. This is the episode's OWN
# outcome (user decision 2026-06-23) — NOT the stricter is_clean, so it
# INCLUDES success / overshoot / wrong_branch_timeout / wire_fold_stall /
# unknown_truncation that ended in D.
_DAUGHTERS = ("RCCA", "LCCA", "RVA", "LVA")


def episode_metadata(ep):
    """Plan v12 R17 — per-Episode quality/outcome metadata, derived STRICTLY
    from infos[-1] (never a runtime latch). One shared definition for both the
    rolling worker-side heatup flush (save_heatup_batch) and the end-of-heatup
    save in runner.training_run."""
    last_info = ep.infos[-1] if getattr(ep, "infos", None) else {}
    fbs = str(last_info.get("final_branch_short", "unknown"))
    tbs = str(last_info.get("target_branch_short", "unknown"))
    rcd = bool(last_info.get("received_correct_daughter", False))
    grader_success = bool(last_info.get("grader_success", False))
    grader_timeout = bool(last_info.get("grader_failure_timeout", False))
    # is_clean = "threaded THIS daughter cleanly": ended in the target daughter,
    # committed at the correct fork, did NOT fail as a wrong-branch/fold timeout.
    # Does NOT require the deep target (that is grader_success, separate).
    is_clean = (
        fbs == tbs
        and fbs not in ("unknown", "other")
        and rcd
        and not grader_timeout
    )
    return {
        "episode_return": float(getattr(ep, "episode_reward", 0.0)),
        "reached_target_daughter": bool(is_clean),
        "is_demo": bool(getattr(ep, "is_demo", False)),
        "is_clean": bool(is_clean),
        "target_branch_idx": int(getattr(ep, "target_branch_idx", -1)),
        "final_branch_short": fbs,
        "target_branch_short": tbs,
        "received_correct_daughter": rcd,
        "received_wrong_daughter": bool(
            last_info.get("received_wrong_daughter", False)
        ),
        "grader_success": grader_success,
        "grader_failure_timeout": grader_timeout,
        "overshoot": bool(last_info.get("overshoot", False)),
    }


def save_heatup_batch(
    episodes,
    base_path,
    ordered_shorts=None,
    worker_id=0,
    chunk_idx=0,
    rolling=True,
    emit_version_b=True,
):
    """Plan v12 Stage 2 — write a batch of heatup Episodes to per-daughter
    .npz files in TWO versions:

      Version A (by target):  partition by ep.target_branch_idx — which daughter
        this Episode was GRADED for (mostly failures for that target).
      Version B ("snapshot"): partition by the episode's OWN physical end == its
        target, final_branch_short == target_branch_short in DAUGHTERS. ALL
        reasons (success/overshoot/WBT/WFS/unknown) — NOT is_clean.

    rolling=True (worker streaming): filenames carry _w{worker}_p{pid}_c{chunk}
        so concurrent workers + successive flushes never collide; the analyzer
        globs + concatenates the chunks.
    rolling=False (finite / main final flush): legacy Version-A filenames
        {base}_{short}{ext} (byte-for-byte parity with the prior end-of-heatup
        save), {base}{ext} for the single-target (-1) bucket; Version B as
        {base}_endedin_{short}{ext}.

    Returns {path: n_episodes} written ({} if nothing to write).
    """
    import os
    from collections import defaultdict

    if not episodes:
        return {}

    base, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".npz"
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    pid = os.getpid()
    written = {}

    def _tuples(eps):
        return [
            (np.array(e.flat_obs), np.array(e.actions),
             np.array(e.rewards), np.array(e.terminals))
            for e in eps
        ]

    pairs = [(e, episode_metadata(e)) for e in episodes]

    # ---- Version A: partition by target_branch_idx ----
    a_buckets = defaultdict(list)
    for e, m in pairs:
        a_buckets[int(m["target_branch_idx"])].append((e, m))
    for tbi, group in sorted(a_buckets.items()):
        if tbi < 0:
            short = None
        elif ordered_shorts is not None and 0 <= tbi < len(ordered_shorts):
            short = str(ordered_shorts[tbi])
        else:
            short = f"t{tbi}"
        eps = [g[0] for g in group]
        md = [g[1] for g in group]
        if rolling:
            tag = short if short is not None else "single"
            path = (f"{base}_targetpart_{tag}"
                    f"_w{worker_id}_p{pid}_c{chunk_idx:03d}{ext}")
        else:
            path = f"{base}_{short}{ext}" if short is not None else f"{base}{ext}"
        save_episodes_npz(path, _tuples(eps), metadata=md)
        written[path] = len(eps)

    # ---- Version B: partition by final==target in DAUGHTERS (all reasons) ----
    if emit_version_b:
        b_buckets = defaultdict(list)
        for e, m in pairs:
            fbs = m["final_branch_short"]
            if fbs == m["target_branch_short"] and fbs in _DAUGHTERS:
                b_buckets[fbs].append((e, m))
        for short, group in sorted(b_buckets.items()):
            eps = [g[0] for g in group]
            md = [g[1] for g in group]
            if rolling:
                path = (f"{base}_endedin_{short}"
                        f"_w{worker_id}_p{pid}_c{chunk_idx:03d}{ext}")
            else:
                path = f"{base}_endedin_{short}{ext}"
            save_episodes_npz(path, _tuples(eps), metadata=md)
            written[path] = len(eps)

    return written


def load_episodes_npz(path):
    """Load episodes from a compressed .npz file.

    Args:
        path: Input file path.

    Returns:
        (episodes, position, metadata) where episodes is a list of
        (flat_obs, actions, rewards, terminals) numpy tuples, position is the
        ring buffer write position, and metadata is a list of per-episode
        quality dicts (Plan v8) or None when the file predates that format.
    """
    data = np.load(path)

    n_episodes = int(data["n_episodes"])
    lengths = data["lengths"]
    position = int(data["position"])

    all_obs = data["flat_obs"]
    all_actions = data["actions"]
    all_rewards = data["rewards"]
    all_terminals = data["terminals"]

    # Split observations: each episode has (length + 1) obs entries
    obs_lengths = lengths + 1
    obs_splits = np.cumsum(obs_lengths)[:-1]
    obs_list = np.split(all_obs, obs_splits, axis=0)

    # Split actions/rewards/terminals: each episode has (length) entries
    step_splits = np.cumsum(lengths)[:-1]
    actions_list = np.split(all_actions, step_splits, axis=0)
    rewards_list = np.split(all_rewards, step_splits, axis=0)
    terminals_list = np.split(all_terminals, step_splits, axis=0)

    episodes = [
        (obs_list[i], actions_list[i], rewards_list[i], terminals_list[i])
        for i in range(n_episodes)
    ]

    # Plan v8 + Plan v12 — per-episode quality flags, when the file
    # carries them. Backward-compat: legacy caches predating Plan v12
    # have only Plan v8 fields; the Plan v12 fields (is_clean,
    # target_branch_idx) default to safe values (False / -1) when absent.
    metadata = None
    if "meta_return" in data.files:
        mr = data["meta_return"]
        mc = data["meta_reached"]
        md = data["meta_is_demo"]
        # Plan v12 — optional fields. Present in multi-target heatup
        # caches; absent in Plan v8 caches.
        mic = data["meta_is_clean"] if "meta_is_clean" in data.files else None
        mti = data["meta_target_branch_idx"] if "meta_target_branch_idx" in data.files else None
        # Plan v12 — traceability fields (which daughter the wire actually
        # ended in; tracking-target; +1.0 bonus signal). All four are
        # backward-compatibly defaulted to safe values when absent.
        mtbs = data["meta_target_branch_short"] if "meta_target_branch_short" in data.files else None
        mfbs = data["meta_final_branch_short"] if "meta_final_branch_short" in data.files else None
        mrcd = data["meta_received_correct_daughter"] if "meta_received_correct_daughter" in data.files else None
        mrwd = data["meta_received_wrong_daughter"] if "meta_received_wrong_daughter" in data.files else None
        mgs = data["meta_grader_success"] if "meta_grader_success" in data.files else None
        mgft = data["meta_grader_failure_timeout"] if "meta_grader_failure_timeout" in data.files else None
        movs = data["meta_overshoot"] if "meta_overshoot" in data.files else None
        metadata = [
            {
                "episode_return": float(mr[i]),
                "reached_target_daughter": bool(mc[i]),
                "is_demo": bool(md[i]),
                "is_clean": bool(mic[i]) if mic is not None else False,
                "target_branch_idx": int(mti[i]) if mti is not None else -1,
                "target_branch_short": str(mtbs[i]) if mtbs is not None else "unknown",
                "final_branch_short": str(mfbs[i]) if mfbs is not None else "unknown",
                "received_correct_daughter": bool(mrcd[i]) if mrcd is not None else False,
                "received_wrong_daughter": bool(mrwd[i]) if mrwd is not None else False,
                "grader_success": bool(mgs[i]) if mgs is not None else False,
                "grader_failure_timeout": bool(mgft[i]) if mgft is not None else False,
                "overshoot": bool(movs[i]) if movs is not None else False,
            }
            for i in range(n_episodes)
        ]

    total_steps = int(lengths.sum())
    logger.info(f"Loaded {n_episodes} episodes ({total_steps} steps) from {path}")
    return episodes, position, metadata
