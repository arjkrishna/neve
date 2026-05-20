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
        metadata: Plan v8 — optional list (one dict per episode) of
            episode-quality flags `{"episode_return", "reached_target_daughter",
            "is_demo"}`, persisted so cache-loaded episodes carry the signals
            the stabilization-suite samplers need. None → flags omitted.
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
    np.savez_compressed(path, **save_kwargs)
    total_steps = int(lengths.sum())
    logger.info(f"Saved {len(episodes)} episodes ({total_steps} steps) to {path}")


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

    # Plan v8 — per-episode quality flags, when the file carries them.
    metadata = None
    if "meta_return" in data.files:
        mr = data["meta_return"]
        mc = data["meta_reached"]
        md = data["meta_is_demo"]
        metadata = [
            {
                "episode_return": float(mr[i]),
                "reached_target_daughter": bool(mc[i]),
                "is_demo": bool(md[i]),
            }
            for i in range(n_episodes)
        ]

    total_steps = int(lengths.sum())
    logger.info(f"Loaded {n_episodes} episodes ({total_steps} steps) from {path}")
    return episodes, position, metadata
