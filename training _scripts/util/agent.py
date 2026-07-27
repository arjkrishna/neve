from torch import optim
import eve_rl
import eve
import numpy as np


class BenchAgentSingle(eve_rl.agent.Single):
    def __init__(
        self,
        device,
        lr,
        lr_end_factor,
        lr_linear_end_steps,
        hidden_layers,
        embedder_nodes,
        embedder_layers,
        gamma,
        batch_size,
        reward_scaling,
        replay_buffer_size,
        env_train: eve.Env,
        env_eval: eve.Env,
        consecutive_action_steps,
        stochastic_eval: bool = False,
        ff_only: bool = False,
        # Log-std band must reach EVERY GaussianPolicy site, not only the
        # Synchron one — same defaults as BenchAgentSynchron.
        log_std_min: float = -2.0,
        log_std_max: float = 2.0,
        # Auto-alpha entropy setpoint override; None -> SAC default
        # (-n_actions).
        target_entropy: float = None,
        # Gen-4 asymmetric critic — see BenchAgentSynchron. Policy width =
        # flat obs width minus the privileged tail; own embedder instance.
        privileged_obs_dim: int = 0,
    ):

        obs_dict = env_train.observation_space.sample()
        obs_list = [obs.flatten() for obs in obs_dict.values()]
        obs_np = np.concatenate(obs_list)

        n_observations = obs_np.shape[0]
        n_obs_policy = n_observations - int(privileged_obs_dim)
        n_actions = env_train.action_space.sample().flatten().shape[0]
        if embedder_layers and embedder_nodes and not ff_only:
            q1_embedder = eve_rl.network.component.LSTM(
                n_layer=embedder_layers, n_nodes=embedder_nodes
            )
            policy_embedder = eve_rl.network.component.LSTM(
                n_layer=embedder_layers, n_nodes=embedder_nodes
            )
        elif embedder_layers and embedder_nodes and ff_only:
            hidden_layers = [embedder_nodes] * embedder_layers
            q1_embedder = eve_rl.network.component.MLP(hidden_layers=hidden_layers)
            policy_embedder = eve_rl.network.component.MLP(
                hidden_layers=hidden_layers
            )
        else:
            q1_embedder = eve_rl.network.component.ComponentDummy()
            policy_embedder = eve_rl.network.component.ComponentDummy()

        q1_base = eve_rl.network.component.MLP(hidden_layers)
        q2_base = eve_rl.network.component.MLP(hidden_layers)
        policy_base = eve_rl.network.component.MLP(hidden_layers)

        q1 = eve_rl.network.QNetwork(q1_base, n_observations, n_actions, q1_embedder)
        q1_optim = eve_rl.optim.Adam(
            q1,
            lr=lr,
        )
        q1_scheduler = optim.lr_scheduler.LinearLR(
            q1_optim,
            start_factor=1.0,
            end_factor=lr_end_factor,
            total_iters=lr_linear_end_steps,
        )

        q2 = eve_rl.network.QNetwork(q2_base, n_observations, n_actions, q1_embedder)
        q2_optim = eve_rl.optim.Adam(
            q2_base,
            lr=lr,
        )
        q2_scheduler = optim.lr_scheduler.LinearLR(
            q2_optim,
            start_factor=1.0,
            end_factor=lr_end_factor,
            total_iters=lr_linear_end_steps,
        )

        policy = eve_rl.network.GaussianPolicy(
            policy_base, n_obs_policy, n_actions, policy_embedder,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
        policy_optim = eve_rl.optim.Adam(
            policy_base,
            lr=lr,
        )
        policy_scheduler = optim.lr_scheduler.LinearLR(
            policy_optim,
            start_factor=1.0,
            end_factor=lr_end_factor,
            total_iters=lr_linear_end_steps,
        )

        sac_model = eve_rl.model.SACModel(
            lr_alpha=lr,
            q1=q1,
            q2=q2,
            policy=policy,
            q1_optimizer=q1_optim,
            q2_optimizer=q2_optim,
            policy_optimizer=policy_optim,
            q1_scheduler=q1_scheduler,
            q2_scheduler=q2_scheduler,
            policy_scheduler=policy_scheduler,
        )

        algo = eve_rl.algo.SAC(
            sac_model,
            n_actions=n_actions,
            gamma=gamma,
            reward_scaling=reward_scaling,
            stochastic_eval=stochastic_eval,
            target_entropy=target_entropy,
        )

        replay_buffer = eve_rl.replaybuffer.VanillaEpisodeShared(
            replay_buffer_size, batch_size, device
        )

        super().__init__(
            algo,
            env_train,
            env_eval,
            replay_buffer,
            consecutive_action_steps=consecutive_action_steps,
            device=device,
            normalize_actions=True,
        )


class BenchAgentSynchron(eve_rl.agent.Synchron):
    def __init__(
        self,
        trainer_device,
        worker_device,
        lr,
        lr_end_factor,
        lr_linear_end_steps,
        hidden_layers,
        embedder_nodes,
        embedder_layers,
        gamma,
        batch_size,
        reward_scaling,
        replay_buffer_size,
        env_train: eve.Env,
        env_eval: eve.Env,
        consecutive_action_steps,
        n_worker,
        stochastic_eval: bool = False,
        ff_only: bool = False,
        diagnostics_config: dict = None,
        replay_mode: str = "episode",
        per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_steps: float = 2e7,
        grad_clip: float = 0.0,
        algo: str = "sac",
        awac_lambda: float = 3.0,
        demo_priority_bonus: float = 0.0,
        priority_mode: str = "td",
        balanced_fraction: float = 0.0,
        log_std_min: float = -2.0,  # Plan v11 anti-rail default (was -20)
        log_std_max: float = 2.0,  # Plan v12 anti-rail — cap the std CEILING
        entropy_beta_per_dim=None,
        action_mean_penalty: float = 0.0,
        # RL_IMPROV_15 collapse forensics — configurable log_alpha clamp
        # rails (SAC/AWAC auto-alpha). Defaults = legacy (-10, 2).
        log_alpha_min: float = -10.0,
        log_alpha_max: float = 2.0,
        # RL_IMPROV_16 E1b/E2.2 — AWAC advantage normalization + aux-label
        # z-scoring (see sac.py docnotes). Defaults OFF = legacy.
        awac_adv_norm_tau: float = 0.0,
        aux_label_znorm: bool = False,
        # RL_IMPROV_17 (RLPD recipe) — see sac.py / mlp.py / pervanillastep
        # docnotes. All default-off = legacy behavior.
        critic_layernorm: bool = False,
        backup_entropy: bool = True,
        offline_fraction: float = 0.0,
        # RL_IMPROV_18 v1c — crunchpass lane (see pervanillastep docnote).
        crunchpass_fraction: float = 0.0,
        crunchpass_la0_index: int = -1,
        crunchpass_la2_index: int = -1,
        crunchpass_radius_index: int = -1,
        crunchpass_engage_thresh: float = 0.25,
        crunchpass_radius_thresh: float = 0.175,
        # RL_IMPROV_16 E3 — stuck-lane sampling knobs (see pervanillastep
        # docnote; indices are flat-obs positions computed by the training
        # script from the env5 Gen-4 layout). Defaults OFF = legacy.
        stuck_fraction: float = 0.0,
        stuck_slack_index: int = -1,
        stuck_slack_thresh: float = 0.174,
        stuck_contact_index: int = -1,
        stuck_contact_thresh: float = 0.0026,
        # Auto-alpha entropy setpoint override; None -> SAC default
        # (-n_actions). SAC/AWAC only — IQL has no entropy regulator.
        target_entropy: float = None,
        offline_mode: bool = False,
        # Plan v11 Stage 1B — IQL-specific hyperparameters (only used when
        # algo == "iql"). When algo != "iql" these are ignored.
        iql_tau: float = 0.7,
        iql_beta: float = 3.0,
        iql_awr_max: float = 5.0,
        lr_value: float = None,
        # FIX 1 (RL_IMPROV_8 KL-anchor iteration) — KL-to-warmstart penalty
        # for the IQL policy update. Only used when algo == "iql".
        #   alpha_kl == 0.0 -> no penalty (default; back-compat).
        #   alpha_kl > 0    -> requires kl_warmstart_state_dict (the frozen
        #                      reference policy weights).
        alpha_kl: float = 0.0,
        kl_warmstart_state_dict: dict = None,
        # Gen-4 asymmetric critic — width of the privileged tail appended
        # LAST in the env's ObsDict. Critics (and IQL V) consume the FULL
        # flat obs; the policy is built n_obs_total - privileged_obs_dim
        # wide and slices its input internally, so a deployed policy never
        # depends on sim-only state. 0 = symmetric (legacy).
        privileged_obs_dim: int = 0,
        # Gen-4 aux heads — indices RELATIVE to the privileged tail start;
        # converted to absolute flat-obs indices here. The policy predicts
        # these privileged values from its deployable prefix (MSE weight
        # aux_coef). Empty/0.0 = off.
        aux_coef: float = 0.0,
        aux_label_rel_indices=None,
        # Gen-4 procedural anatomy — per-worker training-env factory. When
        # set, worker i gets env_train_factory(i) (e.g. a distinctly-seeded
        # DualDeviceNavProcedural) instead of a deepcopy of the master
        # env_train. env_eval stays the shared held-out anatomy.
        env_train_factory=None,
        # RL_IMPROV_18 — per-worker EVAL-env factory (see synchron.py note):
        # lets each worker own its own anatomy stream so eval measures
        # generalization across vessel trees, not one frozen tree.
        env_eval_factory=None,
    ):

        obs_dict = env_train.observation_space.sample()
        obs_list = [obs.flatten() for obs in obs_dict.values()]
        obs_np = np.concatenate(obs_list)

        n_observations = obs_np.shape[0]
        n_obs_policy = n_observations - int(privileged_obs_dim)
        if n_obs_policy <= 0:
            raise ValueError(
                f"privileged_obs_dim={privileged_obs_dim} >= flat obs "
                f"width {n_observations}"
            )
        aux_label_rel_indices = list(aux_label_rel_indices or [])
        # Validate at CONSTRUCTION (not at the first update, hours in): an
        # out-of-range index would crash index_select mid-training; a
        # NEGATIVE index would silently supervise against a deployable-obs
        # column (invisible garbage). Bound to the privileged tail.
        for _i in aux_label_rel_indices:
            if not (0 <= int(_i) < int(privileged_obs_dim)):
                raise ValueError(
                    f"aux label rel index {_i} outside "
                    f"[0, privileged_obs_dim={privileged_obs_dim})"
                )
        aux_label_abs = [n_obs_policy + int(i) for i in aux_label_rel_indices]
        n_actions = env_train.action_space.sample().flatten().shape[0]
        if embedder_layers and embedder_nodes and not ff_only:
            q1_embedder = eve_rl.network.component.LSTM(
                n_layer=embedder_layers, n_nodes=embedder_nodes
            )
            policy_embedder = eve_rl.network.component.LSTM(
                n_layer=embedder_layers, n_nodes=embedder_nodes
            )
        elif embedder_layers and embedder_nodes and ff_only:
            hidden_layers = [embedder_nodes] * embedder_layers
            q1_embedder = eve_rl.network.component.MLP(hidden_layers=hidden_layers)
            policy_embedder = eve_rl.network.component.MLP(
                hidden_layers=hidden_layers
            )
        else:
            q1_embedder = eve_rl.network.component.ComponentDummy()
            policy_embedder = eve_rl.network.component.ComponentDummy()
        # The policy gets its OWN embedder instance: with the asymmetric
        # widths a shared head would raise on the n_inputs re-set (and the
        # old sharing meant only q1's optimizer ever stepped it while all
        # three losses fed it gradients).

        # RL_IMPROV_17 (RLPD) — LayerNorm on the CRITIC bodies only (the
        # RLPD-decisive stabilizer); the policy body stays legacy.
        q1_base = eve_rl.network.component.MLP(
            hidden_layers, use_layernorm=critic_layernorm
        )
        q2_base = eve_rl.network.component.MLP(
            hidden_layers, use_layernorm=critic_layernorm
        )
        policy_base = eve_rl.network.component.MLP(hidden_layers)

        q1 = eve_rl.network.QNetwork(q1_base, n_observations, n_actions, q1_embedder)
        q1_optim = eve_rl.optim.Adam(
            q1,
            lr=lr,
        )
        q1_scheduler = optim.lr_scheduler.LinearLR(
            q1_optim,
            start_factor=1.0,
            end_factor=lr_end_factor,
            total_iters=lr_linear_end_steps,
        )

        q2 = eve_rl.network.QNetwork(q2_base, n_observations, n_actions, q1_embedder)
        q2_optim = eve_rl.optim.Adam(
            q2_base,
            lr=lr,
        )
        q2_scheduler = optim.lr_scheduler.LinearLR(
            q2_optim,
            start_factor=1.0,
            end_factor=lr_end_factor,
            total_iters=lr_linear_end_steps,
        )

        policy = eve_rl.network.GaussianPolicy(
            policy_base, n_obs_policy, n_actions, policy_embedder,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            n_aux=len(aux_label_abs),
        )
        policy_optim = eve_rl.optim.Adam(
            policy_base,
            lr=lr,
        )
        policy_scheduler = optim.lr_scheduler.LinearLR(
            policy_optim,
            start_factor=1.0,
            end_factor=lr_end_factor,
            total_iters=lr_linear_end_steps,
        )

        # Plan v11 Stage 1B — choose algo / model construction by name.
        # The SAC path covers "sac" + "awac" (AWAC is a flag inside the
        # SAC algo). The IQL path constructs an IQLModel with an extra V
        # network + its own algo class.
        if algo == "iql":
            v_base = eve_rl.network.component.MLP(hidden_layers)
            v = eve_rl.network.VNetwork(v_base, n_observations, q1_embedder)
            # Default V-network LR follows the critic LR if not provided —
            # keeps the trainer CLI backward-compatible.
            _lr_v = lr if lr_value is None else lr_value
            v_optim = eve_rl.optim.Adam(v_base, lr=_lr_v)
            v_scheduler = optim.lr_scheduler.LinearLR(
                v_optim,
                start_factor=1.0,
                end_factor=lr_end_factor,
                total_iters=lr_linear_end_steps,
            )
            iql_model = eve_rl.model.IQLModel(
                q1=q1,
                q2=q2,
                v=v,
                policy=policy,
                q1_optimizer=q1_optim,
                q2_optimizer=q2_optim,
                v_optimizer=v_optim,
                policy_optimizer=policy_optim,
                q1_scheduler=q1_scheduler,
                q2_scheduler=q2_scheduler,
                v_scheduler=v_scheduler,
                policy_scheduler=policy_scheduler,
            )
            algo = eve_rl.algo.IQL(
                iql_model,
                n_actions=n_actions,
                gamma=gamma,
                reward_scaling=reward_scaling,
                stochastic_eval=stochastic_eval,
                grad_clip=grad_clip,
                iql_expectile_tau=iql_tau,
                iql_beta=iql_beta,
                iql_awr_max=iql_awr_max,
                offline_mode=offline_mode,
                # FIX 1 — KL-to-warmstart anchor (no-op when alpha_kl == 0).
                alpha_kl=alpha_kl,
                kl_warmstart_state_dict=kl_warmstart_state_dict,
            )
        else:
            sac_model = eve_rl.model.SACModel(
                lr_alpha=lr,
                q1=q1,
                q2=q2,
                policy=policy,
                q1_optimizer=q1_optim,
                q2_optimizer=q2_optim,
                policy_optimizer=policy_optim,
                q1_scheduler=q1_scheduler,
                q2_scheduler=q2_scheduler,
                policy_scheduler=policy_scheduler,
            )

            algo = eve_rl.algo.SAC(
                sac_model,
                n_actions=n_actions,
                gamma=gamma,
                reward_scaling=reward_scaling,
                stochastic_eval=stochastic_eval,
                grad_clip=grad_clip,
                algo=algo,
                awac_lambda=awac_lambda,
                awac_adv_norm_tau=awac_adv_norm_tau,
                aux_label_znorm=aux_label_znorm,
                backup_entropy=backup_entropy,
                entropy_beta_per_dim=entropy_beta_per_dim,
                action_mean_penalty=action_mean_penalty,
                log_alpha_min=log_alpha_min,
                log_alpha_max=log_alpha_max,
                target_entropy=target_entropy,
                offline_mode=offline_mode,
                aux_coef=aux_coef,
                aux_label_indices=aux_label_abs or None,
            )

        # Plan v6 — replay-mode selects the buffer class.
        #   "episode" — VanillaEpisodeShared: stores whole episodes, sample()
        #     returns padded sequences (the buffer for the LSTM-embedder
        #     setup; capacity counted in episodes).
        #   "step"    — VanillaStepShared: stores individual (s,a,r,s',done)
        #     transitions, sample() returns random transition batches
        #     (canonical step-SAC; capacity counted in transitions).
        # Plan v7 — `per` is an orthogonal switch on step mode: when set,
        #   the uniform VanillaStepShared is swapped for PERVanillaStepShared
        #   (proportional sampling + IS weights). PER is step-only.
        if replay_mode == "step" and per:
            replay_buffer = eve_rl.replaybuffer.PERVanillaStepShared(
                replay_buffer_size,
                batch_size,
                trainer_device,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_steps=per_beta_steps,
                demo_priority_bonus=demo_priority_bonus,
                priority_mode=priority_mode,
                balanced_fraction=balanced_fraction,
                stuck_fraction=stuck_fraction,
                stuck_slack_index=stuck_slack_index,
                stuck_slack_thresh=stuck_slack_thresh,
                stuck_contact_index=stuck_contact_index,
                stuck_contact_thresh=stuck_contact_thresh,
                offline_fraction=offline_fraction,
                crunchpass_fraction=crunchpass_fraction,
                crunchpass_la0_index=crunchpass_la0_index,
                crunchpass_la2_index=crunchpass_la2_index,
                crunchpass_radius_index=crunchpass_radius_index,
                crunchpass_engage_thresh=crunchpass_engage_thresh,
                crunchpass_radius_thresh=crunchpass_radius_thresh,
            )
        elif replay_mode == "step":
            replay_buffer = eve_rl.replaybuffer.VanillaStepShared(
                replay_buffer_size, batch_size, trainer_device,
                balanced_fraction=balanced_fraction,
            )
        else:
            replay_buffer = eve_rl.replaybuffer.VanillaEpisodeShared(
                replay_buffer_size, batch_size, trainer_device
            )

        super().__init__(
            algo,
            env_train,
            env_eval,
            replay_buffer,
            consecutive_action_steps=consecutive_action_steps,
            trainer_device=trainer_device,
            worker_device=worker_device,
            n_worker=n_worker,
            normalize_actions=True,
            timeout_worker_after_reaching_limit=180,
            diagnostics_config=diagnostics_config,
            env_train_factory=env_train_factory,
            env_eval_factory=env_eval_factory,
        )


def create_bench_agent(
    device_trainer,
    device_worker,
    lr,
    lr_end_factor,
    lr_linear_end_steps,
    hidden_layers,
    embedder_nodes,
    embedder_layers,
    gamma,
    batch_size,
    reward_scaling,
    replay_buffer_size,
    train_env: eve.Env,
    eval_env: eve.Env,
    consecutive_action_steps,
    n_worker,
    stochastic_eval: bool = False,
    single: bool = False,
    ff_only: bool = False,
    # Log-std band must reach EVERY GaussianPolicy site, not only the
    # Synchron one — same defaults as BenchAgentSynchron.
    log_std_min: float = -2.0,
    log_std_max: float = 2.0,
    # Auto-alpha entropy setpoint override; None -> SAC default
    # (-n_actions).
    target_entropy: float = None,
    # Gen-4 asymmetric critic — see BenchAgentSynchron.
    privileged_obs_dim: int = 0,
):
    obs_dict = train_env.observation_space.sample()
    obs_list = [obs.flatten() for obs in obs_dict.values()]
    obs_np = np.concatenate(obs_list)

    n_observations = obs_np.shape[0]
    n_obs_policy = n_observations - int(privileged_obs_dim)
    n_actions = train_env.action_space.sample().flatten().shape[0]
    if embedder_layers and embedder_nodes and not ff_only:
        q1_embedder = eve_rl.network.component.LSTM(
            n_layer=embedder_layers, n_nodes=embedder_nodes
        )
        policy_embedder = eve_rl.network.component.LSTM(
            n_layer=embedder_layers, n_nodes=embedder_nodes
        )
    elif embedder_layers and embedder_nodes and ff_only:
        hidden_layers = [embedder_nodes] * embedder_layers
        q1_embedder = eve_rl.network.component.MLP(hidden_layers=hidden_layers)
        policy_embedder = eve_rl.network.component.MLP(
            hidden_layers=hidden_layers
        )
    else:
        q1_embedder = eve_rl.network.component.ComponentDummy()
        policy_embedder = eve_rl.network.component.ComponentDummy()

    q1_base = eve_rl.network.component.MLP(hidden_layers)
    q2_base = eve_rl.network.component.MLP(hidden_layers)
    policy_base = eve_rl.network.component.MLP(hidden_layers)

    q1 = eve_rl.network.QNetwork(q1_base, n_observations, n_actions, q1_embedder)
    q1_optim = eve_rl.optim.Adam(
        q1,
        lr=lr,
    )
    q1_scheduler = optim.lr_scheduler.LinearLR(
        q1_optim,
        start_factor=1.0,
        end_factor=lr_end_factor,
        total_iters=lr_linear_end_steps,
    )

    q2 = eve_rl.network.QNetwork(q2_base, n_observations, n_actions, q1_embedder)
    q2_optim = eve_rl.optim.Adam(
        q2_base,
        lr=lr,
    )
    q2_scheduler = optim.lr_scheduler.LinearLR(
        q2_optim,
        start_factor=1.0,
        end_factor=lr_end_factor,
        total_iters=lr_linear_end_steps,
    )

    policy = eve_rl.network.GaussianPolicy(
        policy_base, n_obs_policy, n_actions, policy_embedder,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
    )
    policy_optim = eve_rl.optim.Adam(
        policy_base,
        lr=lr,
    )
    policy_scheduler = optim.lr_scheduler.LinearLR(
        policy_optim,
        start_factor=1.0,
        end_factor=lr_end_factor,
        total_iters=lr_linear_end_steps,
    )

    sac_model = eve_rl.model.SACModel(
        lr_alpha=lr,
        q1=q1,
        q2=q2,
        policy=policy,
        q1_optimizer=q1_optim,
        q2_optimizer=q2_optim,
        policy_optimizer=policy_optim,
        q1_scheduler=q1_scheduler,
        q2_scheduler=q2_scheduler,
        policy_scheduler=policy_scheduler,
    )

    algo = eve_rl.algo.SAC(
        sac_model,
        n_actions=n_actions,
        gamma=gamma,
        reward_scaling=reward_scaling,
        stochastic_eval=stochastic_eval,
        target_entropy=target_entropy,
    )

    replay_buffer = eve_rl.replaybuffer.VanillaEpisodeShared(
        replay_buffer_size, batch_size, device_trainer
    )
    if not single:
        agent = eve_rl.agent.Synchron(
            algo,
            train_env,
            eval_env,
            replay_buffer,
            consecutive_action_steps=consecutive_action_steps,
            trainer_device=device_trainer,
            worker_device=device_worker,
            n_worker=n_worker,
            normalize_actions=True,
            timeout_worker_after_reaching_limit=180,
        )
    else:
        agent = eve_rl.agent.Single(
            algo,
            train_env,
            eval_env,
            replay_buffer,
            consecutive_action_steps=consecutive_action_steps,
            device=device_trainer,
            normalize_actions=True,
        )

    return agent


class OfflineDummyEnv:
    """Plan v11 Stage 1 — shape-only env stub for offline RL.

    `BenchAgentSingle` / `BenchAgentSynchron` infer `n_observations` and
    `n_actions` from the env's `observation_space.sample()` (a Dict of
    Boxes) and `action_space.sample()` (a flat Box). Stage 1 trains on
    saved transitions only — no SOFA, no env — so we hand the agent
    constructor an env whose ONLY job is to advertise the right shapes.

    The Dict-with-one-key layout matches how the real env's flatten path
    works (concat all Box values), so the SAC nets end up with the same
    input dim as the buffer's stored flat_obs vectors.
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 action_low: float = -1.0, action_high: float = 1.0):
        import gymnasium as gym
        self.observation_space = gym.spaces.Dict({
            "flat": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
            ),
        })
        self.action_space = gym.spaces.Box(
            low=action_low, high=action_high,
            shape=(action_dim,), dtype=np.float32,
        )

    def reset(self, *args, **kwargs):
        return {"flat": np.zeros(self.observation_space["flat"].shape, dtype=np.float32)}, {}

    def step(self, action):
        zero = {"flat": np.zeros(self.observation_space["flat"].shape, dtype=np.float32)}
        return zero, 0.0, False, False, {}

    def close(self):
        return None
