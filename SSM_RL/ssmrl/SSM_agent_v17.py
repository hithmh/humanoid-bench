"""
SSM Agent v14 – JAX + qpax GPU-accelerated MPC.

Architecture is identical to v11 except for the inference-time controller:
  * MPC cost is evaluated entirely in JAX (JIT-compiled, runs on GPU/CPU)
  * The QP is solved by ``qpax`` (pure-JAX interior-point QP solver)
  * Dynamics constraints are analytically eliminated: the full state trajectory
    is expressed as a linear function of the stacked control sequence U,
    reducing the MPC to a single dense QP in U only.
  * The ensemble terminal cost ``max_i V_i(z)`` is replaced by a latent-space
    UCB: ``mean_i V_i(z) + beta * std_i V_i(z)``, encouraging optimistic
    exploration into uncertain regions of state space.  ``beta`` is controlled
    by ``cfg.ucb_beta`` (default 1.0).
  * Zero-copy PyTorch ↔ JAX tensor bridge via ``torch.utils.dlpack`` /
    ``jax.dlpack`` when both tensors live on the same CUDA device.
  * ``jax.jit`` compiles the full matrix-build + solve once; subsequent calls
    are fast.

QP form passed to qpax:
  min   ½ U^T Q_qp U + c_qp^T U
  s.t.  G U ≤ h   (per-step box constraints on actions)

State trajectory (A is diagonal):
  z_{t+1} = A^{t+1} ⊙ z_0  +  T_u[t] @ U_flat

Training loop (``update``) mirrors ``SSMRL.update`` from ``ssmrl.py``:
  sample buffer → encode → latent rollout → consistency / reward / value losses
  → update world model → update policy → soft-update targets.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

# ---- JAX stack (required) -----------------------------------------------
import jax
import jax.numpy as jnp
import qpax  # pip install qpax


from ssmrl.common.ssm_world_model_v5 import SSMWorldModel
from ssmrl.common.scale import RunningScale


class SSMAgent:
    """
    SSM-RL agent.  Implements training + inference.
    Mirrors the public API of ``SSMRL`` from ``ssmrl.py``.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # World model
        self.model = SSMWorldModel(cfg).to(self.device)

        # Dimensions (convenience)
        self.latent_dim = self.model.latent_dim
        self.act_dim = self.model.act_dim
        self.state_dim = self.model.state_dim
        self.history_horizon = self.model.history_horizon
        self.num_ensembles = self.model.num_ensembles

        # ---- Optimizers (following ssmrl.py pattern: model optim + pi optim) ----
        enc_lr_scale = getattr(cfg, 'enc_lr_scale', 0.3)
        lr = cfg.lr

        # Group 1 – world-model parameters (encoder_mean at scaled lr)
        self.model_optim = torch.optim.Adam([
            {'params': self.model._encoder_mean.parameters(),
             'lr': lr * enc_lr_scale},
            {'params': self.model._transformer.parameters()},
            {'params': self.model._A_net.parameters()},
            {'params': self.model._B_net.parameters()},
            {'params': self.model._Q_net.parameters()},
            {'params': self.model._q_net.parameters()},
            {'params': [self.model._b]},
            {'params': self.model._P_indices_net.parameters()},
            {'params': self.model._p_net.parameters()},
            {'params': self.model._pb_net.parameters()},
        ], lr=lr)


        # Group 2 – policy (all three heads)
        self.pi_optim = torch.optim.Adam(
            list(self.model._pi_trunk.parameters())
            + list(self.model._pi_mean_head.parameters())
            + list(self.model._pi_log_std_head.parameters()),
            lr=lr, eps=1e-5
        )

        self.model.eval()
        self.scale = RunningScale(cfg)

        # Discount
        self.discount = self._get_discount(
            getattr(cfg, 'episode_length', 1000)
        )

        # Gradient clipping
        self.grad_clip_norm = getattr(cfg, 'grad_clip_norm', 20.0)

        # Loss weights (mirror ssmrl.py config keys where available)
        self.consistency_coef = getattr(cfg, 'consistency_coef', 20.0)
        self.reward_coef = getattr(cfg, 'reward_coef', 0.1)
        self.value_coef = getattr(cfg, 'value_coef', 0.1)
        self.entropy_coef = getattr(cfg, 'entropy_coef', 1e-4)
        self.rho = getattr(cfg, 'rho', 0.5)
        self.horizon = getattr(cfg, 'horizon', 3)

        # Shift / scale for observation and action normalisation
        self.shift = np.zeros(self.state_dim)
        self.scale_obs = np.ones(self.state_dim)
        self.shift_u = np.zeros(self.act_dim)
        self.scale_u = np.ones(self.act_dim)

        # History buffers for inference (transformer context)
        self.state_history = []
        self.action_history = []

        # Build JAX + qpax MPC controller
        self._build_jax_controller()

    # ------------------------------------------------------------------
    # Discount helper (same as SSMRL)
    # ------------------------------------------------------------------
    def _get_discount(self, episode_length):
        denom = getattr(self.cfg, 'discount_denom', 5)
        d_min = getattr(self.cfg, 'discount_min', 0.95)
        d_max = getattr(self.cfg, 'discount_max', 0.995)
        frac = episode_length / denom
        return min(max((frac - 1) / frac, d_min), d_max)

    # ------------------------------------------------------------------
    # Save / Load (mirror ssmrl.py)
    # ------------------------------------------------------------------
    def save(self, fp):
        """Save agent state dict."""
        torch.save({'model': self.model.state_dict()}, fp)

    def load(self, fp):
        """Load agent state dict."""
        state_dict = fp if isinstance(fp, dict) else torch.load(fp, weights_only=False)
        self.model.load_state_dict(state_dict['model'])

    # ------------------------------------------------------------------
    # Shift / scale management
    # ------------------------------------------------------------------
    def set_shift_and_scale(self, shift, scale, shift_u, scale_u):
        self.shift = np.asarray(shift, dtype=np.float32)
        self.scale_obs = np.asarray(scale, dtype=np.float32)
        self.shift_u = np.asarray(shift_u, dtype=np.float32)
        self.scale_u = np.asarray(scale_u, dtype=np.float32)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False):
        """
        Select an action.

        Args:
            obs:  raw observation (numpy or torch), shape [state_dim].
            t0:   True at the first step of an episode.
            eval_mode: deterministic action if True.
        Returns:
            action (numpy), shape [act_dim], in *original* (un-normalised) scale.
        """
        # Normalise observation
        if isinstance(obs, torch.Tensor):
            obs_np = obs.cpu().numpy()
        else:
            obs_np = np.asarray(obs, dtype=np.float32)
        x0 = obs_np
        obs_t = torch.tensor(x0, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Encode
        z = self.model.encode(obs_t)

        # Decide action
        if len(self.state_history) < self.history_horizon:
            # Not enough history for transformer – use policy net
            u_norm = self.model.pi(z, deterministic=eval_mode)[0].cpu().numpy()
        else:
            # Attempt JAX/qpax planning
            u_norm = self._plan_jax(z, x0, eval_mode)
        # Update history
        self.action_history.append(u_norm.copy())
        self.state_history.append(x0.copy())

        ## convert to float32
        u = np.asarray(u_norm, dtype=np.float32)
        return u

    @torch.no_grad()
    def _perturb_encoded_params(self, A_diag, B, Q_diag, q):
        """
        Add relative Gaussian noise to encoded SSM parameters (A_diag, B, Q_diag, q)
        for exploration.  Noise magnitude is proportional to each tensor's mean
        absolute value, so the perturbation is scale-invariant.

        Controlled by ``cfg.param_noise_std`` (default 0.02).
        """
        std = float(getattr(self.cfg, 'param_noise_std', 0.02))

        def _noisy(t: torch.Tensor) -> torch.Tensor:
            scale = t.abs().mean() + 1e-6
            return t + std * scale * torch.randn_like(t)

        # Q_diag must stay non-negative so that Tᵤᵀ diag(Q_diag) Tᵤ remains PSD.
        # P_diags is already guarded by F.relu in _plan_jax; Q_diag is not.
        noisy_Q_diag = _noisy(Q_diag).clamp(min=0.0)

        return _noisy(A_diag), _noisy(B), noisy_Q_diag, _noisy(q)

    def _plan_jax(self, z: torch.Tensor, x0: np.ndarray, eval_mode: bool):
        """
        Solve the JAX/qpax MPC problem.  Falls back to the policy net on
        numerical failure or solver non-convergence.

        Returns:
            u_norm: normalised action, shape [act_dim]
        """
        state_seq = np.array(self.state_history[-self.history_horizon:],
                             dtype=np.float32)
        action_seq = np.array(self.action_history[-self.history_horizon:],
                              dtype=np.float32)
        state_t  = torch.tensor(state_seq,  device=self.device).unsqueeze(0)
        action_t = torch.tensor(action_seq, device=self.device).unsqueeze(0)
        obs_t    = torch.tensor(x0, dtype=torch.float32, device=self.device).unsqueeze(0)

        A_diag, B, Q_diag, q, encoder_in = self.model.encode_context(state_t, action_t, obs_t)

        # ---- Parameter noise for exploration ----
        if not eval_mode:
            A_diag, B, Q_diag, q = self._perturb_encoded_params(A_diag, B, Q_diag, q)

        # Extract single-sample tensors
        A_diag_t = A_diag[0]   # (D,)
        B_t      = B[0]         # (D, nU)
        Q_diag_t = Q_diag[0]   # (D,)
        q_t      = q[0]         # (D,)
        z_t      = z[0]         # (D,)

        # Critic params from MLP heads conditioned on encoder_in
        E, D = self.num_ensembles, self.latent_dim
        P_diags_t = F.relu(self.model._P_indices_net(encoder_in)).view(E, D)   # (E, D)
        p_t       = self.model._p_net(encoder_in).view(E, D)                   # (E, D)
        pb_t      = self.model._pb_net(encoder_in).squeeze(0)                  # (E,)

        # ---- PyTorch → JAX (zero-copy DLPack on CUDA) ----
        def to_jax(t: torch.Tensor):
            return jax.dlpack.from_dlpack(
                torch.utils.dlpack.to_dlpack(t.detach().contiguous()))

        try:
            U_sol, converged = self._jax_solve_mpc(
                to_jax(z_t),
                to_jax(A_diag_t),
                to_jax(B_t),
                to_jax(Q_diag_t),
                to_jax(q_t),
                to_jax(P_diags_t),
                to_jax(p_t),
                to_jax(pb_t),
                self._a_low_jax,
                self._a_high_jax,
            )
            if not bool(converged):
                return self.model.pi(z, deterministic=eval_mode)[0].cpu().numpy()
            u = np.asarray(U_sol[0], dtype=np.float32)   # first control step
        except Exception:
            return self.model.pi(z, deterministic=eval_mode)[0].cpu().numpy()

        if not eval_mode:
            std = self.model.get_pi_std(z)[0]
            epsilon = (std * torch.randn(self.act_dim, device=std.device)
                       ).detach().cpu().numpy()
            u = u + epsilon

        return np.clip(u, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # JAX + qpax MPC controller builder
    # ------------------------------------------------------------------
    def _build_jax_controller(self):
        """
        Build and JIT-compile the JAX/qpax MPC solve function.

        Dynamics constraints are analytically eliminated by expressing the full
        state trajectory as a linear map of the stacked control vector U_flat:

            z_{t+1} = A_diag^{t+1} ⊙ z_0   +   T_u[t] @ U_flat

        Substituting into the MPC cost yields a strict convex QP in U_flat:
            min   ½ U^T Q_qp U + c_qp^T U
            s.t.  G U ≤ h   (box constraints on each u_t)

        The ensemble terminal cost ``max_i V_i`` is approximated by the ensemble
        *average* to keep the terminal term quadratic (strict QP).
        A small ridge (1e-6 I) is added to Q_qp for numerical stability.
        """
        D   = self.latent_dim
        nU  = self.act_dim
        H   = self.horizon
        CH  = H
        E   = self.num_ensembles
        discount = self.discount
        u_penalty = float(getattr(self.cfg, 'u_penalty', 0.0))
        ucb_beta  = float(getattr(self.cfg, 'ucb_beta', 1.0))

        self._control_horizon = CH
        n = CH * nU   # total QP decision-variable size

        def _build_and_solve(z0, A_diag, B, Q_diag, q_vec,
                             P_diags, p_mat, pb_vec, a_low, a_high):
            """
            Args
            ----
            z0      : (D,)      initial latent state
            A_diag  : (D,)      diagonal of dynamics matrix A
            B       : (D, nU)   input matrix
            Q_diag  : (D,)      diagonal of stage state-cost matrix
            q_vec   : (D,)      linear state-cost coefficient
            P_diags : (E, D)    diagonal of terminal state-critic per ensemble
            p_mat   : (E, D)    linear terminal state-critic coefficient
            pb_vec  : (E,)      terminal critic bias (constant, included for completeness)
            a_low   : (nU,)     per-dim action lower bound
            a_high  : (nU,)     per-dim action upper bound

            Returns
            -------
            U         : (CH, nU)  optimal control sequence
            converged : bool
            """
            # ----------------------------------------------------------
            # 1. State trajectory matrices (unrolled, compile-time loops)
            # ----------------------------------------------------------
            # z_{t+1} = f_list[t] + T_u_list[t] @ U_flat
            f_list   = []
            T_u_list = []

            for t in range(H):
                f_list.append((A_diag ** (t + 1)) * z0)

                T_u_t = jnp.zeros((D, n))
                for k in range(CH):
                    s_k = k * nU
                    e_k = (k + 1) * nU
                    if k < CH - 1:
                        if k <= t:
                            coeff = (A_diag ** (t - k))[:, None] * B   # (D, nU)
                            T_u_t = T_u_t.at[:, s_k:e_k].set(coeff)
                    else:
                        # Last ZOH block
                        accum = jnp.zeros((D, nU))
                        for j in range(CH - 1, t + 1):
                            accum = accum + (A_diag ** (t - j))[:, None] * B
                        T_u_t = T_u_t.at[:, s_k:e_k].set(accum)

                T_u_list.append(T_u_t)

            # ----------------------------------------------------------
            # 2. Assemble Q_qp and c_qp
            #    qpax minimises  ½ U^T Q_qp U + c_qp^T U
            #    so  Q_qp = 2 * H_UU,  c_qp = h_U
            # ----------------------------------------------------------
            Q_qp = jnp.zeros((n, n))
            c_qp = jnp.zeros(n)

            for t in range(H):
                k_u = min(t, CH - 1)
                s_u = k_u * nU
                e_u = (k_u + 1) * nU
                Tu  = T_u_list[t]   # (D, n)
                f   = f_list[t]     # (D,)

                # Stage state cost: z^T diag(Q) z + q^T z
                QTu  = Q_diag[:, None] * Tu          # (D, n)
                Q_qp = Q_qp + (discount ** t) * 2.0 * (Tu.T @ QTu)
                c_qp = c_qp + (discount ** t) * (2.0 * (QTu.T @ f) + Tu.T @ q_vec)

                # Stage control cost: u_penalty * ||u||^2  (block k_u)
                Q_qp = Q_qp.at[s_u:e_u, s_u:e_u].add(
                    (discount ** t) * 2.0 * u_penalty * jnp.eye(nU))

            # Terminal cost – ensemble UCB: mean + beta * std  (optimistic planning)
            Tu_H = T_u_list[H - 1]   # (D, n)
            f_H  = f_list[H - 1]     # (D,)

            # Latent-Space UCB: mean + beta * std over ensemble
            # This gives an optimistic upper bound on the terminal value,
            # encouraging the agent to explore uncertain regions of state space.
            P_mean = jnp.mean(P_diags, axis=0)                         # (D,)
            P_std  = jnp.std( P_diags, axis=0)                         # (D,)
            P_ucb  = P_mean + ucb_beta * P_std                         # (D,)

            p_mean = jnp.mean(p_mat, axis=0)                           # (D,)
            p_std  = jnp.std( p_mat, axis=0)                           # (D,)
            p_ucb  = p_mean + ucb_beta * p_std                         # (D,)

            PTu  = P_ucb[:, None] * Tu_H                               # (D, n)
            Q_qp = Q_qp + (discount ** H) * 2.0 * (Tu_H.T @ PTu)
            c_qp = c_qp + (discount ** H) * (2.0 * (PTu.T @ f_H) + Tu_H.T @ p_ucb)

            # Ridge for numerical stability
            Q_qp = Q_qp + 1e-6 * jnp.eye(n)

            # ----------------------------------------------------------
            # 3. Inequality constraints: a_low ≤ u_t ≤ a_high  ∀t
            # ----------------------------------------------------------
            a_high_t = jnp.tile(a_high, CH)           # (n,)
            a_low_t  = jnp.tile(a_low,  CH)           # (n,)
            G = jnp.concatenate([ jnp.eye(n), -jnp.eye(n)], axis=0)   # (2n, n)
            h = jnp.concatenate([a_high_t, -a_low_t], axis=0)          # (2n,)

            # No equality constraints
            A_eq = jnp.zeros((0, n))
            b_eq = jnp.zeros((0,))

            # ----------------------------------------------------------
            # 4. Solve with qpax
            # ----------------------------------------------------------
            x, _s, _z, _y, converged, _iters = qpax.solve_qp(
                Q_qp, c_qp, A_eq, b_eq, G, h)
            return x.reshape(CH, nU), converged

        self._jax_solve_mpc = jax.jit(_build_and_solve)

        # Pre-fetch action bound arrays (static across calls)
        a_high = np.asarray(
            getattr(self.cfg, 'a_bound_high', np.ones(nU)), dtype=np.float32)
        a_low = np.asarray(
            getattr(self.cfg, 'a_bound_low', -np.ones(nU)), dtype=np.float32)
        self._a_high_jax = jnp.array(a_high)
        self._a_low_jax  = jnp.array(a_low)

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------
    def reset_for_control(self):
        self.state_history = []
        self.action_history = []

    def store_cached_control_info(self):
        self._cached = copy.deepcopy({
            'state_history': self.state_history,
            'action_history': self.action_history,
        })

    def restore_control_info(self):
        if hasattr(self, '_cached'):
            self.state_history = self._cached['state_history']
            self.action_history = self._cached['action_history']
        else:
            print('No cached control info found.')

    # ------------------------------------------------------------------
    # Policy update (mirror SSMRL.update_pi)
    # ------------------------------------------------------------------
    def update_pi(self, zs, A_diag, B_mat, encoder_in):
        """
        Update policy using a sequence of latent states.

        The gradient path is:
            zs (detached) → _pi(zs) → actions → model.next(zs, actions, A, B)
            → z_next → Q_value(z_next, encoder_in)
        so that the policy parameters receive gradients.

        Args:
            zs:         [T, batch, latent_dim]  (detached)
            A_diag:     [batch, latent_dim]     (detached)
            B_mat:      [batch, latent_dim, act_dim] (detached)
            encoder_in: [batch, ctx_dim]        (detached)
        Returns:
            pi_loss (float)
        """
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_critic_grad(False)

        T, B, _ = zs.shape

        # Policy produces actions + log-probs (SAC stochastic); gradient flows through
        # reparameterised samples → next(z, action) → Q_value.
        actions, log_probs = self.model.pi(zs, return_log_prob=True)  # [T, B, act_dim], [T, B, 1]
        z_next_list = []
        for t in range(T):
            z_next_t = self.model.next(zs[t], actions[t], A_diag, B_mat)
            z_next_list.append(z_next_t)
        z_next = torch.stack(z_next_list, dim=0)  # [T, B, D]

        z_next_flat = z_next.reshape(T * B, -1)
        # Expand encoder_in from [B, ctx_dim] to [T*B, ctx_dim]
        enc_in_flat = encoder_in.unsqueeze(0).expand(T, -1, -1).reshape(T * B, -1)
        vals = self.model.Q_value(z_next_flat, enc_in_flat, target=False, return_type='min')
        vals = vals.view(T, B, 1)

        self.scale.update(vals[0])
        vals = self.scale(vals)

        # Weighted loss over horizon
        rho = torch.pow(
            torch.tensor(self.rho, device=self.device),
            torch.arange(T, device=self.device, dtype=torch.float32),
        )
        # SAC loss: maximise (Q - alpha * log_pi)
        pi_loss = -(
            (vals - self.entropy_coef * log_probs).mean(dim=(1, 2)) * rho
        ).mean()

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model._pi_trunk.parameters())
            + list(self.model._pi_mean_head.parameters())
            + list(self.model._pi_log_std_head.parameters()),
            self.grad_clip_norm
        )
        self.pi_optim.step()
        self.model.track_critic_grad(True)

        return pi_loss.item()

    # ------------------------------------------------------------------
    # TD target (mirror SSMRL._td_target)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _td_target(self, next_z, reward, encoder_in):
        """
        Compute TD target: r + gamma * V_target(next_z).

        Args:
            next_z:     [T, batch, latent_dim]
            reward:     [T, batch, 1]
            encoder_in: [batch, ctx_dim]
        Returns:
            td_target: [T, batch, 1]
        """
        T, B, _ = next_z.shape
        z_flat = next_z.reshape(T * B, -1)
        enc_in_flat = encoder_in.unsqueeze(0).expand(T, -1, -1).reshape(T * B, -1)
        next_val = self.model.Q_value(z_flat, enc_in_flat, target=True, return_type='min')
        next_val = next_val.view(T, B, 1)
        return reward + self.discount * next_val

    # ------------------------------------------------------------------
    # Main update (mirror SSMRL.update)
    # ------------------------------------------------------------------
    def update(self, buffer):
        """
        Main update function.  Corresponds to one iteration of model learning.

        Args:
            buffer: replay buffer that yields (obs, action, reward, task)
                    via ``buffer.sample()``.  We ignore task for SSM agent
                    but add done support if provided.
        Returns:
            dict of training statistics.
        """
        # ---- Sample from buffer ----
        sample = buffer.sample()
        # Unpack – buffer may or may not include done

        obs, action, reward, _ = sample
        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)



        # obs:    [horizon+1, batch, state_dim]
        # action: [horizon, batch, act_dim]
        # reward: [horizon, batch, 1]
        H = self.horizon  # horizon

        # ---- Compute targets (no grad) ----
        with torch.no_grad():
            next_mean = self.model.encode(obs[1:])  # [H, B, D]
            # Build encoder_in for target computation using history
            ctx_state_tgt = obs[:self.history_horizon].permute(1, 0, 2)
            ctx_action_tgt = action[:self.history_horizon].permute(1, 0, 2)
            _, _, _, _, encoder_in = self.model.encode_context(
                ctx_state_tgt, ctx_action_tgt, obs[self.history_horizon]
            )
            td_targets = self._td_target(next_mean, reward, encoder_in)

        # ---- Prepare for update ----
        self.model_optim.zero_grad(set_to_none=True)
        self.model.train()

        B = obs.shape[1]
        D = self.latent_dim

        # ---- Encode first obs ----
        z = self.model.encode(obs[self.history_horizon])  # [B, D]
        z_target = self.model.encode(obs[self.history_horizon+1], target=True)
        z_random = z

        # ---- Build context for transformer ----
        ctx_state = obs[:self.history_horizon]
        ctx_action = action[:self.history_horizon]
        ## rearrange the dimension from  [history_horizon, batch, state_dim] to  [batch, history_horizon, state_dim]
        ctx_state = ctx_state.permute(1, 0, 2)
        ctx_action = ctx_action.permute(1, 0, 2)
        A_diag, B_mat, Q_diag, q, encoder_in = self.model.encode_context(
            ctx_state, ctx_action, obs[self.history_horizon]
        )

        # ---- Latent rollout ----
        zs = torch.empty(H + 1, B, D, device=self.device)
        z_randoms = torch.empty(H + 1, B, D, device=self.device)
        zs[0] = z
        z_randoms[0] = z_random
        consistency_loss = torch.tensor(0.0, device=self.device)
        for t in range(H):
            z = self.model.next(z, action[t+self.history_horizon], A_diag, B_mat)
            z_random = self.model.next(z_random, action[t+self.history_horizon], A_diag, B_mat)
            consistency_loss += F.mse_loss(z, next_mean[t+self.history_horizon]) * (self.rho ** t)
            zs[t + 1] = z
            z_randoms[t + 1] = z_random

        # ---- Reward predictions ----
        reward_loss = torch.tensor(0.0, device=self.device)
        for t in range(H):
            r_pred = self.model.reward(z_randoms[t+1], Q_diag, q)
            reward_loss += F.mse_loss(r_pred, reward[t+self.history_horizon]) * (self.rho ** t)

        # ---- Value loss (P-critic) ----
        z_for_p = zs[0]
        # Bellman: P(z) should match r(z) + gamma * (1-d) * P_target(z')

        p_target_val = reward[self.history_horizon] + self.discount * self.model.Q_value(
            z_target, encoder_in, target=True, return_type='min'
        )
        p_pred_all = self.model.Q_value(z_for_p, encoder_in, target=False, return_type='all')
        p_loss = F.mse_loss(
            p_pred_all, p_target_val.detach().expand_as(p_pred_all)
        )
        # Normalise
        consistency_loss = consistency_loss / H
        reward_loss = reward_loss / H
        value_loss = p_loss

        total_loss = (
            self.consistency_coef * consistency_loss
            + self.reward_coef * reward_loss
            + self.value_coef * value_loss
        )

        # ---- Backward & step (world model) ----
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_norm
        )
        self.model_optim.step()


        # ---- Update policy ----
        pi_loss = self.update_pi(zs.detach(), A_diag.detach(), B_mat.detach(), encoder_in.detach())

        # ---- Soft update targets ----
        self.model.soft_update_targets()

        # ---- Return diagnostics ----
        self.model.eval()
        return {
            'consistency_loss': float(consistency_loss.item()),
            'reward_loss': float(reward_loss.item()),
            'value_loss': float(value_loss.item()),
            'p_loss': float(p_loss.item()),
            'pi_loss': pi_loss,
            'total_loss': float(total_loss.item()),
            'grad_norm': float(grad_norm),
            'pi_scale': float(self.scale.value),
        }
