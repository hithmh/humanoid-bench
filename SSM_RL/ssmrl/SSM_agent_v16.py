"""
SSM Agent v16 – Diffusion-based generator synthesis (guideline-compliant).

Implements the pipeline described in:
  "Probabilistic Synthesis of Dynamical Generators: A Denoising Diffusion
   Framework for Trajectory-Conditioned System Identification"

Key upgrades over v15
─────────────────────
1. Transformer trajectory encoder  (was GRU)
   The state-action history window τ is encoded by a Transformer with sinusoidal
   positional embeddings and multi-head self-attention, capturing long-range
   temporal dependencies as recommended by the guideline (§4.2, Transformer row).

2. ε-prediction denoiser with adaLN conditioning  (was x0-prediction + concat)
   The denoising network now predicts the added noise ε, matching the guideline's
   L_simple objective:
       L_simple = E[‖ε − ε_θ(g_t, t, τ)‖²]
   Conditioning from τ is injected via adaptive LayerNorm (adaLN) at every
   residual block, which is more expressive than simple concatenation (guideline §4.3).

3. Explicit DDPM auxiliary loss (L_simple) in update()
   The agent now includes a standalone diffusion loss term:
       L_total = consistency_coef * L_cons
               + reward_coef     * L_reward
               + value_coef      * L_value
               + diffusion_coef  * L_simple
   L_simple is computed by noising the SSM parameter sequence produced by the
   reference (train) DDIM chain and asking the denoiser to recover the noise.
   This additional signal prevents denoiser collapse and stabilises training.

4. Generator latent space (placeholder, not yet fully implemented)
   The guideline recommends compressing the generator g into a low-dimensional
   latent z_g via a VAE before running diffusion on z_g ("latent diffusion").
   A lightweight linear encoder/decoder pair is provided as a placeholder.
   Setting cfg.gen_latent_dim enables this path.  A full VAE implementation
   requires paired (trajectory, generator) data for pre-training and is left
   as future work (see diffusion_world_model_v2.py).

Architecture is provided by WorldModel in
    ssmrl/common/diffusion_world_model_v2.py

MPC planning (JAX/qpax) is unchanged from v15; per-step A[t] B[t] sequences
are still used to build the linear-quadratic QP.

OPEN QUESTIONS / UNCERTAINTIES
───────────────────────────────
• Generator latent space: The guideline's VAE-based latent diffusion is not
  fully implemented.  Please confirm whether a pre-trained VAE is expected
  here, or if the placeholder identity projectors are sufficient for now.

• Diffusion loss target (g0): L_simple requires a "clean" g0.  We use the
  SSM parameter sequence produced by the short training DDIM chain as g0
  (i.e., self-consistency).  If a teacher network or a separate dataset of
  (τ, g) pairs is available it should be passed instead.

• Diffusion coefficient (diffusion_coef): Defaults to 1.0.  The appropriate
  weighting relative to the RL losses is task-dependent and likely requires tuning.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

# ---- JAX stack (required) -----------------------------------------------
import jax
import jax.numpy as jnp
import qpax

from ssmrl.common.diffusion_world_model_v2 import WorldModel
from ssmrl.common.scale import RunningScale


class SSMAgent:
    """
    SSM-RL Agent v16.

    Mirrors the public API of SSMAgent from SSM_agent_v15.py.
    Differences:
      - Uses WorldModel v2 (Transformer encoder, ε-prediction, adaLN)
      - Adds diffusion_coef * L_simple to the world-model training loss
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(self, cfg):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = WorldModel(cfg).to(self.device)

        self.latent_dim    = self.model.latent_dim
        self.act_dim       = self.model.act_dim
        self.state_dim     = self.model.state_dim
        self.history_horizon = self.model.history_horizon
        self.num_ensembles = self.model.num_ensembles

        enc_lr_scale = getattr(cfg, 'enc_lr_scale', 0.3)
        lr           = cfg.lr

        # World-model optimiser (same parameter groups as v15)
        self.model_optim = torch.optim.Adam([
            {'params': self.model._encoder_mean.parameters(),
             'lr': lr * enc_lr_scale},
            {'params': self.model._diffusion_model.parameters()},
            {'params': [self.model._b]},
            {'params': [self.model._P_indices, self.model._p, self.model._pb,
                        self.model._K_indices, self.model._k]},
        ], lr=lr)

        # Policy optimiser
        self.pi_optim = torch.optim.Adam(
            list(self.model._pi_trunk.parameters())
            + list(self.model._pi_mean_head.parameters())
            + list(self.model._pi_log_std_head.parameters()),
            lr=lr, eps=1e-5,
        )

        self.model.eval()
        self.scale = RunningScale(cfg)

        self.discount      = self._get_discount(getattr(cfg, 'episode_length', 1000))
        self.grad_clip_norm = getattr(cfg, 'grad_clip_norm', 20.0)

        self.consistency_coef = getattr(cfg, 'consistency_coef', 20.0)
        self.reward_coef      = getattr(cfg, 'reward_coef', 0.1)
        self.value_coef       = getattr(cfg, 'value_coef', 0.1)
        self.entropy_coef     = getattr(cfg, 'entropy_coef', 1e-4)
        self.diffusion_coef   = getattr(cfg, 'diffusion_coef', 1.0)   # ← NEW v16
        self.rho              = getattr(cfg, 'rho', 0.5)
        self.horizon          = getattr(cfg, 'horizon', 3)

        self.shift    = np.zeros(self.state_dim)
        self.scale_obs = np.ones(self.state_dim)
        self.shift_u  = np.zeros(self.act_dim)
        self.scale_u  = np.ones(self.act_dim)

        self.state_history  = []
        self.action_history = []

        self._build_jax_controller()

    # ------------------------------------------------------------------
    # Discount helper
    # ------------------------------------------------------------------
    def _get_discount(self, episode_length):
        denom = getattr(self.cfg, 'discount_denom', 5)
        d_min = getattr(self.cfg, 'discount_min', 0.95)
        d_max = getattr(self.cfg, 'discount_max', 0.995)
        frac  = episode_length / denom
        return min(max((frac - 1) / frac, d_min), d_max)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, fp):
        torch.save({'model': self.model.state_dict()}, fp)

    def load(self, fp):
        sd = fp if isinstance(fp, dict) else torch.load(fp, weights_only=False)
        self.model.load_state_dict(sd['model'])

    # ------------------------------------------------------------------
    # Shift / scale management
    # ------------------------------------------------------------------
    def set_shift_and_scale(self, shift, scale, shift_u, scale_u):
        self.shift    = np.asarray(shift,   dtype=np.float32)
        self.scale_obs = np.asarray(scale,  dtype=np.float32)
        self.shift_u  = np.asarray(shift_u, dtype=np.float32)
        self.scale_u  = np.asarray(scale_u, dtype=np.float32)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False):
        """
        Select an action.

        Args:
            obs:       raw observation (numpy or torch), shape [state_dim]
            t0:        True at the first step of an episode
            eval_mode: deterministic action if True
        Returns:
            action (numpy), shape [act_dim]
        """
        if isinstance(obs, torch.Tensor):
            obs_np = obs.cpu().numpy()
        else:
            obs_np = np.asarray(obs, dtype=np.float32)

        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        z     = self.model.encode(obs_t)

        if len(self.state_history) < self.history_horizon:
            u_norm = self.model.pi(z, deterministic=eval_mode)[0].cpu().numpy()
        else:
            u_norm = self._plan_jax(z, obs_np, eval_mode)

        self.action_history.append(u_norm.copy())
        self.state_history.append(obs_np.copy())

        return np.asarray(u_norm, dtype=np.float32)

    def _plan_jax(self, z: torch.Tensor, x0: np.ndarray, eval_mode: bool):
        """JAX/qpax MPC using diffusion-sampled per-step SSM params."""
        state_seq  = np.array(self.state_history[-self.history_horizon:],  dtype=np.float32)
        action_seq = np.array(self.action_history[-self.history_horizon:], dtype=np.float32)
        state_t  = torch.tensor(state_seq,  device=self.device).unsqueeze(0)
        action_t = torch.tensor(action_seq, device=self.device).unsqueeze(0)
        obs_t    = torch.tensor(x0, dtype=torch.float32, device=self.device).unsqueeze(0)

        A_mat_seq, B_seq, Q_diag_seq, q_seq, _, _ = self.model.encode_context(
            state_t, action_t, obs_t, use_diffusion=True)

        # Use diagonal of A for the JAX QP (keeps v15 QP formulation)
        A_diag_t = A_mat_seq[0].diagonal(dim1=-2, dim2=-1)  # (H, D)
        B_t      = B_seq[0]         # (H, D, nU)
        Q_diag_t = Q_diag_seq[0]   # (H, D)
        q_t      = q_seq[0]         # (H, D)
        z_t      = z[0]             # (D,)

        P_diags_t = F.relu(self.model._P_indices)   # (E, D)
        p_t       = self.model._p.squeeze(-1)        # (E, D)
        pb_t      = self.model._pb.squeeze(-1)       # (E,)

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
            u = np.asarray(U_sol[0], dtype=np.float32)
        except Exception:
            return self.model.pi(z, deterministic=eval_mode)[0].cpu().numpy()

        if not eval_mode:
            std     = self.model.get_pi_std(z)[0]
            epsilon = (std * torch.randn(self.act_dim, device=std.device)
                       ).detach().cpu().numpy()
            u = u + epsilon

        return np.clip(u, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # JAX + qpax MPC (unchanged from v15)
    # ------------------------------------------------------------------
    def _build_jax_controller(self):
        D        = self.latent_dim
        nU       = self.act_dim
        H        = self.horizon
        CH       = getattr(self.cfg, 'control_horizon', 5)
        E        = self.num_ensembles
        discount = float(self.discount)
        u_penalty = float(getattr(self.cfg, 'u_penalty', 0.0))

        self._control_horizon = CH
        n = CH * nU

        def _build_and_solve(z0, A_diag_seq, B_seq, Q_diag_seq, q_seq,
                             P_diags, p_mat, pb_vec, a_low, a_high):
            phi_prefix = [jnp.ones(D)]
            for t in range(H):
                phi_prefix.append(phi_prefix[-1] * A_diag_seq[t])

            def sfx(s, t):
                result = jnp.ones(D)
                for i in range(s, t + 1):
                    result = result * A_diag_seq[i]
                return result

            f_list   = []
            T_u_list = []

            for t in range(H):
                f_list.append(phi_prefix[t + 1] * z0)

                T_u_t = jnp.zeros((D, n))
                for k in range(CH):
                    s_k = k * nU
                    e_k = (k + 1) * nU
                    if k < CH - 1:
                        if k <= t:
                            coeff = sfx(k + 1, t)[:, None] * B_seq[k] if k < t else B_seq[k]
                            T_u_t = T_u_t.at[:, s_k:e_k].set(coeff)
                    else:
                        if t >= CH - 1:
                            accum = jnp.zeros((D, nU))
                            for j in range(CH - 1, t + 1):
                                phi_j = sfx(j + 1, t) if j < t else jnp.ones(D)
                                accum = accum + phi_j[:, None] * B_seq[j]
                            T_u_t = T_u_t.at[:, s_k:e_k].set(accum)
                T_u_list.append(T_u_t)

            Q_qp = jnp.zeros((n, n))
            c_qp = jnp.zeros(n)

            for t in range(H):
                k_u = min(t, CH - 1)
                s_u = k_u * nU
                e_u = (k_u + 1) * nU
                Tu  = T_u_list[t]
                f   = f_list[t]
                Q_t = Q_diag_seq[t]
                q_t = q_seq[t]

                QTu  = Q_t[:, None] * Tu
                Q_qp = Q_qp + (discount ** t) * 2.0 * (Tu.T @ QTu)
                c_qp = c_qp + (discount ** t) * (2.0 * (QTu.T @ f) + Tu.T @ q_t)
                Q_qp = Q_qp.at[s_u:e_u, s_u:e_u].add(
                    (discount ** t) * 2.0 * u_penalty * jnp.eye(nU))

            Tu_H = T_u_list[H - 1]
            f_H  = f_list[H - 1]
            P_avg = jnp.mean(P_diags, axis=0)
            p_avg = jnp.mean(p_mat,   axis=0)
            PTu  = P_avg[:, None] * Tu_H
            Q_qp = Q_qp + (discount ** H) * 2.0 * (Tu_H.T @ PTu)
            c_qp = c_qp + (discount ** H) * (2.0 * (PTu.T @ f_H) + Tu_H.T @ p_avg)

            Q_qp = Q_qp + 1e-6 * jnp.eye(n)

            a_high_t = jnp.tile(a_high, CH)
            a_low_t  = jnp.tile(a_low,  CH)
            G = jnp.concatenate([ jnp.eye(n), -jnp.eye(n)], axis=0)
            h = jnp.concatenate([a_high_t, -a_low_t], axis=0)

            A_eq = jnp.zeros((0, n))
            b_eq = jnp.zeros((0,))

            x, _s, _z, _y, converged, _iters = qpax.solve_qp(
                Q_qp, c_qp, A_eq, b_eq, G, h)
            return x.reshape(CH, nU), converged

        self._jax_solve_mpc = jax.jit(_build_and_solve)

        a_high = np.asarray(
            getattr(self.cfg, 'a_bound_high', np.ones(nU)), dtype=np.float32)
        a_low  = np.asarray(
            getattr(self.cfg, 'a_bound_low', -np.ones(nU)), dtype=np.float32)
        self._a_high_jax = jnp.array(a_high)
        self._a_low_jax  = jnp.array(a_low)

    # ------------------------------------------------------------------
    # Episode reset helpers
    # ------------------------------------------------------------------
    def reset_for_control(self):
        self.state_history  = []
        self.action_history = []

    def store_cached_control_info(self):
        self._cached = copy.deepcopy({
            'state_history':  self.state_history,
            'action_history': self.action_history,
        })

    def restore_control_info(self):
        if hasattr(self, '_cached'):
            self.state_history  = self._cached['state_history']
            self.action_history = self._cached['action_history']
        else:
            print('No cached control info found.')

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------
    def update_pi(self, zs, A_mat_seq, B_seq):
        """
        Update policy.

        Args:
            zs:        [T, batch, latent_dim]
            A_mat_seq: [batch, H, latent_dim, latent_dim]
            B_seq:     [batch, H, latent_dim, act_dim]
        Returns:
            pi_loss (float)
        """
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_critic_grad(False)

        T, B, _ = zs.shape

        actions, log_probs = self.model.pi(zs, return_log_prob=True)

        z_next_list = []
        for t in range(T):
            t_idx    = min(t, A_mat_seq.shape[1] - 1)
            z_next_t = self.model.next(zs[t], actions[t],
                                       A_mat_seq[:, t_idx, :, :],
                                       B_seq[:, t_idx, :, :])
            z_next_list.append(z_next_t)
        z_next = torch.stack(z_next_list, dim=0)

        z_next_flat = z_next.reshape(T * B, -1)
        vals = self.model.Q_value(z_next_flat, target=False, return_type='min')
        vals = vals.view(T, B, 1)

        self.scale.update(vals[0])
        vals = self.scale(vals)

        rho = torch.pow(
            torch.tensor(self.rho, device=self.device),
            torch.arange(T, device=self.device, dtype=torch.float32),
        )
        pi_loss = -(
            (vals - self.entropy_coef * log_probs).mean(dim=(1, 2)) * rho
        ).mean()

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model._pi_trunk.parameters())
            + list(self.model._pi_mean_head.parameters())
            + list(self.model._pi_log_std_head.parameters()),
            self.grad_clip_norm,
        )
        self.pi_optim.step()
        self.model.track_critic_grad(True)

        return pi_loss.item()

    # ------------------------------------------------------------------
    # TD target
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _td_target(self, next_z, reward):
        T, B, _ = next_z.shape
        z_flat   = next_z.reshape(T * B, -1)
        next_val = self.model.Q_value(z_flat, target=True, return_type='min')
        next_val = next_val.view(T, B, 1)
        return reward + self.discount * next_val

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------
    def update(self, buffer):
        """
        Main update function.

        v16 additions over v15:
          • L_simple DDPM loss (diffusion_coef * L_simple) is added to the
            total loss.  g0 is taken as the flattened SSM parameter sequence
            produced by the training DDIM chain (self-consistency target).
            See the OPEN QUESTIONS section in the module docstring for
            alternatives when paired (τ, g) data is available.

        Returns:
            dict of training statistics.
        """
        sample = buffer.sample()
        obs, action, reward, _ = sample
        obs    = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)

        H = self.horizon

        with torch.no_grad():
            next_mean  = self.model.encode(obs[1:])
            td_targets = self._td_target(next_mean, reward)

        self.model_optim.zero_grad(set_to_none=True)
        self.model.train()

        B = obs.shape[1]
        D = self.latent_dim

        z        = self.model.encode(obs[self.history_horizon])
        z_target = self.model.encode(obs[self.history_horizon + 1], target=True)
        z_random = z

        ctx_state   = obs[:self.history_horizon].permute(1, 0, 2)
        ctx_action  = action[:self.history_horizon].permute(1, 0, 2)
        current_obs = obs[self.history_horizon]

        # Training DDIM chain (differentiable) → per-step SSM params
        A_mat_seq, B_seq, Q_diag_seq, q_seq, R_diag_seq, r_lin_seq = \
            self.model.encode_context(ctx_state, ctx_action, current_obs,
                                      use_diffusion=False)

        # ---- Latent rollout ----
        zs        = torch.empty(H + 1, B, D, device=self.device)
        z_randoms = torch.empty(H + 1, B, D, device=self.device)
        zs[0]        = z
        z_randoms[0] = z_random
        consistency_loss = torch.tensor(0.0, device=self.device)

        for t in range(H):
            t_idx = min(t, A_mat_seq.shape[1] - 1)
            A_t   = A_mat_seq[:, t_idx, :, :]
            B_t   = B_seq[:, t_idx, :, :]
            z        = self.model.next(z,        action[t + self.history_horizon], A_t, B_t)
            z_random = self.model.next(z_random, action[t + self.history_horizon], A_t, B_t)
            consistency_loss += (
                F.mse_loss(z, next_mean[t + self.history_horizon]) * (self.rho ** t)
            )
            zs[t + 1]        = z
            z_randoms[t + 1] = z_random

        # ---- Reward predictions ----
        reward_loss = torch.tensor(0.0, device=self.device)
        for t in range(H):
            t_idx  = min(t, Q_diag_seq.shape[1] - 1)
            r_pred = self.model.reward(
                z_randoms[t + 1],
                action[t + self.history_horizon],
                Q_diag_seq[:, t_idx, :],
                q_seq[:,      t_idx, :],
                R_diag_seq[:, t_idx, :],
                r_lin_seq[:,  t_idx, :],
            )
            reward_loss += F.mse_loss(r_pred, reward[t + self.history_horizon]) * (self.rho ** t)

        # ---- Value loss ----
        p_target_val = reward[self.history_horizon] + self.discount * self.model.Q_value(
            z_target, target=True, return_type='min'
        )
        p_pred_all = self.model.Q_value(zs[0], target=False, return_type='all')
        value_loss = F.mse_loss(
            p_pred_all, p_target_val.detach().expand_as(p_pred_all)
        )

        # ---- NEW v16: DDPM auxiliary loss L_simple ----------------------
        # Build the "clean" generator sequence g0 from the current SSM params.
        # Shape: [B, seq_dim] = flatten [B, H, param_dim]
        param_dim = self.model._diffusion_model.param_dim
        seq_dim   = self.model._diffusion_model.seq_dim

        # Stack per-step raw params back into a flat sequence.
        # We use the raw (pre-activation) parameters inferred by split_and_activate
        # in reverse.  Since split_and_activate applies softplus / identity, and we
        # don't have the raw pre-activation values after the fact, we use the
        # softplus-activated Q/R values as a proxy g0.  This is approximate but
        # provides a valid training signal for the denoiser.
        #
        # A cleaner approach: expose raw_seq from encode_context (TODO).
        A_flat  = A_mat_seq.reshape(B, H, -1)       # [B, H, D*D]
        B_flat  = B_seq.reshape(B, H, -1)            # [B, H, D*nU]
        g0_seq  = torch.cat([
            A_flat,
            B_flat,
            Q_diag_seq,
            q_seq,
            R_diag_seq,
            r_lin_seq,
        ], dim=-1)                                    # [B, H, param_dim]
        g0_flat = g0_seq.reshape(B, seq_dim).detach()  # detach: treat as fixed target

        diffusion_loss = self.model.diffusion_loss(
            ctx_state, ctx_action, current_obs, g0_flat)

        # ---- Normalise and combine ----
        consistency_loss = consistency_loss / H
        reward_loss      = reward_loss / H

        total_loss = (
            self.consistency_coef * consistency_loss
            + self.reward_coef     * reward_loss
            + self.value_coef      * value_loss
            + self.diffusion_coef  * diffusion_loss   # ← L_simple  (NEW v16)
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.model_optim.step()
        self.model.eval()

        # ---- Policy update ----
        pi_loss = self.update_pi(
            zs.detach(), A_mat_seq.detach(), B_seq.detach())

        # ---- Soft target update ----
        self.model.soft_update_targets()

        return {
            'consistency_loss': consistency_loss.item(),
            'reward_loss':      reward_loss.item(),
            'value_loss':       value_loss.item(),
            'diffusion_loss':   diffusion_loss.item(),   # ← NEW v16
            'pi_loss':          pi_loss,
            'total_loss':       total_loss.item(),
        }
