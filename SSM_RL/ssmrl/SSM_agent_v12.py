"""
SSM Agent v11 – rewritten following the architecture of ``ssmrl.py`` (SSMRL / TD-MPC2).

This is a **standalone PyTorch class** (no TensorFlow, no ``base_agent`` inheritance).
It preserves the SSM-specific components from the original TF implementation:
  * Variational encoder (mean + log_sigma)
  * Transformer-based temporal context → time-varying diagonal-A / dense-B dynamics
  * Quadratic reward model (z^T Q z + q^T z + b)
  * Ensemble quadratic P-critics (z^T P z + p^T z + pb)
  * CVXPY-based MPC controller at inference time (with policy-net fallback)

Training loop (``update``) mirrors ``SSMRL.update`` from ``ssmrl.py``:
  sample buffer → encode → latent rollout → consistency / reward / value losses
  → update world model → update policy → soft-update targets.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
from cvxpy import Variable, Parameter, Problem, Minimize, quad_form, hstack, SCS
import cvxpy


from ssmrl.common.ssm_world_model_v4 import SSMWorldModel
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
            {'params': [self.model._P_indices, self.model._p, self.model._pb]},
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

        # Build CVXPY controller (if cvxpy is available)

        self._build_controller()

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
            # Attempt CVXPY planning
            u_norm = self._plan_cvxpy(z, x0, z, eval_mode)
        # Update history
        self.action_history.append(u_norm.copy())
        self.state_history.append(x0.copy())

        ## convert to float32
        u = np.asarray(u_norm, dtype=np.float32)
        return u

    def _plan_cvxpy(self, z, x0, mean_latent, eval_mode):
        """
        Solve the CVXPY MPC problem.  Falls back to the policy net if the
        solver fails.

        Returns:
            u_norm: normalised action, shape [act_dim]
        """

        # Build context
        state_seq = np.array(self.state_history[-self.history_horizon:], dtype=np.float32)
        action_seq = np.array(self.action_history[-self.history_horizon:], dtype=np.float32)
        state_t = torch.tensor(state_seq, device=self.device).unsqueeze(0)
        action_t = torch.tensor(action_seq, device=self.device).unsqueeze(0)
        obs_t = torch.tensor(x0, dtype=torch.float32, device=self.device).unsqueeze(0)

        A_diag, B, Q_diag, q = self.model.encode_context(state_t, action_t, obs_t)
        A_np = torch.diag(A_diag[0]).cpu().numpy()
        B_np = B[0].cpu().numpy()
        Q_np = torch.diag(Q_diag[0]).cpu().numpy()
        q_np = q[0].cpu().numpy()
        z_np = z.cpu().numpy()

        # Get P-critic params
        b_np, P_list, p_np, pb_np = self._get_critic_numpy()

        # Set CVXPY parameters
        self._mean_t_param.value = z_np.reshape(-1, 1)
        self._A_param.value = A_np
        self._B_param.value = B_np
        self._Q_param.value = Q_np
        self._q_param.value = q_np.reshape(1, -1)
        self._b_param.value = b_np

        for i in range(self.num_ensembles):
            self._P_params[i].value = P_list[i]
        self._p_param.value = p_np.squeeze(-1)
        self._pb_param.value = pb_np

        # Solve
        try:
            self._prob.solve(solver=self.solver, warm_start=False)
        except Exception:
            pass

        if (self._prob.status not in ('optimal', 'optimal_inaccurate')
                or self._u_var[:, 0].value is None):
            return self.model.pi(z, deterministic=eval_mode)[0].cpu().numpy()
        else:
            u = np.array(self._u_var[:, 0].value, dtype=np.float32)
            if not eval_mode:
                std = self.model.get_pi_std(z)[0]
                epsilon = std * torch.randn(self.cfg.action_dim, device=std.device)
                epsilon = epsilon.detach().cpu().numpy()
                u += epsilon
            return np.clip(u, -1, 1)

    def _get_critic_numpy(self):
        """Extract P-critic parameters as numpy arrays."""
        b_np = self.model._b.detach().cpu().numpy()
        P_list = []
        P_idx = F.relu(self.model._P_indices).detach().cpu()
        for i in range(self.num_ensembles):
            P_list.append(torch.diag(P_idx[i]).numpy())
        p_np = self.model._p.detach().cpu().numpy()
        pb_np = self.model._pb.detach().cpu().numpy()
        return b_np, P_list, p_np, pb_np

    # ------------------------------------------------------------------
    # CVXPY controller builder (ported from TF version)
    # ------------------------------------------------------------------
    def _build_controller(self):
        if self.cfg.solver == 'SCS':
            self.solver = cvxpy.SCS
        elif self.cfg.solver == 'OSQP':
            self.solver = cvxpy.OSQP

        control_horizon = getattr(self.cfg, 'control_horizon', 5)
        pred_horizon = self.horizon

        self._u_var = Variable((self.act_dim, control_horizon))
        self._A_param = Parameter((self.latent_dim, self.latent_dim))
        self._B_param = Parameter((self.latent_dim, self.act_dim))
        self._Q_param = Parameter((self.latent_dim, self.latent_dim), PSD=True)
        self._q_param = Parameter((1, self.latent_dim))
        self._b_param = Parameter((1,))

        self._P_params = [
            Parameter((self.latent_dim, self.latent_dim), PSD=True)
            for _ in range(self.num_ensembles)
        ]
        self._p_param = Parameter((self.num_ensembles, self.latent_dim))
        self._pb_param = Parameter((self.num_ensembles, 1))

        a_high = getattr(self.cfg, 'a_bound_high', np.ones(self.act_dim))
        a_low = getattr(self.cfg, 'a_bound_low', -np.ones(self.act_dim))
        self._a_high_param = Parameter((self.act_dim,))
        self._a_low_param = Parameter((self.act_dim,))
        self._a_high_param.value = np.asarray(a_high, dtype=np.float64)
        self._a_low_param.value = np.asarray(a_low, dtype=np.float64)

        mean = Variable((self.latent_dim, pred_horizon + 1))
        self._mean_t_param = Parameter((self.latent_dim, 1))
        objective = 0.0
        constraints = [mean[:, 0] == self._mean_t_param[:, 0]]

        for k in range(pred_horizon):
            k_u = min(k, control_horizon - 1)
            mean_k = mean[:, k + 1]
            stage_cost = (
                quad_form(mean_k, self._Q_param)
                + self._q_param @ mean_k
            )
            objective += np.power(self.discount, k) * stage_cost
            u = self._u_var[:, k_u]
            constraints += [
                mean[:, k + 1] == self._A_param @ mean[:, k] + self._B_param @ u
            ]
            objective += cvxpy.norm(u, 2) * self.cfg.u_penalty
            apply_action_constraints = getattr(self.cfg, 'apply_action_constraints', True)
            if apply_action_constraints and k < control_horizon:
                constraints += [self._a_low_param <= u, u <= self._a_high_param]

        # Terminal cost (max over ensemble)
        critic_vals = []
        for i in range(self.num_ensembles):
            mean_k = mean[:, -1]
            cv = (
                quad_form(mean_k, self._P_params[i])
                + self._p_param[i] @ mean_k
                + self._pb_param[i]
            )
            critic_vals.append(cv)
        objective += np.power(self.discount, pred_horizon) * cvxpy.max(hstack(critic_vals))

        self._prob = Problem(Minimize(objective), constraints)

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
    def update_pi(self, zs, A_diag, B_mat):
        """
        Update policy using a sequence of latent states.

        The gradient path is:
            zs (detached) → _pi(zs) → actions → model.next(zs, actions, A, B)
            → z_next → Q_value(z_next)
        so that the policy parameters receive gradients.

        Args:
            zs:     [T, batch, latent_dim]  (detached)
            A_diag: [batch, latent_dim]     (detached)
            B_mat:  [batch, latent_dim, act_dim] (detached)
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
        vals = self.model.Q_value(z_next_flat, target=False, return_type='min')
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
    def _td_target(self, next_z, reward):
        """
        Compute TD target: r + gamma * (1 - done) * V(next_z).

        Args:
            next_z: [T, batch, latent_dim]
            reward:  [T, batch, 1]
            done:    [T, batch, 1]
        Returns:
            td_target: [T, batch, 1]
        """
        T, B, _ = next_z.shape
        z_flat = next_z.reshape(T * B, -1)
        next_val = self.model.Q_value(z_flat, target=True, return_type='min')
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
            td_targets = self._td_target(next_mean, reward)

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
        A_diag, B_mat, Q_diag, q = self.model.encode_context(
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
            z_target, target=True, return_type='min'
        )
        p_pred_all = self.model.Q_value(z_for_p, target=False, return_type='all')
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
        pi_loss = self.update_pi(zs.detach(), A_diag.detach(), B_mat.detach())

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
