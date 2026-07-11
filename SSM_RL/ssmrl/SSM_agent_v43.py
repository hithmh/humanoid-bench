"""
SSM Agent v42- transformer-conditioned SSM-RL with JAX smooth MPC.

This agent wraps ``SSMWorldModel`` and provides both model learning and
inference-time control. During inference, recent state/action history is used
by ``encode_context`` to produce local linear SSM parameters, softplus
reward parameters, and critic context. Until enough history is available, or
if the MPC solver fails, actions come from the learned policy.

MPC controller:
  * The finite-horizon control objective is assembled once as a JAX function
    and JIT-compiled.
  * A projected first-order optimizer solves over the stacked action sequence
    directly with box projection.
  * The optimizer is initialized from a policy rollout through the local
    predictive latent dynamics; stochastic policy actions are used during
    training control and deterministic actions during evaluation.
  * Dense latent dynamics are analytically unrolled, eliminating equality
    dynamics constraints:
      z_{t+1} = A_t ... A_0 z_0 + T_u[t] @ U_flat
  * Per-step action bounds are encoded as box constraints.
  * The stage reward and terminal arrival Q are softplus networks. With signed
    output weights, minimizing their negation is generally nonconvex; the JAX
    projected Adam optimizer is a local smooth box-constrained solver.
  * Training-time exploration uses stochastic policy rollout initialization
    and adds policy-standard-deviation action noise after planning.
  * PyTorch tensors are passed to JAX through DLPack when the backends share a
    device, avoiding per-step NumPy conversion.

MPC objective:
  min_U -sum_t gamma^t w1_t^T softplus(W2_t z_t(U) + W3_t u_t + b1_t)
        -gamma^H w1_H^T softplus(W4_H softplus(W2_H z_H(U) + W3_H u_H + b1_H) + b3_H)
  s.t.  a_low <= u_t <= a_high

Training loop:
  sample replay buffer -> encode context -> latent rollout -> optimize
  consistency, reward, critic value, and observation reconstruction losses ->
  update SAC-style policy with entropy regularization -> soft-update targets.
"""

import os
import copy
import numpy as np
import torch
import torch.nn.functional as F

import jax
import jax.numpy as jnp


from ssmrl.common.ssm_world_model_v43 import SSMWorldModel
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
        self.device = self._resolve_device(getattr(cfg, "device", "auto"))
        self._jax_cpu_device = jax.devices("cpu")[0]
        jax_platform = str(getattr(cfg, "jax_mpc_platform", "cpu")).lower()
        self._jax_mpc_cuda_requested = jax_platform in {"cuda", "gpu"}
        self._jax_has_cuda = (
            self._jax_mpc_cuda_requested
            and any(d.platform in ('cuda', 'gpu') for d in jax.devices())
        )
        self._jax_disable_cuda_after_error = bool(getattr(
            cfg, 'jax_mpc_disable_cuda_after_error', True))
        self._jax_mpc_cuda_disabled = False
        self._jax_mpc_cuda_disable_reason = ''

        # World model
        self.model = SSMWorldModel(cfg).to(self.device)

        # Dimensions (convenience)
        self.latent_dim = self.model.latent_dim
        self.act_dim = self.model.act_dim
        self.state_dim = self.model.state_dim
        self.history_horizon = self.model.history_horizon

        # ---- Optimizers (following ssmrl.py pattern: model optim + pi optim) ----
        enc_lr_scale = getattr(cfg, 'enc_lr_scale', 0.3)
        lr = cfg.lr
        self.l2_regularizer = float(getattr(cfg, 'l2_regularizer', 0.0))

        # Group 1 – world-model parameters (encoder_mean at scaled lr)
        self.model_optim = torch.optim.Adam([
            {'params': self.model._encoder_mean.parameters(),
             'lr': lr * enc_lr_scale},
            {'params': self.model._transformer.parameters()},
            {'params': self.model._A_net.parameters()},
            {'params': [self.model._A_basis]},
            {'params': self.model._B_net.parameters()},
            {'params': [self.model._B_basis]},
            {'params': self.model.reward_head_parameters()},
            {'params': self.model.arrival_head_parameters()},
        ], lr=lr, weight_decay=self.l2_regularizer)


        # Group 2 - TD-MPC2-style latent policy prior
        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(),
            lr=lr, eps=1e-5, weight_decay=self.l2_regularizer
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

        # History buffers and episode-scoped control diagnostics
        self.state_history = []
        self.action_history = []
        self._state_history_tensors = []
        self._action_history_tensors = []
        self._reset_convex_diagnostics()

        # Build JAX smooth MPC controller
        self._build_convex_controller()

    def _resolve_device(self, requested):
        requested = str(requested).lower()
        if requested in {"auto", "none", "???", ""}:
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        return torch.device(requested)

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
            obs_t = obs.to(device=self.device, dtype=torch.float32).reshape(1, -1)
            obs_np = None
        else:
            obs_np = np.asarray(obs, dtype=np.float32)
            obs_t = torch.as_tensor(
                obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Encode
        z = self.model.encode(obs_t)

        # Decide action
        if len(self._state_history_tensors) < self.history_horizon:
            # Not enough history for transformer - use policy net on latent state
            u_norm = self.model.pi(z, deterministic=eval_mode)[0].cpu().numpy()
        else:
            # Attempt JAX smooth convex planning
            u_norm = self._plan_convex(z, obs_t, eval_mode)
        # u_norm = self._sanitize_action(u_norm)
        # Update history
        self._append_history(obs_np, obs_t, u_norm)

        ## convert to float32
        u = np.asarray(u_norm, dtype=np.float32)
        return u

    def _sanitize_action(self, action):
        """Return a finite clipped action so MuJoCo never receives NaN controls."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.act_dim:
            safe = np.zeros(self.act_dim, dtype=np.float32)
            safe[:min(action.shape[0], self.act_dim)] = action[:min(action.shape[0], self.act_dim)]
            action = safe
        if not np.isfinite(action).all():
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _is_jax_cuda_error(exc):
        msg = str(exc).lower()
        return any(token in msg for token in (
            'cuda_error',
            'cuda error',
            'failed to allocate device memory',
            'unknown backend cuda',
            'gpu backend',
        ))

    def _disable_jax_cuda_mpc(self, reason):
        if not self._jax_disable_cuda_after_error:
            return
        self._jax_has_cuda = False
        self._jax_mpc_cuda_disabled = True
        self._jax_mpc_cuda_disable_reason = str(reason)[:512]

    def _append_history(self, obs_np: np.ndarray, obs_t: torch.Tensor,
                        action_np: np.ndarray):
        # if obs_np is not None:
        #     self.state_history.append(np.asarray(obs_np, dtype=np.float32).copy())
        # self.action_history.append(np.asarray(action_np, dtype=np.float32).copy())
        self._state_history_tensors.append(obs_t[0].detach().clone())
        self._action_history_tensors.append(
            torch.as_tensor(action_np, dtype=torch.float32, device=self.device).detach().clone())

    # def _rebuild_tensor_histories(self):
    #     self._state_history_tensors = [
    #         torch.as_tensor(x, dtype=torch.float32, device=self.device)
    #         for x in self.state_history
    #     ]
    #     self._action_history_tensors = [
    #         torch.as_tensor(u, dtype=torch.float32, device=self.device)
    #         for u in self.action_history
    #     ]

    @torch.no_grad()
    def _policy_rollout_mpc_init(self, z0: torch.Tensor, A_seq: torch.Tensor,
                                 B_seq: torch.Tensor, eval_mode: bool):
        """
        Generate the MPC initial action sequence by rolling out pi through the
        local predictive latent model.
        """
        z_roll = z0.view(1, -1)
        a_low = torch.as_tensor(
            self._a_low_np, dtype=z_roll.dtype, device=z_roll.device).view(1, -1)
        a_high = torch.as_tensor(
            self._a_high_np, dtype=z_roll.dtype, device=z_roll.device).view(1, -1)
        actions = []
        for t in range(self._control_horizon):
            u = self.model.pi(z_roll, deterministic=eval_mode)
            u = torch.nan_to_num(u, nan=0.0, posinf=1.0, neginf=-1.0)
            u = torch.clamp(u, a_low, a_high)
            actions.append(u[0])
            dyn_t = min(t, A_seq.shape[0] - 1)
            z_roll = self.model.next(
                z_roll,
                u,
                A_seq[dyn_t].unsqueeze(0),
                B_seq[dyn_t].unsqueeze(0),
            )
        return torch.stack(actions, dim=0)

    def _plan_convex(self, z: torch.Tensor, obs_t: torch.Tensor, eval_mode: bool):
        """
        Solve the JAX smooth MPC problem. Falls back to the policy net on
        numerical failure or solver non-convergence.

        Returns:
            u_norm: normalised action, shape [act_dim]
        """
        # if (len(self._state_history_tensors) < self.history_horizon
        #         or len(self._action_history_tensors) < self.history_horizon):
        #     self._rebuild_tensor_histories()
        state_t = torch.stack(
            self._state_history_tensors[-self.history_horizon:], dim=0).unsqueeze(0)
        action_t = torch.stack(
            self._action_history_tensors[-self.history_horizon:], dim=0).unsqueeze(0)

        (A_seq, B_seq,
         reward_w1_seq, reward_w2_seq, reward_w3_seq,
         reward_b1_seq, reward_b2_seq,
         encoder_in) = self.model.encode_context(
            state_t,
            action_t,
            obs_t,
        )

        # Extract single-sample tensors
        A_seq_t = A_seq[0]                         # (H, D, D)
        B_seq_t = B_seq[0]                         # (H, D, nU)
        reward_w1_t = reward_w1_seq[0]             # (H, K), signed output weight
        reward_w2_t = reward_w2_seq[0]             # (H, K, D)
        reward_w3_t = reward_w3_seq[0]             # (H, K, nU)
        reward_b1_t = reward_b1_seq[0]             # (H, K)
        reward_b2_t = reward_b2_seq[0]             # (H, 1), constant for MPC
        z_t = z[0]                                 # (D,)

        (arrival_w1, arrival_w2, arrival_w3, arrival_w4,
         arrival_b1, arrival_b3, arrival_b2) = self.model.arrival_Q_params(
            encoder_in, target=False, return_type='all')
        if eval_mode:
            arrival_head_idx = slice(None)
        else:
            idx = torch.randint(arrival_w1.shape[0], (1,), device=arrival_w1.device).item()
            arrival_head_idx = slice(idx, idx + 1)
        arrival_w1_t = arrival_w1[arrival_head_idx, 0]  # (E_arr, K2_arr)
        arrival_w2_t = arrival_w2[arrival_head_idx, 0]  # (E_arr, K1_arr, D)
        arrival_w3_t = arrival_w3[arrival_head_idx, 0]  # (E_arr, K1_arr, nU)
        arrival_w4_t = arrival_w4[arrival_head_idx, 0]  # (E_arr, K2_arr, K1_arr)
        arrival_b1_t = arrival_b1[arrival_head_idx, 0]  # (E_arr, K1_arr)
        arrival_b3_t = arrival_b3[arrival_head_idx, 0]  # (E_arr, K2_arr)
        arrival_b2_t = arrival_b2[arrival_head_idx, 0]  # (E_arr, 1), constant for MPC

        self.convex_solver_attempts += 1
        input_tensors = {
            'z': z_t,
            'A': A_seq_t,
            'B': B_seq_t,
            'reward_w1': reward_w1_t,
            'reward_w2': reward_w2_t,
            'reward_w3': reward_w3_t,
            'reward_b1': reward_b1_t,
            'reward_b2': reward_b2_t,
            'arrival_w1': arrival_w1_t,
            'arrival_w2': arrival_w2_t,
            'arrival_w3': arrival_w3_t,
            'arrival_w4': arrival_w4_t,
            'arrival_b1': arrival_b1_t,
            'arrival_b3': arrival_b3_t,
            'arrival_b2': arrival_b2_t,
        }
        # bad_inputs = [name for name, tensor in input_tensors.items()
        #               if not torch.isfinite(tensor).all().item()]
        # if bad_inputs:
        #     self._record_convex_failure('input_nonfinite', ','.join(bad_inputs))
        #     for name in bad_inputs:
        #         self.convex_nonfinite_inputs[name] += 1
        #     return self._sanitize_action(self.model.pi(z, deterministic=eval_mode)[0].cpu().numpy())

        U_init_t = self._policy_rollout_mpc_init(
            z_t, A_seq_t, B_seq_t, eval_mode)
        if not torch.isfinite(U_init_t).all().item():
            self._record_convex_failure('input_nonfinite', 'U_init')
            self.convex_nonfinite_inputs['U_init'] += 1
            return self._sanitize_action(self.model.pi(z, deterministic=eval_mode)[0].cpu().numpy())

        def to_jax_cpu(t: torch.Tensor):
            return jax.device_put(t.detach().cpu().numpy(), self._jax_cpu_device)

        def to_jax(t: torch.Tensor):
            t = t.detach().contiguous()
            if t.is_cuda and not self._jax_has_cuda:
                return to_jax_cpu(t)
            try:
                return jax.dlpack.from_dlpack(t)
            except TypeError:
                try:
                    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(t))
                except RuntimeError as exc:
                    if t.is_cuda and self._is_jax_cuda_error(exc):
                        self._disable_jax_cuda_mpc(exc)
                        return to_jax_cpu(t)
                    raise
            except RuntimeError as exc:
                if t.is_cuda and self._is_jax_cuda_error(exc):
                    self._disable_jax_cuda_mpc(exc)
                    return to_jax_cpu(t)
                raise

        try:
            z_jax = to_jax(z_t)
            A_jax = to_jax(A_seq_t)
            U_init = jax.device_put(to_jax(U_init_t), z_jax.device).astype(z_jax.dtype)
            a_low_jax = jax.device_put(self._a_low_np, z_jax.device).astype(z_jax.dtype)
            a_high_jax = jax.device_put(self._a_high_np, z_jax.device).astype(z_jax.dtype)
            U_sol, converged = self._jax_solve_mpc(
                z_jax,
                A_jax,
                to_jax(B_seq_t),
                to_jax(reward_w1_t),
                to_jax(reward_w2_t),
                to_jax(reward_w3_t),
                to_jax(reward_b1_t),
                to_jax(arrival_w1_t),
                to_jax(arrival_w2_t),
                to_jax(arrival_w3_t),
                to_jax(arrival_w4_t),
                to_jax(arrival_b1_t),
                to_jax(arrival_b3_t),
                U_init,
                a_low_jax,
                a_high_jax,
            )
            if not bool(converged):
                self._record_convex_failure('nonconverged')
                return self._sanitize_action(self.model.pi(z, deterministic=eval_mode)[0].cpu().numpy())
            u = np.asarray(U_sol[0], dtype=np.float32)   # first control step
            if not np.isfinite(u).all():
                self._record_convex_failure('solution_nonfinite')
                return self._sanitize_action(self.model.pi(z, deterministic=eval_mode)[0].cpu().numpy())
        except Exception as exc:
            if self._is_jax_cuda_error(exc):
                self._disable_jax_cuda_mpc(exc)
            self._record_convex_failure('exception', exc)
            return self._sanitize_action(self.model.pi(z, deterministic=eval_mode)[0].cpu().numpy())

        if not eval_mode:
            std = self.model.get_pi_std(z)[0]
            epsilon = (std * torch.randn(self.act_dim, device=std.device)
                       ).detach().cpu().numpy()
            # u = u + epsilon

        return self._sanitize_action(u)

    # ------------------------------------------------------------------
    # JAX smooth MPC controller
    # ------------------------------------------------------------------
    def _build_convex_controller(self):
        """
        Build and JIT-compile the projected optimizer for smooth MPC.

        Dynamics are unrolled inside the objective. Signed W1 removes the
        concavity guarantee, so this is generally a nonconvex smooth
        box-constrained problem optimized locally with projected Adam.
        """
        nU  = self.act_dim
        self._control_horizon = max(
            1, min(int(getattr(self.cfg, 'control_horizon', self.horizon)), self.horizon))
        self._softplus_beta_reward = float(getattr(
            self.cfg, 'reward_softplus_beta', 1.0))
        self._softplus_beta_arrival = float(getattr(
            self.cfg, 'arrival_softplus_beta', self._softplus_beta_reward))
        self._convex_action_l2 = float(getattr(
            self.cfg, 'mpc_convex_action_l2', 0.0))
        self._convex_num_iters = int(getattr(self.cfg, 'mpc_convex_num_iters', 64))
        self._convex_step_size = float(getattr(self.cfg, 'mpc_convex_step_size', 0.03))
        self._convex_adam_beta1 = float(getattr(self.cfg, 'mpc_convex_adam_beta1', 0.9))
        self._convex_adam_beta2 = float(getattr(self.cfg, 'mpc_convex_adam_beta2', 0.999))
        self._convex_adam_eps = float(getattr(self.cfg, 'mpc_convex_adam_eps', 1e-8))
        self._mpc_objective_convex = False

        a_high = np.asarray(
            getattr(self.cfg, 'a_bound_high', np.ones(nU)), dtype=np.float32)
        a_low = np.asarray(
            getattr(self.cfg, 'a_bound_low', -np.ones(nU)), dtype=np.float32)
        self._a_high_np = a_high
        self._a_low_np = a_low

        H = self.horizon
        CH = self._control_horizon
        discount = 1
        beta_reward = self._softplus_beta_reward
        beta_arrival = self._softplus_beta_arrival
        action_l2 = self._convex_action_l2
        num_iters = self._convex_num_iters
        step_size = self._convex_step_size
        adam_beta1 = self._convex_adam_beta1
        adam_beta2 = self._convex_adam_beta2
        adam_eps = self._convex_adam_eps

        def _softplus(x, beta):
            return jax.nn.softplus(beta * x) / beta

        def _objective(U, z0, A_seq, B_seq,
                       reward_w1, reward_w2, reward_w3, reward_b1,
                       arrival_w1, arrival_w2, arrival_w3, arrival_w4,
                       arrival_b1, arrival_b3):
            z = z0
            cost = jnp.array(0.0, dtype=U.dtype)
            for t in range(H):
                k_u = min(t, CH - 1)
                u = U[k_u]
                z = A_seq[t] @ z + B_seq[t] @ u
                preact = reward_w2[t] @ z + reward_w3[t] @ u + reward_b1[t]
                cost = cost - (discount ** t) * jnp.sum(
                    reward_w1[t] * _softplus(preact, beta_reward))

            u_terminal = U[CH - 1]
            arrival_preact1 = (
                jnp.einsum('ekd,d->ek', arrival_w2, z)
                + jnp.einsum('eka,a->ek', arrival_w3, u_terminal)
                + arrival_b1
            )
            arrival_hidden1 = _softplus(arrival_preact1, beta_arrival)
            arrival_preact2 = (
                jnp.einsum('elk,ek->el', arrival_w4, arrival_hidden1)
                + arrival_b3
            )
            arrival_value = jnp.sum(
                arrival_w1 * _softplus(arrival_preact2, beta_arrival),
                axis=-1,
            )
            cost = cost - (discount ** H) * jnp.mean(arrival_value)
            # if action_l2 > 0.0:
            #     cost = cost + action_l2 * jnp.sum(U * U)
            return cost

        value_and_grad = jax.value_and_grad(_objective)

        def _solve(z0, A_seq, B_seq,
                   reward_w1, reward_w2, reward_w3, reward_b1,
                   arrival_w1, arrival_w2, arrival_w3, arrival_w4,
                   arrival_b1, arrival_b3,
                   U_init, a_low, a_high):
            U = jnp.clip(U_init, a_low, a_high)
            m = jnp.zeros_like(U)
            v = jnp.zeros_like(U)

            def body(i, state):
                U, m, v = state
                _value, grad = value_and_grad(
                    U, z0, A_seq, B_seq,
                    reward_w1, reward_w2, reward_w3, reward_b1,
                    arrival_w1, arrival_w2, arrival_w3, arrival_w4,
                    arrival_b1, arrival_b3,
                )
                grad = jnp.nan_to_num(grad, nan=0.0, posinf=1e6, neginf=-1e6)
                m = adam_beta1 * m + (1.0 - adam_beta1) * grad
                v = adam_beta2 * v + (1.0 - adam_beta2) * (grad * grad)
                step = i + 1
                m_hat = m / (1.0 - adam_beta1 ** step)
                v_hat = v / (1.0 - adam_beta2 ** step)
                U = jnp.clip(U - step_size * m_hat / (jnp.sqrt(v_hat) + adam_eps),
                             a_low, a_high)
                return U, m, v

            U, _m, _v = jax.lax.fori_loop(0, num_iters, body, (U, m, v))
            final_value = _objective(
                U, z0, A_seq, B_seq,
                reward_w1, reward_w2, reward_w3, reward_b1,
                arrival_w1, arrival_w2, arrival_w3, arrival_w4,
                arrival_b1, arrival_b3,
            )
            converged = jnp.isfinite(final_value) & jnp.all(jnp.isfinite(U))
            return U, converged

        self._jax_solve_mpc = jax.jit(_solve)

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------
    def _reset_convex_diagnostics(self):
        self.convex_solver_attempts = 0
        self.convex_solver_failures = 0
        self.convex_solver_nonconverged = 0
        self.convex_solver_exceptions = 0
        self.convex_solver_input_nonfinite = 0
        self.convex_solver_solution_nonfinite = 0
        self.convex_nonfinite_inputs = {
            name: 0 for name in (
                'z', 'A', 'B', 'reward_w1', 'reward_w2',
                'reward_w3', 'reward_b1', 'reward_b2',
                'arrival_w1', 'arrival_w2', 'arrival_w3',
                'arrival_w4', 'arrival_b1', 'arrival_b3',
                'arrival_b2', 'U_init',
            )
        }
        self.convex_last_failure_reason = 'none'
        self.convex_last_exception_type = 'none'
        self.convex_last_exception_message = ''
        self.convex_exception_samples = []

    def _add_convex_exception_sample(self, sample):
        sample = str(sample)[:512]
        if sample and sample not in self.convex_exception_samples:
            self.convex_exception_samples.append(sample)
            self.convex_exception_samples = self.convex_exception_samples[-5:]

    def _record_convex_failure(self, reason, detail=None):
        self.convex_solver_failures += 1
        self.convex_last_failure_reason = reason
        if reason == 'nonconverged':
            self.convex_solver_nonconverged += 1
        elif reason == 'exception':
            self.convex_solver_exceptions += 1
            self.convex_last_exception_type = type(detail).__name__
            self.convex_last_exception_message = str(detail)[:512]
            self._add_convex_exception_sample(
                f'{self.convex_last_exception_type}: {self.convex_last_exception_message}')
        elif reason == 'input_nonfinite':
            self.convex_solver_input_nonfinite += 1
            self.convex_last_exception_type = 'input_nonfinite'
            self.convex_last_exception_message = str(detail)
            self._add_convex_exception_sample(f'input_nonfinite: {detail}')
        elif reason == 'solution_nonfinite':
            self.convex_solver_solution_nonfinite += 1
            self.convex_last_exception_type = 'solution_nonfinite'
            self.convex_last_exception_message = ''
            self._add_convex_exception_sample('solution_nonfinite')

    def reset_for_control(self):
        self.state_history = []
        self.action_history = []
        self._state_history_tensors = []
        self._action_history_tensors = []
        self._reset_convex_diagnostics()

    def get_control_metrics(self):
        successes = self.convex_solver_attempts - self.convex_solver_failures
        failure_rate = (self.convex_solver_failures / self.convex_solver_attempts
                        if self.convex_solver_attempts else 0.0)
        metrics = {
            'convex_solver_attempts': int(self.convex_solver_attempts),
            'convex_solver_successes': int(successes),
            'convex_solver_failures': int(self.convex_solver_failures),
            'convex_solver_failure_rate': float(failure_rate),
            'convex_solver_nonconverged': int(self.convex_solver_nonconverged),
            'convex_solver_exceptions': int(self.convex_solver_exceptions),
            'convex_solver_input_nonfinite': int(self.convex_solver_input_nonfinite),
            'convex_solver_solution_nonfinite': int(self.convex_solver_solution_nonfinite),
        }
        metrics.update({
            f'convex_nonfinite_{name}': int(count)
            for name, count in self.convex_nonfinite_inputs.items()
        })
        metrics.update({
            'convex_last_failure_reason': self.convex_last_failure_reason,
            'convex_last_exception_type': self.convex_last_exception_type,
            'convex_last_exception_message': self.convex_last_exception_message,
            'convex_exception_samples': ' | '.join(self.convex_exception_samples),
            'convex_jax_cuda_disabled': bool(self._jax_mpc_cuda_disabled),
            'convex_jax_cuda_disable_reason': self._jax_mpc_cuda_disable_reason,
            'mpc_objective_convex': bool(self._mpc_objective_convex),
        })
        return metrics


    # ------------------------------------------------------------------
    # Policy update (mirror SSMRL.update_pi)
    # ------------------------------------------------------------------
    def _weighted_mean(self, per_sample_loss, sample_weight):
        sample_weight = sample_weight.to(per_sample_loss.device)
        while sample_weight.dim() < per_sample_loss.dim():
            sample_weight = sample_weight.unsqueeze(-1)
        return (per_sample_loss * sample_weight).mean()

    def update_pi(self, zs, sample_weight=None):
        """
        Update policy from detached encoded observations.

        The gradient path is:
            obs0 -> encode(obs0).detach() -> pi(z) -> arrival_Q_value(z, action)
        so policy parameters receive gradients through the action, while the
        actor update does not backpropagate into the encoder.

        Args:
            obs0:       [batch, state_dim]           raw observation for critic input
        Returns:
            pi_loss (float)
        """
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_critic_grad(False)

        H = zs.size(0)
        B = zs.size(1)

        z0 = zs.detach().view(-1, self.latent_dim)
        action, log_prob = self.model.pi(z0, return_log_prob=True)  # [B, act_dim], [B, 1]

        # Arrival Q at (z0, action) - critic grad frozen, actor grad flows via action.
        val = self.model.arrival_Q_value(
            z0, action, z0, target=False, return_type='avg')  # [B, 1]

        self.scale.update(val.mean(0))
        val = self.scale(val)

        # SAC loss: maximise (Q - alpha * log_pi)
        pi_loss_per_sample = (self.entropy_coef * log_prob - val).squeeze(-1)
        pi_loss_per_sample = pi_loss_per_sample.view(H, B, 1)
        pi_loss_per_sample = pi_loss_per_sample.mean(dim=0)

        if sample_weight is None:
            pi_loss = pi_loss_per_sample.mean()
        else:
            pi_loss = self._weighted_mean(pi_loss_per_sample, sample_weight)
        # if not torch.isfinite(pi_loss):
        #     self.pi_optim.zero_grad(set_to_none=True)
        #     self.model.track_critic_grad(True)
        #     return 0.0

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(),
            self.grad_clip_norm
        )
        self.pi_optim.step()
        self.model.track_critic_grad(True)

        return pi_loss.item()

    # ------------------------------------------------------------------
    # Main update (mirror SSMRL.update)
    # ------------------------------------------------------------------
    def _skipped_update_stats(self, nonfinite_batch=False, nonfinite_loss=False):
        self.model.eval()
        return {
            'consistency_loss': 0.0,
            'reward_loss': 0.0,
            'value_loss': 0.0,
            'q_loss': 0.0,
            'arrival_q_loss': 0.0,
            'pi_loss': 0.0,
            'total_loss': 0.0,
            'grad_norm': 0.0,
            'pi_scale': float(self.scale.value),
            'skipped_update': 1.0,
            'nonfinite_batch': float(nonfinite_batch),
            'nonfinite_loss': float(nonfinite_loss),
        }

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
        try:
            sample = buffer.sample(return_weights=True)
        except TypeError:
            sample = buffer.sample()
        if len(sample) == 5:
            obs, action, reward, _, sample_weight = sample
        else:
            obs, action, reward, _ = sample
            sample_weight = torch.ones(obs.shape[1], 1, device=self.device)
        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        sample_weight = sample_weight.to(self.device).view(-1)
        # if not (torch.isfinite(obs).all() and torch.isfinite(action).all()
        #         and torch.isfinite(reward).all()
        #         and torch.isfinite(sample_weight).all()):
        #     return self._skipped_update_stats(nonfinite_batch=True)

        H = self.horizon  # horizon

        # ---- Compute consistency targets without letting the encoder move both
        # sides of the prediction target.

        next_mean = self.model.encode(obs[1:])  # [H, B, D]

        # ---- Prepare for update ----
        self.model_optim.zero_grad(set_to_none=True)
        self.model.train()

        B = obs.shape[1]
        D = self.latent_dim

        # ---- Encode first obs ----
        z = self.model.encode(obs[self.history_horizon])  # [B, D]

        # ---- Build context for transformer ----
        ctx_state = obs[:self.history_horizon]
        ctx_action = action[:self.history_horizon]
        ## rearrange the dimension from  [history_horizon, batch, state_dim] to  [batch, history_horizon, state_dim]
        ctx_state = ctx_state.permute(1, 0, 2)
        ctx_action = ctx_action.permute(1, 0, 2)
        (A_seq, B_seq,
         reward_w1_seq, reward_w2_seq, reward_w3_seq,
         reward_b1_seq, reward_b2_seq,
         encoder_in) = self.model.encode_context(
            ctx_state, ctx_action, obs[self.history_horizon]
        )

        # ---- Latent rollout ----
        zs = torch.empty(H + 1, B, D, device=self.device)
        zs[0] = z
        consistency_loss = torch.tensor(0.0, device=self.device)
        for t in range(H):
            z = self.model.next(z, action[t+self.history_horizon], A_seq[:, t], B_seq[:, t])
            consistency_step = F.mse_loss(
                z, next_mean[t+self.history_horizon], reduction='none'
            ).mean(dim=-1)
            consistency_loss += self._weighted_mean(
                consistency_step, sample_weight
            ) * (self.rho ** t)
            zs[t + 1] = z

        # ---- Reward predictions ----
        reward_loss = torch.tensor(0.0, device=self.device)
        for t in range(H):
            r_pred = self.model.reward(
                zs[t+1], action[t+self.history_horizon],
                reward_w1_seq[:, t], reward_w2_seq[:, t],
                reward_w3_seq[:, t], reward_b1_seq[:, t],
                reward_b2_seq[:, t],
            )
            reward_step = F.smooth_l1_loss(
                r_pred, reward[t+self.history_horizon], reduction='none'
            ).mean(dim=-1)
            reward_loss += self._weighted_mean(
                reward_step, sample_weight
            ) * (self.rho ** t)

        # ---- Value loss (arrival Q is the only Q-function) ----
        q_loss = torch.tensor(0.0, device=self.device)
        arrival_q_loss = torch.tensor(0.0, device=self.device)
        encoder_in_target = encoder_in.detach()
        encoded_zs = self.model.encode(obs)
        for t in range(H+self.history_horizon):
            z_for_q = encoded_zs[t]
            a_for_q = action[t]
            with torch.no_grad():
                z_next = encoded_zs[t + 1].detach()
                a_next = self.model.pi(z_next, deterministic=True)
                q_next = self.model.arrival_Q_value(
                    z_next, a_next, encoder_in_target,
                    target=True, return_type='min')
                q_target = reward[t] + self.discount * q_next

            arrival_q_pred = self.model.arrival_Q_value(
                z_for_q, a_for_q, encoder_in, target=False, return_type='all')
            arrival_q_step = F.smooth_l1_loss(
                arrival_q_pred, q_target.unsqueeze(0).expand_as(arrival_q_pred),
                reduction='none'
            ).mean(dim=(0, 2))
            arrival_q_loss += self._weighted_mean(
                arrival_q_step, sample_weight
            ) * (self.rho ** t)
        # Normalise
        consistency_loss = consistency_loss / H
        reward_loss = reward_loss / H
        arrival_q_loss = arrival_q_loss / (H+self.history_horizon)
        value_loss = arrival_q_loss

        total_loss = (
            self.consistency_coef * consistency_loss
            + self.reward_coef * reward_loss
            + self.value_coef * value_loss
        )
        # if not torch.isfinite(total_loss):
        #     self.model_optim.zero_grad(set_to_none=True)
        #     return self._skipped_update_stats(nonfinite_loss=True)

        # ---- Backward & step (world model) ----
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_norm
        )
        self.model_optim.step()

        # ---- Update policy from detached latent states; critic still uses raw observations ----
        pi_loss = self.update_pi(
            encoded_zs, sample_weight)

        # ---- Soft update targets ----
        self.model.soft_update_targets()

        # ---- Return diagnostics ----
        self.model.eval()
        return {
            'consistency_loss': float(consistency_loss.item()),
            'reward_loss': float(reward_loss.item()),
            'value_loss': float(value_loss.item()),
            'q_loss': float(q_loss.item()),
            'arrival_q_loss': float(arrival_q_loss.item()),
            'pi_loss': pi_loss,
            'total_loss': float(total_loss.item()),
            'grad_norm': float(grad_norm),
            'pi_scale': float(self.scale.value),
            'sample_weight_mean': float(sample_weight.mean().item()),
            'sample_weight_min': float(sample_weight.min().item()),
            'sample_weight_max': float(sample_weight.max().item()),
            'skipped_update': 0.0,
            'nonfinite_batch': 0.0,
            'nonfinite_loss': 0.0,
        }

    # ------------------------------------------------------------------
    # Reward prediction visualization
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_reward_predictions(self, buffer, num_samples=1):
        """
        Compute predicted vs actual rewards for visualization.

        Args:
            buffer: replay buffer
            num_samples: number of samples to compute predictions for

        Returns:
            dict with 'actual_rewards' and 'predicted_rewards' arrays (numpy)
        """
        self.model.eval()

        actual_rewards_list = []
        predicted_rewards_list = []

        for _ in range(num_samples):
            # Sample from buffer
            sample = buffer.sample()
            obs, action, reward, _ = sample
            obs = obs.to(self.device)
            action = action.to(self.device)
            reward = reward.to(self.device)

            H = self.horizon
            B = obs.shape[1]

            # Build context
            ctx_state = obs[:self.history_horizon].permute(1, 0, 2)
            ctx_action = action[:self.history_horizon].permute(1, 0, 2)
            (A_seq, B_seq,
             reward_w1_seq, reward_w2_seq, reward_w3_seq,
             reward_b1_seq, reward_b2_seq,
             encoder_in) = self.model.encode_context(
                ctx_state, ctx_action, obs[self.history_horizon]
            )

            # Get initial latent state
            z = self.model.encode(obs[self.history_horizon])

            # Rollout and predict rewards
            predicted_rewards = []
            for t in range(H):
                z = self.model.next(z, action[t+self.history_horizon], A_seq[:, t], B_seq[:, t])
                r_pred = self.model.reward(
                    z, action[t+self.history_horizon],
                    reward_w1_seq[:, t], reward_w2_seq[:, t],
                    reward_w3_seq[:, t], reward_b1_seq[:, t],
                    reward_b2_seq[:, t],
                )
                predicted_rewards.append(r_pred.detach().cpu().numpy())

            # Stack predictions
            predicted_rewards = np.concatenate(predicted_rewards, axis=0)  # (H*B, 1)
            actual_rewards = reward[self.history_horizon:self.history_horizon+H].detach().cpu().numpy()  # (H, B, 1)
            actual_rewards = actual_rewards.reshape(-1, 1)

            actual_rewards_list.append(actual_rewards)
            predicted_rewards_list.append(predicted_rewards)

        # Concatenate all samples
        actual_rewards_all = np.concatenate(actual_rewards_list, axis=0)
        predicted_rewards_all = np.concatenate(predicted_rewards_list, axis=0)

        return {
            'actual_rewards': actual_rewards_all.flatten(),
            'predicted_rewards': predicted_rewards_all.flatten(),
        }
