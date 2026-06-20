"""
SSM Agent v19 - transformer-conditioned SSM-RL with JAX/qpax MPC.

This agent wraps ``SSMWorldModel`` and provides both model learning and
inference-time control. During inference, recent state/action history is used
by ``encode_context`` to produce local linear SSM parameters, reconstruction
matrix ``C``, reward quadratics, and critic context. Until enough history is
available, or if the QP solver fails, actions come from the learned policy.

MPC controller:
  * The finite-horizon control problem is assembled in JAX and JIT-compiled.
  * ``qpax`` solves the dense QP over the stacked action sequence only.
  * Diagonal latent dynamics are analytically unrolled, eliminating equality
    dynamics constraints:
      z_{t+1} = A^{t+1} * z_0 + T_u[t] @ U_flat
  * Per-step action bounds are encoded as box inequalities ``G U <= h``.
  * The terminal Q cost uses the critic ensemble mean plus
    ``cfg.ucb_beta`` times the ensemble standard deviation for state and action
    quadratic terms.
  * Training-time exploration perturbs encoded SSM/reward parameters by
    relative Gaussian noise (``cfg.param_noise_std``) and adds policy-standard-
    deviation action noise after planning.
  * PyTorch tensors are passed to JAX through DLPack before the JIT solve.

QP form passed to ``qpax``:
  min   0.5 * U^T Q_qp U + c_qp^T U
  s.t.  G U <= h

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
import qpax  # pip install qpax


from ssmrl.common.ssm_world_model_v10 import SSMWorldModel
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
        self._jax_has_cuda = any(d.platform in ('cuda', 'gpu') for d in jax.devices())

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
            {'params': self.model._R_net.parameters()},
            {'params': self.model._r_net.parameters()},
            {'params': [self.model._b]},
            {'params': self.model._q_func.parameters()},
        ], lr=lr)


        # Group 2 - raw-observation policy trunk and action heads
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

        # History buffers and episode-scoped control diagnostics
        self.state_history = []
        self.action_history = []
        self._reset_qpax_diagnostics()

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
            # Not enough history for transformer - use policy net on raw obs
            u_norm = self.model.pi(obs_t, deterministic=eval_mode)[0].cpu().numpy()
        else:
            # Attempt JAX/qpax planning
            u_norm = self._plan_jax(z, x0, eval_mode)
            # u_norm = self.model.pi(obs_t, deterministic=eval_mode)[0].cpu().numpy()
        # Update history
        self.action_history.append(u_norm.copy())
        self.state_history.append(x0.copy())

        ## convert to float32
        u = np.asarray(u_norm, dtype=np.float32)
        return u

    @torch.no_grad()
    def _perturb_encoded_params(self, A_diag, B, Q, q, R, r_vec):
        """
        Add relative Gaussian noise to encoded SSM parameters for exploration.
        Full quadratic matrices are perturbed by congruence transforms,
        preserving PSD structure.
        """
        std = float(getattr(self.cfg, 'param_noise_std', 0.02))

        def _noisy(t: torch.Tensor) -> torch.Tensor:
            scale = t.abs().mean() + 1e-6
            return t + std * scale * torch.randn_like(t)

        def _noisy_psd(M: torch.Tensor) -> torch.Tensor:
            dim = M.shape[-1]
            eye = torch.eye(dim, device=M.device, dtype=M.dtype)
            S = eye.expand(*M.shape[:-2], dim, dim) + std * torch.randn_like(M)
            return S @ M @ S.transpose(-1, -2)

        return _noisy(A_diag), _noisy(B), _noisy_psd(Q), _noisy(q), _noisy_psd(R), _noisy(r_vec)

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

        A_diag_seq, B, Q_seq, q_seq, R_seq, r_seq, encoder_in = self.model.encode_context(
            state_t, action_t, obs_t)

        # # ---- Parameter noise for exploration ----
        if not eval_mode:
            std = float(getattr(self.cfg, 'param_noise_std', 0.02))
            def _noisy(t):
                scale = t.abs().mean() + 1e-6
                return t + std * scale * torch.randn_like(t)
            def _noisy_psd(M):
                dim = M.shape[-1]
                eye = torch.eye(dim, device=M.device, dtype=M.dtype)
                S = eye.expand(*M.shape[:-2], dim, dim) + std * torch.randn_like(M)
                return S @ M @ S.transpose(-1, -2)
            A_diag_seq = _noisy(A_diag_seq)
            B = _noisy(B)
            Q_seq = _noisy_psd(Q_seq)
            q_seq = _noisy(q_seq)
            R_seq = _noisy_psd(R_seq)
            r_seq = _noisy(r_seq)

        # Extract single-sample tensors
        A_diag_seq_t = A_diag_seq[0]    # (H, D)
        B_t          = B[0]              # (D, nU)
        Q_seq_t      = Q_seq[0]          # (H, D, D)
        q_seq_t      = q_seq[0]          # (H, D)
        R_seq_t      = R_seq[0]          # (H, nU, nU)
        r_seq_t      = r_seq[0]          # (H, nU)
        z_t          = z[0]              # (D,)


        self.qpax_solver_attempts += 1
        input_tensors = {
            'z': z_t,
            'A': A_diag_seq_t,
            'B': B_t,
            'Q': Q_seq_t,
            'q': q_seq_t,
            'R': R_seq_t,
            'r': r_seq_t,
        }
        bad_inputs = [name for name, tensor in input_tensors.items()
                      if not torch.isfinite(tensor).all().item()]
        if bad_inputs:
            self._record_qpax_failure('input_nonfinite', ','.join(bad_inputs))
            for name in bad_inputs:
                self.qpax_nonfinite_inputs[name] += 1
            return self.model.pi(obs_t, deterministic=eval_mode)[0].cpu().numpy()

        # ---- PyTorch -> JAX. Use DLPack when both libraries share a backend;
        # otherwise copy through CPU so CUDA Torch + CPU-only JAX still works.
        def to_jax(t: torch.Tensor):
            t = t.detach().contiguous()
            if t.is_cuda and not self._jax_has_cuda:
                return jnp.asarray(t.cpu().numpy())
            try:
                return jax.dlpack.from_dlpack(t)
            except TypeError:
                try:
                    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(t))
                except RuntimeError as exc:
                    if t.is_cuda and 'Unknown backend cuda' in str(exc):
                        return jnp.asarray(t.cpu().numpy())
                    raise
            except RuntimeError as exc:
                if t.is_cuda and 'Unknown backend cuda' in str(exc):
                    return jnp.asarray(t.cpu().numpy())
                raise

        try:
            U_sol, converged = self._jax_solve_mpc(
                to_jax(z_t),
                to_jax(A_diag_seq_t),
                to_jax(B_t),
                to_jax(Q_seq_t),
                to_jax(q_seq_t),
                to_jax(R_seq_t),
                to_jax(r_seq_t),
                self._a_low_jax,
                self._a_high_jax,
            )
            if not bool(converged):
                self._record_qpax_failure('nonconverged')
                return self.model.pi(obs_t, deterministic=eval_mode)[0].cpu().numpy()
            u = np.asarray(U_sol[0], dtype=np.float32)   # first control step
            if not np.isfinite(u).all():
                self._record_qpax_failure('solution_nonfinite')
                return self.model.pi(obs_t, deterministic=eval_mode)[0].cpu().numpy()
        except Exception as exc:
            self._record_qpax_failure('exception', exc)
            return self.model.pi(obs_t, deterministic=eval_mode)[0].cpu().numpy()

        # if not eval_mode:
        #     std = self.model.get_pi_std(obs_t)[0]
        #     epsilon = (std * torch.randn(self.act_dim, device=std.device)
        #                ).detach().cpu().numpy()
        #     u = u + epsilon

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

        Stage reward:  r(z_t, u_t) = -(z_t^T Q_r z_t + q_r^T z_t
                                        + u_t^T R_r u_t + r_r^T u_t + b)
        Terminal cost (Q-value): Q(z_H, u_H) UCB over ensemble.
        """
        D   = self.latent_dim
        nU  = self.act_dim
        H   = self.horizon
        CH  = H
        discount = self.discount
        discount = 1.0
        u_penalty = float(getattr(self.cfg, 'u_penalty', 0.0))

        self._control_horizon = CH
        n = CH * nU   # total QP decision-variable size

        def _build_and_solve(z0, A_diag_seq, B, Q_seq, q_seq,
                             R_seq, r_seq,
                             a_low, a_high):
            """
            Args
            ----
            z0          : (D,)       initial latent state
            A_diag_seq  : (H, D)     per-step diagonal of dynamics matrix A
            B           : (D, nU)    input matrix
            Q_seq       : (H, D, D)  per-step PSD stage state-cost matrix
            q_seq       : (H, D)     per-step linear state-cost coefficient
            R_seq       : (H, nU, nU) per-step PSD stage action-cost matrix
            r_seq       : (H, nU)    per-step linear action-cost coefficient
            a_low       : (nU,)      per-dim action lower bound
            a_high      : (nU,)      per-dim action upper bound

            Returns
            -------
            U         : (CH, nU)  optimal control sequence
            converged : bool
            """
            # ----------------------------------------------------------
            # 1. State trajectory matrices (per-step A_diag)
            # ----------------------------------------------------------
            f_list   = []
            T_u_list = []

            for t in range(H):
                # f_t = A_0 * A_1 * ... * A_t * z0  (element-wise cumulative product)
                A_cum = jnp.ones(D)
                for s in range(t + 1):
                    A_cum = A_cum * A_diag_seq[s]
                f_list.append(A_cum * z0)

                T_u_t = jnp.zeros((D, n))
                for k in range(CH):
                    s_k = k * nU
                    e_k = (k + 1) * nU
                    if k < CH - 1:
                        if k <= t:
                            # coeff = A_{k+1} * ... * A_t * B  (element-wise)
                            A_prod = jnp.ones(D)
                            for s in range(k + 1, t + 1):
                                A_prod = A_prod * A_diag_seq[s]
                            coeff = A_prod[:, None] * B   # (D, nU)
                            T_u_t = T_u_t.at[:, s_k:e_k].set(coeff)
                    else:
                        # Last ZOH block
                        accum = jnp.zeros((D, nU))
                        for j in range(CH - 1, t + 1):
                            A_prod = jnp.ones(D)
                            for s in range(j + 1, t + 1):
                                A_prod = A_prod * A_diag_seq[s]
                            accum = accum + A_prod[:, None] * B
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

                # Stage state cost: z^T Q[t] z + q[t]^T z  (per-step Q and q)
                Q_t = Q_seq[t]                   # (D, D)
                q_t = q_seq[t]                   # (D,)
                QTu = Q_t @ Tu                   # (D, n)
                Q_qp = Q_qp + (discount ** t) * 2.0 * (Tu.T @ QTu)
                c_qp = c_qp + (discount ** t) * (2.0 * (QTu.T @ f) + Tu.T @ q_t)

                # Stage action cost: u_t^T R[t] u_t + r[t]^T u_t  (per-step, block k_u)
                Q_qp = Q_qp.at[s_u:e_u, s_u:e_u].add(
                    (discount ** t) * 2.0 * R_seq[t])
                c_qp = c_qp.at[s_u:e_u].add(
                    (discount ** t) * r_seq[t])

                # # Stage control regularisation: u_penalty * ||u||^2  (block k_u)
                # Q_qp = Q_qp.at[s_u:e_u, s_u:e_u].add(
                #     (discount ** t) * 2.0 * u_penalty * jnp.eye(nU))

            # Note: Terminal cost Q(z_H, u_H) is now handled by a separate MLP
            # and not included in the QP. The MPC only optimizes the stage costs.

            # # Ridge for numerical stability
            # Q_qp = Q_qp + 1e-6 * jnp.eye(n)

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
    def _reset_qpax_diagnostics(self):
        self.qpax_solver_attempts = 0
        self.qpax_solver_failures = 0
        self.qpax_solver_nonconverged = 0
        self.qpax_solver_exceptions = 0
        self.qpax_solver_input_nonfinite = 0
        self.qpax_solver_solution_nonfinite = 0
        self.qpax_nonfinite_inputs = {name: 0 for name in ('z', 'A', 'B', 'Q', 'q', 'R', 'r')}
        self.qpax_last_failure_reason = 'none'
        self.qpax_last_exception_type = 'none'
        self.qpax_last_exception_message = ''
        self.qpax_exception_samples = []

    def _add_qpax_exception_sample(self, sample):
        sample = str(sample)[:512]
        if sample and sample not in self.qpax_exception_samples:
            self.qpax_exception_samples.append(sample)
            self.qpax_exception_samples = self.qpax_exception_samples[-5:]

    def _record_qpax_failure(self, reason, detail=None):
        self.qpax_solver_failures += 1
        self.qpax_last_failure_reason = reason
        if reason == 'nonconverged':
            self.qpax_solver_nonconverged += 1
        elif reason == 'exception':
            self.qpax_solver_exceptions += 1
            self.qpax_last_exception_type = type(detail).__name__
            self.qpax_last_exception_message = str(detail)[:512]
            self._add_qpax_exception_sample(
                f'{self.qpax_last_exception_type}: {self.qpax_last_exception_message}')
        elif reason == 'input_nonfinite':
            self.qpax_solver_input_nonfinite += 1
            self.qpax_last_exception_type = 'input_nonfinite'
            self.qpax_last_exception_message = str(detail)
            self._add_qpax_exception_sample(f'input_nonfinite: {detail}')
        elif reason == 'solution_nonfinite':
            self.qpax_solver_solution_nonfinite += 1
            self.qpax_last_exception_type = 'solution_nonfinite'
            self.qpax_last_exception_message = ''
            self._add_qpax_exception_sample('solution_nonfinite')

    def reset_for_control(self):
        self.state_history = []
        self.action_history = []
        self._reset_qpax_diagnostics()

    def get_control_metrics(self):
        successes = self.qpax_solver_attempts - self.qpax_solver_failures
        failure_rate = (self.qpax_solver_failures / self.qpax_solver_attempts
                        if self.qpax_solver_attempts else 0.0)
        metrics = {
            'qpax_solver_attempts': int(self.qpax_solver_attempts),
            'qpax_solver_successes': int(successes),
            'qpax_solver_failures': int(self.qpax_solver_failures),
            'qpax_solver_failure_rate': float(failure_rate),
            'qpax_solver_nonconverged': int(self.qpax_solver_nonconverged),
            'qpax_solver_exceptions': int(self.qpax_solver_exceptions),
            'qpax_solver_input_nonfinite': int(self.qpax_solver_input_nonfinite),
            'qpax_solver_solution_nonfinite': int(self.qpax_solver_solution_nonfinite),
        }
        metrics.update({
            f'qpax_nonfinite_{name}': int(count)
            for name, count in self.qpax_nonfinite_inputs.items()
        })
        metrics.update({
            'qpax_last_failure_reason': self.qpax_last_failure_reason,
            'qpax_last_exception_type': self.qpax_last_exception_type,
            'qpax_last_exception_message': self.qpax_last_exception_message,
            'qpax_exception_samples': ' | '.join(self.qpax_exception_samples),
        })
        return metrics

    def get_control_diagnostics(self):
        return self.get_control_metrics()

    def store_cached_control_info(self):
        self._cached = copy.deepcopy({
            'state_history': self.state_history,
            'action_history': self.action_history,
            'qpax_diagnostics': self.get_control_diagnostics(),
        })

    def restore_control_info(self):
        if hasattr(self, '_cached'):
            self.state_history = self._cached['state_history']
            self.action_history = self._cached['action_history']
            self._reset_qpax_diagnostics()
            diagnostics = self._cached.get('qpax_diagnostics')
            if diagnostics is None:
                self.qpax_solver_failures = self._cached.get('qpax_solver_failures', 0)
                self.qpax_solver_attempts = self.qpax_solver_failures
            else:
                self.qpax_solver_attempts = diagnostics.get('qpax_solver_attempts', 0)
                self.qpax_solver_failures = diagnostics.get('qpax_solver_failures', 0)
                self.qpax_solver_nonconverged = diagnostics.get('qpax_solver_nonconverged', 0)
                self.qpax_solver_exceptions = diagnostics.get('qpax_solver_exceptions', 0)
                self.qpax_solver_input_nonfinite = diagnostics.get('qpax_solver_input_nonfinite', 0)
                self.qpax_solver_solution_nonfinite = diagnostics.get('qpax_solver_solution_nonfinite', 0)
                for name in self.qpax_nonfinite_inputs:
                    self.qpax_nonfinite_inputs[name] = diagnostics.get(f'qpax_nonfinite_{name}', 0)
                self.qpax_last_failure_reason = diagnostics.get('qpax_last_failure_reason', 'none')
                self.qpax_last_exception_type = diagnostics.get('qpax_last_exception_type', 'none')
                self.qpax_last_exception_message = diagnostics.get('qpax_last_exception_message', '')
                samples = diagnostics.get('qpax_exception_samples', '')
                self.qpax_exception_samples = [s for s in samples.split(' | ') if s]
        else:
            print('No cached control info found.')

    # ------------------------------------------------------------------
    # Policy update (mirror SSMRL.update_pi)
    # ------------------------------------------------------------------
    def update_pi(self, obs0, z0, encoder_in):
        """
        Update policy using raw observations for the actor and latent state for the critic.

        The gradient path is:
            obs0 -> pi(obs0) -> action -> Q_value(z0, action, encoder_in)
        so policy parameters receive gradients through the action without using
        the world-model encoder as the actor input.

        Args:
            obs0:       [batch, state_dim]           raw observation for policy input
            z0:         [batch, latent_dim]          detached latent state for Q input
            encoder_in: [batch, ctx_dim]             detached context for Q input
        Returns:
            pi_loss (float)
        """
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_critic_grad(False)

        # Sample action from policy at raw observation obs0
        action, log_prob = self.model.pi(obs0, return_log_prob=True)  # [B, act_dim], [B, 1]

        # Q value at (z0, action) - critic grad frozen, actor grad flows via action
        val = self.model.Q_value(z0, action, encoder_in, target=False, return_type='min')  # [B, 1]

        self.scale.update(val)
        val = self.scale(val)

        # SAC loss: maximise (Q - alpha * log_pi)
        pi_loss = -(val - self.entropy_coef * log_prob).mean()

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
    def _td_target(self, next_z, next_obs, reward, encoder_in):
        """
        Compute TD target: r + gamma * V_target(next_z).

        Args:
            next_z:     [T, batch, latent_dim]
            next_obs:   [T, batch, state_dim]
            reward:     [T, batch, 1]
            encoder_in: [batch, ctx_dim]
        Returns:
            td_target: [T, batch, 1]
        """
        T, B, _ = next_z.shape
        z_flat = next_z.reshape(T * B, -1)
        obs_flat = next_obs.reshape(T * B, -1)
        enc_in_flat = encoder_in.unsqueeze(0).expand(T, -1, -1).reshape(T * B, -1)
        # Sample next action from target policy for Q(z, a)
        a_next = self.model.pi(obs_flat, target=True, deterministic=True)
        next_val = self.model.Q_value(z_flat, a_next, enc_in_flat, target=True, return_type='min')
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
        obs, action, reward, _ = sample
        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)

        H = self.horizon  # horizon

        # # ---- Compute targets (no grad) ----
        # with torch.no_grad():
        next_mean = self.model.encode(obs[1:])  # [H, B, D]
        #     # Build encoder_in for target computation using history
        #     ctx_state_tgt = obs[:self.history_horizon].permute(1, 0, 2)
        #     ctx_action_tgt = action[:self.history_horizon].permute(1, 0, 2)
        #     _, _, _, _, _, _, encoder_in = self.model.encode_context(
        #         ctx_state_tgt, ctx_action_tgt, obs[self.history_horizon]
        #     )
        #     td_targets = self._td_target(next_mean, reward, encoder_in)

        # ---- Prepare for update ----
        self.model_optim.zero_grad(set_to_none=True)
        self.model.train()

        B = obs.shape[1]
        D = self.latent_dim

        # ---- Encode first obs ----
        z = self.model.encode(obs[self.history_horizon])  # [B, D]
        z_target = self.model.encode(obs[self.history_horizon+ H], target=True)
        z_random = z

        # ---- Build context for transformer ----
        ctx_state = obs[:self.history_horizon]
        ctx_action = action[:self.history_horizon]
        ## rearrange the dimension from  [history_horizon, batch, state_dim] to  [batch, history_horizon, state_dim]
        ctx_state = ctx_state.permute(1, 0, 2)
        ctx_action = ctx_action.permute(1, 0, 2)
        A_diag_seq, B_mat, Q_seq, q_seq, R_seq, r_seq, encoder_in = self.model.encode_context(
            ctx_state, ctx_action, obs[self.history_horizon]
        )

        # ---- Latent rollout ----
        zs = torch.empty(H + 1, B, D, device=self.device)
        z_randoms = torch.empty(H + 1, B, D, device=self.device)
        zs[0] = z
        z_randoms[0] = z_random
        consistency_loss = torch.tensor(0.0, device=self.device)
        for t in range(H):
            z = self.model.next(z, action[t+self.history_horizon], A_diag_seq[:, t, :], B_mat)
            z_random = self.model.next(z_random, action[t+self.history_horizon], A_diag_seq[:, t, :], B_mat)
            consistency_loss += F.mse_loss(z, next_mean[t+self.history_horizon]) * (self.rho ** t)
            zs[t + 1] = z
            z_randoms[t + 1] = z_random

        # ---- Reward predictions ----
        reward_loss = torch.tensor(0.0, device=self.device)
        for t in range(H):
            r_pred = self.model.reward(
                z_randoms[t+1], action[t+self.history_horizon],
                Q_seq[:, t], q_seq[:, t, :], R_seq[:, t], r_seq[:, t, :]
            )
            reward_loss += F.mse_loss(r_pred, reward[t+self.history_horizon]) * (self.rho ** t)

        # ---- Value loss (Q-function) ----
        z_for_q = self.model.encode(obs[self.history_horizon + H-1])
        # Sample target action from raw obs; evaluate it with latent z_target
        with torch.no_grad():
            a_target = self.model.pi(obs[self.history_horizon + H], target=True, deterministic=True)

        q_target_val = reward[self.history_horizon] + self.discount * self.model.Q_value(
            z_target, a_target, encoder_in, target=True
        )
        # Sample action from current policy at z_for_q for Q(z_for_q, a)
        a_for_q = action[self.history_horizon + H-1]
        q_pred = self.model.Q_value(z_for_q, a_for_q, encoder_in, target=False)
        q_loss = F.mse_loss(q_pred, q_target_val.detach())
        # Normalise
        consistency_loss = consistency_loss / H
        reward_loss = reward_loss / H
        value_loss = q_loss

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

        # ---- Update policy from raw observations; critic still uses latent z0 ----
        with torch.no_grad():
            z0_pi = self.model.encode(obs[self.history_horizon])
        pi_loss = self.update_pi(
            obs[self.history_horizon], z0_pi.detach(), encoder_in.detach())

        # ---- Soft update targets ----
        self.model.soft_update_targets()

        # ---- Return diagnostics ----
        self.model.eval()
        return {
            'consistency_loss': float(consistency_loss.item()),
            'reward_loss': float(reward_loss.item()),
            'value_loss': float(value_loss.item()),
            'q_loss': float(q_loss.item()),
            'pi_loss': pi_loss,
            'total_loss': float(total_loss.item()),
            'grad_norm': float(grad_norm),
            'pi_scale': float(self.scale.value),
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
            A_diag_seq, B_mat, Q_seq, q_seq, R_seq, r_seq, encoder_in = self.model.encode_context(
                ctx_state, ctx_action, obs[self.history_horizon]
            )

            # Get initial latent state
            z = self.model.encode(obs[self.history_horizon])

            # Rollout and predict rewards
            predicted_rewards = []
            for t in range(H):
                z = self.model.next(z, action[t+self.history_horizon], A_diag_seq[:, t, :], B_mat)
                r_pred = self.model.reward(
                    z, action[t+self.history_horizon],
                    Q_seq[:, t], q_seq[:, t, :], R_seq[:, t], r_seq[:, t, :]
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
