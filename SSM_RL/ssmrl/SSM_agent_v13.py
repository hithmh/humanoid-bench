            u = _jax_to_numpy(U_sol[0])   # first control step
SSM Agent v13 – JAX + qpax GPU-accelerated MPC.
            # Fallback to policy net
                to_jax(pb_t),
  * Dynamics constraints are analytically eliminated: the full state trajectory
    is expressed as a linear function of the stacked control sequence U,
    reducing the MPC to a single dense QP in U only.
  * The ensemble terminal cost ``max_i V_i(z, u)`` is replaced by its
    ensemble average, which keeps the problem a strict convex QP (no QCQP
    or conic constraints needed).
  * The resulting QP is solved by ``qpax`` (pure-JAX interior-point QP solver),
    JIT-compiled and executed on GPU.
        pb_t = pb_t.squeeze(-1)  # (E,)
    ``jax.dlpack`` when both tensors live on the same CUDA device.
  * ``jax.jit`` compiles the full matrix-build + solve once; subsequent
    calls are fast.
        # Critic params (stay on same device as model)
QP form passed to qpax
  min   ½ U^T Q_qp U + c_qp^T U
  s.t.  G U ≤ h   (per-step box constraints on actions)

State trajectory (A is diagonal):
  z_{t+1} = A^{t+1} z_0  +  T_u[t] @ U_flat
        pb_t      = self.model._pb                          # (E, 1)
        K_diags_t = F.relu(self.model._K_indices)          # (E, nU)
        k_t       = self.model._k                           # (E, nU, 1) or (E, nU)
        # Extract single-sample tensors [D] / [D, nU] / etc.
        obs_t = torch.tensor(x0, dtype=torch.float32, device=self.device).unsqueeze(0)
        state_seq = np.array(self.state_history[-self.history_horizon:],
                             dtype=np.float32)
        action_seq = np.array(self.action_history[-self.history_horizon:],
                              dtype=np.float32)
        state_t = torch.tensor(state_seq, device=self.device).unsqueeze(0)
import qpax  # pip install qpax
        numerical failure.
        self._a_low_jax = jnp.array(a_low)
        self._jax_solve_mpc = jax.jit(_solve)
            (U_final, _), _ = jax.lax.scan(step_fn, (U_init, opt_state),
                                            None, length=n_steps)
            return U_final.reshape(CH, nU)
            def step_fn(carry, _):
                U, opt_state = carry
                _, grads = loss_and_grad(U, z0, A_diag, B, Q_diag, q_vec,
                                         R_diag, r_vec, P_diags, p_mat,
                                         K_diags, k_mat, pb_vec)
                updates, new_opt_state = optimizer.update(grads, opt_state)
                new_U = optax.apply_updates(U, updates)
                if apply_action_constraints:
                    # Box-project each time step's control
                    new_U = jnp.clip(new_U.reshape(CH, nU),
                                     a_low[None, :], a_high[None, :]).reshape(-1)
                return (new_U, new_opt_state), None
        def _solve(z0, A_diag, B, Q_diag, q_vec, R_diag, r_vec,
                   P_diags, p_mat, K_diags, k_mat, pb_vec, a_low, a_high):
            U_init = jnp.zeros(CH * nU)
            opt_state = optimizer.init(U_init)
        # ----------------------------------------------------------------
        # JIT-compiled projected-Adam solve
        # ----------------------------------------------------------------
        optimizer = optax.adam(mpc_lr)
        loss_and_grad = jax.value_and_grad(_objective)
            critic_vals = jax.vmap(single_critic)(P_diags, p_mat, K_diags, k_mat, pb_vec)
            cost = cost + (discount ** H) * jnp.max(critic_vals)
            return cost
            def single_critic(P_d, p_v, K_d, k_v, pb_s):
                return (jnp.dot(P_d * z, z)
                        + jnp.dot(p_v, z)
                        + jnp.dot(K_d * u_term, u_term)
                        + jnp.dot(k_v, u_term)
                        + pb_s)
            # Terminal cost – max over ensemble critics (vmap over E)
            u_term = U[CH - 1]
                u = U[k_u]
                z = A_diag * z + B @ u          # diagonal A → element-wise
                stage = (jnp.dot(Q_diag * z, z)
                         + jnp.dot(q_vec, z)
                         + jnp.dot(R_diag * u, u)
                         + jnp.dot(r_vec, u))
                cost = cost + (discount ** t) * stage
            U = U_flat.reshape(CH, nU)
            z = z0
            cost = jnp.zeros(())
            Args:
                U_flat : (CH * nU,)   – stacked control sequence
                z0     : (D,)         – initial latent state
                A_diag : (D,)         – diagonal of A
                B      : (D, nU)      – input matrix
                Q_diag : (D,)         – diagonal of stage state cost
                q_vec  : (D,)
                R_diag : (nU,)        – diagonal of stage control cost
                r_vec  : (nU,)
                P_diags: (E, D)       – diagonal of terminal state critic per ensemble
                p_mat  : (E, D)
                K_diags: (E, nU)      – diagonal of terminal control critic
                k_mat  : (E, nU)
                pb_vec : (E,)         – terminal critic bias
        def _objective(U_flat, z0, A_diag, B, Q_diag, q_vec,
                       R_diag, r_vec, P_diags, p_mat, K_diags, k_mat, pb_vec):
        # Pure JAX objective (static graph; H, CH, E are compile-time consts)
        n_steps = int(getattr(self.cfg, 'mpc_gd_steps', 200))
        mpc_lr = float(getattr(self.cfg, 'mpc_lr', 1e-2))
        D = self.latent_dim
        nU = self.act_dim
        H = self.horizon
        CH = getattr(self.cfg, 'control_horizon', 5)
        E = self.num_ensembles
        Dynamics:  z_{t+1} = A_diag ⊙ z_t + B u_t   (A is diagonal)
        Objective (all quantities are diagonal matrices passed as vectors):
          J(U) = Σ_{t=0}^{H-1}  γ^t [ z_{t+1}^T diag(Q) z_{t+1}
                                       + q^T z_{t+1}
                                       + u_t^T diag(R) u_t
                                       + r^T u_t ]
               + γ^H  max_i [ z_H^T diag(P_i) z_H + p_i^T z_H
                               + u_{H-1}^T diag(K_i) u_{H-1}
                               + k_i^T u_{H-1} + pb_i ]
        The MPC is solved as an unconstrained differentiable program in the
        stacked control sequence ``U ∈ R^{CH × act_dim}``, with box
        constraints enforced by clipping after each Adam step.
        Build and JIT-compile the JAX MPC solve function.
import optax  # pip install optax
The epigraph reformulation for ``max_i V_i(z, u)`` is handled naturally by
``jnp.max`` inside the differentiable objective – no conic variables needed.
    ``jax.dlpack`` when both tensors live on the same CUDA device
  * ``jax.vmap`` vectorises the ensemble critic evaluation
  * ``jax.jit`` compiles the full solve once; subsequent calls are fast
  * MPC cost is evaluated entirely in JAX (JIT-compiled, runs on GPU/CPU)
  * Projected-Adam gradient-descent loop solved with ``jax.lax.scan`` +
    ``optax.adam`` (no external QP solver dependency)
"""
SSM Agent v13 – JAX-accelerated MPC replaces the previous CVXPY CPU solver.

Architecture is identical to v12 except for the inference-time controller:
  * MPC cost is evaluated entirely in JAX (JIT-compiled, runs on GPU/CPU)
  * Projected-Adam gradient-descent loop solved with ``jax.lax.scan`` +
    ``optax.adam`` (no external QP solver dependency)
  * Zero-copy PyTorch ↔ JAX tensor bridge via ``torch.utils.dlpack`` /
    ``jax.dlpack`` when both tensors live on the same CUDA device
  * ``jax.vmap`` vectorises the ensemble critic evaluation
  * ``jax.jit`` compiles the full solve once; subsequent calls are fast

The epigraph reformulation for ``max_i V_i(z, u)`` is handled naturally by
``jnp.max`` inside the differentiable objective – no conic variables needed.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

# ---- JAX stack (required) -----------------------------------------------
import jax
import jax.numpy as jnp
import optax  # pip install optax

from ssmrl.common.ssm_world_model_v4 import SSMWorldModel
from ssmrl.common.scale import RunningScale


# ---------------------------------------------------------------------------
# Utility: zero-copy bridge between PyTorch and JAX
# ---------------------------------------------------------------------------
def _torch_to_jax(t: torch.Tensor):
    """Transfer a PyTorch tensor to a JAX array (zero-copy via DLPack on CUDA)."""
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(t.contiguous()))


def _jax_to_numpy(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


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
        Build and JIT-compile the qpax MPC solve function.

        Dynamics are analytically eliminated by expressing the full state
        trajectory as a linear map of the stacked control vector U_flat:

            z_{t+1} = A_diag^{t+1} ⊙ z_0   +   T_u[t] @ U_flat
                      \___ free response ___/   \__ forced response __/

        where A is diagonal (element-wise multiply) and T_u[t] ∈ R^{D×n}
        (n = CH×nU) is built by unrolling the dynamics with zero-order hold.
        # ---- Optimizers ----
        Substituting into the MPC cost yields a strict convex QP in U_flat:
        self.model_optim = torch.optim.Adam([
            min   ½ U^T Q_qp U + c_qp^T U
            s.t.  G U ≤ h          (box constraints on each u_t)
            {'params': self.model._r_net.parameters()},
        The ensemble terminal cost ``max_i V_i`` is approximated by the
        ensemble *average*, which keeps the terminal term quadratic so the
        whole problem remains a standard QP solvable by qpax.

        A small ridge (1e-6 I) is added to Q_qp for numerical stability.
            {'params': [self.model._P_indices, self.model._p, self.model._pb,
        D   = self.latent_dim
        nU  = self.act_dim
        H   = self.horizon
        CH  = getattr(self.cfg, 'control_horizon', 5)
            + list(self.model._pi_mean_head.parameters())
        )

        self.model.eval()
        n = CH * nU   # total QP decision-variable size
        self.scale = RunningScale(cfg)

        # JIT-compiled: build QP matrices + call qpax.solve_qp
        # All Python for-loops are over compile-time-constant integers
        # (H, CH) and are fully unrolled by JAX at trace time.
        self.grad_clip_norm = getattr(cfg, 'grad_clip_norm', 20.0)
        def _build_and_solve(z0, A_diag, B, Q_diag, q_vec,
                             R_diag, r_vec, P_diags, p_mat,
                             K_diags, k_mat, a_low, a_high):
        self.reward_coef = getattr(cfg, 'reward_coef', 0.1)
            Args
            ----
            z0      : (D,)      initial latent state
            A_diag  : (D,)      diagonal of dynamics matrix A
            B       : (D, nU)   input matrix
            Q_diag  : (D,)      diagonal of stage state-cost matrix
            q_vec   : (D,)      linear state-cost coefficient
            R_diag  : (nU,)     diagonal of stage control-cost matrix
            r_vec   : (nU,)     linear control-cost coefficient
            P_diags : (E, D)    diagonal of terminal state-critic per ensemble
            p_mat   : (E, D)    linear terminal state-critic coefficient
            K_diags : (E, nU)   diagonal of terminal control-critic per ensemble
            k_mat   : (E, nU)   linear terminal control-critic coefficient
            a_low   : (nU,)     per-dim action lower bound
            a_high  : (nU,)     per-dim action upper bound

            Returns
            -------
            U       : (CH, nU)  optimal control sequence
            converged : bool
        self._build_jax_controller()

            # ----------------------------------------------------------
            # 1. State trajectory matrices (unrolled, compile-time loops)
            # ----------------------------------------------------------
            # f_list[t]   = A_diag^{t+1} * z0         shape (D,)
            # T_u_list[t] = forced-response matrix     shape (D, n)
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
                        # Only input-step j = k maps to u_k (ZOH)
                        if k <= t:
                            coeff = (A_diag ** (t - k))[:, None] * B  # (D, nU)
                            T_u_t = T_u_t.at[:, s_k:e_k].set(coeff)
                    else:
                        # Last ZOH block: j = CH-1, CH, …, t all map to u_{CH-1}
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
    # ------------------------------------------------------------------
    def _get_discount(self, episode_length):
        denom = getattr(self.cfg, 'discount_denom', 5)
                s_u = k_u * nU
                e_u = (k_u + 1) * nU
                Tu  = T_u_list[t]   # (D, n)
                f   = f_list[t]     # (D,)

                # Stage state cost:  z^T diag(Q) z + q^T z
                QTu   = Q_diag[:, None] * Tu                  # (D, n)
                Q_qp  = Q_qp + (discount ** t) * 2.0 * (Tu.T @ QTu)
                c_qp  = c_qp + (discount ** t) * (2.0 * (QTu.T @ f) + Tu.T @ q_vec)

                # Stage control cost:  u^T diag(R) u + r^T u  (block k_u)
                Q_qp = Q_qp.at[s_u:e_u, s_u:e_u].add(
                    (discount ** t) * 2.0 * jnp.diag(R_diag))
                c_qp = c_qp.at[s_u:e_u].add((discount ** t) * r_vec)

            # Terminal cost – ensemble average (keeps problem a strict QP)
            Tu_H = T_u_list[H - 1]   # (D, n)
            f_H  = f_list[H - 1]     # (D,)
            s_t  = (CH - 1) * nU
            e_t  = CH * nU
    # ------------------------------------------------------------------
            P_avg = jnp.mean(P_diags, axis=0)   # (D,)
            p_avg = jnp.mean(p_mat,   axis=0)   # (D,)
            K_avg = jnp.mean(K_diags, axis=0)   # (nU,)
            k_avg = jnp.mean(k_mat,   axis=0)   # (nU,)

            PTu  = P_avg[:, None] * Tu_H         # (D, n)
            Q_qp = Q_qp + (discount ** H) * 2.0 * (Tu_H.T @ PTu)
            c_qp = c_qp + (discount ** H) * (2.0 * (PTu.T @ f_H) + Tu_H.T @ p_avg)
    # ------------------------------------------------------------------
            Q_qp = Q_qp.at[s_t:e_t, s_t:e_t].add(
                (discount ** H) * 2.0 * jnp.diag(K_avg))
            c_qp = c_qp.at[s_t:e_t].add((discount ** H) * k_avg)
        self.shift_u = np.asarray(shift_u, dtype=np.float32)
            # Ridge for numerical stability
            Q_qp = Q_qp + 1e-6 * jnp.eye(n)
    @torch.no_grad()
            # ----------------------------------------------------------
            # 3. Inequality constraints: a_low ≤ u_t ≤ a_high  ∀t
            #    Tiled over CH steps → G U ≤ h
            # ----------------------------------------------------------
            a_high_t = jnp.tile(a_high, CH)           # (n,)
            a_low_t  = jnp.tile(a_low,  CH)           # (n,)
            G = jnp.concatenate([ jnp.eye(n), -jnp.eye(n)], axis=0)   # (2n, n)
            h = jnp.concatenate([a_high_t, -a_low_t], axis=0)          # (2n,)
        Args:
            # No equality constraints
            A_eq = jnp.zeros((0, n))
            b_eq = jnp.zeros((0,))

            # ----------------------------------------------------------
            # 4. Solve with qpax (JAX interior-point QP solver)
            # ----------------------------------------------------------
            x, _s, _z, _y, converged, _iters = qpax.solve_qp(
                Q_qp, c_qp, A_eq, b_eq, G, h)
            u_norm = self.model.pi(z, deterministic=eval_mode)[0].cpu().numpy()
            return x.reshape(CH, nU), converged

        self._jax_solve_mpc = jax.jit(_build_and_solve)
            u_norm = self._plan_jax(z, x0, eval_mode)

        self.action_history.append(u_norm.copy())
        self.state_history.append(x0.copy())

        return np.asarray(u_norm, dtype=np.float32)

        self._a_low_jax  = jnp.array(a_low)
    # JAX MPC controller builder
    # ------------------------------------------------------------------
    def _build_jax_controller(self):
        """
        Build and JIT-compile the JAX MPC solve function.

        Build QP matrices on-the-fly and solve with qpax.
        Falls back to the policy net if the solver does not converge.
        constraints enforced by clipping after each Adam step.

        Objective (all quantities are diagonal matrices passed as vectors):
          J(U) = Σ_{t=0}^{H-1}  γ^t [ z_{t+1}^T diag(Q) z_{t+1}
        state_seq  = np.array(self.state_history[-self.history_horizon:], dtype=np.float32)
        action_seq = np.array(self.action_history[-self.history_horizon:], dtype=np.float32)
        state_t  = torch.tensor(state_seq,  device=self.device).unsqueeze(0)
                               + k_i^T u_{H-1} + pb_i ]
        obs_t    = torch.tensor(x0, dtype=torch.float32, device=self.device).unsqueeze(0)
        Dynamics:  z_{t+1} = A_diag ⊙ z_t + B u_t   (A is diagonal)
        """
        D = self.latent_dim
        nU = self.act_dim
        CH = getattr(self.cfg, 'control_horizon', 5)
        E = self.num_ensembles
        discount = float(self.discount)
        n_steps = int(getattr(self.cfg, 'mpc_gd_steps', 200))
        mpc_lr = float(getattr(self.cfg, 'mpc_lr', 1e-2))
        apply_action_constraints = getattr(self.cfg, 'apply_action_constraints', True)

        self._control_horizon = CH
        # Critic params
        P_diags_t = F.relu(self.model._P_indices)   # (E, D)
        K_diags_t = F.relu(self.model._K_indices)   # (E, nU)
        p_t  = self.model._p.squeeze(-1)             # (E, D)
        k_t  = self.model._k.squeeze(-1)             # (E, nU)
        # pb_vec not needed: it is a constant offset that does not affect
        # the optimal U and is therefore omitted from the QP.
            """
                q_vec  : (D,)
                R_diag : (nU,)        – diagonal of stage control cost
                r_vec  : (nU,)
                P_diags: (E, D)       – diagonal of terminal state critic per ensemble
            U_sol, converged = self._jax_solve_mpc(
                K_diags: (E, nU)      – diagonal of terminal control critic
                k_mat  : (E, nU)
                pb_vec : (E,)         – terminal critic bias
            """
            U = U_flat.reshape(CH, nU)
            z = z0
            cost = jnp.zeros(())

            for t in range(H):
                k_u = min(t, CH - 1)
                u = U[k_u]
                stage = (jnp.dot(Q_diag * z, z)
                         + jnp.dot(q_vec, z)
                         + jnp.dot(R_diag * u, u)
            if not bool(converged):
                return self.model.pi(z, deterministic=eval_mode)[0].cpu().numpy()
            u = _jax_to_numpy(U_sol[0])   # first control step, shape (nU,)
        except Exception:
            # Terminal cost – max over ensemble critics (vmap over E)
            u_term = U[CH - 1]

            def single_critic(P_d, p_v, K_d, k_v, pb_s):
                return (jnp.dot(P_d * z, z)
                        + jnp.dot(p_v, z)
                        + jnp.dot(K_d * u_term, u_term)
                        + jnp.dot(k_v, u_term)
                        + pb_s)

            critic_vals = jax.vmap(single_critic)(P_diags, p_mat, K_diags, k_mat, pb_vec)
            cost = cost + (discount ** H) * jnp.max(critic_vals)
            return cost

        # ----------------------------------------------------------------
        # JIT-compiled projected-Adam solve
        # ----------------------------------------------------------------
        optimizer = optax.adam(mpc_lr)
        loss_and_grad = jax.value_and_grad(_objective)

        def _solve(z0, A_diag, B, Q_diag, q_vec, R_diag, r_vec,
                   P_diags, p_mat, K_diags, k_mat, pb_vec, a_low, a_high):
            U_init = jnp.zeros(CH * nU)
            opt_state = optimizer.init(U_init)

            def step_fn(carry, _):
                U, opt_state = carry
                _, grads = loss_and_grad(U, z0, A_diag, B, Q_diag, q_vec,
                                         R_diag, r_vec, P_diags, p_mat,
                                         K_diags, k_mat, pb_vec)
                updates, new_opt_state = optimizer.update(grads, opt_state)
                new_U = optax.apply_updates(U, updates)
                if apply_action_constraints:
                    # Box-project each time step's control
                    new_U = jnp.clip(new_U.reshape(CH, nU),
                                     a_low[None, :], a_high[None, :]).reshape(-1)
                return (new_U, new_opt_state), None

            (U_final, _), _ = jax.lax.scan(step_fn, (U_init, opt_state),
                                            None, length=n_steps)
            return U_final.reshape(CH, nU)

        self._jax_solve_mpc = jax.jit(_solve)

        # Pre-fetch action bound arrays (static across calls)
        a_high = np.asarray(
            getattr(self.cfg, 'a_bound_high', np.ones(nU)), dtype=np.float32)
        a_low = np.asarray(
            getattr(self.cfg, 'a_bound_low', -np.ones(nU)), dtype=np.float32)
        self._a_high_jax = jnp.array(a_high)
        self._a_low_jax = jnp.array(a_low)

    # ------------------------------------------------------------------
    # JAX-based MPC planning
    # ------------------------------------------------------------------
    def _plan_jax(self, z: torch.Tensor, x0: np.ndarray, eval_mode: bool):
        """
        Solve the JAX/optax MPC problem.  Falls back to the policy net on
        numerical failure.

        Returns:
            u_norm: normalised action, shape [act_dim]
        """
        state_seq = np.array(self.state_history[-self.history_horizon:],
                             dtype=np.float32)
        action_seq = np.array(self.action_history[-self.history_horizon:],
                              dtype=np.float32)
        state_t = torch.tensor(state_seq, device=self.device).unsqueeze(0)
        action_t = torch.tensor(action_seq, device=self.device).unsqueeze(0)
        obs_t = torch.tensor(x0, dtype=torch.float32, device=self.device).unsqueeze(0)

        A_diag, B, Q_diag, q, R_diag, r_lin = self.model.encode_context(
            state_t, action_t, obs_t)

        # Extract single-sample tensors [D] / [D, nU] / etc.
        A_diag_t = A_diag[0]   # (D,)
        B_t      = B[0]         # (D, nU)
        Q_diag_t = Q_diag[0]   # (D,)
        q_t      = q[0]         # (D,)
        R_diag_t = R_diag[0]   # (nU,)
        r_t      = r_lin[0]     # (nU,)
        z_t      = z[0]         # (D,)

        # Critic params (stay on same device as model)
        P_diags_t = F.relu(self.model._P_indices)          # (E, D)
        p_t       = self.model._p                           # (E, D, 1) or (E, D)
        pb_t      = self.model._pb                          # (E, 1)
        K_diags_t = F.relu(self.model._K_indices)          # (E, nU)
        k_t       = self.model._k                           # (E, nU, 1) or (E, nU)

        # Squeeze trailing dim if present
        p_t  = p_t.squeeze(-1)   # (E, D)
        pb_t = pb_t.squeeze(-1)  # (E,)
        k_t  = k_t.squeeze(-1)   # (E, nU)

        # ---- PyTorch → JAX (zero-copy DLPack on CUDA) ----
        def to_jax(t: torch.Tensor):
            return _torch_to_jax(t.detach())

        try:
            U_sol = self._jax_solve_mpc(
                to_jax(z_t),
                to_jax(A_diag_t),
                to_jax(B_t),
                to_jax(Q_diag_t),
                to_jax(q_t),
                to_jax(R_diag_t),
                to_jax(r_t),
                to_jax(P_diags_t),
                to_jax(p_t),
                to_jax(K_diags_t),
                to_jax(k_t),
                to_jax(pb_t),
                self._a_low_jax,
                self._a_high_jax,
            )
            u = _jax_to_numpy(U_sol[0])   # first control step
        except Exception as e:
            # Fallback to policy net
            return self.model.pi(z, deterministic=eval_mode)[0].cpu().numpy()

        if not eval_mode:
            std = self.model.get_pi_std(z)[0]
            epsilon = (std * torch.randn(self.act_dim, device=std.device)
                       ).detach().cpu().numpy()
            u = u + epsilon

        return np.clip(u, -1.0, 1.0).astype(np.float32)

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
    # Policy update
    # ------------------------------------------------------------------
    def update_pi(self, zs, A_diag, B_mat):
        """
        Update policy using a sequence of latent states.

        Args:
            zs:     [T, batch, latent_dim]  (detached)
            A_diag: unused – kept for API compatibility
            B_mat:  unused – kept for API compatibility
        Returns:
            pi_loss (float)
        """
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_critic_grad(False)

        T, B, _ = zs.shape

        actions, log_probs = self.model.pi(zs, return_log_prob=True)

        zs_flat = zs.reshape(T * B, -1)
        actions_flat = actions.reshape(T * B, -1)
        vals = self.model.Q_value(zs_flat, actions_flat, target=False, return_type='min')
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
            self.grad_clip_norm
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
        z_flat = next_z.reshape(T * B, -1)
        next_a = self.model.pi(z_flat, target=True, deterministic=True)
        next_val = self.model.Q_value(z_flat, next_a, target=True, return_type='min')
        next_val = next_val.view(T, B, 1)
        return reward + self.discount * next_val

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------
    def update(self, buffer):
        """
        Main update function.

        Args:
            buffer: replay buffer with ``buffer.sample()`` returning
                    ``(obs, action, reward, task)``.
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
            next_mean = self.model.encode(obs[1:])
            td_targets = self._td_target(next_mean, reward)

        self.model_optim.zero_grad(set_to_none=True)
        self.model.train()

        B = obs.shape[1]
        D = self.latent_dim

        z        = self.model.encode(obs[self.history_horizon])
        z_target = self.model.encode(obs[self.history_horizon + 1], target=True)
        z_random = z

        ctx_state  = obs[:self.history_horizon].permute(1, 0, 2)
        ctx_action = action[:self.history_horizon].permute(1, 0, 2)
        A_diag, B_mat, Q_diag, q, R_diag, r_lin = self.model.encode_context(
            ctx_state, ctx_action, obs[self.history_horizon]
        )

        zs       = torch.empty(H + 1, B, D, device=self.device)
        z_randoms = torch.empty(H + 1, B, D, device=self.device)
        zs[0]       = z
        z_randoms[0] = z_random
        consistency_loss = torch.tensor(0.0, device=self.device)

        for t in range(H):
            z        = self.model.next(z,        action[t + self.history_horizon], A_diag, B_mat)
            z_random = self.model.next(z_random, action[t + self.history_horizon], A_diag, B_mat)
            consistency_loss += F.mse_loss(z, next_mean[t + self.history_horizon]) * (self.rho ** t)
            zs[t + 1]        = z
            z_randoms[t + 1] = z_random

        reward_loss = torch.tensor(0.0, device=self.device)
        for t in range(H):
            act_t  = action[t + self.history_horizon]
            r_pred = self.model.reward(z_randoms[t + 1], act_t, Q_diag, q, R_diag, r_lin)
            reward_loss += F.mse_loss(r_pred, reward[t + self.history_horizon]) * (self.rho ** t)

        z_for_p  = zs[0]
        act_for_p = action[self.history_horizon]
        with torch.no_grad():
            a_target = self.model.pi(z_target, target=True, deterministic=True)
        p_target_val = reward[self.history_horizon] + self.discount * self.model.Q_value(
            z_target, a_target, target=True, return_type='min'
        )
        p_pred_all = self.model.Q_value(z_for_p, act_for_p, target=False, return_type='all')
        p_loss = F.mse_loss(p_pred_all, p_target_val.detach().expand_as(p_pred_all))

        consistency_loss = consistency_loss / H
        reward_loss      = reward_loss / H
        value_loss       = p_loss

        total_loss = (
            self.consistency_coef * consistency_loss
            + self.reward_coef    * reward_loss
            + self.value_coef     * value_loss
        )

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_norm
        )
        self.model_optim.step()

        pi_loss = self.update_pi(zs.detach(), A_diag.detach(), B_mat.detach())

        self.model.soft_update_targets()

        self.model.eval()
        return {
            'consistency_loss': float(consistency_loss.item()),
            'reward_loss':       float(reward_loss.item()),
            'value_loss':        float(value_loss.item()),
            'p_loss':            float(p_loss.item()),
            'pi_loss':           pi_loss,
            'total_loss':        float(total_loss.item()),
            'grad_norm':         float(grad_norm),
            'pi_scale':          float(self.scale.value),
        }
