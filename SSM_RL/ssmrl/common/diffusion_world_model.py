"""
SSM World Model for SSM-RL.
Implements a state-space model with:
  - Variational encoder (mean + log_sigma)
  - Diffusion-based temporal context encoder that generates sequences of
    (A[t], B[t], Q_diag[t], q[t], R_diag[t], r_lin[t]) over the prediction horizon
  - Full A / dense B linear dynamics: z' = A*z + B*u
  - Quadratic reward: z^T Q z + q^T z + a^T R a + r^T a + b
  - Ensemble P-critics (quadratic value): z^T P z + p^T z + pb
  - Policy network (tanh-squashed)

Follows the architecture of WorldModel in ssmrl/common/world_model.py.
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: simple MLP builder
# ---------------------------------------------------------------------------
class _ShiftedELU(nn.Module):
    """ELU(x) + 1  — ensures strictly positive output (useful for PSD diagonals)."""
    def forward(self, x):
        return F.elu(x) + 1.0


def _mlp(in_dim, hidden_dims, out_dim, act=nn.ReLU, output_act=None, dropout=0.0):
    """Build a simple MLP with optional output activation."""
    dims = [in_dim] + list(hidden_dims) + [out_dim]
    mods = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(nn.LayerNorm(dims[i + 1]))
            mods.append(act())
            if dropout > 0:
                mods.append(nn.Dropout(dropout))
    if output_act is not None:
        mods.append(output_act)
    return nn.Sequential(*mods)


# ---------------------------------------------------------------------------
# Diffusion model components
# ---------------------------------------------------------------------------
class SinusoidalTimestepEmbed(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B] integer tensor
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)   # [B, half_dim]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)   # [B, dim]


class HistoryGRUEncoder(nn.Module):
    """
    Encodes (state_history, action_history, current_obs) into a conditioning vector.
    Uses a GRU over the history window then concatenates current_obs.
    """
    def __init__(self, state_dim, act_dim, cond_dim, gru_hidden=256):
        super().__init__()
        self.gru = nn.GRU(input_size=state_dim + act_dim,
                          hidden_size=gru_hidden, num_layers=1, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(gru_hidden + state_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.SiLU(),
        )

    def forward(self, state_history, action_history, current_obs):
        """
        Args:
            state_history:  [B, T, state_dim]
            action_history: [B, T, act_dim]
            current_obs:    [B, state_dim]
        Returns:
            condition: [B, cond_dim]
        """
        inp = torch.cat([state_history, action_history], dim=-1)   # [B, T, s+a]
        _, h = self.gru(inp)                                        # h: [1, B, gru_hidden]
        h = h.squeeze(0)                                            # [B, gru_hidden]
        return self.proj(torch.cat([h, current_obs], dim=-1))       # [B, cond_dim]


class SequenceDenoiser(nn.Module):
    """
    Denoising network for SSM parameter sequences.
    Takes (noisy_seq_flat, timestep, condition) and predicts the clean sequence.

    Architecture: project inputs → residual MLP blocks → output projection.
    """
    def __init__(self, seq_dim, cond_dim, t_emb_dim=128, hidden_dim=512, num_layers=4):
        super().__init__()
        self.t_embed = SinusoidalTimestepEmbed(t_emb_dim)

        in_dim = seq_dim + t_emb_dim + cond_dim
        layers = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, seq_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_noisy, t, condition):
        """
        Args:
            x_noisy:   [B, seq_dim]  noisy parameter sequence (flattened)
            t:         [B]           integer diffusion timestep
            condition: [B, cond_dim]
        Returns:
            x0_pred:   [B, seq_dim]  predicted clean sequence
        """
        t_emb = self.t_embed(t)
        return self.net(torch.cat([x_noisy, t_emb, condition], dim=-1))


class DiffusionSSMModel(nn.Module):
    """
    Conditional diffusion model that generates sequences of SSM parameters
    (A[t], B[t], Q_diag[t], q[t], R_diag[t], r_lin[t]) for t=0..H-1,
    conditioned on (state_history, action_history, current_obs).

    Training scheme (fully unsupervised via RL losses):
      - The ``denoiser`` is trained end-to-end by minimising the downstream
        consistency and reward losses.  No supervised DDPM reconstruction loss
        is used.  During the training forward pass, a short differentiable DDIM
        chain (``num_train_steps``, default 3) is run with fresh Gaussian noise,
        and gradients propagate back through every denoising step into the
        denoiser and history encoder.
      - At inference the same denoiser is used with a longer DDIM chain
        (``num_sample_steps``), starting from pure noise.

    Why this works:
      The denoiser maps (noise, t, condition) → clean SSM params.  When the
      downstream losses penalise bad dynamics / reward predictions, the
      gradient signal teaches the denoiser which param sequences are consistent
      with the observed transitions.  Random noise provides stochasticity that
      acts as implicit regularisation and allows diverse planning at inference.
    """

    def __init__(self, state_dim, act_dim, latent_dim, history_horizon,
                 prediction_horizon, cond_dim=256, denoiser_hidden=512,
                 num_diffusion_steps=100, num_sample_steps=10,
                 num_train_steps=3, gru_hidden=256, t_emb_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.history_horizon = history_horizon
        self.prediction_horizon = prediction_horizon
        self.num_diffusion_steps = num_diffusion_steps
        self.num_sample_steps = num_sample_steps
        self.num_train_steps = num_train_steps

        # Per-step SSM parameter sizes
        self.A_size = latent_dim * latent_dim       # A (full matrix, flattened)
        self.B_size = latent_dim * act_dim          # B (flattened)
        self.Q_size = latent_dim                    # Q_diag
        self.q_size = latent_dim                    # q
        self.R_size = act_dim                       # R_diag
        self.r_size = act_dim                       # r_lin
        self.param_dim = (self.A_size + self.B_size + self.Q_size
                          + self.q_size + self.R_size + self.r_size)
        self.seq_dim = prediction_horizon * self.param_dim

        # History → condition
        self.history_encoder = HistoryGRUEncoder(state_dim, act_dim, cond_dim, gru_hidden)

        # Diffusion denoiser (no separate reference_net; trained purely by RL losses)
        self.denoiser = SequenceDenoiser(self.seq_dim, cond_dim, t_emb_dim,
                                         denoiser_hidden, num_layers=4)

        # DDPM linear beta schedule
        betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1.0 - alphas_cumprod).sqrt())

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------
    def encode_condition(self, state_history, action_history, current_obs):
        """→ [B, cond_dim]"""
        return self.history_encoder(state_history, action_history, current_obs)

    # ------------------------------------------------------------------
    # Differentiable DDIM for training
    # ------------------------------------------------------------------
    def sample_train(self, condition, num_steps=None):
        """
        Differentiable DDIM chain used during training.

        Runs ``num_steps`` (default ``num_train_steps``) denoising steps with
        gradient enabled so that consistency / reward losses can backpropagate
        through the entire chain into the denoiser and history encoder.

        Args:
            condition: [B, cond_dim]
        Returns:
            raw_seq: [B, H, param_dim]  (pre-activation, gradients attached)
        """
        if num_steps is None:
            num_steps = self.num_train_steps
        B = condition.shape[0]
        device = condition.device

        T = self.num_diffusion_steps
        step_size = max(T // num_steps, 1)
        timesteps = list(range(0, T, step_size))[::-1]   # high → low

        # Start from fresh Gaussian noise (stochastic, acts as regularisation)
        x = torch.randn(B, self.seq_dim, device=device)

        for idx, t_val in enumerate(timesteps):
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            x0_pred = self.denoiser(x, t_batch, condition)   # differentiable

            if t_val == 0 or idx == len(timesteps) - 1:
                x = x0_pred
            else:
                alpha_t    = self.alphas_cumprod[t_val]
                t_prev     = timesteps[idx + 1]
                alpha_prev = self.alphas_cumprod[t_prev]
                # DDIM deterministic update (all ops differentiable)
                eps_pred = (x - alpha_t.sqrt() * x0_pred) / (1.0 - alpha_t).sqrt().clamp(min=1e-8)
                x = alpha_prev.sqrt() * x0_pred + (1.0 - alpha_prev).sqrt() * eps_pred

        return x.view(B, self.prediction_horizon, self.param_dim)

    # ------------------------------------------------------------------
    # DDIM sampling (inference, no grad)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, condition, num_steps=None):
        """
        DDIM sampling from pure noise to SSM parameter sequence.

        Args:
            condition: [B, cond_dim]
        Returns:
            raw_seq: [B, H, param_dim]  (pre-activation, apply split_and_activate)
        """
        if num_steps is None:
            num_steps = self.num_sample_steps
        B = condition.shape[0]
        device = condition.device

        T = self.num_diffusion_steps
        step_size = max(T // num_steps, 1)
        # Timesteps from high to low
        timesteps = list(range(0, T, step_size))[::-1]

        x = torch.randn(B, self.seq_dim, device=device)

        for idx, t_val in enumerate(timesteps):
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            x0_pred = self.denoiser(x, t_batch, condition)

            if t_val == 0:
                x = x0_pred
            else:
                alpha_t    = self.alphas_cumprod[t_val]
                t_prev     = timesteps[idx + 1] if idx + 1 < len(timesteps) else 0
                alpha_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)

                # DDIM deterministic update
                eps_pred = (x - alpha_t.sqrt() * x0_pred) / (1.0 - alpha_t).sqrt().clamp(min=1e-8)
                x = alpha_prev.sqrt() * x0_pred + (1.0 - alpha_prev).sqrt() * eps_pred

        return x.view(B, self.prediction_horizon, self.param_dim)

    # ------------------------------------------------------------------
    # Split raw sequence into named SSM tensors with proper activations
    # ------------------------------------------------------------------
    def split_and_activate(self, seq):
        """
        Args:
            seq: [B, H, param_dim]  raw (pre-activation) parameter sequence
        Returns:
            A_mat:  [B, H, latent_dim, latent_dim]   full dynamics matrix via tanh
            B_mat:  [B, H, latent_dim, act_dim]
            Q_diag: [B, H, latent_dim]           ≥ 0 via softplus
            q:      [B, H, latent_dim]
            R_diag: [B, H, act_dim]              ≥ 0 via softplus
            r_lin:  [B, H, act_dim]
        """
        B, H, _ = seq.shape
        D, nU = self.latent_dim, self.act_dim

        idx = 0
        A_raw = seq[..., idx:idx + self.A_size]; idx += self.A_size
        B_raw = seq[..., idx:idx + self.B_size]; idx += self.B_size
        Q_raw = seq[..., idx:idx + self.Q_size]; idx += self.Q_size
        q_raw = seq[..., idx:idx + self.q_size]; idx += self.q_size
        R_raw = seq[..., idx:idx + self.R_size]; idx += self.R_size
        r_raw = seq[..., idx:idx + self.r_size]

        A_mat  = A_raw.view(B, H, D, D)     # [B, H, D, D]
        B_mat  = B_raw.view(B, H, D, nU)                # [B, H, D, nU]
        Q_diag = F.softplus(Q_raw)                      # [B, H, D]
        q      = q_raw                                   # [B, H, D]
        R_diag = F.softplus(R_raw)                      # [B, H, nU]
        r_lin  = r_raw                                   # [B, H, nU]

        return A_mat, B_mat, Q_diag, q, R_diag, r_lin


# ---------------------------------------------------------------------------
# SSM World Model
# ---------------------------------------------------------------------------
class WorldModel(nn.Module):
    """
    State-Space-Model world model.

    Key components:
    * Deterministic encoder: obs → latent
    * DiffusionSSMModel: generates sequences of SSM parameters
      (A_diag[t], B[t], Q_diag[t], q[t], R_diag[t], r_lin[t]) for t=0..H-1
      conditioned on (state_history, action_history, current_obs).
    * Reward: quadratic in z and a using the per-step Q, q, R, r.
    * Ensemble quadratic critics.
    * SAC-style stochastic policy.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        state_dim        = cfg.obs_shape['state'][0]
        act_dim          = cfg.action_dim
        latent_dim       = cfg.latent_dim
        history_horizon  = getattr(cfg, 'history_horizon', 10)
        prediction_horizon = getattr(cfg, 'horizon', 5)
        num_ensembles    = getattr(cfg, 'num_ensembles', 3)
        encoder_hidden   = getattr(cfg, 'encoder_struct', [256, 256])
        policy_hidden    = getattr(cfg, 'policy_struct', [256, 256])

        # Diffusion model hyper-parameters
        cond_dim             = getattr(cfg, 'diffusion_cond_dim', 256)
        denoiser_hidden      = getattr(cfg, 'diffusion_denoiser_hidden', 512)
        num_diffusion_steps  = getattr(cfg, 'num_diffusion_steps', 100)
        num_sample_steps     = getattr(cfg, 'num_sample_steps', 10)
        num_train_steps      = getattr(cfg, 'num_train_steps', 3)
        gru_hidden           = getattr(cfg, 'diffusion_gru_hidden', 256)
        t_emb_dim            = getattr(cfg, 'diffusion_t_emb_dim', 128)

        self.latent_dim      = latent_dim
        self.act_dim         = act_dim
        self.state_dim       = state_dim
        self.history_horizon = history_horizon
        self.prediction_horizon = prediction_horizon
        self.num_ensembles   = num_ensembles

        # ---- Deterministic encoder: obs → latent ----
        self._encoder_mean = _mlp(state_dim, encoder_hidden, latent_dim)
        self._encoder_mean_target = deepcopy(self._encoder_mean)
        for p in self._encoder_mean_target.parameters():
            p.requires_grad_(False)

        # ---- Diffusion SSM model (replaces transformer + MLP heads) ----
        self._diffusion_model = DiffusionSSMModel(
            state_dim=state_dim,
            act_dim=act_dim,
            latent_dim=latent_dim,
            history_horizon=history_horizon,
            prediction_horizon=prediction_horizon,
            cond_dim=cond_dim,
            denoiser_hidden=denoiser_hidden,
            num_diffusion_steps=num_diffusion_steps,
            num_sample_steps=num_sample_steps,
            num_train_steps=num_train_steps,
            gru_hidden=gru_hidden,
            t_emb_dim=t_emb_dim,
        )

        # Scalar reward bias
        self._b = nn.Parameter(torch.zeros(1))

        # ---- Policy (SAC-style stochastic) ----
        log_std_min = getattr(cfg, 'log_std_min', -5)
        log_std_max = getattr(cfg, 'log_std_max', 2)
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

        self._pi_trunk        = _mlp(latent_dim, policy_hidden, policy_hidden[-1])
        self._pi_mean_head    = nn.Linear(policy_hidden[-1], act_dim)
        self._pi_log_std_head = nn.Linear(policy_hidden[-1], act_dim)
        self._pi = nn.ModuleList([self._pi_trunk, self._pi_mean_head, self._pi_log_std_head])

        self._pi_target_trunk        = deepcopy(self._pi_trunk).requires_grad_(False)
        self._pi_target_mean_head    = deepcopy(self._pi_mean_head).requires_grad_(False)
        self._pi_target_log_std_head = deepcopy(self._pi_log_std_head).requires_grad_(False)
        self._pi_target = nn.ModuleList([
            self._pi_target_trunk, self._pi_target_mean_head, self._pi_target_log_std_head
        ])

        # ---- Ensemble quadratic critics ----
        P_indices_init = torch.empty(num_ensembles, latent_dim)
        nn.init.xavier_uniform_(P_indices_init)
        self._P_indices = nn.Parameter(P_indices_init)

        p_init = torch.empty(num_ensembles, latent_dim, 1)
        nn.init.xavier_uniform_(p_init)
        self._p = nn.Parameter(p_init)

        pb_init = torch.empty(num_ensembles, 1)
        nn.init.uniform_(pb_init, -0.1, 0.1)
        self._pb = nn.Parameter(pb_init)

        K_indices_init = torch.empty(num_ensembles, act_dim)
        nn.init.xavier_uniform_(K_indices_init)
        self._K_indices = nn.Parameter(K_indices_init)

        k_init = torch.empty(num_ensembles, act_dim, 1)
        nn.init.xavier_uniform_(k_init)
        self._k = nn.Parameter(k_init)

        # Target copies
        self._P_indices_target = nn.Parameter(P_indices_init.clone(), requires_grad=False)
        self._p_target         = nn.Parameter(p_init.clone(),         requires_grad=False)
        self._pb_target        = nn.Parameter(pb_init.clone(),        requires_grad=False)
        self._K_indices_target = nn.Parameter(K_indices_init.clone(), requires_grad=False)
        self._k_target         = nn.Parameter(k_init.clone(),         requires_grad=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Train / eval overrides
    # ------------------------------------------------------------------
    def train(self, mode=True):
        super().train(mode)
        for m in self._pi_target:
            m.train(False)
        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(self, obs, target=False):
        encoder = self._encoder_mean_target if target else self._encoder_mean
        return encoder(obs)

    def encode_context(self, state_history, action_history, current_obs,
                       use_diffusion=False):
        """
        Generate per-step SSM parameter sequences over the prediction horizon.

        Args:
            state_history:  [B, history_horizon, state_dim]
            action_history: [B, history_horizon, act_dim]
            current_obs:    [B, state_dim]
            use_diffusion:  if True, use full DDIM sampling (inference, no grad);
                            if False, use short differentiable DDIM chain (training).
        Returns:
            A_mat:  [B, H, latent_dim, latent_dim]
            B:      [B, H, latent_dim, act_dim]
            Q_diag: [B, H, latent_dim]
            q:      [B, H, latent_dim]
            R_diag: [B, H, act_dim]
            r_lin:  [B, H, act_dim]
        """
        condition = self._diffusion_model.encode_condition(
            state_history, action_history, current_obs)

        if use_diffusion:
            # Inference: full DDIM chain, no grad
            raw_seq = self._diffusion_model.sample(condition)
        else:
            # Training: short differentiable DDIM chain so that
            # consistency / reward losses backprop into the denoiser
            raw_seq = self._diffusion_model.sample_train(condition)

        return self._diffusion_model.split_and_activate(raw_seq)

    def diffusion_loss(self, state_history, action_history, current_obs):
        """Deprecated – denoiser is now trained purely via downstream RL losses."""
        raise NotImplementedError(
            "diffusion_loss() is removed. The denoiser is trained end-to-end "
            "through consistency_loss and reward_loss in SSMAgent.update()."
        )

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def next(self, z, a, A_mat, B):
        """
        z' = A*z + B*u  (A is a full matrix)

        Args:
            z:     [B, latent_dim]
            a:     [B, act_dim]
            A_mat: [B, latent_dim, latent_dim]
            B:     [B, latent_dim, act_dim]
        Returns:
            z': [B, latent_dim]
        """
        Az = torch.bmm(A_mat, z.unsqueeze(-1)).squeeze(-1)
        Bu = torch.bmm(B, a.unsqueeze(-1)).squeeze(-1)
        return Az + Bu

    # ------------------------------------------------------------------
    # Reward (quadratic in z and a)
    # ------------------------------------------------------------------
    def reward(self, z, a, Q_diag, q, R_diag, r_lin):
        """
        r = -(z^T diag(Q) z + q^T z + a^T diag(R) a + r_lin^T a + b)
        """
        z_quad = (Q_diag * z * z).sum(dim=-1, keepdim=True)
        z_lin  = (q * z).sum(dim=-1, keepdim=True)
        a_quad = (R_diag * a * a).sum(dim=-1, keepdim=True)
        a_lin  = (r_lin * a).sum(dim=-1, keepdim=True)
        return -(z_quad + z_lin + a_quad + a_lin + self._b)

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    def pi(self, z, target=False, deterministic=False, return_log_prob=False):
        trunk        = self._pi_target[0] if target else self._pi_trunk
        mean_head    = self._pi_target[1] if target else self._pi_mean_head
        log_std_head = self._pi_target[2] if target else self._pi_log_std_head

        h = trunk(z)
        mean    = mean_head(h)
        log_std = log_std_head(h).clamp(self._log_std_min, self._log_std_max)
        std     = log_std.exp()

        if deterministic:
            action = torch.tanh(mean)
            if return_log_prob:
                return action, self._gaussian_log_prob(mean, mean, std)
            return action

        eps    = torch.randn_like(std)
        u      = mean + std * eps
        action = torch.tanh(u)

        if return_log_prob:
            return action, self._gaussian_log_prob(u, mean, std)
        return action

    def get_pi_std(self, z, target=False):
        trunk        = self._pi_target[0] if target else self._pi_trunk
        log_std_head = self._pi_target[2] if target else self._pi_log_std_head
        h = trunk(z)
        return log_std_head(h).clamp(self._log_std_min, self._log_std_max).exp()

    @staticmethod
    def _gaussian_log_prob(u, mean, std):
        log_prob_gaussian = (
            -0.5 * ((u - mean) / std).pow(2)
            - std.log()
            - 0.5 * torch.tensor(2 * torch.pi).log().to(u.device)
        )
        log_det_jacobian = 2.0 * (torch.log(torch.tensor(2.0, device=u.device))
                                  - u - F.softplus(-2.0 * u))
        return (log_prob_gaussian - log_det_jacobian).sum(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # Ensemble critics
    # ------------------------------------------------------------------
    def Q_value(self, z, a=None, target=False, return_type='max'):
        """
        Q_i(z, a) = -(z^T diag(P_i) z + p_i^T z + [a^T diag(K_i) a + k_i^T a] + pb_i)
        Action terms are included only when a is provided.
        """
        P_idx  = self._P_indices_target if target else self._P_indices
        p_vec  = self._p_target         if target else self._p
        pb_val = self._pb_target        if target else self._pb
        K_idx  = self._K_indices_target if target else self._K_indices
        k_vec  = self._k_target         if target else self._k

        P_diag  = F.relu(P_idx)
        z_quad  = torch.einsum('bd,ed->be', z * z, P_diag)
        z_lin   = torch.einsum('bd,ed->be', z, p_vec.squeeze(-1))
        pb_sq   = pb_val.squeeze(-1)

        all_vals = -(z_quad + z_lin + pb_sq.unsqueeze(0))

        if a is not None:
            K_diag  = F.relu(K_idx)
            a_quad  = torch.einsum('bd,ed->be', a * a, K_diag)
            a_lin   = torch.einsum('bd,ed->be', a, k_vec.squeeze(-1))
            all_vals = all_vals - (a_quad + a_lin)

        if return_type == 'all':
            return all_vals
        elif return_type == 'max':
            return all_vals.max(dim=1, keepdim=True).values
        elif return_type == 'min':
            return all_vals.min(dim=1, keepdim=True).values
        elif return_type == 'avg':
            return all_vals.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown return_type: {return_type}")

    # ------------------------------------------------------------------
    # Gradient control helpers
    # ------------------------------------------------------------------
    def track_critic_grad(self, mode=True):
        for p in [self._P_indices, self._p, self._pb, self._K_indices, self._k]:
            p.requires_grad_(mode)

    # ------------------------------------------------------------------
    # Soft target updates
    # ------------------------------------------------------------------
    def soft_update_targets(self, tau=None):
        if tau is None:
            tau = self.cfg.tau
        with torch.no_grad():
            for p_tgt, p in zip(self._encoder_mean_target.parameters(),
                                 self._encoder_mean.parameters()):
                p_tgt.data.lerp_(p.data, tau)
            self._P_indices_target.data.lerp_(self._P_indices.data, tau)
            self._p_target.data.lerp_(self._p.data, tau)
            self._pb_target.data.lerp_(self._pb.data, tau)
            self._K_indices_target.data.lerp_(self._K_indices.data, tau)
            self._k_target.data.lerp_(self._k.data, tau)
            pi_pairs = [
                (self._pi_target_trunk,        self._pi_trunk),
                (self._pi_target_mean_head,    self._pi_mean_head),
                (self._pi_target_log_std_head, self._pi_log_std_head),
            ]
            for tgt_mod, src_mod in pi_pairs:
                for p_tgt, p in zip(tgt_mod.parameters(), src_mod.parameters()):
                    p_tgt.data.lerp_(p.data, tau)
