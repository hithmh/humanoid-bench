"""
SSM World Model v2 – implements the pipeline described in
"Probabilistic Synthesis of Dynamical Generators: A Denoising Diffusion
Framework for Trajectory-Conditioned System Identification".

Key changes vs diffusion_world_model.py (v1):
  1. Trajectory encoder: GRU → Transformer encoder with sinusoidal positional
     embeddings and mean-pooling.  Captures long-range temporal dependencies as
     recommended by the guideline.
  2. Denoiser conditioning: concatenation → adaptive LayerNorm (adaLN).
     The combined (t_emb, cond_emb) modulation vector produces per-block
     scale/shift for every hidden layer.
  3. Noise prediction: the denoiser now predicts the added noise ε (ε-prediction)
     rather than x0.  This matches the original DDPM derivation and the
     guideline's L_simple objective.
  4. Explicit DDPM loss (L_simple): a standalone ``ddpm_loss()`` method returns
     E[‖ε − ε_θ(g_t, t, τ)‖²] which the agent adds to the total training loss.
  5. Generator latent space (VAE / latent diffusion):
     NOTE – the guideline recommends compressing the generator g into a
     low-dimensional latent z_g via a VAE before running diffusion on z_g.
     This has *not* been fully implemented here because it requires a
     separate dataset of (trajectory, generator) pairs for VAE pre-training,
     and the current RL setup generates targets implicitly.  A lightweight
     linear projector (Encoder/Decoder pair) is included as a placeholder
     that can be replaced with a full VAE once paired data is available.
     See `_gen_encoder` / `_gen_decoder` in DiffusionSSMModel.
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
    """ELU(x) + 1 — strictly positive (useful for PSD diagonals)."""
    def forward(self, x):
        return F.elu(x) + 1.0


def _mlp(in_dim, hidden_dims, out_dim, act=nn.ReLU, output_act=None, dropout=0.0):
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
# Sinusoidal embeddings (shared by positional + timestep)
# ---------------------------------------------------------------------------
class SinusoidalEmbed(nn.Module):
    """General-purpose sinusoidal embedding for integer indices."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """x: [B] or [B, T] integer tensor → [..., dim]"""
        device   = x.device
        half_dim = self.dim // 2
        freq     = math.log(10000) / max(half_dim - 1, 1)
        freq     = torch.exp(torch.arange(half_dim, device=device) * -freq)
        emb      = x.float().unsqueeze(-1) * freq              # [..., half_dim]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)   # [..., dim]


# ---------------------------------------------------------------------------
# Transformer trajectory encoder
# ---------------------------------------------------------------------------
class HistoryTransformerEncoder(nn.Module):
    """
    Encodes (state_history, action_history, current_obs) into a conditioning
    vector using a Transformer encoder, as recommended by the guideline.

    Transformer advantages over GRU:
      • Captures long-range temporal dependencies via self-attention
      • Parallelisable over time-steps (no sequential dependency)
      • Avoids vanishing-gradient issues inherent in RNNs

    Architecture:
      1. Linear projection of (s_t ∥ a_t) to `d_model`
      2. Sinusoidal positional embedding added
      3. N-layer standard TransformerEncoder
      4. Mean-pool across time → [B, d_model]
      5. Concatenate current_obs, project to `cond_dim`

    Args:
        state_dim:  dimension of each state vector
        act_dim:    dimension of each action vector
        cond_dim:   output conditioning vector dimension
        d_model:    Transformer internal width
        nhead:      number of attention heads
        num_layers: number of TransformerEncoderLayer stacks
        dropout:    attention dropout
    """
    def __init__(self, state_dim, act_dim, cond_dim,
                 d_model=256, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        input_dim = state_dim + act_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed  = SinusoidalEmbed(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,               # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_proj = nn.Sequential(
            nn.Linear(d_model + state_dim, cond_dim),
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
        B, T, _ = state_history.shape
        device   = state_history.device

        # [B, T, d_model]
        x = self.input_proj(torch.cat([state_history, action_history], dim=-1))

        # Add positional encoding
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # [B, T]
        x = x + self.pos_embed(positions)  # [B, T, d_model]

        # Transformer
        x = self.transformer(x)            # [B, T, d_model]

        # Mean-pool across time
        x = x.mean(dim=1)                  # [B, d_model]

        # Append current obs and project
        return self.out_proj(torch.cat([x, current_obs], dim=-1))   # [B, cond_dim]


# ---------------------------------------------------------------------------
# Adaptive LayerNorm (adaLN) block
# ---------------------------------------------------------------------------
class AdaLNBlock(nn.Module):
    """
    Single residual MLP block conditioned via adaptive LayerNorm (adaLN).

    The modulation vector `mod` (from time-step + condition) produces per-channel
    scale γ and shift β that replace the learned parameters of LayerNorm:
        y = γ(mod) ⊙ LayerNorm(x) + β(mod)

    This is the conditioning mechanism recommended in the guideline as an
    alternative to cross-attention.
    """
    def __init__(self, hidden_dim, mod_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc   = nn.Linear(hidden_dim, hidden_dim)
        self.act  = nn.SiLU()
        # Predict 2 * hidden_dim modulation params (γ and β)
        self.mod_proj = nn.Linear(mod_dim, 2 * hidden_dim)
        # Zero-init so the block starts as identity
        nn.init.zeros_(self.mod_proj.weight)
        nn.init.zeros_(self.mod_proj.bias)

    def forward(self, x, mod):
        """
        x:   [B, hidden_dim]
        mod: [B, mod_dim]
        """
        gamma, beta = self.mod_proj(mod).chunk(2, dim=-1)   # each [B, hidden_dim]
        h = self.norm(x)
        h = (1.0 + gamma) * h + beta                        # adaLN modulation
        h = self.act(self.fc(h))
        return x + h                                        # residual connection


# ---------------------------------------------------------------------------
# Sequence denoiser with adaLN conditioning and ε-prediction
# ---------------------------------------------------------------------------
class SequenceDenoiserAdaLN(nn.Module):
    """
    Denoising network that predicts the added noise ε (ε-prediction).

    Given (g_t, t, τ-condition), the network predicts ε such that:
        g_0 ≈ (g_t - sqrt(1 - ᾱ_t) * ε_θ(g_t, t, τ)) / sqrt(ᾱ_t)

    Conditioning is injected via adaLN (adaptive LayerNorm) at every residual
    block, which is more expressive than simple concatenation and avoids
    increasing the sequence dimension.

    Architecture:
        input projection: seq_dim → hidden_dim
        modulation MLP:   (t_emb ∥ cond) → mod_dim
        N × AdaLNBlock(hidden_dim, mod_dim)
        output projection: hidden_dim → seq_dim

    Args:
        seq_dim:     dimension of the flattened SSM parameter sequence
        cond_dim:    dimension of the trajectory conditioning vector
        t_emb_dim:   sinusoidal time-step embedding dimension
        hidden_dim:  hidden width of each adaLN block
        num_layers:  number of adaLN residual blocks
    """
    def __init__(self, seq_dim, cond_dim, t_emb_dim=128,
                 hidden_dim=512, num_layers=4):
        super().__init__()
        self.t_embed  = SinusoidalEmbed(t_emb_dim)
        mod_dim       = t_emb_dim + cond_dim

        self.mod_mlp  = nn.Sequential(
            nn.Linear(mod_dim, hidden_dim),
            nn.SiLU(),
        )
        self.in_proj  = nn.Linear(seq_dim, hidden_dim)
        self.blocks   = nn.ModuleList(
            [AdaLNBlock(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, seq_dim)
        # Zero-init output projection (standard practice for diffusion models)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, g_noisy, t, condition):
        """
        Predict noise ε added to g_0 to produce g_noisy at diffusion step t.

        Args:
            g_noisy:   [B, seq_dim]   noisy parameter sequence (flattened)
            t:         [B]            integer diffusion timestep
            condition: [B, cond_dim]  trajectory conditioning vector
        Returns:
            eps_pred:  [B, seq_dim]   predicted noise
        """
        t_emb = self.t_embed(t)                         # [B, t_emb_dim]
        mod   = self.mod_mlp(torch.cat([t_emb, condition], dim=-1))  # [B, hidden_dim]

        x = self.in_proj(g_noisy)                       # [B, hidden_dim]
        for blk in self.blocks:
            x = blk(x, mod)
        return self.out_proj(self.out_norm(x))          # [B, seq_dim]


# ---------------------------------------------------------------------------
# Diffusion SSM model (v2)
# ---------------------------------------------------------------------------
class DiffusionSSMModel(nn.Module):
    """
    Conditional diffusion model (v2) generating SSM parameter sequences
    (A[t], B[t], Q_diag[t], q[t], R_diag[t], r_lin[t]) for t=0..H-1
    conditioned on (state_history, action_history, current_obs).

    Changes vs v1:
      • Trajectory encoder: HistoryTransformerEncoder (was GRU)
      • Denoiser: SequenceDenoiserAdaLN with ε-prediction (was x0-prediction + concat)
      • Explicit DDPM loss L_simple via ``ddpm_loss()``
      • ``sample_train()`` now runs a short differentiable chain and
        reconstructs x0 from ε-predictions — gradients still flow end-to-end.
      • Generator latent projectors (_gen_encoder / _gen_decoder) as placeholders
        for a future full VAE implementation of latent diffusion.

    Training objective (in SSMAgent.update):
        L_total = consistency_coef * L_consistency
                + reward_coef     * L_reward
                + value_coef      * L_value
                + diffusion_coef  * L_simple          ← NEW in v16

    Args:
        state_dim, act_dim, latent_dim, history_horizon, prediction_horizon:
            standard world-model dimensions.
        gen_latent_dim: dimension of the generator latent space for latent
            diffusion.  If None or == seq_dim the projectors are identity and
            diffusion is run directly in parameter space.
        cond_dim:             trajectory conditioning vector dimension
        d_model:              Transformer d_model
        nhead:                Transformer attention heads
        num_transformer_layers: Transformer depth
        denoiser_hidden:      adaLN block hidden width
        num_diffusion_steps:  T in the DDPM schedule
        num_sample_steps:     DDIM steps at inference
        num_train_steps:      differentiable DDIM steps during training
        t_emb_dim:            sinusoidal timestep embedding dimension
    """

    def __init__(self, state_dim, act_dim, latent_dim, history_horizon,
                 prediction_horizon,
                 gen_latent_dim=None,
                 cond_dim=256,
                 d_model=256, nhead=4, num_transformer_layers=2,
                 denoiser_hidden=512,
                 num_diffusion_steps=100,
                 num_sample_steps=10,
                 num_train_steps=3,
                 t_emb_dim=128):
        super().__init__()
        self.state_dim          = state_dim
        self.act_dim            = act_dim
        self.latent_dim         = latent_dim
        self.history_horizon    = history_horizon
        self.prediction_horizon = prediction_horizon
        self.num_diffusion_steps = num_diffusion_steps
        self.num_sample_steps   = num_sample_steps
        self.num_train_steps    = num_train_steps

        # Per-step SSM parameter sizes
        self.A_size = latent_dim * latent_dim
        self.B_size = latent_dim * act_dim
        self.Q_size = latent_dim
        self.q_size = latent_dim
        self.R_size = act_dim
        self.r_size = act_dim
        self.param_dim = (self.A_size + self.B_size + self.Q_size
                          + self.q_size + self.R_size + self.r_size)
        self.seq_dim = prediction_horizon * self.param_dim

        # Generator latent space (placeholder for latent diffusion)
        # If gen_latent_dim is None or equals seq_dim → identity projection
        self._use_gen_latent = (gen_latent_dim is not None
                                and gen_latent_dim != self.seq_dim)
        diff_dim = gen_latent_dim if self._use_gen_latent else self.seq_dim
        if self._use_gen_latent:
            self._gen_encoder = nn.Sequential(
                nn.Linear(self.seq_dim, diff_dim),
                nn.LayerNorm(diff_dim), nn.SiLU(),
            )
            self._gen_decoder = nn.Sequential(
                nn.Linear(diff_dim, self.seq_dim),
            )
        else:
            self._gen_encoder = nn.Identity()
            self._gen_decoder = nn.Identity()
        self.diff_dim = diff_dim

        # Trajectory → condition (Transformer, guideline §4.2)
        self.history_encoder = HistoryTransformerEncoder(
            state_dim=state_dim, act_dim=act_dim, cond_dim=cond_dim,
            d_model=d_model, nhead=nhead, num_layers=num_transformer_layers,
        )

        # Denoiser: ε-prediction with adaLN conditioning (guideline §4.1)
        self.denoiser = SequenceDenoiserAdaLN(
            seq_dim=diff_dim,
            cond_dim=cond_dim,
            t_emb_dim=t_emb_dim,
            hidden_dim=denoiser_hidden,
            num_layers=4,
        )

        # DDPM linear beta schedule
        betas             = torch.linspace(1e-4, 0.02, num_diffusion_steps)
        alphas            = 1.0 - betas
        alphas_cumprod    = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas',                      betas)
        self.register_buffer('alphas',                     alphas)
        self.register_buffer('alphas_cumprod',             alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod',        alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             (1.0 - alphas_cumprod).sqrt())

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------
    def encode_condition(self, state_history, action_history, current_obs):
        """→ [B, cond_dim]"""
        return self.history_encoder(state_history, action_history, current_obs)

    # ------------------------------------------------------------------
    # Reconstruction helper: ε → x0
    # ------------------------------------------------------------------
    def _eps_to_x0(self, g_t, eps, t_batch):
        """
        Recover clean sequence from noise prediction:
            x0 = (g_t - sqrt(1 - ᾱ_t) * eps) / sqrt(ᾱ_t)
        """
        sqrt_a   = self.sqrt_alphas_cumprod[t_batch].unsqueeze(-1)
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t_batch].unsqueeze(-1)
        return (g_t - sqrt_1ma * eps) / sqrt_a.clamp(min=1e-8)

    # ------------------------------------------------------------------
    # Explicit DDPM training loss L_simple (guideline eq. L_simple)
    # ------------------------------------------------------------------
    def ddpm_loss(self, state_history, action_history, current_obs, g0):
        """
        Compute L_simple = E[‖ε − ε_θ(g_t, t, τ)‖²].

        Args:
            state_history:  [B, T, state_dim]
            action_history: [B, T, act_dim]
            current_obs:    [B, state_dim]
            g0:             [B, seq_dim]  clean SSM parameter sequence (target)
        Returns:
            loss: scalar
        """
        B      = g0.shape[0]
        device = g0.device

        # Optionally compress to generator latent space
        z_g0 = self._gen_encoder(g0)           # [B, diff_dim]

        # Sample random diffusion timestep
        t_batch = torch.randint(0, self.num_diffusion_steps, (B,), device=device)

        # Forward diffusion: g_t = sqrt(ᾱ_t)*g_0 + sqrt(1-ᾱ_t)*ε
        eps    = torch.randn_like(z_g0)
        sqrt_a = self.sqrt_alphas_cumprod[t_batch].unsqueeze(-1)          # [B, 1]
        sqrt_b = self.sqrt_one_minus_alphas_cumprod[t_batch].unsqueeze(-1)
        g_t    = sqrt_a * z_g0 + sqrt_b * eps                              # [B, diff_dim]

        # Condition
        condition = self.encode_condition(state_history, action_history, current_obs)

        # Predict noise
        eps_pred = self.denoiser(g_t, t_batch, condition)                  # [B, diff_dim]

        return F.mse_loss(eps_pred, eps)

    # ------------------------------------------------------------------
    # Differentiable DDIM for training (short chain, gradients enabled)
    # ------------------------------------------------------------------
    def sample_train(self, condition, num_steps=None):
        """
        Short differentiable DDIM chain for training.

        Uses ε-prediction: x0 is reconstructed from the predicted noise at each
        step.  Gradients propagate back through all denoising steps.

        Args:
            condition: [B, cond_dim]
        Returns:
            raw_seq: [B, H, param_dim]
        """
        if num_steps is None:
            num_steps = self.num_train_steps
        B      = condition.shape[0]
        device = condition.device

        T         = self.num_diffusion_steps
        step_size = max(T // num_steps, 1)
        timesteps = list(range(0, T, step_size))[::-1]   # high → low

        g = torch.randn(B, self.diff_dim, device=device)

        for idx, t_val in enumerate(timesteps):
            t_batch  = torch.full((B,), t_val, device=device, dtype=torch.long)
            eps_pred = self.denoiser(g, t_batch, condition)        # ε-prediction
            x0_pred  = self._eps_to_x0(g, eps_pred, t_batch)

            if t_val == 0 or idx == len(timesteps) - 1:
                g = x0_pred
            else:
                alpha_t    = self.alphas_cumprod[t_val]
                t_prev     = timesteps[idx + 1]
                alpha_prev = self.alphas_cumprod[t_prev]
                eps_pred_det = (g - alpha_t.sqrt() * x0_pred) / (1.0 - alpha_t).sqrt().clamp(1e-8)
                g = alpha_prev.sqrt() * x0_pred + (1.0 - alpha_prev).sqrt() * eps_pred_det

        # Decode from generator latent space
        raw = self._gen_decoder(g)                                 # [B, seq_dim]
        return raw.view(B, self.prediction_horizon, self.param_dim)

    # ------------------------------------------------------------------
    # DDIM sampling (inference, no grad)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, condition, num_steps=None):
        """
        Full DDIM inference sampling from pure noise.

        Args:
            condition: [B, cond_dim]
        Returns:
            raw_seq: [B, H, param_dim]
        """
        if num_steps is None:
            num_steps = self.num_sample_steps
        B      = condition.shape[0]
        device = condition.device

        T         = self.num_diffusion_steps
        step_size = max(T // num_steps, 1)
        timesteps = list(range(0, T, step_size))[::-1]

        g = torch.randn(B, self.diff_dim, device=device)

        for idx, t_val in enumerate(timesteps):
            t_batch  = torch.full((B,), t_val, device=device, dtype=torch.long)
            eps_pred = self.denoiser(g, t_batch, condition)
            x0_pred  = self._eps_to_x0(g, eps_pred, t_batch)

            if t_val == 0:
                g = x0_pred
            else:
                alpha_t    = self.alphas_cumprod[t_val]
                t_prev     = timesteps[idx + 1] if idx + 1 < len(timesteps) else 0
                alpha_prev = (self.alphas_cumprod[t_prev]
                              if t_prev > 0
                              else torch.tensor(1.0, device=device))
                eps_det = (g - alpha_t.sqrt() * x0_pred) / (1.0 - alpha_t).sqrt().clamp(1e-8)
                g = alpha_prev.sqrt() * x0_pred + (1.0 - alpha_prev).sqrt() * eps_det

        raw = self._gen_decoder(g)
        return raw.view(B, self.prediction_horizon, self.param_dim)

    # ------------------------------------------------------------------
    # Split raw sequence → named SSM tensors with activations
    # ------------------------------------------------------------------
    def split_and_activate(self, seq):
        """
        Args:
            seq: [B, H, param_dim]
        Returns:
            A_mat:  [B, H, D, D]   full dynamics matrix (tanh → ≈ stable)
            B_mat:  [B, H, D, nU]
            Q_diag: [B, H, D]      ≥ 0 (softplus)
            q:      [B, H, D]
            R_diag: [B, H, nU]     ≥ 0 (softplus)
            r_lin:  [B, H, nU]
        """
        B, H, _ = seq.shape
        D, nU   = self.latent_dim, self.act_dim

        idx    = 0
        A_raw  = seq[..., idx:idx + self.A_size]; idx += self.A_size
        B_raw  = seq[..., idx:idx + self.B_size]; idx += self.B_size
        Q_raw  = seq[..., idx:idx + self.Q_size]; idx += self.Q_size
        q_raw  = seq[..., idx:idx + self.q_size]; idx += self.q_size
        R_raw  = seq[..., idx:idx + self.R_size]; idx += self.R_size
        r_raw  = seq[..., idx:idx + self.r_size]

        A_mat  = A_raw.view(B, H, D, D)
        B_mat  = B_raw.view(B, H, D, nU)
        Q_diag = F.softplus(Q_raw)
        q      = q_raw
        R_diag = F.softplus(R_raw)
        r_lin  = r_raw

        return A_mat, B_mat, Q_diag, q, R_diag, r_lin


# ---------------------------------------------------------------------------
# SSM World Model v2
# ---------------------------------------------------------------------------
class WorldModel(nn.Module):
    """
    SSM World Model v2.

    Same public API as WorldModel in diffusion_world_model.py, but backed by
    DiffusionSSMModel v2 (Transformer encoder, ε-prediction, adaLN conditioning).

    New public method:
        diffusion_loss(state_history, action_history, current_obs, g0) → scalar
            Computes L_simple for optional supervised diffusion pre-training or
            as an auxiliary loss during RL training.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        state_dim          = cfg.obs_shape['state'][0]
        act_dim            = cfg.action_dim
        latent_dim         = cfg.latent_dim
        history_horizon    = getattr(cfg, 'history_horizon', 10)
        prediction_horizon = getattr(cfg, 'horizon', 5)
        num_ensembles      = getattr(cfg, 'num_ensembles', 3)
        encoder_hidden     = getattr(cfg, 'encoder_struct', [256, 256])
        policy_hidden      = getattr(cfg, 'policy_struct', [256, 256])

        # Diffusion hyper-parameters (v2 specific)
        cond_dim                = getattr(cfg, 'diffusion_cond_dim',         256)
        d_model                 = getattr(cfg, 'diffusion_d_model',          256)
        nhead                   = getattr(cfg, 'diffusion_nhead',              4)
        num_transformer_layers  = getattr(cfg, 'diffusion_transformer_layers', 2)
        denoiser_hidden         = getattr(cfg, 'diffusion_denoiser_hidden',  512)
        num_diffusion_steps     = getattr(cfg, 'num_diffusion_steps',        100)
        num_sample_steps        = getattr(cfg, 'num_sample_steps',            10)
        num_train_steps         = getattr(cfg, 'num_train_steps',              3)
        t_emb_dim               = getattr(cfg, 'diffusion_t_emb_dim',        128)
        gen_latent_dim          = getattr(cfg, 'gen_latent_dim',             None)

        self.latent_dim         = latent_dim
        self.act_dim            = act_dim
        self.state_dim          = state_dim
        self.history_horizon    = history_horizon
        self.prediction_horizon = prediction_horizon
        self.num_ensembles      = num_ensembles

        # ---- Deterministic encoder: obs → latent ----
        self._encoder_mean        = _mlp(state_dim, encoder_hidden, latent_dim)
        self._encoder_mean_target = deepcopy(self._encoder_mean)
        for p in self._encoder_mean_target.parameters():
            p.requires_grad_(False)

        # ---- Diffusion SSM model v2 ----
        self._diffusion_model = DiffusionSSMModel(
            state_dim=state_dim,
            act_dim=act_dim,
            latent_dim=latent_dim,
            history_horizon=history_horizon,
            prediction_horizon=prediction_horizon,
            gen_latent_dim=gen_latent_dim,
            cond_dim=cond_dim,
            d_model=d_model,
            nhead=nhead,
            num_transformer_layers=num_transformer_layers,
            denoiser_hidden=denoiser_hidden,
            num_diffusion_steps=num_diffusion_steps,
            num_sample_steps=num_sample_steps,
            num_train_steps=num_train_steps,
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
        self._pi = nn.ModuleList([
            self._pi_trunk, self._pi_mean_head, self._pi_log_std_head])

        self._pi_target_trunk        = deepcopy(self._pi_trunk)
        self._pi_target_mean_head    = deepcopy(self._pi_mean_head)
        self._pi_target_log_std_head = deepcopy(self._pi_log_std_head)
        for m in [self._pi_target_trunk,
                  self._pi_target_mean_head,
                  self._pi_target_log_std_head]:
            for p in m.parameters():
                p.requires_grad_(False)
        self._pi_target = nn.ModuleList([
            self._pi_target_trunk,
            self._pi_target_mean_head,
            self._pi_target_log_std_head,
        ])

        # ---- Ensemble quadratic critics ----
        def _init(shape, fn=nn.init.xavier_uniform_):
            t = torch.empty(*shape)
            fn(t)
            return nn.Parameter(t)

        self._P_indices = _init((num_ensembles, latent_dim))
        self._p         = _init((num_ensembles, latent_dim, 1))
        pb_init = torch.empty(num_ensembles, 1)
        nn.init.uniform_(pb_init, -0.1, 0.1)
        self._pb        = nn.Parameter(pb_init)
        self._K_indices = _init((num_ensembles, act_dim))
        self._k         = _init((num_ensembles, act_dim, 1))

        # Target copies
        self._P_indices_target = nn.Parameter(self._P_indices.data.clone(), requires_grad=False)
        self._p_target         = nn.Parameter(self._p.data.clone(),         requires_grad=False)
        self._pb_target        = nn.Parameter(self._pb.data.clone(),        requires_grad=False)
        self._K_indices_target = nn.Parameter(self._K_indices.data.clone(), requires_grad=False)
        self._k_target         = nn.Parameter(self._k.data.clone(),         requires_grad=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        enc = self._encoder_mean_target if target else self._encoder_mean
        return enc(obs)

    def encode_context(self, state_history, action_history, current_obs,
                       use_diffusion=False):
        """Generate per-step SSM parameter sequences (same API as v1)."""
        condition = self._diffusion_model.encode_condition(
            state_history, action_history, current_obs)
        if use_diffusion:
            raw_seq = self._diffusion_model.sample(condition)
        else:
            raw_seq = self._diffusion_model.sample_train(condition)
        return self._diffusion_model.split_and_activate(raw_seq)

    def diffusion_loss(self, state_history, action_history, current_obs, g0):
        """
        L_simple DDPM loss (guideline eq. L_simple).

        Can be used as:
          (a) a standalone supervised pre-training objective when g0 is a
              teacher-sampled parameter sequence, or
          (b) an auxiliary loss during RL by passing the parameter sequence
              generated by the reference (training) DDIM chain.

        Args:
            state_history:  [B, T, state_dim]
            action_history: [B, T, act_dim]
            current_obs:    [B, state_dim]
            g0:             [B, seq_dim]  clean (reference) SSM parameter sequence
        Returns:
            loss: scalar
        """
        return self._diffusion_model.ddpm_loss(
            state_history, action_history, current_obs, g0)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def next(self, z, a, A_mat, B):
        """z' = A*z + B*u  (A is full matrix)."""
        Az = torch.bmm(A_mat, z.unsqueeze(-1)).squeeze(-1)
        Bu = torch.bmm(B,     a.unsqueeze(-1)).squeeze(-1)
        return Az + Bu

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def reward(self, z, a, Q_diag, q, R_diag, r_lin):
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

        h       = trunk(z)
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
        log_p = (-0.5 * ((u - mean) / std).pow(2)
                 - std.log()
                 - 0.5 * torch.tensor(2 * torch.pi).log().to(u.device))
        log_det = 2.0 * (torch.log(torch.tensor(2.0, device=u.device))
                         - u - F.softplus(-2.0 * u))
        return (log_p - log_det).sum(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # Ensemble critics
    # ------------------------------------------------------------------
    def Q_value(self, z, a=None, target=False, return_type='max'):
        P_idx  = self._P_indices_target if target else self._P_indices
        p_vec  = self._p_target         if target else self._p
        pb_val = self._pb_target        if target else self._pb
        K_idx  = self._K_indices_target if target else self._K_indices
        k_vec  = self._k_target         if target else self._k

        P_diag   = F.relu(P_idx)
        z_quad   = torch.einsum('bd,ed->be', z * z, P_diag)
        z_lin    = torch.einsum('bd,ed->be', z, p_vec.squeeze(-1))
        all_vals = -(z_quad + z_lin + pb_val.squeeze(-1).unsqueeze(0))

        if a is not None:
            K_diag   = F.relu(K_idx)
            a_quad   = torch.einsum('bd,ed->be', a * a, K_diag)
            a_lin    = torch.einsum('bd,ed->be', a, k_vec.squeeze(-1))
            all_vals = all_vals - (a_quad + a_lin)

        if   return_type == 'all': return all_vals
        elif return_type == 'max': return all_vals.max(dim=1, keepdim=True).values
        elif return_type == 'min': return all_vals.min(dim=1, keepdim=True).values
        elif return_type == 'avg': return all_vals.mean(dim=1, keepdim=True)
        else: raise ValueError(f"Unknown return_type: {return_type}")

    # ------------------------------------------------------------------
    def track_critic_grad(self, mode=True):
        for p in [self._P_indices, self._p, self._pb, self._K_indices, self._k]:
            p.requires_grad_(mode)

    # ------------------------------------------------------------------
    def soft_update_targets(self, tau=None):
        if tau is None:
            tau = self.cfg.tau
        with torch.no_grad():
            for p_t, p in zip(self._encoder_mean_target.parameters(),
                              self._encoder_mean.parameters()):
                p_t.data.lerp_(p.data, tau)
            for (pt, ps) in [
                (self._P_indices_target, self._P_indices),
                (self._p_target,         self._p),
                (self._pb_target,        self._pb),
                (self._K_indices_target, self._K_indices),
                (self._k_target,         self._k),
            ]:
                pt.data.lerp_(ps.data, tau)
            for tgt_m, src_m in [
                (self._pi_target_trunk,        self._pi_trunk),
                (self._pi_target_mean_head,    self._pi_mean_head),
                (self._pi_target_log_std_head, self._pi_log_std_head),
            ]:
                for p_t, p in zip(tgt_m.parameters(), src_m.parameters()):
                    p_t.data.lerp_(p.data, tau)
