"""
SSM World Model for SSM-RL.
Implements a state-space model with:
  - Variational encoder (mean + log_sigma)
  - Transformer-based temporal context encoder
  - Diagonal A / dense B linear dynamics: z' = diag(A)*z + B*u
  - Quadratic reward: z^T Q z + q^T z + b
  - Ensemble P-critics (quadratic value): z^T P z + p^T z + pb
  - Policy network (tanh-squashed)

Follows the architecture of WorldModel in ssmrl/common/world_model.py.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: simple MLP builder (uses NormedLinear from layers when available,
# but here we keep it lightweight / self-contained for SSM-specific heads).
# ---------------------------------------------------------------------------
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
# Transformer temporal-context encoder
# ---------------------------------------------------------------------------
class TransformerContextEncoder(nn.Module):
    """
    Takes a history window of (state, action) pairs and produces a fixed-size
    context vector via multi-head self-attention + global average pooling.
    """

    def __init__(self, input_dim, seq_length, d_model=128, num_heads=4,
                 dff=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.seq_length = seq_length
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Embedding(seq_length, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dff,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, x):
        """
        Args:
            x: [batch, seq_length, input_dim]
        Returns:
            context: [batch, d_model]
        """
        x = self.input_proj(x)  # [B, T, d_model]
        positions = torch.arange(self.seq_length, device=x.device)
        x = x + self.pos_emb(positions).unsqueeze(0)  # broadcast over batch
        x = self.transformer(x)  # [B, T, d_model]
        return x.mean(dim=1)  # global average pooling → [B, d_model]


# ---------------------------------------------------------------------------
# SSM World Model
# ---------------------------------------------------------------------------
class SSMWorldModel(nn.Module):
    """
    State-Space-Model world model, following the same structural conventions as
    ``WorldModel`` in ``ssmrl.common.world_model`` (TD-MPC2 style).

    Key differences from the vanilla TD-MPC2 WorldModel:
    * Dynamics are linear: z' = diag(A)*z + B*u  (A, B predicted per-step by
      MLP heads conditioned on a transformer context).
    * Reward is a learned quadratic form: z^T Q z + q^T z + b.
    * Critic is an ensemble of quadratic forms: z^T P_i z + p_i^T z + pb_i.
    * Encoder is variational (mean + log_sigma).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ---- dimensions (from cfg, expected to be set by the caller) ----
        state_dim = cfg.obs_shape['state'][0]
        act_dim = cfg.action_dim
        latent_dim = cfg.latent_dim
        history_horizon = getattr(cfg, 'history_horizon', 10)
        num_ensembles = getattr(cfg, 'num_ensembles', 3)
        encoder_hidden = getattr(cfg, 'encoder_struct', [256, 256])
        policy_hidden = getattr(cfg, 'policy_struct', [256, 256])
        transformer_d_model = getattr(cfg, 'transformer_d_model', 128)
        transformer_num_heads = getattr(cfg, 'transformer_num_heads', 4)
        transformer_dff = getattr(cfg, 'transformer_dff', 512)
        transformer_num_layers = getattr(cfg, 'transformer_num_layers', 2)
        transformer_dropout = getattr(cfg, 'transformer_dropout', 0.1)

        self.latent_dim = latent_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.history_horizon = history_horizon
        self.num_ensembles = num_ensembles

        # ---- Variational encoder: obs → (mean, log_sigma) ----
        self._encoder_mean = _mlp(state_dim, encoder_hidden, latent_dim)
        self._encoder_log_sigma = _mlp(state_dim, encoder_hidden, latent_dim)

        # ---- Transformer context encoder ----
        self._transformer = TransformerContextEncoder(
            input_dim=state_dim + act_dim,
            seq_length=history_horizon,
            d_model=transformer_d_model,
            num_heads=transformer_num_heads,
            dff=transformer_dff,
            num_layers=transformer_num_layers,
            dropout=transformer_dropout,
        )

        # ---- A, B dynamics heads (conditioned on transformer_output || current_obs) ----
        ctx_dim = transformer_d_model + state_dim
        self._A_net = _mlp(ctx_dim, encoder_hidden, latent_dim, output_act=nn.Tanh())
        self._B_net = _mlp(ctx_dim, encoder_hidden, latent_dim * act_dim)

        # ---- Quadratic reward heads ----
        self._Q_net = _mlp(ctx_dim, encoder_hidden, latent_dim, output_act=nn.ReLU())
        self._q_net = _mlp(ctx_dim, encoder_hidden, latent_dim)
        self._b = nn.Parameter(torch.zeros(1))

        # ---- Policy ----
        self._pi = _mlp(latent_dim, policy_hidden, act_dim, output_act=nn.Tanh())
        self._pi_target = deepcopy(self._pi).requires_grad_(False)

        # ---- Ensemble P critics (quadratic value function) ----
        self._P_indices = nn.Parameter(torch.ones(num_ensembles, latent_dim))
        self._p = nn.Parameter(torch.zeros(num_ensembles, latent_dim, 1))
        self._pb = nn.Parameter(torch.zeros(num_ensembles, 1))

        # Target copies
        self._P_indices_target = nn.Parameter(
            torch.ones(num_ensembles, latent_dim), requires_grad=False
        )
        self._p_target = nn.Parameter(
            torch.zeros(num_ensembles, latent_dim, 1), requires_grad=False
        )
        self._pb_target = nn.Parameter(
            torch.zeros(num_ensembles, 1), requires_grad=False
        )

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
        self._pi_target.train(False)
        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(self, obs):
        """
        Encode observations → latent (mean, sigma).

        Args:
            obs: [batch, state_dim] or [T, batch, state_dim]
        Returns:
            mean, sigma   (same leading shape as obs, last dim = latent_dim)
        """
        mean = self._encoder_mean(obs)
        log_sigma = self._encoder_log_sigma(obs).clamp(-2, 2)
        sigma = log_sigma.exp()
        return mean, sigma

    def encode_context(self, state_history, action_history, current_obs):
        """
        Run transformer on history and produce (A, B, Q, q) for the current step.

        Args:
            state_history:  [batch, history_horizon, state_dim]
            action_history: [batch, history_horizon, act_dim]
            current_obs:    [batch, state_dim]
        Returns:
            A_diag: [batch, latent_dim]          (diagonal of A)
            B:      [batch, latent_dim, act_dim]
            Q_diag: [batch, latent_dim]          (diagonal of Q)
            q:      [batch, latent_dim]
        """
        ctx_input = torch.cat([state_history, action_history], dim=-1)
        transformer_out = self._transformer(ctx_input)  # [batch, d_model]
        encoder_in = torch.cat([transformer_out, current_obs], dim=-1)

        A_diag = self._A_net(encoder_in)  # [batch, latent_dim]
        B_flat = self._B_net(encoder_in)  # [batch, latent_dim * act_dim]
        B = B_flat.view(-1, self.latent_dim, self.act_dim)

        Q_diag = self._Q_net(encoder_in)  # [batch, latent_dim]
        q = self._q_net(encoder_in)       # [batch, latent_dim]

        return A_diag, B, Q_diag, q

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def next(self, z, a, A_diag, B):
        """
        Predict next latent: z' = diag(A)*z + B*u

        Args:
            z:      [batch, latent_dim]
            a:      [batch, act_dim]
            A_diag: [batch, latent_dim]
            B:      [batch, latent_dim, act_dim]
        Returns:
            z': [batch, latent_dim]
        """
        # diag(A) * z
        Az = A_diag * z
        # B @ u  → [batch, latent_dim, 1] → squeeze
        Bu = torch.bmm(B, a.unsqueeze(-1)).squeeze(-1)
        return Az + Bu

    # ------------------------------------------------------------------
    # Reward (quadratic)
    # ------------------------------------------------------------------
    def reward(self, z, Q_diag, q):
        """
        Compute quadratic reward: z^T diag(Q) z + q^T z + b

        Args:
            z:      [batch, latent_dim]
            Q_diag: [batch, latent_dim]
            q:      [batch, latent_dim]
        Returns:
            r: [batch, 1]
        """
        quad = (Q_diag * z * z).sum(dim=-1, keepdim=True)
        lin = (q * z).sum(dim=-1, keepdim=True)
        return quad + lin + self._b

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    def pi(self, z, target=False):
        """
        Sample action from policy.

        Args:
            z: [batch, latent_dim]  or [T, batch, latent_dim]
            target: use target policy network
        Returns:
            action: same leading dims, last dim = act_dim
        """
        net = self._pi_target if target else self._pi
        return net(z)

    # ------------------------------------------------------------------
    # Ensemble P-Critic (quadratic value)
    # ------------------------------------------------------------------
    def Q_value(self, z, target=False, return_type='max'):
        """
        Compute ensemble quadratic critic value.

        Args:
            z: [batch, latent_dim]
            target: whether to use target parameters
            return_type: 'max' (pessimistic, take max across ensemble),
                         'all' (return all), 'min', 'avg'
        Returns:
            If 'max'/'min'/'avg': [batch, 1]
            If 'all': [batch, num_ensembles]
        """
        P_idx = self._P_indices_target if target else self._P_indices
        p_vec = self._p_target if target else self._p
        pb_val = self._pb_target if target else self._pb

        # P_idx: [E, D],  z: [B, D]
        P_diag = F.relu(P_idx)  # ensure PSD diagonal
        # quadratic: z^T diag(P) z  →  (P_diag * z^2).sum  →  [B, E]
        quad = torch.einsum('bd,ed->be', z * z, P_diag)

        # linear: p^T z  →  [B, E]
        p_sq = p_vec.squeeze(-1)  # [E, D]
        lin = torch.einsum('bd,ed->be', z, p_sq)

        # constant: pb  →  [E] broadcast to [B, E]
        pb_sq = pb_val.squeeze(-1)  # [E]
        all_vals = quad + lin + pb_sq.unsqueeze(0)

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
        """Enable / disable gradients for P-critic parameters."""
        for p in [self._P_indices, self._p, self._pb]:
            p.requires_grad_(mode)

    # ------------------------------------------------------------------
    # Soft target updates
    # ------------------------------------------------------------------
    def soft_update_targets(self, tau=None):
        """Polyak-average update of target P-params and target policy."""
        if tau is None:
            tau = self.cfg.tau
        with torch.no_grad():
            self._P_indices_target.data.lerp_(self._P_indices.data, tau)
            self._p_target.data.lerp_(self._p.data, tau)
            self._pb_target.data.lerp_(self._pb.data, tau)
            for p_tgt, p in zip(self._pi_target.parameters(), self._pi.parameters()):
                p_tgt.data.lerp_(p.data, tau)
