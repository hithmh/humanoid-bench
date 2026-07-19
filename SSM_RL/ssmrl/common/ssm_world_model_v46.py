"""
SSM World Model for SSM-RL.
Implements a state-space model with:
  - Variational encoder (mean + log_sigma)
  - Transformer-based temporal context encoder
  - Dense per-step linear dynamics: z' = A*z + B*u
  - Quadratic reward over X = [z, u]:
      r = X^T Q X + q^T X + b_q, with Q positive definite
  - Quadratic arrival Q-function ensemble over X = [z, u]
  - Latent-state policy network (tanh-squashed)

Follows the architecture of WorldModel in ssmrl/common/world_model.py.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from ssmrl.common import math
from ssmrl.common import layers
import numpy as np


# ---------------------------------------------------------------------------
# Helper: simple MLP builder (uses NormedLinear from layers when available,
# but here we keep it lightweight / self-contained for SSM-specific heads).
# ---------------------------------------------------------------------------
class _ShiftedELU(nn.Module):
    """ELU(x) + 1  — ensures strictly positive output (useful for PSD diagonals)."""
    def forward(self, x):
        return F.elu(x) + 1.0


def _mlp(in_dim, hidden_dims, out_dim, act=nn.Mish, output_act=None, dropout=0.0):
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
    * Dynamics are linear: z' = A*z + B*u  (dense A and B are mixed from
      learned basis matrices).
    * Reward is a learned quadratic form over latent state and action.
    * Arrival critic is a learned quadratic-form ensemble over latent state
      and action.
    * The policy consumes latent states from the encoder.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ---- dimensions (from cfg, expected to be set by the caller) ----
        state_dim = cfg.obs_shape['state'][0]
        act_dim = cfg.action_dim
        latent_dim = cfg.latent_dim
        history_horizon = getattr(cfg, 'history_horizon', 10)
        transformer_d_model = getattr(cfg, 'transformer_d_model', 128)
        transformer_num_heads = getattr(cfg, 'transformer_num_heads', 4)
        transformer_dff = getattr(cfg, 'transformer_dff', 512)
        transformer_num_layers = getattr(cfg, 'transformer_num_layers', 2)
        transformer_dropout = getattr(cfg, 'transformer_dropout', 0.1)

        prediction_horizon = getattr(cfg, 'horizon', 3)

        self.latent_dim = latent_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.history_horizon = history_horizon
        self.prediction_horizon = prediction_horizon

        # ---- Deterministic encoder: obs → latent (for world model) ----
        self._encoder_mean = layers.enc(cfg)

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
        self._A_num_bases = int(getattr(cfg, 'dynamics_a_num_bases', 16))
        self._A_identity_scale = float(getattr(cfg, 'dynamics_a_identity_scale', 1.0))
        a_basis_init = float(getattr(cfg, 'dynamics_a_basis_init', 0.01))
        self._B_num_bases = int(getattr(cfg, 'dynamics_b_num_bases', self._A_num_bases))
        b_basis_init = float(getattr(cfg, 'dynamics_b_basis_init', a_basis_init))
        # Each dynamics head predicts one coefficient sequence directly.
        self._A_net = layers.mlp(
            ctx_dim,
            2 * [cfg.mlp_dim],
            self._A_num_bases * prediction_horizon,
        )
        self._A_basis = nn.Parameter(torch.randn(self._A_num_bases, latent_dim, latent_dim) * a_basis_init)
        self._B_net = layers.mlp(
            ctx_dim,
            2 * [cfg.mlp_dim],
            self._B_num_bases * prediction_horizon,
        )
        self._B_basis = nn.Parameter(torch.randn(self._B_num_bases, latent_dim, act_dim) * b_basis_init)

        # ---- Context-conditioned quadratic reward head ----
        # r(z, u | c) = X^T Q(c) X + q(c)^T X + b, where X = [z, u].
        # The transformer context predicts Q and q; b remains a learned
        # context-independent constant.
        self.quadratic_dim = latent_dim + act_dim
        self.quadratic_pd_eps = float(getattr(cfg, 'quadratic_pd_eps', 1e-6))

        param_split = self.quadratic_dim * self.quadratic_dim
        self._reward_param_split = param_split
        self._reward_param_net = layers.mlp(
            ctx_dim,
            2 * [cfg.mlp_dim],
            param_split + self.quadratic_dim,
        )
        self._reward_b = nn.Parameter(torch.zeros(1))
        # ---- Policy (SAC-style stochastic: outputs mean + log_std) ----
        log_std_min = torch.tensor(getattr(cfg, 'log_std_min', -5), dtype=torch.float32)
        log_std_max = torch.tensor(getattr(cfg, 'log_std_max', 2), dtype=torch.float32)
        self.register_buffer('log_std_min', log_std_min, persistent=False)
        self.register_buffer('log_std_dif', log_std_max - log_std_min, persistent=False)

        self._pi = layers.mlp(
            latent_dim,
            2 * [cfg.mlp_dim],
            2 * act_dim,
        )

        # ---- Context-conditioned quadratic arrival Q-function ensemble ----
        # Q_arr(z, u | c) = X^T Q(c) X + q(c)^T X + b.
        # Each ensemble head has its own basis banks for Q and q. The context
        # net predicts nonnegative mixture weights over those bases.
        self.num_arrival_q = max(1, int(getattr(
            cfg, 'arrival_num_q', getattr(cfg, 'num_q', 2))))
        self._arrival_Q_num_bases = int(getattr(cfg, 'arrival_Q_num_bases', 16))
        self._arrival_q_num_bases = int(getattr(cfg, 'arrival_q_num_bases', 16))
        arrival_basis_init = float(getattr(cfg, 'arrival_basis_init', 0.01))
        self._arrival_param_nets = nn.ModuleList([
            layers.mlp(
                ctx_dim,
                2 * [cfg.mlp_dim],
                self._arrival_Q_num_bases + self._arrival_q_num_bases,
            )
            for _ in range(self.num_arrival_q)
        ])
        self._arrival_Q_basis_raw = nn.Parameter(
            torch.randn(
                self.num_arrival_q,
                self._arrival_Q_num_bases,
                self.quadratic_dim,
                self.quadratic_dim,
            ) * arrival_basis_init
        )
        self._arrival_q_basis = nn.Parameter(
            torch.randn(
                self.num_arrival_q,
                self._arrival_q_num_bases,
                self.quadratic_dim,
            ) * arrival_basis_init
        )
        self._arrival_b = nn.Parameter(torch.zeros(self.num_arrival_q, 1))
        self._arrival_param_nets_target = deepcopy(self._arrival_param_nets).requires_grad_(False)
        self._arrival_Q_basis_raw_target = nn.Parameter(
            self._arrival_Q_basis_raw.detach().clone(), requires_grad=False)
        self._arrival_q_basis_target = nn.Parameter(
            self._arrival_q_basis.detach().clone(), requires_grad=False)
        self._arrival_b_target = nn.Parameter(
            self._arrival_b.detach().clone(), requires_grad=False)


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
        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(self, obs, target=False):
        """
        Encode observations → latent (deterministic).

        Args:
            obs:    [batch, state_dim] or [T, batch, state_dim]
            target: if True, use the target (EMA) encoder
        Returns:
            z   (same leading shape as obs, last dim = latent_dim)
        """
        encoder = self._encoder_mean
        return encoder[self.cfg.obs](obs)

    def encode_context(
        self,
        state_history,
        action_history,
        current_obs,
    ):
        """
        Run transformer on history and produce dynamics plus quadratic
        reward parameters for the current step.

        Args:
            state_history:  [batch, history_horizon, state_dim]
            action_history: [batch, history_horizon, act_dim]
            current_obs:    [batch, state_dim]
        Returns:
            A_seq:      [batch, prediction_horizon, latent_dim, latent_dim]
            B_seq:      [batch, prediction_horizon, latent_dim, act_dim]
            reward_Q:   [batch, prediction_horizon, latent_dim + act_dim, latent_dim + act_dim]
            reward_q:   [batch, prediction_horizon, latent_dim + act_dim]
            reward_b:   [batch, prediction_horizon, 1]
            encoder_in: [batch, ctx_dim]  (transformer_out || current_obs)
        """
        ctx_input = torch.cat([state_history, action_history], dim=-1)
        transformer_out = self._transformer(ctx_input)  # [batch, d_model]
        encoder_in = torch.cat([transformer_out, current_obs], dim=-1)

        H = self.prediction_horizon
        A_weights = self._dynamics_weights(self._A_net, encoder_in, H, self._A_num_bases)
        A_seq = self._basis_dynamics_matrix(
            A_weights, self._A_basis)

        B_weights = self._dynamics_weights(self._B_net, encoder_in, H, self._B_num_bases)
        B_seq = self._basis_matrix(B_weights, self._B_basis)

        reward_params = self.reward_params(encoder_in)

        return A_seq, B_seq, *reward_params, encoder_in

    @staticmethod
    def _dynamics_weights(net, encoder_in, horizon, num_bases):
        """Return single-head basis weights from a context predictor."""
        weight_flat = net(encoder_in)
        return weight_flat.view(-1, horizon, num_bases)

    @staticmethod
    def _basis_matrix(weights, basis):
        """
        Mix learned dense basis matrices using bounded context coefficients.

        Args:
            weights: [..., horizon, num_bases] context-dependent coefficients
            basis: [num_bases, rows, cols] learned dense basis matrices
        Returns:
            [..., horizon, rows, cols] dense matrix sequence
        """
        num_bases = basis.shape[0]
        coeffs = torch.tanh(weights)
        coeffs = nn.Softmax(dim=-1)(weights)
        bounded_basis = torch.tanh(basis)
        mixed = torch.einsum('...k,kij->...ij', coeffs, basis) #/ (num_bases ** 0.5)
        return mixed

    @staticmethod
    def _basis_dynamics_matrix(weights, basis):
        """
        Mix learned dense basis matrices into bounded near-identity dynamics.

        Args:
            weights: [..., horizon, num_bases] context-dependent coefficients
            basis: [num_bases, dim, dim] learned dense basis matrices
            identity_scale: coefficient on identity matrix for stable initialization
            dynamics_scale: bound for learned dense residual entries
        Returns:
            [..., horizon, dim, dim] dense dynamics matrices
        """
        residual = SSMWorldModel._basis_matrix(weights, basis)
        return residual

    @staticmethod
    def _psd_from_raw_factor(raw):
        """
        Convert unconstrained lower-triangular factors to PSD matrices.

        Args:
            raw: [..., dim, dim] unconstrained factor parameters
        Returns:
            [..., dim, dim] PSD matrices L @ L.T
        """
        L_raw = torch.tril(raw)
        diag_raw = torch.diagonal(L_raw, dim1=-2, dim2=-1)
        diag_pos = F.relu(diag_raw)
        L = L_raw - torch.diag_embed(diag_raw) + torch.diag_embed(diag_pos)
        return L @ L.transpose(-1, -2)

    @staticmethod
    def _pd_from_raw_factor(raw, eps=1e-6):
        """Convert raw factors to positive definite matrices."""
        psd = SSMWorldModel._psd_from_raw_factor(raw)
        return psd

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def next(self, z, a, A, B):
        """
        Predict next latent: z' = A*z + B*u

        Args:
            z: [batch, latent_dim]
            a: [batch, act_dim]
            A: [batch, latent_dim, latent_dim]
            B: [batch, latent_dim, act_dim]
        Returns:
            z': [batch, latent_dim]
        """
        Az = torch.bmm(A, z.unsqueeze(-1)).squeeze(-1)
        Bu = torch.bmm(B, a.unsqueeze(-1)).squeeze(-1)
        return Az + Bu

    # ------------------------------------------------------------------
    # Reward (quadratic in z and action)
    # ------------------------------------------------------------------
    def reward_head_parameters(self):
        """Return the trainable parameters of the quadratic reward head."""
        return (
            list(self._reward_param_net.parameters())
            + [self._reward_b]
        )

    def reward_params(self, encoder_in=None, batch_size=None, horizon=None,
                      device=None, dtype=None):
        """
        Return context-conditioned reward parameters.

        The reward represents r = X^T Q X + q^T X + b for X = [z, a].
        """

        batch_size = encoder_in.shape[0]
        horizon = self.prediction_horizon
        params = self._reward_param_net(encoder_in)
        Q_raw, q = params.split(
            [self._reward_param_split, self.quadratic_dim], dim=-1)
        Q_raw = Q_raw.view(
            batch_size, 1, self.quadratic_dim, self.quadratic_dim)
        Q = self._pd_from_raw_factor(Q_raw, eps=self.quadratic_pd_eps)
        Q = Q.expand(batch_size, horizon, -1, -1)
        q = q.view(batch_size, 1, self.quadratic_dim)
        q = q.expand(batch_size, horizon, -1)
        b = self._reward_b.to(device=encoder_in.device, dtype=encoder_in.dtype)
        b = b.view(1, 1, 1).expand(batch_size, horizon, -1)
        return Q, q, b

    def reward(self, z, a, Q=None, q=None, b=None):
        """
        Compute r(z, a) = X^T Q X + q^T X + b for X = [z, a].

        Args:
            z:      [batch, latent_dim]
            a:      [batch, act_dim]
            Q:      [batch, latent_dim + act_dim, latent_dim + act_dim]
            q:      [batch, latent_dim + act_dim]
            b:      [batch, 1]
        Returns:
            r: [batch, 1]
        """
        if Q is None:
            Q, q, b = self.reward_params(
                encoder_in=None,
                batch_size=z.shape[0],
                horizon=1,
                device=z.device,
                dtype=z.dtype,
            )
            Q, q, b = Q[:, 0], q[:, 0], b[:, 0]

        x = torch.cat([z, a], dim=-1)
        quad = torch.bmm(x.unsqueeze(1), torch.bmm(Q, x.unsqueeze(-1)))
        lin = (q * x).sum(dim=-1, keepdim=True)
        return -quad.squeeze(-1) + lin + b

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    def pi(self, z, deterministic=False, return_log_prob=False):
        """
        SAC-style stochastic policy over latent states.

        Samples action via reparameterisation: a = tanh(mean + std * eps).
        Log-probability accounts for the tanh squashing:
            log pi(a|z) = sum(log N(u; mean, std) - log(1 - tanh(u)^2))

        Args:
            z:               [..., latent_dim]
            deterministic:   if True, return tanh(mean) (eval / MPC use)
            return_log_prob: if True, also return log pi(a|z)
        Returns:
            action              [..., act_dim]           (always)
            log_prob (optional) [..., 1]
        """
        mean, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.zeros_like(mean) if deterministic else torch.randn_like(mean)

        log_prob = math.gaussian_logprob(eps, log_std)
        action = mean + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)
        if deterministic:
            action = mean
        if return_log_prob:
            return action, log_prob
        return action

    def get_pi_std(self, z):
        """Helper to get policy std (for exploration diagnostics)."""
        _, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        return log_std.exp()

    # ------------------------------------------------------------------
    # Quadratic Q-Function for MPC arrival value
    # ------------------------------------------------------------------
    def arrival_head_parameters(self):
        """Return the trainable parameters of the quadratic arrival Q head."""
        return (
            list(self._arrival_param_nets.parameters())
            + [self._arrival_Q_basis_raw, self._arrival_q_basis]
            + [self._arrival_b]
        )

    def arrival_target_head_parameters(self):
        """Return target-network parameters of the quadratic arrival Q head."""
        return (
            list(self._arrival_param_nets_target.parameters())
            + [self._arrival_Q_basis_raw_target, self._arrival_q_basis_target]
            + [self._arrival_b_target]
        )

    def arrival_Q_params(self, encoder_in, target=False, return_type='first'):
        """
        Return quadratic arrival Q parameters.

        The arrival Q-function represents:
            Q(z, a) = X^T Q X + q^T X + b, where X = [z, a].

        Args:
            encoder_in: [batch, ctx_dim]
            target:     whether to use target parameters
            return_type: 'first' returns one head for MPC parameter extraction;
                         'all' returns every ensemble head.
        Returns:
            If return_type == 'first':
                Q: [batch, latent_dim + act_dim, latent_dim + act_dim]
                q: [batch, latent_dim + act_dim]
                b: [batch, 1]
            If return_type == 'all':
                same tensors with leading [num_arrival_q, batch, ...].
        """
        if return_type is None:
            return_type = 'first'
        assert return_type in {'first', 'all'}
        batch_size = encoder_in.shape[0]
        if target:
            param_nets = self._arrival_param_nets_target
            Q_basis_raw = self._arrival_Q_basis_raw_target
            q_basis = self._arrival_q_basis_target
            b = self._arrival_b_target
        else:
            param_nets = self._arrival_param_nets
            Q_basis_raw = self._arrival_Q_basis_raw
            q_basis = self._arrival_q_basis
            b = self._arrival_b

        params = torch.stack([net(encoder_in) for net in param_nets], dim=1)
        Q_weights, q_weights = params.split(
            [self._arrival_Q_num_bases, self._arrival_q_num_bases], dim=-1)
        Q_weights = torch.softmax(Q_weights, dim=-1)
        q_weights = torch.softmax(q_weights, dim=-1)

        Q_basis_raw = Q_basis_raw.to(device=encoder_in.device, dtype=encoder_in.dtype)
        q_basis = q_basis.to(device=encoder_in.device, dtype=encoder_in.dtype)
        Q_basis = self._pd_from_raw_factor(
            Q_basis_raw, eps=self.quadratic_pd_eps)
        Q = torch.einsum('bek,ekij->beij', Q_weights, Q_basis)
        q = torch.einsum('bek,ekd->bed', q_weights, q_basis)
        b = b.to(device=encoder_in.device, dtype=encoder_in.dtype)
        if return_type == 'first':
            Q = Q[:, 0]
            q = q[:, 0]
            b = b[0].view(1, 1).expand(batch_size, -1)
            return Q, q, b
        Q = Q.permute(1, 0, 2, 3).contiguous()
        q = q.permute(1, 0, 2).contiguous()
        b = b.view(self.num_arrival_q, 1, 1).expand(-1, batch_size, -1)
        return Q, q, b

    def arrival_Q_value(self, z, a, encoder_in, target=False, return_type='min'):
        """
        Evaluate the quadratic arrival Q-function.

        Args:
            z:          [batch, latent_dim]
            a:          [batch, act_dim]
            encoder_in: [batch, ctx_dim]
            target:     whether to use the target network
            return_type: 'min', 'avg', or 'all' over ensemble heads.
        Returns:
            If 'min'/'avg': arrival Q value [batch, 1].
            If 'all':       arrival Q values [num_arrival_q, batch, 1].
        """
        if return_type is None:
            return_type = 'min'
        assert return_type in {'min', 'avg', 'all'}
        Q, q, b = self.arrival_Q_params(
            encoder_in, target=target, return_type='all')
        x = torch.cat([z, a], dim=-1)
        quad = torch.einsum('bd,ebdf,bf->eb', x, Q, x).unsqueeze(-1)
        lin = torch.einsum('ebd,bd->eb', q, x).unsqueeze(-1)
        value = -quad + lin + b
        if return_type == 'all':
            return value

        Q1, Q2 = value[np.random.choice(self.cfg.num_q, 2, replace=False)]
        return torch.min(Q1, Q2) if return_type == "min" else (Q1 + Q2) / 2
        if return_type == 'avg':
            return value.mean(dim=0)
        return value.min(dim=0).values


    # ------------------------------------------------------------------
    # Gradient control helpers
    # ------------------------------------------------------------------
    def track_critic_grad(self, mode=True):
        """Enable / disable gradients for arrival-Q parameters."""
        for p in self.arrival_head_parameters():
            p.requires_grad_(mode)

    # ------------------------------------------------------------------
    # Soft target updates
    # ------------------------------------------------------------------
    def soft_update_targets(self, tau=None):
        """Polyak-average update of target encoder and arrival critic."""
        if tau is None:
            tau = self.cfg.tau
        with torch.no_grad():
            for p_tgt, p in zip(self._arrival_param_nets_target.parameters(),
                                self._arrival_param_nets.parameters()):
                p_tgt.data.lerp_(p.data, tau)
            self._arrival_Q_basis_raw_target.data.lerp_(
                self._arrival_Q_basis_raw.data, tau)
            self._arrival_q_basis_target.data.lerp_(
                self._arrival_q_basis.data, tau)
            self._arrival_b_target.data.lerp_(self._arrival_b.data, tau)
