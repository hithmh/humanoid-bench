"""
SSM World Model for SSM-RL.
Implements a state-space model with:
  - Variational encoder (mean + log_sigma)
  - Transformer-based temporal context encoder
  - Diagonal A / dense B linear dynamics: z' = diag(A)*z + B*u
  - Quadratic reward: z^T Q z + q^T z + u^T R u + r^T u + b
    with full PSD Q and R matrices
  - Scalar Q-function conditioned on latent state, action, and context
  - Raw-observation policy network (tanh-squashed)

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
    * Reward is a learned quadratic form over latent state and action.
    * Critic is a scalar MLP over (latent state, action, context).
    * The policy consumes raw observations directly.
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

        prediction_horizon = getattr(cfg, 'horizon', 3)

        self.latent_dim = latent_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.history_horizon = history_horizon
        self.num_ensembles = num_ensembles
        self.prediction_horizon = prediction_horizon

        # ---- Deterministic encoder: obs → latent (for world model) ----
        self._encoder_mean = _mlp(state_dim, encoder_hidden, latent_dim)
        self._encoder_mean_target = deepcopy(self._encoder_mean)
        for p in self._encoder_mean_target.parameters():
            p.requires_grad_(False)

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
        self._A_net = _mlp(ctx_dim, encoder_hidden, latent_dim * prediction_horizon)
        self._B_net = _mlp(ctx_dim, encoder_hidden, latent_dim * act_dim)

        # ---- Quadratic reward heads (state part) ----
        # Each outputs lower-triangular factors for a sequence of full PSD matrices.
        self._Q_net = _mlp(ctx_dim, encoder_hidden, latent_dim * latent_dim * prediction_horizon)
        self._q_net = _mlp(ctx_dim, encoder_hidden, latent_dim * prediction_horizon)
        self._b = nn.Parameter(torch.zeros(1))

        # ---- Quadratic reward heads (action part) ----
        self._R_net = _mlp(ctx_dim, encoder_hidden, act_dim * act_dim * prediction_horizon)
        self._r_net = _mlp(ctx_dim, encoder_hidden, act_dim * prediction_horizon)

        # ---- Policy (SAC-style stochastic: outputs mean + log_std) ----
        log_std_min = getattr(cfg, 'log_std_min', -5)
        log_std_max = getattr(cfg, 'log_std_max', 2)
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

        # Raw-observation policy trunk -> mean head and log_std head
        self._pi_trunk = _mlp(state_dim, policy_hidden, policy_hidden[-1])
        self._pi_mean_head = nn.Linear(policy_hidden[-1], act_dim)
        self._pi_log_std_head = nn.Linear(policy_hidden[-1], act_dim)

        # Bundle into a single module list for easy deepcopy / param access
        self._pi = nn.ModuleList([self._pi_trunk, self._pi_mean_head, self._pi_log_std_head])

        # Target copies (used by pi_target for Bellman backup)
        self._pi_target_trunk = deepcopy(self._pi_trunk).requires_grad_(False)
        self._pi_target_mean_head = deepcopy(self._pi_mean_head).requires_grad_(False)
        self._pi_target_log_std_head = deepcopy(self._pi_log_std_head).requires_grad_(False)
        self._pi_target = nn.ModuleList([
            self._pi_target_trunk, self._pi_target_mean_head, self._pi_target_log_std_head
        ])

        # ---- Single Q-function MLP ----
        # Takes (z, a, encoder_in) and outputs a single Q-value scalar
        # Input: [z, a, encoder_in] concatenated
        q_func_input_dim = latent_dim + act_dim + ctx_dim
        critic_hidden = getattr(cfg, 'critic_struct', encoder_hidden)

        self._q_func = _mlp(q_func_input_dim, critic_hidden, 1)
        self._q_func_target = deepcopy(self._q_func)
        for p in self._q_func_target.parameters():
            p.requires_grad_(False)

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
        """
        Encode observations → latent (deterministic).

        Args:
            obs:    [batch, state_dim] or [T, batch, state_dim]
            target: if True, use the target (EMA) encoder
        Returns:
            z   (same leading shape as obs, last dim = latent_dim)
        """
        encoder = self._encoder_mean_target if target else self._encoder_mean
        return encoder(obs)

    def encode_context(self, state_history, action_history, current_obs):
        """
        Run transformer on history and produce (A, B, Q_seq, q_seq, R_seq, r_seq, encoder_in) for the current step.

        Args:
            state_history:  [batch, history_horizon, state_dim]
            action_history: [batch, history_horizon, act_dim]
            current_obs:    [batch, state_dim]
        Returns:
            A_diag_seq: [batch, prediction_horizon, latent_dim]
            B:          [batch, latent_dim, act_dim]
            Q_seq:      [batch, prediction_horizon, latent_dim, latent_dim] per-step state PSD matrix
            q_seq:      [batch, prediction_horizon, latent_dim]   per-step state linear coefficient
            R_seq:      [batch, prediction_horizon, act_dim, act_dim] per-step action PSD matrix
            r_seq:      [batch, prediction_horizon, act_dim]      per-step action linear coefficient
            encoder_in: [batch, ctx_dim]  (transformer_out || current_obs)
        """
        ctx_input = torch.cat([state_history, action_history], dim=-1)
        transformer_out = self._transformer(ctx_input)  # [batch, d_model]
        encoder_in = torch.cat([transformer_out, current_obs], dim=-1)

        H = self.prediction_horizon
        A_flat = self._A_net(encoder_in)  # [batch, latent_dim * H]
        A_diag_seq = A_flat.view(-1, H, self.latent_dim)  # [batch, H, latent_dim]

        B_flat = self._B_net(encoder_in)  # [batch, latent_dim * act_dim]
        B = B_flat.view(-1, self.latent_dim, self.act_dim)

        H = self.prediction_horizon
        # Q_net and q_net output sequences of length H
        Q_flat = self._Q_net(encoder_in)  # [batch, latent_dim * latent_dim * H]
        Q_raw_seq = Q_flat.view(-1, H, self.latent_dim, self.latent_dim)
        Q_seq = self._psd_from_raw_factor(Q_raw_seq)

        q_flat = self._q_net(encoder_in)  # [batch, latent_dim * H]
        q_seq = q_flat.view(-1, H, self.latent_dim)       # [batch, H, latent_dim]

        R_flat = self._R_net(encoder_in)  # [batch, act_dim * act_dim * H]
        R_raw_seq = R_flat.view(-1, H, self.act_dim, self.act_dim)
        R_seq = self._psd_from_raw_factor(R_raw_seq)

        r_flat = self._r_net(encoder_in)  # [batch, act_dim * H]
        r_seq = r_flat.view(-1, H, self.act_dim)       # [batch, H, act_dim]

        return A_diag_seq, B, Q_seq, q_seq, R_seq, r_seq, encoder_in

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
        diag_pos = F.elu(diag_raw) + 1.0
        L = L_raw - torch.diag_embed(diag_raw) + torch.diag_embed(diag_pos)
        return L @ L.transpose(-1, -2)

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
    # Reward (quadratic in z and action)
    # ------------------------------------------------------------------
    def reward(self, z, a, Q, q, R, r_vec):
        """
        Compute quadratic reward: -(z^T Q z + q^T z + a^T R a + r^T a + b)

        Args:
            z:      [batch, latent_dim]
            a:      [batch, act_dim]
            Q:      [batch, latent_dim, latent_dim] state quadratic PSD matrix
            q:      [batch, latent_dim]   state linear coefficient
            R:      [batch, act_dim, act_dim] action quadratic PSD matrix
            r_vec:  [batch, act_dim]      action linear coefficient
        Returns:
            r: [batch, 1]
        """
        quad_z = torch.bmm(z.unsqueeze(1), torch.bmm(Q, z.unsqueeze(-1)))
        lin_z  = (q * z).sum(dim=-1, keepdim=True)
        quad_a = torch.bmm(a.unsqueeze(1), torch.bmm(R, a.unsqueeze(-1)))
        lin_a  = (r_vec * a).sum(dim=-1, keepdim=True)
        return -(quad_z.squeeze(-1) + lin_z + quad_a.squeeze(-1) + lin_a + self._b)

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    def pi(self, obs, target=False, deterministic=False, return_log_prob=False):
        """
        SAC-style stochastic policy over raw observations.

        Samples action via reparameterisation: a = tanh(mean + std * eps).
        Log-probability accounts for the tanh squashing:
            log pi(a|obs) = sum(log N(u; mean, std) - log(1 - tanh(u)^2))

        Args:
            obs:             [..., state_dim]
            target:          use target network weights
            deterministic:   if True, return tanh(mean) (eval / MPC use)
            return_log_prob: if True, also return log pi(a|obs)
        Returns:
            action              [..., act_dim]           (always)
            log_prob (optional) [..., 1]
        """
        trunk       = self._pi_target[0] if target else self._pi_trunk
        mean_head   = self._pi_target[1] if target else self._pi_mean_head
        log_std_head = self._pi_target[2] if target else self._pi_log_std_head

        h = trunk(obs)
        mean = mean_head(h)
        log_std = log_std_head(h).clamp(self._log_std_min, self._log_std_max)
        std = log_std.exp()

        if deterministic:
            action = torch.tanh(mean)
            if return_log_prob:
                # log-prob at the deterministic point
                log_prob = self._gaussian_log_prob(mean, mean, std)
                return action, log_prob
            return action

        # Reparameterised sample
        eps = torch.randn_like(std)
        u = mean + std * eps                  # pre-squash
        action = torch.tanh(u)

        if return_log_prob:
            log_prob = self._gaussian_log_prob(u, mean, std)
            return action, log_prob
        return action

    def get_pi_std(self, obs, target=False):
        """Helper to get policy std (for exploration diagnostics)."""
        trunk       = self._pi_target[0] if target else self._pi_trunk
        log_std_head = self._pi_target[2] if target else self._pi_log_std_head
        h = trunk(obs)
        log_std = log_std_head(h).clamp(self._log_std_min, self._log_std_max)
        return log_std.exp()

    @staticmethod
    def _gaussian_log_prob(u, mean, std):
        """
        Log prob of Gaussian with tanh squashing correction.
            log pi(a|obs) = sum(log N(u; mu, sigma) - log(1 - tanh(u)^2))
        Returns: [..., 1]
        """
        log_prob_gaussian = (
            -0.5 * ((u - mean) / std).pow(2)
            - std.log()
            - 0.5 * torch.tensor(2 * torch.pi).log().to(u.device)
        )
        # Tanh squashing correction (numerically stable)
        log_det_jacobian = 2.0 * (torch.log(torch.tensor(2.0, device=u.device))
                                  - u - F.softplus(-2.0 * u))
        return (log_prob_gaussian - log_det_jacobian).sum(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # Q-Function MLP (single scalar Q-value approximator)
    # ------------------------------------------------------------------
    def Q_value(self, z, a, encoder_in, target=False, return_type=None):
        """
        Compute Q-value using a single MLP: Q(z, a, encoder_in).

        Args:
            z:          [batch, latent_dim]
            a:          [batch, act_dim]
            encoder_in: [batch, ctx_dim]
            target:     whether to use target network
            return_type: unused (kept for API compatibility)
        Returns:
            q_val: [batch, 1]
        """
        q_func = self._q_func_target if target else self._q_func
        # Concatenate inputs
        x = torch.cat([z, a, encoder_in], dim=-1)  # [batch, latent_dim + act_dim + ctx_dim]
        q_val = q_func(x)  # [batch, 1]
        return q_val

    # ------------------------------------------------------------------
    # Gradient control helpers
    # ------------------------------------------------------------------
    def track_critic_grad(self, mode=True):
        """Enable / disable gradients for Q-function MLP parameters."""
        for p in self._q_func.parameters():
            p.requires_grad_(mode)

    # ------------------------------------------------------------------
    # Soft target updates
    # ------------------------------------------------------------------
    def soft_update_targets(self, tau=None):
        """Polyak-average update of target encoder, Q-function and target policy."""
        if tau is None:
            tau = self.cfg.tau
        with torch.no_grad():
            # World model encoder target
            for p_tgt, p in zip(self._encoder_mean_target.parameters(),
                                 self._encoder_mean.parameters()):
                p_tgt.data.lerp_(p.data, tau)
            # Q-function target
            for p_tgt, p in zip(self._q_func_target.parameters(), self._q_func.parameters()):
                p_tgt.data.lerp_(p.data, tau)
            # Update all three policy heads
            pi_pairs = [
                (self._pi_target_trunk,        self._pi_trunk),
                (self._pi_target_mean_head,    self._pi_mean_head),
                (self._pi_target_log_std_head, self._pi_log_std_head),
            ]
            for tgt_mod, src_mod in pi_pairs:
                for p_tgt, p in zip(tgt_mod.parameters(), src_mod.parameters()):
                    p_tgt.data.lerp_(p.data, tau)
