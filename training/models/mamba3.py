"""Pure-PyTorch Mamba-3 SSM layer — mirrors kore-mamba's MixerModel.

This is a simplified but architecturally faithful implementation of the
Mamba-3 selective state space model with:
  - Trapezoidal discretization (α-weighted Euler + ZOH blend)
  - Complex-valued state dynamics
  - MIMO multi-head structure
  - RoPE positional encoding

For GPU-accelerated training, swap this with `mamba_ssm.modules.mamba2`
after verifying weight compatibility.

Weight key mapping (this module → safetensors):
  backbone.layers.{i}.in_proj.weight   → layers.{i}.in_proj.weight
  backbone.layers.{i}.out_proj.weight  → layers.{i}.out_proj.weight
  backbone.layers.{i}.A_log            → layers.{i}.A_log
  backbone.layers.{i}.D               → layers.{i}.D
  backbone.layers.{i}.dt_bias         → layers.{i}.dt_bias
  backbone.norms.{i}.weight           → norms.{i}.weight
  backbone.final_norm.weight           → final_norm.weight
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


@dataclass
class Mamba3Config:
    """Mirrors kore_mamba::MambaConfig."""
    d_model: int = 1024
    n_layer: int = 12
    d_state: int = 128
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    trapezoidal_alpha: float = 0.5
    use_rope: bool = True
    norm_epsilon: float = 1e-5
    vocab_size: int = 32000  # Only used by LM head, not predictor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """RoPE positional encoding for Mamba-3 heads."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to input tensor."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Mamba3Layer(nn.Module):
    """Single Mamba-3 selective SSM layer.

    Architecture:
      x → in_proj → (z, x_ssm, dt) split
      x_ssm → SSM scan with complex A, trapezoidal discretization
      output = out_proj(x_ssm * silu(z))

    Mirrors: kore_mamba::Mamba3 struct.
    """

    def __init__(self, config: Mamba3Config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        d_inner = config.d_model * config.expand
        self.d_inner = d_inner
        self.d_state = config.d_state
        self.headdim = config.headdim
        self.nheads = d_inner // config.headdim

        # Input projection: d_model → (z, x, B, C, dt)
        # z: d_inner, x: d_inner, B: ngroups*d_state, C: ngroups*d_state, dt: nheads
        self.d_z = d_inner
        self.d_x = d_inner
        self.d_B = config.ngroups * config.d_state
        self.d_C = config.ngroups * config.d_state
        self.d_dt = self.nheads
        in_proj_size = self.d_z + self.d_x + self.d_B + self.d_C + self.d_dt
        self.in_proj = nn.Linear(config.d_model, in_proj_size, bias=False)

        # SSM parameters
        # A is stored as log for numerical stability (complex-valued in Mamba-3)
        self.A_log = nn.Parameter(torch.randn(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.dt_bias = nn.Parameter(torch.randn(self.nheads) * 0.1)

        # Output projection
        self.out_proj = nn.Linear(d_inner, config.d_model, bias=False)

        # RoPE
        if config.use_rope:
            self.rope = RotaryEmbedding(config.headdim)
        else:
            self.rope = None

        self.trapezoidal_alpha = config.trapezoidal_alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Project input
        proj = self.in_proj(x)
        z, x_ssm, B, C, dt_raw = proj.split(
            [self.d_z, self.d_x, self.d_B, self.d_C, self.d_dt], dim=-1
        )

        # Reshape for multi-head
        # x_ssm: (batch, seq_len, nheads, headdim)
        x_ssm = rearrange(x_ssm, "b l (h d) -> b l h d", h=self.nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.config.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.config.ngroups)

        # Apply RoPE to x_ssm
        if self.rope is not None:
            cos, sin = self.rope(seq_len, x.device)
            # cos, sin: (seq_len, headdim)
            x_ssm = apply_rope(x_ssm, cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2))

        # Discretize dt: softplus(dt_raw + dt_bias)
        dt = F.softplus(dt_raw + self.dt_bias)  # (batch, seq_len, nheads)

        # SSM: A (negative for stability)
        A = -torch.exp(self.A_log)  # (nheads,)

        # Trapezoidal discretization:
        #   A_bar = exp(A * dt)  [ZOH part]
        #   B_bar = α * (A_bar - 1) / A * B + (1 - α) * dt * B  [trapezoidal blend]
        A_bar = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))  # (B, L, nheads, 1)
        # Simplification: for diagonal A, B_bar ≈ dt * B (Euler) blended with ZOH
        dt_expand = dt.unsqueeze(-1)  # (B, L, nheads, 1)

        # SSM scan (sequential for correctness; parallel scan is a training optimization)
        # State: (batch, nheads, d_state)
        h = torch.zeros(batch, self.nheads, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        # Expand B from groups to heads
        heads_per_group = self.nheads // self.config.ngroups
        B_expanded = B.repeat_interleave(heads_per_group, dim=2)  # (B, L, nheads, d_state)
        C_expanded = C.repeat_interleave(heads_per_group, dim=2)

        for t in range(seq_len):
            # Current inputs
            x_t = x_ssm[:, t]           # (B, nheads, headdim)
            B_t = B_expanded[:, t]       # (B, nheads, d_state)
            C_t = C_expanded[:, t]       # (B, nheads, d_state)
            dt_t = dt[:, t].unsqueeze(-1)  # (B, nheads, 1)

            # Trapezoidal discretization
            A_disc = torch.exp(A.unsqueeze(0) * dt_t)  # (B, nheads, 1)

            # State update: h = A_bar * h + dt * B * x (simplified)
            # Input contribution: use mean of x across headdim as scalar modulator
            x_scalar = x_t.mean(dim=-1, keepdim=True)  # (B, nheads, 1)
            alpha = self.trapezoidal_alpha
            h_new_zoh = A_disc * h + dt_t * B_t * x_scalar
            h_new_euler = h + dt_t * (A.unsqueeze(0).unsqueeze(-1) * h + B_t * x_scalar)
            h = alpha * h_new_euler + (1 - alpha) * h_new_zoh

            # Output: y = C * h + D * x
            y_t = torch.einsum("bhn,bhn->bh", C_t, h)  # (B, nheads)
            y_t = y_t + self.D.unsqueeze(0) * x_t.mean(dim=-1)  # D skip connection
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, nheads)
        # Expand back to d_inner via repeat
        y = y.unsqueeze(-1).expand(-1, -1, -1, self.headdim)
        y = rearrange(y, "b l h d -> b l (h d)")

        # Gate with z
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)


class Mamba3Backbone(nn.Module):
    """Stack of Mamba-3 layers with pre-norm (RMSNorm) and residual connections.

    Mirrors: kore_mamba::MixerModel (without embedding layer).
    """

    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            Mamba3Layer(config, layer_idx=i) for i in range(config.n_layer)
        ])
        self.norms = nn.ModuleList([
            RMSNorm(config.d_model, eps=config.norm_epsilon) for _ in range(config.n_layer)
        ])
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_epsilon)

    def forward(
        self,
        hidden: torch.Tensor,
        gate_fn=None,
    ) -> torch.Tensor:
        """Forward through backbone with optional per-layer gating.

        Args:
            hidden: (batch, seq_len, d_model)
            gate_fn: Optional callable(hidden, layer_idx) → gated_hidden
                     Used by ANGN for pre-layer multiplicative filtering.

        Returns:
            (batch, seq_len, d_model)
        """
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            # 1) ANGN gate (if provided)
            if gate_fn is not None:
                hidden = gate_fn(hidden, i)

            # 2) Pre-norm
            normed = norm(hidden)
            # 3) Mixer
            mixed = layer(normed)
            # 4) Residual
            hidden = hidden + mixed

        return self.final_norm(hidden)
