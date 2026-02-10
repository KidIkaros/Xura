"""Adaptive Neural Gating Network — PyTorch mirror of kore-vljepa's angn.rs.

Multiplicative input gate for Mamba-3 backbone layers:
  gate = σ(W_gate · h + b_gate)
  h_gated = h ⊙ gate

Weight key mapping:
  angn.gates.{i}.weight → angn.gates.{i}.weight
  angn.gates.{i}.bias   → angn.gates.{i}.bias
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ANGNConfig:
    """Mirrors kore-vljepa's ANGNConfig."""
    enabled: bool = False
    d_model: int = 64
    n_gates: int = 0
    ema_smoothing: float = 0.1
    context_reset_threshold: float = 0.2
    bias_init: float = 1.0

    @classmethod
    def small(cls) -> "ANGNConfig":
        return cls(
            enabled=True,
            d_model=1024,
            n_gates=12,
            ema_smoothing=0.1,
            context_reset_threshold=0.15,
            bias_init=1.5,
        )

    @classmethod
    def tiny(cls) -> "ANGNConfig":
        return cls(
            enabled=True,
            d_model=64,
            n_gates=2,
            ema_smoothing=0.5,
            context_reset_threshold=0.2,
            bias_init=1.0,
        )


class LayerGate(nn.Module):
    """Single learned gate for one backbone layer.

    Computes: gate = σ(w · h + b), output = h ⊙ gate

    Weights are zero-initialized so bias controls initial gate state.
    Mirrors: angn.rs LayerGate.
    """

    def __init__(self, d_model: int, bias_init: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model))
        self.bias = nn.Parameter(torch.full((d_model,), bias_init))

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply multiplicative gating.

        Args:
            hidden: (batch, seq_len, d_model)

        Returns:
            (gated_hidden, gate_activation) where gate_activation is (batch, seq_len, d_model)
        """
        gate = torch.sigmoid(self.weight * hidden + self.bias)
        return hidden * gate, gate


class AdaptiveNeuralGate(nn.Module):
    """Adaptive Neural Gating Network — per-layer multiplicative input filter.

    Contains one LayerGate per backbone layer. Tracks EMA of gate activation
    and detects context resets.

    Mirrors: angn.rs AdaptiveNeuralGate.
    """

    def __init__(self, config: ANGNConfig, n_layers: int):
        super().__init__()
        self.config = config

        n_gates = n_layers if config.n_gates == 0 or config.n_gates > n_layers else config.n_gates
        self.gates = nn.ModuleList([
            LayerGate(config.d_model, config.bias_init) for _ in range(n_gates)
        ])

        # EMA tracking (not a parameter — inference-only state)
        self.register_buffer("ema_activation", torch.ones(1))
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))

    def gate_layer(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply gating for a specific backbone layer.

        Args:
            hidden: (batch, seq_len, d_model)
            layer_idx: 0-indexed backbone layer

        Returns:
            Gated hidden states, same shape.
        """
        if not self.config.enabled or layer_idx >= len(self.gates):
            return hidden

        gated, gate_activation = self.gates[layer_idx](hidden)

        # Update EMA on last gated layer (only during eval/inference)
        if not self.training and layer_idx == len(self.gates) - 1:
            avg = gate_activation.mean().item()
            alpha = self.config.ema_smoothing
            self.ema_activation.fill_(alpha * avg + (1 - alpha) * self.ema_activation.item())
            self.step_count += 1

        return gated

    def get_gate_fn(self):
        """Return a callable for use with Mamba3Backbone.forward(gate_fn=...)."""
        def _gate(hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
            return self.gate_layer(hidden, layer_idx)
        return _gate

    @property
    def context_reset_detected(self) -> bool:
        """Whether avg gate activation dropped below threshold."""
        return self.ema_activation.item() < self.config.context_reset_threshold

    def reset_ema(self):
        """Reset EMA tracking state."""
        self.ema_activation.fill_(1.0)
        self.step_count.zero_()

    def gate_regularization_loss(self) -> torch.Tensor:
        """Auxiliary loss to encourage sparse gating.

        Returns L1 penalty on gate biases — encourages gates to close
        on irrelevant layers, reducing compute and noise.
        """
        loss = torch.tensor(0.0, device=self.gates[0].bias.device)
        for gate in self.gates:
            # Penalize gates that are always open (bias >> 0)
            loss = loss + torch.sigmoid(gate.bias).mean()
        return loss / len(self.gates)
