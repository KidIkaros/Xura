"""Mamba-3 Predictor — PyTorch mirror of kore-vljepa's predictor.rs.

Maps (visual embeddings + text query) → predicted target embedding
using a Mamba-3 backbone with optional ANGN gating.

Weight key mapping (this module → safetensors):
  predictor.vision_proj.weight  → predictor.vision_proj.weight
  predictor.vision_proj.bias    → predictor.vision_proj.bias
  predictor.query_proj.weight   → predictor.query_proj.weight
  predictor.query_proj.bias     → predictor.query_proj.bias
  predictor.backbone.*          → predictor.backbone.*
  predictor.pred_head.weight    → predictor.pred_head.weight
  predictor.pred_head.bias      → predictor.pred_head.bias
  predictor.angn.*              → predictor.angn.*
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba3 import Mamba3Backbone, Mamba3Config
from .angn import AdaptiveNeuralGate, ANGNConfig


class Mamba3Predictor(nn.Module):
    """Mamba-3 Predictor for JEPA.

    Architecture:
      visual_tokens → vision_proj → concat with query_proj(query_embeds)
      → Mamba-3 backbone (with ANGN gating) → mean pool → pred_head → L2 norm

    Mirrors: kore-vljepa's Mamba3Predictor.
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_layers: int = 12,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        trapezoidal_alpha: float = 0.5,
        embed_dim: int = 1536,
        vision_dim: int = 1024,
        query_embed_dim: int = 1024,
        angn_config: ANGNConfig | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.embed_dim = embed_dim

        # Input projections
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.query_proj = nn.Linear(query_embed_dim, d_model)

        # Mamba-3 backbone
        mamba_config = Mamba3Config(
            d_model=d_model,
            n_layer=n_layers,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            trapezoidal_alpha=trapezoidal_alpha,
        )
        self.backbone = Mamba3Backbone(mamba_config)

        # Prediction head: d_model → embed_dim
        self.pred_head = nn.Linear(d_model, embed_dim)

        # ANGN (optional)
        self.angn = None
        if angn_config is not None and angn_config.enabled:
            self.angn = AdaptiveNeuralGate(angn_config, n_layers)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        query_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            visual_tokens: (batch, n_vis, vision_dim) from X-Encoder
            query_embeds: (batch, n_qry, query_embed_dim) embedded query tokens

        Returns:
            (batch, embed_dim) — L2-normalized predicted embedding
        """
        # Project inputs to d_model
        vis = self.vision_proj(visual_tokens)   # (B, n_vis, d_model)
        qry = self.query_proj(query_embeds)     # (B, n_qry, d_model)

        # Concatenate: [visual; query]
        hidden = torch.cat([vis, qry], dim=1)   # (B, n_vis+n_qry, d_model)

        # Backbone with optional ANGN gating
        gate_fn = self.angn.get_gate_fn() if self.angn is not None else None
        hidden = self.backbone(hidden, gate_fn=gate_fn)

        # Mean pool over sequence
        pooled = hidden.mean(dim=1)             # (B, d_model)

        # Prediction head → L2 normalize
        projected = self.pred_head(pooled)      # (B, embed_dim)
        return F.normalize(projected, p=2, dim=-1)

    def forward_text_only(self, query_embeds: torch.Tensor) -> torch.Tensor:
        """Text-only forward (skips vision projection).

        Args:
            query_embeds: (batch, n_qry, query_embed_dim)

        Returns:
            (batch, embed_dim) — L2-normalized predicted embedding
        """
        qry = self.query_proj(query_embeds)

        gate_fn = self.angn.get_gate_fn() if self.angn is not None else None
        hidden = self.backbone(qry, gate_fn=gate_fn)

        pooled = hidden.mean(dim=1)
        projected = self.pred_head(pooled)
        return F.normalize(projected, p=2, dim=-1)

    @classmethod
    def from_config(cls, config: dict) -> "Mamba3Predictor":
        """Build from a config dict (matches Rust Mamba3PredictorConfig fields)."""
        angn_cfg = None
        if "angn" in config:
            angn_cfg = ANGNConfig(**config["angn"])
        return cls(
            d_model=config.get("d_model", 1024),
            n_layers=config.get("n_layers", 12),
            d_state=config.get("d_state", 128),
            expand=config.get("expand", 2),
            headdim=config.get("headdim", 64),
            ngroups=config.get("ngroups", 1),
            trapezoidal_alpha=config.get("trapezoidal_alpha", 0.5),
            embed_dim=config.get("embed_dim", 1536),
            vision_dim=config.get("vision_dim", 1024),
            query_embed_dim=config.get("query_embed_dim", 1024),
            angn_config=angn_cfg,
        )
