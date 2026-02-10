"""Mamba-3 Text Encoder (Y-Encoder) — PyTorch mirror of kore-vljepa's y_encoder.rs.

Encodes text tokens into the shared embedding space to produce JEPA
training targets. The predictor learns to match these target embeddings.

Weight key mapping:
  y_encoder.embedding.weight       → y_encoder.embedding.weight
  y_encoder.backbone.*             → y_encoder.backbone.*
  y_encoder.projection.weight      → y_encoder.projection.weight
  y_encoder.projection.bias        → y_encoder.projection.bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba3 import Mamba3Backbone, Mamba3Config


class Mamba3TextEncoder(nn.Module):
    """Y-Encoder: text → shared embedding space.

    Architecture:
      tokens → embedding → Mamba-3 backbone → mean pool → projection → L2 norm

    The Y-Encoder produces the *target* embeddings that the predictor
    learns to predict. During Phase 1 JEPA training, both the predictor
    and Y-encoder are trained jointly via InfoNCE.

    After Phase 1, the Y-encoder can optionally be frozen or used as an
    EMA target (like BYOL/DINO).

    Mirrors: kore-vljepa's Mamba3TextEncoder.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        n_layers: int = 12,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 32,
        ngroups: int = 1,
        max_seq_len: int = 512,
        embed_dim: int = 1536,
    ):
        super().__init__()

        self.d_model = d_model
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Mamba-3 backbone
        mamba_config = Mamba3Config(
            d_model=d_model,
            n_layer=n_layers,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            vocab_size=vocab_size,
        )
        self.backbone = Mamba3Backbone(mamba_config)

        # Projection to shared embedding space
        self.projection = nn.Linear(d_model, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode text tokens to shared embedding space.

        Args:
            token_ids: (batch, seq_len) integer token IDs

        Returns:
            (batch, embed_dim) — L2-normalized target embedding
        """
        # Embed tokens
        x = self.embedding(token_ids)           # (B, L, d_model)

        # Backbone
        x = self.backbone(x)                    # (B, L, d_model)

        # Mean pool over sequence
        pooled = x.mean(dim=1)                  # (B, d_model)

        # Project → L2 normalize
        projected = self.projection(pooled)     # (B, embed_dim)
        return F.normalize(projected, p=2, dim=-1)

    @classmethod
    def small(cls) -> "Mamba3TextEncoder":
        """Small preset matching Rust Mamba3TextEncoderConfig::small()."""
        return cls(
            vocab_size=32000, d_model=768, n_layers=12,
            d_state=64, expand=2, headdim=32, ngroups=1,
            max_seq_len=512, embed_dim=1536,
        )

    @classmethod
    def tiny(cls) -> "Mamba3TextEncoder":
        """Tiny preset for testing."""
        return cls(
            vocab_size=256, d_model=64, n_layers=2,
            d_state=16, expand=2, headdim=16, ngroups=1,
            max_seq_len=64, embed_dim=32,
        )
