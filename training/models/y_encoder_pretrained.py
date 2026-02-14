"""Pretrained Text Encoder (Y-Encoder) — uses a frozen sentence embedding model.

Replaces the random-init Mamba3TextEncoder with a pretrained model that
provides meaningful target embeddings for InfoNCE training from the start.

The VL-JEPA paper (arXiv:2512.10942) uses EmbeddingGemma-300M as the
Y-Encoder. We use all-MiniLM-L6-v2 (22M params) as a T4-friendly
alternative, with a trainable linear projection to the shared 1536-dim space.

Weight key mapping:
  y_encoder.projection.weight  → y_encoder.projection.weight
  y_encoder.projection.bias    → y_encoder.projection.bias
  y_encoder.backbone.*         → (frozen, not exported)
"""

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


class PretrainedTextEncoder(nn.Module):
    """Y-Encoder using a pretrained sentence embedding model.

    Architecture:
      raw_text → tokenizer → frozen backbone → mean pool → projection → L2 norm

    The backbone is frozen by default. Only the projection head is trained.
    The VL-JEPA paper recommends a LR multiplier of ×0.05 on Y-Encoder params;
    with a frozen backbone, this only applies to the projection.
    """

    # Default: all-MiniLM-L6-v2 (384-dim, 22M params, fast on T4)
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        embed_dim: int = 1536,
        max_length: int = 512,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel

        self.model_name = model_name
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.freeze_backbone = freeze_backbone

        # Load pretrained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        base_dim = self.backbone.config.hidden_size  # 384 for MiniLM

        # Trainable projection to shared embedding space
        self.projection = nn.Linear(base_dim, embed_dim)

        # Freeze backbone if requested
        if freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode raw text strings to the shared embedding space.

        Args:
            texts: List of raw text strings (batch_size,)

        Returns:
            (batch, embed_dim) — L2-normalized target embedding
        """
        device = self.projection.weight.device

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Encode through backbone
        ctx = torch.no_grad() if self.freeze_backbone else nullcontext()
        with ctx:
            outputs = self.backbone(**encoded)

        # Mean pooling over non-padding tokens (per VL-JEPA paper §3.1)
        token_embeds = outputs.last_hidden_state  # (B, L, base_dim)
        mask = encoded["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
        pooled = (token_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # Project + normalize
        projected = self.projection(pooled)  # (B, embed_dim)
        return F.normalize(projected, p=2, dim=-1)
