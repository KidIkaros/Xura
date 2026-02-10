"""Full Mamba3-JEPA model — PyTorch mirror of kore-vljepa's vljepa.rs.

Combines all components into a single module for training and inference.
This is the top-level model used by the training scripts.

Weight key mapping:
  All sub-module keys are prefixed with their component name:
    x_encoder.*   → (frozen, not exported)
    predictor.*   → predictor.*
    y_encoder.*   → y_encoder.*
    y_decoder.*   → y_decoder.*
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import VisionEncoder, VitConfig
from .predictor import Mamba3Predictor
from .angn import ANGNConfig
from .y_encoder import Mamba3TextEncoder
from .y_decoder import Mamba3Decoder


class Mamba3Jepa(nn.Module):
    """Full Mamba3-JEPA: X-Encoder + Predictor + Y-Encoder + Y-Decoder.

    Training phases:
      Phase 1 (JEPA): Train predictor + y_encoder jointly via InfoNCE.
                       x_encoder frozen. y_decoder disabled.
      Phase 2 (Decoder): Train y_decoder via cross-entropy.
                          predictor + y_encoder frozen.
      Phase 3 (Agent): Fine-tune recursion + tools.
                        Core model frozen or LoRA.

    Mirrors: kore-vljepa's Mamba3Jepa.
    """

    def __init__(
        self,
        x_encoder: VisionEncoder,
        predictor: Mamba3Predictor,
        y_encoder: Mamba3TextEncoder,
        y_decoder: Mamba3Decoder,
        shared_embed_dim: int = 1536,
    ):
        super().__init__()
        self.x_encoder = x_encoder
        self.predictor = predictor
        self.y_encoder = y_encoder
        self.y_decoder = y_decoder
        self.shared_embed_dim = shared_embed_dim

    def forward_jepa(
        self,
        images: torch.Tensor,
        query_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        temperature: float = 0.07,
    ) -> dict[str, torch.Tensor]:
        """Phase 1: JEPA forward pass with InfoNCE loss.

        Args:
            images: (batch, C, H, W) input images
            query_tokens: (batch, n_qry, query_embed_dim) query embeddings
            target_tokens: (batch, seq_len) target text token IDs
            temperature: InfoNCE temperature

        Returns:
            Dict with 'loss', 'pred_embed', 'target_embed', 'accuracy'
        """
        batch = images.shape[0]

        # X-Encoder (frozen)
        with torch.no_grad():
            visual_tokens = self.x_encoder(images)  # (B, N, vision_dim)

        # Predictor
        pred_embed = self.predictor(visual_tokens, query_tokens)  # (B, embed_dim)

        # Y-Encoder (target)
        target_embed = self.y_encoder(target_tokens)  # (B, embed_dim)

        # InfoNCE loss
        # Similarity matrix: (B, B)
        logits = torch.mm(pred_embed, target_embed.t()) / temperature
        labels = torch.arange(batch, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        # Accuracy (top-1)
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean()

        return {
            "loss": loss,
            "pred_embed": pred_embed,
            "target_embed": target_embed,
            "accuracy": accuracy,
        }

    def forward_jepa_text_only(
        self,
        query_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        temperature: float = 0.07,
    ) -> dict[str, torch.Tensor]:
        """Phase 1 text-only variant: skip ViT, use query-only predictor path.

        Args:
            query_tokens: (batch, n_qry, query_embed_dim)
            target_tokens: (batch, seq_len)
            temperature: InfoNCE temperature

        Returns:
            Dict with 'loss', 'pred_embed', 'target_embed', 'accuracy'
        """
        batch = query_tokens.shape[0]

        pred_embed = self.predictor.forward_text_only(query_tokens)
        target_embed = self.y_encoder(target_tokens)

        logits = torch.mm(pred_embed, target_embed.t()) / temperature
        labels = torch.arange(batch, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean()

        return {
            "loss": loss,
            "pred_embed": pred_embed,
            "target_embed": target_embed,
            "accuracy": accuracy,
        }

    def forward_decoder(
        self,
        images: torch.Tensor,
        query_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Phase 2: Decoder training forward pass.

        Predictor is frozen. Decoder trains on cross-entropy.

        Args:
            images: (batch, C, H, W)
            query_tokens: (batch, n_qry, query_embed_dim)
            target_tokens: (batch, seq_len) ground-truth token IDs

        Returns:
            Dict with 'loss', 'logits'
        """
        # Get predicted embedding (frozen)
        with torch.no_grad():
            visual_tokens = self.x_encoder(images)
            pred_embed = self.predictor(visual_tokens, query_tokens)

        # Decoder: predicted embedding → logits
        # Shift tokens for teacher forcing: input = tokens[:-1], target = tokens[1:]
        input_tokens = target_tokens[:, :-1]
        target_labels = target_tokens[:, 1:]

        logits = self.y_decoder(pred_embed, input_tokens)  # (B, seq_len-1, vocab_size)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_labels.reshape(-1),
            ignore_index=-100,
        )

        return {"loss": loss, "logits": logits}

    @classmethod
    def small(cls, angn_config: ANGNConfig | None = None) -> "Mamba3Jepa":
        """Build small preset matching Rust Mamba3JepaConfig::small()."""
        return cls(
            x_encoder=VisionEncoder(VitConfig.vjepa2_vit_l()),
            predictor=Mamba3Predictor(
                d_model=1024, n_layers=12, d_state=128,
                expand=2, headdim=64, embed_dim=1536,
                vision_dim=1024, query_embed_dim=1024,
                angn_config=angn_config,
            ),
            y_encoder=Mamba3TextEncoder.small(),
            y_decoder=Mamba3Decoder.small(),
            shared_embed_dim=1536,
        )

    @classmethod
    def tiny(cls) -> "Mamba3Jepa":
        """Build tiny preset for testing."""
        return cls(
            x_encoder=VisionEncoder(VitConfig.tiny()),
            predictor=Mamba3Predictor(
                d_model=64, n_layers=2, d_state=16,
                expand=2, headdim=16, embed_dim=32,
                vision_dim=32, query_embed_dim=32,
            ),
            y_encoder=Mamba3TextEncoder.tiny(),
            y_decoder=Mamba3Decoder.tiny(),
            shared_embed_dim=32,
        )
