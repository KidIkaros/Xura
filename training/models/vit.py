"""Vision Transformer (X-Encoder) — loads frozen V-JEPA 2 ViT-L weights.

This module wraps timm's ViT to match the Rust VisionEncoder interface.
The ViT is ALWAYS frozen during training — it provides pre-trained visual
features that the Mamba-3 predictor learns to process.

Weight key mapping:
  NOT exported to safetensors (frozen, loaded from Meta checkpoint).
  The Rust side loads ViT weights separately via load_vit_weights().
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class VitConfig:
    """Mirrors kore-vljepa's VitConfig."""
    patch_size: int = 14
    image_size: int = 224
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 24
    d_ff: int = 4096
    in_channels: int = 3

    @classmethod
    def vjepa2_vit_l(cls) -> "VitConfig":
        return cls()

    @classmethod
    def tiny(cls) -> "VitConfig":
        return cls(
            patch_size=4, image_size=16, d_model=32,
            n_heads=4, n_layers=2, d_ff=64,
        )

    @property
    def num_patches(self) -> int:
        grid = self.image_size // self.patch_size
        return grid * grid


class VisionEncoder(nn.Module):
    """Frozen Vision Transformer X-Encoder.

    For training, this loads V-JEPA 2 ViT-L weights from a checkpoint
    and keeps all parameters frozen. Output is patch-level features:
      (batch, num_patches, d_model)

    Mirrors: kore-vljepa's VisionEncoder.
    """

    def __init__(self, config: VitConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            config.in_channels, config.d_model,
            kernel_size=config.patch_size, stride=config.patch_size,
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.num_patches, config.d_model) * 0.02
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to patch features.

        Args:
            images: (batch, channels, height, width)

        Returns:
            (batch, num_patches, d_model)
        """
        # Patch embedding: (B, C, H, W) → (B, d_model, grid, grid) → (B, N, d_model)
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer
        x = self.encoder(x)
        return self.norm(x)

    @classmethod
    def from_vjepa2_checkpoint(cls, checkpoint_path: str, config: VitConfig | None = None) -> "VisionEncoder":
        """Load V-JEPA 2 ViT-L weights from Meta checkpoint.

        Args:
            checkpoint_path: Path to V-JEPA 2 .pth or .safetensors file.
            config: Override config (defaults to vjepa2_vit_l).

        Returns:
            Frozen VisionEncoder.
        """
        if config is None:
            config = VitConfig.vjepa2_vit_l()

        model = cls(config)

        # Load checkpoint (adapt key names as needed for Meta's format)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if "encoder" in state_dict:
            state_dict = state_dict["encoder"]

        # Try to load, allowing missing/unexpected keys for flexibility
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[VisionEncoder] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[VisionEncoder] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

        # Freeze all parameters
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        return model
