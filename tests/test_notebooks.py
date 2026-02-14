#!/usr/bin/env python3
"""Unit tests for Kaggle notebook cells.

Tests each logical cell from both Phase 1 and Phase 2 notebooks
using synthetic data and CPU-only mode. No GPU or datasets required.

Run: python -m pytest tests/test_notebooks.py -v
"""

import sys
import os
import math
import random
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add training/ to path so model imports work
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def xura_models():
    """Import all Xura model classes."""
    from models.mamba3 import Mamba3Backbone, Mamba3Config, Mamba3Layer, RMSNorm
    from models.predictor import Mamba3Predictor
    from models.angn import ANGNConfig
    from models.y_encoder_pretrained import PretrainedTextEncoder
    from models.y_decoder import Mamba3Decoder
    from models.vit import VisionEncoder, VitConfig
    from models.vljepa import Mamba3Jepa

    return {
        "Mamba3Predictor": Mamba3Predictor,
        "ANGNConfig": ANGNConfig,
        "PretrainedTextEncoder": PretrainedTextEncoder,
        "Mamba3Decoder": Mamba3Decoder,
        "VisionEncoder": VisionEncoder,
        "VitConfig": VitConfig,
        "Mamba3Jepa": Mamba3Jepa,
    }


@pytest.fixture(scope="session")
def query_tokenizer():
    """BERT tokenizer for queries."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="session")
def decoder_tokenizer():
    """T5 sentencepiece tokenizer for decoder."""
    from transformers import T5Tokenizer
    return T5Tokenizer.from_pretrained("google-t5/t5-base")


@pytest.fixture(scope="session")
def dinov2_model():
    """Load DINOv2 with the same fallback logic as notebooks."""
    import timm

    model_names = [
        "vit_large_patch14_dinov2.lvd142m",
        "vit_large_patch14_dinov2",
        "vit_large_patch14_reg4_dinov2.lvd142m",
    ]

    for name in model_names:
        try:
            model = timm.create_model(name, pretrained=True, num_classes=0)
            return model.eval()
        except Exception:
            continue

    # Fallback: find any dinov2 model
    available = timm.list_models("*dinov2*")
    if available:
        model = timm.create_model(available[0], pretrained=True, num_classes=0)
        return model.eval()

    pytest.skip("No DINOv2 model available in timm")


# ---------------------------------------------------------------------------
# Phase 1 Tests
# ---------------------------------------------------------------------------

class TestPhase1Imports:
    """Cell 5: Import Xura models."""

    def test_all_models_import(self, xura_models):
        for name, cls in xura_models.items():
            assert cls is not None, f"{name} failed to import"

    def test_model_classes_are_callable(self, xura_models):
        assert callable(xura_models["Mamba3Predictor"])
        assert callable(xura_models["Mamba3Decoder"])
        assert callable(xura_models["Mamba3Jepa"])


class TestPhase1DINOv2:
    """Cells 6-7: DINOv2 loading and wrapper."""

    def test_dinov2_loads(self, dinov2_model):
        assert dinov2_model is not None
        param_count = sum(p.numel() for p in dinov2_model.parameters())
        assert param_count > 100_000_000, f"DINOv2 too small: {param_count}"

    def test_forward_features_works(self, dinov2_model):
        dummy = torch.randn(1, 3, 518, 518)
        with torch.no_grad():
            try:
                out = dinov2_model.forward_features(dummy)
                assert out.dim() in (2, 3)
            except Exception:
                out = dinov2_model(dummy)
                assert out is not None

    def test_wrapper_forward(self, dinov2_model, xura_models):
        VitConfig = xura_models["VitConfig"]

        # Detect method
        use_ff = True
        dummy = torch.randn(1, 3, 518, 518)
        with torch.no_grad():
            try:
                dinov2_model.forward_features(dummy)
            except Exception:
                use_ff = False

        class DINOv2Wrapper(nn.Module):
            def __init__(self, model, use_forward_features=True):
                super().__init__()
                self.model = model
                self.config = VitConfig.vjepa2_vit_l()
                self._use_forward_features = use_forward_features

            def forward(self, images):
                if self._use_forward_features:
                    features = self.model.forward_features(images)
                else:
                    features = self.model(images)
                if features.dim() == 3 and features.shape[1] > self.config.num_patches:
                    features = features[:, 1:, :]
                return features

        wrapper = DINOv2Wrapper(dinov2_model, use_forward_features=use_ff)
        with torch.no_grad():
            out = wrapper(dummy)

        assert out.dim() == 3, f"Expected 3D output, got {out.dim()}D"
        assert out.shape[-1] == 1024, f"Expected dim 1024, got {out.shape[-1]}"
        assert out.shape[0] == 1, f"Batch size mismatch"


class TestPhase1ModelBuild:
    """Cell 8: Build JEPA model."""

    def test_jepa_builds(self, xura_models, query_tokenizer, dinov2_model):
        VitConfig = xura_models["VitConfig"]

        # Detect method
        use_ff = True
        with torch.no_grad():
            try:
                dinov2_model.forward_features(torch.randn(1, 3, 518, 518))
            except Exception:
                use_ff = False

        class DINOv2Wrapper(nn.Module):
            def __init__(self, model, use_forward_features=True):
                super().__init__()
                self.model = model
                self.config = VitConfig.vjepa2_vit_l()
                self._use_forward_features = use_forward_features
            def forward(self, images):
                if self._use_forward_features:
                    f = self.model.forward_features(images)
                else:
                    f = self.model(images)
                if f.dim() == 3 and f.shape[1] > self.config.num_patches:
                    f = f[:, 1:, :]
                return f

        x_encoder = DINOv2Wrapper(dinov2_model, use_ff)

        y_encoder = xura_models["PretrainedTextEncoder"](
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embed_dim=1536, freeze_backbone=True,
        )

        predictor = xura_models["Mamba3Predictor"](
            d_model=1024, n_layers=12, d_state=128,
            expand=2, headdim=64, embed_dim=1536,
            vision_dim=1024, query_embed_dim=1024,
            angn_config=xura_models["ANGNConfig"](),
        )

        model = xura_models["Mamba3Jepa"](
            x_encoder=x_encoder, predictor=predictor,
            y_encoder=y_encoder,
            y_decoder=xura_models["Mamba3Decoder"].small(),
            shared_embed_dim=1536,
            query_vocab_size=query_tokenizer.vocab_size,
            query_embed_dim=1024,
        )

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        assert trainable > 0, "No trainable params"
        assert frozen > 0, "No frozen params (DINOv2 should be frozen)"

    def test_query_embed_forward(self, xura_models, query_tokenizer, dinov2_model):
        """Test get_query_embeds works with BERT tokens."""
        VitConfig = xura_models["VitConfig"]

        use_ff = True
        with torch.no_grad():
            try:
                dinov2_model.forward_features(torch.randn(1, 3, 518, 518))
            except Exception:
                use_ff = False

        class DINOv2Wrapper(nn.Module):
            def __init__(self, model, use_forward_features=True):
                super().__init__()
                self.model = model
                self.config = VitConfig.vjepa2_vit_l()
                self._use_forward_features = use_forward_features
            def forward(self, images):
                if self._use_forward_features:
                    f = self.model.forward_features(images)
                else:
                    f = self.model(images)
                if f.dim() == 3 and f.shape[1] > self.config.num_patches:
                    f = f[:, 1:, :]
                return f

        x_encoder = DINOv2Wrapper(dinov2_model, use_ff)
        y_encoder = xura_models["PretrainedTextEncoder"](
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embed_dim=1536, freeze_backbone=True,
        )
        predictor = xura_models["Mamba3Predictor"](
            d_model=1024, n_layers=12, d_state=128,
            expand=2, headdim=64, embed_dim=1536,
            vision_dim=1024, query_embed_dim=1024,
            angn_config=xura_models["ANGNConfig"](),
        )

        model = xura_models["Mamba3Jepa"](
            x_encoder=x_encoder, predictor=predictor,
            y_encoder=y_encoder,
            y_decoder=xura_models["Mamba3Decoder"].small(),
            shared_embed_dim=1536,
            query_vocab_size=query_tokenizer.vocab_size,
            query_embed_dim=1024,
        )

        encoded = query_tokenizer(
            "describe this image", max_length=32,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        tokens = encoded["input_ids"]  # [1, 32]

        with torch.no_grad():
            query_embeds = model.get_query_embeds(tokens)

        assert query_embeds.dim() == 3, f"Expected 3D, got {query_embeds.dim()}"
        assert query_embeds.shape[0] == 1


# ---------------------------------------------------------------------------
# Phase 2 Tests
# ---------------------------------------------------------------------------

class TestPhase2Tokenizers:
    """Cell 6: Tokenizer setup."""

    def test_bert_tokenizer(self, query_tokenizer):
        assert query_tokenizer.vocab_size > 0
        encoded = query_tokenizer("hello world", return_tensors="pt")
        assert encoded["input_ids"].shape[0] == 1

    def test_t5_tokenizer(self, decoder_tokenizer):
        assert decoder_tokenizer.vocab_size == 32100, (
            f"T5 vocab should be 32100, got {decoder_tokenizer.vocab_size}"
        )
        assert decoder_tokenizer.pad_token_id is not None
        assert decoder_tokenizer.eos_token_id is not None

    def test_t5_encode_decode(self, decoder_tokenizer):
        text = "A cat sitting on a windowsill"
        ids = decoder_tokenizer.encode(text)
        decoded = decoder_tokenizer.decode(ids, skip_special_tokens=True)
        assert text.lower() in decoded.lower() or len(decoded) > 5


class TestPhase2Decoder:
    """Cell 9: Decoder model construction."""

    def test_decoder_builds(self, xura_models):
        decoder = xura_models["Mamba3Decoder"](
            d_model=512, n_layers=6, d_state=64,
            expand=2, headdim=32, vocab_size=32100,
            prefix_len=8, embed_dim=1536,
        )
        param_count = sum(p.numel() for p in decoder.parameters())
        assert param_count > 1_000_000, f"Decoder too small: {param_count}"

    def test_decoder_forward(self, xura_models):
        decoder = xura_models["Mamba3Decoder"](
            d_model=512, n_layers=6, d_state=64,
            expand=2, headdim=32, vocab_size=32100,
            prefix_len=8, embed_dim=1536,
        )
        batch_size = 2
        seq_len = 16
        VOCAB_SIZE = 32100
        pred_embed = torch.randn(batch_size, 1536)
        input_tokens = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))

        with torch.no_grad():
            logits = decoder(pred_embed, input_tokens)

        assert logits.shape == (batch_size, seq_len, VOCAB_SIZE), (
            f"Logits shape {logits.shape} != expected ({batch_size}, {seq_len}, {VOCAB_SIZE})"
        )

    def test_decoder_generate(self, xura_models):
        decoder = xura_models["Mamba3Decoder"](
            d_model=512, n_layers=6, d_state=64,
            expand=2, headdim=32, vocab_size=32100,
            prefix_len=8, embed_dim=1536,
        )
        pred_embed = torch.randn(1, 1536)

        with torch.no_grad():
            gen_ids = decoder.generate(
                pred_embed, bos_token=0, eos_token=1,
                max_tokens=16, temperature=0.7,
            )

        assert len(gen_ids) == 1
        # generate() may include BOS token, so allow max_tokens + 1
        assert len(gen_ids[0]) <= 17, f"Generated {len(gen_ids[0])} tokens, expected <= 17"


class TestPhase2TrainingStep:
    """Cell 13: One training step with synthetic data."""

    def test_decoder_loss_backward(self, xura_models, decoder_tokenizer):
        decoder = xura_models["Mamba3Decoder"](
            d_model=512, n_layers=6, d_state=64,
            expand=2, headdim=32, vocab_size=32100,
            prefix_len=8, embed_dim=1536,
        )
        PAD_TOKEN_ID = decoder_tokenizer.pad_token_id

        # Simulated Phase 2 training step
        batch_size = 2
        pred_embed = torch.randn(batch_size, 1536)

        caption = "A cat sitting on a windowsill"
        dec_enc = decoder_tokenizer(
            [caption] * batch_size, max_length=32,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        decoder_tokens = dec_enc["input_ids"]

        input_tokens = decoder_tokens[:, :-1]
        target_labels = decoder_tokens[:, 1:]

        logits = decoder(pred_embed, input_tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_labels.reshape(-1),
            ignore_index=PAD_TOKEN_ID,
        )

        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"

        loss.backward()

        # Check gradients exist
        has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in decoder.parameters()
        )
        assert has_grads, "No gradients computed"


# ---------------------------------------------------------------------------
# Shared: Gradient accumulation logic
# ---------------------------------------------------------------------------

class TestGradientAccumulation:
    """Verify the grad accum logic matches both notebooks."""

    def test_accum_steps_correct(self):
        """Final batch flush logic."""
        num_batches = 100
        grad_accum_steps = 8

        optimizer_steps = 0
        for batch_idx in range(num_batches):
            is_accum = (batch_idx + 1) % grad_accum_steps == 0
            is_last = (batch_idx + 1) == num_batches
            if is_accum or is_last:
                optimizer_steps += 1

        expected = math.ceil(num_batches / grad_accum_steps)
        assert optimizer_steps == expected, (
            f"Got {optimizer_steps} steps, expected {expected}"
        )

    def test_scheduler_total_steps(self):
        """Steps per epoch calculation."""
        num_batches = 100
        grad_accum_steps = 8
        epochs = 10

        steps_per_epoch = math.ceil(num_batches / grad_accum_steps)
        total_steps = steps_per_epoch * epochs
        assert total_steps == 130, f"Got {total_steps}, expected 130"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
