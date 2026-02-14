#!/usr/bin/env python3
"""Kaggle Notebook: Xura Phase 1 JEPA Training on T4 GPUs.

This script is designed as a Kaggle notebook (.py format for Greptile review).
Upload to Kaggle and run with GPU accelerator enabled (T4 x2).

What it does:
  1. Installs dependencies
  2. Downloads DINOv2 ViT-L weights (public, via timm)
  3. Downloads CC3M subset via HuggingFace datasets (or uses synthetic)
  4. Runs Phase 1 JEPA pretraining with:
     - Pretrained Y-Encoder (all-MiniLM-L6-v2)
     - FP16 mixed precision
     - Gradient accumulation (effective batch 256)
  5. Logs metrics to W&B (optional)
  6. Exports best checkpoint to safetensors

Estimated time: ~2 hours for 10 epochs on 50k CC3M samples.
Estimated VRAM: ~12GB peak (fits in T4's 15GB).

Compatibility: PyTorch 2.2.x-2.5.x (uses torch.cuda.amp API)
"""

# %% [markdown]
# # Xura Phase 1: JEPA Pretraining on Kaggle T4
#
# Training the Mamba-3 predictor to match pretrained Y-Encoder targets
# via InfoNCE contrastive loss. The X-Encoder (DINOv2 ViT-L) is frozen.

# %% Install dependencies
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("Installing dependencies...")
install("torch>=2.2.0,<2.6.0")
install("torchvision>=0.17.0,<0.21.0")
install("timm>=0.9.12,<1.0.0")
install("transformers>=4.36.0,<5.0.0")
install("sentence-transformers>=2.3.0,<3.0.0")
install("safetensors>=0.4.0,<1.0.0")
install("einops>=0.7.0,<1.0.0")
install("datasets>=2.16.0,<3.0.0")
install("wandb>=0.16.0,<1.0.0")
install("tqdm>=4.66.0")
install("pillow>=10.0.0")
print("Dependencies installed.")

# %% Imports
import os
import math
import time
import json
import random
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Configuration
#
# Tuned for T4 GPU (15GB VRAM). Uses gradient accumulation to achieve
# effective batch size of 256 while fitting in memory.

# %% Config
class Config:
    # Model
    d_model = 1024          # Mamba-3 predictor hidden dim
    n_layers = 12           # Mamba-3 predictor layers
    d_state = 128           # SSM state dimension
    embed_dim = 1536        # Shared embedding space
    vision_dim = 1024       # ViT output dim (DINOv2 ViT-L)
    query_embed_dim = 1024  # Match predictor.query_proj.in_features

    # Training
    epochs = 10
    batch_size = 16         # Per-GPU batch (T4 VRAM constraint)
    grad_accum_steps = 16   # Effective batch = 16 * 16 = 256
    lr = 3e-4
    weight_decay = 0.01
    temperature = 0.07
    warmup_steps = 500
    max_seq_len = 128       # Shorter for memory efficiency on T4
    image_size = 224
    fp16 = True

    # Y-Encoder
    y_encoder_model = "sentence-transformers/all-MiniLM-L6-v2"
    y_encoder_lr_mult = 0.05

    # Data
    num_samples = 50000     # CC3M subset size
    num_workers = 2         # Kaggle has limited workers

    # Logging
    use_wandb = False       # Set True if you have W&B API key
    wandb_project = "xura-mamba3-jepa"
    log_interval = 25
    save_every = 2
    output_dir = "/kaggle/working/checkpoints"

cfg = Config()

# %% [markdown]
# ## Clone Xura Repository
#
# We clone the repo to get the model definitions.

# %% Clone repo
XURA_DIR = Path("/kaggle/working/Xura")
if not XURA_DIR.exists():
    print("Cloning Xura repository...")
    subprocess.check_call([
        "git", "clone", "--branch", "main",
        "--depth", "1",
        "https://github.com/KidIkaros/Xura.git",
        str(XURA_DIR),
    ])
    print(f"Cloned to {XURA_DIR}")
else:
    print(f"Xura already exists at {XURA_DIR}")

# Add training dir to path
sys.path.insert(0, str(XURA_DIR / "training"))

# Import Xura models
from models.mamba3 import Mamba3Backbone, Mamba3Config, Mamba3Layer, RMSNorm
from models.predictor import Mamba3Predictor
from models.angn import ANGNConfig
from models.y_encoder_pretrained import PretrainedTextEncoder
from models.y_decoder import Mamba3Decoder
from models.vit import VisionEncoder, VitConfig
from models.vljepa import Mamba3Jepa

print("Xura models imported successfully.")

# %% [markdown]
# ## Load DINOv2 ViT-L as X-Encoder
#
# Using DINOv2 ViT-L/14 (public, via timm) instead of V-JEPA 2 ViT-L
# (requires Meta approval). Both are ViT-L/14 with ~304M params.

# %% X-Encoder: DINOv2 ViT-L
import timm

def load_dinov2_vit_l(device: torch.device) -> nn.Module:
    """Load DINOv2 ViT-L as frozen X-Encoder.

    DINOv2 produces 1024-dim patch features, matching VitConfig.vjepa2_vit_l().
    We wrap timm's DINOv2 to match VisionEncoder's output interface.
    """
    print("Loading DINOv2 ViT-L/14 from timm...")

    # Load DINOv2 via timm (auto-downloads weights)
    dinov2 = timm.create_model(
        "vit_large_patch14_dinov2.lvd142m",
        pretrained=True,
        num_classes=0,  # Remove classification head
    )
    dinov2 = dinov2.eval()
    for p in dinov2.parameters():
        p.requires_grad = False

    # Verify output dim
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        features = dinov2.forward_features(dummy)
        if features.dim() == 3:
            features = features[:, 1:, :]  # Remove CLS token
        print(f"DINOv2 output: {features.shape}")
        output_dim = features.shape[-1]
        if output_dim != 1024:
            import warnings
            warnings.warn(
                f"Expected DINOv2 output dim 1024, got {output_dim}. "
                f"The model may produce incorrect results. Check timm version.",
                stacklevel=2,
            )

    class DINOv2Wrapper(nn.Module):
        """Wraps timm DINOv2 to match VisionEncoder interface."""
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.config = VitConfig.vjepa2_vit_l()

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            features = self.model.forward_features(images)
            if features.dim() == 3 and features.shape[1] > self.config.num_patches:
                features = features[:, 1:, :]  # Remove CLS token
            return features

    wrapper = DINOv2Wrapper(dinov2).to(device)
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in wrapper.parameters()) / 1e6
    print(f"DINOv2 ViT-L loaded: {n_params:.1f}M params (frozen)")
    return wrapper

x_encoder = load_dinov2_vit_l(DEVICE)

# %% [markdown]
# ## Build Mamba3-JEPA Model
#
# - X-Encoder: DINOv2 ViT-L (frozen, 304M params)
# - Predictor: Mamba-3 backbone (trainable, ~150M params)
# - Y-Encoder: all-MiniLM-L6-v2 (frozen backbone + trainable projection)

# %% Build model
print("Building Mamba3-JEPA model...")

# Pretrained Y-Encoder
y_encoder = PretrainedTextEncoder(
    model_name=cfg.y_encoder_model,
    embed_dim=cfg.embed_dim,
    freeze_backbone=True,
)

# HuggingFace tokenizer for query tokens
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size
print(f"Query tokenizer: bert-base-uncased (vocab={vocab_size})")

# Mamba-3 Predictor (THE CORE — this is what we're training)
predictor = Mamba3Predictor(
    d_model=cfg.d_model,
    n_layers=cfg.n_layers,
    d_state=cfg.d_state,
    expand=2,
    headdim=64,
    embed_dim=cfg.embed_dim,
    vision_dim=cfg.vision_dim,
    query_embed_dim=cfg.d_model,
    angn_config=ANGNConfig(),
)

# Full model
model = Mamba3Jepa(
    x_encoder=x_encoder,
    predictor=predictor,
    y_encoder=y_encoder,
    y_decoder=Mamba3Decoder.small(),
    shared_embed_dim=cfg.embed_dim,
    query_vocab_size=vocab_size,
    query_embed_dim=cfg.query_embed_dim,  # 1024 = matches predictor.query_proj.in_features
).to(DEVICE)

# Count params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Trainable: {trainable:,} | Frozen: {frozen:,}")
print(f"VRAM after model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# %% [markdown]
# ## Dataset: CC3M Subset via HuggingFace
#
# Uses streaming mode to avoid downloading the full 590GB CC3M dataset.
# Falls back to synthetic data if unavailable.

# %% Dataset
class CC3MDataset(Dataset):
    """CC3M dataset from HuggingFace, with synthetic fallback.

    Uses streaming=True to avoid downloading the full 590GB dataset.
    Materializes only num_samples into memory.
    """

    def __init__(self, num_samples: int, image_size: int, max_seq_len: int, tokenizer):
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.use_synthetic = False
        self.materialized_samples = []

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Try loading CC3M from HuggingFace with streaming
        try:
            from datasets import load_dataset
            print(f"Loading CC3M subset ({num_samples} samples) via streaming...")
            stream = load_dataset(
                "pixparse/cc3m-wds",
                split="train",
                streaming=True,
            )
            # Materialize only the samples we need
            for i, sample in enumerate(stream):
                if i >= num_samples:
                    break
                self.materialized_samples.append(sample)
            print(f"Materialized {len(self.materialized_samples)} CC3M samples.")
        except Exception as e:
            print(f"CC3M unavailable ({e}). Using synthetic data.")
            self.use_synthetic = True
            self.num_samples = num_samples

    def __len__(self):
        if self.use_synthetic:
            return self.num_samples
        return len(self.materialized_samples)

    def __getitem__(self, idx):
        if self.use_synthetic:
            return self._synthetic_sample()

        sample = self.materialized_samples[idx]

        # Load image
        try:
            if isinstance(sample.get("image"), Image.Image):
                img = sample["image"].convert("RGB")
            elif "jpg" in sample:
                import io
                img = Image.open(io.BytesIO(sample["jpg"])).convert("RGB")
            else:
                img = Image.new("RGB", (self.image_size, self.image_size))
            image_tensor = self.transform(img)
        except Exception:
            image_tensor = torch.randn(3, self.image_size, self.image_size)

        # Get caption
        raw_text = sample.get("caption", sample.get("txt", "a photo"))
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)

        # Tokenize
        encoded = self.tokenizer(
            raw_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "image": image_tensor,
            "tokens": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "raw_text": raw_text,
        }

    def _synthetic_sample(self):
        captions = [
            "A cat sitting on a windowsill in warm sunlight",
            "A dog playing fetch in the park on a sunny day",
            "A beautiful sunset over the ocean with orange clouds",
            "A city skyline at night with lights reflecting on water",
            "A plate of fresh pasta with basil and tomato sauce",
            "A mountain landscape covered in snow during winter",
            "Children playing in a playground on a summer afternoon",
            "A stack of old books on a wooden library shelf",
            "A red bicycle leaning against a brick wall",
            "A garden full of colorful flowers after the rain",
            "An astronaut floating in space above the blue Earth",
            "A wooden cabin by a lake surrounded by pine trees",
        ]
        raw_text = random.choice(captions)
        encoded = self.tokenizer(
            raw_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "image": torch.randn(3, self.image_size, self.image_size),
            "tokens": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "raw_text": raw_text,
        }

dataset = CC3MDataset(cfg.num_samples, cfg.image_size, cfg.max_seq_len, tokenizer)
dataloader = DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
    drop_last=True,
)

print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")
print(f"Effective batch size: {cfg.batch_size * cfg.grad_accum_steps}")

# %% [markdown]
# ## Training Loop
#
# Phase 1 JEPA: Train Mamba-3 predictor + Y-Encoder projection via InfoNCE.
# X-Encoder (DINOv2) is frozen. Mixed precision FP16 for T4 efficiency.

# %% Optimizer
# Separate param groups with Y-Encoder LR multiplier (per VL-JEPA paper)
param_groups = [
    {"params": [p for p in model.predictor.parameters() if p.requires_grad],
     "lr": cfg.lr, "name": "predictor"},
    {"params": [p for p in model.y_encoder.parameters() if p.requires_grad],
     "lr": cfg.lr * cfg.y_encoder_lr_mult, "name": "y_encoder"},
]
# Query embedding + adapt layers
if model.query_embedding is not None:
    qe_params = list(model.query_embedding.parameters())
    if not isinstance(model.query_adapt, nn.Identity):
        qe_params.extend(list(model.query_adapt.parameters()))
    if qe_params:  # Only add if there are params to avoid empty group
        param_groups.append(
            {"params": qe_params, "lr": cfg.lr, "name": "query_embed"},
        )

optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

# Compute total optimizer steps, accounting for grad accumulation and
# the final incomplete batch (which also triggers an optimizer step).
num_batches = len(dataloader)
steps_per_epoch = math.ceil(num_batches / cfg.grad_accum_steps)
total_steps = steps_per_epoch * cfg.epochs
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=cfg.lr,
    total_steps=max(total_steps, 1),
    pct_start=min(cfg.warmup_steps / max(total_steps, 1), 0.3),
)

# Mixed precision scaler
# Use torch.cuda.amp API for PyTorch 2.2-2.5 compatibility
scaler = torch.cuda.amp.GradScaler() if cfg.fp16 and DEVICE.type == "cuda" else None
if scaler:
    print("FP16 mixed precision enabled")

# W&B
if cfg.use_wandb:
    import wandb
    wandb.init(project=cfg.wandb_project, config=vars(cfg))

# %% Training loop
print(f"\n{'='*60}")
print("Starting Phase 1 JEPA Training")
print(f"{'='*60}")

output_dir = Path(cfg.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
best_loss = float("inf")

for epoch in range(1, cfg.epochs + 1):
    model.train()
    model.x_encoder.eval()  # Keep frozen
    if hasattr(model.y_encoder, 'backbone'):
        model.y_encoder.backbone.eval()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    epoch_start = time.time()

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(DEVICE)
        tokens = batch["tokens"].to(DEVICE)
        raw_texts = batch["raw_text"]

        # Query embeddings — NO torch.no_grad()!
        # query_embedding and query_adapt are trainable layers.
        query_embeds = model.get_query_embeds(tokens)

        # Target embeddings (from pretrained Y-Encoder)
        target_embeds = model.y_encoder(raw_texts)

        # Forward with mixed precision
        # Use torch.cuda.amp.autocast for PyTorch 2.2-2.5 compatibility
        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model.forward_jepa(
                images=images,
                query_tokens=query_embeds,
                target_embeds=target_embeds,
                temperature=cfg.temperature,
            )
            loss = outputs["loss"] / cfg.grad_accum_steps

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step every grad_accum_steps, OR on the final batch
        is_accum_step = (batch_idx + 1) % cfg.grad_accum_steps == 0
        is_last_batch = (batch_idx + 1) == num_batches
        if is_accum_step or is_last_batch:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += outputs["loss"].item()
        total_acc += outputs["accuracy"].item()
        n_batches += 1

        if (batch_idx + 1) % cfg.log_interval == 0:
            avg_loss = total_loss / n_batches
            avg_acc = total_acc / n_batches
            elapsed = time.time() - epoch_start
            samples_per_sec = (batch_idx + 1) * cfg.batch_size / elapsed
            vram_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(
                f"  [{epoch}][{batch_idx+1}/{num_batches}] "
                f"loss={avg_loss:.4f} acc={avg_acc:.3f} "
                f"({samples_per_sec:.1f} samp/s, {vram_used:.1f}GB VRAM)"
            )

    # Epoch summary
    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = total_acc / max(n_batches, 1)
    epoch_time = time.time() - epoch_start

    print(f"\nEpoch {epoch}/{cfg.epochs}: loss={avg_loss:.4f} acc={avg_acc:.3f} ({epoch_time:.0f}s)")

    # Checkpointing
    if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
        ckpt_path = output_dir / f"phase1_epoch{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "accuracy": avg_acc,
        }, ckpt_path)
        print(f"  Saved checkpoint -> {ckpt_path}")

    # Best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_path = output_dir / "phase1_best.pt"
        torch.save({"model_state_dict": model.state_dict()}, best_path)
        print(f"  New best model (loss={best_loss:.4f}) -> {best_path}")

    # W&B
    if cfg.use_wandb:
        import wandb
        wandb.log({
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": avg_acc,
            "epoch_time_s": epoch_time,
        })

# %% [markdown]
# ## Export to Safetensors
#
# Export the trained Mamba-3 predictor weights to safetensors format
# for loading into the Rust inference engine.

# %% Export
print(f"\n{'='*60}")
print("Exporting weights to safetensors...")
print(f"{'='*60}")

best_path = output_dir / "phase1_best.pt"
if best_path.exists():
    try:
        from utils.export_weights import export_to_safetensors
        export_to_safetensors(str(best_path), str(output_dir / "safetensors"), phase=1)
        print("Safetensors export complete.")
    except ImportError:
        print("export_weights utility not available. Saving raw state_dict instead.")
        # Use map_location only — weights_only requires PyTorch 2.6+
        ckpt = torch.load(best_path, map_location="cpu")
        predictor_state = {
            k.replace("predictor.", ""): v
            for k, v in ckpt["model_state_dict"].items()
            if k.startswith("predictor.")
        }
        safetensors_dir = output_dir / "safetensors"
        safetensors_dir.mkdir(parents=True, exist_ok=True)
        torch.save(predictor_state, safetensors_dir / "predictor.pt")
        print(f"Predictor state_dict saved to {safetensors_dir / 'predictor.pt'}")
    except Exception as e:
        print(f"Safetensors export failed: {e}")
        print("Manual export: load the .pt checkpoint and save predictor state_dict.")
else:
    print("No best checkpoint found. Training may not have completed.")

# %% Summary
print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Best loss: {best_loss:.4f}")
print(f"Checkpoints: {output_dir}")
print(f"")
print("Next steps:")
print("  1. Download the checkpoint (.pt) from Kaggle output")
print("  2. Run Phase 2 (decoder training) locally or on Kaggle")
print("  3. Export safetensors for Rust inference engine")
