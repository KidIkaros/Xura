#!/usr/bin/env python3
"""Kaggle Notebook: Xura Phase 1 JEPA Training on T4 GPUs.

Split into small cells for easier debugging on Kaggle.
Compatibility: PyTorch 2.2.x-2.5.x (uses torch.cuda.amp API)
"""

# %% [markdown]
# # Xura Phase 1: JEPA Pretraining on Kaggle T4

# %% Cell 1: Install dependencies
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

# %% Cell 2: Imports and GPU check
import os
import io
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

# %% Cell 3: Configuration
class Config:
    d_model = 1024
    n_layers = 12
    d_state = 128
    embed_dim = 1536
    vision_dim = 1024
    query_embed_dim = 1024
    epochs = 10
    batch_size = 16
    grad_accum_steps = 16
    lr = 3e-4
    weight_decay = 0.01
    temperature = 0.07
    warmup_steps = 500
    max_seq_len = 128
    image_size = 224
    fp16 = True
    y_encoder_model = "sentence-transformers/all-MiniLM-L6-v2"
    y_encoder_lr_mult = 0.05
    num_samples = 50000
    num_workers = 2
    use_wandb = False
    wandb_project = "xura-mamba3-jepa"
    log_interval = 25
    save_every = 2
    output_dir = "/kaggle/working/checkpoints"

cfg = Config()
print("Config loaded.")

# %% Cell 4: Clone Xura repo
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

sys.path.insert(0, str(XURA_DIR))
sys.path.insert(0, str(XURA_DIR / "training"))

# %% Cell 5: Import Xura models
from models.mamba3 import Mamba3Backbone, Mamba3Config, Mamba3Layer, RMSNorm
from models.predictor import Mamba3Predictor
from models.angn import ANGNConfig
from models.y_encoder_pretrained import PretrainedTextEncoder
from models.y_decoder import Mamba3Decoder
from models.vit import VisionEncoder, VitConfig
from models.vljepa import Mamba3Jepa

print("Xura models imported successfully.")

# %% Cell 6: Load DINOv2 ViT-L as X-Encoder
import timm

print(f"timm version: {timm.__version__}")

DINOV2_MODEL_NAMES = [
    "vit_large_patch14_dinov2.lvd142m",
    "vit_large_patch14_dinov2",
    "vit_large_patch14_reg4_dinov2.lvd142m",
]

dinov2 = None
for model_name in DINOV2_MODEL_NAMES:
    try:
        print(f"Trying: {model_name}...")
        dinov2 = timm.create_model(model_name, pretrained=True, num_classes=0)
        print(f"Loaded: {model_name}")
        break
    except Exception as e:
        print(f"  Failed: {e}")
        continue

if dinov2 is None:
    available = [m for m in timm.list_models("*dinov2*")]
    print(f"Available DINOv2 models in timm: {available[:10]}")
    if available:
        dinov2 = timm.create_model(available[0], pretrained=True, num_classes=0)
        print(f"Loaded fallback: {available[0]}")
    else:
        raise RuntimeError("No DINOv2 model found in timm. Try: pip install timm>=0.9.12")

dinov2 = dinov2.eval()
for p in dinov2.parameters():
    p.requires_grad = False

print(f"DINOv2 params: {sum(p.numel() for p in dinov2.parameters()) / 1e6:.1f}M")

# %% Cell 7: Verify DINOv2 output and create wrapper

# Detect whether forward_features or forward should be used
_use_forward_features = True
with torch.no_grad():
    dummy = torch.randn(1, 3, 224, 224)
    try:
        features = dinov2.forward_features(dummy)
        print(f"forward_features works — output: {features.shape}")
    except Exception as e:
        print(f"forward_features failed ({e}), falling back to forward()")
        _use_forward_features = False
        features = dinov2(dummy)
        print(f"forward() output: {features.shape}")

    if features.dim() == 3:
        features = features[:, 1:, :]
    print(f"Final feature shape (after CLS removal): {features.shape}")
    output_dim = features.shape[-1]
    if output_dim != 1024:
        import warnings
        warnings.warn(f"Expected dim 1024, got {output_dim}.", stacklevel=2)

class DINOv2Wrapper(nn.Module):
    """Wraps timm DINOv2 to match VisionEncoder interface."""
    def __init__(self, model, use_forward_features: bool = True):
        super().__init__()
        self.model = model
        self.config = VitConfig.vjepa2_vit_l()
        self._use_forward_features = use_forward_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self._use_forward_features:
            features = self.model.forward_features(images)
        else:
            features = self.model(images)
        if features.dim() == 3 and features.shape[1] > self.config.num_patches:
            features = features[:, 1:, :]
        return features

x_encoder = DINOv2Wrapper(dinov2, use_forward_features=_use_forward_features).to(DEVICE)
x_encoder.eval()
for p in x_encoder.parameters():
    p.requires_grad = False

print(f"X-Encoder ready on {DEVICE}")

# %% Cell 8: Build JEPA model
print("Building Mamba3-JEPA model...")

y_encoder = PretrainedTextEncoder(
    model_name=cfg.y_encoder_model,
    embed_dim=cfg.embed_dim,
    freeze_backbone=True,
)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size
print(f"Query tokenizer: bert-base-uncased (vocab={vocab_size})")

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

model = Mamba3Jepa(
    x_encoder=x_encoder,
    predictor=predictor,
    y_encoder=y_encoder,
    y_decoder=Mamba3Decoder.small(),
    shared_embed_dim=cfg.embed_dim,
    query_vocab_size=vocab_size,
    query_embed_dim=cfg.query_embed_dim,
).to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Trainable: {trainable:,} | Frozen: {frozen:,}")
print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# %% Cell 9: Load dataset
class CC3MDataset(Dataset):
    """CC3M with streaming + compressed JPEG storage (~500MB RAM for 50k)."""

    def __init__(self, num_samples, image_size, max_seq_len, tokenizer):
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.use_synthetic = False
        self.captions: list[str] = []
        self.image_bytes: list[bytes | None] = []
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        try:
            from datasets import load_dataset
            print(f"Loading CC3M subset ({num_samples} samples) via streaming...")
            stream = load_dataset("pixparse/cc3m-wds", split="train", streaming=True)
            for i, sample in enumerate(stream):
                if i >= num_samples:
                    break
                caption = sample.get("caption", sample.get("txt", "a photo"))
                if not isinstance(caption, str):
                    caption = str(caption)
                self.captions.append(caption)
                img = sample.get("image")
                if isinstance(img, Image.Image):
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    self.image_bytes.append(buf.getvalue())
                elif isinstance(sample.get("jpg"), bytes):
                    self.image_bytes.append(sample["jpg"])
                else:
                    self.image_bytes.append(None)
                if (i + 1) % 5000 == 0:
                    print(f"  ... {i + 1}/{num_samples}")
            print(f"Loaded {len(self.captions)} CC3M samples")
        except Exception as e:
            print(f"CC3M unavailable ({e}). Using synthetic data.")
            self.use_synthetic = True
            self.num_samples = num_samples

    def __len__(self):
        return self.num_samples if self.use_synthetic else len(self.captions)

    def __getitem__(self, idx):
        if self.use_synthetic:
            return self._synthetic_sample()
        raw_text = self.captions[idx]
        try:
            img_data = self.image_bytes[idx]
            if img_data is not None:
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
            else:
                img = Image.new("RGB", (self.image_size, self.image_size))
            image_tensor = self.transform(img)
        except Exception:
            image_tensor = torch.randn(3, self.image_size, self.image_size)
        encoded = self.tokenizer(
            raw_text, max_length=self.max_seq_len, truncation=True,
            padding="max_length", return_tensors="pt",
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
        ]
        raw_text = random.choice(captions)
        encoded = self.tokenizer(
            raw_text, max_length=self.max_seq_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        return {
            "image": torch.randn(3, self.image_size, self.image_size),
            "tokens": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "raw_text": raw_text,
        }

dataset = CC3MDataset(cfg.num_samples, cfg.image_size, cfg.max_seq_len, tokenizer)
dataloader = DataLoader(
    dataset, batch_size=cfg.batch_size, shuffle=True,
    num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
)
print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")

# %% Cell 10: Setup optimizer and scheduler
param_groups = [
    {"params": [p for p in model.predictor.parameters() if p.requires_grad],
     "lr": cfg.lr, "name": "predictor"},
    {"params": [p for p in model.y_encoder.parameters() if p.requires_grad],
     "lr": cfg.lr * cfg.y_encoder_lr_mult, "name": "y_encoder"},
]
if model.query_embedding is not None:
    qe_params = list(model.query_embedding.parameters())
    if not isinstance(model.query_adapt, nn.Identity):
        qe_params.extend(list(model.query_adapt.parameters()))
    if qe_params:
        param_groups.append({"params": qe_params, "lr": cfg.lr, "name": "query_embed"})

optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

num_batches = len(dataloader)
steps_per_epoch = math.ceil(num_batches / cfg.grad_accum_steps)
total_steps = steps_per_epoch * cfg.epochs
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=cfg.lr, total_steps=max(total_steps, 1),
    pct_start=min(cfg.warmup_steps / max(total_steps, 1), 0.3),
)

scaler = torch.cuda.amp.GradScaler() if cfg.fp16 and DEVICE.type == "cuda" else None
if scaler:
    print("FP16 mixed precision enabled")

if cfg.use_wandb:
    import wandb
    wandb.init(project=cfg.wandb_project, config=vars(cfg))

print("Optimizer ready.")

# %% Cell 11: Training loop
print(f"\n{'='*60}")
print("Starting Phase 1 JEPA Training")
print(f"{'='*60}")

output_dir = Path(cfg.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
best_loss = float("inf")

for epoch in range(1, cfg.epochs + 1):
    model.train()
    model.x_encoder.eval()
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

        query_embeds = model.get_query_embeds(tokens)
        target_embeds = model.y_encoder(raw_texts)

        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model.forward_jepa(
                images=images, query_tokens=query_embeds,
                target_embeds=target_embeds, temperature=cfg.temperature,
            )
            loss = outputs["loss"] / cfg.grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

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
            sps = (batch_idx + 1) * cfg.batch_size / elapsed
            vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"  [{epoch}][{batch_idx+1}/{num_batches}] "
                  f"loss={avg_loss:.4f} acc={avg_acc:.3f} ({sps:.1f} samp/s, {vram:.1f}GB)")

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = total_acc / max(n_batches, 1)
    epoch_time = time.time() - epoch_start
    print(f"\nEpoch {epoch}/{cfg.epochs}: loss={avg_loss:.4f} acc={avg_acc:.3f} ({epoch_time:.0f}s)")

    if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
        ckpt_path = output_dir / f"phase1_epoch{epoch:03d}.pt"
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss, "accuracy": avg_acc,
        }, ckpt_path)
        print(f"  Saved -> {ckpt_path}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_path = output_dir / "phase1_best.pt"
        torch.save({"model_state_dict": model.state_dict()}, best_path)
        print(f"  New best (loss={best_loss:.4f}) -> {best_path}")

    if cfg.use_wandb:
        import wandb
        wandb.log({"epoch": epoch, "loss": avg_loss, "accuracy": avg_acc})

# %% Cell 12: Export to safetensors
print(f"\n{'='*60}")
print("Exporting weights...")
print(f"{'='*60}")

best_path = output_dir / "phase1_best.pt"
if best_path.exists():
    try:
        try:
            from training.utils.export_weights import export_to_safetensors
        except ImportError:
            from utils.export_weights import export_to_safetensors
        export_to_safetensors(str(best_path), str(output_dir / "safetensors"), phase=1)
        print("Safetensors export complete.")
    except ImportError:
        print("export_weights not available. Saving raw state_dict.")
        ckpt = torch.load(best_path, map_location="cpu")
        predictor_state = {
            k.replace("predictor.", ""): v
            for k, v in ckpt["model_state_dict"].items()
            if k.startswith("predictor.")
        }
        sf_dir = output_dir / "safetensors"
        sf_dir.mkdir(parents=True, exist_ok=True)
        torch.save(predictor_state, sf_dir / "predictor.pt")
        print(f"Saved to {sf_dir / 'predictor.pt'}")
    except Exception as e:
        print(f"Export failed: {e}")
else:
    print("No best checkpoint found.")

print(f"\nTRAINING COMPLETE — Best loss: {best_loss:.4f}")
print(f"Checkpoints: {output_dir}")
