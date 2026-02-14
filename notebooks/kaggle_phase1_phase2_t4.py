#!/usr/bin/env python3
"""Kaggle Notebook: Xura Phase 1 + Phase 2 Training on T4 GPUs.

Runs both phases in a single notebook:
  Phase 1: JEPA pretraining (InfoNCE contrastive) on CC3M
  Phase 2: Decoder finetuning (cross-entropy) on COCO Captions

Split into small cells for easier debugging on Kaggle.
Compatibility: PyTorch 2.2.x-2.5.x (uses torch.cuda.amp API)
"""

# %% [markdown]
# # Xura Phase 1 + 2: JEPA Pretraining & Decoder Training on Kaggle T4

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
install("sentencepiece>=0.1.99,<1.0.0")
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
class Phase1Config:
    """Phase 1: JEPA pretraining on CC3M."""
    epochs = 10
    batch_size = 16
    grad_accum_steps = 16
    lr = 3e-4
    weight_decay = 0.01
    temperature = 0.07
    warmup_steps = 500
    max_seq_len = 128
    y_encoder_lr_mult = 0.05
    num_samples = 50000
    log_interval = 25
    save_every = 2
    use_wandb = False
    wandb_project = "xura-mamba3-jepa"

class Phase2Config:
    """Phase 2: Decoder finetuning on COCO Captions."""
    epochs = 10
    batch_size = 24
    grad_accum_steps = 8
    lr = 1e-4
    weight_decay = 0.01
    warmup_steps = 300
    max_seq_len = 64
    num_samples = 50000
    log_interval = 25
    save_every = 2
    use_wandb = False
    wandb_project = "xura-mamba3-decoder"

class SharedConfig:
    """Shared model architecture settings."""
    d_model = 1024
    n_layers = 12
    d_state = 128
    embed_dim = 1536
    vision_dim = 1024
    query_embed_dim = 1024
    decoder_d_model = 512
    decoder_n_layers = 6
    decoder_vocab_size = 32100
    decoder_prefix_len = 8
    image_size = 518
    fp16 = True
    y_encoder_model = "sentence-transformers/all-MiniLM-L6-v2"
    num_workers = 2
    output_dir = "/kaggle/working/checkpoints"

p1 = Phase1Config()
p2 = Phase2Config()
cfg = SharedConfig()
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
    dummy = torch.randn(1, 3, 518, 518)
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

# %% Cell 8: Tokenizers
from transformers import AutoTokenizer, T5Tokenizer

query_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(f"Query tokenizer: bert-base-uncased (vocab={query_tokenizer.vocab_size})")

decoder_tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
print(f"Decoder tokenizer: t5-base (vocab={decoder_tokenizer.vocab_size})")
assert decoder_tokenizer.vocab_size == cfg.decoder_vocab_size, (
    f"T5 vocab {decoder_tokenizer.vocab_size} != config {cfg.decoder_vocab_size}"
)

PAD_TOKEN_ID = decoder_tokenizer.pad_token_id
EOS_TOKEN_ID = decoder_tokenizer.eos_token_id
BOS_TOKEN_ID = decoder_tokenizer.pad_token_id

# %% Cell 9: Build JEPA model (used for both phases)
print("Building Mamba3-JEPA model...")

y_encoder = PretrainedTextEncoder(
    model_name=cfg.y_encoder_model,
    embed_dim=cfg.embed_dim,
    freeze_backbone=True,
)

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

y_decoder = Mamba3Decoder(
    d_model=cfg.decoder_d_model, n_layers=cfg.decoder_n_layers,
    d_state=64, expand=2, headdim=32,
    vocab_size=cfg.decoder_vocab_size, prefix_len=cfg.decoder_prefix_len,
    embed_dim=cfg.embed_dim,
)

model = Mamba3Jepa(
    x_encoder=x_encoder,
    predictor=predictor,
    y_encoder=y_encoder,
    y_decoder=y_decoder,
    shared_embed_dim=cfg.embed_dim,
    query_vocab_size=query_tokenizer.vocab_size,
    query_embed_dim=cfg.query_embed_dim,
).to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Trainable: {trainable:,} | Frozen: {frozen:,}")
print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# %% [markdown]
# ---
# # Phase 1: JEPA Pretraining (InfoNCE contrastive on CC3M)

# %% Cell 10: Load Phase 1 dataset (CC3M)
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

p1_dataset = CC3MDataset(p1.num_samples, cfg.image_size, p1.max_seq_len, query_tokenizer)
p1_dataloader = DataLoader(
    p1_dataset, batch_size=p1.batch_size, shuffle=True,
    num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
)
print(f"Phase 1 Dataset: {len(p1_dataset)} samples, {len(p1_dataloader)} batches")

# %% Cell 11: Phase 1 optimizer and scheduler
param_groups = [
    {"params": [p for p in model.predictor.parameters() if p.requires_grad],
     "lr": p1.lr, "name": "predictor"},
    {"params": [p for p in model.y_encoder.parameters() if p.requires_grad],
     "lr": p1.lr * p1.y_encoder_lr_mult, "name": "y_encoder"},
]
if model.query_embedding is not None:
    qe_params = list(model.query_embedding.parameters())
    if not isinstance(model.query_adapt, nn.Identity):
        qe_params.extend(list(model.query_adapt.parameters()))
    if qe_params:
        param_groups.append({"params": qe_params, "lr": p1.lr, "name": "query_embed"})

p1_optimizer = torch.optim.AdamW(param_groups, weight_decay=p1.weight_decay)

p1_num_batches = len(p1_dataloader)
p1_steps_per_epoch = math.ceil(p1_num_batches / p1.grad_accum_steps)
p1_total_steps = p1_steps_per_epoch * p1.epochs
p1_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    p1_optimizer, max_lr=p1.lr, total_steps=max(p1_total_steps, 1),
    pct_start=min(p1.warmup_steps / max(p1_total_steps, 1), 0.3),
)

scaler = torch.cuda.amp.GradScaler() if cfg.fp16 and DEVICE.type == "cuda" else None
if scaler:
    print("FP16 mixed precision enabled")

if p1.use_wandb:
    import wandb
    wandb.init(project=p1.wandb_project, config={**vars(cfg), **vars(p1)})

print("Phase 1 optimizer ready.")

# %% Cell 12: Phase 1 training loop
print(f"\n{'='*60}")
print("Starting Phase 1 JEPA Training")
print(f"{'='*60}")

output_dir = Path(cfg.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
p1_best_loss = float("inf")

for epoch in range(1, p1.epochs + 1):
    model.train()
    model.x_encoder.eval()
    if hasattr(model.y_encoder, 'backbone'):
        model.y_encoder.backbone.eval()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    epoch_start = time.time()
    p1_optimizer.zero_grad()

    for batch_idx, batch in enumerate(p1_dataloader):
        images = batch["image"].to(DEVICE)
        tokens = batch["tokens"].to(DEVICE)
        raw_texts = batch["raw_text"]

        query_embeds = model.get_query_embeds(tokens)
        target_embeds = model.y_encoder(raw_texts)

        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model.forward_jepa(
                images=images, query_tokens=query_embeds,
                target_embeds=target_embeds, temperature=p1.temperature,
            )
            loss = outputs["loss"] / p1.grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_accum_step = (batch_idx + 1) % p1.grad_accum_steps == 0
        is_last_batch = (batch_idx + 1) == p1_num_batches
        if is_accum_step or is_last_batch:
            if scaler is not None:
                scaler.unscale_(p1_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(p1_optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                p1_optimizer.step()
            p1_scheduler.step()
            p1_optimizer.zero_grad()

        total_loss += outputs["loss"].item()
        total_acc += outputs["accuracy"].item()
        n_batches += 1

        if (batch_idx + 1) % p1.log_interval == 0:
            avg_loss = total_loss / n_batches
            avg_acc = total_acc / n_batches
            elapsed = time.time() - epoch_start
            sps = (batch_idx + 1) * p1.batch_size / elapsed
            vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"  [P1][{epoch}][{batch_idx+1}/{p1_num_batches}] "
                  f"loss={avg_loss:.4f} acc={avg_acc:.3f} ({sps:.1f} samp/s, {vram:.1f}GB)")

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = total_acc / max(n_batches, 1)
    epoch_time = time.time() - epoch_start
    print(f"\nPhase 1 Epoch {epoch}/{p1.epochs}: loss={avg_loss:.4f} acc={avg_acc:.3f} ({epoch_time:.0f}s)")

    if epoch % p1.save_every == 0 or epoch == p1.epochs:
        ckpt_path = output_dir / f"phase1_epoch{epoch:03d}.pt"
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": p1_optimizer.state_dict(),
            "loss": avg_loss, "accuracy": avg_acc,
        }, ckpt_path)
        print(f"  Saved -> {ckpt_path}")

    if avg_loss < p1_best_loss:
        p1_best_loss = avg_loss
        best_path = output_dir / "phase1_best.pt"
        torch.save({"model_state_dict": model.state_dict()}, best_path)
        print(f"  New best (loss={p1_best_loss:.4f}) -> {best_path}")

    if p1.use_wandb:
        import wandb
        wandb.log({"phase": 1, "epoch": epoch, "loss": avg_loss, "accuracy": avg_acc})

print(f"\nPHASE 1 COMPLETE — Best loss: {p1_best_loss:.4f}")

# %% Cell 13: Phase 1 export
print(f"\n{'='*60}")
print("Exporting Phase 1 weights...")
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

# %% [markdown]
# ---
# # Phase 2: Decoder Finetuning (cross-entropy on COCO Captions)

# %% Cell 14: Freeze Phase 1 weights, unfreeze decoder
print(f"\n{'='*60}")
print("Transitioning to Phase 2...")
print(f"{'='*60}")

# Free Phase 1 data and optimizer from memory
del p1_dataset, p1_dataloader, p1_optimizer, p1_scheduler
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Freeze everything except decoder
for name, param in model.named_parameters():
    if not name.startswith("y_decoder."):
        param.requires_grad = False
    else:
        param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Trainable (decoder only): {trainable:,} | Frozen: {frozen:,}")

# %% Cell 15: Load Phase 2 dataset (COCO Captions)
class COCOCaptionsDataset(Dataset):
    """COCO Captions streaming + compressed JPEG storage (~500MB RAM)."""

    def __init__(self, num_samples, image_size, max_seq_len, query_tok, decoder_tok):
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.query_tok = query_tok
        self.decoder_tok = decoder_tok
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
            print(f"Loading COCO Captions ({num_samples} samples) via streaming...")
            stream = load_dataset("HuggingFaceM4/COCO", split="train", streaming=True)
            for i, sample in enumerate(stream):
                if i >= num_samples:
                    break
                caption = sample.get("caption", sample.get("sentences", ["a photo"]))
                if isinstance(caption, list):
                    caption = caption[0] if caption else "a photo"
                if isinstance(caption, dict):
                    caption = caption.get("raw", str(caption))
                if not isinstance(caption, str):
                    caption = str(caption)
                self.captions.append(caption)
                img = sample.get("image")
                if isinstance(img, Image.Image):
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    self.image_bytes.append(buf.getvalue())
                else:
                    self.image_bytes.append(None)
                if (i + 1) % 5000 == 0:
                    print(f"  ... {i + 1}/{num_samples}")
            print(f"Loaded {len(self.captions)} COCO samples")
        except Exception as e:
            print(f"COCO unavailable ({e}). Using synthetic data.")
            self.use_synthetic = True
            self.num_samples = num_samples

    def __len__(self):
        return self.num_samples if self.use_synthetic else len(self.captions)

    def __getitem__(self, idx):
        if self.use_synthetic:
            return self._synthetic_sample()
        caption = self.captions[idx]
        try:
            img_data = self.image_bytes[idx]
            if img_data is not None:
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
            else:
                img = Image.new("RGB", (self.image_size, self.image_size))
            image_tensor = self.transform(img)
        except Exception:
            image_tensor = torch.randn(3, self.image_size, self.image_size)
        query_enc = self.query_tok(
            "describe this image", max_length=32, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        dec_enc = self.decoder_tok(
            caption, max_length=self.max_seq_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        return {
            "image": image_tensor,
            "query_tokens": query_enc["input_ids"].squeeze(0),
            "decoder_tokens": dec_enc["input_ids"].squeeze(0),
            "raw_caption": caption,
        }

    def _synthetic_sample(self):
        captions = [
            "A cat sitting on a windowsill",
            "A dog playing in the park",
            "A sunset over the ocean",
            "A city skyline at night",
            "Children playing in a garden",
            "A mountain covered in snow",
        ]
        caption = random.choice(captions)
        query_enc = self.query_tok(
            "describe this image", max_length=32, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        dec_enc = self.decoder_tok(
            caption, max_length=self.max_seq_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        return {
            "image": torch.randn(3, self.image_size, self.image_size),
            "query_tokens": query_enc["input_ids"].squeeze(0),
            "decoder_tokens": dec_enc["input_ids"].squeeze(0),
            "raw_caption": caption,
        }

p2_dataset = COCOCaptionsDataset(
    p2.num_samples, cfg.image_size, p2.max_seq_len,
    query_tokenizer, decoder_tokenizer,
)
p2_dataloader = DataLoader(
    p2_dataset, batch_size=p2.batch_size, shuffle=True,
    num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
)
print(f"Phase 2 Dataset: {len(p2_dataset)} samples, {len(p2_dataloader)} batches")

# %% Cell 16: Phase 2 optimizer and scheduler
p2_optimizer = torch.optim.AdamW(
    model.y_decoder.parameters(), lr=p2.lr, weight_decay=p2.weight_decay,
)

p2_num_batches = len(p2_dataloader)
p2_steps_per_epoch = math.ceil(p2_num_batches / p2.grad_accum_steps)
p2_total_steps = p2_steps_per_epoch * p2.epochs
p2_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    p2_optimizer, max_lr=p2.lr, total_steps=max(p2_total_steps, 1),
    pct_start=min(p2.warmup_steps / max(p2_total_steps, 1), 0.3),
)

# Reuse scaler from Phase 1 (or create new one if Phase 1 didn't use it)
if scaler is None and cfg.fp16 and DEVICE.type == "cuda":
    scaler = torch.cuda.amp.GradScaler()

if p2.use_wandb:
    import wandb
    if not wandb.run:
        wandb.init(project=p2.wandb_project, config={**vars(cfg), **vars(p2)})

print("Phase 2 optimizer ready.")

# %% Cell 17: Phase 2 training loop
print(f"\n{'='*60}")
print("Starting Phase 2 Decoder Training")
print(f"{'='*60}")

p2_best_loss = float("inf")

for epoch in range(1, p2.epochs + 1):
    model.y_decoder.train()
    model.x_encoder.eval()
    model.predictor.eval()
    if hasattr(model.y_encoder, 'backbone'):
        model.y_encoder.backbone.eval()

    total_loss = 0.0
    n_batches = 0
    epoch_start = time.time()
    p2_optimizer.zero_grad()

    for batch_idx, batch in enumerate(p2_dataloader):
        images = batch["image"].to(DEVICE)
        query_tokens = batch["query_tokens"].to(DEVICE)
        decoder_tokens = batch["decoder_tokens"].to(DEVICE)

        with torch.no_grad():
            visual_tokens = model.x_encoder(images)
            query_embeds = model.get_query_embeds(query_tokens)
            pred_embed = model.predictor(visual_tokens, query_embeds)

        input_tokens = decoder_tokens[:, :-1]
        target_labels = decoder_tokens[:, 1:]

        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model.y_decoder(pred_embed, input_tokens)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_labels.reshape(-1),
                ignore_index=PAD_TOKEN_ID,
            )
            loss = loss / p2.grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_accum_step = (batch_idx + 1) % p2.grad_accum_steps == 0
        is_last_batch = (batch_idx + 1) == p2_num_batches
        if is_accum_step or is_last_batch:
            if scaler is not None:
                scaler.unscale_(p2_optimizer)
                torch.nn.utils.clip_grad_norm_(model.y_decoder.parameters(), max_norm=1.0)
                scaler.step(p2_optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.y_decoder.parameters(), max_norm=1.0)
                p2_optimizer.step()
            p2_scheduler.step()
            p2_optimizer.zero_grad()

        total_loss += loss.item() * p2.grad_accum_steps
        n_batches += 1

        if (batch_idx + 1) % p2.log_interval == 0:
            avg_loss = total_loss / n_batches
            elapsed = time.time() - epoch_start
            sps = (batch_idx + 1) * p2.batch_size / elapsed
            vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"  [P2][{epoch}][{batch_idx+1}/{p2_num_batches}] "
                  f"loss={avg_loss:.4f} ({sps:.1f} samp/s, {vram:.1f}GB)")

    avg_loss = total_loss / max(n_batches, 1)
    epoch_time = time.time() - epoch_start

    # Sample generation
    model.y_decoder.eval()
    with torch.no_grad():
        sample_batch = next(iter(p2_dataloader))
        s_img = sample_batch["image"][:2].to(DEVICE)
        s_qry = sample_batch["query_tokens"][:2].to(DEVICE)
        s_cap = sample_batch["raw_caption"][:2]
        v_tok = model.x_encoder(s_img)
        q_emb = model.get_query_embeds(s_qry)
        s_pred = model.predictor(v_tok, q_emb)
        gen_ids = model.y_decoder.generate(
            s_pred, bos_token=BOS_TOKEN_ID,
            eos_token=EOS_TOKEN_ID, max_tokens=48, temperature=0.7,
        )
        print(f"\nPhase 2 Epoch {epoch}/{p2.epochs}: loss={avg_loss:.4f} ({epoch_time:.0f}s)")
        for i in range(min(2, len(gen_ids))):
            gen_text = decoder_tokenizer.decode(gen_ids[i], skip_special_tokens=True)
            print(f"  GT:  {s_cap[i][:80]}")
            print(f"  Gen: {gen_text[:80]}")

    if epoch % p2.save_every == 0 or epoch == p2.epochs:
        ckpt_path = output_dir / f"phase2_epoch{epoch:03d}.pt"
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": p2_optimizer.state_dict(), "loss": avg_loss,
        }, ckpt_path)
        print(f"  Saved -> {ckpt_path}")

    if avg_loss < p2_best_loss:
        p2_best_loss = avg_loss
        best_path = output_dir / "phase2_best.pt"
        torch.save({"model_state_dict": model.state_dict()}, best_path)
        print(f"  New best (loss={p2_best_loss:.4f}) -> {best_path}")

    if p2.use_wandb:
        import wandb
        wandb.log({"phase": 2, "epoch": epoch, "loss": avg_loss})

# %% Cell 18: Phase 2 export
print(f"\n{'='*60}")
print("Exporting Phase 2 weights...")
print(f"{'='*60}")

best_path = output_dir / "phase2_best.pt"
if best_path.exists():
    try:
        try:
            from training.utils.export_weights import export_to_safetensors
        except ImportError:
            from utils.export_weights import export_to_safetensors
        export_to_safetensors(str(best_path), str(output_dir / "safetensors"), phase=2)
        print("Safetensors export complete.")
    except ImportError:
        print("export_weights not available. Saving raw state_dict.")
        ckpt = torch.load(best_path, map_location="cpu")
        decoder_state = {
            k.replace("y_decoder.", ""): v
            for k, v in ckpt["model_state_dict"].items()
            if k.startswith("y_decoder.")
        }
        sf_dir = output_dir / "safetensors"
        sf_dir.mkdir(parents=True, exist_ok=True)
        torch.save(decoder_state, sf_dir / "decoder.pt")
        print(f"Saved to {sf_dir / 'decoder.pt'}")
    except Exception as e:
        print(f"Export failed: {e}")
else:
    print("No best checkpoint found.")

print(f"\n{'='*60}")
print(f"ALL TRAINING COMPLETE")
print(f"  Phase 1 best loss: {p1_best_loss:.4f}")
print(f"  Phase 2 best loss: {p2_best_loss:.4f}")
print(f"  Checkpoints: {output_dir}")
print(f"{'='*60}")
