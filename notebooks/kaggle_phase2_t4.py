#!/usr/bin/env python3
"""Kaggle Notebook: Xura Phase 2 Decoder Training on T4 GPUs.

Runs AFTER Phase 1. Loads Phase 1 checkpoint, freezes everything except
Y-Decoder, trains decoder on COCO Captions via teacher-forced cross-entropy.

Split into small cells for easier debugging on Kaggle.
Compatibility: PyTorch 2.2.x-2.5.x (uses torch.cuda.amp API)
"""

# %% [markdown]
# # Xura Phase 2: Decoder Training on Kaggle T4
#
# **Requires**: Phase 1 checkpoint at `/kaggle/working/checkpoints/phase1_best.pt`

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
import io
import os
import math
import time
import json
import random
from pathlib import Path

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
    # Model (must match Phase 1)
    d_model = 1024
    n_layers = 12
    d_state = 128
    embed_dim = 1536
    vision_dim = 1024
    query_embed_dim = 1024
    # Decoder
    decoder_d_model = 512
    decoder_n_layers = 6
    decoder_vocab_size = 32000
    decoder_prefix_len = 8
    # Training
    epochs = 10
    batch_size = 24
    grad_accum_steps = 8
    lr = 1e-4
    weight_decay = 0.01
    warmup_steps = 300
    max_seq_len = 64
    image_size = 224
    fp16 = True
    # Y-Encoder (must match Phase 1)
    y_encoder_model = "sentence-transformers/all-MiniLM-L6-v2"
    # Data
    num_samples = 50000
    num_workers = 2
    # Logging
    use_wandb = False
    wandb_project = "xura-mamba3-decoder"
    log_interval = 25
    save_every = 2
    output_dir = "/kaggle/working/checkpoints"
    phase1_checkpoint = "/kaggle/working/checkpoints/phase1_best.pt"

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

# %% Cell 6: Tokenizers
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

# %% Cell 7: Load DINOv2 X-Encoder
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

if dinov2 is None:
    available = [m for m in timm.list_models("*dinov2*")]
    print(f"Available DINOv2 models: {available[:10]}")
    if available:
        dinov2 = timm.create_model(available[0], pretrained=True, num_classes=0)
        print(f"Loaded fallback: {available[0]}")
    else:
        raise RuntimeError("No DINOv2 model found in timm.")

dinov2 = dinov2.eval()
for p in dinov2.parameters():
    p.requires_grad = False
print(f"DINOv2 params: {sum(p.numel() for p in dinov2.parameters()) / 1e6:.1f}M")

# %% Cell 8: Create DINOv2 wrapper
with torch.no_grad():
    dummy = torch.randn(1, 3, 224, 224)
    try:
        features = dinov2.forward_features(dummy)
    except Exception as e:
        print(f"forward_features failed: {e}, trying forward...")
        features = dinov2(dummy)
    if features.dim() == 3:
        features = features[:, 1:, :]
    print(f"DINOv2 output shape: {features.shape}")
    if features.shape[-1] != 1024:
        import warnings
        warnings.warn(f"Expected dim 1024, got {features.shape[-1]}", stacklevel=2)

class DINOv2Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = VitConfig.vjepa2_vit_l()
    def forward(self, images):
        features = self.model.forward_features(images)
        if features.dim() == 3 and features.shape[1] > self.config.num_patches:
            features = features[:, 1:, :]
        return features

x_encoder = DINOv2Wrapper(dinov2).to(DEVICE)
x_encoder.eval()
for p in x_encoder.parameters():
    p.requires_grad = False
print(f"X-Encoder ready on {DEVICE}")

# %% Cell 9: Build model
print("Building Mamba3-JEPA model...")

y_encoder = PretrainedTextEncoder(
    model_name=cfg.y_encoder_model, embed_dim=cfg.embed_dim, freeze_backbone=True,
)

predictor = Mamba3Predictor(
    d_model=cfg.d_model, n_layers=cfg.n_layers, d_state=cfg.d_state,
    expand=2, headdim=64, embed_dim=cfg.embed_dim,
    vision_dim=cfg.vision_dim, query_embed_dim=cfg.d_model,
    angn_config=ANGNConfig(),
)

y_decoder = Mamba3Decoder(
    d_model=cfg.decoder_d_model, n_layers=cfg.decoder_n_layers,
    d_state=64, expand=2, headdim=32,
    vocab_size=cfg.decoder_vocab_size, prefix_len=cfg.decoder_prefix_len,
    embed_dim=cfg.embed_dim,
)

model = Mamba3Jepa(
    x_encoder=x_encoder, predictor=predictor, y_encoder=y_encoder,
    y_decoder=y_decoder, shared_embed_dim=cfg.embed_dim,
    query_vocab_size=query_tokenizer.vocab_size, query_embed_dim=cfg.query_embed_dim,
).to(DEVICE)

print(f"Model built. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# %% Cell 10: Load Phase 1 checkpoint and freeze
phase1_path = Path(cfg.phase1_checkpoint)
if phase1_path.exists():
    print(f"Loading Phase 1 checkpoint from {phase1_path}...")
    ckpt = torch.load(phase1_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    decoder_missing = [k for k in missing if k.startswith("y_decoder.")]
    other_missing = [k for k in missing if not k.startswith("y_decoder.")]
    print(f"  Decoder keys (expected missing): {len(decoder_missing)}")
    if other_missing:
        print(f"  WARNING — non-decoder missing: {other_missing[:5]}...")
    print("Phase 1 weights loaded.")
else:
    print(f"WARNING: No Phase 1 checkpoint at {phase1_path}")
    print("Training decoder from scratch (no pretrained predictor).")

for name, param in model.named_parameters():
    if not name.startswith("y_decoder."):
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Trainable (decoder): {trainable:,} | Frozen: {frozen:,}")

# %% Cell 11: Load dataset
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

dataset = COCOCaptionsDataset(
    cfg.num_samples, cfg.image_size, cfg.max_seq_len,
    query_tokenizer, decoder_tokenizer,
)
dataloader = DataLoader(
    dataset, batch_size=cfg.batch_size, shuffle=True,
    num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
)
print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")

# %% Cell 12: Setup optimizer
optimizer = torch.optim.AdamW(
    model.y_decoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
)

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

# %% Cell 13: Training loop
print(f"\n{'='*60}")
print("Starting Phase 2 Decoder Training")
print(f"{'='*60}")

output_dir = Path(cfg.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
best_loss = float("inf")

for epoch in range(1, cfg.epochs + 1):
    model.y_decoder.train()
    model.x_encoder.eval()
    model.predictor.eval()
    if hasattr(model.y_encoder, 'backbone'):
        model.y_encoder.backbone.eval()

    total_loss = 0.0
    n_batches = 0
    epoch_start = time.time()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
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
            loss = loss / cfg.grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_accum_step = (batch_idx + 1) % cfg.grad_accum_steps == 0
        is_last_batch = (batch_idx + 1) == num_batches
        if is_accum_step or is_last_batch:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.y_decoder.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.y_decoder.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * cfg.grad_accum_steps
        n_batches += 1

        if (batch_idx + 1) % cfg.log_interval == 0:
            avg_loss = total_loss / n_batches
            elapsed = time.time() - epoch_start
            sps = (batch_idx + 1) * cfg.batch_size / elapsed
            vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"  [{epoch}][{batch_idx+1}/{num_batches}] "
                  f"loss={avg_loss:.4f} ({sps:.1f} samp/s, {vram:.1f}GB)")

    avg_loss = total_loss / max(n_batches, 1)
    epoch_time = time.time() - epoch_start

    # Sample generation
    model.y_decoder.eval()
    with torch.no_grad():
        sample_batch = next(iter(dataloader))
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
        print(f"\nEpoch {epoch}/{cfg.epochs}: loss={avg_loss:.4f} ({epoch_time:.0f}s)")
        for i in range(min(2, len(gen_ids))):
            gen_text = decoder_tokenizer.decode(gen_ids[i], skip_special_tokens=True)
            print(f"  GT:  {s_cap[i][:80]}")
            print(f"  Gen: {gen_text[:80]}")

    if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
        ckpt_path = output_dir / f"phase2_epoch{epoch:03d}.pt"
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "loss": avg_loss,
        }, ckpt_path)
        print(f"  Saved -> {ckpt_path}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_path = output_dir / "phase2_best.pt"
        torch.save({"model_state_dict": model.state_dict()}, best_path)
        print(f"  New best (loss={best_loss:.4f}) -> {best_path}")

    if cfg.use_wandb:
        import wandb
        wandb.log({"epoch": epoch, "loss": avg_loss})

# %% Cell 14: Export
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

print(f"\nPHASE 2 COMPLETE — Best loss: {best_loss:.4f}")
print(f"Checkpoints: {output_dir}")
