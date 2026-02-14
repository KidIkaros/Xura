#!/usr/bin/env python3
"""Kaggle Notebook: Xura Phase 2 Decoder Training on T4 GPUs.

This script runs AFTER Phase 1 completes. It loads the Phase 1 checkpoint
and trains the Y-Decoder (Mamba-3 language model) to generate text from
the predictor's learned visual embeddings.

What it does:
  1. Installs dependencies (including sentencepiece for 32k decoder vocab)
  2. Loads Phase 1 checkpoint (predictor + X-Encoder + Y-Encoder)
  3. Freezes everything except Y-Decoder
  4. Downloads COCO Captions via HuggingFace datasets (streaming)
  5. Trains decoder via teacher-forced cross-entropy
  6. Exports decoder weights to safetensors

Architecture:
  images → X-Encoder(frozen) → visual_tokens
  query_text → BERT tokenizer → query_embedding → Predictor(frozen) → pred_embed
  pred_embed → Y-Decoder(trainable) → text logits

Estimated time: ~1.5 hours for 10 epochs on 50k COCO Captions.
Estimated VRAM: ~10GB peak (fits in T4's 15GB — decoder is smaller).

Compatibility: PyTorch 2.2.x-2.5.x (uses torch.cuda.amp API)
"""

# %% [markdown]
# # Xura Phase 2: Decoder Training on Kaggle T4
#
# Phase 1 taught the predictor to produce good embeddings.
# Phase 2 teaches the decoder to turn those embeddings into text.
#
# **Requires**: Phase 1 checkpoint at `/kaggle/working/checkpoints/phase1_best.pt`

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
install("sentencepiece>=0.1.99,<1.0.0")
install("safetensors>=0.4.0,<1.0.0")
install("einops>=0.7.0,<1.0.0")
install("datasets>=2.16.0,<3.0.0")
install("wandb>=0.16.0,<1.0.0")
install("tqdm>=4.66.0")
install("pillow>=10.0.0")
print("Dependencies installed.")

# %% Imports
import io
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
# Phase 2 trains ONLY the decoder (~25M params). Everything else is frozen.
# Memory footprint is lower than Phase 1, so we can use larger batches.

# %% Config
class Config:
    # Model (must match Phase 1)
    d_model = 1024          # Mamba-3 predictor hidden dim
    n_layers = 12           # Mamba-3 predictor layers
    d_state = 128           # SSM state dimension
    embed_dim = 1536        # Shared embedding space
    vision_dim = 1024       # ViT output dim (DINOv2 ViT-L)
    query_embed_dim = 1024  # Match predictor.query_proj.in_features

    # Decoder
    decoder_d_model = 512   # Decoder hidden dim (from Mamba3Decoder.small())
    decoder_n_layers = 6    # Decoder layers
    decoder_vocab_size = 32000  # Sentencepiece vocab
    decoder_prefix_len = 8  # Prefix tokens from predicted embedding

    # Training
    epochs = 10
    batch_size = 24         # Larger than Phase 1 (decoder is smaller)
    grad_accum_steps = 8    # Effective batch = 24 * 8 = 192
    lr = 1e-4               # Lower than Phase 1 (finetuning, not pretraining)
    weight_decay = 0.01
    warmup_steps = 300
    max_seq_len = 64        # Captions are short (~10-15 tokens)
    image_size = 224
    fp16 = True

    # Y-Encoder (must match Phase 1)
    y_encoder_model = "sentence-transformers/all-MiniLM-L6-v2"

    # Data
    num_samples = 50000     # COCO Captions subset
    num_workers = 2

    # Logging
    use_wandb = False
    wandb_project = "xura-mamba3-decoder"
    log_interval = 25
    save_every = 2
    output_dir = "/kaggle/working/checkpoints"

    # Phase 1 checkpoint
    phase1_checkpoint = "/kaggle/working/checkpoints/phase1_best.pt"

cfg = Config()

# %% [markdown]
# ## Clone Xura Repository

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

# Add both project root AND training dir to path
sys.path.insert(0, str(XURA_DIR))
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
# ## Tokenizers
#
# - **Query tokenizer**: BERT (same as Phase 1)
# - **Decoder tokenizer**: T5 sentencepiece (32000 vocab, matches Mamba3Decoder.small())

# %% Tokenizers
from transformers import AutoTokenizer, T5Tokenizer

# Query tokenizer (BERT, same as Phase 1)
query_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(f"Query tokenizer: bert-base-uncased (vocab={query_tokenizer.vocab_size})")

# Decoder tokenizer (T5 sentencepiece, 32000 vocab)
decoder_tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
print(f"Decoder tokenizer: t5-base (vocab={decoder_tokenizer.vocab_size})")
assert decoder_tokenizer.vocab_size == cfg.decoder_vocab_size, (
    f"T5 vocab size {decoder_tokenizer.vocab_size} != config {cfg.decoder_vocab_size}"
)

# Special tokens for decoder
PAD_TOKEN_ID = decoder_tokenizer.pad_token_id     # 0
EOS_TOKEN_ID = decoder_tokenizer.eos_token_id     # 1
BOS_TOKEN_ID = decoder_tokenizer.pad_token_id     # T5 uses pad as BOS

# %% [markdown]
# ## Load DINOv2 ViT-L (X-Encoder)
#
# Same as Phase 1 — frozen.

# %% X-Encoder
import timm

def load_dinov2_vit_l(device: torch.device) -> nn.Module:
    """Load DINOv2 ViT-L as frozen X-Encoder (same as Phase 1)."""
    print("Loading DINOv2 ViT-L/14 from timm...")
    dinov2 = timm.create_model(
        "vit_large_patch14_dinov2.lvd142m",
        pretrained=True,
        num_classes=0,
    )
    dinov2 = dinov2.eval()
    for p in dinov2.parameters():
        p.requires_grad = False

    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        features = dinov2.forward_features(dummy)
        if features.dim() == 3:
            features = features[:, 1:, :]
        output_dim = features.shape[-1]
        if output_dim != 1024:
            import warnings
            warnings.warn(
                f"Expected DINOv2 output dim 1024, got {output_dim}.",
                stacklevel=2,
            )

    class DINOv2Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.config = VitConfig.vjepa2_vit_l()

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            features = self.model.forward_features(images)
            if features.dim() == 3 and features.shape[1] > self.config.num_patches:
                features = features[:, 1:, :]
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
# ## Build Model & Load Phase 1 Checkpoint

# %% Build model
print("Building Mamba3-JEPA model...")

# Pretrained Y-Encoder (same as Phase 1)
y_encoder = PretrainedTextEncoder(
    model_name=cfg.y_encoder_model,
    embed_dim=cfg.embed_dim,
    freeze_backbone=True,
)

# Mamba-3 Predictor (will load Phase 1 weights)
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

# Y-Decoder (THIS IS WHAT WE TRAIN IN PHASE 2)
y_decoder = Mamba3Decoder(
    d_model=cfg.decoder_d_model,
    n_layers=cfg.decoder_n_layers,
    d_state=64,
    expand=2,
    headdim=32,
    vocab_size=cfg.decoder_vocab_size,
    prefix_len=cfg.decoder_prefix_len,
    embed_dim=cfg.embed_dim,
)

# Full model
model = Mamba3Jepa(
    x_encoder=x_encoder,
    predictor=predictor,
    y_encoder=y_encoder,
    y_decoder=y_decoder,
    shared_embed_dim=cfg.embed_dim,
    query_vocab_size=query_tokenizer.vocab_size,
    query_embed_dim=cfg.query_embed_dim,
).to(DEVICE)

# %% Load Phase 1 checkpoint
phase1_path = Path(cfg.phase1_checkpoint)
if phase1_path.exists():
    print(f"Loading Phase 1 checkpoint from {phase1_path}...")
    # No weights_only — not available in PyTorch 2.2-2.5
    ckpt = torch.load(phase1_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Expected: y_decoder keys missing (not trained in Phase 1)
    decoder_missing = [k for k in missing if k.startswith("y_decoder.")]
    other_missing = [k for k in missing if not k.startswith("y_decoder.")]
    print(f"  Decoder keys (expected missing): {len(decoder_missing)}")
    if other_missing:
        print(f"  WARNING: Non-decoder missing keys: {other_missing[:5]}...")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}...")
    print("Phase 1 weights loaded successfully.")
else:
    print(f"WARNING: No Phase 1 checkpoint at {phase1_path}")
    print("Training decoder from scratch (no pretrained predictor).")
    print("For best results, run Phase 1 notebook first!")

# %% Freeze everything except decoder
for name, param in model.named_parameters():
    if not name.startswith("y_decoder."):
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Trainable (decoder): {trainable:,} | Frozen: {frozen:,}")
print(f"VRAM after model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# %% [markdown]
# ## Dataset: COCO Captions via HuggingFace
#
# Uses streaming to avoid downloading the full dataset.
# Stores compressed JPEG bytes + caption strings to keep RAM low.

# %% Dataset
class COCOCaptionsDataset(Dataset):
    """COCO Captions from HuggingFace, with synthetic fallback.

    Streams data and stores captions + compressed image bytes.
    RAM usage: ~500MB for 50k samples.
    """

    def __init__(
        self,
        num_samples: int,
        image_size: int,
        max_seq_len: int,
        query_tokenizer,
        decoder_tokenizer,
    ):
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.query_tokenizer = query_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.use_synthetic = False
        self.captions: list[str] = []
        self.image_bytes: list[bytes | None] = []

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Try loading COCO Captions from HuggingFace
        try:
            from datasets import load_dataset
            print(f"Loading COCO Captions ({num_samples} samples) via streaming...")
            stream = load_dataset(
                "HuggingFaceM4/COCO",
                split="train",
                streaming=True,
            )
            for i, sample in enumerate(stream):
                if i >= num_samples:
                    break

                # COCO has multiple captions per image — use the first
                caption = sample.get("caption", sample.get("sentences", ["a photo"]))
                if isinstance(caption, list):
                    caption = caption[0] if caption else "a photo"
                if isinstance(caption, dict):
                    caption = caption.get("raw", str(caption))
                if not isinstance(caption, str):
                    caption = str(caption)
                self.captions.append(caption)

                # Store image as compressed JPEG bytes
                img = sample.get("image")
                if isinstance(img, Image.Image):
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    self.image_bytes.append(buf.getvalue())
                else:
                    self.image_bytes.append(None)

                if (i + 1) % 5000 == 0:
                    print(f"  ... {i + 1}/{num_samples} samples loaded")

            print(f"Loaded {len(self.captions)} COCO Captions "
                  f"(~{sum(len(b) for b in self.image_bytes if b) / 1e6:.0f}MB image bytes)")
        except Exception as e:
            print(f"COCO Captions unavailable ({e}). Using synthetic data.")
            self.use_synthetic = True
            self.num_samples = num_samples

    def __len__(self):
        if self.use_synthetic:
            return self.num_samples
        return len(self.captions)

    def __getitem__(self, idx):
        if self.use_synthetic:
            return self._synthetic_sample()

        caption = self.captions[idx]

        # Decode image from compressed bytes
        try:
            img_data = self.image_bytes[idx]
            if img_data is not None:
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
            else:
                img = Image.new("RGB", (self.image_size, self.image_size))
            image_tensor = self.transform(img)
        except Exception:
            image_tensor = torch.randn(3, self.image_size, self.image_size)

        # Query tokens (BERT) — "describe this image" or use the caption itself
        query_text = "describe this image"
        query_encoded = self.query_tokenizer(
            query_text,
            max_length=32,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Decoder target tokens (sentencepiece/T5)
        decoder_encoded = self.decoder_tokenizer(
            caption,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "image": image_tensor,
            "query_tokens": query_encoded["input_ids"].squeeze(0),
            "query_mask": query_encoded["attention_mask"].squeeze(0),
            "decoder_tokens": decoder_encoded["input_ids"].squeeze(0),
            "decoder_mask": decoder_encoded["attention_mask"].squeeze(0),
            "raw_query": query_text,
            "raw_caption": caption,
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
        ]
        caption = random.choice(captions)
        query_text = "describe this image"

        query_encoded = self.query_tokenizer(
            query_text, max_length=32, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        decoder_encoded = self.decoder_tokenizer(
            caption, max_length=self.max_seq_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )

        return {
            "image": torch.randn(3, self.image_size, self.image_size),
            "query_tokens": query_encoded["input_ids"].squeeze(0),
            "query_mask": query_encoded["attention_mask"].squeeze(0),
            "decoder_tokens": decoder_encoded["input_ids"].squeeze(0),
            "decoder_mask": decoder_encoded["attention_mask"].squeeze(0),
            "raw_query": query_text,
            "raw_caption": caption,
        }

dataset = COCOCaptionsDataset(
    cfg.num_samples, cfg.image_size, cfg.max_seq_len,
    query_tokenizer, decoder_tokenizer,
)
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
# Phase 2: Only Y-Decoder trains. Loss is cross-entropy next-token prediction.
# The predictor generates embeddings, decoder converts them to text.

# %% Optimizer
optimizer = torch.optim.AdamW(
    model.y_decoder.parameters(),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
)

num_batches = len(dataloader)
steps_per_epoch = math.ceil(num_batches / cfg.grad_accum_steps)
total_steps = steps_per_epoch * cfg.epochs
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=cfg.lr,
    total_steps=max(total_steps, 1),
    pct_start=min(cfg.warmup_steps / max(total_steps, 1), 0.3),
)

# Mixed precision
scaler = torch.cuda.amp.GradScaler() if cfg.fp16 and DEVICE.type == "cuda" else None
if scaler:
    print("FP16 mixed precision enabled")

# W&B
if cfg.use_wandb:
    import wandb
    wandb.init(project=cfg.wandb_project, config=vars(cfg))

# %% Training loop
print(f"\n{'='*60}")
print("Starting Phase 2 Decoder Training")
print(f"{'='*60}")

output_dir = Path(cfg.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
best_loss = float("inf")

for epoch in range(1, cfg.epochs + 1):
    # Only decoder trains
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

        # Get predicted embedding from frozen predictor
        with torch.no_grad():
            visual_tokens = model.x_encoder(images)
            query_embeds = model.get_query_embeds(query_tokens)
            pred_embed = model.predictor(visual_tokens, query_embeds)

        # Decoder teacher forcing: input = tokens[:-1], target = tokens[1:]
        input_tokens = decoder_tokens[:, :-1]
        target_labels = decoder_tokens[:, 1:]

        # Forward with mixed precision
        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model.y_decoder(pred_embed, input_tokens)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_labels.reshape(-1),
                ignore_index=PAD_TOKEN_ID,
            )
            loss = loss / cfg.grad_accum_steps

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step
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

        # Track unscaled loss
        total_loss += loss.item() * cfg.grad_accum_steps
        n_batches += 1

        if (batch_idx + 1) % cfg.log_interval == 0:
            avg_loss = total_loss / n_batches
            elapsed = time.time() - epoch_start
            samples_per_sec = (batch_idx + 1) * cfg.batch_size / elapsed
            vram_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

            # Decode a sample prediction for qualitative check
            sample_text = ""
            if (batch_idx + 1) % (cfg.log_interval * 4) == 0:
                with torch.no_grad():
                    gen_ids = model.y_decoder.generate(
                        pred_embed[:1],
                        bos_token=BOS_TOKEN_ID,
                        eos_token=EOS_TOKEN_ID,
                        max_tokens=32,
                        temperature=0.8,
                    )
                    sample_text = decoder_tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                    sample_text = f' | sample="{sample_text[:60]}"'

            print(
                f"  [{epoch}][{batch_idx+1}/{num_batches}] "
                f"loss={avg_loss:.4f} "
                f"({samples_per_sec:.1f} samp/s, {vram_used:.1f}GB VRAM)"
                f"{sample_text}"
            )

    # Epoch summary
    avg_loss = total_loss / max(n_batches, 1)
    epoch_time = time.time() - epoch_start

    # Generate a sample at end of each epoch
    model.y_decoder.eval()
    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        sample_images = sample_batch["image"][:2].to(DEVICE)
        sample_queries = sample_batch["query_tokens"][:2].to(DEVICE)
        sample_captions = sample_batch["raw_caption"][:2]

        visual_tok = model.x_encoder(sample_images)
        query_emb = model.get_query_embeds(sample_queries)
        sample_pred = model.predictor(visual_tok, query_emb)
        gen_ids = model.y_decoder.generate(
            sample_pred, bos_token=BOS_TOKEN_ID,
            eos_token=EOS_TOKEN_ID, max_tokens=48, temperature=0.7,
        )
        print(f"\nEpoch {epoch}/{cfg.epochs}: loss={avg_loss:.4f} ({epoch_time:.0f}s)")
        for i in range(min(2, len(gen_ids))):
            gen_text = decoder_tokenizer.decode(gen_ids[i], skip_special_tokens=True)
            print(f"  GT:  {sample_captions[i][:80]}")
            print(f"  Gen: {gen_text[:80]}")

    # Checkpointing
    if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
        ckpt_path = output_dir / f"phase2_epoch{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, ckpt_path)
        print(f"  Saved checkpoint -> {ckpt_path}")

    # Best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_path = output_dir / "phase2_best.pt"
        torch.save({"model_state_dict": model.state_dict()}, best_path)
        print(f"  New best model (loss={best_loss:.4f}) -> {best_path}")

    # W&B
    if cfg.use_wandb:
        import wandb
        wandb.log({"epoch": epoch, "loss": avg_loss, "epoch_time_s": epoch_time})

# %% [markdown]
# ## Export to Safetensors

# %% Export
print(f"\n{'='*60}")
print("Exporting Phase 2 weights to safetensors...")
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
        print("export_weights utility not available. Saving raw state_dict instead.")
        ckpt = torch.load(best_path, map_location="cpu")
        decoder_state = {
            k.replace("y_decoder.", ""): v
            for k, v in ckpt["model_state_dict"].items()
            if k.startswith("y_decoder.")
        }
        safetensors_dir = output_dir / "safetensors"
        safetensors_dir.mkdir(parents=True, exist_ok=True)
        torch.save(decoder_state, safetensors_dir / "decoder.pt")
        print(f"Decoder state_dict saved to {safetensors_dir / 'decoder.pt'}")
    except Exception as e:
        print(f"Safetensors export failed: {e}")
else:
    print("No best checkpoint found.")

# %% Summary
print(f"\n{'='*60}")
print("PHASE 2 TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Best loss: {best_loss:.4f}")
print(f"Checkpoints: {output_dir}")
print(f"")
print("Next steps:")
print("  1. Download phase2_best.pt from Kaggle output")
print("  2. Export safetensors for Rust inference engine")
print("  3. Deploy with xura-vljepa crate for inference")
