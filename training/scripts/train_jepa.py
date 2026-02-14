"""Phase 1: JEPA Pretraining — Predictor + Y-Encoder + ANGN.

Trains the Mamba-3 predictor to match Y-Encoder target embeddings via
InfoNCE contrastive loss. The X-Encoder (ViT) is frozen.

Usage:
    # Single GPU (with pretrained Y-Encoder — RECOMMENDED)
    python scripts/train_jepa.py \
        --pretrained-y-encoder \
        --data-dir /path/to/image_text_pairs \
        --output-dir checkpoints/phase1 \
        --epochs 50 --batch-size 64 --lr 3e-4

    # Single GPU (with Mamba Y-Encoder — legacy, random init)
    python scripts/train_jepa.py \
        --vit-checkpoint /path/to/vjepa2_vitl.pth \
        --data-dir /path/to/image_text_pairs \
        --output-dir checkpoints/phase1

    # Multi-GPU (via accelerate)
    accelerate launch scripts/train_jepa.py \
        --pretrained-y-encoder \
        --data-dir /path/to/image_text_pairs \
        --output-dir checkpoints/phase1

What trains:   Predictor backbone (Mamba-3) + ANGN gates + Y-Encoder
What's frozen: X-Encoder (V-JEPA 2 ViT-L / DINOv2)
What's off:    Y-Decoder, RecursionLayer
Loss:          InfoNCE (symmetric contrastive)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.vljepa import Mamba3Jepa
from models.vit import VisionEncoder, VitConfig
from models.predictor import Mamba3Predictor
from models.angn import ANGNConfig
from models.y_encoder import Mamba3TextEncoder
from models.y_encoder_pretrained import PretrainedTextEncoder
from models.y_decoder import Mamba3Decoder
from utils.export_weights import export_to_safetensors


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class ImageTextDataset(Dataset):
    """Dataset for image-text pairs with proper tokenization.

    Supports:
      - Directory with {id}.jpg + {id}.txt file pairs
      - JSONL manifest with {"image": path, "text": path_or_string}
      - Synthetic data fallback for testing
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        max_seq_len: int = 512,
        tokenizer_name: str = "bert-base-uncased",
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.max_seq_len = max_seq_len

        # Load HuggingFace tokenizer (no auth required for bert-base-uncased)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vocab_size = self.tokenizer.vocab_size
        print(f"[Dataset] Tokenizer: {tokenizer_name} (vocab_size={self.vocab_size})")

        # Image transforms
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Discover samples
        self.samples = []
        if (self.data_dir / "manifest.jsonl").exists():
            with open(self.data_dir / "manifest.jsonl") as f:
                for line in f:
                    self.samples.append(json.loads(line))
        else:
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                for img_path in self.data_dir.glob(ext):
                    txt_path = img_path.with_suffix(".txt")
                    if txt_path.exists():
                        self.samples.append({
                            "image": str(img_path),
                            "text": str(txt_path),
                        })

        if not self.samples:
            print(f"[WARNING] No samples found in {data_dir}. Using synthetic data.")
            self.synthetic = True
        else:
            self.synthetic = False
            print(f"[Dataset] Found {len(self.samples)} image-text pairs in {data_dir}")

    def __len__(self) -> int:
        return max(len(self.samples), 1000) if self.synthetic else len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if self.synthetic:
            return self._synthetic_sample()

        sample = self.samples[idx]

        # Load image
        try:
            from PIL import Image
            img = Image.open(sample["image"]).convert("RGB")
            image_tensor = self.transform(img)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load image {sample.get('image', '?')}: {e}", stacklevel=2)
            image_tensor = torch.randn(3, self.image_size, self.image_size)

        # Load text and tokenize with HuggingFace tokenizer
        try:
            text_source = sample["text"]
            if text_source.endswith(".txt") and Path(text_source).exists():
                with open(text_source) as f:
                    raw_text = f.read().strip()
            else:
                raw_text = text_source  # text is inline in manifest

            encoded = self.tokenizer(
                raw_text,
                max_length=self.max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            token_tensor = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load text {sample.get('text', '?')}: {e}", stacklevel=2)
            raw_text = "[error loading text]"
            token_tensor = torch.zeros(self.max_seq_len, dtype=torch.long)
            attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)

        return {
            "image": image_tensor,
            "tokens": token_tensor,
            "attention_mask": attention_mask,
            "raw_text": raw_text,
        }

    def _synthetic_sample(self) -> dict:
        """Generate a synthetic sample for testing the training loop."""
        captions = [
            "A cat sitting on a windowsill in the sunlight",
            "A dog playing fetch in the park on a sunny day",
            "A beautiful sunset over the ocean with orange clouds",
            "A city skyline at night with lights reflecting on water",
            "A plate of fresh pasta with basil and tomato sauce",
            "A mountain landscape covered in snow during winter",
            "Children playing in a playground on a summer afternoon",
            "A stack of old books on a wooden library shelf",
        ]
        import random
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


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: Mamba3Jepa,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    epoch: int,
    temperature: float = 0.07,
    angn_reg_weight: float = 0.01,
    log_interval: int = 50,
    use_pretrained_y_encoder: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> dict:
    """Train one epoch of Phase 1 JEPA pretraining."""
    model.train()
    # Keep X-encoder frozen
    model.x_encoder.eval()
    # Keep pretrained Y-encoder backbone frozen
    if use_pretrained_y_encoder and hasattr(model.y_encoder, 'backbone'):
        model.y_encoder.backbone.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_angn_loss = 0.0
    n_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        tokens = batch["tokens"].to(device)

        # Get query embeddings — NO torch.no_grad() here!
        # query_embedding and query_adapt are trainable layers.
        query_embeds = model.get_query_embeds(tokens)

        # Get target embeddings
        if use_pretrained_y_encoder:
            # Pretrained Y-Encoder: encode raw text strings
            raw_texts = batch["raw_text"]  # list of strings
            target_embeds = model.y_encoder(raw_texts)  # (B, embed_dim)
            target_tokens = None
        else:
            # Mamba Y-Encoder: encode token IDs
            target_embeds = None
            target_tokens = tokens

        # Forward JEPA (Mamba-3 predictor backbone untouched)
        use_amp = scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model.forward_jepa(
                images=images,
                query_tokens=query_embeds,
                target_tokens=target_tokens,
                target_embeds=target_embeds,
                temperature=temperature,
            )

            loss = outputs["loss"]

            # ANGN regularization (encourage sparse gating)
            angn_loss = torch.tensor(0.0, device=device)
            if model.predictor.angn is not None and angn_reg_weight > 0:
                angn_loss = model.predictor.angn.gate_regularization_loss()
                loss = loss + angn_reg_weight * angn_loss

        # Backward with optional mixed precision
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += outputs["loss"].item()
        total_acc += outputs["accuracy"].item()
        total_angn_loss += angn_loss.item()
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / n_batches
            avg_acc = total_acc / n_batches
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * images.shape[0] / elapsed
            print(
                f"  [{epoch}][{batch_idx+1}/{len(dataloader)}] "
                f"loss={avg_loss:.4f} acc={avg_acc:.3f} "
                f"angn_reg={total_angn_loss/n_batches:.4f} "
                f"({samples_per_sec:.1f} samples/s)"
            )

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": total_acc / max(n_batches, 1),
        "angn_reg": total_angn_loss / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    model: Mamba3Jepa,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 0.07,
    use_pretrained_y_encoder: bool = False,
) -> dict:
    """Validate Phase 1 JEPA.

    Note: @torch.no_grad() is correct here — validation does not update weights.
    The query embeddings don't need gradients for metric computation.
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        tokens = batch["tokens"].to(device)

        query_embeds = model.get_query_embeds(tokens)

        if use_pretrained_y_encoder:
            raw_texts = batch["raw_text"]
            target_embeds = model.y_encoder(raw_texts)
            target_tokens = None
        else:
            target_embeds = None
            target_tokens = tokens

        outputs = model.forward_jepa(
            images, query_embeds, target_tokens=target_tokens,
            target_embeds=target_embeds, temperature=temperature,
        )
        total_loss += outputs["loss"].item()
        total_acc += outputs["accuracy"].item()
        n_batches += 1

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_accuracy": total_acc / max(n_batches, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 1: JEPA Pretraining")
    parser.add_argument("--vit-checkpoint", type=str, default=None,
                        help="Path to V-JEPA 2 ViT-L checkpoint (.pth or .safetensors)")
    parser.add_argument("--data-dir", type=str, default="data/image_text",
                        help="Directory containing image-text pairs")
    parser.add_argument("--output-dir", type=str, default="checkpoints/phase1",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--angn-reg-weight", type=float, default=0.01,
                        help="ANGN gate sparsity regularization weight")
    parser.add_argument("--angn-enabled", action="store_true",
                        help="Enable ANGN gating during training")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--tiny", action="store_true",
                        help="Use tiny config for testing")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="mamba3-jepa")
    # --- NEW: pretrained Y-Encoder flags ---
    parser.add_argument("--pretrained-y-encoder", action="store_true",
                        help="Use pretrained sentence embedding model as Y-Encoder (recommended)")
    parser.add_argument("--y-encoder-model", type=str,
                        default=PretrainedTextEncoder.DEFAULT_MODEL,
                        help="HuggingFace model name for pretrained Y-Encoder")
    parser.add_argument("--y-encoder-lr-multiplier", type=float, default=0.05,
                        help="LR multiplier for Y-Encoder params (paper: 0.05)")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased",
                        help="HuggingFace tokenizer name (no auth required for bert-base-uncased)")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable FP16 mixed precision training (recommended for T4 GPUs)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Phase 1] Device: {device} ({gpu_name}, {vram_gb:.1f}GB VRAM)")
    else:
        device = torch.device("cpu")
        print(f"[Phase 1] Device: {device} (no GPU detected)")

    # W&B
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # ─── Build model ───
    use_pretrained = args.pretrained_y_encoder

    if args.tiny:
        print("[Phase 1] Using TINY config for testing")
        torch.backends.cudnn.benchmark = False
        model = Mamba3Jepa.tiny()
        image_size = 16
        vocab_size = 256
        use_pretrained = False  # tiny always uses Mamba Y-Encoder
    elif use_pretrained:
        print(f"[Phase 1] Using PRETRAINED Y-Encoder: {args.y_encoder_model}")
        angn_config = ANGNConfig.small() if args.angn_enabled else ANGNConfig()
        if args.vit_checkpoint:
            x_encoder = VisionEncoder.from_vjepa2_checkpoint(args.vit_checkpoint)
        else:
            print("[Phase 1] No ViT checkpoint — using random ViT weights")
            x_encoder = VisionEncoder(VitConfig.vjepa2_vit_l())
            x_encoder.eval()
            for p in x_encoder.parameters():
                p.requires_grad = False

        # Pretrained Y-Encoder
        y_encoder = PretrainedTextEncoder(
            model_name=args.y_encoder_model,
            embed_dim=1536,
        )

        # Get vocab size from dataset tokenizer
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        vocab_size = tok.vocab_size

        model = Mamba3Jepa(
            x_encoder=x_encoder,
            predictor=Mamba3Predictor(
                d_model=1024, n_layers=12, d_state=128,
                expand=2, headdim=64, embed_dim=1536,
                vision_dim=1024, query_embed_dim=1024,
                angn_config=angn_config,
            ),
            y_encoder=y_encoder,
            y_decoder=Mamba3Decoder.small(),
            shared_embed_dim=1536,
            query_vocab_size=vocab_size,
            query_embed_dim=1024,  # Match predictor.query_proj.in_features
        )
        image_size = 224
    else:
        angn_config = ANGNConfig.small() if args.angn_enabled else ANGNConfig()
        if args.vit_checkpoint:
            x_encoder = VisionEncoder.from_vjepa2_checkpoint(args.vit_checkpoint)
        else:
            print("[Phase 1] No ViT checkpoint — using random ViT weights")
            x_encoder = VisionEncoder(VitConfig.vjepa2_vit_l())
            x_encoder.eval()
            for p in x_encoder.parameters():
                p.requires_grad = False

        model = Mamba3Jepa(
            x_encoder=x_encoder,
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
        image_size = 224
        vocab_size = 32000

    model = model.to(device)

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[Phase 1] Trainable: {trainable:,} | Frozen: {frozen:,}")

    # ─── Data ───
    dataset = ImageTextDataset(
        args.data_dir,
        image_size=image_size,
        max_seq_len=args.max_seq_len,
        tokenizer_name=args.tokenizer if not args.tiny else "bert-base-uncased",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    # ─── Optimizer ───
    # Separate param groups with Y-Encoder LR multiplier (per VL-JEPA paper §3.1)
    param_groups = [
        {"params": [p for p in model.predictor.parameters() if p.requires_grad],
         "lr": args.lr},
        {"params": [p for p in model.y_encoder.parameters() if p.requires_grad],
         "lr": args.lr * args.y_encoder_lr_multiplier},
    ]
    # Query embedding + adapt (only exist in pretrained mode)
    if model.query_embedding is not None:
        qe_params = list(model.query_embedding.parameters())
        if not isinstance(model.query_adapt, nn.Identity):
            qe_params.extend(list(model.query_adapt.parameters()))
        if qe_params:  # Only add if there are params
            param_groups.append({"params": qe_params, "lr": args.lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=min(args.warmup_steps / max(total_steps, 1), 0.3),
    )

    # Mixed precision scaler (for T4 GPUs)
    scaler = torch.amp.GradScaler('cuda') if args.fp16 and device.type == "cuda" else None
    if scaler:
        print("[Phase 1] FP16 mixed precision enabled")

    # ─── Training ───
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        train_metrics = train_one_epoch(
            model, dataloader, optimizer, scheduler, device,
            epoch, args.temperature, args.angn_reg_weight,
            use_pretrained_y_encoder=use_pretrained,
            scaler=scaler,
        )
        print(f"  Train: loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.3f}")

        # Checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = output_dir / f"phase1_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

        # Best model
        if train_metrics["loss"] < best_loss:
            best_loss = train_metrics["loss"]
            best_path = output_dir / "phase1_best.pt"
            torch.save({"model_state_dict": model.state_dict()}, best_path)
            print(f"  New best model (loss={best_loss:.4f}) → {best_path}")

        # W&B
        if args.wandb:
            import wandb
            wandb.log({"epoch": epoch, **train_metrics})

    # ─── Export to safetensors ───
    print(f"\n{'='*60}")
    print("Exporting weights to safetensors...")
    best_path = output_dir / "phase1_best.pt"
    if best_path.exists():
        export_to_safetensors(str(best_path), str(output_dir / "safetensors"), phase=1)

    print("\n[Phase 1] Training complete!")


if __name__ == "__main__":
    main()
