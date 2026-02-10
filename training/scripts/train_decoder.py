"""Phase 2: Decoder Finetuning — Y-Decoder learns to generate text.

The predictor and Y-encoder are frozen from Phase 1. The decoder learns
to generate text tokens conditioned on predicted embeddings via cross-entropy.

Usage:
    python scripts/train_decoder.py \
        --phase1-checkpoint checkpoints/phase1/phase1_best.pt \
        --data-dir /path/to/vqa_data \
        --output-dir checkpoints/phase2 \
        --epochs 20 --batch-size 32 --lr 1e-4

What trains:   Y-Decoder only
What's frozen: X-Encoder, Predictor, Y-Encoder, ANGN (all from Phase 1)
Loss:          Cross-entropy next-token prediction
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.vljepa import Mamba3Jepa
from utils.export_weights import export_to_safetensors


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class VQADataset(Dataset):
    """Placeholder VQA/captioning dataset for decoder training.

    Expected format: JSONL with fields:
      {"image": "path/to/img.jpg", "question": "...", "answer": "..."}
    Or for captioning:
      {"image": "path/to/img.jpg", "caption": "..."}

    Replace with actual VQAv2, OK-VQA, COCO Captions, or LLaVA-Instruct.
    """

    def __init__(self, data_dir: str, image_size: int = 224, max_seq_len: int = 128):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.samples = []
        self.synthetic = False

        manifest = self.data_dir / "manifest.jsonl"
        if manifest.exists():
            with open(manifest) as f:
                for line in f:
                    self.samples.append(json.loads(line))
            print(f"[VQA Dataset] Loaded {len(self.samples)} samples from {manifest}")
        else:
            print(f"[WARNING] No manifest at {manifest}. Using synthetic data.")
            self.synthetic = True

    def __len__(self) -> int:
        return max(len(self.samples), 500) if self.synthetic else len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if self.synthetic:
            return {
                "image": torch.randn(3, self.image_size, self.image_size),
                "query_tokens": torch.randint(0, 256, (32,)),
                "target_tokens": torch.randint(0, 256, (self.max_seq_len,)),
            }

        sample = self.samples[idx]

        # Load image
        try:
            from PIL import Image
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            img = Image.open(sample["image"]).convert("RGB")
            image_tensor = transform(img)
        except Exception:
            image_tensor = torch.randn(3, self.image_size, self.image_size)

        # Tokenize question/query (placeholder — use real tokenizer)
        question = sample.get("question", sample.get("caption", "describe"))
        query_ids = [ord(c) % 32000 for c in question[:32]]
        query_ids += [0] * (32 - len(query_ids))

        # Tokenize answer/caption (target for decoder)
        answer = sample.get("answer", sample.get("caption", ""))
        target_ids = [ord(c) % 32000 for c in answer[:self.max_seq_len]]
        target_ids += [0] * (self.max_seq_len - len(target_ids))

        return {
            "image": image_tensor,
            "query_tokens": torch.tensor(query_ids, dtype=torch.long),
            "target_tokens": torch.tensor(target_ids, dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: Mamba3Jepa,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 50,
) -> dict:
    """Train decoder for one epoch."""
    # Only decoder trains
    model.y_decoder.train()
    model.x_encoder.eval()
    model.predictor.eval()
    model.y_encoder.eval()

    total_loss = 0.0
    n_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        query_tokens = batch["query_tokens"].to(device)
        target_tokens = batch["target_tokens"].to(device)

        # Get predicted embedding (frozen predictor)
        with torch.no_grad():
            visual_tokens = model.x_encoder(images)
            query_embeds = model.y_encoder.embedding(query_tokens)
            qry_dim = model.predictor.query_proj.in_features
            if query_embeds.shape[-1] != qry_dim:
                if query_embeds.shape[-1] > qry_dim:
                    query_embeds = query_embeds[..., :qry_dim]
                else:
                    pad = torch.zeros(*query_embeds.shape[:-1],
                                      qry_dim - query_embeds.shape[-1], device=device)
                    query_embeds = torch.cat([query_embeds, pad], dim=-1)
            pred_embed = model.predictor(visual_tokens, query_embeds)

        # Decoder forward (teacher forcing)
        input_tokens = target_tokens[:, :-1]
        target_labels = target_tokens[:, 1:]

        logits = model.y_decoder(pred_embed, input_tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_labels.reshape(-1),
            ignore_index=0,  # Ignore padding
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.y_decoder.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / n_batches
            elapsed = time.time() - start_time
            print(
                f"  [{epoch}][{batch_idx+1}/{len(dataloader)}] "
                f"loss={avg_loss:.4f} ({elapsed:.1f}s)"
            )

    return {"loss": total_loss / max(n_batches, 1)}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Decoder Finetuning")
    parser.add_argument("--phase1-checkpoint", type=str, required=True,
                        help="Path to Phase 1 best checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/vqa",
                        help="Directory with VQA/captioning data")
    parser.add_argument("--output-dir", type=str, default="checkpoints/phase2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase 2] Device: {device}")

    # ─── Build model & load Phase 1 weights ───
    if args.tiny:
        model = Mamba3Jepa.tiny()
        image_size = 16
    else:
        model = Mamba3Jepa.small()
        image_size = 224

    # Load Phase 1 checkpoint
    ckpt = torch.load(args.phase1_checkpoint, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Phase 2] Loaded Phase 1: {len(missing)} missing, {len(unexpected)} unexpected keys")

    model = model.to(device)

    # Freeze everything except decoder
    for name, param in model.named_parameters():
        if not name.startswith("y_decoder."):
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Phase 2] Trainable (decoder only): {trainable:,}")

    # ─── Data ───
    dataset = VQADataset(args.data_dir, image_size=image_size, max_seq_len=args.max_seq_len)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # ─── Optimizer ───
    optimizer = torch.optim.AdamW(
        model.y_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # ─── Training ───
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Phase 2 — Epoch {epoch}/{args.epochs}")

        metrics = train_one_epoch(model, dataloader, optimizer, device, epoch)
        print(f"  Loss: {metrics['loss']:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            path = output_dir / f"phase2_epoch{epoch:03d}.pt"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, path)
            print(f"  Saved → {path}")

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_path = output_dir / "phase2_best.pt"
            torch.save({"model_state_dict": model.state_dict()}, best_path)
            print(f"  New best (loss={best_loss:.4f}) → {best_path}")

    # Export
    print("\nExporting weights...")
    best_path = output_dir / "phase2_best.pt"
    if best_path.exists():
        export_to_safetensors(str(best_path), str(output_dir / "safetensors"), phase=2)

    print("[Phase 2] Training complete!")


if __name__ == "__main__":
    main()
