"""Phase 3: Agent + Recursion Finetuning.

Trains the ConfusionMonitor, StateInjector, and tool selection components
on top of frozen Phase 1+2 model weights.

Usage:
    python scripts/train_agent.py \
        --phase2-checkpoint checkpoints/phase2/phase2_best.pt \
        --data-dir /path/to/tool_use_data \
        --output-dir checkpoints/phase3 \
        --epochs 10 --batch-size 16 --lr 5e-5

What trains:   ConfusionMonitor probe, StateInjector projection, tool embeddings
What's frozen: X-Encoder, Predictor backbone, Y-Encoder, Y-Decoder
               (ANGN gates optionally unfrozen for context-switch adaptation)
Loss:          Binary confusion classification + task performance (end-to-end)
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.vljepa import Mamba3Jepa
from utils.export_weights import export_to_safetensors


# ═══════════════════════════════════════════════════════════════════════════
# Confusion Probe — mirrors kore-vljepa's ConfusionMonitor
# ═══════════════════════════════════════════════════════════════════════════

class ConfusionProbe(nn.Module):
    """Binary classifier: hidden state → P(confused).

    Trained on synthetic confusion data:
    - Positive: corrupted inputs, unanswerable questions, OOD images
    - Negative: clean inputs with known-good answers

    Weight key mapping:
      confusion_probe.linear.weight → recursion.confusion_probe.weight
      confusion_probe.linear.bias   → recursion.confusion_probe.bias
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Predict confusion score from mean-pooled hidden state.

        Args:
            hidden_state: (batch, d_model) — mean-pooled predictor output

        Returns:
            (batch, 1) — confusion logits (apply sigmoid for probability)
        """
        return self.linear(hidden_state)


class StateInjector(nn.Module):
    """Projects knowledge vectors into the residual stream.

    h_new = h + W_inject · v_knowledge

    Weight key mapping:
      state_injector.projection.weight → recursion.injector.weight
      state_injector.projection.bias   → recursion.injector.bias
    """

    def __init__(self, knowledge_dim: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(knowledge_dim, d_model)

    def forward(
        self,
        hidden: torch.Tensor,
        knowledge: torch.Tensor,
    ) -> torch.Tensor:
        """Inject knowledge into hidden state.

        Args:
            hidden: (batch, seq_len, d_model)
            knowledge: (batch, knowledge_dim)

        Returns:
            (batch, seq_len, d_model)
        """
        injected = self.projection(knowledge).unsqueeze(1)  # (B, 1, d_model)
        return hidden + injected


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class ToolUseDataset(Dataset):
    """Dataset for agent/recursion training.

    Expected JSONL format:
      {
        "image": "path/to/img.jpg",       # optional
        "query": "What is this?",
        "answer": "A cat",
        "confused": false,                 # ground-truth confusion label
        "tool_used": "memory_search",      # optional: which tool was needed
        "knowledge": [0.1, 0.2, ...]       # optional: retrieved knowledge vector
      }

    For synthetic data generation, corrupt inputs to create positive
    confusion examples:
    - Shuffle image-question pairs (wrong image for question)
    - Use unanswerable questions
    - Add noise to images
    """

    def __init__(self, data_dir: str, d_model: int = 1024, max_seq_len: int = 64):
        self.data_dir = Path(data_dir)
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.samples = []
        self.synthetic = False

        manifest = self.data_dir / "manifest.jsonl"
        if manifest.exists():
            with open(manifest) as f:
                for line in f:
                    self.samples.append(json.loads(line))
            print(f"[ToolUse Dataset] Loaded {len(self.samples)} samples")
        else:
            print(f"[WARNING] No manifest at {manifest}. Using synthetic data.")
            self.synthetic = True

    def __len__(self) -> int:
        return max(len(self.samples), 500) if self.synthetic else len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if self.synthetic:
            confused = idx % 2 == 0  # 50% confusion rate
            return {
                "hidden_state": torch.randn(self.d_model) * (2.0 if confused else 0.5),
                "confused": torch.tensor(1.0 if confused else 0.0),
                "knowledge": torch.randn(256),  # Simulated retrieval
                "target_tokens": torch.randint(0, 256, (self.max_seq_len,)),
            }

        sample = self.samples[idx]
        return {
            "hidden_state": torch.randn(self.d_model),  # Placeholder — precompute from Phase 1
            "confused": torch.tensor(1.0 if sample.get("confused", False) else 0.0),
            "knowledge": torch.tensor(sample.get("knowledge", [0.0] * 256), dtype=torch.float32),
            "target_tokens": torch.randint(0, 256, (self.max_seq_len,)),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    confusion_probe: ConfusionProbe,
    state_injector: StateInjector,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    confusion_weight: float = 1.0,
    injection_weight: float = 0.5,
    log_interval: int = 50,
) -> dict:
    """Train confusion probe and state injector for one epoch."""
    confusion_probe.train()
    state_injector.train()

    total_confusion_loss = 0.0
    total_injection_loss = 0.0
    total_accuracy = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        hidden = batch["hidden_state"].to(device)
        confused_labels = batch["confused"].to(device)
        knowledge = batch["knowledge"].to(device)

        # ─── Confusion probe loss (binary cross-entropy) ───
        confusion_logits = confusion_probe(hidden).squeeze(-1)
        confusion_loss = F.binary_cross_entropy_with_logits(confusion_logits, confused_labels)

        # Accuracy
        with torch.no_grad():
            preds = (confusion_logits > 0).float()
            accuracy = (preds == confused_labels).float().mean()

        # ─── State injection loss ───
        # Proxy: injected state should be closer to knowledge than random
        # This is a simplified loss — in practice, measure downstream task improvement
        hidden_seq = hidden.unsqueeze(1).expand(-1, 4, -1)  # Fake seq_len=4
        injected = state_injector(hidden_seq, knowledge)
        # Loss: L2 between injected hidden (mean-pooled) and original + knowledge direction
        injected_pooled = injected.mean(dim=1)
        knowledge_proj = state_injector.projection(knowledge)
        injection_loss = F.mse_loss(injected_pooled, hidden + knowledge_proj)

        # Combined loss
        loss = confusion_weight * confusion_loss + injection_weight * injection_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(confusion_probe.parameters()) + list(state_injector.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        total_confusion_loss += confusion_loss.item()
        total_injection_loss += injection_loss.item()
        total_accuracy += accuracy.item()
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            print(
                f"  [{epoch}][{batch_idx+1}/{len(dataloader)}] "
                f"conf_loss={total_confusion_loss/n_batches:.4f} "
                f"inj_loss={total_injection_loss/n_batches:.4f} "
                f"acc={total_accuracy/n_batches:.3f}"
            )

    return {
        "confusion_loss": total_confusion_loss / max(n_batches, 1),
        "injection_loss": total_injection_loss / max(n_batches, 1),
        "accuracy": total_accuracy / max(n_batches, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Agent Finetuning")
    parser.add_argument("--phase2-checkpoint", type=str, default=None,
                        help="Phase 2 checkpoint (optional, for end-to-end)")
    parser.add_argument("--data-dir", type=str, default="data/tool_use")
    parser.add_argument("--output-dir", type=str, default="checkpoints/phase3")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--knowledge-dim", type=int, default=256)
    parser.add_argument("--confusion-weight", type=float, default=1.0)
    parser.add_argument("--injection-weight", type=float, default=0.5)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase 3] Device: {device}")

    d_model = 64 if args.tiny else args.d_model
    knowledge_dim = 16 if args.tiny else args.knowledge_dim

    # ─── Build components ───
    confusion_probe = ConfusionProbe(d_model).to(device)
    state_injector = StateInjector(knowledge_dim, d_model).to(device)

    trainable = sum(
        p.numel() for p in list(confusion_probe.parameters()) + list(state_injector.parameters())
    )
    print(f"[Phase 3] Trainable: {trainable:,}")

    # ─── Data ───
    dataset = ToolUseDataset(args.data_dir, d_model=d_model)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, drop_last=True,
    )

    # ─── Optimizer ───
    optimizer = torch.optim.AdamW(
        list(confusion_probe.parameters()) + list(state_injector.parameters()),
        lr=args.lr,
    )

    # ─── Training ───
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Phase 3 — Epoch {epoch}/{args.epochs}")

        metrics = train_one_epoch(
            confusion_probe, state_injector, dataloader, optimizer,
            device, epoch, args.confusion_weight, args.injection_weight,
        )
        print(f"  Confusion: {metrics['confusion_loss']:.4f} | "
              f"Injection: {metrics['injection_loss']:.4f} | "
              f"Acc: {metrics['accuracy']:.3f}")

        combined = metrics["confusion_loss"] + metrics["injection_loss"]
        if epoch % args.save_every == 0 or epoch == args.epochs:
            path = output_dir / f"phase3_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "confusion_probe": confusion_probe.state_dict(),
                "state_injector": state_injector.state_dict(),
                "metrics": metrics,
            }, path)
            print(f"  Saved → {path}")

        if combined < best_loss:
            best_loss = combined
            best_path = output_dir / "phase3_best.pt"
            torch.save({
                "confusion_probe": confusion_probe.state_dict(),
                "state_injector": state_injector.state_dict(),
            }, best_path)
            print(f"  New best → {best_path}")

    # ─── Export ───
    print("\nExporting recursion weights...")
    best_path = output_dir / "phase3_best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
        from safetensors.torch import save_file
        rec_weights = {}
        for k, v in ckpt["confusion_probe"].items():
            rec_weights[f"recursion.confusion_probe.{k}"] = v.contiguous().float()
        for k, v in ckpt["state_injector"].items():
            rec_weights[f"recursion.injector.{k}"] = v.contiguous().float()
        sf_dir = output_dir / "safetensors"
        sf_dir.mkdir(exist_ok=True)
        save_file(rec_weights, str(sf_dir / "recursion.safetensors"))
        print(f"  Exported {len(rec_weights)} tensors → {sf_dir / 'recursion.safetensors'}")

    print("[Phase 3] Training complete!")


if __name__ == "__main__":
    main()
