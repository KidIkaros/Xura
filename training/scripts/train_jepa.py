"""Phase 1: JEPA Pretraining — Predictor + Y-Encoder + ANGN.

Trains the Mamba-3 predictor to match Y-Encoder target embeddings via
InfoNCE contrastive loss. The X-Encoder (ViT) is frozen.

KEY DESIGN: Stateful training with BPTT — Mamba-3 hidden state carries
across sequential batches within each episode. This trains the SSM to
maintain coherent state over time, not just match static pairs.

Usage:
    # Single GPU (tiny test)
    python scripts/train_jepa.py --tiny --epochs 2 --batch-size 4

    # Full training with state carry-over
    python scripts/train_jepa.py \
        --vit-checkpoint /path/to/vjepa2_vitl.pth \
        --data-dir /path/to/image_text_pairs \
        --output-dir checkpoints/phase1 \
        --epochs 50 --batch-size 16 --lr 3e-4 \
        --grad-accum 4 --fp16

    # Resume from checkpoint (critical for Kaggle 30h sessions)
    python scripts/train_jepa.py \
        --resume checkpoints/phase1/phase1_epoch010.pt \
        --data-dir /path/to/data --output-dir checkpoints/phase1

    # Phase 1.5: Stream from YouTube playlist (buffered, no disk dataset)
    python scripts/train_jepa.py \
        --playlist-url 'https://www.youtube.com/playlist?list=PLxxx' \
        --total-steps 50000 --frame-skip 5 --batch-size 8 \
        --output-dir checkpoints/phase1_stream --fp16

What trains:   Predictor backbone + ANGN gates + Y-Encoder
What's frozen: X-Encoder (ViT)
What's off:    Y-Decoder, RecursionLayer
Loss:          InfoNCE (symmetric contrastive)
State:         Mamba-3 hidden state carries across sequential batches (BPTT)
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.vljepa import Mamba3Jepa
from models.vit import VisionEncoder, VitConfig
from models.predictor import Mamba3Predictor
from models.angn import ANGNConfig
from models.y_encoder import Mamba3TextEncoder
from models.y_decoder import Mamba3Decoder
from utils.export_weights import export_to_safetensors
from utils.stream_loader import YouTubeStreamDataset


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class TemporalFrameDataset(Dataset):
    """Sequential frame-chunk dataset for temporal/stateful training.

    Yields contiguous chunks of T frames from video episodes. Each sample
    is a (T, C, H, W) tensor that Mamba processes in a single parallel scan.
    Temporal learning comes from:
      1) Mamba scanning T frames in parallel within each chunk.
      2) BPTT state carry-over across sequential chunks.

    Expected directory structure:
      data_dir/
        episode_000/
          frame_000000.jpg
          frame_000001.jpg
          ...
        episode_001/
          ...

    Or a manifest.jsonl with:
      {"episode": "ep0", "frames": ["frame_000.jpg", ...], "captions": ["text", ...]}
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        seq_len: int = 16,
        max_seq_len: int = 64,
        vocab_size: int = 32000,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.seq_len = seq_len
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # Build list of (chunk_paths, is_episode_start) where chunk_paths is a list of T frame paths
        self.chunks: list[tuple[list[str], bool]] = []

        if (self.data_dir / "temporal_manifest.jsonl").exists():
            self._load_from_manifest()
        elif self.data_dir.exists():
            self._discover_episodes()

        if not self.chunks:
            print(f"[WARNING] No temporal data in {data_dir}. Using synthetic temporal data.")
            self.synthetic = True
            self.n_synthetic_episodes = 10
            self.episode_len = 100
        else:
            self.synthetic = False
            print(f"[TemporalDataset] Found {len(self.chunks)} frame chunks (seq_len={seq_len})")

    def _chunk_episode(self, frame_paths: list[str]):
        """Split an episode's frames into disjoint chunks of seq_len."""
        for i in range(0, len(frame_paths) - self.seq_len + 1, self.seq_len):
            chunk = frame_paths[i : i + self.seq_len]
            self.chunks.append((chunk, i == 0))

    def _load_from_manifest(self):
        """Load from temporal_manifest.jsonl."""
        with open(self.data_dir / "temporal_manifest.jsonl") as f:
            for line in f:
                entry = json.loads(line)
                frames = [str(self.data_dir / f) for f in entry["frames"]]
                self._chunk_episode(frames)

    def _discover_episodes(self):
        """Auto-discover episode directories with numbered frames."""
        episode_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and d.name.startswith("episode")
        ])
        for ep_dir in episode_dirs:
            frames = sorted(ep_dir.glob("frame_*.jpg")) + sorted(ep_dir.glob("frame_*.png"))
            self._chunk_episode([str(f) for f in frames])

    def __len__(self) -> int:
        if self.synthetic:
            return self.n_synthetic_episodes * (self.episode_len // self.seq_len)
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        if self.synthetic:
            return self._synthetic_temporal_sample(idx)

        chunk_paths, is_start = self.chunks[idx]

        try:
            from PIL import Image
            from torchvision import transforms

            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            frames = torch.stack([
                transform(Image.open(p).convert("RGB")) for p in chunk_paths
            ])  # (T, C, H, W)
        except Exception:
            frames = torch.randn(self.seq_len, 3, self.image_size, self.image_size)

        tokens = torch.randint(0, self.vocab_size, (self.max_seq_len,))

        return {
            "frames": frames,           # (T, C, H, W) — chunk of sequential video frames
            "tokens": tokens,
            "is_episode_start": is_start,
        }

    def _synthetic_temporal_sample(self, idx: int) -> dict:
        """Generate synthetic temporal data with smooth transitions."""
        chunks_per_ep = self.episode_len // self.seq_len
        ep_idx = idx // chunks_per_ep
        chunk_idx = idx % chunks_per_ep

        # Create smoothly varying synthetic frames
        frames = []
        for t in range(self.seq_len):
            global_frame = chunk_idx * self.seq_len + t
            torch.manual_seed(ep_idx * 10000 + global_frame)
            frames.append(torch.randn(3, self.image_size, self.image_size))

        return {
            "frames": torch.stack(frames),  # (T, C, H, W)
            "tokens": torch.randint(0, self.vocab_size, (self.max_seq_len,)),
            "is_episode_start": chunk_idx == 0,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def detach_states(states: list[torch.Tensor] | None) -> list[torch.Tensor] | None:
    """Detach SSM states for BPTT truncation (carry values, cut gradients)."""
    if states is None:
        return None
    return [s.detach() for s in states]


def mask_states(
    states: list[torch.Tensor] | None,
    is_episode_start: torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor] | None:
    """Zero out SSM states only for samples that start a new episode.

    Per-sample masking: samples mid-episode keep their state, while samples
    at episode boundaries get their state reset to zero. This preserves
    temporal continuity for the rest of the batch.

    Args:
        states: Per-layer SSM states, each (batch, nheads, d_state).
        is_episode_start: (batch,) bool tensor — True for new episodes.
        device: Target device.

    Returns:
        Masked states (same structure), or None if states was None.
    """
    if states is None:
        return None
    if not is_episode_start.any().item():
        return states
    # keep_mask: True = keep state, False = zero out
    keep_mask = (~is_episode_start).float().to(device)
    # Broadcast (B,) → (B, 1, 1) to match (B, nheads, d_state)
    keep_mask = keep_mask.unsqueeze(-1).unsqueeze(-1)
    return [s * keep_mask for s in states]


def state_norms(states: list[torch.Tensor] | None) -> list[float]:
    """Compute L2 norms of each layer's SSM state for monitoring."""
    if states is None:
        return []
    return [s.norm().item() for s in states]


def train_one_epoch(
    model: Mamba3Jepa,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    epoch: int,
    temperature: float = 0.07,
    angn_reg_weight: float = 0.01,
    grad_accum_steps: int = 1,
    use_fp16: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    log_interval: int = 50,
) -> dict:
    """Train one epoch of Phase 1 JEPA pretraining with stateful BPTT.

    Key difference from static training: Mamba-3 hidden state carries
    across sequential batches within each episode. State is reset at
    episode boundaries and detached every batch for BPTT truncation.
    """
    model.train()
    # Keep X-encoder frozen
    model.x_encoder.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_angn_loss = 0.0
    total_state_norm = 0.0
    n_batches = 0
    n_state_resets = 0
    start_time = time.time()

    # Persistent SSM state across batches (BPTT)
    persistent_state: list[torch.Tensor] | None = None

    for batch_idx, batch in enumerate(dataloader):
        images = batch["frames"].to(device)
        tokens = batch["tokens"].to(device)
        is_episode_start = batch["is_episode_start"]

        # Per-sample state masking: zero out state only for samples starting
        # a new episode, preserving temporal continuity for the rest of the batch.
        persistent_state = mask_states(persistent_state, is_episode_start, device)
        if is_episode_start.any().item():
            n_state_resets += 1

        # Query embeddings: use Y-encoder's embedding layer as query source
        with torch.no_grad():
            query_embeds = model.y_encoder.embedding(tokens)  # (B, L, d_model_enc)
            # Truncate/pad to match predictor's query_embed_dim
            qry_dim = model.predictor.query_proj.in_features
            if query_embeds.shape[-1] != qry_dim:
                if query_embeds.shape[-1] > qry_dim:
                    query_embeds = query_embeds[..., :qry_dim]
                else:
                    pad = torch.zeros(*query_embeds.shape[:-1], qry_dim - query_embeds.shape[-1],
                                      device=device)
                    query_embeds = torch.cat([query_embeds, pad], dim=-1)

        # Forward JEPA with state carry-over
        with torch.amp.autocast("cuda", enabled=use_fp16):
            outputs = model.forward_jepa(
                images=images,
                query_tokens=query_embeds,
                target_tokens=tokens,
                temperature=temperature,
                previous_state=persistent_state,
            )

            loss = outputs["loss"]

            # ANGN regularization (encourage sparse gating)
            angn_loss = torch.tensor(0.0, device=device)
            if model.predictor.angn is not None and angn_reg_weight > 0:
                angn_loss = model.predictor.angn.gate_regularization_loss()
                loss = loss + angn_reg_weight * angn_loss

            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps

        # Carry state forward (detach for BPTT truncation)
        persistent_state = detach_states(outputs["final_state"])

        # Backward
        if use_fp16 and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step (with gradient accumulation)
        if (batch_idx + 1) % grad_accum_steps == 0:
            if use_fp16 and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Metrics
        total_loss += outputs["loss"].item()
        total_acc += outputs["accuracy"].item()
        total_angn_loss += angn_loss.item()
        s_norms = state_norms(persistent_state)
        total_state_norm += max(s_norms) if s_norms else 0.0
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / n_batches
            avg_acc = total_acc / n_batches
            avg_state_norm = total_state_norm / n_batches
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * images.shape[0] / elapsed
            print(
                f"  [{epoch}][{batch_idx+1}/{len(dataloader)}] "
                f"loss={avg_loss:.4f} acc={avg_acc:.3f} "
                f"state_norm={avg_state_norm:.2f} "
                f"resets={n_state_resets} "
                f"angn={total_angn_loss/n_batches:.4f} "
                f"({samples_per_sec:.1f} samp/s)"
            )

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": total_acc / max(n_batches, 1),
        "angn_reg": total_angn_loss / max(n_batches, 1),
        "avg_state_norm": total_state_norm / max(n_batches, 1),
        "state_resets": n_state_resets,
    }


@torch.no_grad()
def validate(
    model: Mamba3Jepa,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 0.07,
) -> dict:
    """Validate Phase 1 JEPA."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    persistent_state = None

    for batch in dataloader:
        images = batch["frames"].to(device)
        tokens = batch["tokens"].to(device)
        is_episode_start = batch["is_episode_start"]

        persistent_state = mask_states(persistent_state, is_episode_start, device)

        query_embeds = model.y_encoder.embedding(tokens)
        qry_dim = model.predictor.query_proj.in_features
        if query_embeds.shape[-1] != qry_dim:
            if query_embeds.shape[-1] > qry_dim:
                query_embeds = query_embeds[..., :qry_dim]
            else:
                pad = torch.zeros(*query_embeds.shape[:-1], qry_dim - query_embeds.shape[-1],
                                  device=device)
                query_embeds = torch.cat([query_embeds, pad], dim=-1)

        outputs = model.forward_jepa(
            images, query_embeds, tokens, temperature,
            previous_state=persistent_state,
        )
        persistent_state = detach_states(outputs["final_state"])

        total_loss += outputs["loss"].item()
        total_acc += outputs["accuracy"].item()
        n_batches += 1

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_accuracy": total_acc / max(n_batches, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Streaming training (Phase 1.5)
# ═══════════════════════════════════════════════════════════════════════════

def train_streaming(
    model: Mamba3Jepa,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    total_steps: int,
    temperature: float = 0.07,
    angn_reg_weight: float = 0.01,
    grad_accum_steps: int = 1,
    use_fp16: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    log_interval: int = 50,
    save_every_steps: int = 5000,
    output_dir: Path | None = None,
    use_wandb: bool = False,
    args: argparse.Namespace | None = None,
) -> dict:
    """Step-based training loop for YouTube streaming (Phase 1.5).

    Wraps the same BPTT + InfoNCE logic as train_one_epoch but uses
    an infinite iterator over an IterableDataset. Restarts the playlist
    on StopIteration so training can run for arbitrary step counts.
    """
    model.train()
    model.x_encoder.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_angn_loss = 0.0
    total_state_norm = 0.0
    n_batches = 0
    n_state_resets = 0
    best_loss = float("inf")
    start_time = time.time()

    persistent_state: list[torch.Tensor] | None = None
    iterator = iter(dataloader)

    for step in range(1, total_steps + 1):
        # Get next batch; restart playlist if exhausted
        try:
            batch = next(iterator)
        except StopIteration:
            print(f"[Stream] Playlist exhausted at step {step}, restarting...")
            iterator = iter(dataloader)
            batch = next(iterator)

        images = batch["frames"].to(device)
        tokens = batch["tokens"].to(device)
        is_episode_start = batch["is_episode_start"]

        # Per-sample state masking at episode boundaries
        persistent_state = mask_states(persistent_state, is_episode_start, device)
        if is_episode_start.any().item():
            n_state_resets += 1

        # Query embeddings
        with torch.no_grad():
            query_embeds = model.y_encoder.embedding(tokens)
            qry_dim = model.predictor.query_proj.in_features
            if query_embeds.shape[-1] != qry_dim:
                if query_embeds.shape[-1] > qry_dim:
                    query_embeds = query_embeds[..., :qry_dim]
                else:
                    pad = torch.zeros(*query_embeds.shape[:-1], qry_dim - query_embeds.shape[-1],
                                      device=device)
                    query_embeds = torch.cat([query_embeds, pad], dim=-1)

        # Forward
        with torch.amp.autocast("cuda", enabled=use_fp16):
            outputs = model.forward_jepa(
                images=images,
                query_tokens=query_embeds,
                target_tokens=tokens,
                temperature=temperature,
                previous_state=persistent_state,
            )

            loss = outputs["loss"]

            angn_loss = torch.tensor(0.0, device=device)
            if model.predictor.angn is not None and angn_reg_weight > 0:
                angn_loss = model.predictor.angn.gate_regularization_loss()
                loss = loss + angn_reg_weight * angn_loss

            loss = loss / grad_accum_steps

        persistent_state = detach_states(outputs["final_state"])

        # Backward
        if use_fp16 and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step
        if step % grad_accum_steps == 0:
            if use_fp16 and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Metrics
        total_loss += outputs["loss"].item()
        total_acc += outputs["accuracy"].item()
        total_angn_loss += angn_loss.item()
        s_norms = state_norms(persistent_state)
        total_state_norm += max(s_norms) if s_norms else 0.0
        n_batches += 1

        # Log
        if step % log_interval == 0:
            avg_loss = total_loss / n_batches
            avg_acc = total_acc / n_batches
            avg_state_norm = total_state_norm / n_batches
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            print(
                f"  [step {step}/{total_steps}] "
                f"loss={avg_loss:.4f} acc={avg_acc:.3f} "
                f"state_norm={avg_state_norm:.2f} "
                f"resets={n_state_resets} "
                f"angn={total_angn_loss/n_batches:.4f} "
                f"({steps_per_sec:.1f} steps/s)"
            )
            if use_wandb:
                import wandb
                wandb.log({
                    "step": step,
                    "loss": avg_loss,
                    "accuracy": avg_acc,
                    "state_norm": avg_state_norm,
                    "angn_reg": total_angn_loss / n_batches,
                })

        # Checkpoint
        if output_dir is not None and step % save_every_steps == 0:
            avg_loss = total_loss / n_batches
            ckpt_path = output_dir / f"phase1_stream_step{step:06d}.pt"
            ckpt_data = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_loss": best_loss,
                "args": vars(args) if args else {},
            }
            if scaler is not None:
                ckpt_data["scaler_state_dict"] = scaler.state_dict()
            torch.save(ckpt_data, ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = output_dir / "phase1_best.pt"
                torch.save({"model_state_dict": model.state_dict()}, best_path)
                print(f"  New best model (loss={best_loss:.4f}) → {best_path}")

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": total_acc / max(n_batches, 1),
        "angn_reg": total_angn_loss / max(n_batches, 1),
        "avg_state_norm": total_state_norm / max(n_batches, 1),
        "state_resets": n_state_resets,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 1: JEPA Pretraining (Stateful)")
    parser.add_argument("--vit-checkpoint", type=str, default=None,
                        help="Path to V-JEPA 2 ViT-L checkpoint (.pth or .safetensors)")
    parser.add_argument("--data-dir", type=str, default="data/image_text",
                        help="Directory containing image-text pairs or temporal frames")
    parser.add_argument("--output-dir", type=str, default="checkpoints/phase1",
                        help="Output directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
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
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable fp16 mixed precision training")
    parser.add_argument("--tiny", action="store_true",
                        help="Use tiny config for testing")
    parser.add_argument("--kaggle", action="store_true",
                        help="Use Kaggle-optimized config (fits T4 16GB)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="mamba3-jepa")
    # Phase 1.5: YouTube streaming args
    parser.add_argument("--playlist-url", type=str, default=None,
                        help="YouTube playlist URL (activates buffered streaming mode)")
    parser.add_argument("--total-steps", type=int, default=50000,
                        help="Total training steps for streaming mode")
    parser.add_argument("--seq-len", type=int, default=16,
                        help="Frames per chunk for Mamba parallel scan (default 16)")
    parser.add_argument("--frame-skip", type=int, default=5,
                        help="Keep every Nth frame from video (default 5 → ~6 eff. FPS)")
    parser.add_argument("--save-every-steps", type=int, default=5000,
                        help="Checkpoint interval in streaming mode (steps)")
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
    if args.tiny:
        print("[Phase 1] Using TINY config for testing")
        torch.backends.cudnn.benchmark = False
        model = Mamba3Jepa.tiny()
        image_size = 16
        vocab_size = 256
    elif args.kaggle:
        print("[Phase 1] Using KAGGLE config (fits T4 16GB)")
        model = Mamba3Jepa.kaggle()
        image_size = 224
        vocab_size = 32000
    else:
        angn_config = ANGNConfig.small() if args.angn_enabled else ANGNConfig()
        if args.vit_checkpoint:
            x_encoder = VisionEncoder.from_vjepa2_checkpoint(args.vit_checkpoint)
        else:
            print("[Phase 1] No ViT checkpoint provided — using random ViT weights")
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
    streaming_mode = args.playlist_url is not None

    if streaming_mode:
        print(f"[Phase 1.5] YouTube streaming mode (buffered)")
        print(f"  playlist: {args.playlist_url}")
        print(f"  frame_skip={args.frame_skip}, total_steps={args.total_steps}")
        dataset = YouTubeStreamDataset(
            playlist_url=args.playlist_url,
            image_size=image_size,
            seq_len=args.seq_len,
            frame_skip=args.frame_skip,
            max_seq_len=args.max_seq_len,
            vocab_size=vocab_size,
            max_resolution=480,
        )
        # IterableDataset: no sampler, no shuffle, cap workers to avoid IP ban
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=min(args.num_workers, 2),
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
    else:
        print("[Phase 1] Temporal frame dataset (sequential, stateful)")
        dataset = TemporalFrameDataset(
            args.data_dir,
            image_size=image_size,
            seq_len=args.seq_len,
            max_seq_len=args.max_seq_len,
            vocab_size=vocab_size,
        )
        # Sequential sampler — NO shuffling. Temporal order is the whole point.
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=SequentialSampler(dataset),
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )

    # ─── Optimizer ───
    # Only optimize trainable parameters (excludes frozen ViT)
    param_groups = [
        {"params": [p for p in model.predictor.parameters() if p.requires_grad], "lr": args.lr},
        {"params": [p for p in model.y_encoder.parameters() if p.requires_grad], "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    if streaming_mode:
        total_steps = args.total_steps
    else:
        total_steps = (len(dataloader) // args.grad_accum) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max(total_steps, 1),
        pct_start=min(args.warmup_steps / max(total_steps, 1), 0.3),
    )

    # fp16 scaler
    scaler = torch.amp.GradScaler("cuda") if args.fp16 and device.type == "cuda" else None

    # ─── Resume from checkpoint ───
    start_epoch = 1
    best_loss = float("inf")

    if args.resume:
        print(f"[Phase 1] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, best_loss={best_loss:.4f}")

    # ─── Training ───
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if streaming_mode:
        # Phase 1.5: step-based streaming loop
        print(f"\n{'='*60}")
        print(f"Phase 1.5: YouTube Streaming Training ({total_steps} steps)")
        print(f"{'='*60}")

        train_metrics = train_streaming(
            model, dataloader, optimizer, scheduler, device,
            total_steps=total_steps,
            temperature=args.temperature,
            angn_reg_weight=args.angn_reg_weight,
            grad_accum_steps=args.grad_accum,
            use_fp16=args.fp16,
            scaler=scaler,
            save_every_steps=args.save_every_steps,
            output_dir=output_dir,
            use_wandb=args.wandb,
            args=args,
        )
        print(
            f"\n  Stream done: loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.3f} "
            f"state_norm={train_metrics['avg_state_norm']:.2f} "
            f"resets={train_metrics['state_resets']}"
        )

    else:
        # Phase 1: epoch-based disk training
        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*60}")

            train_metrics = train_one_epoch(
                model, dataloader, optimizer, scheduler, device,
                epoch, args.temperature, args.angn_reg_weight,
                grad_accum_steps=args.grad_accum,
                use_fp16=args.fp16,
                scaler=scaler,
            )
            print(
                f"  Train: loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['accuracy']:.3f} "
                f"state_norm={train_metrics['avg_state_norm']:.2f} "
                f"resets={train_metrics['state_resets']}"
            )

            # Checkpoint (includes everything needed to resume)
            if epoch % args.save_every == 0 or epoch == args.epochs:
                ckpt_path = output_dir / f"phase1_epoch{epoch:03d}.pt"
                ckpt_data = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "train_metrics": train_metrics,
                    "args": vars(args),
                }
                if scaler is not None:
                    ckpt_data["scaler_state_dict"] = scaler.state_dict()
                torch.save(ckpt_data, ckpt_path)
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
