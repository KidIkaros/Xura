# Mamba3-JEPA Training

PyTorch training scripts for the Kore Mamba3-JEPA architecture. Train in Python, export to safetensors, load in Rust for inference.

## Setup

```bash
cd training
pip install -r requirements.txt

# For GPU-accelerated Mamba (optional):
pip install mamba-ssm>=2.0.0
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Mamba3-JEPA                            │
│                                                             │
│  Image ──► X-Encoder (ViT-L, frozen) ──► visual_tokens      │
│                                              │              │
│  Query ──► query_proj ──────────────────► concat            │
│                                              │              │
│                                         ANGN gate (×N)      │
│                                              │              │
│                                    Mamba-3 Backbone (×12)   │
│                                              │              │
│                                         pred_head           │
│                                              │              │
│                                    predicted_embedding       │
│                                         │         │         │
│                              ┌──────────┘         │         │
│                              ▼                    ▼         │
│                     Y-Encoder (target)     Y-Decoder (gen)  │
│                              │                    │         │
│                     InfoNCE loss           Cross-entropy     │
└─────────────────────────────────────────────────────────────┘
```

## Training Phases

### Phase 1: JEPA Pretraining (Foundation)

| | |
|---|---|
| **Trains** | Predictor + Y-Encoder + ANGN |
| **Frozen** | X-Encoder (V-JEPA 2 ViT-L) |
| **Loss** | InfoNCE contrastive |
| **Data** | Image-text pairs (CC3M, LAION) |

```bash
# Tiny test run (no GPU, synthetic data)
python scripts/train_jepa.py --tiny --epochs 2 --batch-size 4

# Full training
python scripts/train_jepa.py \
    --vit-checkpoint /path/to/vjepa2_vitl.pth \
    --data-dir /path/to/image_text_pairs \
    --output-dir checkpoints/phase1 \
    --epochs 50 --batch-size 64 --lr 3e-4 \
    --angn-enabled --wandb
```

### Phase 2: Decoder Finetuning (Speech)

| | |
|---|---|
| **Trains** | Y-Decoder only |
| **Frozen** | Everything from Phase 1 |
| **Loss** | Cross-entropy next-token |
| **Data** | VQA, captioning, instruction-following |

```bash
python scripts/train_decoder.py \
    --phase1-checkpoint checkpoints/phase1/phase1_best.pt \
    --data-dir /path/to/vqa_data \
    --output-dir checkpoints/phase2 \
    --epochs 20 --batch-size 32 --lr 1e-4
```

### Phase 3: Agent + Recursion (Intelligence)

| | |
|---|---|
| **Trains** | ConfusionMonitor, StateInjector |
| **Frozen** | Core model from Phase 1+2 |
| **Loss** | Binary confusion + injection MSE |
| **Data** | Tool-use demonstrations, synthetic confusion |

```bash
python scripts/train_agent.py \
    --phase2-checkpoint checkpoints/phase2/phase2_best.pt \
    --data-dir /path/to/tool_use_data \
    --output-dir checkpoints/phase3 \
    --epochs 10 --batch-size 16 --lr 5e-5
```

## Weight Export

After training, export weights to safetensors for Rust:

```bash
# Phase 1: predictor + y_encoder + angn
python -m utils.export_weights \
    --checkpoint checkpoints/phase1/phase1_best.pt \
    --output weights/ --phase 1

# Phase 2: adds y_decoder
python -m utils.export_weights \
    --checkpoint checkpoints/phase2/phase2_best.pt \
    --output weights/ --phase 2

# Phase 3: exports recursion weights separately
# (handled automatically by train_agent.py)
```

Output safetensors files:
- `predictor.safetensors` — Mamba-3 predictor backbone + projections
- `y_encoder.safetensors` — Text encoder backbone + embedding
- `angn.safetensors` — ANGN gate weights + biases
- `y_decoder.safetensors` — Decoder backbone + LM head
- `recursion.safetensors` — Confusion probe + state injector

## Loading in Rust

```rust
use kore_vljepa::loader::load_safetensors;

let weights = load_safetensors("weights/predictor.safetensors")?;
// Apply weights to Mamba3Predictor...
```

See [WEIGHT_MAPPING.md](WEIGHT_MAPPING.md) for the complete key mapping.

## Data Format

All training scripts accept a data directory with a `manifest.jsonl` file:

```jsonl
{"image": "images/001.jpg", "text": "captions/001.txt"}
{"image": "images/002.jpg", "question": "What color?", "answer": "Blue"}
{"image": null, "query": "Hello", "confused": false, "tool_used": null}
```

If no manifest exists, scripts fall back to synthetic data for testing.

## File Structure

```
training/
├── README.md
├── WEIGHT_MAPPING.md
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── mamba3.py       # Mamba-3 SSM layer + backbone
│   ├── angn.py         # Adaptive Neural Gating Network
│   ├── vit.py          # Vision Transformer (X-Encoder)
│   ├── predictor.py    # Mamba-3 Predictor
│   ├── y_encoder.py    # Y-Encoder (text → embedding)
│   ├── y_decoder.py    # Y-Decoder (embedding → text)
│   └── vljepa.py       # Full Mamba3-JEPA model
├── scripts/
│   ├── train_jepa.py   # Phase 1: JEPA pretraining
│   ├── train_decoder.py # Phase 2: Decoder finetuning
│   └── train_agent.py  # Phase 3: Agent finetuning
└── utils/
    ├── __init__.py
    └── export_weights.py # PyTorch → safetensors conversion
```
