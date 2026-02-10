# Xura

**Mamba3-JEPA Agent** — A vision-language state-space world model with adaptive neural gating, state-space delegation, and a 3-phase training pipeline.

Built on [KORE](https://github.com/KidIkaros/KORE) foundations.

## Architecture

```
Image ──► X-Encoder (ViT-L, frozen) ──► visual_tokens
                                             │
Query ──► query_proj ──────────────────► concat
                                             │
                                        ANGN gate (×N)      ← Adaptive Neural Gating
                                             │
                                   Mamba-3 Backbone (×12)   ← State-Space Model
                                             │
                                   RecursionLayer            ← State-Space Delegation
                                   (ConfusionMonitor → ToolRegistry → StateInjector)
                                             │
                                        pred_head
                                             │
                                   predicted_embedding
                                        │         │
                              ┌─────────┘         │
                              ▼                   ▼
                    Y-Encoder (target)     Y-Decoder (gen)
                              │                   │
                    InfoNCE loss           Cross-entropy
```

## Key Components

- **`xura-mamba`** — Mamba SSM (v1, v2, v3), S4, SSD, trapezoidal discretization, complex-valued state dynamics, MIMO multi-head, RoPE
- **`xura-vljepa`** — VL-JEPA model, Mamba3 Predictor, ANGN gating, SSD Agent loop, ConfusionMonitor, ToolRegistry, SelectiveDecoder
- **`training/`** — PyTorch training scripts for 3-phase training pipeline

## Training Phases

| Phase | Trains | Frozen | Loss |
|---|---|---|---|
| **1: JEPA** | Predictor + Y-Encoder + ANGN | X-Encoder (ViT-L) | InfoNCE |
| **2: Decoder** | Y-Decoder | Phase 1 model | Cross-entropy |
| **3: Agent** | ConfusionMonitor + StateInjector | Phase 1+2 model | Binary CE + MSE |

See [training/README.md](training/README.md) for full training documentation.

## Quick Start

```bash
# Build
cargo build --release

# Run tests
cargo test

# Training (PyTorch)
cd training
pip install -r requirements.txt
python scripts/train_jepa.py --tiny --epochs 2 --batch-size 4
```

## Crate Structure

```
Xura/
├── crates/
│   ├── xura-mamba/     # Mamba SSM engine
│   └── xura-vljepa/    # VL-JEPA model + Agent
├── examples/
├── training/           # PyTorch training pipeline
└── Cargo.toml          # Workspace
```

## Dependencies

Xura depends on [KORE](https://github.com/KidIkaros/KORE) foundation crates:
- `kore-core` — Tensor engine
- `kore-nn` — Neural network layers
- `kore-transformer` — Transformer building blocks
- `kore-attention` — Attention mechanisms

## License

Apache-2.0 — see [LICENSE](LICENSE).
