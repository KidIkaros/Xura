# Weight Mapping: PyTorch → safetensors → Rust

This document defines the exact key mapping between PyTorch module paths
and the safetensors keys that the Rust `kore-vljepa` loader expects.

## Overview

```
PyTorch training → .pt checkpoint → export_weights.py → .safetensors → Rust loader
```

The Rust loader (`kore-vljepa/src/loader.rs`) reads flat safetensors keys.
PyTorch produces hierarchical keys. The export utility handles remapping.

---

## Predictor (Phase 1)

| PyTorch Key | safetensors Key | Shape |
|---|---|---|
| `predictor.vision_proj.weight` | `predictor.vision_proj.weight` | (d_model, vision_dim) |
| `predictor.vision_proj.bias` | `predictor.vision_proj.bias` | (d_model,) |
| `predictor.query_proj.weight` | `predictor.query_proj.weight` | (d_model, query_embed_dim) |
| `predictor.query_proj.bias` | `predictor.query_proj.bias` | (d_model,) |
| `predictor.pred_head.weight` | `predictor.pred_head.weight` | (embed_dim, d_model) |
| `predictor.pred_head.bias` | `predictor.pred_head.bias` | (embed_dim,) |
| `predictor.backbone.layers.{i}.in_proj.weight` | `predictor.layers.{i}.in_proj.weight` | (proj_size, d_model) |
| `predictor.backbone.layers.{i}.out_proj.weight` | `predictor.layers.{i}.out_proj.weight` | (d_model, d_inner) |
| `predictor.backbone.layers.{i}.A_log` | `predictor.layers.{i}.A_log` | (nheads,) |
| `predictor.backbone.layers.{i}.D` | `predictor.layers.{i}.D` | (nheads,) |
| `predictor.backbone.layers.{i}.dt_bias` | `predictor.layers.{i}.dt_bias` | (nheads,) |
| `predictor.backbone.norms.{i}.weight` | `predictor.norms.{i}.weight` | (d_model,) |
| `predictor.backbone.final_norm.weight` | `predictor.final_norm.weight` | (d_model,) |

## ANGN (Phase 1)

| PyTorch Key | safetensors Key | Shape |
|---|---|---|
| `predictor.angn.gates.{i}.weight` | `angn.gates.{i}.weight` | (d_model,) |
| `predictor.angn.gates.{i}.bias` | `angn.gates.{i}.bias` | (d_model,) |

## Y-Encoder (Phase 1)

| PyTorch Key | safetensors Key | Shape |
|---|---|---|
| `y_encoder.embedding.weight` | `y_encoder.embedding.weight` | (vocab_size, d_model) |
| `y_encoder.projection.weight` | `y_encoder.projection.weight` | (embed_dim, d_model) |
| `y_encoder.projection.bias` | `y_encoder.projection.bias` | (embed_dim,) |
| `y_encoder.backbone.layers.{i}.*` | `y_encoder.layers.{i}.*` | (varies) |
| `y_encoder.backbone.norms.{i}.weight` | `y_encoder.norms.{i}.weight` | (d_model,) |
| `y_encoder.backbone.final_norm.weight` | `y_encoder.final_norm.weight` | (d_model,) |

## Y-Decoder (Phase 2)

| PyTorch Key | safetensors Key | Shape |
|---|---|---|
| `y_decoder.embed_to_prefix.weight` | `y_decoder.embed_to_prefix.weight` | (prefix_len*d_model, embed_dim) |
| `y_decoder.embed_to_prefix.bias` | `y_decoder.embed_to_prefix.bias` | (prefix_len*d_model,) |
| `y_decoder.token_embedding.weight` | `y_decoder.token_embedding.weight` | (vocab_size, d_model) |
| `y_decoder.lm_head.weight` | `y_decoder.lm_head.weight` | (vocab_size, d_model) |
| `y_decoder.backbone.layers.{i}.*` | `y_decoder.layers.{i}.*` | (varies) |
| `y_decoder.backbone.norms.{i}.weight` | `y_decoder.norms.{i}.weight` | (d_model,) |
| `y_decoder.backbone.final_norm.weight` | `y_decoder.final_norm.weight` | (d_model,) |

## Recursion (Phase 3)

| PyTorch Key | safetensors Key | Shape |
|---|---|---|
| `confusion_probe.linear.weight` | `recursion.confusion_probe.linear.weight` | (1, d_model) |
| `confusion_probe.linear.bias` | `recursion.confusion_probe.linear.bias` | (1,) |
| `state_injector.projection.weight` | `recursion.injector.projection.weight` | (d_model, knowledge_dim) |
| `state_injector.projection.bias` | `recursion.injector.projection.bias` | (d_model,) |

## X-Encoder (ViT) — NOT EXPORTED

The ViT is frozen and loaded separately from Meta's V-JEPA 2 checkpoint.
It is not included in the training export. The Rust loader handles ViT
weights via `load_vit_weights()`.

---

## Small Config Dimensions

| Component | d_model | embed_dim | Other |
|---|---|---|---|
| ViT (X-Encoder) | 1024 | — | 24 layers, 16 heads |
| Predictor | 1024 | 1536 | 12 layers, d_state=128 |
| Y-Encoder | 768 | 1536 | 12 layers, d_state=64 |
| Y-Decoder | 512 | 1536 | 6 layers, prefix_len=8 |
| ANGN | 1024 | — | 12 gates |
| Confusion Probe | 1024 | — | linear → 1 |
| State Injector | 1024 | — | knowledge_dim=256 |
