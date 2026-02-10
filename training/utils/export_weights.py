"""Weight export utility: PyTorch → safetensors for Rust inference.

Converts trained PyTorch model state dicts into safetensors files that
the Rust kore-vljepa loader can read. Handles key name remapping between
PyTorch module paths and the flat key namespace expected by Rust.

Usage:
    python -m utils.export_weights \
        --checkpoint checkpoints/phase1_best.pt \
        --output weights/ \
        --phase 1
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


# ═══════════════════════════════════════════════════════════════════════════
# Weight key mapping: PyTorch module path → safetensors key
# ═══════════════════════════════════════════════════════════════════════════

# The Rust loader (kore-vljepa/src/loader.rs) expects flat keys.
# PyTorch modules produce hierarchical keys like "predictor.backbone.layers.0.in_proj.weight".
# This mapping defines how to translate them.

PREDICTOR_KEY_MAP = {
    # Input projections
    "predictor.vision_proj.weight": "predictor.vision_proj.weight",
    "predictor.vision_proj.bias": "predictor.vision_proj.bias",
    "predictor.query_proj.weight": "predictor.query_proj.weight",
    "predictor.query_proj.bias": "predictor.query_proj.bias",
    # Prediction head
    "predictor.pred_head.weight": "predictor.pred_head.weight",
    "predictor.pred_head.bias": "predictor.pred_head.bias",
    # Backbone layers: predictor.backbone.layers.{i}.* → predictor.layers.{i}.*
    # Backbone norms: predictor.backbone.norms.{i}.weight → predictor.norms.{i}.weight
    # Backbone final_norm: predictor.backbone.final_norm.weight → predictor.final_norm.weight
}

ANGN_KEY_MAP = {
    # Gates: predictor.angn.gates.{i}.weight → angn.gates.{i}.weight
    # Gates: predictor.angn.gates.{i}.bias → angn.gates.{i}.bias
}

Y_ENCODER_KEY_MAP = {
    "y_encoder.embedding.weight": "y_encoder.embedding.weight",
    "y_encoder.projection.weight": "y_encoder.projection.weight",
    "y_encoder.projection.bias": "y_encoder.projection.bias",
}

Y_DECODER_KEY_MAP = {
    "y_decoder.embed_to_prefix.weight": "y_decoder.embed_to_prefix.weight",
    "y_decoder.embed_to_prefix.bias": "y_decoder.embed_to_prefix.bias",
    "y_decoder.lm_head.weight": "y_decoder.lm_head.weight",
    "y_decoder.token_embedding.weight": "y_decoder.token_embedding.weight",
}


def remap_key(pytorch_key: str) -> str:
    """Remap a PyTorch state dict key to safetensors key for Rust.

    Handles both explicit mappings and pattern-based backbone key remapping.
    """
    # Check explicit maps first
    all_maps = {**PREDICTOR_KEY_MAP, **ANGN_KEY_MAP, **Y_ENCODER_KEY_MAP, **Y_DECODER_KEY_MAP}
    if pytorch_key in all_maps:
        return all_maps[pytorch_key]

    # Pattern: predictor.backbone.layers.{i}.* → predictor.layers.{i}.*
    if pytorch_key.startswith("predictor.backbone.layers."):
        return pytorch_key.replace("predictor.backbone.layers.", "predictor.layers.")

    # Pattern: predictor.backbone.norms.{i}.* → predictor.norms.{i}.*
    if pytorch_key.startswith("predictor.backbone.norms."):
        return pytorch_key.replace("predictor.backbone.norms.", "predictor.norms.")

    # Pattern: predictor.backbone.final_norm.* → predictor.final_norm.*
    if pytorch_key.startswith("predictor.backbone.final_norm."):
        return pytorch_key.replace("predictor.backbone.final_norm.", "predictor.final_norm.")

    # Pattern: predictor.angn.* → angn.*
    if pytorch_key.startswith("predictor.angn."):
        return pytorch_key.replace("predictor.angn.", "angn.")

    # Pattern: y_encoder.backbone.* → y_encoder.*
    if pytorch_key.startswith("y_encoder.backbone.layers."):
        return pytorch_key.replace("y_encoder.backbone.layers.", "y_encoder.layers.")
    if pytorch_key.startswith("y_encoder.backbone.norms."):
        return pytorch_key.replace("y_encoder.backbone.norms.", "y_encoder.norms.")
    if pytorch_key.startswith("y_encoder.backbone.final_norm."):
        return pytorch_key.replace("y_encoder.backbone.final_norm.", "y_encoder.final_norm.")

    # Pattern: y_decoder.backbone.* → y_decoder.*
    if pytorch_key.startswith("y_decoder.backbone.layers."):
        return pytorch_key.replace("y_decoder.backbone.layers.", "y_decoder.layers.")
    if pytorch_key.startswith("y_decoder.backbone.norms."):
        return pytorch_key.replace("y_decoder.backbone.norms.", "y_decoder.norms.")
    if pytorch_key.startswith("y_decoder.backbone.final_norm."):
        return pytorch_key.replace("y_decoder.backbone.final_norm.", "y_decoder.final_norm.")

    # Default: pass through unchanged
    return pytorch_key


def build_weight_map(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap all keys in a state dict for Rust consumption."""
    remapped = {}
    for key, tensor in state_dict.items():
        # Skip non-parameter buffers (EMA tracking, step counts)
        if "ema_activation" in key or "step_count" in key:
            continue
        new_key = remap_key(key)
        remapped[new_key] = tensor.contiguous().to(torch.float32)
    return remapped


def export_to_safetensors(
    checkpoint_path: str,
    output_dir: str,
    phase: int = 1,
) -> list[str]:
    """Export a training checkpoint to safetensors files.

    Args:
        checkpoint_path: Path to PyTorch .pt checkpoint.
        output_dir: Directory to write safetensors files.
        phase: Training phase (1=JEPA, 2=Decoder, 3=Agent).
            Determines which components to export.

    Returns:
        List of exported file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    remapped = build_weight_map(state_dict)
    exported = []

    if phase >= 1:
        # Export predictor weights
        pred_weights = {k: v for k, v in remapped.items() if k.startswith("predictor.")}
        if pred_weights:
            path = str(output_path / "predictor.safetensors")
            save_file(pred_weights, path)
            exported.append(path)
            print(f"[Phase 1] Exported {len(pred_weights)} predictor tensors → {path}")

        # Export Y-encoder weights
        enc_weights = {k: v for k, v in remapped.items() if k.startswith("y_encoder.")}
        if enc_weights:
            path = str(output_path / "y_encoder.safetensors")
            save_file(enc_weights, path)
            exported.append(path)
            print(f"[Phase 1] Exported {len(enc_weights)} y_encoder tensors → {path}")

        # Export ANGN weights
        angn_weights = {k: v for k, v in remapped.items() if k.startswith("angn.")}
        if angn_weights:
            path = str(output_path / "angn.safetensors")
            save_file(angn_weights, path)
            exported.append(path)
            print(f"[Phase 1] Exported {len(angn_weights)} ANGN tensors → {path}")

    if phase >= 2:
        # Export Y-decoder weights
        dec_weights = {k: v for k, v in remapped.items() if k.startswith("y_decoder.")}
        if dec_weights:
            path = str(output_path / "y_decoder.safetensors")
            save_file(dec_weights, path)
            exported.append(path)
            print(f"[Phase 2] Exported {len(dec_weights)} y_decoder tensors → {path}")

    if phase >= 3:
        # Export recursion weights (confusion probe, state injector)
        rec_weights = {k: v for k, v in remapped.items() if k.startswith("recursion.")}
        if rec_weights:
            path = str(output_path / "recursion.safetensors")
            save_file(rec_weights, path)
            exported.append(path)
            print(f"[Phase 3] Exported {len(rec_weights)} recursion tensors → {path}")

    return exported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch weights to safetensors")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", required=True, help="Output directory for safetensors")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                        help="Training phase (determines which components to export)")
    args = parser.parse_args()

    files = export_to_safetensors(args.checkpoint, args.output, args.phase)
    print(f"\nExported {len(files)} files total.")
