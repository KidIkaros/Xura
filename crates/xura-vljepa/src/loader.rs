//! Weight loading for Mamba3-JEPA components.
//!
//! Supports loading pretrained weights from SafeTensors format:
//! - V-JEPA 2 ViT weights → X-Encoder (frozen)
//! - Mamba-3 checkpoints → Predictor / Y-Encoder / Y-Decoder

use std::path::Path;

/// Errors during weight loading.
#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    SafeTensors(String),
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    MissingKey(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "IO error: {}", e),
            LoadError::SafeTensors(e) => write!(f, "SafeTensors error: {}", e),
            LoadError::ShapeMismatch { expected, got } =>
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got),
            LoadError::MissingKey(k) => write!(f, "Missing key: {}", k),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e)
    }
}

/// Load SafeTensors file and return raw tensor data keyed by name.
///
/// Returns a map of tensor_name → (shape, f32 data).
pub fn load_safetensors(
    path: &Path,
) -> Result<std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>, LoadError> {
    let data = std::fs::read(path)?;
    let tensors = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| LoadError::SafeTensors(e.to_string()))?;

    let mut result = std::collections::HashMap::new();
    for (name, tensor) in tensors.tensors() {
        let shape: Vec<usize> = tensor.shape().to_vec();
        let dtype = tensor.dtype();

        // Convert to f32
        let f32_data = match dtype {
            safetensors::Dtype::F32 => {
                let bytes = tensor.data();
                bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            }
            safetensors::Dtype::F16 => {
                let bytes = tensor.data();
                bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half::f16::from_bits(bits).to_f32()
                    })
                    .collect()
            }
            safetensors::Dtype::BF16 => {
                let bytes = tensor.data();
                bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half::bf16::from_bits(bits).to_f32()
                    })
                    .collect()
            }
            _ => {
                return Err(LoadError::SafeTensors(
                    format!("Unsupported dtype {:?} for tensor {}", dtype, name),
                ));
            }
        };

        result.insert(name.to_string(), (shape, f32_data));
    }

    Ok(result)
}

/// Load V-JEPA 2 ViT weights into a VisionEncoder.
///
/// Expected key patterns:
/// - `vision_encoder.patch_embed.proj.weight` → PatchEmbed weight
/// - `vision_encoder.patch_embed.proj.bias` → PatchEmbed bias
/// - `vision_encoder.cls_token` → CLS token
/// - `vision_encoder.pos_embed` → positional embeddings
/// - `vision_encoder.blocks.{i}.norm1.weight` → VitBlock norm1 weight
/// - `vision_encoder.blocks.{i}.attn.qkv.weight` → MHA QKV weight
/// - etc.
pub fn load_vit_weights(
    encoder: &mut crate::vit::VisionEncoder,
    path: &Path,
) -> Result<(), LoadError> {
    let tensors = load_safetensors(path)?;

    // Helper to load a tensor by key
    let get = |key: &str| -> Result<&Vec<f32>, LoadError> {
        tensors
            .get(key)
            .map(|(_, data)| data)
            .ok_or_else(|| LoadError::MissingKey(key.to_string()))
    };

    // Patch embed
    if let Ok(w) = get("vision_encoder.patch_embed.proj.weight") {
        if w.len() == encoder.patch_embed.weight.len() {
            encoder.patch_embed.weight.copy_from_slice(w);
        }
    }
    if let Ok(b) = get("vision_encoder.patch_embed.proj.bias") {
        if b.len() == encoder.patch_embed.bias.len() {
            encoder.patch_embed.bias.copy_from_slice(b);
        }
    }

    // CLS token
    if let Ok(cls) = get("vision_encoder.cls_token") {
        let len = encoder.cls_token.len();
        if cls.len() >= len {
            encoder.cls_token.copy_from_slice(&cls[..len]);
        }
    }

    // Positional embeddings
    if let Ok(pos) = get("vision_encoder.pos_embed") {
        let len = encoder.pos_embed.len();
        if pos.len() >= len {
            encoder.pos_embed.copy_from_slice(&pos[..len]);
        }
    }

    // Blocks
    for (i, block) in encoder.blocks.iter_mut().enumerate() {
        let prefix = format!("vision_encoder.blocks.{}", i);

        // norm1
        if let Ok(w) = get(&format!("{}.norm1.weight", prefix)) {
            if w.len() == block.norm1.weight.len() {
                block.norm1.weight.copy_from_slice(w);
            }
        }
        if let Ok(b) = get(&format!("{}.norm1.bias", prefix)) {
            if b.len() == block.norm1.bias.len() {
                block.norm1.bias.copy_from_slice(b);
            }
        }

        // attention QKV
        if let Ok(w) = get(&format!("{}.attn.qkv.weight", prefix)) {
            if w.len() == block.attn.qkv_weight.len() {
                block.attn.qkv_weight.copy_from_slice(w);
            }
        }
        if let Ok(b) = get(&format!("{}.attn.qkv.bias", prefix)) {
            if b.len() == block.attn.qkv_bias.len() {
                block.attn.qkv_bias.copy_from_slice(b);
            }
        }

        // attention out proj
        if let Ok(w) = get(&format!("{}.attn.proj.weight", prefix)) {
            if w.len() == block.attn.out_weight.len() {
                block.attn.out_weight.copy_from_slice(w);
            }
        }
        if let Ok(b) = get(&format!("{}.attn.proj.bias", prefix)) {
            if b.len() == block.attn.out_bias.len() {
                block.attn.out_bias.copy_from_slice(b);
            }
        }

        // norm2
        if let Ok(w) = get(&format!("{}.norm2.weight", prefix)) {
            if w.len() == block.norm2.weight.len() {
                block.norm2.weight.copy_from_slice(w);
            }
        }
        if let Ok(b) = get(&format!("{}.norm2.bias", prefix)) {
            if b.len() == block.norm2.bias.len() {
                block.norm2.bias.copy_from_slice(b);
            }
        }

        // FFN
        if let Ok(w) = get(&format!("{}.mlp.fc1.weight", prefix)) {
            if w.len() == block.ffn.w1.len() {
                block.ffn.w1.copy_from_slice(w);
            }
        }
        if let Ok(b) = get(&format!("{}.mlp.fc1.bias", prefix)) {
            if b.len() == block.ffn.b1.len() {
                block.ffn.b1.copy_from_slice(b);
            }
        }
        if let Ok(w) = get(&format!("{}.mlp.fc2.weight", prefix)) {
            if w.len() == block.ffn.w2.len() {
                block.ffn.w2.copy_from_slice(w);
            }
        }
        if let Ok(b) = get(&format!("{}.mlp.fc2.bias", prefix)) {
            if b.len() == block.ffn.b2.len() {
                block.ffn.b2.copy_from_slice(b);
            }
        }
    }

    // Final norm
    if let Ok(w) = get("vision_encoder.norm.weight") {
        if w.len() == encoder.final_norm.weight.len() {
            encoder.final_norm.weight.copy_from_slice(w);
        }
    }
    if let Ok(b) = get("vision_encoder.norm.bias") {
        if b.len() == encoder.final_norm.bias.len() {
            encoder.final_norm.bias.copy_from_slice(b);
        }
    }

    // Freeze
    encoder.frozen = true;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_error_display() {
        let e = LoadError::MissingKey("foo.bar".into());
        assert!(format!("{}", e).contains("foo.bar"));
    }
}
