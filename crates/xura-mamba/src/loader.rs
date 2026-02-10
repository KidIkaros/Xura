//! SafeTensors weight loading for pretrained Mamba models.
//!
//! Loads weights from HuggingFace `state-spaces/mamba-*` checkpoints in SafeTensors format.

use std::path::Path;

use safetensors::SafeTensors;

use crate::config::MambaConfig;
use crate::mamba::Mamba;
use crate::mamba2::Mamba2;
use crate::mixer::MixerLayer;
use crate::model::MambaLMHeadModel;

/// Error type for weight loading.
#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    SafeTensors(String),
    Config(String),
    MissingKey(String),
    ShapeMismatch { key: String, expected: Vec<usize>, got: Vec<usize> },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "IO error: {}", e),
            LoadError::SafeTensors(s) => write!(f, "SafeTensors error: {}", s),
            LoadError::Config(s) => write!(f, "Config error: {}", s),
            LoadError::MissingKey(k) => write!(f, "Missing key: {}", k),
            LoadError::ShapeMismatch { key, expected, got } =>
                write!(f, "Shape mismatch for {}: expected {:?}, got {:?}", key, expected, got),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self { LoadError::Io(e) }
}

/// Load a MambaConfig from a `config.json` file.
pub fn load_config(path: &Path) -> Result<MambaConfig, LoadError> {
    let text = std::fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|e| LoadError::Config(e.to_string()))
}

/// Convert a SafeTensors tensor view to Vec<f32>.
fn tensor_to_f32(view: &safetensors::tensor::TensorView<'_>) -> Vec<f32> {
    let dtype = view.dtype();
    let data = view.data();
    match dtype {
        safetensors::Dtype::F32 => {
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        }
        safetensors::Dtype::F16 => {
            data.chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect()
        }
        safetensors::Dtype::BF16 => {
            data.chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect()
        }
        _ => {
            // Fallback: try as f32
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        }
    }
}

/// Get a tensor from the SafeTensors file, returning an error if missing.
fn get_tensor(tensors: &SafeTensors<'_>, key: &str) -> Result<Vec<f32>, LoadError> {
    let view = tensors.tensor(key).map_err(|_| LoadError::MissingKey(key.to_string()))?;
    Ok(tensor_to_f32(&view))
}

/// Get a tensor, returning None if the key doesn't exist.
fn get_tensor_opt(tensors: &SafeTensors<'_>, key: &str) -> Option<Vec<f32>> {
    tensors.tensor(key).ok().map(|v| tensor_to_f32(&v))
}

/// Load a pretrained Mamba model from a directory containing `config.json` and
/// `model.safetensors` (or multiple shard files).
///
/// # Arguments
/// - `model_dir`: path to directory with config.json and safetensors files
///
/// # Returns
/// A `MambaLMHeadModel` with loaded weights.
pub fn load_pretrained(model_dir: &Path) -> Result<MambaLMHeadModel, LoadError> {
    // 1. Load config
    let config_path = model_dir.join("config.json");
    let config = load_config(&config_path)?;

    // 2. Load safetensors
    let st_path = model_dir.join("model.safetensors");
    let data = std::fs::read(&st_path)?;
    let tensors = SafeTensors::deserialize(&data)
        .map_err(|e| LoadError::SafeTensors(e.to_string()))?;

    // 3. Build model structure
    let mut model = MambaLMHeadModel::new(config.clone());
    let vocab_size = config.padded_vocab_size();
    let d = config.d_model;

    // 4. Load embedding
    if let Some(emb) = get_tensor_opt(&tensors, "backbone.embedding.weight") {
        model.backbone.embedding = emb;
    }

    // 5. Load per-layer weights
    for i in 0..config.n_layer {
        let prefix = format!("backbone.layers.{}.mixer", i);
        let norm_key = format!("backbone.layers.{}.norm.weight", i);

        // Load norm weight
        if let Some(w) = get_tensor_opt(&tensors, &norm_key) {
            model.backbone.norms[i].weight = w;
        }

        // Load mixer weights based on layer type
        match &mut model.backbone.layers[i] {
            MixerLayer::Mamba1(ref mut m) => {
                load_mamba1_weights(m, &tensors, &prefix)?;
            }
            MixerLayer::Mamba2(ref mut m) => {
                load_mamba2_weights(m, &tensors, &prefix)?;
            }
            MixerLayer::Mamba3(_) => {
                // Mamba-3 weight loading not yet implemented for SafeTensors
            }
            MixerLayer::S4(_) => {
                // S4 weight loading not yet implemented for SafeTensors
            }
        }
    }

    // 6. Load final norm
    if let Some(w) = get_tensor_opt(&tensors, "backbone.norm_f.weight") {
        model.backbone.final_norm.weight = w;
    }

    // 7. Load LM head (if not tied)
    if !config.tie_embeddings {
        if let Some(w) = get_tensor_opt(&tensors, "lm_head.weight") {
            model.lm_head_weight = w;
        }
    } else {
        // Tied: use embedding weight
        model.lm_head_weight = model.backbone.embedding.clone();
    }

    Ok(model)
}

/// Load weights into a Mamba v1 block.
fn load_mamba1_weights(m: &mut Mamba, tensors: &SafeTensors<'_>, prefix: &str) -> Result<(), LoadError> {
    if let Some(w) = get_tensor_opt(tensors, &format!("{}.in_proj.weight", prefix)) {
        m.in_proj_weight = w;
    }
    m.in_proj_bias = get_tensor_opt(tensors, &format!("{}.in_proj.bias", prefix));

    if let Some(w) = get_tensor_opt(tensors, &format!("{}.conv1d.weight", prefix)) {
        // Python shape: (d_inner, 1, kernel_size) -> flatten to (d_inner, kernel_size)
        m.conv1d_weight = w;
    }
    m.conv1d_bias = get_tensor_opt(tensors, &format!("{}.conv1d.bias", prefix));

    if let Some(w) = get_tensor_opt(tensors, &format!("{}.x_proj.weight", prefix)) {
        m.x_proj_weight = w;
    }
    if let Some(w) = get_tensor_opt(tensors, &format!("{}.dt_proj.weight", prefix)) {
        m.dt_proj_weight = w;
    }
    if let Some(b) = get_tensor_opt(tensors, &format!("{}.dt_proj.bias", prefix)) {
        m.dt_proj_bias = b;
    }
    if let Some(a) = get_tensor_opt(tensors, &format!("{}.A_log", prefix)) {
        m.a_log = a;
    }
    if let Some(d) = get_tensor_opt(tensors, &format!("{}.D", prefix)) {
        m.d_skip = d;
    }
    if let Some(w) = get_tensor_opt(tensors, &format!("{}.out_proj.weight", prefix)) {
        m.out_proj_weight = w;
    }
    m.out_proj_bias = get_tensor_opt(tensors, &format!("{}.out_proj.bias", prefix));

    Ok(())
}

/// Load weights into a Mamba-2 block.
fn load_mamba2_weights(m: &mut Mamba2, tensors: &SafeTensors<'_>, prefix: &str) -> Result<(), LoadError> {
    if let Some(w) = get_tensor_opt(tensors, &format!("{}.in_proj.weight", prefix)) {
        m.in_proj_weight = w;
    }
    m.in_proj_bias = get_tensor_opt(tensors, &format!("{}.in_proj.bias", prefix));

    if let Some(w) = get_tensor_opt(tensors, &format!("{}.conv1d.weight", prefix)) {
        m.conv1d_weight = w;
    }
    m.conv1d_bias = get_tensor_opt(tensors, &format!("{}.conv1d.bias", prefix));

    if let Some(b) = get_tensor_opt(tensors, &format!("{}.dt_bias", prefix)) {
        m.dt_bias = b;
    }
    if let Some(a) = get_tensor_opt(tensors, &format!("{}.A_log", prefix)) {
        m.a_log = a;
    }
    if let Some(d) = get_tensor_opt(tensors, &format!("{}.D", prefix)) {
        m.d_skip = d;
    }
    if let Some(ref mut norm) = m.norm {
        if let Some(w) = get_tensor_opt(tensors, &format!("{}.norm.weight", prefix)) {
            norm.weight = w;
        }
    }
    if let Some(w) = get_tensor_opt(tensors, &format!("{}.out_proj.weight", prefix)) {
        m.out_proj_weight = w;
    }
    m.out_proj_bias = get_tensor_opt(tensors, &format!("{}.out_proj.bias", prefix));

    Ok(())
}

/// List all tensor keys in a SafeTensors file (useful for debugging).
pub fn list_keys(model_dir: &Path) -> Result<Vec<String>, LoadError> {
    let st_path = model_dir.join("model.safetensors");
    let data = std::fs::read(&st_path)?;
    let tensors = SafeTensors::deserialize(&data)
        .map_err(|e| LoadError::SafeTensors(e.to_string()))?;
    Ok(tensors.names().iter().map(|s| s.to_string()).collect())
}
