//! Mamba model configuration.

use serde::{Deserialize, Serialize};

/// SSM-specific configuration within a Mamba block.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SsmConfig {
    /// Which SSM layer to use: "Mamba1", "Mamba2", "Mamba3", "S4", or "S4D"
    #[serde(default = "default_ssm_layer")]
    pub layer: String,
    /// SSM state expansion factor
    #[serde(default = "default_d_state")]
    pub d_state: usize,
    /// Local convolution width
    #[serde(default = "default_d_conv")]
    pub d_conv: usize,
    /// Block expansion factor
    #[serde(default = "default_expand")]
    pub expand: usize,
    /// Head dimension (Mamba-2 only)
    #[serde(default = "default_headdim")]
    pub headdim: usize,
    /// Number of groups (Mamba-2 only)
    #[serde(default = "default_ngroups")]
    pub ngroups: usize,
    /// Chunk size for SSD scan (Mamba-2 only)
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    /// S4 kernel mode: "s4d" (diagonal) or "s4" (DPLR). Only used when layer is "S4" or "S4D".
    #[serde(default = "default_s4_mode")]
    pub s4_mode: String,
    /// S4 initialization: "legs", "diag-inv", "diag-lin", etc.
    #[serde(default = "default_s4_init")]
    pub s4_init: String,
    /// S4 activation function: "gelu", "silu", "id"
    #[serde(default = "default_s4_activation")]
    pub s4_activation: String,
    /// S4 dt minimum
    #[serde(default = "default_s4_dt_min")]
    pub s4_dt_min: f32,
    /// S4 dt maximum
    #[serde(default = "default_s4_dt_max")]
    pub s4_dt_max: f32,
    /// Mamba-3: use data-dependent RoPE (equivalent to complex A)
    #[serde(default = "default_use_rope")]
    pub use_rope: bool,
    /// Mamba-3: trapezoidal interpolation parameter (0.5 = standard trapezoidal)
    #[serde(default = "default_trapezoidal_alpha")]
    pub trapezoidal_alpha: f32,
}

fn default_ssm_layer() -> String { "Mamba1".to_string() }
fn default_d_state() -> usize { 16 }
fn default_d_conv() -> usize { 4 }
fn default_expand() -> usize { 2 }
fn default_headdim() -> usize { 64 }
fn default_ngroups() -> usize { 1 }
fn default_chunk_size() -> usize { 256 }
fn default_s4_mode() -> String { "s4d".to_string() }
fn default_s4_init() -> String { "diag-inv".to_string() }
fn default_s4_activation() -> String { "gelu".to_string() }
fn default_s4_dt_min() -> f32 { 0.001 }
fn default_s4_dt_max() -> f32 { 0.1 }
fn default_use_rope() -> bool { true }
fn default_trapezoidal_alpha() -> f32 { 0.5 }

impl Default for SsmConfig {
    fn default() -> Self {
        Self {
            layer: default_ssm_layer(),
            d_state: default_d_state(),
            d_conv: default_d_conv(),
            expand: default_expand(),
            headdim: default_headdim(),
            ngroups: default_ngroups(),
            chunk_size: default_chunk_size(),
            s4_mode: default_s4_mode(),
            s4_init: default_s4_init(),
            s4_activation: default_s4_activation(),
            s4_dt_min: default_s4_dt_min(),
            s4_dt_max: default_s4_dt_max(),
            use_rope: default_use_rope(),
            trapezoidal_alpha: default_trapezoidal_alpha(),
        }
    }
}

/// Top-level Mamba model configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MambaConfig {
    pub d_model: usize,
    pub n_layer: usize,
    #[serde(default)]
    pub d_intermediate: usize,
    pub vocab_size: usize,
    #[serde(default)]
    pub ssm_cfg: SsmConfig,
    #[serde(default = "default_rms_norm")]
    pub rms_norm: bool,
    #[serde(default)]
    pub residual_in_fp32: bool,
    #[serde(default)]
    pub fused_add_norm: bool,
    #[serde(default = "default_pad_vocab_size_multiple")]
    pub pad_vocab_size_multiple: usize,
    #[serde(default = "default_tie_embeddings")]
    pub tie_embeddings: bool,
    #[serde(default = "default_norm_epsilon")]
    pub norm_epsilon: f32,
}

fn default_rms_norm() -> bool { true }
fn default_pad_vocab_size_multiple() -> usize { 1 }
fn default_tie_embeddings() -> bool { true }
fn default_norm_epsilon() -> f32 { 1e-5 }

impl MambaConfig {
    /// Padded vocab size (rounded up to pad_vocab_size_multiple).
    pub fn padded_vocab_size(&self) -> usize {
        let v = self.vocab_size;
        let m = self.pad_vocab_size_multiple;
        if m <= 1 || v % m == 0 {
            v
        } else {
            v + m - (v % m)
        }
    }

    /// Mamba-130m preset.
    pub fn mamba_130m() -> Self {
        Self {
            d_model: 768,
            n_layer: 24,
            d_intermediate: 0,
            vocab_size: 50280,
            ssm_cfg: SsmConfig { layer: "Mamba1".into(), d_state: 16, ..Default::default() },
            rms_norm: true,
            residual_in_fp32: true,
            fused_add_norm: false,
            pad_vocab_size_multiple: 8,
            tie_embeddings: true,
            norm_epsilon: 1e-5,
        }
    }

    /// Mamba-370m preset.
    pub fn mamba_370m() -> Self {
        Self {
            d_model: 1024,
            n_layer: 48,
            d_intermediate: 0,
            vocab_size: 50280,
            ssm_cfg: SsmConfig { layer: "Mamba1".into(), d_state: 16, ..Default::default() },
            rms_norm: true,
            residual_in_fp32: true,
            fused_add_norm: false,
            pad_vocab_size_multiple: 8,
            tie_embeddings: true,
            norm_epsilon: 1e-5,
        }
    }

    /// Mamba2-2.7b preset.
    pub fn mamba2_2_7b() -> Self {
        Self {
            d_model: 2560,
            n_layer: 64,
            d_intermediate: 0,
            vocab_size: 50280,
            ssm_cfg: SsmConfig {
                layer: "Mamba2".into(),
                d_state: 128,
                headdim: 64,
                ngroups: 1,
                chunk_size: 256,
                ..Default::default()
            },
            rms_norm: true,
            residual_in_fp32: true,
            fused_add_norm: false,
            pad_vocab_size_multiple: 16,
            tie_embeddings: true,
            norm_epsilon: 1e-5,
        }
    }

    /// A tiny config for unit tests (~50K params).
    pub fn tiny() -> Self {
        Self {
            d_model: 64,
            n_layer: 2,
            d_intermediate: 0,
            vocab_size: 256,
            ssm_cfg: SsmConfig {
                layer: "Mamba1".into(),
                d_state: 8,
                d_conv: 4,
                expand: 2,
                ..Default::default()
            },
            rms_norm: true,
            residual_in_fp32: false,
            fused_add_norm: false,
            pad_vocab_size_multiple: 1,
            tie_embeddings: true,
            norm_epsilon: 1e-5,
        }
    }

    /// S4D tiny config for unit tests.
    pub fn tiny_s4d() -> Self {
        Self {
            d_model: 64,
            n_layer: 2,
            d_intermediate: 0,
            vocab_size: 256,
            ssm_cfg: SsmConfig {
                layer: "S4D".into(),
                d_state: 32,
                s4_mode: "s4d".into(),
                s4_init: "diag-inv".into(),
                s4_activation: "gelu".into(),
                s4_dt_min: 0.001,
                s4_dt_max: 0.1,
                ..Default::default()
            },
            rms_norm: true,
            residual_in_fp32: false,
            fused_add_norm: false,
            pad_vocab_size_multiple: 1,
            tie_embeddings: true,
            norm_epsilon: 1e-5,
        }
    }

    /// S4-DPLR tiny config for unit tests.
    pub fn tiny_s4() -> Self {
        Self {
            d_model: 64,
            n_layer: 2,
            d_intermediate: 0,
            vocab_size: 256,
            ssm_cfg: SsmConfig {
                layer: "S4".into(),
                d_state: 32,
                s4_mode: "s4".into(),
                s4_init: "legs".into(),
                s4_activation: "gelu".into(),
                s4_dt_min: 0.001,
                s4_dt_max: 0.1,
                ..Default::default()
            },
            rms_norm: true,
            residual_in_fp32: false,
            fused_add_norm: false,
            pad_vocab_size_multiple: 1,
            tie_embeddings: true,
            norm_epsilon: 1e-5,
        }
    }

    /// A tiny Mamba-3 config for unit tests.
    pub fn tiny_mamba3() -> Self {
        Self {
            d_model: 64,
            n_layer: 2,
            d_intermediate: 0,
            vocab_size: 256,
            ssm_cfg: SsmConfig {
                layer: "Mamba3".into(),
                d_state: 16,
                expand: 2,
                headdim: 16,
                ngroups: 1,
                use_rope: true,
                trapezoidal_alpha: 0.5,
                ..Default::default()
            },
            rms_norm: true,
            residual_in_fp32: false,
            fused_add_norm: false,
            pad_vocab_size_multiple: 1,
            tie_embeddings: true,
            norm_epsilon: 1e-5,
        }
    }

    /// A tiny Mamba-2 config for unit tests.
    pub fn tiny_mamba2() -> Self {
        Self {
            d_model: 64,
            n_layer: 2,
            d_intermediate: 0,
            vocab_size: 256,
            ssm_cfg: SsmConfig {
                layer: "Mamba2".into(),
                d_state: 16,
                d_conv: 4,
                expand: 2,
                headdim: 16,
                ngroups: 1,
                chunk_size: 64,
                ..Default::default()
            },
            rms_norm: true,
            residual_in_fp32: false,
            fused_add_norm: false,
            pad_vocab_size_multiple: 1,
            tie_embeddings: true,
            norm_epsilon: 1e-5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_padded_vocab_size() {
        let cfg = MambaConfig::mamba_130m();
        assert_eq!(cfg.padded_vocab_size(), 50280); // already divisible by 8
    }

    #[test]
    fn test_padded_vocab_rounding() {
        let mut cfg = MambaConfig::tiny();
        cfg.vocab_size = 257;
        cfg.pad_vocab_size_multiple = 8;
        assert_eq!(cfg.padded_vocab_size(), 264);
    }

    #[test]
    fn test_deserialize_config() {
        let json = r#"{
            "d_model": 768,
            "n_layer": 24,
            "vocab_size": 50280,
            "ssm_cfg": {"layer": "Mamba1", "d_state": 16}
        }"#;
        let cfg: MambaConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.d_model, 768);
        assert_eq!(cfg.ssm_cfg.d_state, 16);
        assert!(cfg.rms_norm); // default
    }
}
