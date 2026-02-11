//! Mamba-3 Text Encoder (Y-Encoder) — replaces EmbeddingGemma-300M in VL-JEPA.
//!
//! Encodes target text into a continuous embedding in the shared latent space
//! using a Mamba-3 MixerModel backbone.

use rand::Rng;

use xura_mamba::MixerModel;

use crate::config::Mamba3TextEncoderConfig;

/// Linear projection layer.
struct Linear {
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl Linear {
    fn new(in_dim: usize, out_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (in_dim + out_dim) as f32).sqrt();
        Self {
            weight: (0..out_dim * in_dim)
                .map(|_| rng.gen_range(-std..std))
                .collect(),
            bias: vec![0.0f32; out_dim],
            in_dim,
            out_dim,
        }
    }

    fn forward(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n * self.out_dim];
        for i in 0..n {
            for o in 0..self.out_dim {
                let mut acc = self.bias[o];
                for k in 0..self.in_dim {
                    acc += x[i * self.in_dim + k] * self.weight[o * self.in_dim + k];
                }
                out[i * self.out_dim + o] = acc;
            }
        }
        out
    }
}

/// Mamba-3 Text Encoder.
///
/// Tokenized text → Mamba-3 MixerModel → average pool → projection → L2-normalize.
pub struct Mamba3TextEncoder {
    pub config: Mamba3TextEncoderConfig,
    /// Mamba-3 backbone (uses its own embedding table).
    pub backbone: MixerModel,
    /// Projection head: d_model → embed_dim.
    proj_head: Linear,
}

impl Mamba3TextEncoder {
    pub fn new(config: Mamba3TextEncoderConfig) -> Self {
        let mamba_config = config.to_mamba_config();
        let backbone = MixerModel::new(mamba_config);
        let proj_head = Linear::new(config.d_model, config.embed_dim);

        Self {
            config,
            backbone,
            proj_head,
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// - `token_ids`: shape (batch * seq_len,) flattened token IDs
    /// - `batch`: batch size
    /// - `seq_len`: sequence length
    ///
    /// # Returns
    /// Target embedding: shape (batch, embed_dim), L2-normalized.
    pub fn forward(&self, token_ids: &[usize], batch: usize, seq_len: usize) -> Vec<f32> {
        let dm = self.config.d_model;

        // 1) Run through Mamba-3 backbone: (batch, seq_len, d_model)
        let hidden = self.backbone.forward(token_ids, batch, seq_len);

        // 2) Average pool over sequence → (batch, d_model)
        let mut pooled = vec![0.0f32; batch * dm];
        for b in 0..batch {
            for pos in 0..seq_len {
                for d in 0..dm {
                    pooled[b * dm + d] += hidden[b * seq_len * dm + pos * dm + d];
                }
            }
            let inv_len = 1.0 / seq_len as f32;
            for d in 0..dm {
                pooled[b * dm + d] *= inv_len;
            }
        }

        // 3) Projection head → L2-normalize
        let projected = self.proj_head.forward(&pooled, batch);
        l2_normalize(&projected, batch, self.config.embed_dim)
    }
}

/// L2-normalize each row.
fn l2_normalize(data: &[f32], rows: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let start = r * dim;
        let norm: f32 = data[start..start + dim]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(1e-12);
        for d in 0..dim {
            out[start + d] = data[start + d] / norm;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_encoder_forward_shape() {
        let config = Mamba3TextEncoderConfig::tiny();
        let encoder = Mamba3TextEncoder::new(config.clone());

        let batch = 2;
        let seq_len = 8;
        let tokens: Vec<usize> = (0..batch * seq_len)
            .map(|i| i % config.vocab_size)
            .collect();

        let output = encoder.forward(&tokens, batch, seq_len);
        assert_eq!(output.len(), batch * config.embed_dim);
    }

    #[test]
    fn test_text_encoder_output_normalized() {
        let config = Mamba3TextEncoderConfig::tiny();
        let encoder = Mamba3TextEncoder::new(config.clone());

        let batch = 1;
        let seq_len = 4;
        let tokens: Vec<usize> = vec![1, 2, 3, 4];

        let output = encoder.forward(&tokens, batch, seq_len);

        let norm: f32 = output.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "output should be L2-normalized, got norm={}",
            norm
        );
    }
}
