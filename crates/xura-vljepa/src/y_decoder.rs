//! Mamba-3 Decoder (Y-Decoder) — lightweight text generation from predicted embeddings.
//!
//! Wraps `MambaLMHeadModel` with prefix conditioning: the predicted embedding
//! is projected into a sequence of prefix tokens that prime the Mamba-3 decoder.

use rand::Rng;

use xura_mamba::{InferenceParams, MambaLMHeadModel};

use crate::config::Mamba3DecoderConfig;

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

/// Mamba-3 Decoder with prefix conditioning.
///
/// Converts a predicted embedding into text by:
/// 1. Projecting the embedding into `prefix_len` pseudo-tokens
/// 2. Running those through the Mamba-3 LM to prime the hidden state
/// 3. Autoregressively generating text tokens
pub struct Mamba3Decoder {
    pub config: Mamba3DecoderConfig,
    /// Projects predicted embedding → prefix_len × d_model.
    embed_to_prefix: Linear,
    /// Mamba-3 language model.
    lm: MambaLMHeadModel,
}

impl Mamba3Decoder {
    pub fn new(config: Mamba3DecoderConfig) -> Self {
        let embed_to_prefix = Linear::new(config.embed_dim, config.prefix_len * config.d_model);
        let mamba_config = config.to_mamba_config();
        let lm = MambaLMHeadModel::new(mamba_config);

        Self {
            config,
            embed_to_prefix,
            lm,
        }
    }

    /// Generate text from a predicted embedding.
    ///
    /// # Arguments
    /// - `predicted_embedding`: shape (embed_dim,) — single sample
    /// - `max_tokens`: maximum number of tokens to generate
    /// - `bos_token`: beginning-of-sequence token ID
    ///
    /// # Returns
    /// Generated token IDs.
    pub fn generate(
        &self,
        predicted_embedding: &[f32],
        max_tokens: usize,
        bos_token: usize,
    ) -> Vec<usize> {
        let dm = self.config.d_model;
        let prefix_len = self.config.prefix_len;

        // 1) Project embedding → prefix tokens: (prefix_len, d_model)
        let prefix_flat = self.embed_to_prefix.forward(predicted_embedding, 1);

        // 2) Prime the Mamba-3 hidden state by running prefix through backbone
        let mut inference_params = InferenceParams::new();
        self.prime_with_prefix(&prefix_flat, prefix_len, &mut inference_params);

        // 3) Autoregressive generation starting from BOS
        let mut tokens = vec![bos_token];
        let mut current_token = bos_token;

        for _ in 0..max_tokens {
            let hidden = self
                .lm
                .backbone
                .forward_step(&[current_token], 1, &mut inference_params);

            // Project to vocab logits
            let vocab_size = self.lm.backbone.vocab_size;
            let mut logits = vec![0.0f32; vocab_size];
            for v in 0..vocab_size {
                let mut acc = 0.0f32;
                for d in 0..dm {
                    acc += hidden[d] * self.lm.backbone.embedding[v * dm + d];
                }
                logits[v] = acc;
            }

            // Greedy decode (argmax)
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            tokens.push(next_token);
            current_token = next_token;

            inference_params.advance(1);
        }

        tokens
    }

    /// Prime the decoder's hidden state with prefix tokens.
    ///
    /// Runs the prefix sequence through the backbone layers to build up
    /// the Mamba-3 SSM state (trapezoidal h_t + prev_bx).
    fn prime_with_prefix(
        &self,
        prefix: &[f32],
        prefix_len: usize,
        inference_params: &mut InferenceParams,
    ) {
        let dm = self.config.d_model;

        // Feed prefix tokens one at a time through the backbone
        // We need to find the nearest embedding for each prefix vector
        // and use that token ID, OR we can directly manipulate hidden states.
        //
        // Simpler approach: find nearest-neighbor token for each prefix vector
        // and feed those token IDs through the model.
        let vocab_size = self.lm.backbone.vocab_size;

        for p in 0..prefix_len {
            let prefix_vec = &prefix[p * dm..(p + 1) * dm];

            // Find nearest embedding
            let mut best_token = 0;
            let mut best_sim = f32::NEG_INFINITY;
            for v in 0..vocab_size {
                let mut dot = 0.0f32;
                for d in 0..dm {
                    dot += prefix_vec[d] * self.lm.backbone.embedding[v * dm + d];
                }
                if dot > best_sim {
                    best_sim = dot;
                    best_token = v;
                }
            }

            // Feed through backbone to update state
            let _ = self
                .lm
                .backbone
                .forward_step(&[best_token], 1, inference_params);
            inference_params.advance(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let config = Mamba3DecoderConfig::tiny();
        let decoder = Mamba3Decoder::new(config.clone());
        assert_eq!(decoder.config.prefix_len, 4);
    }

    #[test]
    fn test_decoder_generate() {
        let config = Mamba3DecoderConfig::tiny();
        let decoder = Mamba3Decoder::new(config.clone());

        let embedding = vec![0.1f32; config.embed_dim];
        let tokens = decoder.generate(&embedding, 5, 0);

        // Should have BOS + 5 generated tokens
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[0], 0); // BOS
    }

    #[test]
    fn test_embed_to_prefix_shape() {
        let config = Mamba3DecoderConfig::tiny();
        let decoder = Mamba3Decoder::new(config.clone());

        let embedding = vec![0.1f32; config.embed_dim];
        let prefix = decoder.embed_to_prefix.forward(&embedding, 1);
        assert_eq!(prefix.len(), config.prefix_len * config.d_model);
    }
}
