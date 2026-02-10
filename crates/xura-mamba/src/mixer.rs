//! MixerModel — backbone with stacked Mamba/Mamba2 blocks.
//!
//! Implements the `MixerModel` from `mamba_ssm/models/mixer_seq_simple.py`:
//! Embedding → N × (Norm → Mamba Block + Residual) → Final Norm


use crate::cache::InferenceParams;
use crate::config::MambaConfig;
use crate::mamba::Mamba;
use crate::mamba2::Mamba2;
use crate::mamba3::Mamba3;
use crate::s4_block::S4Block;

/// Which SSM variant a layer uses.
pub enum MixerLayer {
    Mamba1(Mamba),
    Mamba2(Mamba2),
    Mamba3(Mamba3),
    S4(S4Block),
}

impl MixerLayer {
    /// Forward for training (full sequence).
    pub fn forward_train(&self, input: &[f32], batch: usize, seq_len: usize) -> Vec<f32> {
        match self {
            MixerLayer::Mamba1(m) => m.forward_train(input, batch, seq_len),
            MixerLayer::Mamba2(m) => m.forward_train(input, batch, seq_len),
            MixerLayer::Mamba3(m) => m.forward_train(input, batch, seq_len),
            MixerLayer::S4(s) => s.forward_train(input, batch, seq_len),
        }
    }

    /// Forward for single-step decoding.
    pub fn forward_step(
        &self,
        input: &[f32],
        batch: usize,
        conv_state: &mut [f32],
        ssm_state: &mut [f32],
    ) -> Vec<f32> {
        match self {
            MixerLayer::Mamba1(m) => m.forward_step(input, batch, conv_state, ssm_state),
            MixerLayer::Mamba2(m) => m.forward_step(input, batch, conv_state, ssm_state),
            MixerLayer::Mamba3(m) => m.forward_step(input, batch, conv_state, ssm_state),
            MixerLayer::S4(s) => {
                // S4 uses ssm_state for its complex recurrent state
                s.forward_step(input, batch, ssm_state)
            }
        }
    }

    /// Conv state size for a given batch.
    pub fn conv_state_size(&self, batch: usize) -> usize {
        match self {
            MixerLayer::Mamba1(m) => m.conv_state_size(batch),
            MixerLayer::Mamba2(m) => m.conv_state_size(batch),
            MixerLayer::Mamba3(m) => m.conv_state_size(batch),
            MixerLayer::S4(_) => 0, // S4 has no separate conv state
        }
    }

    /// SSM state size for a given batch.
    pub fn ssm_state_size(&self, batch: usize) -> usize {
        match self {
            MixerLayer::Mamba1(m) => m.ssm_state_size(batch),
            MixerLayer::Mamba2(m) => m.ssm_state_size(batch),
            MixerLayer::Mamba3(m) => m.ssm_state_size(batch),
            MixerLayer::S4(s) => s.state_size(batch),
        }
    }
}

/// RMSNorm operating on flat f32 slices.
pub struct RMSNormFlat {
    pub weight: Vec<f32>,
    eps: f32,
    d: usize,
}

impl RMSNormFlat {
    pub fn new(d: usize, eps: f32) -> Self {
        Self { weight: vec![1.0f32; d], eps, d }
    }

    /// Normalize input of shape (n_rows, d).
    pub fn forward(&self, x: &[f32], n_rows: usize) -> Vec<f32> {
        let d = self.d;
        let mut out = vec![0.0f32; n_rows * d];
        for row in 0..n_rows {
            let start = row * d;
            let slice = &x[start..start + d];
            let ms: f32 = slice.iter().map(|v| v * v).sum::<f32>() / d as f32;
            let inv_rms = 1.0 / (ms + self.eps).sqrt();
            for j in 0..d {
                out[start + j] = slice[j] * inv_rms * self.weight[j];
            }
        }
        out
    }
}

/// MixerModel backbone: Embedding → N layers → Final RMSNorm.
pub struct MixerModel {
    pub config: MambaConfig,
    /// Embedding table: (vocab_size, d_model)
    pub embedding: Vec<f32>,
    pub vocab_size: usize,
    /// Stacked layers
    pub layers: Vec<MixerLayer>,
    /// Per-layer pre-norm
    pub norms: Vec<RMSNormFlat>,
    /// Final norm
    pub final_norm: RMSNormFlat,
}

impl MixerModel {
    /// Build a new MixerModel from config with random weights.
    pub fn new(config: MambaConfig) -> Self {
        let vocab_size = config.padded_vocab_size();
        let d = config.d_model;

        // Embedding: Xavier init
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let emb_std = 0.02f32;
        let embedding: Vec<f32> = (0..vocab_size * d)
            .map(|_| rng.gen_range(-emb_std..emb_std))
            .collect();

        // Create layers
        let mut layers = Vec::with_capacity(config.n_layer);
        let mut norms = Vec::with_capacity(config.n_layer);

        for i in 0..config.n_layer {
            let layer = match config.ssm_cfg.layer.as_str() {
                "Mamba2" => MixerLayer::Mamba2(Mamba2::new_full(
                    d,
                    config.ssm_cfg.d_state,
                    config.ssm_cfg.d_conv,
                    config.ssm_cfg.expand,
                    config.ssm_cfg.headdim,
                    None,
                    config.ssm_cfg.ngroups,
                    false,
                    config.rms_norm,
                    false,
                    false,
                    true,
                    config.ssm_cfg.chunk_size,
                    Some(i),
                )),
                "Mamba3" | "mamba3" => MixerLayer::Mamba3(Mamba3::new_full(
                    d,
                    config.ssm_cfg.d_state,
                    config.ssm_cfg.expand,
                    config.ssm_cfg.headdim,
                    None,
                    config.ssm_cfg.ngroups,
                    config.ssm_cfg.use_rope,
                    config.rms_norm,
                    false,
                    false,
                    false,
                    true,
                    config.ssm_cfg.trapezoidal_alpha,
                    Some(i),
                )),
                "S4" | "S4D" | "s4" | "s4d" => {
                    let mut block = S4Block::new(
                        d,
                        config.ssm_cfg.d_state,
                        &config.ssm_cfg.s4_mode,
                        &config.ssm_cfg.s4_init,
                        &config.ssm_cfg.s4_activation,
                        config.ssm_cfg.s4_dt_min,
                        config.ssm_cfg.s4_dt_max,
                        Some(i),
                    );
                    block.setup_step();
                    MixerLayer::S4(block)
                }
                _ => MixerLayer::Mamba1(Mamba::new_full(
                    d,
                    config.ssm_cfg.d_state,
                    config.ssm_cfg.d_conv,
                    config.ssm_cfg.expand,
                    None,
                    true,
                    false,
                    Some(i),
                )),
            };
            layers.push(layer);
            norms.push(RMSNormFlat::new(d, config.norm_epsilon));
        }

        let final_norm = RMSNormFlat::new(d, config.norm_epsilon);

        Self { config, embedding, vocab_size, layers, norms, final_norm }
    }

    /// Look up embeddings for token IDs.
    /// Returns flat f32 vec of shape (batch, seq_len, d_model).
    fn embed(&self, token_ids: &[usize], batch: usize, seq_len: usize) -> Vec<f32> {
        let d = self.config.d_model;
        let mut out = vec![0.0f32; batch * seq_len * d];
        for b in 0..batch {
            for l in 0..seq_len {
                let tid = token_ids[b * seq_len + l];
                let emb_off = tid * d;
                let out_off = b * seq_len * d + l * d;
                out[out_off..out_off + d].copy_from_slice(&self.embedding[emb_off..emb_off + d]);
            }
        }
        out
    }

    /// Forward pass for training/prefill.
    ///
    /// `token_ids`: shape (batch * seq_len,) flattened
    /// Returns: hidden states, shape (batch, seq_len, d_model)
    pub fn forward(
        &self,
        token_ids: &[usize],
        batch: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let d = self.config.d_model;
        let n = batch * seq_len;

        let mut hidden = self.embed(token_ids, batch, seq_len);

        for (i, layer) in self.layers.iter().enumerate() {
            // Pre-norm
            let normed = self.norms[i].forward(&hidden, n);
            // Mixer
            let mixed = layer.forward_train(&normed, batch, seq_len);
            // Residual
            for j in 0..hidden.len() {
                hidden[j] += mixed[j];
            }
        }

        // Final norm
        self.final_norm.forward(&hidden, n)
    }

    /// Forward pass for single-step decoding with inference cache.
    ///
    /// `token_ids`: shape (batch,) — one token per batch element
    /// Returns: hidden states, shape (batch, d_model)
    pub fn forward_step(
        &self,
        token_ids: &[usize],
        batch: usize,
        inference_params: &mut InferenceParams,
    ) -> Vec<f32> {
        let d = self.config.d_model;
        // Embed single tokens
        let mut hidden = vec![0.0f32; batch * d];
        for b in 0..batch {
            let tid = token_ids[b];
            let emb_off = tid * d;
            let out_off = b * d;
            hidden[out_off..out_off + d].copy_from_slice(&self.embedding[emb_off..emb_off + d]);
        }

        for (i, layer) in self.layers.iter().enumerate() {
            // Ensure state is allocated
            let conv_size = layer.conv_state_size(batch);
            let ssm_size = layer.ssm_state_size(batch);
            inference_params.get_or_create_state(i, conv_size, ssm_size);

            // Pre-norm
            let normed = self.norms[i].forward(&hidden, batch);

            // Get mutable state
            let state = inference_params.get_state_mut(i)
                .expect("mixer: inference state not found for layer index");
            let mixed = layer.forward_step(
                &normed, batch,
                &mut state.conv_state,
                &mut state.ssm_state,
            );

            // Residual
            for j in 0..hidden.len() {
                hidden[j] += mixed[j];
            }
        }

        // Final norm
        self.final_norm.forward(&hidden, batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixer_model_forward() {
        let config = MambaConfig::tiny();
        let model = MixerModel::new(config);
        let batch = 2;
        let seq_len = 4;
        let tokens: Vec<usize> = (0..batch * seq_len).map(|i| i % 256).collect();
        let output = model.forward(&tokens, batch, seq_len);
        assert_eq!(output.len(), batch * seq_len * 64);
    }

    #[test]
    fn test_mixer_model_step() {
        let config = MambaConfig::tiny();
        let model = MixerModel::new(config);
        let batch = 1;
        let mut params = InferenceParams::new();

        let tokens = vec![42usize];
        let output = model.forward_step(&tokens, batch, &mut params);
        assert_eq!(output.len(), batch * 64);

        // Second step should also work
        let tokens2 = vec![10usize];
        let output2 = model.forward_step(&tokens2, batch, &mut params);
        assert_eq!(output2.len(), batch * 64);
    }

    #[test]
    fn test_mixer_model_s4d() {
        let config = MambaConfig::tiny_s4d();
        let model = MixerModel::new(config);
        let batch = 1;
        let seq_len = 4;
        let tokens: Vec<usize> = vec![0, 1, 2, 3];
        let output = model.forward(&tokens, batch, seq_len);
        assert_eq!(output.len(), batch * seq_len * 64);
    }

    #[test]
    fn test_mixer_model_s4d_step() {
        let config = MambaConfig::tiny_s4d();
        let model = MixerModel::new(config);
        let batch = 1;
        let mut params = InferenceParams::new();
        let tokens = vec![42usize];
        let output = model.forward_step(&tokens, batch, &mut params);
        assert_eq!(output.len(), batch * 64);
    }

    #[test]
    fn test_mixer_model_mamba3() {
        let config = MambaConfig::tiny_mamba3();
        let model = MixerModel::new(config);
        let batch = 1;
        let seq_len = 4;
        let tokens: Vec<usize> = vec![0, 1, 2, 3];
        let output = model.forward(&tokens, batch, seq_len);
        assert_eq!(output.len(), batch * seq_len * 64);
    }

    #[test]
    fn test_mixer_model_mamba3_step() {
        let config = MambaConfig::tiny_mamba3();
        let model = MixerModel::new(config);
        let batch = 1;
        let mut params = InferenceParams::new();
        let tokens = vec![42usize];
        let output = model.forward_step(&tokens, batch, &mut params);
        assert_eq!(output.len(), batch * 64);

        let tokens2 = vec![10usize];
        let output2 = model.forward_step(&tokens2, batch, &mut params);
        assert_eq!(output2.len(), batch * 64);
    }

    #[test]
    fn test_mixer_model_mamba2() {
        let config = MambaConfig::tiny_mamba2();
        let model = MixerModel::new(config);
        let batch = 1;
        let seq_len = 4;
        let tokens: Vec<usize> = vec![0, 1, 2, 3];
        let output = model.forward(&tokens, batch, seq_len);
        assert_eq!(output.len(), batch * seq_len * 64);
    }
}
