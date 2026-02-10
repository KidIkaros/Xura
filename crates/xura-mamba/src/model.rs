//! MambaLMHeadModel — full language model with generation support.
//!
//! Implements the `MambaLMHeadModel` from `mamba_ssm/models/mixer_seq_simple.py`:
//! MixerModel backbone + LM head (Linear projection to vocab) + autoregressive generation.

use crate::cache::InferenceParams;
use crate::config::MambaConfig;
use crate::mixer::MixerModel;

/// Sampling configuration for text generation.
#[derive(Clone, Debug)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl SamplerConfig {
    /// Greedy decoding (argmax).
    pub fn greedy() -> Self {
        Self { temperature: 1.0, top_k: 1, top_p: 1.0, repetition_penalty: 1.0 }
    }

    /// Nucleus sampling with temperature.
    pub fn nucleus(temperature: f32, top_p: f32) -> Self {
        Self { temperature, top_k: 0, top_p, repetition_penalty: 1.0 }
    }
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self::greedy()
    }
}

/// Full Mamba language model: backbone + LM head.
pub struct MambaLMHeadModel {
    pub config: MambaConfig,
    pub backbone: MixerModel,
    /// LM head weight: (vocab_size, d_model) — may be tied with embedding
    pub lm_head_weight: Vec<f32>,
    pub tie_embeddings: bool,
}

impl MambaLMHeadModel {
    /// Build from config with random weights.
    pub fn new(config: MambaConfig) -> Self {
        let backbone = MixerModel::new(config.clone());
        let vocab_size = config.padded_vocab_size();
        let d = config.d_model;
        let tie = config.tie_embeddings;

        let lm_head_weight = if tie {
            // Tied: share embedding weight
            backbone.embedding.clone()
        } else {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let limit = (6.0 / (d + vocab_size) as f32).sqrt();
            (0..vocab_size * d).map(|_| rng.gen_range(-limit..limit)).collect()
        };

        Self { config, backbone, lm_head_weight, tie_embeddings: tie }
    }

    /// Forward pass: token_ids → logits.
    ///
    /// `token_ids`: shape (batch * seq_len,) flattened
    /// Returns: logits as flat f32 vec, shape (batch, seq_len, vocab_size)
    pub fn forward(
        &self,
        token_ids: &[usize],
        batch: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let d = self.config.d_model;
        let v = self.backbone.vocab_size;

        // Get hidden states: (batch, seq_len, d_model)
        let hidden = self.backbone.forward(token_ids, batch, seq_len);

        // LM head: hidden @ lm_head^T → (batch, seq_len, vocab_size)
        // lm_head_weight is (vocab_size, d_model)
        let n = batch * seq_len;
        let mut logits = vec![0.0f32; n * v];
        for i in 0..n {
            for j in 0..v {
                let mut acc = 0.0f32;
                for k in 0..d {
                    acc += hidden[i * d + k] * self.lm_head_weight[j * d + k];
                }
                logits[i * v + j] = acc;
            }
        }

        logits
    }

    /// Forward pass for single-step decoding.
    ///
    /// `token_ids`: shape (batch,)
    /// Returns: logits, shape (batch, vocab_size)
    pub fn forward_step(
        &self,
        token_ids: &[usize],
        batch: usize,
        inference_params: &mut InferenceParams,
    ) -> Vec<f32> {
        let d = self.config.d_model;
        let v = self.backbone.vocab_size;

        let hidden = self.backbone.forward_step(token_ids, batch, inference_params);

        let mut logits = vec![0.0f32; batch * v];
        for b in 0..batch {
            for j in 0..v {
                let mut acc = 0.0f32;
                for k in 0..d {
                    acc += hidden[b * d + k] * self.lm_head_weight[j * d + k];
                }
                logits[b * v + j] = acc;
            }
        }

        logits
    }

    /// Autoregressive text generation.
    ///
    /// `prompt`: initial token IDs (single batch)
    /// `max_new_tokens`: how many tokens to generate
    /// `sampler`: sampling configuration
    ///
    /// Returns: full sequence including prompt
    pub fn generate(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        sampler: &SamplerConfig,
    ) -> Vec<usize> {
        assert!(!prompt.is_empty(), "generate: prompt must not be empty");
        let batch = 1;
        let v = self.backbone.vocab_size;
        let mut tokens = prompt.to_vec();

        if max_new_tokens == 0 {
            return tokens;
        }

        let mut inference_params = InferenceParams::new();

        // Prefill: process entire prompt
        let logits = self.forward(&tokens, batch, tokens.len());
        let last_logits = &logits[(tokens.len() - 1) * v..tokens.len() * v];
        let next = sample_token(last_logits, &tokens, sampler);
        tokens.push(next);

        if max_new_tokens == 1 {
            return tokens;
        }

        // Advance offset so decode path is used
        inference_params.advance(prompt.len());

        // We need to "warm up" the cache by running prefill through step-by-step
        // Actually, for simplicity in this reference impl, we just use forward_step
        // for all subsequent tokens. The states won't match the prefill exactly
        // (since prefill used forward_train), so we re-process the prompt through
        // the step interface.
        inference_params.reset();
        for &tid in prompt.iter() {
            let _ = self.forward_step(&[tid], batch, &mut inference_params);
            inference_params.advance(1);
        }
        // Process the first generated token
        let logits_step = self.forward_step(&[next], batch, &mut inference_params);
        inference_params.advance(1);

        let next2 = sample_token(&logits_step[..v], &tokens, sampler);
        tokens.push(next2);

        // Continue generating
        for _ in 2..max_new_tokens {
            let last = *tokens.last().unwrap();
            let logits = self.forward_step(&[last], batch, &mut inference_params);
            inference_params.advance(1);
            let next = sample_token(&logits[..v], &tokens, sampler);
            tokens.push(next);
        }

        tokens
    }

    /// Total parameter count.
    pub fn param_count(&self) -> usize {
        let v = self.backbone.vocab_size;
        let d = self.config.d_model;
        let embedding = v * d;
        let layer_params: usize = self.backbone.layers.iter().map(|l| {
            match l {
                crate::mixer::MixerLayer::Mamba1(m) => m.param_count(),
                crate::mixer::MixerLayer::Mamba2(m) => m.param_count(),
                crate::mixer::MixerLayer::Mamba3(m) => m.param_count(),
                crate::mixer::MixerLayer::S4(s) => s.param_count(),
            }
        }).sum();
        let norms = self.config.n_layer * d + d; // per-layer norms + final
        let lm_head = if self.tie_embeddings { 0 } else { v * d };
        embedding + layer_params + norms + lm_head
    }
}

/// Sample a token from logits with the given config.
fn sample_token(logits: &[f32], _context: &[usize], config: &SamplerConfig) -> usize {
    let v = logits.len();

    if config.top_k == 1 {
        // Greedy: argmax
        let mut best = 0;
        let mut best_val = f32::NEG_INFINITY;
        for i in 0..v {
            if logits[i] > best_val {
                best_val = logits[i];
                best = i;
            }
        }
        return best;
    }

    // Temperature scaling
    let temp = config.temperature.max(1e-6);
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();

    // Softmax
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = if sum > 0.0 {
        exps.iter().map(|&x| x / sum).collect()
    } else {
        // Fallback: uniform distribution if all exps underflowed to 0
        let uniform = 1.0 / exps.len() as f32;
        vec![uniform; exps.len()]
    };

    // Top-k filtering
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

    let k = if config.top_k > 0 { config.top_k.min(v) } else { v };
    let top_k_items = &indexed[..k];

    // Top-p (nucleus) filtering
    let mut cumsum = 0.0f32;
    let mut cutoff = k;
    for (i, &(_, p)) in top_k_items.iter().enumerate() {
        cumsum += p;
        if cumsum >= config.top_p {
            cutoff = i + 1;
            break;
        }
    }

    let selected = &top_k_items[..cutoff];
    let total: f32 = selected.iter().map(|&(_, p)| p).sum();

    // Sample from the filtered distribution
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen::<f32>() * total;
    let mut acc = 0.0f32;
    for &(idx, p) in selected {
        acc += p;
        if acc >= r {
            return idx;
        }
    }

    selected.last().map(|&(idx, _)| idx).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba_lm_forward() {
        let config = MambaConfig::tiny();
        let model = MambaLMHeadModel::new(config);
        let batch = 1;
        let seq_len = 4;
        let tokens = vec![0, 1, 2, 3];
        let logits = model.forward(&tokens, batch, seq_len);
        assert_eq!(logits.len(), batch * seq_len * 256);
    }

    #[test]
    fn test_mamba_lm_generate() {
        let config = MambaConfig::tiny();
        let model = MambaLMHeadModel::new(config);
        let prompt = vec![0, 1, 2];
        let output = model.generate(&prompt, 5, &SamplerConfig::greedy());
        assert_eq!(output.len(), 3 + 5); // prompt + generated
        for &t in &output {
            assert!(t < 256);
        }
    }

    #[test]
    fn test_mamba_lm_param_count() {
        let config = MambaConfig::tiny();
        let model = MambaLMHeadModel::new(config);
        let params = model.param_count();
        assert!(params > 10_000, "params = {}", params);
        println!("Tiny Mamba LM params: {}", params);
    }

    #[test]
    fn test_mamba_lm_step() {
        let config = MambaConfig::tiny();
        let model = MambaLMHeadModel::new(config);
        let batch = 1;
        let mut params = InferenceParams::new();
        let logits = model.forward_step(&[42], batch, &mut params);
        assert_eq!(logits.len(), 256);
    }

    #[test]
    fn test_sample_greedy() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let token = sample_token(&logits, &[], &SamplerConfig::greedy());
        assert_eq!(token, 3); // argmax
    }
}
