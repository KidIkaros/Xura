//! Mamba-3 Predictor — replaces Llama-3.2-1B in VL-JEPA.
//!
//! Maps (visual embeddings + text query) → predicted target embedding
//! using a Mamba-3 MixerModel backbone with trapezoidal discretization,
//! complex-valued state dynamics, and MIMO multi-head structure.

use rand::Rng;

use xura_mamba::MixerModel;

use crate::angn::{ANGNConfig, AdaptiveNeuralGate};
use crate::config::{Mamba3PredictorConfig, RecursionConfig};
use crate::recursion::RecursionLayer;

/// Linear projection layer (weight + bias).
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

    /// Forward: (n, in_dim) → (n, out_dim)
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

/// Mamba-3 Predictor.
///
/// Takes visual embeddings from the X-Encoder and a text query,
/// projects them into a shared space, processes through Mamba-3 layers,
/// and outputs a predicted target embedding.
pub struct Mamba3Predictor {
    pub config: Mamba3PredictorConfig,
    /// Projects ViT visual tokens to predictor d_model.
    vision_proj: Linear,
    /// Projects query token embeddings to predictor d_model.
    query_proj: Linear,
    /// Mamba-3 backbone (uses MixerModel internally but we bypass embedding).
    backbone_layers: MixerModel,
    /// Prediction head: d_model → embed_dim.
    pred_head: Linear,
    /// Optional recursion layer for State-Space Delegation.
    pub recursion: Option<RecursionLayer>,
    /// Optional Adaptive Neural Gating Network for input filtering.
    pub angn: Option<AdaptiveNeuralGate>,
    /// Projects tool feedback knowledge → d_model for virtual token injection.
    /// Created when recursion is enabled, so tool results can be fed back
    /// through Mamba's SSM scan on the next step (fixes the "Memento" bug).
    feedback_proj: Option<Linear>,
}

impl Mamba3Predictor {
    pub fn new(config: Mamba3PredictorConfig) -> Self {
        let vision_proj = Linear::new(config.vision_dim, config.d_model);
        let query_proj = Linear::new(config.query_embed_dim, config.d_model);

        // Build MixerModel — we'll use a dummy vocab since we bypass embedding
        let mamba_config = config.to_mamba_config(2); // minimal vocab, unused
        let backbone_layers = MixerModel::new(mamba_config);

        let pred_head = Linear::new(config.d_model, config.embed_dim);

        Self {
            config,
            vision_proj,
            query_proj,
            backbone_layers,
            pred_head,
            recursion: None,
            angn: None,
            feedback_proj: None,
        }
    }

    /// Build a predictor with recursion layer enabled.
    ///
    /// Validates `inject_after_layer` against `n_layers` — if out of bounds,
    /// clamps to the last valid layer index and emits a debug warning.
    pub fn new_with_recursion(
        config: Mamba3PredictorConfig,
        mut rec_config: RecursionConfig,
    ) -> Self {
        let mut predictor = Self::new(config.clone());
        if rec_config.enabled {
            let n_layers = predictor.backbone_layers.layers.len();
            if n_layers > 0 && rec_config.inject_after_layer >= n_layers {
                let clamped = n_layers - 1;
                debug_assert!(
                    false,
                    "inject_after_layer ({}) >= n_layers ({}), clamping to {}",
                    rec_config.inject_after_layer, n_layers, clamped,
                );
                rec_config.inject_after_layer = clamped;
            }
            predictor.recursion = Some(RecursionLayer::new(rec_config.clone(), config.d_model));
            predictor.feedback_proj = Some(Linear::new(rec_config.knowledge_dim, config.d_model));
        }
        predictor
    }

    /// Build a predictor with both recursion and ANGN.
    ///
    /// This is the full-featured constructor. Both subsystems are optional
    /// (controlled by their `enabled` flags).
    pub fn new_with_features(
        config: Mamba3PredictorConfig,
        mut rec_config: RecursionConfig,
        angn_config: ANGNConfig,
    ) -> Self {
        let mut predictor = Self::new(config.clone());
        let n_layers = predictor.backbone_layers.layers.len();

        // Wire recursion
        if rec_config.enabled {
            if n_layers > 0 && rec_config.inject_after_layer >= n_layers {
                let clamped = n_layers - 1;
                debug_assert!(
                    false,
                    "inject_after_layer ({}) >= n_layers ({}), clamping to {}",
                    rec_config.inject_after_layer, n_layers, clamped,
                );
                rec_config.inject_after_layer = clamped;
            }
            predictor.recursion = Some(RecursionLayer::new(rec_config.clone(), config.d_model));
            predictor.feedback_proj = Some(Linear::new(rec_config.knowledge_dim, config.d_model));
        }

        // Wire ANGN
        if angn_config.enabled {
            predictor.angn = Some(AdaptiveNeuralGate::new(angn_config, n_layers));
        }

        predictor
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// - `visual_tokens`: shape (batch, n_vis, vision_dim) — from X-Encoder
    /// - `query_embeds`: shape (batch, n_qry, query_embed_dim) — embedded query tokens
    /// - `batch`: batch size
    /// - `n_vis`: number of visual tokens per sample
    /// - `n_qry`: number of query tokens per sample
    ///
    /// # Returns
    /// Predicted embedding: shape (batch, embed_dim), L2-normalized.
    pub fn forward(
        &mut self,
        visual_tokens: &[f32],
        query_embeds: &[f32],
        batch: usize,
        n_vis: usize,
        n_qry: usize,
    ) -> Vec<f32> {
        let dm = self.config.d_model;
        let seq_len = n_vis + n_qry;

        // 1) Project visual tokens: (batch * n_vis, vision_dim) → (batch * n_vis, d_model)
        let vis_proj = self.vision_proj.forward(visual_tokens, batch * n_vis);

        // 2) Project query embeddings: (batch * n_qry, query_embed_dim) → (batch * n_qry, d_model)
        let qry_proj = self.query_proj.forward(query_embeds, batch * n_qry);

        // 3) Concatenate: (batch, n_vis + n_qry, d_model)
        let mut concat = vec![0.0f32; batch * seq_len * dm];
        for b in 0..batch {
            // Visual tokens
            let vis_src = b * n_vis * dm;
            let dst = b * seq_len * dm;
            concat[dst..dst + n_vis * dm].copy_from_slice(&vis_proj[vis_src..vis_src + n_vis * dm]);
            // Query tokens
            let qry_src = b * n_qry * dm;
            let dst_qry = dst + n_vis * dm;
            concat[dst_qry..dst_qry + n_qry * dm]
                .copy_from_slice(&qry_proj[qry_src..qry_src + n_qry * dm]);
        }

        // 4) Feed through Mamba-3 backbone layers (bypass embedding, use raw hidden states)
        let hidden = self.forward_backbone(&concat, batch, seq_len);

        // 5) Average pool over all positions → (batch, d_model)
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

        // 6) Prediction head → L2-normalize
        let projected = self.pred_head.forward(&pooled, batch);
        l2_normalize(&projected, batch, self.config.embed_dim)
    }

    /// Text-only forward pass — skips visual token projection entirely.
    ///
    /// # Arguments
    /// - `query_embeds`: shape (batch * n_qry, query_embed_dim) — embedded query tokens
    /// - `batch`: batch size
    /// - `n_qry`: number of query tokens per sample
    ///
    /// # Returns
    /// Predicted embedding: shape (batch, embed_dim), L2-normalized.
    pub fn forward_text_only(
        &mut self,
        query_embeds: &[f32],
        batch: usize,
        n_qry: usize,
    ) -> Vec<f32> {
        let dm = self.config.d_model;

        // Project query embeddings only: (batch * n_qry, query_embed_dim) → (batch * n_qry, d_model)
        let qry_proj = self.query_proj.forward(query_embeds, batch * n_qry);

        // Feed through Mamba-3 backbone (query tokens only, no visual tokens)
        let hidden = self.forward_backbone(&qry_proj, batch, n_qry);

        // Average pool → (batch, d_model)
        let mut pooled = vec![0.0f32; batch * dm];
        for b in 0..batch {
            for pos in 0..n_qry {
                for d in 0..dm {
                    pooled[b * dm + d] += hidden[b * n_qry * dm + pos * dm + d];
                }
            }
            let inv_len = 1.0 / n_qry as f32;
            for d in 0..dm {
                pooled[b * dm + d] *= inv_len;
            }
        }

        // Prediction head → L2-normalize
        let projected = self.pred_head.forward(&pooled, batch);
        l2_normalize(&projected, batch, self.config.embed_dim)
    }

    /// Whether the predictor has a feedback projection (for virtual token feedback).
    pub fn has_feedback_proj(&self) -> bool {
        self.feedback_proj.is_some()
    }

    /// Forward pass with optional virtual token feedback from prior tool use.
    ///
    /// When `feedback` is provided, it is projected via `feedback_proj` and
    /// prepended as a virtual token to the sequence. Mamba's SSM scan then
    /// "reads" this token into hidden state, giving the model persistent
    /// memory of tool results (fixes the "Memento" problem).
    ///
    /// # Arguments
    /// - `visual_tokens`: shape (batch * n_vis, vision_dim) from X-Encoder
    /// - `query_embeds`: shape (batch * n_qry, query_embed_dim)
    /// - `feedback`: optional knowledge vector from previous tool call, shape (feedback_dim,)
    /// - `batch`: batch size
    /// - `n_vis`: visual tokens per sample
    /// - `n_qry`: query tokens per sample
    pub fn forward_with_feedback(
        &mut self,
        visual_tokens: &[f32],
        query_embeds: &[f32],
        feedback: Option<&[f32]>,
        batch: usize,
        n_vis: usize,
        n_qry: usize,
    ) -> Vec<f32> {
        let dm = self.config.d_model;

        // Project inputs to d_model
        let vis_proj = self.vision_proj.forward(visual_tokens, batch * n_vis);
        let qry_proj = self.query_proj.forward(query_embeds, batch * n_qry);

        // Project feedback to a virtual token if available
        let (fb_token, n_fb) = match (feedback, &self.feedback_proj) {
            (Some(fb), Some(proj)) => (Some(proj.forward(fb, 1)), 1usize),
            _ => (None, 0usize),
        };

        let seq_len = n_fb + n_vis + n_qry;

        // Concatenate: [feedback_token; visual_tokens; query_tokens]
        let mut concat = vec![0.0f32; batch * seq_len * dm];
        for b in 0..batch {
            let dst = b * seq_len * dm;
            let mut offset = 0;

            // Virtual feedback token (broadcast to all batch items)
            if let Some(ref fb) = fb_token {
                concat[dst..dst + dm].copy_from_slice(&fb[..dm]);
                offset += dm;
            }

            // Visual tokens
            let vis_src = b * n_vis * dm;
            concat[dst + offset..dst + offset + n_vis * dm]
                .copy_from_slice(&vis_proj[vis_src..vis_src + n_vis * dm]);
            offset += n_vis * dm;

            // Query tokens
            let qry_src = b * n_qry * dm;
            concat[dst + offset..dst + offset + n_qry * dm]
                .copy_from_slice(&qry_proj[qry_src..qry_src + n_qry * dm]);
        }

        // Feed through backbone
        let hidden = self.forward_backbone(&concat, batch, seq_len);

        // Average pool → (batch, d_model)
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

        // Prediction head → L2-normalize
        let projected = self.pred_head.forward(&pooled, batch);
        l2_normalize(&projected, batch, self.config.embed_dim)
    }

    /// Text-only forward pass with optional virtual token feedback.
    ///
    /// Same as `forward_text_only` but prepends a feedback virtual token
    /// when available, so Mamba reads tool results into state.
    pub fn forward_text_only_with_feedback(
        &mut self,
        query_embeds: &[f32],
        feedback: Option<&[f32]>,
        batch: usize,
        n_qry: usize,
    ) -> Vec<f32> {
        let dm = self.config.d_model;

        let qry_proj = self.query_proj.forward(query_embeds, batch * n_qry);

        let (fb_token, n_fb) = match (feedback, &self.feedback_proj) {
            (Some(fb), Some(proj)) => (Some(proj.forward(fb, 1)), 1usize),
            _ => (None, 0usize),
        };

        let seq_len = n_fb + n_qry;

        let mut concat = vec![0.0f32; batch * seq_len * dm];
        for b in 0..batch {
            let dst = b * seq_len * dm;
            let mut offset = 0;

            if let Some(ref fb) = fb_token {
                concat[dst..dst + dm].copy_from_slice(&fb[..dm]);
                offset += dm;
            }

            let qry_src = b * n_qry * dm;
            concat[dst + offset..dst + offset + n_qry * dm]
                .copy_from_slice(&qry_proj[qry_src..qry_src + n_qry * dm]);
        }

        let hidden = self.forward_backbone(&concat, batch, seq_len);

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

        let projected = self.pred_head.forward(&pooled, batch);
        l2_normalize(&projected, batch, self.config.embed_dim)
    }

    /// Run the Mamba-3 backbone layers directly on hidden states (bypassing embedding).
    ///
    /// Pipeline per layer:
    /// 1. ANGN gate (multiplicative filter — removes noise before SSM update)
    /// 2. Pre-norm (RMSNorm)
    /// 3. Mamba-3 mixer (SSM state update)
    /// 4. Residual connection
    /// 5. Recursion injection (if configured for this layer)
    fn forward_backbone(&mut self, hidden: &[f32], batch: usize, seq_len: usize) -> Vec<f32> {
        let n = batch * seq_len;
        let dm = self.config.d_model;
        let n_layers = self.backbone_layers.layers.len();
        let mut h = hidden.to_vec();

        for i in 0..n_layers {
            // 1) ANGN gate: multiplicative filtering before SSM update
            if let Some(ref mut angn) = self.angn {
                h = angn.gate_layer(&h, i, n);
            }

            // 2) Pre-norm
            let normed = self.backbone_layers.norms[i].forward(&h, n);
            // 3) Mixer
            let mixed = self.backbone_layers.layers[i].forward_train(&normed, batch, seq_len);
            // 4) Residual
            for j in 0..h.len() {
                h[j] += mixed[j];
            }

            // 5) Recursion injection point: after the configured layer
            if let Some(ref mut recursion) = self.recursion {
                if i == recursion.config.inject_after_layer {
                    let (h_new, _delegated, _score) =
                        recursion.maybe_delegate(&h, batch, seq_len, dm);
                    h = h_new;
                }
            }
        }

        // Final norm
        self.backbone_layers.final_norm.forward(&h, n)
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
    fn test_predictor_forward_shape() {
        let config = Mamba3PredictorConfig::tiny();
        let mut predictor = Mamba3Predictor::new(config.clone());

        let batch = 2;
        let n_vis = 4;
        let n_qry = 3;
        let vis = vec![0.1f32; batch * n_vis * config.vision_dim];
        let qry = vec![0.1f32; batch * n_qry * config.query_embed_dim];

        let output = predictor.forward(&vis, &qry, batch, n_vis, n_qry);
        assert_eq!(output.len(), batch * config.embed_dim);
    }

    #[test]
    fn test_predictor_output_normalized() {
        let config = Mamba3PredictorConfig::tiny();
        let mut predictor = Mamba3Predictor::new(config.clone());

        let batch = 1;
        let n_vis = 4;
        let n_qry = 2;
        let vis = vec![0.5f32; batch * n_vis * config.vision_dim];
        let qry = vec![0.3f32; batch * n_qry * config.query_embed_dim];

        let output = predictor.forward(&vis, &qry, batch, n_vis, n_qry);

        // Check L2 norm ≈ 1
        let norm: f32 = output.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "output should be L2-normalized, got norm={}",
            norm
        );
    }

    #[test]
    fn test_linear_forward() {
        let lin = Linear::new(4, 2);
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let output = lin.forward(&input, 1);
        assert_eq!(output.len(), 2);
    }
}
