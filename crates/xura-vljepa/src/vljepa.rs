//! Mamba3-JEPA assembly — the full State-Space World Model.
//!
//! Combines VisionEncoder (ViT), Mamba3Predictor, Mamba3TextEncoder, and Mamba3Decoder
//! into a unified architecture for vision-language understanding.

use crate::config::Mamba3JepaConfig;
use crate::loss::info_nce_loss;
use crate::predictor::Mamba3Predictor;
use crate::vit::VisionEncoder;
use crate::y_decoder::Mamba3Decoder;
use crate::y_encoder::Mamba3TextEncoder;
use crate::recursion::MemoryTool;

/// Training output from a forward pass.
pub struct TrainOutput {
    /// InfoNCE loss value.
    pub loss: f32,
    /// Predicted embeddings: (batch, embed_dim).
    pub predicted: Vec<f32>,
    /// Target embeddings: (batch, embed_dim).
    pub target: Vec<f32>,
}

/// Inference mode.
pub enum InferenceMode {
    /// Return predicted embedding only (for classification / retrieval).
    Embedding,
    /// Decode predicted embedding to text (for VQA).
    Decode { max_tokens: usize, bos_token: usize },
}

/// Inference output.
pub enum InferenceOutput {
    /// Predicted embedding: (embed_dim,).
    Embedding(Vec<f32>),
    /// Decoded text tokens.
    Tokens(Vec<usize>),
}

/// Mamba3-JEPA: State-Space World Model.
pub struct Mamba3Jepa {
    pub config: Mamba3JepaConfig,
    pub x_encoder: VisionEncoder,
    pub predictor: Mamba3Predictor,
    pub y_encoder: Mamba3TextEncoder,
    pub y_decoder: Mamba3Decoder,
}

impl Mamba3Jepa {
    /// Build a new Mamba3-JEPA model from config with random weights.
    pub fn new(config: Mamba3JepaConfig) -> Self {
        let x_encoder = VisionEncoder::new(config.vit.clone());
        let predictor = Mamba3Predictor::new_with_features(
            config.predictor.clone(),
            config.recursion.clone(),
            config.angn.clone(),
        );
        let y_encoder = Mamba3TextEncoder::new(config.y_encoder.clone());
        let y_decoder = Mamba3Decoder::new(config.y_decoder.clone());

        Self { config, x_encoder, predictor, y_encoder, y_decoder }
    }

    /// Replace the memory tool used by the recursion layer.
    pub fn set_memory_tool(&mut self, tool: Box<dyn MemoryTool>) {
        if let Some(ref mut rec) = self.predictor.recursion {
            rec.set_memory_tool(tool);
        }
    }

    /// Training forward pass.
    ///
    /// # Arguments
    /// - `images`: visual input, shape (batch, in_channels, H, W) flattened
    /// - `query_tokens`: query text token IDs, shape (batch * n_qry,) flattened
    /// - `target_tokens`: target text token IDs, shape (batch * n_tgt,) flattened
    /// - `batch`: batch size
    /// - `h`, `w`: image height and width
    /// - `n_qry`: query sequence length
    /// - `n_tgt`: target sequence length
    /// - `temperature`: InfoNCE temperature
    ///
    /// # Returns
    /// `TrainOutput` with loss, predicted embeddings, and target embeddings.
    pub fn train_forward(
        &mut self,
        images: &[f32],
        query_tokens: &[usize],
        target_tokens: &[usize],
        batch: usize,
        h: usize,
        w: usize,
        n_qry: usize,
        n_tgt: usize,
        temperature: f32,
    ) -> TrainOutput {
        let num_patches = (h / self.config.vit.patch_size) * (w / self.config.vit.patch_size);

        // 1) X-Encoder: images → visual tokens (frozen)
        let visual_tokens = self.x_encoder.forward_patches(images, batch, h, w);

        // 2) Embed query tokens for predictor input
        // Use the Y-Encoder's backbone embedding table to get query embeddings
        let qry_dm = self.config.y_encoder.d_model;
        let query_embeds = embed_tokens(
            query_tokens,
            &self.y_encoder.backbone.embedding,
            qry_dm,
            batch * n_qry,
        );

        // 3) Predictor: (visual_tokens, query_embeds) → predicted embedding
        let predicted = self.predictor.forward(
            &visual_tokens,
            &query_embeds,
            batch,
            num_patches,
            n_qry,
        );

        // 4) Y-Encoder: target_tokens → target embedding
        let target = self.y_encoder.forward(target_tokens, batch, n_tgt);

        // 5) InfoNCE loss
        let embed_dim = self.config.shared_embed_dim;
        let loss = info_nce_loss(&predicted, &target, batch, embed_dim, temperature);

        TrainOutput { loss, predicted, target }
    }

    /// Inference forward pass (single sample).
    ///
    /// # Arguments
    /// - `image`: visual input, shape (1, in_channels, H, W) flattened
    /// - `query_tokens`: query token IDs, shape (n_qry,)
    /// - `h`, `w`: image dimensions
    /// - `mode`: inference mode (embedding only or decode to text)
    pub fn infer(
        &mut self,
        image: &[f32],
        query_tokens: &[usize],
        h: usize,
        w: usize,
        mode: InferenceMode,
    ) -> InferenceOutput {
        let batch = 1;
        let num_patches = (h / self.config.vit.patch_size) * (w / self.config.vit.patch_size);
        let n_qry = query_tokens.len();

        // 1) X-Encoder
        let visual_tokens = self.x_encoder.forward_patches(image, batch, h, w);

        // 2) Embed query
        let qry_dm = self.config.y_encoder.d_model;
        let query_embeds = embed_tokens(
            query_tokens,
            &self.y_encoder.backbone.embedding,
            qry_dm,
            n_qry,
        );

        // 3) Predictor → predicted embedding
        let predicted = self.predictor.forward(
            &visual_tokens,
            &query_embeds,
            batch,
            num_patches,
            n_qry,
        );

        match mode {
            InferenceMode::Embedding => InferenceOutput::Embedding(predicted),
            InferenceMode::Decode { max_tokens, bos_token } => {
                let tokens = self.y_decoder.generate(&predicted, max_tokens, bos_token);
                InferenceOutput::Tokens(tokens)
            }
        }
    }

    /// Text-only inference — skips the ViT X-Encoder entirely.
    ///
    /// This is a fast-path for when the agent has no visual input.
    /// Query tokens are embedded and fed directly to the Mamba-3 Predictor
    /// without any visual token concatenation.
    ///
    /// # Arguments
    /// - `query_tokens`: query token IDs, shape (n_qry,)
    /// - `mode`: inference mode (embedding only or decode to text)
    pub fn infer_text_only(
        &mut self,
        query_tokens: &[usize],
        mode: InferenceMode,
    ) -> InferenceOutput {
        let batch = 1;
        let n_qry = query_tokens.len();

        // Embed query tokens (no ViT call)
        let qry_dm = self.config.y_encoder.d_model;
        let query_embeds = embed_tokens(
            query_tokens,
            &self.y_encoder.backbone.embedding,
            qry_dm,
            n_qry,
        );

        // Predictor text-only path (no visual tokens)
        let predicted = self.predictor.forward_text_only(
            &query_embeds,
            batch,
            n_qry,
        );

        match mode {
            InferenceMode::Embedding => InferenceOutput::Embedding(predicted),
            InferenceMode::Decode { max_tokens, bos_token } => {
                let tokens = self.y_decoder.generate(&predicted, max_tokens, bos_token);
                InferenceOutput::Tokens(tokens)
            }
        }
    }

    /// Classify by comparing predicted embedding against candidate embeddings.
    ///
    /// # Arguments
    /// - `predicted`: shape (embed_dim,)
    /// - `candidates`: shape (n_candidates, embed_dim) flattened
    /// - `n_candidates`: number of candidates
    ///
    /// # Returns
    /// Index of the most similar candidate.
    pub fn classify(
        &self,
        predicted: &[f32],
        candidates: &[f32],
        n_candidates: usize,
    ) -> usize {
        let dim = self.config.shared_embed_dim;
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;

        for c in 0..n_candidates {
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += predicted[d] * candidates[c * dim + d];
            }
            if dot > best_sim {
                best_sim = dot;
                best_idx = c;
            }
        }

        best_idx
    }
}

/// Look up token embeddings from an embedding table.
/// Returns flat f32 of shape (n, d_model).
///
/// Token IDs that are out of range get zero embeddings (no panic).
fn embed_tokens(
    token_ids: &[usize],
    embedding_table: &[f32],
    d_model: usize,
    n: usize,
) -> Vec<f32> {
    assert!(n <= token_ids.len(),
        "embed_tokens: n ({}) exceeds token_ids length ({})", n, token_ids.len());
    let vocab_size = embedding_table.len() / d_model;
    let mut out = vec![0.0f32; n * d_model];
    for i in 0..n {
        let tid = token_ids[i];
        if tid < vocab_size {
            let src = tid * d_model;
            let dst = i * d_model;
            out[dst..dst + d_model].copy_from_slice(&embedding_table[src..src + d_model]);
        } else {
            debug_assert!(false, "embed_tokens: OOV token id {} >= vocab_size {}", tid, vocab_size);
            // Out-of-vocab IDs get zero vectors in release builds
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Mamba3JepaConfig;

    #[test]
    fn test_mamba3_jepa_creation() {
        let config = Mamba3JepaConfig::tiny();
        let model = Mamba3Jepa::new(config);
        assert_eq!(model.config.shared_embed_dim, 32);
    }

    #[test]
    fn test_mamba3_jepa_train_forward() {
        let config = Mamba3JepaConfig::tiny();
        let mut model = Mamba3Jepa::new(config.clone());

        let batch = 2;
        let h = config.vit.image_size;
        let w = config.vit.image_size;
        let n_qry = 3;
        let n_tgt = 4;

        let images = vec![0.1f32; batch * config.vit.in_channels * h * w];
        let query_tokens: Vec<usize> = (0..batch * n_qry).map(|i| i % config.y_encoder.vocab_size).collect();
        let target_tokens: Vec<usize> = (0..batch * n_tgt).map(|i| i % config.y_encoder.vocab_size).collect();

        let output = model.train_forward(
            &images, &query_tokens, &target_tokens,
            batch, h, w, n_qry, n_tgt, 0.07,
        );

        assert_eq!(output.predicted.len(), batch * config.shared_embed_dim);
        assert_eq!(output.target.len(), batch * config.shared_embed_dim);
        assert!(output.loss.is_finite(), "loss should be finite, got {}", output.loss);
    }

    #[test]
    fn test_mamba3_jepa_infer_embedding() {
        let config = Mamba3JepaConfig::tiny();
        let mut model = Mamba3Jepa::new(config.clone());

        let h = config.vit.image_size;
        let w = config.vit.image_size;
        let image = vec![0.1f32; config.vit.in_channels * h * w];
        let query = vec![1usize, 2, 3];

        let output = model.infer(&image, &query, h, w, InferenceMode::Embedding);
        match output {
            InferenceOutput::Embedding(emb) => {
                assert_eq!(emb.len(), config.shared_embed_dim);
            }
            _ => panic!("expected Embedding output"),
        }
    }

    #[test]
    fn test_mamba3_jepa_infer_decode() {
        let config = Mamba3JepaConfig::tiny();
        let mut model = Mamba3Jepa::new(config.clone());

        let h = config.vit.image_size;
        let w = config.vit.image_size;
        let image = vec![0.1f32; config.vit.in_channels * h * w];
        let query = vec![1usize, 2];

        let output = model.infer(
            &image, &query, h, w,
            InferenceMode::Decode { max_tokens: 5, bos_token: 0 },
        );
        match output {
            InferenceOutput::Tokens(tokens) => {
                assert_eq!(tokens.len(), 6); // BOS + 5 generated
            }
            _ => panic!("expected Tokens output"),
        }
    }

    #[test]
    fn test_classify() {
        let config = Mamba3JepaConfig::tiny();
        let model = Mamba3Jepa::new(config.clone());
        let dim = config.shared_embed_dim;

        // Predicted embedding
        let mut predicted = vec![0.0f32; dim];
        predicted[0] = 1.0;

        // 3 candidates: second one matches
        let mut candidates = vec![0.0f32; 3 * dim];
        candidates[0 * dim + 1] = 1.0; // candidate 0: orthogonal
        candidates[1 * dim + 0] = 1.0; // candidate 1: matches
        candidates[2 * dim + 2] = 1.0; // candidate 2: orthogonal

        let idx = model.classify(&predicted, &candidates, 3);
        assert_eq!(idx, 1);
    }
}
