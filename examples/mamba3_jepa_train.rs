//! Mamba3-JEPA Training Example
//!
//! Demonstrates the full training pipeline for the Mamba3-JEPA world model:
//!
//! 1. Build model from config (tiny preset for demo)
//! 2. Generate synthetic image-text training data
//! 3. Run the two-stage training loop:
//!    - Stage 1: Query-free captioning pretrain (query = target)
//!    - Stage 2: Query-conditioned SFT (query ≠ target)
//! 4. Evaluate: embedding similarity, classification, text decoding
//!
//! Run with:
//!   cargo run --example mamba3_jepa_train --release
//!
//! Architecture (from arxiv.org/abs/2512.10942, adapted with Mamba-3):
//!
//!   ┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
//!   │  X-Encoder   │     │   Predictor       │     │  Y-Encoder   │
//!   │  (ViT,frozen)│────▶│  (Mamba-3 SSM)    │     │  (Mamba-3)   │
//!   │  image→Sv    │     │  ⟨Sv,Xq⟩ → Ŝy    │     │  text → Sy   │
//!   └─────────────┘     └────────┬───────────┘     └──────┬──────┘
//!                                │                        │
//!                                ▼                        ▼
//!                        ┌──────────────────────────────────┐
//!                        │   InfoNCE Loss: align Ŝy ↔ Sy   │
//!                        └──────────────────────────────────┘
//!                                │
//!                                ▼ (inference only)
//!                        ┌──────────────────┐
//!                        │   Y-Decoder       │
//!                        │  (Mamba-3 LM)     │
//!                        │   Ŝy → text       │
//!                        └──────────────────┘

use std::time::Instant;

// ── Mamba3-JEPA types ──────────────────────────────────────────────────────
use xura_vljepa::{
    Mamba3Jepa, Mamba3JepaConfig, TrainOutput,
    InferenceMode, InferenceOutput,
    SelectiveDecoder, SelectiveDecodeConfig,
    RecursionConfig, LocalMemoryTool,
};

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Simple pseudo-random number generator (xorshift64) — no external deps.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform f32 in [lo, hi).
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let t = (self.next_u64() & 0xFFFFFF) as f32 / 0xFFFFFF as f32;
        lo + t * (hi - lo)
    }

    /// Random token ID in [0, vocab_size).
    fn token(&mut self, vocab_size: usize) -> usize {
        (self.next_u64() % vocab_size as u64) as usize
    }
}

/// Generate a synthetic training batch.
///
/// Returns (images, query_tokens, target_tokens).
///
/// In a real pipeline you'd load (video_frames, query_text, target_text) triplets
/// from a dataset like CC3M, WebVid, or your own captioned video corpus.
fn generate_batch(
    rng: &mut Rng,
    batch: usize,
    in_channels: usize,
    h: usize,
    w: usize,
    n_qry: usize,
    n_tgt: usize,
    vocab_size: usize,
) -> (Vec<f32>, Vec<usize>, Vec<usize>) {
    let img_size = batch * in_channels * h * w;
    let images: Vec<f32> = (0..img_size).map(|_| rng.uniform(-1.0, 1.0)).collect();
    let query_tokens: Vec<usize> = (0..batch * n_qry).map(|_| rng.token(vocab_size)).collect();
    let target_tokens: Vec<usize> = (0..batch * n_tgt).map(|_| rng.token(vocab_size)).collect();
    (images, query_tokens, target_tokens)
}

/// Naive SGD parameter update (for demonstration).
///
/// In production, use AdamW from kore-optim with:
///   - Predictor LR: 1e-4
///   - Y-Encoder LR: 5e-6 (×0.05 multiplier, per VL-JEPA paper §3.2)
///   - X-Encoder: frozen (no update)
///   - Cosine LR schedule with linear warmup
fn sgd_step_placeholder(step: usize, loss: f32) {
    // This is a placeholder. Real gradient computation requires autograd.
    // The kore-autograd crate provides this; here we just log progress.
    let _ = (step, loss);
}

/// Cosine learning rate schedule with linear warmup.
fn cosine_lr(step: usize, warmup_steps: usize, total_steps: usize, base_lr: f32) -> f32 {
    if step < warmup_steps {
        base_lr * (step as f32 / warmup_steps as f32)
    } else {
        let progress = (step - warmup_steps) as f32 / (total_steps - warmup_steps) as f32;
        base_lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          Mamba3-JEPA Training Pipeline (Kore)               ║");
    println!("║  State-Space World Model for Vision-Language                ║");
    println!("║  Paper: arxiv.org/abs/2512.10942 (adapted with Mamba-3)    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── 1. Configuration ───────────────────────────────────────────────────
    //
    // Using the tiny preset for fast iteration. For real training, use:
    //   Mamba3JepaConfig::small()   (~150M trainable params)
    //
    // The VL-JEPA paper trains with:
    //   - X-Encoder: V-JEPA 2 ViT-L (304M, frozen)
    //   - Predictor: 8 layers, 1B params (we use Mamba-3 instead of Llama)
    //   - Y-Encoder: 300M params (we use Mamba-3 instead of EmbeddingGemma)
    //   - Y-Decoder: lightweight LM (we use Mamba-3 LM)

    let config = Mamba3JepaConfig::tiny();
    println!("── Model Configuration ──");
    println!("  ViT:       d_model={}, layers={}, patch={}×{}, image={}×{}",
        config.vit.d_model, config.vit.n_layers,
        config.vit.patch_size, config.vit.patch_size,
        config.vit.image_size, config.vit.image_size);
    println!("  Predictor: d_model={}, layers={}, d_state={}, headdim={}",
        config.predictor.d_model, config.predictor.n_layers,
        config.predictor.d_state, config.predictor.headdim);
    println!("  Y-Encoder: d_model={}, layers={}, vocab={}",
        config.y_encoder.d_model, config.y_encoder.n_layers,
        config.y_encoder.vocab_size);
    println!("  Y-Decoder: d_model={}, layers={}, prefix_len={}",
        config.y_decoder.d_model, config.y_decoder.n_layers,
        config.y_decoder.prefix_len);
    println!("  Shared embed dim: {}", config.shared_embed_dim);
    println!();

    // ── 2. Build Model ─────────────────────────────────────────────────────

    let t0 = Instant::now();
    let mut model = Mamba3Jepa::new(config.clone());
    println!("Model built in {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    println!();

    // ── 3. Training Hyperparameters ────────────────────────────────────────
    //
    // VL-JEPA paper §3.2:
    //   - Stage 1 (query-free pretrain): 50K steps, batch=2048, LR=1e-4
    //   - Stage 2 (query-conditioned SFT): 30K steps, batch=512, LR=5e-5
    //   - Y-Encoder LR multiplier: ×0.05
    //   - Temperature τ = 0.07
    //   - AdamW, β=(0.9, 0.95), weight_decay=0.05
    //   - Cosine LR schedule with 2K warmup steps
    //
    // For this demo we use tiny numbers:

    let batch = 2;
    let h = config.vit.image_size;
    let w = config.vit.image_size;
    let vocab_size = config.y_encoder.vocab_size;
    let temperature = 0.07;

    let stage1_steps = 20;
    let stage2_steps = 15;
    let warmup_steps = 3;
    let base_lr = 1e-4;

    let mut rng = Rng::new(42);

    // ── 4. Stage 1: Query-Free Captioning Pretrain ─────────────────────────
    //
    // In Stage 1, the query IS the target (self-supervised).
    // The model learns to predict text embeddings from visual input alone.
    // This is analogous to V-JEPA's image-only pretraining phase.

    println!("═══ Stage 1: Query-Free Captioning Pretrain ═══");
    println!("  Steps: {}, Batch: {}, Temperature: {}", stage1_steps, batch, temperature);
    println!();

    let mut stage1_losses = Vec::new();
    let t_stage1 = Instant::now();

    for step in 0..stage1_steps {
        let n_tgt = 4 + (step % 3); // vary sequence length
        let n_qry = n_tgt;          // query = target in Stage 1

        let (images, _, target_tokens) = generate_batch(
            &mut rng, batch, config.vit.in_channels, h, w, n_qry, n_tgt, vocab_size,
        );
        // Stage 1: query tokens = target tokens (self-supervised)
        let query_tokens = target_tokens.clone();

        let lr = cosine_lr(step, warmup_steps, stage1_steps, base_lr);

        let output: TrainOutput = model.train_forward(
            &images, &query_tokens, &target_tokens,
            batch, h, w, n_qry, n_tgt, temperature,
        );

        stage1_losses.push(output.loss);
        sgd_step_placeholder(step, output.loss);

        if step % 5 == 0 || step == stage1_steps - 1 {
            println!("  step {:3}/{:3}  loss={:.4}  lr={:.2e}  pred_norm={:.3}  tgt_norm={:.3}",
                step, stage1_steps, output.loss, lr,
                vec_l2_norm(&output.predicted, config.shared_embed_dim),
                vec_l2_norm(&output.target, config.shared_embed_dim),
            );
        }
    }

    let stage1_time = t_stage1.elapsed();
    let avg_loss_s1: f32 = stage1_losses.iter().sum::<f32>() / stage1_losses.len() as f32;
    println!();
    println!("  Stage 1 complete: avg_loss={:.4}, time={:.1}ms ({:.1}ms/step)",
        avg_loss_s1,
        stage1_time.as_secs_f64() * 1000.0,
        stage1_time.as_secs_f64() * 1000.0 / stage1_steps as f64,
    );
    println!();

    // ── 5. Stage 2: Query-Conditioned SFT ──────────────────────────────────
    //
    // In Stage 2, the query and target are different.
    // The model learns to predict target text embeddings given a visual input
    // and a query prompt (e.g., "What is the person doing?").
    //
    // This is the key VL-JEPA innovation: predicting continuous embeddings
    // instead of autoregressive tokens.

    println!("═══ Stage 2: Query-Conditioned SFT ═══");
    println!("  Steps: {}, Batch: {}, Temperature: {}", stage2_steps, batch, temperature);
    println!();

    let mut stage2_losses = Vec::new();
    let t_stage2 = Instant::now();

    for step in 0..stage2_steps {
        let n_qry = 3 + (step % 2);
        let n_tgt = 5 + (step % 3);

        let (images, query_tokens, target_tokens) = generate_batch(
            &mut rng, batch, config.vit.in_channels, h, w, n_qry, n_tgt, vocab_size,
        );

        let global_step = stage1_steps + step;
        let lr = cosine_lr(global_step, warmup_steps, stage1_steps + stage2_steps, base_lr * 0.5);

        let output: TrainOutput = model.train_forward(
            &images, &query_tokens, &target_tokens,
            batch, h, w, n_qry, n_tgt, temperature,
        );

        stage2_losses.push(output.loss);
        sgd_step_placeholder(global_step, output.loss);

        if step % 5 == 0 || step == stage2_steps - 1 {
            println!("  step {:3}/{:3}  loss={:.4}  lr={:.2e}",
                step, stage2_steps, output.loss, lr);
        }
    }

    let stage2_time = t_stage2.elapsed();
    let avg_loss_s2: f32 = stage2_losses.iter().sum::<f32>() / stage2_losses.len() as f32;
    println!();
    println!("  Stage 2 complete: avg_loss={:.4}, time={:.1}ms ({:.1}ms/step)",
        avg_loss_s2,
        stage2_time.as_secs_f64() * 1000.0,
        stage2_time.as_secs_f64() * 1000.0 / stage2_steps as f64,
    );
    println!();

    // ── 6. Evaluation: Embedding Inference ─────────────────────────────────
    //
    // After training, the model can:
    //   (a) Produce embeddings for retrieval / classification
    //   (b) Decode embeddings to text for VQA
    //   (c) Selectively decode only when the SSM state drifts (streaming)

    println!("═══ Evaluation ═══");
    println!();

    // 6a. Embedding inference
    let image = vec![0.5f32; config.vit.in_channels * h * w];
    let query = vec![1usize, 2, 3];

    let t_infer = Instant::now();
    let output = model.infer(&image, &query, h, w, InferenceMode::Embedding);
    let infer_time = t_infer.elapsed();

    match &output {
        InferenceOutput::Embedding(emb) => {
            println!("  Embedding inference: dim={}, time={:.2}ms",
                emb.len(), infer_time.as_secs_f64() * 1000.0);
            println!("  First 8 dims: {:?}", &emb[..8.min(emb.len())]);
        }
        _ => {}
    }
    println!();

    // 6b. Zero-shot classification via nearest-neighbor
    //
    // VL-JEPA classifies by comparing the predicted embedding against
    // candidate text embeddings (no fine-tuning needed).
    println!("── Zero-Shot Classification ──");

    let dim = config.shared_embed_dim;
    let class_names = ["cat", "dog", "bird"];
    let n_candidates = class_names.len();

    // Encode each class name through Y-Encoder to get candidate embeddings
    let mut candidates = Vec::new();
    for (i, _name) in class_names.iter().enumerate() {
        // In practice: tokenize the class name, run through y_encoder.forward()
        // Here we use synthetic embeddings for demonstration
        let class_tokens: Vec<usize> = (0..3).map(|j| (i * 3 + j) % vocab_size).collect();
        let class_emb = model.y_encoder.forward(&class_tokens, 1, 3);
        candidates.extend_from_slice(&class_emb);
    }

    if let InferenceOutput::Embedding(ref emb) = output {
        let predicted_class = model.classify(emb, &candidates, n_candidates);
        println!("  Candidates: {:?}", class_names);
        println!("  Predicted class index: {} (\"{}\")", predicted_class, class_names[predicted_class]);
    }
    println!();

    // 6c. Text decoding (VQA mode)
    //
    // When text output is needed, the Y-Decoder generates tokens from the
    // predicted embedding using prefix conditioning + autoregressive Mamba-3.
    println!("── Text Decoding (VQA) ──");

    let t_decode = Instant::now();
    let decode_output = model.infer(
        &image, &query, h, w,
        InferenceMode::Decode { max_tokens: 10, bos_token: 0 },
    );
    let decode_time = t_decode.elapsed();

    match decode_output {
        InferenceOutput::Tokens(tokens) => {
            println!("  Generated {} tokens in {:.2}ms: {:?}",
                tokens.len(), decode_time.as_secs_f64() * 1000.0, tokens);
            println!("  (Token IDs — use a tokenizer to decode to text)");
        }
        _ => {}
    }
    println!();

    // ── 7. Selective Decoding Demo ─────────────────────────────────────────
    //
    // Mamba-3's hidden state h_t naturally tracks semantic change.
    // The SelectiveDecoder monitors ‖h_t - h_{t-k}‖ and only triggers
    // the expensive Y-Decoder when the state drifts past a threshold.
    //
    // This is ideal for streaming video: decode only at scene changes.

    println!("── Selective Decoding (Streaming) ──");

    let sel_config = SelectiveDecodeConfig {
        window_size: 4,
        drift_threshold: 0.3,
        smoothing: 0.2,
    };
    let mut selective = SelectiveDecoder::new(sel_config);

    println!("  Simulating 8 frames...");
    for frame in 0..8 {
        // Simulate SSM state: static for frames 0-3, then a "scene change" at frame 4
        let mut state = vec![0.1f32; dim];
        if frame >= 4 {
            // Scene change: large state shift
            for d in 0..dim {
                state[d] = 0.9 + (d as f32) * 0.01;
            }
        }

        let should_decode = selective.should_decode(&state);
        println!("    frame {}: drift={:.4}, decode={}",
            frame, selective.current_drift(), should_decode);
    }
    println!();

    // ── 8. Recursion Layer Demo (State-Space Delegation) ─────────────────
    //
    // The recursion layer monitors the predictor's hidden state for "confusion"
    // via a learned probe. When confusion exceeds a threshold, it delegates to
    // an external memory tool (here: a local vector store), retrieves compressed
    // knowledge, and injects it back into the residual stream:
    //   h_new = h + W_inject · V_knowledge

    println!("── Recursion Layer (State-Space Delegation) ──");

    // Build a model with recursion enabled
    let mut rec_config = config.clone();
    rec_config.recursion = RecursionConfig {
        enabled: true,
        knowledge_dim: config.shared_embed_dim, // match embed dim for demo
        confusion_threshold: 0.0,               // always trigger for demo
        confusion_dims: config.predictor.d_model.min(16),
        inject_after_layer: 0,                  // inject after first layer
        max_depth: 1,
        smoothing: 1.0,                         // instant response
    };

    let mut rec_model = Mamba3Jepa::new(rec_config.clone());

    // Populate the local knowledge base with synthetic entries
    let mut memory = LocalMemoryTool::new(config.shared_embed_dim);
    for i in 0..5 {
        let mut key = vec![0.0f32; dim];
        let mut val = vec![0.0f32; dim];
        key[i % dim] = 1.0;
        for d in 0..dim {
            val[d] = (i as f32 + 1.0) * 0.1;
        }
        memory.add_entry(key, val);
    }
    println!("  Knowledge base: {} entries, dim={}", memory.len(), dim);
    rec_model.set_memory_tool(Box::new(memory));

    // Generate synthetic data for the recursion forward pass
    let rec_n_qry = 4;
    let rec_n_tgt = 4;
    let (rec_images, rec_query, rec_target) = generate_batch(
        &mut rng, 1, config.vit.in_channels, h, w, rec_n_qry, rec_n_tgt, vocab_size,
    );

    // Run a forward pass with recursion active
    let rec_output = rec_model.train_forward(
        &rec_images, &rec_query, &rec_target,
        1, h, w, rec_n_qry, rec_n_tgt, temperature,
    );
    println!("  Recursion forward pass: loss={:.4}", rec_output.loss);
    assert!(rec_output.loss.is_finite(), "recursion loss should be finite");

    // Check delegation count
    if let Some(ref rec) = rec_model.predictor.recursion {
        println!("  Delegations performed: {}", rec.delegation_count());
        println!("  Confusion score: {:.4}", rec.monitor.current_score());
    }
    println!();

    // ── 9. Training Summary ────────────────────────────────────────────────

    let total_time = stage1_time + stage2_time;
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Training Summary                                          ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Stage 1 (pretrain):  {:3} steps, avg_loss={:.4}           ║", stage1_steps, avg_loss_s1);
    println!("║  Stage 2 (SFT):      {:3} steps, avg_loss={:.4}           ║", stage2_steps, avg_loss_s2);
    println!("║  Total time:         {:.1}ms                               ║", total_time.as_secs_f64() * 1000.0);
    println!("║                                                            ║");
    println!("║  Key Mamba-3 advantages over Transformer-based VL-JEPA:    ║");
    println!("║  • O(N) sequence processing (vs O(N²) attention)           ║");
    println!("║  • O(1) per-step decode (vs growing KV-cache)              ║");
    println!("║  • Complex-valued state tracks rotational dynamics         ║");
    println!("║  • Trapezoidal discretization for 2nd-order accuracy       ║");
    println!("║  • Selective decoding via SSM state drift monitoring       ║");
    println!("║  • Recursion layer: external memory via State-Space Deleg.  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Next steps:");
    println!("  1. Replace synthetic data with real image-text pairs (CC3M, WebVid)");
    println!("  2. Load pretrained V-JEPA 2 ViT weights: load_vit_weights()");
    println!("  3. Use AdamW optimizer from kore-optim with cosine LR schedule");
    println!("  4. Scale to Mamba3JepaConfig::small() for production training");
    println!("  5. Add kore-autograd for automatic differentiation");
    println!("  6. Plug in vector DB backend via MemoryTool trait for recursion");
}

/// Compute average L2 norm of the first vector in a batch.
fn vec_l2_norm(data: &[f32], dim: usize) -> f32 {
    let sum_sq: f32 = data[..dim].iter().map(|x| x * x).sum();
    sum_sq.sqrt()
}
