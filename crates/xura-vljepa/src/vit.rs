//! Vision Transformer (ViT) — X-Encoder for Mamba3-JEPA.
//!
//! Implements a standard ViT: PatchEmbed → [CLS] + positional embed → N × VitBlock → LayerNorm.
//! Designed to be frozen after loading pretrained V-JEPA 2 weights.

use rand::Rng;

use crate::config::VitConfig;

/// GELU activation (approximate).
#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
}

/// Patch embedding: Conv2d(in_channels, d_model, kernel=patch_size, stride=patch_size).
pub struct PatchEmbed {
    /// Weight: (d_model, in_channels, patch_size, patch_size)
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub d_model: usize,
    pub patch_size: usize,
    pub in_channels: usize,
}

impl PatchEmbed {
    pub fn new(config: &VitConfig) -> Self {
        let mut rng = rand::thread_rng();
        let fan_in = config.in_channels * config.patch_size * config.patch_size;
        let std = (2.0 / fan_in as f32).sqrt();
        let weight: Vec<f32> = (0..config.d_model * fan_in)
            .map(|_| rng.gen_range(-std..std))
            .collect();
        let bias = vec![0.0f32; config.d_model];
        Self {
            weight,
            bias,
            d_model: config.d_model,
            patch_size: config.patch_size,
            in_channels: config.in_channels,
        }
    }

    /// Forward: (batch, in_channels, H, W) → (batch, num_patches, d_model)
    ///
    /// Input is flat f32: batch * in_channels * H * W
    /// H and W must be divisible by patch_size.
    pub fn forward(&self, input: &[f32], batch: usize, h: usize, w: usize) -> Vec<f32> {
        let ps = self.patch_size;
        let gh = h / ps;
        let gw = w / ps;
        let num_patches = gh * gw;
        let fan_in = self.in_channels * ps * ps;

        let mut output = vec![0.0f32; batch * num_patches * self.d_model];

        for b in 0..batch {
            for py in 0..gh {
                for px in 0..gw {
                    let patch_idx = py * gw + px;
                    for d in 0..self.d_model {
                        let mut acc = self.bias[d];
                        for c in 0..self.in_channels {
                            for ky in 0..ps {
                                for kx in 0..ps {
                                    let iy = py * ps + ky;
                                    let ix = px * ps + kx;
                                    let input_idx = b * self.in_channels * h * w
                                        + c * h * w
                                        + iy * w
                                        + ix;
                                    let weight_idx = d * fan_in
                                        + c * ps * ps
                                        + ky * ps
                                        + kx;
                                    acc += input[input_idx] * self.weight[weight_idx];
                                }
                            }
                        }
                        output[b * num_patches * self.d_model + patch_idx * self.d_model + d] = acc;
                    }
                }
            }
        }

        output
    }
}

/// LayerNorm operating on flat f32 slices.
pub struct LayerNorm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub d: usize,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(d: usize) -> Self {
        Self {
            weight: vec![1.0f32; d],
            bias: vec![0.0f32; d],
            d,
            eps: 1e-6,
        }
    }

    /// Normalize input of shape (n_rows, d).
    pub fn forward(&self, x: &[f32], n_rows: usize) -> Vec<f32> {
        let d = self.d;
        let mut out = vec![0.0f32; n_rows * d];
        for row in 0..n_rows {
            let start = row * d;
            let slice = &x[start..start + d];
            let mean: f32 = slice.iter().sum::<f32>() / d as f32;
            let var: f32 = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let inv_std = 1.0 / (var + self.eps).sqrt();
            for j in 0..d {
                out[start + j] = (slice[j] - mean) * inv_std * self.weight[j] + self.bias[j];
            }
        }
        out
    }
}

/// Multi-head self-attention (bidirectional, no causal mask).
pub struct MultiHeadAttention {
    /// QKV projection: (3 * d_model, d_model)
    pub qkv_weight: Vec<f32>,
    pub qkv_bias: Vec<f32>,
    /// Output projection: (d_model, d_model)
    pub out_weight: Vec<f32>,
    pub out_bias: Vec<f32>,
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        let mut rng = rand::thread_rng();
        let head_dim = d_model / n_heads;
        let std = (2.0 / (d_model + d_model) as f32).sqrt();

        let qkv_weight: Vec<f32> = (0..3 * d_model * d_model)
            .map(|_| rng.gen_range(-std..std))
            .collect();
        let qkv_bias = vec![0.0f32; 3 * d_model];
        let out_weight: Vec<f32> = (0..d_model * d_model)
            .map(|_| rng.gen_range(-std..std))
            .collect();
        let out_bias = vec![0.0f32; d_model];

        Self { qkv_weight, qkv_bias, out_weight, out_bias, d_model, n_heads, head_dim }
    }

    /// Forward: (batch * seq_len, d_model) → (batch * seq_len, d_model)
    /// `n` = batch * seq_len, `seq_len` = tokens per batch element
    pub fn forward(&self, x: &[f32], batch: usize, seq_len: usize) -> Vec<f32> {
        let dm = self.d_model;
        let n = batch * seq_len;
        let hd = self.head_dim;
        let nh = self.n_heads;

        // QKV projection: (n, dm) -> (n, 3*dm)
        let mut qkv = vec![0.0f32; n * 3 * dm];
        for i in 0..n {
            for o in 0..3 * dm {
                let mut acc = self.qkv_bias[o];
                for k in 0..dm {
                    acc += x[i * dm + k] * self.qkv_weight[o * dm + k];
                }
                qkv[i * 3 * dm + o] = acc;
            }
        }

        // Split Q, K, V and compute attention per batch
        let mut attn_out = vec![0.0f32; n * dm];

        for b in 0..batch {
            for h in 0..nh {
                // Compute attention scores
                let scale = 1.0 / (hd as f32).sqrt();
                let mut scores = vec![0.0f32; seq_len * seq_len];

                for qi in 0..seq_len {
                    for ki in 0..seq_len {
                        let mut dot = 0.0f32;
                        for d in 0..hd {
                            let q_idx = (b * seq_len + qi) * 3 * dm + h * hd + d;
                            let k_idx = (b * seq_len + ki) * 3 * dm + dm + h * hd + d;
                            dot += qkv[q_idx] * qkv[k_idx];
                        }
                        scores[qi * seq_len + ki] = dot * scale;
                    }
                }

                // Softmax per row
                for qi in 0..seq_len {
                    let row_start = qi * seq_len;
                    let max_val = scores[row_start..row_start + seq_len]
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for ki in 0..seq_len {
                        scores[row_start + ki] = (scores[row_start + ki] - max_val).exp();
                        sum += scores[row_start + ki];
                    }
                    for ki in 0..seq_len {
                        scores[row_start + ki] /= sum;
                    }
                }

                // Weighted sum of V
                for qi in 0..seq_len {
                    for d in 0..hd {
                        let mut acc = 0.0f32;
                        for ki in 0..seq_len {
                            let v_idx = (b * seq_len + ki) * 3 * dm + 2 * dm + h * hd + d;
                            acc += scores[qi * seq_len + ki] * qkv[v_idx];
                        }
                        attn_out[(b * seq_len + qi) * dm + h * hd + d] = acc;
                    }
                }
            }
        }

        // Output projection: (n, dm) -> (n, dm)
        let mut output = vec![0.0f32; n * dm];
        for i in 0..n {
            for o in 0..dm {
                let mut acc = self.out_bias[o];
                for k in 0..dm {
                    acc += attn_out[i * dm + k] * self.out_weight[o * dm + k];
                }
                output[i * dm + o] = acc;
            }
        }

        output
    }
}

/// Feed-forward network: Linear → GELU → Linear.
pub struct FeedForward {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub d_model: usize,
    pub d_ff: usize,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std1 = (2.0 / (d_model + d_ff) as f32).sqrt();
        let std2 = (2.0 / (d_ff + d_model) as f32).sqrt();
        Self {
            w1: (0..d_ff * d_model).map(|_| rng.gen_range(-std1..std1)).collect(),
            b1: vec![0.0f32; d_ff],
            w2: (0..d_model * d_ff).map(|_| rng.gen_range(-std2..std2)).collect(),
            b2: vec![0.0f32; d_model],
            d_model,
            d_ff,
        }
    }

    /// Forward: (n, d_model) → (n, d_model)
    pub fn forward(&self, x: &[f32], n: usize) -> Vec<f32> {
        let dm = self.d_model;
        let df = self.d_ff;

        // First linear + GELU
        let mut hidden = vec![0.0f32; n * df];
        for i in 0..n {
            for o in 0..df {
                let mut acc = self.b1[o];
                for k in 0..dm {
                    acc += x[i * dm + k] * self.w1[o * dm + k];
                }
                hidden[i * df + o] = gelu(acc);
            }
        }

        // Second linear
        let mut output = vec![0.0f32; n * dm];
        for i in 0..n {
            for o in 0..dm {
                let mut acc = self.b2[o];
                for k in 0..df {
                    acc += hidden[i * df + k] * self.w2[o * df + k];
                }
                output[i * dm + o] = acc;
            }
        }

        output
    }
}

/// ViT block: LayerNorm → MHA → residual → LayerNorm → FFN → residual.
pub struct VitBlock {
    pub norm1: LayerNorm,
    pub attn: MultiHeadAttention,
    pub norm2: LayerNorm,
    pub ffn: FeedForward,
    pub d_model: usize,
}

impl VitBlock {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        Self {
            norm1: LayerNorm::new(d_model),
            attn: MultiHeadAttention::new(d_model, n_heads),
            norm2: LayerNorm::new(d_model),
            ffn: FeedForward::new(d_model, d_ff),
            d_model,
        }
    }

    /// Forward: (batch, seq_len, d_model) as flat → same shape.
    pub fn forward(&self, x: &[f32], batch: usize, seq_len: usize) -> Vec<f32> {
        let n = batch * seq_len;
        // Pre-norm → attention → residual
        let normed1 = self.norm1.forward(x, n);
        let attn_out = self.attn.forward(&normed1, batch, seq_len);
        let mut residual: Vec<f32> = x.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();

        // Pre-norm → FFN → residual
        let normed2 = self.norm2.forward(&residual, n);
        let ffn_out = self.ffn.forward(&normed2, n);
        for i in 0..residual.len() {
            residual[i] += ffn_out[i];
        }

        residual
    }
}

/// Vision Encoder: PatchEmbed + [CLS] + positional embed + N × VitBlock + LayerNorm.
pub struct VisionEncoder {
    pub config: VitConfig,
    pub patch_embed: PatchEmbed,
    /// CLS token: (d_model,)
    pub cls_token: Vec<f32>,
    /// Positional embeddings: (1 + num_patches, d_model)
    pub pos_embed: Vec<f32>,
    pub blocks: Vec<VitBlock>,
    pub final_norm: LayerNorm,
    pub frozen: bool,
}

impl VisionEncoder {
    pub fn new(config: VitConfig) -> Self {
        let mut rng = rand::thread_rng();
        let dm = config.d_model;
        let num_patches = config.num_patches();
        let seq_len = 1 + num_patches; // CLS + patches

        let patch_embed = PatchEmbed::new(&config);
        let cls_token: Vec<f32> = (0..dm).map(|_| rng.gen_range(-0.02..0.02)).collect();
        let pos_embed: Vec<f32> = (0..seq_len * dm)
            .map(|_| rng.gen_range(-0.02..0.02))
            .collect();

        let blocks: Vec<VitBlock> = (0..config.n_layers)
            .map(|_| VitBlock::new(dm, config.n_heads, config.d_ff))
            .collect();
        let final_norm = LayerNorm::new(dm);

        Self {
            config,
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
            final_norm,
            frozen: false,
        }
    }

    /// Forward: (batch, in_channels, H, W) → (batch, 1 + num_patches, d_model)
    ///
    /// Returns flat f32 of shape (batch, seq_len, d_model) where seq_len = 1 + num_patches.
    pub fn forward(&self, input: &[f32], batch: usize, h: usize, w: usize) -> Vec<f32> {
        let dm = self.config.d_model;
        let num_patches = (h / self.config.patch_size) * (w / self.config.patch_size);
        let seq_len = 1 + num_patches;

        // Patch embedding: (batch, num_patches, d_model)
        let patches = self.patch_embed.forward(input, batch, h, w);

        // Prepend CLS token and add positional embeddings
        let mut hidden = vec![0.0f32; batch * seq_len * dm];
        for b in 0..batch {
            // CLS token + pos_embed[0]
            for d in 0..dm {
                hidden[b * seq_len * dm + d] = self.cls_token[d] + self.pos_embed[d];
            }
            // Patch tokens + pos_embed[1..]
            for p in 0..num_patches {
                for d in 0..dm {
                    hidden[b * seq_len * dm + (1 + p) * dm + d] =
                        patches[b * num_patches * dm + p * dm + d]
                            + self.pos_embed[(1 + p) * dm + d];
                }
            }
        }

        // Transformer blocks
        for block in &self.blocks {
            hidden = block.forward(&hidden, batch, seq_len);
        }

        // Final LayerNorm
        self.final_norm.forward(&hidden, batch * seq_len)
    }

    /// Extract only patch tokens (exclude CLS), shape (batch, num_patches, d_model).
    pub fn forward_patches(&self, input: &[f32], batch: usize, h: usize, w: usize) -> Vec<f32> {
        let dm = self.config.d_model;
        let num_patches = (h / self.config.patch_size) * (w / self.config.patch_size);
        let seq_len = 1 + num_patches;

        let full = self.forward(input, batch, h, w);

        // Strip CLS token
        let mut patches = vec![0.0f32; batch * num_patches * dm];
        for b in 0..batch {
            let src_start = b * seq_len * dm + dm; // skip CLS
            let dst_start = b * num_patches * dm;
            patches[dst_start..dst_start + num_patches * dm]
                .copy_from_slice(&full[src_start..src_start + num_patches * dm]);
        }
        patches
    }

    /// Total parameter count.
    pub fn param_count(&self) -> usize {
        let dm = self.config.d_model;
        let np = self.config.num_patches();
        let patch = self.config.in_channels * self.config.patch_size * self.config.patch_size * dm + dm;
        let cls = dm;
        let pos = (1 + np) * dm;
        let block_params = self.blocks.len() * (
            2 * (2 * dm) // 2 LayerNorms (weight + bias)
            + 3 * dm * dm + 3 * dm // QKV
            + dm * dm + dm // out proj
            + self.config.d_ff * dm + self.config.d_ff // FFN w1
            + dm * self.config.d_ff + dm // FFN w2
        );
        let final_norm = 2 * dm;
        patch + cls + pos + block_params + final_norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patch_embed_shape() {
        let config = VitConfig::tiny();
        let pe = PatchEmbed::new(&config);
        let batch = 2;
        let h = config.image_size;
        let w = config.image_size;
        let input = vec![0.1f32; batch * config.in_channels * h * w];
        let output = pe.forward(&input, batch, h, w);
        let num_patches = config.num_patches();
        assert_eq!(output.len(), batch * num_patches * config.d_model);
    }

    #[test]
    fn test_vision_encoder_shape() {
        let config = VitConfig::tiny();
        let dm = config.d_model;
        let np = config.num_patches();
        let encoder = VisionEncoder::new(config.clone());
        let batch = 2;
        let input = vec![0.1f32; batch * config.in_channels * config.image_size * config.image_size];
        let output = encoder.forward(&input, batch, config.image_size, config.image_size);
        assert_eq!(output.len(), batch * (1 + np) * dm);
    }

    #[test]
    fn test_vision_encoder_patches_only() {
        let config = VitConfig::tiny();
        let dm = config.d_model;
        let np = config.num_patches();
        let encoder = VisionEncoder::new(config.clone());
        let batch = 1;
        let input = vec![0.1f32; batch * config.in_channels * config.image_size * config.image_size];
        let patches = encoder.forward_patches(&input, batch, config.image_size, config.image_size);
        assert_eq!(patches.len(), batch * np * dm);
    }

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = ln.forward(&input, 1);
        // Mean=2.5, should be centered
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean should be ~0, got {}", mean);
    }
}
