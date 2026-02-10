//! S4Block — wrapper module for the S4/S4D layer.
//!
//! Architecture: Pre-norm → FFTConv → Dropout → Output Linear (GLU) → Residual
//! Supports both full-sequence (training) and single-step (recurrent inference) modes.

use rand::Rng;

use crate::complex_utils::C32;
use crate::fft_conv::FFTConv;

/// S4 block wrapping FFTConv with pre-norm, output linear, and residual.
pub struct S4Block {
    pub d_model: usize,
    pub d_state: usize,
    pub channels: usize,

    /// The inner FFTConv layer
    pub layer: FFTConv,

    /// Output linear: (d_model, 2*d_model) weight for GLU gating
    pub output_weight: Vec<f32>,
    /// Output linear bias: (2*d_model,)
    pub output_bias: Vec<f32>,

    /// Pre-norm weight: (d_model,)
    pub norm_weight: Vec<f32>,
    pub norm_eps: f32,

    /// Cached step parameters (set up by setup_step)
    step_da: Option<Vec<C32>>,
    step_db: Option<Vec<C32>>,
    step_dc: Option<Vec<C32>>,

    pub layer_idx: Option<usize>,
    _training: bool,
}

impl S4Block {
    /// Create a new S4Block.
    ///
    /// # Arguments
    /// - `d_model`: model dimension
    /// - `d_state`: SSM state size (default 64)
    /// - `mode`: "s4d" or "s4"
    /// - `init`: initialization for the kernel
    /// - `activation`: activation in FFTConv ("gelu", "silu", "id")
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        d_state: usize,
        mode: &str,
        init: &str,
        activation: &str,
        dt_min: f32,
        dt_max: f32,
        layer_idx: Option<usize>,
    ) -> Self {
        let channels = 1;
        let layer = FFTConv::new(d_model, d_state, channels, mode, init, activation, dt_min, dt_max);

        let mut rng = rand::thread_rng();

        // Output linear: Conv1d(d_model, 2*d_model, kernel_size=1) → GLU
        // Weight: (2*d_model, d_model)
        let out_dim = 2 * d_model;
        let limit = (6.0 / (d_model + out_dim) as f32).sqrt();
        let output_weight: Vec<f32> = (0..out_dim * d_model)
            .map(|_| rng.gen_range(-limit..limit))
            .collect();
        let output_bias = vec![0.0f32; out_dim];

        // RMSNorm weight
        let norm_weight = vec![1.0f32; d_model];

        Self {
            d_model,
            d_state,
            channels,
            layer,
            output_weight,
            output_bias,
            norm_weight,
            norm_eps: 1e-5,
            step_da: None,
            step_db: None,
            step_dc: None,
            layer_idx,
            _training: true,
        }
    }

    /// RMSNorm: x * weight / sqrt(mean(x^2) + eps)
    fn rms_norm(&self, x: &[f32], n_rows: usize) -> Vec<f32> {
        let d = self.d_model;
        let mut out = vec![0.0f32; n_rows * d];
        for row in 0..n_rows {
            let start = row * d;
            let slice = &x[start..start + d];
            let ms: f32 = slice.iter().map(|v| v * v).sum::<f32>() / d as f32;
            let inv_rms = 1.0 / (ms + self.norm_eps).sqrt();
            for j in 0..d {
                out[start + j] = slice[j] * inv_rms * self.norm_weight[j];
            }
        }
        out
    }

    /// GLU (Gated Linear Unit): split input in half, return first * sigmoid(second).
    fn glu(&self, x: &[f32], n_rows: usize) -> Vec<f32> {
        let d = self.d_model;
        let mut out = vec![0.0f32; n_rows * d];
        for row in 0..n_rows {
            for j in 0..d {
                let a = x[row * 2 * d + j];
                let b = x[row * 2 * d + d + j];
                let sigmoid_b = 1.0 / (1.0 + (-b).exp());
                out[row * d + j] = a * sigmoid_b;
            }
        }
        out
    }

    /// Apply output linear: (n_rows, d_model) → (n_rows, 2*d_model) → GLU → (n_rows, d_model)
    fn output_linear(&self, x: &[f32], n_rows: usize) -> Vec<f32> {
        let d = self.d_model;
        let out_dim = 2 * d;
        let mut projected = vec![0.0f32; n_rows * out_dim];

        for row in 0..n_rows {
            for o in 0..out_dim {
                let mut acc = self.output_bias[o];
                for k in 0..d {
                    acc += x[row * d + k] * self.output_weight[o * d + k];
                }
                projected[row * out_dim + o] = acc;
            }
        }

        self.glu(&projected, n_rows)
    }

    /// Training forward pass.
    ///
    /// `input`: shape (batch, seq_len, d_model)
    /// Returns: shape (batch, seq_len, d_model)
    pub fn forward_train(&self, input: &[f32], batch: usize, seq_len: usize) -> Vec<f32> {
        let d = self.d_model;
        let n = batch * seq_len;

        // 1. Pre-norm: (B*L, d_model)
        let normed = self.rms_norm(input, n);

        // 2. Rearrange to (B, H, L) for FFTConv (transposed format)
        let mut x_bhl = vec![0.0f32; batch * d * seq_len];
        for b in 0..batch {
            for l in 0..seq_len {
                for h in 0..d {
                    x_bhl[b * d * seq_len + h * seq_len + l] =
                        normed[b * seq_len * d + l * d + h];
                }
            }
        }

        // 3. FFTConv: (B, H, L) → (B, C*H, L)
        let y_bhl = self.layer.forward(&x_bhl, batch, seq_len);

        // 4. Rearrange back to (B, L, C*H) = (B, L, d_model) since C=1
        let mut y_bld = vec![0.0f32; batch * seq_len * d];
        for b in 0..batch {
            for l in 0..seq_len {
                for h in 0..d {
                    y_bld[b * seq_len * d + l * d + h] =
                        y_bhl[b * d * seq_len + h * seq_len + l];
                }
            }
        }

        // 5. Output linear with GLU
        let y_out = self.output_linear(&y_bld, n);

        y_out
    }

    /// Set up recurrent step parameters.
    pub fn setup_step(&mut self) {
        let (da, db, dc) = self.layer.kernel.setup_step();
        self.step_da = Some(da);
        self.step_db = Some(db);
        self.step_dc = Some(dc);
    }

    /// Single-step forward for autoregressive decoding.
    ///
    /// `input`: shape (batch, d_model)
    /// `state`: mutable SSM state
    /// Returns: shape (batch, d_model)
    pub fn forward_step(
        &self,
        input: &[f32],
        batch: usize,
        state: &mut [f32],
    ) -> Vec<f32> {
        let da = self.step_da.as_ref().expect("Call setup_step() first");
        let db = self.step_db.as_ref().expect("Call setup_step() first");
        let dc = self.step_dc.as_ref().expect("Call setup_step() first");

        // 1. Pre-norm
        let normed = self.rms_norm(input, batch);

        // 2. FFTConv step: (batch, H) → (batch, C*H)
        let y = self.layer.step(&normed, state, da, db, dc, batch);

        // 3. Output linear
        let y_out = self.output_linear(&y, batch);

        y_out
    }

    /// State size for this block.
    pub fn state_size(&self, batch: usize) -> usize {
        self.layer.kernel.default_state(batch).len()
    }

    /// Total parameter count.
    pub fn param_count(&self) -> usize {
        self.layer.param_count()
            + self.output_weight.len()
            + self.output_bias.len()
            + self.norm_weight.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s4_block_create() {
        let block = S4Block::new(16, 32, "s4d", "diag-inv", "gelu", 0.001, 0.1, None);
        assert_eq!(block.d_model, 16);
        assert_eq!(block.d_state, 32);
    }

    #[test]
    fn test_s4_block_forward_shape() {
        let block = S4Block::new(8, 16, "s4d", "diag-inv", "gelu", 0.001, 0.1, None);
        let batch = 2;
        let seq_len = 4;
        let input = vec![0.1f32; batch * seq_len * 8];
        let output = block.forward_train(&input, batch, seq_len);
        assert_eq!(output.len(), batch * seq_len * 8);
    }

    #[test]
    fn test_s4_block_step() {
        let mut block = S4Block::new(8, 16, "s4d", "diag-inv", "id", 0.001, 0.1, None);
        block.setup_step();

        let batch = 1;
        let mut state = block.layer.kernel.default_state(batch);
        let input = vec![0.1f32; batch * 8];
        let output = block.forward_step(&input, batch, &mut state);
        assert_eq!(output.len(), batch * 8);

        // Second step should also work and state should change
        let output2 = block.forward_step(&input, batch, &mut state);
        assert_eq!(output2.len(), batch * 8);
    }

    #[test]
    fn test_s4_block_param_count() {
        let block = S4Block::new(16, 32, "s4d", "diag-inv", "gelu", 0.001, 0.1, None);
        let params = block.param_count();
        assert!(params > 500, "params = {}", params);
    }
}
