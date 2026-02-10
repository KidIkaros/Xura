//! FFTConv — FFT-based convolution module wrapping S4 kernels.
//!
//! Generates a convolution kernel from SSM parameters, then applies it to the input
//! using FFT-based convolution: y = IFFT(FFT(x) * FFT(k)) + D*x.

use rustfft::{FftPlanner, num_complex::Complex as RustFftComplex};

use crate::complex_utils::C32;
use crate::s4_kernel::{SSMKernelDiag, SSMKernelDPLR, Discretization};

/// SiLU activation.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU activation (approximate).
#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
}

/// Which S4 kernel variant to use.
pub enum S4KernelType {
    Diag(SSMKernelDiag),
    DPLR(SSMKernelDPLR),
}

impl S4KernelType {
    pub fn forward(&self, l: usize) -> Vec<f32> {
        match self {
            S4KernelType::Diag(k) => k.forward(l),
            S4KernelType::DPLR(k) => k.forward(l),
        }
    }

    pub fn setup_step(&self) -> (Vec<C32>, Vec<C32>, Vec<C32>) {
        match self {
            S4KernelType::Diag(k) => k.setup_step(),
            S4KernelType::DPLR(k) => k.setup_step(),
        }
    }

    pub fn default_state(&self, batch: usize) -> Vec<f32> {
        match self {
            S4KernelType::Diag(k) => k.default_state(batch),
            S4KernelType::DPLR(k) => k.default_state(batch),
        }
    }

    pub fn step(&self, u: &[f32], state: &mut [f32], da: &[C32], db: &[C32], dc: &[C32], batch: usize) -> Vec<f32> {
        match self {
            S4KernelType::Diag(k) => k.step(u, state, da, db, dc, batch),
            S4KernelType::DPLR(k) => k.step(u, state, da, db, dc, batch),
        }
    }

    pub fn h(&self) -> usize {
        match self {
            S4KernelType::Diag(k) => k.h,
            S4KernelType::DPLR(k) => k.diag.h,
        }
    }

    pub fn channels(&self) -> usize {
        match self {
            S4KernelType::Diag(k) => k.channels,
            S4KernelType::DPLR(k) => k.diag.channels,
        }
    }

    pub fn param_count(&self) -> usize {
        match self {
            S4KernelType::Diag(k) => k.param_count(),
            S4KernelType::DPLR(k) => k.param_count(),
        }
    }
}

/// FFTConv module: generates SSM kernel → FFT convolution → skip connection → activation.
pub struct FFTConv {
    pub d_model: usize,
    pub channels: usize,
    pub kernel: S4KernelType,
    /// Skip connection D: shape (channels, d_model)
    pub d_skip: Vec<f32>,
    /// Activation function name
    pub activation: String,
}

impl FFTConv {
    /// Create a new FFTConv module.
    ///
    /// # Arguments
    /// - `d_model`: model dimension (H)
    /// - `d_state`: SSM state dimension (N)
    /// - `channels`: number of output channels (C)
    /// - `mode`: "s4d" for diagonal, "s4" for DPLR
    /// - `init`: initialization ("legs", "diag-inv", etc.)
    /// - `activation`: "gelu", "silu", or "id"
    pub fn new(
        d_model: usize,
        d_state: usize,
        channels: usize,
        mode: &str,
        init: &str,
        activation: &str,
        dt_min: f32,
        dt_max: f32,
    ) -> Self {
        let kernel = match mode {
            "s4" | "nplr" | "dplr" => {
                S4KernelType::DPLR(SSMKernelDPLR::new(
                    d_model, d_state, channels, init, 1,
                    dt_min, dt_max, Discretization::Zoh, None,
                ))
            }
            _ => {
                // Default: S4D (diagonal)
                S4KernelType::Diag(SSMKernelDiag::new(
                    d_model, d_state, channels, init,
                    dt_min, dt_max, Discretization::Zoh, None,
                ))
            }
        };

        // D skip connection: (channels, d_model) random
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let d_skip: Vec<f32> = (0..channels * d_model)
            .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
            .collect();

        Self {
            d_model,
            channels,
            kernel,
            d_skip,
            activation: activation.to_string(),
        }
    }

    /// Forward pass: apply SSM convolution to input.
    ///
    /// `x`: shape (batch, d_model, seq_len) — transposed format
    /// Returns: shape (batch, channels * d_model, seq_len)
    pub fn forward(&self, x: &[f32], batch: usize, seq_len: usize) -> Vec<f32> {
        let h = self.d_model;
        let c = self.channels;

        // 1. Generate convolution kernel: (C, H, L)
        let k = self.kernel.forward(seq_len);

        // 2. FFT-based convolution using rustfft
        let conv_len = seq_len + seq_len; // pad to avoid circular convolution
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(conv_len);
        let ifft = planner.plan_fft_inverse(conv_len);

        let mut output = vec![0.0f32; batch * c * h * seq_len];

        for bi in 0..batch {
            for ci in 0..c {
                for hi in 0..h {
                    // Prepare kernel FFT
                    let mut k_buf: Vec<RustFftComplex<f32>> = vec![RustFftComplex::new(0.0, 0.0); conv_len];
                    for l in 0..seq_len {
                        k_buf[l] = RustFftComplex::new(k[ci * h * seq_len + hi * seq_len + l], 0.0);
                    }
                    fft.process(&mut k_buf);

                    // Prepare input FFT
                    let mut x_buf: Vec<RustFftComplex<f32>> = vec![RustFftComplex::new(0.0, 0.0); conv_len];
                    for l in 0..seq_len {
                        x_buf[l] = RustFftComplex::new(x[bi * h * seq_len + hi * seq_len + l], 0.0);
                    }
                    fft.process(&mut x_buf);

                    // Multiply in frequency domain
                    let mut y_buf: Vec<RustFftComplex<f32>> = k_buf.iter().zip(x_buf.iter())
                        .map(|(&ki, &xi)| ki * xi)
                        .collect();

                    // IFFT
                    ifft.process(&mut y_buf);

                    // Extract and normalize
                    let out_base = bi * c * h * seq_len + ci * h * seq_len + hi * seq_len;
                    for l in 0..seq_len {
                        output[out_base + l] = y_buf[l].re / conv_len as f32;
                    }

                    // Add skip connection: D * x
                    let d_val = self.d_skip[ci * h + hi];
                    for l in 0..seq_len {
                        output[out_base + l] += d_val * x[bi * h * seq_len + hi * seq_len + l];
                    }
                }
            }
        }

        // 3. Apply activation
        apply_activation(&mut output, &self.activation);

        output
    }

    /// Recurrent step for autoregressive inference.
    ///
    /// `x`: shape (batch, H)
    /// `state`: complex SSM state
    /// Returns: (output shape (batch, C*H), updated state)
    pub fn step(
        &self,
        x: &[f32],
        state: &mut [f32],
        da: &[C32],
        db: &[C32],
        dc: &[C32],
        batch: usize,
    ) -> Vec<f32> {
        let h = self.d_model;
        let c = self.channels;

        // SSM step: (batch, C, H)
        let y = self.kernel.step(x, state, da, db, dc, batch);

        // Add skip connection and reshape to (batch, C*H)
        let mut output = vec![0.0f32; batch * c * h];
        for bi in 0..batch {
            for ci in 0..c {
                for hi in 0..h {
                    let y_val = y[bi * c * h + ci * h + hi];
                    let d_val = self.d_skip[ci * h + hi];
                    output[bi * c * h + ci * h + hi] = y_val + d_val * x[bi * h + hi];
                }
            }
        }

        apply_activation(&mut output, &self.activation);
        output
    }

    /// Total parameter count.
    pub fn param_count(&self) -> usize {
        self.kernel.param_count() + self.d_skip.len()
    }
}

fn apply_activation(data: &mut [f32], activation: &str) {
    match activation {
        "gelu" => {
            for v in data.iter_mut() {
                *v = gelu(*v);
            }
        }
        "silu" | "swish" => {
            for v in data.iter_mut() {
                *v = silu(*v);
            }
        }
        _ => {} // identity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_conv_create() {
        let conv = FFTConv::new(8, 16, 1, "s4d", "diag-inv", "gelu", 0.001, 0.1);
        assert_eq!(conv.d_model, 8);
        assert_eq!(conv.channels, 1);
    }

    #[test]
    fn test_fft_conv_forward_shape() {
        let conv = FFTConv::new(4, 8, 1, "s4d", "diag-inv", "gelu", 0.001, 0.1);
        let batch = 2;
        let seq_len = 16;
        let x = vec![0.1f32; batch * 4 * seq_len];
        let y = conv.forward(&x, batch, seq_len);
        assert_eq!(y.len(), batch * 1 * 4 * seq_len); // B * C * H * L
    }

    #[test]
    fn test_fft_conv_dplr_forward() {
        let conv = FFTConv::new(4, 8, 1, "s4", "diag-inv", "gelu", 0.001, 0.1);
        let batch = 1;
        let seq_len = 8;
        let x = vec![0.1f32; batch * 4 * seq_len];
        let y = conv.forward(&x, batch, seq_len);
        assert_eq!(y.len(), batch * 1 * 4 * seq_len);
    }

    #[test]
    fn test_fft_conv_step() {
        let conv = FFTConv::new(4, 8, 1, "s4d", "diag-inv", "id", 0.001, 0.1);
        let (da, db, dc) = conv.kernel.setup_step();
        let mut state = conv.kernel.default_state(1);
        let x = vec![0.1f32; 4]; // batch=1, H=4
        let y = conv.step(&x, &mut state, &da, &db, &dc, 1);
        assert_eq!(y.len(), 1 * 4); // batch * C * H
    }
}
