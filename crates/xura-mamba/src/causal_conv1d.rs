//! Causal depthwise 1D convolution for Mamba blocks.
//!
//! Implements left-padded (causal) depthwise convolution where output at position t
//! only depends on inputs at positions ≤ t.

/// SiLU (Swish) activation: x * sigmoid(x)
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Causal depthwise Conv1d forward pass.
///
/// # Arguments
/// - `x`: input, shape (batch, channels, seq_len)
/// - `weight`: depthwise kernel, shape (channels, kernel_size)
/// - `bias`: optional bias, shape (channels,)
/// - `activation`: if true, applies SiLU activation after convolution
///
/// # Returns
/// Output of shape (batch, channels, seq_len)
pub fn causal_conv1d_fn(
    x: &[f32],
    batch: usize,
    channels: usize,
    seq_len: usize,
    weight: &[f32],
    kernel_size: usize,
    bias: Option<&[f32]>,
    activation: bool,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * channels * seq_len];

    for b in 0..batch {
        for ch in 0..channels {
            for l in 0..seq_len {
                let mut acc = 0.0f32;

                for k in 0..kernel_size {
                    // Causal: we look at positions l, l-1, ..., l-(kernel_size-1)
                    // Weight index 0 corresponds to the most recent input
                    let input_pos = l as isize - k as isize;
                    if input_pos >= 0 {
                        let x_idx = b * channels * seq_len + ch * seq_len + input_pos as usize;
                        let w_idx = ch * kernel_size + k;
                        acc += x[x_idx] * weight[w_idx];
                    }
                    // Positions before 0 are implicitly zero-padded
                }

                if let Some(bias_data) = bias {
                    acc += bias_data[ch];
                }

                if activation {
                    acc = silu(acc);
                }

                let out_idx = b * channels * seq_len + ch * seq_len + l;
                output[out_idx] = acc;
            }
        }
    }

    output
}

/// Single-step causal conv1d update for autoregressive decoding.
///
/// Updates `conv_state` in-place (rolling buffer) and returns the output for one step.
///
/// # Arguments
/// - `x`: new input, shape (batch, channels)
/// - `conv_state`: rolling buffer, shape (batch, channels, kernel_size) — mutated in-place
/// - `weight`: depthwise kernel, shape (channels, kernel_size)
/// - `bias`: optional bias, shape (channels,)
/// - `activation`: if true, applies SiLU activation
///
/// # Returns
/// Output for this step, shape (batch, channels)
pub fn causal_conv1d_update(
    x: &[f32],
    batch: usize,
    channels: usize,
    conv_state: &mut [f32],
    kernel_size: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    activation: bool,
) -> Vec<f32> {
    assert!(kernel_size > 0, "causal_conv1d_update: kernel_size must be > 0");
    let mut output = vec![0.0f32; batch * channels];

    for b in 0..batch {
        for ch in 0..channels {
            // Shift state left by 1 (drop oldest, append new)
            let state_base = b * channels * kernel_size + ch * kernel_size;
            for k in 0..(kernel_size - 1) {
                conv_state[state_base + k] = conv_state[state_base + k + 1];
            }
            conv_state[state_base + kernel_size - 1] = x[b * channels + ch];

            // Compute convolution over the state buffer
            let mut acc = 0.0f32;
            for k in 0..kernel_size {
                let w_idx = ch * kernel_size + k;
                // State is stored oldest-first, weight[0] is most recent
                // So weight[k] multiplies state[kernel_size - 1 - k]
                acc += conv_state[state_base + kernel_size - 1 - k] * weight[w_idx];
            }

            if let Some(bias_data) = bias {
                acc += bias_data[ch];
            }

            if activation {
                acc = silu(acc);
            }

            output[b * channels + ch] = acc;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_conv1d_shape() {
        let batch = 2;
        let channels = 4;
        let seq_len = 8;
        let kernel_size = 3;

        let x = vec![0.1f32; batch * channels * seq_len];
        let weight = vec![0.1f32; channels * kernel_size];
        let bias = vec![0.0f32; channels];

        let out = causal_conv1d_fn(&x, batch, channels, seq_len, &weight, kernel_size, Some(&bias), false);
        assert_eq!(out.len(), batch * channels * seq_len);
    }

    #[test]
    fn test_causal_conv1d_causality() {
        // Verify output at position t doesn't depend on input at t+1
        let batch = 1;
        let channels = 1;
        let seq_len = 4;
        let kernel_size = 2;

        // x = [1, 0, 0, 0], weight = [1, 1]
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let weight = vec![1.0, 1.0]; // w[0]=recent, w[1]=older

        let out = causal_conv1d_fn(&x, batch, channels, seq_len, &weight, kernel_size, None, false);

        // t=0: w[0]*x[0] + w[1]*0 = 1.0
        assert!((out[0] - 1.0).abs() < 1e-5);
        // t=1: w[0]*x[1] + w[1]*x[0] = 0 + 1 = 1.0
        assert!((out[1] - 1.0).abs() < 1e-5);
        // t=2: w[0]*x[2] + w[1]*x[1] = 0
        assert!((out[2] - 0.0).abs() < 1e-5);
        // t=3: w[0]*x[3] + w[1]*x[2] = 0
        assert!((out[3] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_causal_conv1d_update_matches_forward() {
        let batch = 1;
        let channels = 2;
        let kernel_size = 3;
        let seq_len = 5;

        let weight = vec![0.5, 0.3, 0.1, 0.2, 0.4, 0.6]; // (2, 3)
        let bias = vec![0.1, -0.1];

        // Create input sequence
        let x_seq: Vec<f32> = (0..channels * seq_len).map(|i| (i as f32) * 0.1).collect();

        // Full forward pass
        let full_out = causal_conv1d_fn(
            &x_seq, batch, channels, seq_len, &weight, kernel_size, Some(&bias), false,
        );

        // Step-by-step update
        let mut conv_state = vec![0.0f32; batch * channels * kernel_size];
        let mut step_out = Vec::new();

        for l in 0..seq_len {
            let mut x_step = vec![0.0f32; batch * channels];
            for ch in 0..channels {
                x_step[ch] = x_seq[ch * seq_len + l];
            }
            let out = causal_conv1d_update(
                &x_step, batch, channels, &mut conv_state, kernel_size, &weight, Some(&bias), false,
            );
            step_out.push(out);
        }

        // Compare: full_out is (batch, channels, seq_len), step_out[l] is (batch, channels)
        for l in 0..seq_len {
            for ch in 0..channels {
                let full_val = full_out[ch * seq_len + l];
                let step_val = step_out[l][ch];
                assert!(
                    (full_val - step_val).abs() < 1e-4,
                    "Mismatch at ch={}, l={}: full={}, step={}",
                    ch, l, full_val, step_val,
                );
            }
        }
    }

    #[test]
    fn test_causal_conv1d_with_activation() {
        let x = vec![1.0, -1.0, 0.5, -0.5];
        let weight = vec![1.0]; // identity kernel
        let out = causal_conv1d_fn(&x, 1, 1, 4, &weight, 1, None, true);

        // SiLU(1.0) ≈ 0.7311, SiLU(-1.0) ≈ -0.2689
        assert!((out[0] - silu(1.0)).abs() < 1e-4);
        assert!((out[1] - silu(-1.0)).abs() < 1e-4);
    }
}
