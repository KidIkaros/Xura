//! Structured State Space Duality (SSD) — chunked parallel scan for Mamba-2.
//!
//! Reference implementation of `mamba_chunk_scan_combined` from
//! `mamba_ssm/ops/triton/ssd_combined.py`. This is the CPU reference path
//! (non-fused), matching the slow path in `mamba2.py`.

/// SiLU (Swish) activation: x * sigmoid(x)
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softplus activation: log(1 + exp(x))
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else if x < -20.0 { 0.0 } else { (1.0 + x.exp()).ln() }
}

/// Output from chunked scan.
pub struct ChunkedScanOutput {
    /// Output tensor, shape (batch, seq_len, nheads, headdim)
    pub output: Vec<f32>,
    /// Final SSM state, shape (batch, nheads, headdim, d_state)
    pub last_state: Vec<f32>,
}

/// Chunked scan combined — the core Mamba-2 scan operation.
///
/// This implements the sequential reference path (not the Triton-optimized version).
/// The scan processes the sequence in chunks but each chunk is still sequential.
///
/// # Arguments
/// - `x`: input, shape (batch, seq_len, nheads, headdim)
/// - `dt`: time delta, shape (batch, seq_len, nheads)
/// - `a`: state matrix (negative), shape (nheads,)
/// - `b`: input-dependent B, shape (batch, seq_len, ngroups, d_state)
/// - `c`: input-dependent C, shape (batch, seq_len, ngroups, d_state)
/// - `d`: skip connection, shape (nheads,) or (nheads, headdim) if d_has_hdim
/// - `dt_bias`: bias for dt, shape (nheads,) — optional
/// - `dt_softplus`: whether to apply softplus to dt
/// - `d_has_hdim`: whether D has per-headdim values
///
/// # Layout
/// - batch, seq_len, nheads, headdim, ngroups, d_state are all explicit
/// - heads_per_group = nheads / ngroups
#[allow(clippy::too_many_arguments)]
pub fn mamba_chunk_scan_combined(
    x: &[f32],
    batch: usize,
    seq_len: usize,
    nheads: usize,
    headdim: usize,
    dt: &[f32],
    a: &[f32],
    b: &[f32],
    ngroups: usize,
    d_state: usize,
    c: &[f32],
    d: Option<&[f32]>,
    z: Option<&[f32]>,
    dt_bias: Option<&[f32]>,
    dt_softplus: bool,
    d_has_hdim: bool,
) -> ChunkedScanOutput {
    // Shape validation
    assert_eq!(x.len(), batch * seq_len * nheads * headdim,
        "x shape mismatch: expected {}, got {}", batch * seq_len * nheads * headdim, x.len());
    assert_eq!(dt.len(), batch * seq_len * nheads,
        "dt shape mismatch: expected {}, got {}", batch * seq_len * nheads, dt.len());
    assert_eq!(a.len(), nheads, "a shape mismatch: expected {}, got {}", nheads, a.len());
    assert_eq!(b.len(), batch * seq_len * ngroups * d_state,
        "b shape mismatch: expected {}, got {}", batch * seq_len * ngroups * d_state, b.len());
    assert_eq!(c.len(), batch * seq_len * ngroups * d_state,
        "c shape mismatch: expected {}, got {}", batch * seq_len * ngroups * d_state, c.len());
    assert!(nheads % ngroups == 0, "nheads ({}) must be divisible by ngroups ({})", nheads, ngroups);
    if let Some(d_skip) = d {
        if d_has_hdim {
            assert_eq!(d_skip.len(), nheads * headdim, "D shape mismatch (hdim)");
        } else {
            assert_eq!(d_skip.len(), nheads, "D shape mismatch");
        }
    }
    if let Some(z_data) = z { assert_eq!(z_data.len(), batch * seq_len * nheads * headdim, "z shape mismatch"); }
    if let Some(bias) = dt_bias { assert_eq!(bias.len(), nheads, "dt_bias shape mismatch"); }

    let heads_per_group = nheads / ngroups;
    let mut output = vec![0.0f32; batch * seq_len * nheads * headdim];
    // State: (batch, nheads, headdim, d_state)
    let mut state = vec![0.0f32; batch * nheads * headdim * d_state];

    // Process sequentially over time steps
    for b_idx in 0..batch {
        // Reset state for each batch element
        let state_base = b_idx * nheads * headdim * d_state;
        for v in &mut state[state_base..state_base + nheads * headdim * d_state] {
            *v = 0.0;
        }

        for l in 0..seq_len {
            for h in 0..nheads {
                let g = h / heads_per_group; // which group this head belongs to

                // dt value for this (batch, time, head)
                let mut dt_val = dt[b_idx * seq_len * nheads + l * nheads + h];
                if let Some(bias) = dt_bias {
                    dt_val += bias[h];
                }
                if dt_softplus {
                    dt_val = softplus(dt_val);
                }

                // dA = exp(dt * A[h])  — A is already negative
                let da = (dt_val * a[h]).exp();

                for p in 0..headdim {
                    // x value: (batch, seq_len, nheads, headdim)
                    let x_val = x[b_idx * seq_len * nheads * headdim
                        + l * nheads * headdim
                        + h * headdim
                        + p];

                    let mut y = 0.0f32;

                    for n in 0..d_state {
                        // B: (batch, seq_len, ngroups, d_state)
                        let b_val = b[b_idx * seq_len * ngroups * d_state
                            + l * ngroups * d_state
                            + g * d_state
                            + n];

                        // dBx = dt * B * x
                        let dbx = dt_val * b_val * x_val;

                        // State update: state = state * dA + dBx
                        let s_idx = state_base + h * headdim * d_state + p * d_state + n;
                        state[s_idx] = state[s_idx] * da + dbx;

                        // C: (batch, seq_len, ngroups, d_state)
                        let c_val = c[b_idx * seq_len * ngroups * d_state
                            + l * ngroups * d_state
                            + g * d_state
                            + n];

                        y += state[s_idx] * c_val;
                    }

                    // Skip connection: y += D * x
                    if let Some(d_skip) = d {
                        if d_has_hdim {
                            y += d_skip[h * headdim + p] * x_val;
                        } else {
                            y += d_skip[h] * x_val;
                        }
                    }

                    // Gating with z (if not using rmsnorm — rmsnorm handles gating separately)
                    if let Some(z_data) = z {
                        let z_val = z_data[b_idx * seq_len * nheads * headdim
                            + l * nheads * headdim
                            + h * headdim
                            + p];
                        y *= silu(z_val);
                    }

                    let out_idx = b_idx * seq_len * nheads * headdim
                        + l * nheads * headdim
                        + h * headdim
                        + p;
                    output[out_idx] = y;
                }
            }
        }
    }

    // Extract last state
    let last_state = state;

    ChunkedScanOutput { output, last_state }
}

/// Single-step SSM update for Mamba-2 decoding.
///
/// # Arguments
/// - `x`: input, shape (batch, nheads, headdim)
/// - `dt`: time delta, shape (batch, nheads)
/// - `a`: state matrix (negative), shape (nheads,)
/// - `b`: B for this step, shape (batch, ngroups, d_state)
/// - `c`: C for this step, shape (batch, ngroups, d_state)
/// - `d`: skip connection, shape (nheads,)
/// - `z`: gate, shape (batch, nheads, headdim) — optional
/// - `dt_bias`: shape (nheads,) — optional
/// - `ssm_state`: mutable, shape (batch, nheads, headdim, d_state)
///
/// # Returns
/// Output, shape (batch, nheads, headdim)
#[allow(clippy::too_many_arguments)]
pub fn mamba2_ssm_step(
    x: &[f32],
    batch: usize,
    nheads: usize,
    headdim: usize,
    dt: &[f32],
    a: &[f32],
    b: &[f32],
    ngroups: usize,
    d_state: usize,
    c: &[f32],
    d: Option<&[f32]>,
    z: Option<&[f32]>,
    dt_bias: Option<&[f32]>,
    dt_softplus: bool,
    ssm_state: &mut [f32],
) -> Vec<f32> {
    // Shape validation
    let bhd = batch * nheads * headdim;
    assert_eq!(x.len(), bhd, "x shape mismatch: expected {}, got {}", bhd, x.len());
    assert_eq!(dt.len(), batch * nheads, "dt shape mismatch: expected {}, got {}", batch * nheads, dt.len());
    assert_eq!(a.len(), nheads, "a shape mismatch: expected {}, got {}", nheads, a.len());
    assert_eq!(b.len(), batch * ngroups * d_state, "b shape mismatch: expected {}, got {}", batch * ngroups * d_state, b.len());
    assert_eq!(c.len(), batch * ngroups * d_state, "c shape mismatch: expected {}, got {}", batch * ngroups * d_state, c.len());
    assert_eq!(ssm_state.len(), batch * nheads * headdim * d_state,
        "ssm_state shape mismatch: expected {}, got {}", batch * nheads * headdim * d_state, ssm_state.len());
    assert!(nheads % ngroups == 0, "nheads must be divisible by ngroups");
    if let Some(d_skip) = d { assert_eq!(d_skip.len(), nheads, "d shape mismatch"); }
    if let Some(z_data) = z { assert_eq!(z_data.len(), bhd, "z shape mismatch"); }
    if let Some(bias) = dt_bias { assert_eq!(bias.len(), nheads, "dt_bias shape mismatch"); }

    let heads_per_group = nheads / ngroups;
    let mut output = vec![0.0f32; batch * nheads * headdim];

    for b_idx in 0..batch {
        for h in 0..nheads {
            let g = h / heads_per_group;

            let mut dt_val = dt[b_idx * nheads + h];
            if let Some(bias) = dt_bias {
                dt_val += bias[h];
            }
            if dt_softplus {
                dt_val = softplus(dt_val);
            }

            let da = (dt_val * a[h]).exp();

            for p in 0..headdim {
                let x_val = x[b_idx * nheads * headdim + h * headdim + p];
                let mut y = 0.0f32;

                for n in 0..d_state {
                    let b_val = b[b_idx * ngroups * d_state + g * d_state + n];
                    let dbx = dt_val * b_val * x_val;

                    let s_idx = b_idx * nheads * headdim * d_state
                        + h * headdim * d_state
                        + p * d_state
                        + n;
                    ssm_state[s_idx] = ssm_state[s_idx] * da + dbx;

                    let c_val = c[b_idx * ngroups * d_state + g * d_state + n];
                    y += ssm_state[s_idx] * c_val;
                }

                if let Some(d_skip) = d {
                    y += d_skip[h] * x_val;
                }

                if let Some(z_data) = z {
                    y *= silu(z_data[b_idx * nheads * headdim + h * headdim + p]);
                }

                output[b_idx * nheads * headdim + h * headdim + p] = y;
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunked_scan_shape() {
        let batch = 2;
        let seq_len = 8;
        let nheads = 4;
        let headdim = 8;
        let ngroups = 1;
        let d_state = 4;

        let x = vec![0.1f32; batch * seq_len * nheads * headdim];
        let dt = vec![0.05f32; batch * seq_len * nheads];
        let a = vec![-1.0f32; nheads];
        let b = vec![0.1f32; batch * seq_len * ngroups * d_state];
        let c = vec![0.1f32; batch * seq_len * ngroups * d_state];
        let d_skip = vec![1.0f32; nheads];

        let result = mamba_chunk_scan_combined(
            &x, batch, seq_len, nheads, headdim,
            &dt, &a, &b, ngroups, d_state, &c,
            Some(&d_skip), None, None, false, false,
        );

        assert_eq!(result.output.len(), batch * seq_len * nheads * headdim);
        assert_eq!(result.last_state.len(), batch * nheads * headdim * d_state);
    }

    #[test]
    fn test_chunked_scan_skip_only() {
        // With A=0, B=0, state stays zero. Output = D * x
        let batch = 1;
        let seq_len = 2;
        let nheads = 1;
        let headdim = 2;
        let ngroups = 1;
        let d_state = 1;

        let x = vec![1.0, 2.0, 3.0, 4.0]; // (1, 2, 1, 2)
        let dt = vec![0.0, 0.0]; // zero dt means no state update
        let a = vec![0.0];
        let b = vec![0.0, 0.0];
        let c = vec![0.0, 0.0];
        let d_skip = vec![2.0];

        let result = mamba_chunk_scan_combined(
            &x, batch, seq_len, nheads, headdim,
            &dt, &a, &b, ngroups, d_state, &c,
            Some(&d_skip), None, None, false, false,
        );

        assert!((result.output[0] - 2.0).abs() < 1e-5); // 2 * 1
        assert!((result.output[1] - 4.0).abs() < 1e-5); // 2 * 2
        assert!((result.output[2] - 6.0).abs() < 1e-5); // 2 * 3
        assert!((result.output[3] - 8.0).abs() < 1e-5); // 2 * 4
    }

    #[test]
    fn test_mamba2_ssm_step() {
        let batch = 1;
        let nheads = 2;
        let headdim = 4;
        let ngroups = 1;
        let d_state = 2;

        let x = vec![0.1f32; batch * nheads * headdim];
        let dt = vec![0.1, 0.2];
        let a = vec![-1.0, -1.0];
        let b = vec![1.0, 0.5];
        let c = vec![1.0, 1.0];
        let d_skip = vec![1.0, 1.0];
        let mut ssm_state = vec![0.0f32; batch * nheads * headdim * d_state];

        let out = mamba2_ssm_step(
            &x, batch, nheads, headdim, &dt, &a, &b, ngroups, d_state, &c,
            Some(&d_skip), None, None, false, &mut ssm_state,
        );

        assert_eq!(out.len(), batch * nheads * headdim);
        // State should be updated
        assert!(ssm_state.iter().any(|&v| v != 0.0));
    }
}
