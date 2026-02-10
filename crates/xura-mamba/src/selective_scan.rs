//! Selective Scan — the core SSM recurrence (Algorithm 2 from the Mamba paper).
//!
//! Reference implementation matching `selective_scan_ref` from
//! `mamba_ssm/ops/selective_scan_interface.py`.
//!
//! All tensors are stored as flat f32 slices with explicit shape indexing.

/// Softplus activation: log(1 + exp(x))
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// SiLU (Swish) activation: x * sigmoid(x)
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Result of a selective scan operation.
pub struct SelectiveScanOutput {
    /// Output tensor data, shape (batch, dim, seq_len)
    pub output: Vec<f32>,
    /// Last hidden state, shape (batch, dim, d_state)
    pub last_state: Vec<f32>,
}

/// Reference implementation of the selective scan algorithm.
///
/// # Arguments
/// - `u`: input, shape (batch, dim, seq_len)
/// - `delta`: time step, shape (batch, dim, seq_len)
/// - `a`: state matrix (log-space, already negated), shape (dim, d_state)
/// - `b`: input-dependent B, shape (batch, d_state, seq_len)
/// - `c`: input-dependent C, shape (batch, d_state, seq_len)
/// - `d`: skip connection, shape (dim,)
/// - `z`: gate, shape (batch, dim, seq_len) — optional
/// - `delta_bias`: bias added to delta before softplus, shape (dim,) — optional
/// - `delta_softplus`: whether to apply softplus to delta
///
/// # Returns
/// `SelectiveScanOutput` with output (batch, dim, seq_len) and last_state (batch, dim, d_state)
pub fn selective_scan_ref(
    u: &[f32],
    batch: usize,
    dim: usize,
    seq_len: usize,
    delta: &[f32],
    a: &[f32],
    d_state: usize,
    b: &[f32],
    c: &[f32],
    d: Option<&[f32]>,
    z: Option<&[f32]>,
    delta_bias: Option<&[f32]>,
    delta_softplus: bool,
) -> Result<SelectiveScanOutput, String> {
    // Shape validation
    let bdl = batch * dim * seq_len;
    if u.len() != bdl { return Err(format!("u shape mismatch: expected {} (batch*dim*seq_len), got {}", bdl, u.len())); }
    if delta.len() != bdl { return Err(format!("delta shape mismatch: expected {}, got {}", bdl, delta.len())); }
    if a.len() != dim * d_state { return Err(format!("a shape mismatch: expected {} (dim*d_state), got {}", dim * d_state, a.len())); }
    if b.len() != batch * d_state * seq_len { return Err(format!("b shape mismatch: expected {}, got {}", batch * d_state * seq_len, b.len())); }
    if c.len() != batch * d_state * seq_len { return Err(format!("c shape mismatch: expected {}, got {}", batch * d_state * seq_len, c.len())); }
    if let Some(d_skip) = d { if d_skip.len() != dim { return Err(format!("d shape mismatch: expected {}, got {}", dim, d_skip.len())); } }
    if let Some(z_data) = z { if z_data.len() != bdl { return Err(format!("z shape mismatch: expected {}, got {}", bdl, z_data.len())); } }
    if let Some(bias) = delta_bias { if bias.len() != dim { return Err(format!("delta_bias shape mismatch: expected {}, got {}", dim, bias.len())); } }

    // Prepare delta with bias and softplus
    let mut dt = delta.to_vec();
    if let Some(bias) = delta_bias {
        for b_idx in 0..batch {
            for d_idx in 0..dim {
                for l in 0..seq_len {
                    let idx = b_idx * dim * seq_len + d_idx * seq_len + l;
                    dt[idx] += bias[d_idx];
                }
            }
        }
    }
    if delta_softplus {
        for v in dt.iter_mut() {
            *v = softplus(*v);
        }
    }

    // State: x[batch, dim, d_state] — initialized to zero
    let mut x = vec![0.0f32; batch * dim * d_state];
    let mut output = vec![0.0f32; batch * dim * seq_len];
    let mut last_state = vec![0.0f32; batch * dim * d_state];

    // Sequential scan over time steps
    for l in 0..seq_len {
        for b_idx in 0..batch {
            for d_idx in 0..dim {
                let dt_val = dt[b_idx * dim * seq_len + d_idx * seq_len + l];
                let u_val = u[b_idx * dim * seq_len + d_idx * seq_len + l];

                let mut y = 0.0f32;

                for n in 0..d_state {
                    // A is (dim, d_state) — already negative exponential
                    let a_val = a[d_idx * d_state + n];
                    // deltaA = exp(dt * A)
                    let da = (dt_val * a_val).exp();
                    // B is (batch, d_state, seq_len)
                    let b_val = b[b_idx * d_state * seq_len + n * seq_len + l];
                    // deltaB * u
                    let db_u = dt_val * b_val * u_val;

                    let x_idx = b_idx * dim * d_state + d_idx * d_state + n;
                    x[x_idx] = da * x[x_idx] + db_u;

                    // C is (batch, d_state, seq_len)
                    let c_val = c[b_idx * d_state * seq_len + n * seq_len + l];
                    y += x[x_idx] * c_val;
                }

                // Skip connection: y += D * u
                if let Some(d_skip) = d {
                    y += d_skip[d_idx] * u_val;
                }

                let out_idx = b_idx * dim * seq_len + d_idx * seq_len + l;
                output[out_idx] = y;

                // Save last state at the final timestep
                if l == seq_len - 1 {
                    for n in 0..d_state {
                        let x_idx = b_idx * dim * d_state + d_idx * d_state + n;
                        last_state[x_idx] = x[x_idx];
                    }
                }
            }
        }
    }

    // Apply gating: output = output * silu(z)
    if let Some(z_data) = z {
        for i in 0..output.len() {
            output[i] *= silu(z_data[i]);
        }
    }

    Ok(SelectiveScanOutput { output, last_state })
}

/// Single-step selective scan update for autoregressive decoding.
///
/// Updates ssm_state in-place and returns the output for one time step.
///
/// # Arguments
/// - `x`: input for this step, shape (batch, dim)
/// - `dt`: delta for this step (after linear projection), shape (batch, dim)
/// - `a`: state matrix, shape (dim, d_state) — negative values
/// - `b`: B for this step, shape (batch, d_state)
/// - `c`: C for this step, shape (batch, d_state)
/// - `d`: skip connection, shape (dim,)
/// - `z`: gate for this step, shape (batch, dim) — optional
/// - `dt_bias`: bias added to dt before softplus, shape (dim,) — optional
/// - `ssm_state`: mutable state, shape (batch, dim, d_state)
///
/// # Returns
/// Output for this step, shape (batch, dim)
pub fn selective_state_update(
    x: &[f32],
    batch: usize,
    dim: usize,
    dt: &[f32],
    a: &[f32],
    d_state: usize,
    b: &[f32],
    c: &[f32],
    d: Option<&[f32]>,
    z: Option<&[f32]>,
    dt_bias: Option<&[f32]>,
    dt_softplus: bool,
    ssm_state: &mut [f32],
) -> Result<Vec<f32>, String> {
    // Shape validation
    let bd = batch * dim;
    if x.len() != bd { return Err(format!("x shape mismatch: expected {} (batch*dim), got {}", bd, x.len())); }
    if dt.len() != bd { return Err(format!("dt shape mismatch: expected {}, got {}", bd, dt.len())); }
    if a.len() != dim * d_state { return Err(format!("a shape mismatch: expected {}, got {}", dim * d_state, a.len())); }
    if b.len() != batch * d_state { return Err(format!("b shape mismatch: expected {}, got {}", batch * d_state, b.len())); }
    if c.len() != batch * d_state { return Err(format!("c shape mismatch: expected {}, got {}", batch * d_state, c.len())); }
    if ssm_state.len() != batch * dim * d_state { return Err(format!("ssm_state shape mismatch: expected {}, got {}", batch * dim * d_state, ssm_state.len())); }
    if let Some(d_skip) = d { if d_skip.len() != dim { return Err(format!("d shape mismatch: expected {}, got {}", dim, d_skip.len())); } }
    if let Some(z_data) = z { if z_data.len() != bd { return Err(format!("z shape mismatch: expected {}, got {}", bd, z_data.len())); } }
    if let Some(bias) = dt_bias { if bias.len() != dim { return Err(format!("dt_bias shape mismatch: expected {}, got {}", dim, bias.len())); } }

    let mut output = vec![0.0f32; batch * dim];

    for b_idx in 0..batch {
        for d_idx in 0..dim {
            let bd_idx = b_idx * dim + d_idx;
            let mut dt_val = dt[bd_idx];

            if let Some(bias) = dt_bias {
                dt_val += bias[d_idx];
            }
            if dt_softplus {
                dt_val = softplus(dt_val);
            }

            let x_val = x[bd_idx];
            let mut y = 0.0f32;

            for n in 0..d_state {
                let a_val = a[d_idx * d_state + n];
                let da = (dt_val * a_val).exp();
                let b_val = b[b_idx * d_state + n];
                let db_x = dt_val * b_val * x_val;

                let s_idx = b_idx * dim * d_state + d_idx * d_state + n;
                ssm_state[s_idx] = ssm_state[s_idx] * da + db_x;

                let c_val = c[b_idx * d_state + n];
                y += ssm_state[s_idx] * c_val;
            }

            // Skip connection
            if let Some(d_skip) = d {
                y += d_skip[d_idx] * x_val;
            }

            // Gating
            if let Some(z_data) = z {
                y *= silu(z_data[bd_idx]);
            }

            output[bd_idx] = y;
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selective_scan_shape() {
        let batch = 2;
        let dim = 4;
        let seq_len = 8;
        let d_state = 3;

        let u = vec![0.1f32; batch * dim * seq_len];
        let delta = vec![0.05f32; batch * dim * seq_len];
        let a = vec![-1.0f32; dim * d_state]; // negative A
        let b = vec![0.1f32; batch * d_state * seq_len];
        let c = vec![0.1f32; batch * d_state * seq_len];
        let d_skip = vec![1.0f32; dim];

        let result = selective_scan_ref(
            &u, batch, dim, seq_len,
            &delta, &a, d_state, &b, &c,
            Some(&d_skip), None, None, false,
        ).unwrap();

        assert_eq!(result.output.len(), batch * dim * seq_len);
        assert_eq!(result.last_state.len(), batch * dim * d_state);
    }

    #[test]
    fn test_selective_scan_skip_connection() {
        // With A=0, B=0, the state stays zero, so output = D * u
        let batch = 1;
        let dim = 2;
        let seq_len = 3;
        let d_state = 2;

        let u = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // (1, 2, 3)
        let delta = vec![0.0f32; batch * dim * seq_len];
        let a = vec![0.0f32; dim * d_state];
        let b = vec![0.0f32; batch * d_state * seq_len];
        let c = vec![0.0f32; batch * d_state * seq_len];
        let d_skip = vec![2.0, 3.0]; // D = [2, 3]

        let result = selective_scan_ref(
            &u, batch, dim, seq_len,
            &delta, &a, d_state, &b, &c,
            Some(&d_skip), None, None, false,
        ).unwrap();

        // output[d=0] = 2 * [1, 2, 3] = [2, 4, 6]
        assert!((result.output[0] - 2.0).abs() < 1e-5);
        assert!((result.output[1] - 4.0).abs() < 1e-5);
        assert!((result.output[2] - 6.0).abs() < 1e-5);
        // output[d=1] = 3 * [4, 5, 6] = [12, 15, 18]
        assert!((result.output[3] - 12.0).abs() < 1e-5);
        assert!((result.output[4] - 15.0).abs() < 1e-5);
        assert!((result.output[5] - 18.0).abs() < 1e-5);
    }

    #[test]
    fn test_selective_scan_delta_softplus() {
        let batch = 1;
        let dim = 1;
        let seq_len = 1;
        let d_state = 1;

        let u = vec![1.0];
        let delta = vec![0.0]; // softplus(0) = ln(2) ≈ 0.693
        let a = vec![-1.0];
        let b = vec![1.0];
        let c = vec![1.0];

        let result = selective_scan_ref(
            &u, batch, dim, seq_len,
            &delta, &a, d_state, &b, &c,
            None, None, None, true,
        ).unwrap();

        let dt = softplus(0.0);
        let expected_state = dt * 1.0 * 1.0; // da * 0 + dt * b * u
        let expected_y = expected_state * 1.0; // state * c
        assert!((result.output[0] - expected_y).abs() < 1e-5);
    }

    #[test]
    fn test_selective_state_update() {
        let batch = 1;
        let dim = 2;
        let d_state = 2;

        let x = vec![1.0, 2.0];
        let dt = vec![0.1, 0.2];
        let a = vec![-1.0, -1.0, -1.0, -1.0]; // (dim, d_state)
        let b = vec![1.0, 0.5]; // (batch, d_state)
        let c = vec![1.0, 1.0]; // (batch, d_state)
        let d_skip = vec![1.0, 1.0];
        let mut ssm_state = vec![0.0f32; batch * dim * d_state];

        let out = selective_state_update(
            &x, batch, dim, &dt, &a, d_state, &b, &c,
            Some(&d_skip), None, None, false, &mut ssm_state,
        ).unwrap();

        assert_eq!(out.len(), batch * dim);
        // State should be non-zero now
        assert!(ssm_state.iter().any(|&v| v != 0.0));
    }
}
