//! Mamba v1 block — the core selective SSM module.
//!
//! Implements the Mamba architecture block from "Mamba: Linear-Time Sequence Modeling
//! with Selective State Spaces" (Gu & Dao, 2023).
//!
//! Reference: `mamba_ssm/modules/mamba_simple.py`

use rand::Rng;

use kore_core::Tensor;
use kore_nn::module::Module;

use crate::causal_conv1d;
use crate::selective_scan;

/// Mamba v1 block.
///
/// Architecture: in_proj → split(x,z) → causal_conv1d → x_proj → split(dt,B,C) →
/// dt_proj → selective_scan(x, dt, A, B, C, D, z) → out_proj
pub struct Mamba {
    // Dimensions
    pub d_model: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub expand: usize,
    pub d_inner: usize,
    pub dt_rank: usize,
    pub layer_idx: Option<usize>,

    // Parameters (stored as raw f32 vecs for direct manipulation)
    /// in_proj weight: (2 * d_inner, d_model)
    pub in_proj_weight: Vec<f32>,
    /// in_proj bias: optional (2 * d_inner,)
    pub in_proj_bias: Option<Vec<f32>>,

    /// conv1d weight: (d_inner, kernel_size) — depthwise
    pub conv1d_weight: Vec<f32>,
    /// conv1d bias: (d_inner,)
    pub conv1d_bias: Option<Vec<f32>>,

    /// x_proj weight: (dt_rank + 2*d_state, d_inner) — no bias
    pub x_proj_weight: Vec<f32>,

    /// dt_proj weight: (d_inner, dt_rank)
    pub dt_proj_weight: Vec<f32>,
    /// dt_proj bias: (d_inner,)
    pub dt_proj_bias: Vec<f32>,

    /// A_log: (d_inner, d_state) — log of the state matrix
    pub a_log: Vec<f32>,

    /// D "skip" parameter: (d_inner,)
    pub d_skip: Vec<f32>,

    /// out_proj weight: (d_model, d_inner)
    pub out_proj_weight: Vec<f32>,
    /// out_proj bias: optional (d_model,)
    pub out_proj_bias: Option<Vec<f32>>,

    training: bool,
}

impl Mamba {
    /// Create a new Mamba block with initialized parameters.
    ///
    /// # Arguments
    /// - `d_model`: model dimension
    /// - `d_state`: SSM state expansion factor (default 16)
    /// - `d_conv`: local convolution width (default 4)
    /// - `expand`: block expansion factor (default 2)
    pub fn new(d_model: usize, d_state: usize, d_conv: usize, expand: usize) -> Self {
        Self::new_full(d_model, d_state, d_conv, expand, None, true, false, None)
    }

    /// Create with full configuration options.
    #[allow(clippy::too_many_arguments)]
    pub fn new_full(
        d_model: usize,
        d_state: usize,
        d_conv: usize,
        expand: usize,
        dt_rank: Option<usize>,
        conv_bias: bool,
        bias: bool,
        layer_idx: Option<usize>,
    ) -> Self {
        let d_inner = expand * d_model;
        let dt_rank = dt_rank.unwrap_or_else(|| (d_model + 15) / 16); // ceil(d_model / 16)

        let mut rng = rand::thread_rng();

        // in_proj: Linear(d_model, 2 * d_inner)
        let in_proj_limit = (6.0 / (d_model + 2 * d_inner) as f32).sqrt();
        let in_proj_weight: Vec<f32> = (0..2 * d_inner * d_model)
            .map(|_| rng.gen_range(-in_proj_limit..in_proj_limit))
            .collect();
        let in_proj_bias = if bias {
            Some(vec![0.0f32; 2 * d_inner])
        } else {
            None
        };

        // conv1d: depthwise, kernel_size weights
        let conv_limit = (6.0 / d_conv as f32).sqrt();
        let conv1d_weight: Vec<f32> = (0..d_inner * d_conv)
            .map(|_| rng.gen_range(-conv_limit..conv_limit))
            .collect();
        let conv1d_bias = if conv_bias {
            Some(vec![0.0f32; d_inner])
        } else {
            None
        };

        // x_proj: Linear(d_inner, dt_rank + 2*d_state, bias=False)
        let x_proj_out = dt_rank + 2 * d_state;
        let x_proj_limit = (6.0 / (d_inner + x_proj_out) as f32).sqrt();
        let x_proj_weight: Vec<f32> = (0..x_proj_out * d_inner)
            .map(|_| rng.gen_range(-x_proj_limit..x_proj_limit))
            .collect();

        // dt_proj: Linear(dt_rank, d_inner, bias=True)
        let dt_init_std = (dt_rank as f32).powf(-0.5);
        let dt_proj_weight: Vec<f32> = (0..d_inner * dt_rank)
            .map(|_| rng.gen_range(-dt_init_std..dt_init_std))
            .collect();

        // dt_proj bias: initialized so that softplus(bias) is in [dt_min, dt_max]
        let dt_min: f32 = 0.001;
        let dt_max: f32 = 0.1;
        let dt_init_floor: f32 = 1e-4;
        let dt_proj_bias: Vec<f32> = (0..d_inner)
            .map(|_| {
                let dt = (rng.gen::<f32>() * (dt_max.ln() - dt_min.ln()) + dt_min.ln())
                    .exp()
                    .max(dt_init_floor);
                // Inverse softplus: x + log(-expm1(-x)), clamped for numerical safety
                let neg_expm1 = (-(-dt).exp_m1()).max(1e-20);
                let inv_sp = dt + neg_expm1.ln();
                if inv_sp.is_nan() || inv_sp.is_infinite() {
                    debug_assert!(false, "inverse softplus produced NaN/Inf for dt={}", dt);
                    dt
                } else {
                    inv_sp
                }
            })
            .collect();

        // A_log: S4D real initialization — log(arange(1, d_state+1)) repeated d_inner times
        let a_log: Vec<f32> = (0..d_inner)
            .flat_map(|_| {
                (1..=d_state).map(|n| (n as f32).ln())
            })
            .collect();

        // D: skip parameter, initialized to ones
        let d_skip = vec![1.0f32; d_inner];

        // out_proj: Linear(d_inner, d_model)
        let out_proj_limit = (6.0 / (d_inner + d_model) as f32).sqrt();
        let out_proj_weight: Vec<f32> = (0..d_model * d_inner)
            .map(|_| rng.gen_range(-out_proj_limit..out_proj_limit))
            .collect();
        let out_proj_bias = if bias {
            Some(vec![0.0f32; d_model])
        } else {
            None
        };

        Self {
            d_model,
            d_state,
            d_conv,
            expand,
            d_inner,
            dt_rank,
            layer_idx,
            in_proj_weight,
            in_proj_bias,
            conv1d_weight,
            conv1d_bias,
            x_proj_weight,
            dt_proj_weight,
            dt_proj_bias,
            a_log,
            d_skip,
            out_proj_weight,
            out_proj_bias,
            training: true,
        }
    }

    /// Training/prefill forward pass.
    ///
    /// `input`: shape (batch, seq_len, d_model)
    /// Returns: shape (batch, seq_len, d_model)
    pub fn forward_train(&self, input: &[f32], batch: usize, seq_len: usize) -> Vec<f32> {
        let d = self.d_model;
        let di = self.d_inner;

        // 1) in_proj: (B, L, d_model) @ W^T -> (B, L, 2*d_inner)
        //    W is (2*d_inner, d_model), so y = x @ W^T
        let mut xz = vec![0.0f32; batch * seq_len * 2 * di];
        for b in 0..batch {
            for l in 0..seq_len {
                let x_off = b * seq_len * d + l * d;
                let out_off = b * seq_len * 2 * di + l * 2 * di;
                for o in 0..(2 * di) {
                    let mut acc = 0.0f32;
                    for k in 0..d {
                        acc += input[x_off + k] * self.in_proj_weight[o * d + k];
                    }
                    if let Some(ref bias) = self.in_proj_bias {
                        acc += bias[o];
                    }
                    xz[out_off + o] = acc;
                }
            }
        }

        // 2) Split into x and z, each (B, L, d_inner), then rearrange to (B, d_inner, L)
        let mut x_bdl = vec![0.0f32; batch * di * seq_len];
        let mut z_bdl = vec![0.0f32; batch * di * seq_len];
        for b in 0..batch {
            for l in 0..seq_len {
                let src = b * seq_len * 2 * di + l * 2 * di;
                for i in 0..di {
                    x_bdl[b * di * seq_len + i * seq_len + l] = xz[src + i];
                    z_bdl[b * di * seq_len + i * seq_len + l] = xz[src + di + i];
                }
            }
        }

        // 3) Causal conv1d on x: (B, d_inner, L) -> (B, d_inner, L) with SiLU
        let x_conv = causal_conv1d::causal_conv1d_fn(
            &x_bdl,
            batch,
            di,
            seq_len,
            &self.conv1d_weight,
            self.d_conv,
            self.conv1d_bias.as_deref(),
            true, // SiLU activation
        );

        // 4) x_proj: rearrange x to (B*L, d_inner) then matmul with x_proj_weight^T
        //    x_proj_weight is (dt_rank + 2*d_state, d_inner)
        //    output: (B*L, dt_rank + 2*d_state)
        let x_proj_out_dim = self.dt_rank + 2 * self.d_state;
        let mut x_dbl = vec![0.0f32; batch * seq_len * x_proj_out_dim];
        for b in 0..batch {
            for l in 0..seq_len {
                let bl = b * seq_len + l;
                for o in 0..x_proj_out_dim {
                    let mut acc = 0.0f32;
                    for k in 0..di {
                        // x_conv is (B, d_inner, L), so x_conv[b*di*L + k*L + l]
                        acc += x_conv[b * di * seq_len + k * seq_len + l]
                            * self.x_proj_weight[o * di + k];
                    }
                    x_dbl[bl * x_proj_out_dim + o] = acc;
                }
            }
        }

        // 5) Split x_dbl into dt_raw, B_raw, C_raw
        //    dt_raw: (B*L, dt_rank)
        //    B_raw: (B*L, d_state)
        //    C_raw: (B*L, d_state)

        // 6) dt_proj: dt_raw @ dt_proj_weight^T -> (B*L, d_inner)
        //    dt_proj_weight is (d_inner, dt_rank)
        //    Then rearrange to (B, d_inner, L)
        let mut delta = vec![0.0f32; batch * di * seq_len];
        for b in 0..batch {
            for l in 0..seq_len {
                let bl = b * seq_len + l;
                for o in 0..di {
                    let mut acc = 0.0f32;
                    for k in 0..self.dt_rank {
                        acc += x_dbl[bl * x_proj_out_dim + k]
                            * self.dt_proj_weight[o * self.dt_rank + k];
                    }
                    delta[b * di * seq_len + o * seq_len + l] = acc;
                }
            }
        }

        // 7) Rearrange B to (B, d_state, L) and C to (B, d_state, L)
        let mut b_bnl = vec![0.0f32; batch * self.d_state * seq_len];
        let mut c_bnl = vec![0.0f32; batch * self.d_state * seq_len];
        for b in 0..batch {
            for l in 0..seq_len {
                let bl = b * seq_len + l;
                for n in 0..self.d_state {
                    b_bnl[b * self.d_state * seq_len + n * seq_len + l] =
                        x_dbl[bl * x_proj_out_dim + self.dt_rank + n];
                    c_bnl[b * self.d_state * seq_len + n * seq_len + l] =
                        x_dbl[bl * x_proj_out_dim + self.dt_rank + self.d_state + n];
                }
            }
        }

        // 8) A = -exp(A_log)
        let a_neg: Vec<f32> = self.a_log.iter().map(|&v| -v.exp()).collect();

        // 9) Selective scan
        let scan_result = selective_scan::selective_scan_ref(
            &x_conv,
            batch,
            di,
            seq_len,
            &delta,
            &a_neg,
            self.d_state,
            &b_bnl,
            &c_bnl,
            Some(&self.d_skip),
            Some(&z_bdl),
            Some(&self.dt_proj_bias),
            true, // delta_softplus
        ).expect("selective_scan: shape mismatch in Mamba forward");

        // 10) Rearrange scan output from (B, d_inner, L) to (B, L, d_inner)
        //     Then out_proj: (B, L, d_inner) @ out_proj_weight^T -> (B, L, d_model)
        //     out_proj_weight is (d_model, d_inner)
        let mut output = vec![0.0f32; batch * seq_len * d];
        for b in 0..batch {
            for l in 0..seq_len {
                for o in 0..d {
                    let mut acc = 0.0f32;
                    for k in 0..di {
                        // scan output is (B, d_inner, L)
                        let y_val = scan_result.output[b * di * seq_len + k * seq_len + l];
                        acc += y_val * self.out_proj_weight[o * di + k];
                    }
                    if let Some(ref bias) = self.out_proj_bias {
                        acc += bias[o];
                    }
                    output[b * seq_len * d + l * d + o] = acc;
                }
            }
        }

        output
    }

    /// Single-step decode forward using cached conv_state and ssm_state.
    ///
    /// `input`: shape (batch, 1, d_model)
    /// `conv_state`: mutable, shape (batch, d_inner, d_conv)
    /// `ssm_state`: mutable, shape (batch, d_inner, d_state)
    ///
    /// Returns: output shape (batch, 1, d_model)
    pub fn forward_step(
        &self,
        input: &[f32],
        batch: usize,
        conv_state: &mut [f32],
        ssm_state: &mut [f32],
    ) -> Vec<f32> {
        let d = self.d_model;
        let di = self.d_inner;

        // 1) in_proj: (B, d_model) -> (B, 2*d_inner)
        let mut xz = vec![0.0f32; batch * 2 * di];
        for b in 0..batch {
            for o in 0..(2 * di) {
                let mut acc = 0.0f32;
                for k in 0..d {
                    acc += input[b * d + k] * self.in_proj_weight[o * d + k];
                }
                if let Some(ref bias) = self.in_proj_bias {
                    acc += bias[o];
                }
                xz[b * 2 * di + o] = acc;
            }
        }

        // 2) Split x, z: each (B, d_inner)
        let mut x_bd = vec![0.0f32; batch * di];
        let mut z_bd = vec![0.0f32; batch * di];
        for b in 0..batch {
            for i in 0..di {
                x_bd[b * di + i] = xz[b * 2 * di + i];
                z_bd[b * di + i] = xz[b * 2 * di + di + i];
            }
        }

        // 3) Conv step: update conv_state and compute output with SiLU
        let x_conv = causal_conv1d::causal_conv1d_update(
            &x_bd,
            batch,
            di,
            conv_state,
            self.d_conv,
            &self.conv1d_weight,
            self.conv1d_bias.as_deref(),
            true,
        );

        // 4) x_proj: (B, d_inner) @ x_proj_weight^T -> (B, dt_rank + 2*d_state)
        let x_proj_out_dim = self.dt_rank + 2 * self.d_state;
        let mut x_db = vec![0.0f32; batch * x_proj_out_dim];
        for b in 0..batch {
            for o in 0..x_proj_out_dim {
                let mut acc = 0.0f32;
                for k in 0..di {
                    acc += x_conv[b * di + k] * self.x_proj_weight[o * di + k];
                }
                x_db[b * x_proj_out_dim + o] = acc;
            }
        }

        // 5) dt = x_db[:, :dt_rank] @ dt_proj_weight^T -> (B, d_inner)
        let mut dt = vec![0.0f32; batch * di];
        for b in 0..batch {
            for o in 0..di {
                let mut acc = 0.0f32;
                for k in 0..self.dt_rank {
                    acc += x_db[b * x_proj_out_dim + k] * self.dt_proj_weight[o * self.dt_rank + k];
                }
                dt[b * di + o] = acc;
            }
        }

        // 6) Extract B, C: each (B, d_state)
        let mut b_step = vec![0.0f32; batch * self.d_state];
        let mut c_step = vec![0.0f32; batch * self.d_state];
        for b in 0..batch {
            for n in 0..self.d_state {
                b_step[b * self.d_state + n] = x_db[b * x_proj_out_dim + self.dt_rank + n];
                c_step[b * self.d_state + n] =
                    x_db[b * x_proj_out_dim + self.dt_rank + self.d_state + n];
            }
        }

        // 7) A = -exp(A_log)
        let a_neg: Vec<f32> = self.a_log.iter().map(|&v| -v.exp()).collect();

        // 8) SSM step update
        let y = selective_scan::selective_state_update(
            &x_conv,
            batch,
            di,
            &dt,
            &a_neg,
            self.d_state,
            &b_step,
            &c_step,
            Some(&self.d_skip),
            Some(&z_bd),
            Some(&self.dt_proj_bias),
            true,
            ssm_state,
        ).expect("selective_state_update: shape mismatch in Mamba step");

        // 9) out_proj: (B, d_inner) @ out_proj_weight^T -> (B, d_model)
        let mut output = vec![0.0f32; batch * d];
        for b in 0..batch {
            for o in 0..d {
                let mut acc = 0.0f32;
                for k in 0..di {
                    acc += y[b * di + k] * self.out_proj_weight[o * di + k];
                }
                if let Some(ref bias) = self.out_proj_bias {
                    acc += bias[o];
                }
                output[b * d + o] = acc;
            }
        }

        output
    }

    /// Get the conv_state size for this block: batch * d_inner * d_conv
    pub fn conv_state_size(&self, batch: usize) -> usize {
        batch * self.d_inner * self.d_conv
    }

    /// Get the ssm_state size for this block: batch * d_inner * d_state
    pub fn ssm_state_size(&self, batch: usize) -> usize {
        batch * self.d_inner * self.d_state
    }

    /// Total parameter count.
    pub fn param_count(&self) -> usize {
        let in_proj = 2 * self.d_inner * self.d_model
            + if self.in_proj_bias.is_some() { 2 * self.d_inner } else { 0 };
        let conv = self.d_inner * self.d_conv
            + if self.conv1d_bias.is_some() { self.d_inner } else { 0 };
        let x_proj = (self.dt_rank + 2 * self.d_state) * self.d_inner;
        let dt_proj = self.d_inner * self.dt_rank + self.d_inner; // weight + bias
        let a_log = self.d_inner * self.d_state;
        let d_skip = self.d_inner;
        let out_proj = self.d_model * self.d_inner
            + if self.out_proj_bias.is_some() { self.d_model } else { 0 };
        in_proj + conv + x_proj + dt_proj + a_log + d_skip + out_proj
    }
}

impl Module for Mamba {
    fn forward(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        let dims = input.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(kore_core::KoreError::ShapeMismatch {
                expected: vec![0, 0, 0],
                got: dims,
            });
        }
        let batch = dims[0];
        let seq_len = dims[1];
        let d = dims[2];
        if d != self.d_model {
            return Err(kore_core::KoreError::ShapeMismatch {
                expected: vec![batch, seq_len, self.d_model],
                got: dims,
            });
        }

        let data = input.contiguous();
        let x = data.as_f32_slice().ok_or_else(|| {
            kore_core::KoreError::UnsupportedDType(input.dtype())
        })?;

        let output = self.forward_train(x, batch, seq_len);
        Ok(Tensor::from_f32(&output, &[batch, seq_len, self.d_model]))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        // Raw parameter storage — return empty since we store as Vec<f32>
        // For full autograd support, these would be Tensor fields
        vec![]
    }

    fn named_parameters(&self) -> Vec<(&str, &Tensor)> {
        vec![]
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba_creation() {
        let mamba = Mamba::new(64, 16, 4, 2);
        assert_eq!(mamba.d_model, 64);
        assert_eq!(mamba.d_inner, 128);
        assert_eq!(mamba.dt_rank, 4); // ceil(64/16)
        assert_eq!(mamba.d_state, 16);
    }

    #[test]
    fn test_mamba_forward_shape() {
        let mamba = Mamba::new(16, 8, 4, 2);
        let batch = 2;
        let seq_len = 8;
        let input = vec![0.1f32; batch * seq_len * 16];
        let output = mamba.forward_train(&input, batch, seq_len);
        assert_eq!(output.len(), batch * seq_len * 16);
    }

    #[test]
    fn test_mamba_step_shape() {
        let mamba = Mamba::new(16, 8, 4, 2);
        let batch = 2;
        let di = mamba.d_inner;
        let mut conv_state = vec![0.0f32; mamba.conv_state_size(batch)];
        let mut ssm_state = vec![0.0f32; mamba.ssm_state_size(batch)];

        let input = vec![0.1f32; batch * 16];
        let output = mamba.forward_step(&input, batch, &mut conv_state, &mut ssm_state);
        assert_eq!(output.len(), batch * 16);
    }

    #[test]
    fn test_mamba_param_count() {
        let mamba = Mamba::new(64, 16, 4, 2);
        let params = mamba.param_count();
        // ~3 * expand * d_model^2 ≈ 3 * 2 * 64^2 = 24576
        assert!(params > 20000);
        assert!(params < 50000);
    }

    #[test]
    fn test_mamba_module_trait() {
        let mamba = Mamba::new(16, 8, 4, 2);
        let input = Tensor::from_f32(&vec![0.1f32; 2 * 4 * 16], &[2, 4, 16]);
        let output = mamba.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 4, 16]);
    }
}
