//! Mamba-3 block — trapezoidal SSD with data-dependent RoPE and MIMO.
//!
//! Implements the Mamba-3 architecture from "Mamba-3: Improved Sequence Modeling
//! using State Space Principles" (ICLR 2026).
//!
//! Key differences from Mamba-2:
//! - No causal conv1d layer (trapezoidal discretization + B/C biases replace it)
//! - Data-dependent RoPE on B, C (equivalent to complex-valued A)
//! - MIMO rank-r formulation for higher arithmetic intensity
//!
//! Reference: `mamba_ssm/modules/mamba2.py` (adapted)

use rand::Rng;

use kore_core::Tensor;
use kore_nn::module::Module;

use crate::norm::RMSNormGated;
use crate::ssd3;

/// SiLU activation
#[inline]
fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Mamba-3 block.
pub struct Mamba3 {
    // Dimensions
    pub d_model: usize,
    pub d_state: usize,
    pub expand: usize,
    pub d_inner: usize,
    pub d_ssm: usize,
    pub headdim: usize,
    pub nheads: usize,
    pub ngroups: usize,
    pub layer_idx: Option<usize>,

    // Mamba-3 specific
    pub use_rope: bool,
    pub trapezoidal_alpha: f32,

    // Parameters
    /// in_proj weight: (d_in_proj, d_model)
    /// Projects to: [z(d_inner), x_ssm(d_ssm), B(ngroups*d_state), C(ngroups*d_state), dt(nheads)]
    pub in_proj_weight: Vec<f32>,
    pub in_proj_bias: Option<Vec<f32>>,

    /// Channel-specific biases on B and C (new in Mamba-3)
    /// B bias: (ngroups, d_state)
    pub b_bias: Vec<f32>,
    /// C bias: (ngroups, d_state)
    pub c_bias: Vec<f32>,

    /// dt_bias: (nheads,)
    pub dt_bias: Vec<f32>,

    /// A_log_real: (nheads,) — log of real part of state matrix
    pub a_log_real: Vec<f32>,
    /// A_imag: (nheads,) — imaginary part of A (for RoPE angle)
    pub a_imag: Vec<f32>,

    /// D skip parameter: (nheads,)
    pub d_skip: Vec<f32>,

    /// RMSNormGated (if enabled)
    pub norm: Option<RMSNormGated>,
    pub use_rmsnorm: bool,
    pub norm_before_gate: bool,

    /// out_proj weight: (d_model, d_inner)
    pub out_proj_weight: Vec<f32>,
    pub out_proj_bias: Option<Vec<f32>>,

    training: bool,
}

impl Mamba3 {
    /// Create a new Mamba-3 block with default options.
    pub fn new(
        d_model: usize,
        d_state: usize,
        expand: usize,
        headdim: usize,
    ) -> Self {
        Self::new_full(
            d_model, d_state, expand, headdim,
            None, 1, true, true, false, false, false,
            true, 0.5, None,
        )
    }

    /// Create with full configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn new_full(
        d_model: usize,
        d_state: usize,
        expand: usize,
        headdim: usize,
        d_ssm: Option<usize>,
        ngroups: usize,
        use_rope: bool,
        rmsnorm: bool,
        norm_before_gate: bool,
        bias: bool,
        _conv_bias: bool,
        dt_softplus: bool,
        trapezoidal_alpha: f32,
        layer_idx: Option<usize>,
    ) -> Self {
        let d_inner = expand * d_model;
        let d_ssm = d_ssm.unwrap_or(d_inner);
        assert!(d_ssm % headdim == 0, "d_ssm must be divisible by headdim");
        let nheads = d_ssm / headdim;

        let mut rng = rand::thread_rng();

        // in_proj: d_model -> d_in_proj
        // Mamba-3: no conv1d, so xBC goes directly through projection
        // Layout: [z(d_inner), x_ssm(d_ssm), B(ngroups*d_state), C(ngroups*d_state), dt(nheads)]
        let d_in_proj = d_inner + d_ssm + 2 * ngroups * d_state + nheads;
        let in_limit = (6.0 / (d_model + d_in_proj) as f32).sqrt();
        let in_proj_weight: Vec<f32> = (0..d_in_proj * d_model)
            .map(|_| rng.gen_range(-in_limit..in_limit))
            .collect();
        let in_proj_bias = if bias {
            Some(vec![0.0f32; d_in_proj])
        } else {
            None
        };

        // B and C channel biases (new in Mamba-3)
        let b_bias = vec![0.0f32; ngroups * d_state];
        let c_bias = vec![0.0f32; ngroups * d_state];

        // dt_bias
        let dt_min: f32 = 0.001;
        let dt_max: f32 = 0.1;
        let dt_init_floor: f32 = 1e-4;
        let dt_bias: Vec<f32> = (0..nheads)
            .map(|_| {
                let dt = (rng.gen::<f32>() * (dt_max.ln() - dt_min.ln()) + dt_min.ln())
                    .exp()
                    .max(dt_init_floor);
                if dt_softplus {
                    dt + (-(-dt).exp_m1()).ln()
                } else {
                    dt
                }
            })
            .collect();

        // A_log_real: uniform in [1, 16], then log (real part, negative on use)
        let a_log_real: Vec<f32> = (0..nheads)
            .map(|_| rng.gen_range(1.0f32..16.0).ln())
            .collect();

        // A_imag: imaginary part for RoPE, initialized near pi for oscillation
        let a_imag: Vec<f32> = if use_rope {
            (0..nheads)
                .map(|i| {
                    // Spread frequencies across heads
                    std::f32::consts::PI * (1.0 + i as f32) / nheads as f32
                })
                .collect()
        } else {
            vec![0.0f32; nheads]
        };

        // D: skip parameter
        let d_skip = vec![1.0f32; nheads];

        // RMSNormGated
        let norm = if rmsnorm {
            Some(RMSNormGated::new(d_ssm, 1e-5, norm_before_gate, d_ssm / ngroups))
        } else {
            None
        };

        // out_proj
        let out_limit = (6.0 / (d_inner + d_model) as f32).sqrt();
        let out_proj_weight: Vec<f32> = (0..d_model * d_inner)
            .map(|_| rng.gen_range(-out_limit..out_limit))
            .collect();
        let out_proj_bias = if bias {
            Some(vec![0.0f32; d_model])
        } else {
            None
        };

        Self {
            d_model,
            d_state,
            expand,
            d_inner,
            d_ssm,
            headdim,
            nheads,
            ngroups,
            layer_idx,
            use_rope,
            trapezoidal_alpha,
            in_proj_weight,
            in_proj_bias,
            b_bias,
            c_bias,
            dt_bias,
            a_log_real,
            a_imag,
            d_skip,
            norm,
            use_rmsnorm: rmsnorm,
            norm_before_gate,
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
        let dm = self.d_model;
        let di = self.d_inner;
        let ngs = self.ngroups * self.d_state;
        let d_in_proj = di + self.d_ssm + 2 * ngs + self.nheads;

        // 1) in_proj: (B, L, d_model) -> (B, L, d_in_proj)
        let mut proj = vec![0.0f32; batch * seq_len * d_in_proj];
        for b in 0..batch {
            for l in 0..seq_len {
                let x_off = b * seq_len * dm + l * dm;
                let o_off = b * seq_len * d_in_proj + l * d_in_proj;
                for o in 0..d_in_proj {
                    let mut acc = 0.0f32;
                    for k in 0..dm {
                        acc += input[x_off + k] * self.in_proj_weight[o * dm + k];
                    }
                    if let Some(ref bias) = self.in_proj_bias {
                        acc += bias[o];
                    }
                    proj[o_off + o] = acc;
                }
            }
        }

        // 2) Split projection
        // Layout: [z(d_inner), x_ssm(d_ssm), B(ngs), C(ngs), dt(nheads)]
        // Compute MLP portion size
        let d_mlp = di.saturating_sub(self.d_ssm);
        let z_start = 0;
        let x_start = d_mlp; // MLP z portion
        let z_ssm_start = 2 * d_mlp; // SSM z portion (if d_mlp > 0)
        let x_ssm_start = z_ssm_start + self.d_ssm;
        let b_start = x_ssm_start + self.d_ssm;
        let c_start = b_start + ngs;
        let dt_start = c_start + ngs;

        // 3) Extract x_ssm: (B, L, d_ssm) -> (B, L, nheads, headdim)
        let mut x_scan = vec![0.0f32; batch * seq_len * self.nheads * self.headdim];
        for b in 0..batch {
            for l in 0..seq_len {
                let src = b * seq_len * d_in_proj + l * d_in_proj + x_ssm_start;
                for i in 0..self.d_ssm {
                    let h = i / self.headdim;
                    let p = i % self.headdim;
                    x_scan[b * seq_len * self.nheads * self.headdim
                        + l * self.nheads * self.headdim
                        + h * self.headdim + p] = proj[src + i];
                }
            }
        }

        // 4) Extract B: (B, L, ngroups, d_state)
        let mut b_scan = vec![0.0f32; batch * seq_len * ngs];
        for b in 0..batch {
            for l in 0..seq_len {
                let src = b * seq_len * d_in_proj + l * d_in_proj + b_start;
                let dst = b * seq_len * ngs + l * ngs;
                for i in 0..ngs {
                    b_scan[dst + i] = proj[src + i];
                }
            }
        }

        // 5) Extract C: (B, L, ngroups, d_state)
        let mut c_scan = vec![0.0f32; batch * seq_len * ngs];
        for b in 0..batch {
            for l in 0..seq_len {
                let src = b * seq_len * d_in_proj + l * d_in_proj + c_start;
                let dst = b * seq_len * ngs + l * ngs;
                for i in 0..ngs {
                    c_scan[dst + i] = proj[src + i];
                }
            }
        }

        // 6) Extract dt: (B, L, nheads)
        let mut dt_scan = vec![0.0f32; batch * seq_len * self.nheads];
        for b in 0..batch {
            for l in 0..seq_len {
                let src = b * seq_len * d_in_proj + l * d_in_proj + dt_start;
                let dst = b * seq_len * self.nheads + l * self.nheads;
                for h in 0..self.nheads {
                    dt_scan[dst + h] = proj[src + h];
                }
            }
        }

        // 7) A = -exp(A_log_real)
        let a_real: Vec<f32> = self.a_log_real.iter().map(|&v| -v.exp()).collect();

        // 8) z for gating (if no rmsnorm)
        let z_for_scan = if !self.use_rmsnorm {
            let mut z_scan = vec![0.0f32; batch * seq_len * self.nheads * self.headdim];
            for b in 0..batch {
                for l in 0..seq_len {
                    let src = b * seq_len * d_in_proj + l * d_in_proj + z_ssm_start;
                    for i in 0..self.d_ssm {
                        let h = i / self.headdim;
                        let p = i % self.headdim;
                        z_scan[b * seq_len * self.nheads * self.headdim
                            + l * self.nheads * self.headdim
                            + h * self.headdim + p] = proj[src + i];
                    }
                }
            }
            Some(z_scan)
        } else {
            None
        };

        // 9) Run Mamba-3 SSD scan
        let scan_result = ssd3::mamba3_scan_combined(
            &x_scan, batch, seq_len, self.nheads, self.headdim,
            &dt_scan, &a_real, &self.a_imag,
            &b_scan, self.ngroups, self.d_state, &c_scan,
            Some(&self.b_bias), Some(&self.c_bias),
            Some(&self.d_skip),
            z_for_scan.as_deref(),
            Some(&self.dt_bias),
            true, // dt_softplus
            self.trapezoidal_alpha,
            self.use_rope,
        );

        // 10) Rearrange scan output: (B, L, nheads, headdim) -> (B, L, d_ssm)
        let mut y_flat = vec![0.0f32; batch * seq_len * self.d_ssm];
        for b in 0..batch {
            for l in 0..seq_len {
                for i in 0..self.d_ssm {
                    let h = i / self.headdim;
                    let p = i % self.headdim;
                    y_flat[b * seq_len * self.d_ssm + l * self.d_ssm + i] =
                        scan_result.output[b * seq_len * self.nheads * self.headdim
                            + l * self.nheads * self.headdim
                            + h * self.headdim + p];
                }
            }
        }

        // 11) Apply RMSNormGated if enabled
        if let Some(ref norm) = self.norm {
            let mut z_for_norm = vec![0.0f32; batch * seq_len * self.d_ssm];
            for b in 0..batch {
                for l in 0..seq_len {
                    let src = b * seq_len * d_in_proj + l * d_in_proj + z_ssm_start;
                    for i in 0..self.d_ssm {
                        z_for_norm[b * seq_len * self.d_ssm + l * self.d_ssm + i] = proj[src + i];
                    }
                }
            }
            let normed = norm.forward(&y_flat, Some(&z_for_norm), batch * seq_len);
            y_flat = normed;
        }

        // 12) Concatenate with MLP output if d_mlp > 0
        let mut y_full = vec![0.0f32; batch * seq_len * di];
        for b in 0..batch {
            for l in 0..seq_len {
                let dst_base = b * seq_len * di + l * di;
                if d_mlp > 0 {
                    let proj_base = b * seq_len * d_in_proj + l * d_in_proj;
                    for i in 0..d_mlp {
                        let z0_val = proj[proj_base + z_start + i];
                        let x0_val = proj[proj_base + x_start + i];
                        y_full[dst_base + i] = silu_f32(z0_val) * x0_val;
                    }
                    let y_base = b * seq_len * self.d_ssm + l * self.d_ssm;
                    for i in 0..self.d_ssm {
                        y_full[dst_base + d_mlp + i] = y_flat[y_base + i];
                    }
                } else {
                    let y_base = b * seq_len * self.d_ssm + l * self.d_ssm;
                    for i in 0..di {
                        y_full[dst_base + i] = y_flat[y_base + i];
                    }
                }
            }
        }

        // 13) out_proj: (B, L, d_inner) -> (B, L, d_model)
        let mut output = vec![0.0f32; batch * seq_len * dm];
        for b in 0..batch {
            for l in 0..seq_len {
                let y_off = b * seq_len * di + l * di;
                let o_off = b * seq_len * dm + l * dm;
                for o in 0..dm {
                    let mut acc = 0.0f32;
                    for k in 0..di {
                        acc += y_full[y_off + k] * self.out_proj_weight[o * di + k];
                    }
                    if let Some(ref bias) = self.out_proj_bias {
                        acc += bias[o];
                    }
                    output[o_off + o] = acc;
                }
            }
        }

        output
    }

    /// Single-step decode forward.
    ///
    /// `input`: shape (batch, d_model)
    /// `conv_state`: unused (Mamba-3 has no conv1d)
    /// `ssm_state`: shape (batch, nheads, d_state, headdim) + prev_bx appended
    ///
    /// Returns: shape (batch, d_model)
    pub fn forward_step(
        &self,
        input: &[f32],
        batch: usize,
        _conv_state: &mut [f32],
        ssm_state: &mut [f32],
    ) -> Vec<f32> {
        let dm = self.d_model;
        let di = self.d_inner;
        let ngs = self.ngroups * self.d_state;
        let d_in_proj = di + self.d_ssm + 2 * ngs + self.nheads;
        let d_mlp = di.saturating_sub(self.d_ssm);

        let z_start = 0;
        let x_start = d_mlp;
        let z_ssm_start = 2 * d_mlp;
        let x_ssm_start = z_ssm_start + self.d_ssm;
        let b_start = x_ssm_start + self.d_ssm;
        let c_start = b_start + ngs;
        let dt_start = c_start + ngs;

        // 1) in_proj: (B, d_model) -> (B, d_in_proj)
        let mut proj = vec![0.0f32; batch * d_in_proj];
        for b in 0..batch {
            for o in 0..d_in_proj {
                let mut acc = 0.0f32;
                for k in 0..dm {
                    acc += input[b * dm + k] * self.in_proj_weight[o * dm + k];
                }
                if let Some(ref bias) = self.in_proj_bias {
                    acc += bias[o];
                }
                proj[b * d_in_proj + o] = acc;
            }
        }

        // 2) Extract components
        let mut x_step = vec![0.0f32; batch * self.nheads * self.headdim];
        let mut b_step = vec![0.0f32; batch * ngs];
        let mut c_step = vec![0.0f32; batch * ngs];
        let mut dt_step = vec![0.0f32; batch * self.nheads];

        for b in 0..batch {
            let base = b * d_in_proj;
            for i in 0..self.d_ssm {
                let h = i / self.headdim;
                let p = i % self.headdim;
                x_step[b * self.nheads * self.headdim + h * self.headdim + p] =
                    proj[base + x_ssm_start + i];
            }
            for i in 0..ngs {
                b_step[b * ngs + i] = proj[base + b_start + i];
                c_step[b * ngs + i] = proj[base + c_start + i];
            }
            for h in 0..self.nheads {
                dt_step[b * self.nheads + h] = proj[base + dt_start + h];
            }
        }

        // 3) A = -exp(A_log_real)
        let a_real: Vec<f32> = self.a_log_real.iter().map(|&v| -v.exp()).collect();

        // 4) z for gating
        let z_for_step = if !self.use_rmsnorm {
            let mut z = vec![0.0f32; batch * self.nheads * self.headdim];
            for b in 0..batch {
                let base = b * d_in_proj;
                for i in 0..self.d_ssm {
                    let h = i / self.headdim;
                    let p = i % self.headdim;
                    z[b * self.nheads * self.headdim + h * self.headdim + p] =
                        proj[base + z_ssm_start + i];
                }
            }
            Some(z)
        } else {
            None
        };

        // 5) Split ssm_state into actual state and prev_bx
        let state_size = batch * self.nheads * self.d_state * self.headdim;
        let (actual_state, prev_bx) = ssm_state.split_at_mut(state_size);

        // 6) Run Mamba-3 SSM step
        let y = ssd3::mamba3_ssm_step(
            &x_step, batch, self.nheads, self.headdim,
            &dt_step, &a_real, &self.a_imag,
            &b_step, self.ngroups, self.d_state, &c_step,
            Some(&self.b_bias), Some(&self.c_bias),
            Some(&self.d_skip),
            z_for_step.as_deref(),
            Some(&self.dt_bias),
            true,
            actual_state, prev_bx,
            self.trapezoidal_alpha,
            self.use_rope,
        );

        // 7) Rearrange: (B, nheads, headdim) -> (B, d_ssm)
        let mut y_flat = vec![0.0f32; batch * self.d_ssm];
        for b in 0..batch {
            for i in 0..self.d_ssm {
                let h = i / self.headdim;
                let p = i % self.headdim;
                y_flat[b * self.d_ssm + i] =
                    y[b * self.nheads * self.headdim + h * self.headdim + p];
            }
        }

        // 8) Apply RMSNormGated
        if let Some(ref norm) = self.norm {
            let mut z_for_norm = vec![0.0f32; batch * self.d_ssm];
            for b in 0..batch {
                let base = b * d_in_proj;
                for i in 0..self.d_ssm {
                    z_for_norm[b * self.d_ssm + i] = proj[base + z_ssm_start + i];
                }
            }
            let normed = norm.forward(&y_flat, Some(&z_for_norm), batch);
            y_flat = normed;
        }

        // 9) MLP + concat
        let mut y_full = vec![0.0f32; batch * di];
        for b in 0..batch {
            let dst_base = b * di;
            if d_mlp > 0 {
                let proj_base = b * d_in_proj;
                for i in 0..d_mlp {
                    let z0_val = proj[proj_base + z_start + i];
                    let x0_val = proj[proj_base + x_start + i];
                    y_full[dst_base + i] = silu_f32(z0_val) * x0_val;
                }
                for i in 0..self.d_ssm {
                    y_full[dst_base + d_mlp + i] = y_flat[b * self.d_ssm + i];
                }
            } else {
                for i in 0..di {
                    y_full[dst_base + i] = y_flat[b * self.d_ssm + i];
                }
            }
        }

        // 10) out_proj
        let mut output = vec![0.0f32; batch * dm];
        for b in 0..batch {
            for o in 0..dm {
                let mut acc = 0.0f32;
                for k in 0..di {
                    acc += y_full[b * di + k] * self.out_proj_weight[o * di + k];
                }
                if let Some(ref bias) = self.out_proj_bias {
                    acc += bias[o];
                }
                output[b * dm + o] = acc;
            }
        }

        output
    }

    /// Conv state size — Mamba-3 has no conv1d, returns 0.
    pub fn conv_state_size(&self, _batch: usize) -> usize {
        0
    }

    /// SSM state size: state + prev_bx for trapezoidal rule.
    pub fn ssm_state_size(&self, batch: usize) -> usize {
        // state: (batch, nheads, d_state, headdim)
        // prev_bx: (batch, nheads, d_state, headdim)
        2 * batch * self.nheads * self.d_state * self.headdim
    }

    /// Total parameter count.
    pub fn param_count(&self) -> usize {
        let di = self.d_inner;
        let ngs = self.ngroups * self.d_state;
        let d_in_proj = di + self.d_ssm + 2 * ngs + self.nheads;

        let in_proj = d_in_proj * self.d_model
            + if self.in_proj_bias.is_some() { d_in_proj } else { 0 };
        let b_c_bias = 2 * ngs;
        let dt_bias = self.nheads;
        let a_params = 2 * self.nheads; // a_log_real + a_imag
        let d_skip = self.nheads;
        let norm_params = if self.norm.is_some() { 2 * self.d_ssm } else { 0 };
        let out_proj = self.d_model * di
            + if self.out_proj_bias.is_some() { self.d_model } else { 0 };

        in_proj + b_c_bias + dt_bias + a_params + d_skip + norm_params + out_proj
    }
}

impl Module for Mamba3 {
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
    fn test_mamba3_creation() {
        let m = Mamba3::new(64, 16, 2, 16);
        assert_eq!(m.d_model, 64);
        assert_eq!(m.d_inner, 128);
        assert_eq!(m.nheads, 8); // 128 / 16
        assert_eq!(m.headdim, 16);
        assert!(m.use_rope);
    }

    #[test]
    fn test_mamba3_forward_shape() {
        let m = Mamba3::new(64, 16, 2, 16);
        let batch = 2;
        let seq_len = 4;
        let input = vec![0.1f32; batch * seq_len * 64];
        let output = m.forward_train(&input, batch, seq_len);
        assert_eq!(output.len(), batch * seq_len * 64);
    }

    #[test]
    fn test_mamba3_step_shape() {
        let m = Mamba3::new(64, 16, 2, 16);
        let batch = 1;
        let input = vec![0.1f32; batch * 64];
        let mut conv_state = vec![]; // unused
        let ssm_size = m.ssm_state_size(batch);
        let mut ssm_state = vec![0.0f32; ssm_size];

        let output = m.forward_step(&input, batch, &mut conv_state, &mut ssm_state);
        assert_eq!(output.len(), batch * 64);

        // Second step should also work
        let output2 = m.forward_step(&input, batch, &mut conv_state, &mut ssm_state);
        assert_eq!(output2.len(), batch * 64);
    }

    #[test]
    fn test_mamba3_no_rope() {
        let m = Mamba3::new_full(
            64, 16, 2, 16,
            None, 1, false, true, false, false, false, true, 0.5, None,
        );
        assert!(!m.use_rope);
        assert!(m.a_imag.iter().all(|&v| v == 0.0));

        let batch = 1;
        let seq_len = 2;
        let input = vec![0.1f32; batch * seq_len * 64];
        let output = m.forward_train(&input, batch, seq_len);
        assert_eq!(output.len(), batch * seq_len * 64);
    }

    #[test]
    fn test_mamba3_param_count() {
        let m = Mamba3::new(64, 16, 2, 16);
        let count = m.param_count();
        assert!(count > 0);
    }

    #[test]
    fn test_mamba3_module_trait() {
        let m = Mamba3::new(64, 16, 2, 16);
        let input = Tensor::from_f32(&vec![0.1f32; 2 * 4 * 64], &[2, 4, 64]);
        let result = m.forward(&input);
        assert!(result.is_ok());
        let out = result.unwrap();
        assert_eq!(out.shape().dims(), &[2, 4, 64]);
    }
}
