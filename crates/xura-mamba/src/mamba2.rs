//! Mamba-2 block — multi-head SSD architecture.
//!
//! Implements the Mamba-2 block from "Transformers are SSMs: Generalized Models and
//! Efficient Algorithms Through Structured State Space Duality" (Dao & Gu, 2024).
//!
//! Reference: `mamba_ssm/modules/mamba2.py`

use rand::Rng;

use kore_core::Tensor;
use kore_nn::module::Module;

use crate::causal_conv1d;
use crate::norm::RMSNormGated;
use crate::ssd;

/// Mamba-2 block with multi-head structured state space duality.
pub struct Mamba2 {
    // Dimensions
    pub d_model: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub expand: usize,
    pub d_inner: usize,
    pub d_ssm: usize,
    pub headdim: usize,
    pub nheads: usize,
    pub ngroups: usize,
    pub chunk_size: usize,
    pub d_has_hdim: bool,
    pub use_rmsnorm: bool,
    pub norm_before_gate: bool,
    pub layer_idx: Option<usize>,

    // Parameters
    /// in_proj weight: (d_in_proj, d_model) where d_in_proj = 2*d_inner + 2*ngroups*d_state + nheads
    pub in_proj_weight: Vec<f32>,
    pub in_proj_bias: Option<Vec<f32>>,

    /// conv1d weight: (conv_dim, kernel_size) depthwise, conv_dim = d_ssm + 2*ngroups*d_state
    pub conv1d_weight: Vec<f32>,
    pub conv1d_bias: Option<Vec<f32>>,
    pub conv_dim: usize,

    /// dt_bias: (nheads,)
    pub dt_bias: Vec<f32>,

    /// A_log: (nheads,) — log of the state matrix diagonal
    pub a_log: Vec<f32>,

    /// D skip parameter: (nheads,) or (d_ssm,) if d_has_hdim
    pub d_skip: Vec<f32>,

    /// RMSNormGated (if use_rmsnorm)
    pub norm: Option<RMSNormGated>,

    /// out_proj weight: (d_model, d_inner)
    pub out_proj_weight: Vec<f32>,
    pub out_proj_bias: Option<Vec<f32>>,

    training: bool,
}

impl Mamba2 {
    /// Create a new Mamba-2 block with default options.
    pub fn new(d_model: usize, d_state: usize, d_conv: usize, expand: usize, headdim: usize) -> Self {
        Self::new_full(d_model, d_state, d_conv, expand, headdim, None, 1, false, true, false, false, true, 256, None)
    }

    /// Create with full configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn new_full(
        d_model: usize,
        d_state: usize,
        d_conv: usize,
        expand: usize,
        headdim: usize,
        d_ssm: Option<usize>,
        ngroups: usize,
        d_has_hdim: bool,
        rmsnorm: bool,
        norm_before_gate: bool,
        bias: bool,
        conv_bias: bool,
        chunk_size: usize,
        layer_idx: Option<usize>,
    ) -> Self {
        let d_inner = expand * d_model;
        let d_ssm = d_ssm.unwrap_or(d_inner);
        assert!(d_ssm % headdim == 0, "d_ssm must be divisible by headdim");
        let nheads = d_ssm / headdim;

        let mut rng = rand::thread_rng();

        // in_proj: d_model -> d_in_proj
        // Order: [z(d_inner), x(d_inner → but only d_ssm used for SSM), B, C, dt]
        // Actually: d_in_proj = 2*d_inner + 2*ngroups*d_state + nheads
        let d_in_proj = 2 * d_inner + 2 * ngroups * d_state + nheads;
        let in_limit = (6.0 / (d_model + d_in_proj) as f32).sqrt();
        let in_proj_weight: Vec<f32> = (0..d_in_proj * d_model)
            .map(|_| rng.gen_range(-in_limit..in_limit))
            .collect();
        let in_proj_bias = if bias {
            Some(vec![0.0f32; d_in_proj])
        } else {
            None
        };

        // conv1d: depthwise over conv_dim = d_ssm + 2*ngroups*d_state
        let conv_dim = d_ssm + 2 * ngroups * d_state;
        let conv_limit = (6.0 / d_conv as f32).sqrt();
        let conv1d_weight: Vec<f32> = (0..conv_dim * d_conv)
            .map(|_| rng.gen_range(-conv_limit..conv_limit))
            .collect();
        let conv1d_bias = if conv_bias {
            Some(vec![0.0f32; conv_dim])
        } else {
            None
        };

        // dt_bias: initialized so softplus(bias) in [dt_min, dt_max]
        let dt_min: f32 = 0.001;
        let dt_max: f32 = 0.1;
        let dt_init_floor: f32 = 1e-4;
        let dt_bias: Vec<f32> = (0..nheads)
            .map(|_| {
                let dt = (rng.gen::<f32>() * (dt_max.ln() - dt_min.ln()) + dt_min.ln())
                    .exp()
                    .max(dt_init_floor);
                dt + (-(-dt).exp_m1()).ln()
            })
            .collect();

        // A_log: uniform in [1, 16], then log
        let a_log: Vec<f32> = (0..nheads)
            .map(|_| {
                let a = rng.gen_range(1.0f32..16.0);
                a.ln()
            })
            .collect();

        // D: skip parameter
        let d_skip_len = if d_has_hdim { d_ssm } else { nheads };
        let d_skip = vec![1.0f32; d_skip_len];

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
            d_conv,
            expand,
            d_inner,
            d_ssm,
            headdim,
            nheads,
            ngroups,
            chunk_size,
            d_has_hdim,
            use_rmsnorm: rmsnorm,
            norm_before_gate,
            layer_idx,
            in_proj_weight,
            in_proj_bias,
            conv1d_weight,
            conv1d_bias,
            conv_dim,
            dt_bias,
            a_log,
            d_skip,
            norm,
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
        let d_in_proj = 2 * di + 2 * self.ngroups * self.d_state + self.nheads;

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

        // 2) Split: [z(d_inner), x_ssm_and_mlp, xBC(d_ssm + 2*ngroups*d_state), dt(nheads)]
        //    d_mlp = (d_in_proj - 2*d_ssm - 2*ngroups*d_state - nheads) / 2
        let d_mlp = (d_in_proj - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) / 2;
        // Offsets within d_in_proj:
        let z0_start = 0;
        let x0_start = d_mlp;
        let z_start = 2 * d_mlp;
        let xbc_start = z_start + self.d_ssm;
        let dt_start = xbc_start + self.d_ssm + 2 * self.ngroups * self.d_state;

        // 3) Extract xBC and apply causal conv1d
        // xBC: (B, L, conv_dim) -> rearrange to (B, conv_dim, L) for conv
        let cd = self.conv_dim;
        let mut xbc_bcl = vec![0.0f32; batch * cd * seq_len];
        for b in 0..batch {
            for l in 0..seq_len {
                let src = b * seq_len * d_in_proj + l * d_in_proj + xbc_start;
                for ch in 0..cd {
                    xbc_bcl[b * cd * seq_len + ch * seq_len + l] = proj[src + ch];
                }
            }
        }

        let xbc_conv = causal_conv1d::causal_conv1d_fn(
            &xbc_bcl, batch, cd, seq_len,
            &self.conv1d_weight, self.d_conv,
            self.conv1d_bias.as_deref(), true,
        );

        // 4) Split convolved xBC back to (B, L, ...): x_ssm, B_ssm, C_ssm
        // Rearrange back to (B, L, conv_dim), then split
        let ngs = self.ngroups * self.d_state;

        // x_ssm: (B, L, nheads, headdim) for scan
        let mut x_scan = vec![0.0f32; batch * seq_len * self.nheads * self.headdim];
        // B_scan: (B, L, ngroups, d_state)
        let mut b_scan = vec![0.0f32; batch * seq_len * self.ngroups * self.d_state];
        // C_scan: (B, L, ngroups, d_state)
        let mut c_scan = vec![0.0f32; batch * seq_len * self.ngroups * self.d_state];

        for b in 0..batch {
            for l in 0..seq_len {
                // x: first d_ssm channels
                for i in 0..self.d_ssm {
                    let h = i / self.headdim;
                    let p = i % self.headdim;
                    x_scan[b * seq_len * self.nheads * self.headdim
                        + l * self.nheads * self.headdim
                        + h * self.headdim + p] =
                        xbc_conv[b * cd * seq_len + i * seq_len + l];
                }
                // B: next ngroups*d_state channels
                for i in 0..ngs {
                    b_scan[b * seq_len * ngs + l * ngs + i] =
                        xbc_conv[b * cd * seq_len + (self.d_ssm + i) * seq_len + l];
                }
                // C: next ngroups*d_state channels
                for i in 0..ngs {
                    c_scan[b * seq_len * ngs + l * ngs + i] =
                        xbc_conv[b * cd * seq_len + (self.d_ssm + ngs + i) * seq_len + l];
                }
            }
        }

        // 5) Extract dt: (B, L, nheads)
        let mut dt_scan = vec![0.0f32; batch * seq_len * self.nheads];
        for b in 0..batch {
            for l in 0..seq_len {
                let src = b * seq_len * d_in_proj + l * d_in_proj + dt_start;
                for h in 0..self.nheads {
                    dt_scan[b * seq_len * self.nheads + l * self.nheads + h] = proj[src + h];
                }
            }
        }

        // 6) A = -exp(A_log)
        let a_neg: Vec<f32> = self.a_log.iter().map(|&v| -v.exp()).collect();

        // 7) Extract z for SSM gating (only if no rmsnorm — rmsnorm handles it)
        let z_for_scan = if !self.use_rmsnorm {
            // z: (B, L, d_ssm) -> rearrange to (B, L, nheads, headdim)
            let mut z_scan = vec![0.0f32; batch * seq_len * self.nheads * self.headdim];
            for b in 0..batch {
                for l in 0..seq_len {
                    let src = b * seq_len * d_in_proj + l * d_in_proj + z_start;
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

        // 8) Run SSD scan
        let scan_result = ssd::mamba_chunk_scan_combined(
            &x_scan, batch, seq_len, self.nheads, self.headdim,
            &dt_scan, &a_neg, &b_scan, self.ngroups, self.d_state, &c_scan,
            Some(&self.d_skip),
            z_for_scan.as_deref(),
            Some(&self.dt_bias),
            true, // dt_softplus
            self.d_has_hdim,
        );

        // 9) Rearrange scan output: (B, L, nheads, headdim) -> (B, L, d_ssm)
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

        // 10) Apply RMSNormGated if enabled
        if let Some(ref norm) = self.norm {
            // z: (B, L, d_ssm)
            let mut z_for_norm = vec![0.0f32; batch * seq_len * self.d_ssm];
            for b in 0..batch {
                for l in 0..seq_len {
                    let src = b * seq_len * d_in_proj + l * d_in_proj + z_start;
                    for i in 0..self.d_ssm {
                        z_for_norm[b * seq_len * self.d_ssm + l * self.d_ssm + i] = proj[src + i];
                    }
                }
            }
            let normed = norm.forward(&y_flat, Some(&z_for_norm), batch * seq_len);
            y_flat = normed;
        }

        // 11) Concatenate with MLP output if d_mlp > 0
        // y_full: (B, L, d_inner)
        let mut y_full = vec![0.0f32; batch * seq_len * di];
        for b in 0..batch {
            for l in 0..seq_len {
                let dst_base = b * seq_len * di + l * di;
                if d_mlp > 0 {
                    // MLP part: silu(z0) * x0
                    let proj_base = b * seq_len * d_in_proj + l * d_in_proj;
                    for i in 0..d_mlp {
                        let z0_val = proj[proj_base + z0_start + i];
                        let x0_val = proj[proj_base + x0_start + i];
                        y_full[dst_base + i] = silu_f32(z0_val) * x0_val;
                    }
                    // SSM part
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

        // 12) out_proj: (B, L, d_inner) -> (B, L, d_model)
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
    /// `conv_state`: shape (batch, conv_dim, d_conv)
    /// `ssm_state`: shape (batch, nheads, headdim, d_state)
    ///
    /// Returns: shape (batch, d_model)
    pub fn forward_step(
        &self,
        input: &[f32],
        batch: usize,
        conv_state: &mut [f32],
        ssm_state: &mut [f32],
    ) -> Vec<f32> {
        let dm = self.d_model;
        let di = self.d_inner;
        let d_in_proj = 2 * di + 2 * self.ngroups * self.d_state + self.nheads;
        let d_mlp = (d_in_proj - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) / 2;

        let z0_start = 0;
        let x0_start = d_mlp;
        let z_start = 2 * d_mlp;
        let xbc_start = z_start + self.d_ssm;
        let dt_start = xbc_start + self.d_ssm + 2 * self.ngroups * self.d_state;

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

        // 2) Extract xBC and conv update
        let cd = self.conv_dim;
        let ngs = self.ngroups * self.d_state;
        let mut xbc_step = vec![0.0f32; batch * cd];
        for b in 0..batch {
            for ch in 0..cd {
                xbc_step[b * cd + ch] = proj[b * d_in_proj + xbc_start + ch];
            }
        }

        let xbc_conv = causal_conv1d::causal_conv1d_update(
            &xbc_step, batch, cd, conv_state, self.d_conv,
            &self.conv1d_weight, self.conv1d_bias.as_deref(), true,
        );

        // 3) Split: x_ssm (d_ssm), B (ngs), C (ngs)
        let mut x_step = vec![0.0f32; batch * self.nheads * self.headdim];
        let mut b_step = vec![0.0f32; batch * self.ngroups * self.d_state];
        let mut c_step = vec![0.0f32; batch * self.ngroups * self.d_state];
        for b in 0..batch {
            for i in 0..self.d_ssm {
                let h = i / self.headdim;
                let p = i % self.headdim;
                x_step[b * self.nheads * self.headdim + h * self.headdim + p] =
                    xbc_conv[b * cd + i];
            }
            for i in 0..ngs {
                b_step[b * ngs + i] = xbc_conv[b * cd + self.d_ssm + i];
                c_step[b * ngs + i] = xbc_conv[b * cd + self.d_ssm + ngs + i];
            }
        }

        // 4) dt: (B, nheads)
        let mut dt_step = vec![0.0f32; batch * self.nheads];
        for b in 0..batch {
            for h in 0..self.nheads {
                dt_step[b * self.nheads + h] = proj[b * d_in_proj + dt_start + h];
            }
        }

        // 5) A = -exp(A_log)
        let a_neg: Vec<f32> = self.a_log.iter().map(|&v| -v.exp()).collect();

        // 6) z for gating (if no rmsnorm)
        let z_for_step = if !self.use_rmsnorm {
            let mut z = vec![0.0f32; batch * self.nheads * self.headdim];
            for b in 0..batch {
                for i in 0..self.d_ssm {
                    let h = i / self.headdim;
                    let p = i % self.headdim;
                    z[b * self.nheads * self.headdim + h * self.headdim + p] =
                        proj[b * d_in_proj + z_start + i];
                }
            }
            Some(z)
        } else {
            None
        };

        // 7) SSM step
        let y_bhp = ssd::mamba2_ssm_step(
            &x_step, batch, self.nheads, self.headdim,
            &dt_step, &a_neg, &b_step, self.ngroups, self.d_state, &c_step,
            Some(&self.d_skip),
            z_for_step.as_deref(),
            Some(&self.dt_bias), true,
            ssm_state,
        );

        // 8) Rearrange to (B, d_ssm)
        let mut y_flat = vec![0.0f32; batch * self.d_ssm];
        for b in 0..batch {
            for i in 0..self.d_ssm {
                let h = i / self.headdim;
                let p = i % self.headdim;
                y_flat[b * self.d_ssm + i] = y_bhp[b * self.nheads * self.headdim + h * self.headdim + p];
            }
        }

        // 9) RMSNormGated
        if let Some(ref norm) = self.norm {
            let mut z_for_norm = vec![0.0f32; batch * self.d_ssm];
            for b in 0..batch {
                for i in 0..self.d_ssm {
                    z_for_norm[b * self.d_ssm + i] = proj[b * d_in_proj + z_start + i];
                }
            }
            y_flat = norm.forward(&y_flat, Some(&z_for_norm), batch);
        }

        // 10) Concatenate MLP if needed, then out_proj
        let mut y_full = vec![0.0f32; batch * di];
        for b in 0..batch {
            if d_mlp > 0 {
                for i in 0..d_mlp {
                    let z0 = proj[b * d_in_proj + z0_start + i];
                    let x0 = proj[b * d_in_proj + x0_start + i];
                    y_full[b * di + i] = silu_f32(z0) * x0;
                }
                for i in 0..self.d_ssm {
                    y_full[b * di + d_mlp + i] = y_flat[b * self.d_ssm + i];
                }
            } else {
                for i in 0..di {
                    y_full[b * di + i] = y_flat[b * self.d_ssm + i];
                }
            }
        }

        // out_proj
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

    /// Conv state size: batch * conv_dim * d_conv
    pub fn conv_state_size(&self, batch: usize) -> usize {
        batch * self.conv_dim * self.d_conv
    }

    /// SSM state size: batch * nheads * headdim * d_state
    pub fn ssm_state_size(&self, batch: usize) -> usize {
        batch * self.nheads * self.headdim * self.d_state
    }

    /// Total parameter count.
    pub fn param_count(&self) -> usize {
        let d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads;
        let in_proj = d_in_proj * self.d_model;
        let conv = self.conv_dim * self.d_conv + if self.conv1d_bias.is_some() { self.conv_dim } else { 0 };
        let dt_bias = self.nheads;
        let a_log = self.nheads;
        let d_skip = self.d_skip.len();
        let norm = if self.use_rmsnorm { self.d_ssm } else { 0 };
        let out_proj = self.d_model * self.d_inner;
        in_proj + conv + dt_bias + a_log + d_skip + norm + out_proj
    }
}

#[inline]
fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

impl Module for Mamba2 {
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

    fn parameters(&self) -> Vec<&Tensor> { vec![] }
    fn named_parameters(&self) -> Vec<(&str, &Tensor)> { vec![] }
    fn train(&mut self, mode: bool) { self.training = mode; }
    fn is_training(&self) -> bool { self.training }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba2_creation() {
        let m2 = Mamba2::new(64, 16, 4, 2, 16);
        assert_eq!(m2.d_model, 64);
        assert_eq!(m2.d_inner, 128);
        assert_eq!(m2.d_ssm, 128);
        assert_eq!(m2.nheads, 8); // 128 / 16
        assert_eq!(m2.headdim, 16);
    }

    #[test]
    fn test_mamba2_forward_shape() {
        let m2 = Mamba2::new(32, 8, 4, 2, 16);
        let batch = 2;
        let seq_len = 4;
        let input = vec![0.1f32; batch * seq_len * 32];
        let output = m2.forward_train(&input, batch, seq_len);
        assert_eq!(output.len(), batch * seq_len * 32);
    }

    #[test]
    fn test_mamba2_step_shape() {
        let m2 = Mamba2::new(32, 8, 4, 2, 16);
        let batch = 1;
        let mut conv_state = vec![0.0f32; m2.conv_state_size(batch)];
        let mut ssm_state = vec![0.0f32; m2.ssm_state_size(batch)];

        let input = vec![0.1f32; batch * 32];
        let output = m2.forward_step(&input, batch, &mut conv_state, &mut ssm_state);
        assert_eq!(output.len(), batch * 32);
    }

    #[test]
    fn test_mamba2_module_trait() {
        let m2 = Mamba2::new(32, 8, 4, 2, 16);
        let input = Tensor::from_f32(&vec![0.1f32; 2 * 4 * 32], &[2, 4, 32]);
        let output = m2.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 4, 32]);
    }
}
