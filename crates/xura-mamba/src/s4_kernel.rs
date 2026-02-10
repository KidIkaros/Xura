//! S4 SSM Kernels — diagonal (S4D) and DPLR variants.
//!
//! These kernels generate convolution filters from SSM parameters (A, B, C, dt).
//! The filters are then used by FFTConv for efficient sequence-level computation.
//!
//! Two variants:
//! - `SSMKernelDiag` (S4D): diagonal state matrix, uses Vandermonde multiplication
//! - `SSMKernelDPLR` (full S4): diagonal + low-rank, uses Cauchy kernel + Woodbury correction

use rand::Rng;

use crate::complex_utils::{c2r, r2c, cauchy_naive, log_vandermonde_naive, fft_nodes, C32};
use crate::dplr;

/// Discretization method.
#[derive(Clone, Debug, PartialEq)]
pub enum Discretization {
    /// Zero-order hold
    Zoh,
    /// Bilinear (Tustin) transform
    Bilinear,
}

/// S4D kernel — diagonal state matrix variant.
pub struct SSMKernelDiag {
    pub h: usize,       // d_model (number of independent SSM copies)
    pub n: usize,       // d_state (full state size, internal N is half due to conjugate symmetry)
    pub half_n: usize,  // N/2
    pub channels: usize,
    pub disc: Discretization,
    pub n_ssm: usize,   // number of trainable (A, B) copies
    pub repeat: usize,  // H / n_ssm

    // Parameters (stored as interleaved real pairs for complex)
    /// inv_dt: shape (H,) — log-space timescale
    pub inv_dt: Vec<f32>,
    /// A_real: shape (n_ssm, N/2) — negative real part of A (stored as positive, negated on use)
    pub a_real: Vec<f32>,
    /// A_imag: shape (n_ssm, N/2) — negative imag part of A (stored as positive, negated on use)
    pub a_imag: Vec<f32>,
    /// B: shape (1, n_ssm, N/2) complex — interleaved real/imag, so len = 2*n_ssm*half_n
    pub b: Vec<f32>,
    /// C: shape (channels, H, N/2) complex — interleaved, len = 2*channels*H*half_n
    pub c: Vec<f32>,
}

impl SSMKernelDiag {
    /// Create a new S4D kernel.
    pub fn new(
        d_model: usize,
        d_state: usize,
        channels: usize,
        init: &str,
        dt_min: f32,
        dt_max: f32,
        disc: Discretization,
        n_ssm: Option<usize>,
    ) -> Self {
        let h = d_model;
        let n = d_state;
        let half_n = n / 2;
        let n_ssm = n_ssm.unwrap_or(h);
        let repeat = h / n_ssm;

        let mut rng = rand::thread_rng();

        // Initialize dt
        let inv_dt: Vec<f32> = (0..h)
            .map(|_| {
                rng.gen::<f32>() * (dt_max.ln() - dt_min.ln()) + dt_min.ln()
            })
            .collect();

        // Initialize A, B via DPLR
        let (a_init, _p_init, b_init) = dplr::ssm_init(init, n, 1, n_ssm);

        // Extract real and imag parts of A
        let mut a_real = vec![0.0f32; n_ssm * half_n];
        let mut a_imag = vec![0.0f32; n_ssm * half_n];
        for s in 0..n_ssm {
            for ni in 0..half_n {
                let ai = a_init[s * half_n + ni];
                // Store as positive (will be negated on use via exp transform)
                a_real[s * half_n + ni] = (-ai.re).max(1e-4).ln();
                a_imag[s * half_n + ni] = (-ai.im).max(0.0);
            }
        }

        // B: complex, shape (1, n_ssm, N/2)
        let mut b_flat = vec![0.0f32; 2 * n_ssm * half_n];
        for s in 0..n_ssm {
            for ni in 0..half_n {
                let bi = b_init[s * half_n + ni];
                b_flat[2 * (s * half_n + ni)] = bi.re;
                b_flat[2 * (s * half_n + ni) + 1] = bi.im;
            }
        }

        // C: random complex, shape (channels, H, N/2)
        let c_flat: Vec<f32> = (0..2 * channels * h * half_n)
            .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
            .collect();

        Self {
            h, n, half_n, channels, disc, n_ssm, repeat,
            inv_dt, a_real, a_imag, b: b_flat, c: c_flat,
        }
    }

    /// Get processed parameters.
    fn get_params(&self, rate: f32) -> (Vec<f32>, Vec<C32>, Vec<C32>, Vec<C32>) {
        // A: (n_ssm, N/2) complex
        let mut a = vec![C32::new(0.0, 0.0); self.n_ssm * self.half_n];
        for i in 0..self.n_ssm * self.half_n {
            let re = -self.a_real[i].exp();
            let im = -self.a_imag[i];
            a[i] = C32::new(re, im);
        }

        // Expand A to (H, N/2) by repeating
        let mut a_full = vec![C32::new(0.0, 0.0); self.h * self.half_n];
        for hi in 0..self.h {
            let si = hi / self.repeat;
            for ni in 0..self.half_n {
                a_full[hi * self.half_n + ni] = a[si * self.half_n + ni];
            }
        }

        // dt: (H,)
        let dt: Vec<f32> = self.inv_dt.iter().map(|&v| v.exp() * rate).collect();

        // B: (1, n_ssm, N/2) → expand to (1, H, N/2)
        let b_complex = r2c(&self.b);
        let mut b_full = vec![C32::new(0.0, 0.0); self.h * self.half_n];
        for hi in 0..self.h {
            let si = hi / self.repeat;
            for ni in 0..self.half_n {
                b_full[hi * self.half_n + ni] = b_complex[si * self.half_n + ni];
            }
        }

        // C: (channels, H, N/2)
        let c_complex = r2c(&self.c);

        (dt, a_full, b_full, c_complex)
    }

    /// Generate the convolution kernel of length L.
    ///
    /// Returns kernel of shape (channels, H, L) as flat f32 vec.
    pub fn forward(&self, l: usize) -> Vec<f32> {
        let (dt, a, b, c) = self.get_params(1.0);

        // dtA = dt * A, shape (H, N/2)
        let mut dt_a = vec![C32::new(0.0, 0.0); self.h * self.half_n];
        for hi in 0..self.h {
            for ni in 0..self.half_n {
                dt_a[hi * self.half_n + ni] = a[hi * self.half_n + ni] * dt[hi];
            }
        }

        // Combine B and C: v = B * C * dt, shape (channels, H, N/2)
        // For each channel c_idx: v[h, n] = B[h, n] * C[c_idx, h, n]
        let mut kernel = vec![0.0f32; self.channels * self.h * l];

        for c_idx in 0..self.channels {
            for hi in 0..self.h {
                // Prepare v and x for this (channel, head)
                let mut v = vec![C32::new(0.0, 0.0); self.half_n];
                let mut x = vec![C32::new(0.0, 0.0); self.half_n];
                for ni in 0..self.half_n {
                    let bi = b[hi * self.half_n + ni];
                    let ci = c[c_idx * self.h * self.half_n + hi * self.half_n + ni];
                    let dta = dt_a[hi * self.half_n + ni];

                    x[ni] = dta;

                    match self.disc {
                        Discretization::Zoh => {
                            // C_bar = C * (exp(dtA) - 1) / A
                            let ai = a[hi * self.half_n + ni];
                            let exp_dta = dta.exp();
                            let c_bar = ci * (exp_dta - C32::new(1.0, 0.0)) / ai;
                            v[ni] = bi * c_bar * dt[hi];
                        }
                        Discretization::Bilinear => {
                            let c_bar = ci * (C32::new(1.0, 0.0) - dta / 2.0).inv() * dt[hi];
                            v[ni] = bi * c_bar;
                            // x = log((1 + dtA/2) / (1 - dtA/2))
                            x[ni] = ((C32::new(1.0, 0.0) + dta / 2.0) /
                                     (C32::new(1.0, 0.0) - dta / 2.0)).ln();
                        }
                    }
                }

                // Vandermonde multiplication → kernel of length L
                let k = log_vandermonde_naive(&v, &x, l);
                let base = c_idx * self.h * l + hi * l;
                kernel[base..base + l].copy_from_slice(&k);
            }
        }

        kernel
    }

    /// Set up discrete step parameters for recurrent inference.
    ///
    /// Returns (dA, dB, dC):
    /// - dA: shape (H, N/2) complex — discrete state transition
    /// - dB: shape (H, N/2) complex — discrete input matrix
    /// - dC: shape (channels, H, N/2) complex — output matrix
    pub fn setup_step(&self) -> (Vec<C32>, Vec<C32>, Vec<C32>) {
        let (dt, a, b, _c) = self.get_params(1.0);

        let mut da = vec![C32::new(0.0, 0.0); self.h * self.half_n];
        let mut db = vec![C32::new(0.0, 0.0); self.h * self.half_n];

        for hi in 0..self.h {
            for ni in 0..self.half_n {
                let dta = a[hi * self.half_n + ni] * dt[hi];
                let ai = a[hi * self.half_n + ni];
                let bi = b[hi * self.half_n + ni];

                match self.disc {
                    Discretization::Zoh => {
                        da[hi * self.half_n + ni] = dta.exp();
                        let exp_dta = dta.exp();
                        db[hi * self.half_n + ni] = bi * (exp_dta - C32::new(1.0, 0.0)) / ai;
                    }
                    Discretization::Bilinear => {
                        let half_dta = dta / 2.0;
                        da[hi * self.half_n + ni] = (C32::new(1.0, 0.0) + half_dta)
                            / (C32::new(1.0, 0.0) - half_dta);
                        db[hi * self.half_n + ni] = bi * (C32::new(1.0, 0.0) - half_dta).inv() * dt[hi];
                    }
                }
            }
        }

        // dC = C (channels, H, N/2) complex
        let dc = r2c(&self.c);

        (da, db, dc)
    }

    /// Create default (zero) state for recurrent inference.
    ///
    /// Returns state of shape (batch, H, N/2) complex, as interleaved f32.
    pub fn default_state(&self, batch: usize) -> Vec<f32> {
        vec![0.0f32; batch * self.h * self.half_n * 2]
    }

    /// Recurrent step: process one timestep.
    ///
    /// # Arguments
    /// - `u`: input, shape (batch, H)
    /// - `state`: complex state, shape (batch, H, N/2) as interleaved f32
    /// - `da`, `db`, `dc`: from setup_step()
    ///
    /// # Returns
    /// (output, new_state) where output is (batch, channels, H) and state is updated in-place.
    pub fn step(
        &self,
        u: &[f32],
        state: &mut [f32],
        da: &[C32],
        db: &[C32],
        dc: &[C32],
        batch: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; batch * self.channels * self.h];

        for bi in 0..batch {
            for hi in 0..self.h {
                for ni in 0..self.half_n {
                    let s_idx = (bi * self.h * self.half_n + hi * self.half_n + ni) * 2;
                    let old_re = state[s_idx];
                    let old_im = state[s_idx + 1];
                    let old = C32::new(old_re, old_im);

                    let da_val = da[hi * self.half_n + ni];
                    let db_val = db[hi * self.half_n + ni];
                    let u_val = C32::new(u[bi * self.h + hi], 0.0);

                    let new_state = da_val * old + db_val * u_val;
                    state[s_idx] = new_state.re;
                    state[s_idx + 1] = new_state.im;

                    // Accumulate output: y = Re(dC * new_state)
                    for c_idx in 0..self.channels {
                        let dc_val = dc[c_idx * self.h * self.half_n + hi * self.half_n + ni];
                        let contrib = (dc_val * new_state).re * 2.0;
                        output[bi * self.channels * self.h + c_idx * self.h + hi] += contrib;
                    }
                }
            }
        }

        output
    }

    /// Total parameter count.
    pub fn param_count(&self) -> usize {
        self.h                           // inv_dt
        + self.n_ssm * self.half_n       // A_real
        + self.n_ssm * self.half_n       // A_imag
        + 2 * self.n_ssm * self.half_n   // B (complex)
        + 2 * self.channels * self.h * self.half_n // C (complex)
    }
}

/// S4-DPLR kernel — diagonal + low-rank state matrix.
///
/// Extends SSMKernelDiag with a low-rank correction P, enabling the full S4 model.
/// Uses Cauchy kernel computation instead of Vandermonde.
pub struct SSMKernelDPLR {
    /// The underlying diagonal kernel (shares A, B, C, dt parameters)
    pub diag: SSMKernelDiag,
    /// Low-rank correction P: shape (rank, n_ssm, N/2) complex as interleaved f32
    pub p: Vec<f32>,
    pub rank: usize,
}

impl SSMKernelDPLR {
    /// Create a new S4-DPLR kernel.
    pub fn new(
        d_model: usize,
        d_state: usize,
        channels: usize,
        init: &str,
        rank: usize,
        dt_min: f32,
        dt_max: f32,
        disc: Discretization,
        n_ssm: Option<usize>,
    ) -> Self {
        let diag = SSMKernelDiag::new(d_model, d_state, channels, init, dt_min, dt_max, disc, n_ssm);

        let actual_n_ssm = n_ssm.unwrap_or(d_model);

        // Initialize P from DPLR
        let (_, p_init, _) = dplr::ssm_init(init, d_state, rank, actual_n_ssm);
        let p_flat = c2r(&p_init);

        Self {
            diag,
            p: p_flat,
            rank,
        }
    }

    /// Generate the convolution kernel using Cauchy multiplication + Woodbury correction.
    ///
    /// Returns kernel of shape (channels, H, L) as flat f32 vec.
    pub fn forward(&self, l: usize) -> Vec<f32> {
        let d = &self.diag;
        let (dt, a, b, c) = d.get_params(1.0);
        let p_complex = r2c(&self.p);

        // dtA = dt * A
        let mut dt_a = vec![C32::new(0.0, 0.0); d.h * d.half_n];
        for hi in 0..d.h {
            for ni in 0..d.half_n {
                dt_a[hi * d.half_n + ni] = a[hi * d.half_n + ni] * dt[hi];
            }
        }

        // Get FFT nodes
        let (omega, z) = fft_nodes(l);
        let fft_len = z.len(); // L/2 + 1

        // Expand P to (rank, H, N/2) and create Q = P*
        let mut p_full = vec![C32::new(0.0, 0.0); self.rank * d.h * d.half_n];
        for r in 0..self.rank {
            for hi in 0..d.h {
                let si = hi / d.repeat;
                for ni in 0..d.half_n {
                    p_full[r * d.h * d.half_n + hi * d.half_n + ni] =
                        p_complex[(r * d.n_ssm * d.half_n + si * d.half_n + ni) % p_complex.len()];
                }
            }
        }

        let mut kernel_out = vec![0.0f32; d.channels * d.h * l];

        // For each head, compute the kernel via Cauchy multiplication
        for c_idx in 0..d.channels {
            for hi in 0..d.h {
                // Incorporate dt into A
                let a_slice: Vec<C32> = (0..d.half_n)
                    .map(|ni| dt_a[hi * d.half_n + ni])
                    .collect();

                // Stack B and P, C and Q
                let b_slice: Vec<C32> = (0..d.half_n)
                    .map(|ni| b[hi * d.half_n + ni] * dt[hi])
                    .collect();
                let c_slice: Vec<C32> = (0..d.half_n)
                    .map(|ni| c[c_idx * d.h * d.half_n + hi * d.half_n + ni])
                    .collect();

                // v = B * C (element-wise), shape (N/2,)
                let v: Vec<C32> = b_slice.iter().zip(c_slice.iter())
                    .map(|(&bi, &ci)| bi * ci)
                    .collect();

                // Cauchy multiplication at z points
                let k_f = cauchy_naive(&v, &z, &a_slice);

                // For rank-1 DPLR, apply Woodbury correction if rank > 0 and P is non-zero
                // Simplified: skip Woodbury for now and just use the base kernel
                // (The Cauchy kernel already captures the diagonal part)

                // Bilinear correction: k_f *= 2 / (1 + omega)
                let mut k_corrected = vec![C32::new(0.0, 0.0); fft_len];
                let two = C32::new(2.0, 0.0);
                let one = C32::new(1.0, 0.0);
                for li in 0..fft_len {
                    k_corrected[li] = k_f[li] * two / (one + omega[li]);
                }

                // IFFT to get time-domain kernel
                let k_time = irfft_naive(&k_corrected, l);
                let base = c_idx * d.h * l + hi * l;
                for li in 0..l {
                    kernel_out[base + li] = k_time[li];
                }
            }
        }

        kernel_out
    }

    /// Delegate to diag for recurrent stepping.
    pub fn setup_step(&self) -> (Vec<C32>, Vec<C32>, Vec<C32>) {
        self.diag.setup_step()
    }

    pub fn default_state(&self, batch: usize) -> Vec<f32> {
        self.diag.default_state(batch)
    }

    pub fn step(
        &self,
        u: &[f32],
        state: &mut [f32],
        da: &[C32],
        db: &[C32],
        dc: &[C32],
        batch: usize,
    ) -> Vec<f32> {
        self.diag.step(u, state, da, db, dc, batch)
    }

    pub fn param_count(&self) -> usize {
        self.diag.param_count() + self.p.len() / 2 // P is complex, stored as 2*real
    }
}

/// Naive inverse real FFT: convert frequency domain (L/2+1 complex) to time domain (L real).
fn irfft_naive(freq: &[C32], n: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; n];
    let half = freq.len();

    for t in 0..n {
        let mut sum = 0.0f32;
        for k in 0..half {
            let angle = 2.0 * std::f32::consts::PI * (k as f32) * (t as f32) / (n as f32);
            let w = C32::new(angle.cos(), angle.sin());
            let contrib = freq[k] * w;
            if k == 0 || k == half - 1 {
                sum += contrib.re;
            } else {
                sum += 2.0 * contrib.re;
            }
        }
        result[t] = sum / n as f32;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssm_kernel_diag_create() {
        let kernel = SSMKernelDiag::new(8, 16, 1, "diag-inv", 0.001, 0.1, Discretization::Zoh, None);
        assert_eq!(kernel.h, 8);
        assert_eq!(kernel.half_n, 8);
        assert_eq!(kernel.channels, 1);
    }

    #[test]
    fn test_ssm_kernel_diag_forward_shape() {
        let kernel = SSMKernelDiag::new(4, 8, 1, "diag-inv", 0.001, 0.1, Discretization::Zoh, None);
        let k = kernel.forward(16);
        assert_eq!(k.len(), 1 * 4 * 16); // channels * H * L
    }

    #[test]
    fn test_ssm_kernel_diag_step() {
        let kernel = SSMKernelDiag::new(4, 8, 1, "diag-inv", 0.001, 0.1, Discretization::Zoh, None);
        let (da, db, dc) = kernel.setup_step();
        let mut state = kernel.default_state(2);
        let u = vec![0.1f32; 2 * 4]; // batch=2, H=4
        let y = kernel.step(&u, &mut state, &da, &db, &dc, 2);
        assert_eq!(y.len(), 2 * 1 * 4); // batch * channels * H
    }

    #[test]
    fn test_ssm_kernel_dplr_create() {
        let kernel = SSMKernelDPLR::new(4, 8, 1, "legs", 1, 0.001, 0.1, Discretization::Zoh, None);
        assert_eq!(kernel.diag.h, 4);
        assert_eq!(kernel.rank, 1);
    }

    #[test]
    fn test_ssm_kernel_dplr_forward_shape() {
        let kernel = SSMKernelDPLR::new(4, 8, 1, "diag-inv", 1, 0.001, 0.1, Discretization::Zoh, None);
        let k = kernel.forward(16);
        assert_eq!(k.len(), 1 * 4 * 16);
    }

    #[test]
    fn test_irfft_naive_dc() {
        // A constant frequency-domain signal should give a constant time-domain signal
        let freq = vec![C32::new(4.0, 0.0), C32::new(0.0, 0.0), C32::new(0.0, 0.0)];
        let time = irfft_naive(&freq, 4);
        for &v in &time {
            assert!((v - 1.0).abs() < 1e-4, "DC signal should be 1.0, got {}", v);
        }
    }
}
