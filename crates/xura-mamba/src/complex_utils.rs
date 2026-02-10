//! Complex number utilities for S4 kernels.
//!
//! Provides Vandermonde multiplication, Cauchy kernel, and complex↔real conversion
//! helpers used by the S4 and S4D kernel implementations.

use num_complex::Complex;
use std::f32::consts::PI;

/// Type alias for complex f32.
pub type C32 = Complex<f32>;

/// Create a complex number from real and imaginary parts.
#[inline]
pub fn c(re: f32, im: f32) -> C32 {
    C32::new(re, im)
}

/// Concatenate a vector with its complex conjugate: [v, v*].
pub fn conj_pairs(v: &[C32]) -> Vec<C32> {
    let mut out = Vec::with_capacity(v.len() * 2);
    out.extend_from_slice(v);
    for &x in v {
        out.push(x.conj());
    }
    out
}

/// Convert complex slice to interleaved real pairs: [re0, im0, re1, im1, ...].
pub fn c2r(v: &[C32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        out.push(x.re);
        out.push(x.im);
    }
    out
}

/// Convert interleaved real pairs back to complex: [re0, im0, re1, im1, ...] → [c0, c1, ...].
pub fn r2c(v: &[f32]) -> Vec<C32> {
    assert!(v.len() % 2 == 0, "r2c requires even-length input");
    v.chunks_exact(2).map(|pair| c(pair[0], pair[1])).collect()
}

/// Naive log-Vandermonde multiplication (reference implementation).
///
/// Computes: K[l] = Σ_n v[n] * exp(x[n] * l)  for l = 0..L
/// Then returns 2 * Re(K).
///
/// # Arguments
/// - `v`: coefficients, shape (N,) complex
/// - `x`: exponents (dtA), shape (N,) complex
/// - `l_len`: sequence length L
///
/// # Returns
/// Real-valued kernel of length L.
pub fn log_vandermonde_naive(v: &[C32], x: &[C32], l_len: usize) -> Vec<f32> {
    let n = v.len();
    assert_eq!(n, x.len());
    let mut kernel = vec![0.0f32; l_len];

    for l in 0..l_len {
        let mut sum = C32::new(0.0, 0.0);
        for i in 0..n {
            // exp(x[i] * l)
            let exponent = x[i] * (l as f32);
            sum += v[i] * exponent.exp();
        }
        kernel[l] = 2.0 * sum.re;
    }

    kernel
}

/// Naive log-Vandermonde transpose.
///
/// Computes: result[n] = Σ_l u[l] * v[n] * exp(x[n] * l)
///
/// # Arguments
/// - `u`: input sequence, shape (L,) real (converted to complex)
/// - `v`: coefficients, shape (N,) complex
/// - `x`: exponents, shape (N,) complex
/// - `l_len`: sequence length L
///
/// # Returns
/// Complex vector of length N.
pub fn log_vandermonde_transpose_naive(
    u: &[f32],
    v: &[C32],
    x: &[C32],
    l_len: usize,
) -> Vec<C32> {
    let n = v.len();
    let mut result = vec![C32::new(0.0, 0.0); n];

    for i in 0..n {
        let mut sum = C32::new(0.0, 0.0);
        for l in 0..l_len {
            let exponent = x[i] * (l as f32);
            sum += C32::new(u[l], 0.0) * exponent.exp();
        }
        result[i] = v[i] * sum;
    }

    result
}

/// Naive Cauchy kernel multiplication.
///
/// Computes: result[l] = Σ_n v[n] / (z[l] - w[n])
/// where v, w are conjugated (doubled) internally.
///
/// # Arguments
/// - `v`: coefficients, shape (N,) complex (half — will be conjugated)
/// - `z`: evaluation points, shape (L,) complex
/// - `w`: poles, shape (N,) complex (half — will be conjugated)
///
/// # Returns
/// Complex vector of length L.
pub fn cauchy_naive(v: &[C32], z: &[C32], w: &[C32]) -> Vec<C32> {
    let v_full = conj_pairs(v);
    let w_full = conj_pairs(w);
    let n = v_full.len();
    let l = z.len();
    let mut result = vec![C32::new(0.0, 0.0); l];

    for li in 0..l {
        let mut sum = C32::new(0.0, 0.0);
        for ni in 0..n {
            let denom = z[li] - w_full[ni];
            if denom.norm_sqr() > 1e-12 {
                sum += v_full[ni] / denom;
            }
        }
        result[li] = sum;
    }

    result
}

/// Compute bilinear transform nodes for FFT: z = 2(1 - ω) / (1 + ω)
/// where ω = exp(-2πi/L) ^ k for k = 0..L/2+1.
///
/// Returns (omega, z) each of length L/2+1.
pub fn fft_nodes(l_len: usize) -> (Vec<C32>, Vec<C32>) {
    let half_l = l_len / 2 + 1;
    let mut omega = Vec::with_capacity(half_l);
    let mut z = Vec::with_capacity(half_l);

    let base = C32::new(0.0, -2.0 * PI / l_len as f32).exp();
    let mut w = C32::new(1.0, 0.0);

    for _ in 0..half_l {
        omega.push(w);
        let one = C32::new(1.0, 0.0);
        let two = C32::new(2.0, 0.0);
        z.push(two * (one - w) / (one + w));
        w *= base;
    }

    (omega, z)
}

/// Element-wise complex multiply two slices of the same length.
pub fn cmul(a: &[C32], b: &[C32]) -> Vec<C32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}

/// Element-wise complex division.
pub fn cdiv(a: &[C32], b: &[C32]) -> Vec<C32> {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            if y.norm_sqr() > 1e-12 {
                x / y
            } else {
                C32::new(0.0, 0.0)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conj_pairs() {
        let v = vec![c(1.0, 2.0), c(3.0, 4.0)];
        let cp = conj_pairs(&v);
        assert_eq!(cp.len(), 4);
        assert_eq!(cp[2], c(1.0, -2.0));
        assert_eq!(cp[3], c(3.0, -4.0));
    }

    #[test]
    fn test_c2r_r2c_roundtrip() {
        let v = vec![c(1.0, 2.0), c(3.0, -4.0)];
        let flat = c2r(&v);
        assert_eq!(flat, vec![1.0, 2.0, 3.0, -4.0]);
        let back = r2c(&flat);
        assert_eq!(back, v);
    }

    #[test]
    fn test_log_vandermonde_constant() {
        // v = [1+0i], x = [0+0i] → kernel should be [2, 2, 2, ...] (2*Re(1*1))
        let v = vec![c(1.0, 0.0)];
        let x = vec![c(0.0, 0.0)];
        let k = log_vandermonde_naive(&v, &x, 4);
        for &val in &k {
            assert!((val - 2.0).abs() < 1e-5, "got {}", val);
        }
    }

    #[test]
    fn test_log_vandermonde_decay() {
        // v = [1+0i], x = [-0.1+0i] → kernel decays: 2*exp(-0.1*l)
        let v = vec![c(1.0, 0.0)];
        let x = vec![c(-0.1, 0.0)];
        let k = log_vandermonde_naive(&v, &x, 5);
        for l in 0..5 {
            let expected = 2.0 * (-0.1 * l as f32).exp();
            assert!((k[l] - expected).abs() < 1e-4, "l={}: got {}, expected {}", l, k[l], expected);
        }
    }

    #[test]
    fn test_cauchy_simple() {
        // Single pole: v=[1], z=[2+0i], w=[0+0i]
        // Full: v_full=[1, 1], w_full=[0, 0]
        // result = 1/(2-0) + 1/(2-0) = 1
        let v = vec![c(1.0, 0.0)];
        let z = vec![c(2.0, 0.0)];
        let w = vec![c(0.0, 0.0)];
        let r = cauchy_naive(&v, &z, &w);
        assert!((r[0].re - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_fft_nodes_length() {
        let (omega, z) = fft_nodes(16);
        assert_eq!(omega.len(), 9); // 16/2 + 1
        assert_eq!(z.len(), 9);
        // omega[0] should be 1+0i
        assert!((omega[0].re - 1.0).abs() < 1e-5);
        assert!(omega[0].im.abs() < 1e-5);
    }
}
