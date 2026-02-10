//! DPLR (Diagonal Plus Low-Rank) initialization for S4 kernels.
//!
//! Constructs the (A, P, B, V) parameterization used by S4 and S4D models.
//! Supports multiple initialization schemes: HiPPO, diagonal-inverse, diagonal-linear, etc.

use std::f32::consts::PI;
use rand::Rng;

use crate::complex_utils::C32;
use crate::hippo;

/// Construct DPLR parameters directly (without HiPPO eigendecomposition).
///
/// # Arguments
/// - `init`: Initialization type: "hippo"/"legs", "real", "lin"/"linear", "inv"/"inverse", "rand"/"random"
/// - `n`: State size (d_state)
/// - `rank`: Rank of low-rank correction
/// - `h`: Number of independent SSM copies (d_model)
/// - `real_scale`: Scaling for real part of A
/// - `imag_scale`: Scaling for imaginary part of A
///
/// # Returns
/// (A, P, B, V) where:
/// - A: shape (H, N/2) complex — diagonal of state matrix
/// - P: shape (rank, H, N/2) complex — low-rank correction
/// - B: shape (H, N/2) complex — input matrix
/// - V: shape (H, N, N/2) complex — change of basis (identity for diagonal inits)
#[allow(clippy::too_many_arguments)]
pub fn dplr(
    init: &str,
    n: usize,
    rank: usize,
    h: usize,
    real_scale: f32,
    imag_scale: f32,
    b_init: &str,
    b_scale: f32,
    p_scale: f32,
) -> (Vec<C32>, Vec<C32>, Vec<C32>, Vec<C32>) {
    let half_n = n / 2;

    // Construct real part of diagonal A (non-negative, will be negated)
    let real_part: Vec<f32> = vec![0.5 * real_scale; h * half_n];

    // Construct imaginary part based on init
    let mut imag_part = vec![0.0f32; h * half_n];

    match init {
        "hippo" | "legs" => {
            // Use HiPPO eigenvalues
            let (w, p_nplr, b_nplr, _v) = hippo::nplr("legs", n, rank, Some(2.0))
                .expect("HiPPO NPLR initialization failed");
            // Broadcast to H copies
            for hi in 0..h {
                for ni in 0..half_n {
                    let idx = hi * half_n + ni;
                    imag_part[idx] = -w[ni % w.len()].im * imag_scale; // positive imag
                }
            }

            // For HiPPO init, use the NPLR B and P
            let mut a = vec![C32::new(0.0, 0.0); h * half_n];
            for hi in 0..h {
                for ni in 0..half_n {
                    let idx = hi * half_n + ni;
                    a[idx] = C32::new(-real_part[idx], -imag_part[idx]);
                }
            }

            // B from NPLR
            let mut b = vec![C32::new(0.0, 0.0); h * half_n];
            for hi in 0..h {
                for ni in 0..half_n {
                    b[hi * half_n + ni] = b_nplr[ni % b_nplr.len()] * b_scale;
                }
            }

            // P from NPLR
            let mut p = vec![C32::new(0.0, 0.0); rank * h * half_n];
            for r in 0..rank {
                for hi in 0..h {
                    for ni in 0..half_n {
                        p[r * h * half_n + hi * half_n + ni] =
                            p_nplr[(r * half_n + ni) % p_nplr.len()] * p_scale;
                    }
                }
            }

            // V = identity (already in diagonal form)
            let mut v = vec![C32::new(0.0, 0.0); h * n * half_n];
            for hi in 0..h {
                for ni in 0..half_n {
                    v[hi * n * half_n + ni * half_n + ni] = C32::new(1.0, 0.0);
                }
            }

            return (a, p, b, v);
        }
        "real" => {
            // S4D-Real: no imaginary part, real eigenvalues -1, -2, ..., -N/2
            for hi in 0..h {
                for ni in 0..half_n {
                    imag_part[hi * half_n + ni] = 0.0;
                }
            }
            // Override real part: 1+arange(N/2)
            let mut real_override = vec![0.0f32; h * half_n];
            for hi in 0..h {
                for ni in 0..half_n {
                    real_override[hi * half_n + ni] = (1.0 + ni as f32) * real_scale;
                }
            }
            let mut a = vec![C32::new(0.0, 0.0); h * half_n];
            for i in 0..h * half_n {
                a[i] = C32::new(-real_override[i], 0.0);
            }
            return build_non_hippo(a, half_n, h, rank, b_init, b_scale, p_scale, n);
        }
        "linear" | "lin" => {
            for hi in 0..h {
                for ni in 0..half_n {
                    imag_part[hi * half_n + ni] = PI * ni as f32 * imag_scale;
                }
            }
        }
        "inverse" | "inv" => {
            for hi in 0..h {
                for ni in 0..half_n {
                    let k = ni as f32;
                    imag_part[hi * half_n + ni] =
                        (1.0 / PI) * (n as f32) * ((n as f32) / (1.0 + 2.0 * k) - 1.0) * imag_scale;
                }
            }
        }
        "random" | "rand" => {
            let mut rng = rand::thread_rng();
            for i in 0..h * half_n {
                imag_part[i] = rng.gen::<f32>().exp() * imag_scale;
            }
        }
        _ => panic!("Unknown DPLR init: {}", init),
    }

    // Construct A
    let mut a = vec![C32::new(0.0, 0.0); h * half_n];
    for i in 0..h * half_n {
        a[i] = C32::new(-real_part[i], -imag_part[i]);
    }

    build_non_hippo(a, half_n, h, rank, b_init, b_scale, p_scale, n)
}

/// Build P, B, V for non-HiPPO initializations.
fn build_non_hippo(
    a: Vec<C32>,
    half_n: usize,
    h: usize,
    rank: usize,
    b_init: &str,
    b_scale: f32,
    p_scale: f32,
    n: usize,
) -> (Vec<C32>, Vec<C32>, Vec<C32>, Vec<C32>) {
    let mut rng = rand::thread_rng();

    // Initialize B
    let mut b = vec![C32::new(0.0, 0.0); h * half_n];
    match b_init {
        "constant" => {
            for v in b.iter_mut() {
                *v = C32::new(b_scale, 0.0);
            }
        }
        "random" => {
            for v in b.iter_mut() {
                *v = C32::new(
                    rng.gen::<f32>() * b_scale,
                    rng.gen::<f32>() * b_scale,
                );
            }
        }
        _ => {
            for v in b.iter_mut() {
                *v = C32::new(b_scale, 0.0);
            }
        }
    }

    // Initialize P (random complex)
    let mut p = vec![C32::new(0.0, 0.0); rank * h * half_n];
    for v in p.iter_mut() {
        *v = C32::new(
            rng.gen::<f32>() * p_scale,
            rng.gen::<f32>() * p_scale,
        );
    }

    // V = identity
    let mut v = vec![C32::new(0.0, 0.0); h * n * half_n];
    for hi in 0..h {
        for ni in 0..half_n {
            v[hi * n * half_n + ni * half_n + ni] = C32::new(1.0, 0.0);
        }
    }

    (a, p, b, v)
}

/// Dispatcher for SSM initialization: handles both "diag-*" and full NPLR inits.
///
/// Returns (A, P, B) flattened for H independent copies.
/// - A: shape (H, N/2) complex
/// - P: shape (rank, H, N/2) complex
/// - B: shape (H, N/2) complex
pub fn ssm_init(
    init: &str,
    n: usize,
    rank: usize,
    h: usize,
) -> (Vec<C32>, Vec<C32>, Vec<C32>) {
    let p_scale = if init.starts_with("diag") { 0.0 } else { 1.0 };

    let actual_init = if init.starts_with("diag-") {
        &init[5..]
    } else if init.starts_with("dplr-") {
        &init[5..]
    } else {
        init
    };

    let (a, p, b, _v) = dplr(
        actual_init,
        n, rank, h,
        1.0,  // real_scale
        1.0,  // imag_scale
        "constant",
        1.0,  // b_scale
        p_scale,
    );

    (a, p, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dplr_hippo_shapes() {
        let (a, p, b, v) = dplr("hippo", 16, 1, 4, 1.0, 1.0, "constant", 1.0, 1.0);
        assert_eq!(a.len(), 4 * 8); // H * N/2
        assert_eq!(p.len(), 1 * 4 * 8); // rank * H * N/2
        assert_eq!(b.len(), 4 * 8);
        assert_eq!(v.len(), 4 * 16 * 8); // H * N * N/2
    }

    #[test]
    fn test_dplr_diag_shapes() {
        let (a, p, b, _v) = dplr("inv", 16, 1, 4, 1.0, 1.0, "constant", 1.0, 0.0);
        assert_eq!(a.len(), 4 * 8);
        assert_eq!(b.len(), 4 * 8);
        // All A should have negative real part
        for &ai in &a {
            assert!(ai.re <= 0.0, "A should have non-positive real: {}", ai.re);
        }
    }

    #[test]
    fn test_dplr_real_init() {
        let (a, _, _, _) = dplr("real", 8, 1, 2, 1.0, 1.0, "constant", 1.0, 0.0);
        // Real init: imag should be 0
        for &ai in &a {
            assert!(ai.im.abs() < 1e-6, "Real init should have 0 imag: {}", ai.im);
        }
    }

    #[test]
    fn test_ssm_init_diag() {
        let (a, p, b) = ssm_init("diag-inv", 16, 1, 4);
        assert_eq!(a.len(), 32);
        // P_scale = 0 for diag, so P should be zero
        for &pi in &p {
            assert!(pi.norm() < 1e-6, "Diag init P should be zero");
        }
    }

    #[test]
    fn test_ssm_init_hippo() {
        let (a, _p, b) = ssm_init("legs", 16, 1, 4);
        assert_eq!(a.len(), 32);
        assert_eq!(b.len(), 32);
    }
}
