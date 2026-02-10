//! RMSNorm with optional gating for Mamba-2.
//!
//! Implements the gated RMSNorm used in Mamba-2 blocks, where the norm output
//! can be optionally multiplied by SiLU(z) (gate).

/// SiLU (Swish) activation: x * sigmoid(x)
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Gated RMSNorm for Mamba-2.
///
/// When `norm_before_gate` is true: output = RMSNorm(y) * SiLU(z)
/// When `norm_before_gate` is false: output = RMSNorm(y * SiLU(z))
pub struct RMSNormGated {
    /// Learnable scale: shape (d,)
    pub weight: Vec<f32>,
    pub eps: f32,
    pub d: usize,
    /// Group size for group normalization (d / ngroups)
    pub group_size: usize,
    pub norm_before_gate: bool,
}

impl RMSNormGated {
    /// Create a new gated RMSNorm.
    ///
    /// # Arguments
    /// - `d`: dimension to normalize over
    /// - `eps`: epsilon for numerical stability
    /// - `norm_before_gate`: if true, norm then gate; if false, gate then norm
    /// - `group_size`: group size for normalization (usually d / ngroups)
    pub fn new(d: usize, eps: f32, norm_before_gate: bool, group_size: usize) -> Self {
        assert!(group_size > 0, "RMSNormGated: group_size must be > 0");
        assert!(d % group_size == 0, "RMSNormGated: d ({}) must be divisible by group_size ({})", d, group_size);
        Self {
            weight: vec![1.0f32; d],
            eps,
            d,
            group_size,
            norm_before_gate,
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// - `y`: input to normalize, shape (..., d) stored as flat slice
    /// - `z`: optional gate, shape (..., d) — if None, no gating applied
    /// - `n_rows`: number of rows (product of all dims except last)
    ///
    /// # Returns
    /// Normalized (and optionally gated) output, shape (..., d)
    pub fn forward(&self, y: &[f32], z: Option<&[f32]>, n_rows: usize) -> Vec<f32> {
        let d = self.d;
        let mut output = vec![0.0f32; n_rows * d];

        for row in 0..n_rows {
            let y_off = row * d;

            if self.norm_before_gate {
                // Norm then gate: output = RMSNorm(y) * SiLU(z)
                // Normalize each group
                let n_groups = d / self.group_size;
                for g in 0..n_groups {
                    let g_start = g * self.group_size;
                    let g_end = g_start + self.group_size;

                    // Compute RMS for this group
                    let mut ms = 0.0f32;
                    for i in g_start..g_end {
                        let v = y[y_off + i];
                        ms += v * v;
                    }
                    ms /= self.group_size as f32;
                    let inv_rms = 1.0 / (ms + self.eps).sqrt();

                    for i in g_start..g_end {
                        let normed = y[y_off + i] * inv_rms * self.weight[i];
                        output[y_off + i] = if let Some(z_data) = z {
                            normed * silu(z_data[y_off + i])
                        } else {
                            normed
                        };
                    }
                }
            } else {
                // Gate then norm: output = RMSNorm(y * SiLU(z))
                let n_groups = d / self.group_size;
                for g in 0..n_groups {
                    let g_start = g * self.group_size;
                    let g_end = g_start + self.group_size;

                    // Apply gating first
                    let mut gated = vec![0.0f32; self.group_size];
                    for (j, i) in (g_start..g_end).enumerate() {
                        gated[j] = if let Some(z_data) = z {
                            y[y_off + i] * silu(z_data[y_off + i])
                        } else {
                            y[y_off + i]
                        };
                    }

                    // RMS of gated values
                    let mut ms = 0.0f32;
                    for &v in &gated {
                        ms += v * v;
                    }
                    ms /= self.group_size as f32;
                    let inv_rms = 1.0 / (ms + self.eps).sqrt();

                    for (j, i) in (g_start..g_end).enumerate() {
                        output[y_off + i] = gated[j] * inv_rms * self.weight[i];
                    }
                }
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_gated_shape() {
        let norm = RMSNormGated::new(8, 1e-5, true, 8);
        let y = vec![1.0f32; 4 * 8];
        let z = vec![1.0f32; 4 * 8];
        let out = norm.forward(&y, Some(&z), 4);
        assert_eq!(out.len(), 4 * 8);
    }

    #[test]
    fn test_rmsnorm_gated_no_gate() {
        let norm = RMSNormGated::new(4, 1e-6, true, 4);
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let out = norm.forward(&y, None, 1);
        // Should be plain RMSNorm
        let ms = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
        let inv_rms = 1.0 / (ms + 1e-6_f32).sqrt();
        assert!((out[0] - 1.0 * inv_rms).abs() < 1e-4);
        assert!((out[1] - 2.0 * inv_rms).abs() < 1e-4);
    }

    #[test]
    fn test_rmsnorm_gated_with_gate() {
        let norm = RMSNormGated::new(2, 1e-6, true, 2);
        let y = vec![1.0, 1.0];
        let z = vec![0.0, 0.0]; // SiLU(0) = 0
        let out = norm.forward(&y, Some(&z), 1);
        // RMSNorm(1,1) * SiLU(0) = something * 0 = 0
        assert!((out[0]).abs() < 1e-5);
        assert!((out[1]).abs() < 1e-5);
    }

    #[test]
    fn test_rmsnorm_gated_groups() {
        // d=8, group_size=4 => 2 groups, each normalized independently
        let norm = RMSNormGated::new(8, 1e-6, true, 4);
        let mut y = vec![0.0f32; 8];
        y[0..4].copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        y[4..8].copy_from_slice(&[10.0, 10.0, 10.0, 10.0]);
        let out = norm.forward(&y, None, 1);
        // Both groups should normalize to ~1.0 (all same values)
        assert!((out[0] - out[4]).abs() < 1e-4);
    }
}
