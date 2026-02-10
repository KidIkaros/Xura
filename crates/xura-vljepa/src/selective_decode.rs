//! Selective Decoding — decode only when the SSM state signals semantic change.
//!
//! Mamba-3's hidden state h_t and trapezoidal prev_bx cache naturally track
//! the rate-of-change of the input signal. We monitor state drift to decide
//! when to invoke the expensive Y-Decoder.

/// Selective decoder configuration.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SelectiveDecodeConfig {
    /// Number of recent states to keep in the window.
    pub window_size: usize,
    /// Drift threshold: decode when ‖h_t - h_{t-k}‖ > threshold.
    pub drift_threshold: f32,
    /// Exponential smoothing factor for drift signal (0..1).
    pub smoothing: f32,
}

impl Default for SelectiveDecodeConfig {
    fn default() -> Self {
        Self {
            window_size: 8,
            drift_threshold: 0.5,
            smoothing: 0.3,
        }
    }
}

/// Monitors SSM hidden state drift to trigger selective decoding.
pub struct SelectiveDecoder {
    pub config: SelectiveDecodeConfig,
    /// Ring buffer of recent state snapshots: (window_size, state_dim)
    state_history: Vec<Vec<f32>>,
    /// Current write position in the ring buffer.
    write_pos: usize,
    /// Number of states stored so far.
    count: usize,
    /// Exponentially smoothed drift value.
    smoothed_drift: f32,
}

impl SelectiveDecoder {
    pub fn new(config: SelectiveDecodeConfig) -> Self {
        let window_size = config.window_size;
        Self {
            config,
            state_history: Vec::with_capacity(window_size),
            write_pos: 0,
            count: 0,
            smoothed_drift: 0.0,
        }
    }

    /// Update with a new SSM state snapshot and return whether to decode.
    ///
    /// # Arguments
    /// - `state`: current SSM hidden state vector (any dimensionality, flattened)
    ///
    /// # Returns
    /// `true` if the state drift exceeds the threshold (should decode).
    pub fn should_decode(&mut self, state: &[f32]) -> bool {
        let ws = self.config.window_size;

        // Compute drift against oldest state in window
        let drift = if self.count > 0 {
            let oldest_idx = if self.count >= ws {
                self.write_pos % ws
            } else {
                0
            };
            let oldest = &self.state_history[oldest_idx];
            l2_distance(state, oldest)
        } else {
            0.0
        };

        // Exponential smoothing
        self.smoothed_drift = self.config.smoothing * drift
            + (1.0 - self.config.smoothing) * self.smoothed_drift;

        // Store state
        if self.state_history.len() < ws {
            self.state_history.push(state.to_vec());
        } else {
            self.state_history[self.write_pos % ws] = state.to_vec();
        }
        self.write_pos = (self.write_pos + 1) % ws;
        self.count += 1;

        self.smoothed_drift > self.config.drift_threshold
    }

    /// Reset the decoder state (e.g., on new video/session).
    pub fn reset(&mut self) {
        self.state_history.clear();
        self.write_pos = 0;
        self.count = 0;
        self.smoothed_drift = 0.0;
    }

    /// Current smoothed drift value.
    pub fn current_drift(&self) -> f32 {
        self.smoothed_drift
    }
}

/// L2 distance between two vectors.
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_decode_on_static_state() {
        let config = SelectiveDecodeConfig {
            window_size: 4,
            drift_threshold: 0.1,
            smoothing: 0.5,
        };
        let mut sd = SelectiveDecoder::new(config);

        let state = vec![1.0, 0.0, 0.0, 0.0];
        // Same state repeatedly should not trigger decode
        for _ in 0..10 {
            let should = sd.should_decode(&state);
            // After initial ramp-up, drift should be 0
            if sd.count > 2 {
                assert!(!should, "static state should not trigger decode");
            }
        }
    }

    #[test]
    fn test_decode_on_state_change() {
        let config = SelectiveDecodeConfig {
            window_size: 4,
            drift_threshold: 0.1,
            smoothing: 0.9, // high smoothing = fast response
        };
        let mut sd = SelectiveDecoder::new(config);

        // Prime with static state
        let state_a = vec![0.0; 8];
        for _ in 0..5 {
            sd.should_decode(&state_a);
        }

        // Big state change
        let state_b = vec![10.0; 8];
        let should = sd.should_decode(&state_b);
        assert!(should, "large state change should trigger decode");
    }

    #[test]
    fn test_reset() {
        let mut sd = SelectiveDecoder::new(SelectiveDecodeConfig::default());
        sd.should_decode(&vec![1.0; 4]);
        sd.should_decode(&vec![2.0; 4]);
        sd.reset();
        assert_eq!(sd.count, 0);
        assert_eq!(sd.smoothed_drift, 0.0);
    }

    #[test]
    fn test_l2_distance() {
        let a = vec![3.0, 0.0];
        let b = vec![0.0, 4.0];
        assert!((l2_distance(&a, &b) - 5.0).abs() < 1e-5);
    }
}
