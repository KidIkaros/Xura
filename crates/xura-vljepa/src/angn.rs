//! Adaptive Neural Gating Network (ANGN) — multiplicative input gate for Mamba-3.
//!
//! Placed between VL-JEPA perception and Mamba-3 state updates, the ANGN
//! computes a per-dimension sigmoid gate that decides what information
//! enters the SSM hidden state:
//!
//!   gate = σ(W_gate · h + b_gate)
//!   h_gated = h ⊙ gate
//!
//! This prevents "memory wash" — where older, still-relevant information
//! is gradually faded out by noisy new input. The gate can:
//!
//! 1. **Selective memory**: Zero out noise before it enters the SSM state
//! 2. **Context reset**: Drop gate → 0 on task switches for clean slate
//! 3. **Signal quality monitoring**: Track EMA of gate activation for
//!    integration with the ConfusionMonitor
//!
//! # Multiplicative vs Additive
//!
//! Multiplicative gating (h ⊙ σ(·)) outperforms additive updates because
//! it can completely discard garbage output (gate → 0), whereas additive
//! always leaks. This is critical for long sessions with noisy input
//! (terminal logs, blank screens, etc.).

use rand::Rng;
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════
// Config
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for the Adaptive Neural Gating Network.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ANGNConfig {
    /// Whether the ANGN is active. When false, gating is identity (pass-through).
    pub enabled: bool,
    /// Hidden dimension that the gate operates on (must match d_model).
    pub d_model: usize,
    /// Number of backbone layers to gate. One gate per layer.
    /// If 0 or greater than actual layers, gates all layers.
    pub n_gates: usize,
    /// EMA smoothing factor for tracking average gate activation (0..1).
    /// Higher = more responsive, lower = smoother.
    pub ema_smoothing: f32,
    /// If the EMA-smoothed average gate activation drops below this
    /// threshold, a context reset is signaled.
    pub context_reset_threshold: f32,
    /// Bias initialization value. Positive = gates start open (pass-through).
    /// Recommended: 1.0–2.0 so the gate defaults to "let everything through"
    /// and learns to close on noise.
    pub bias_init: f32,
}

impl Default for ANGNConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            d_model: 64,
            n_gates: 0,
            ema_smoothing: 0.1,
            context_reset_threshold: 0.2,
            bias_init: 1.0,
        }
    }
}

impl ANGNConfig {
    /// Tiny preset for unit tests.
    pub fn tiny() -> Self {
        Self {
            enabled: true,
            d_model: 64,
            n_gates: 2,
            ema_smoothing: 0.5,
            context_reset_threshold: 0.2,
            bias_init: 1.0,
        }
    }

    /// Small preset for production.
    pub fn small() -> Self {
        Self {
            enabled: true,
            d_model: 1024,
            n_gates: 12,
            ema_smoothing: 0.1,
            context_reset_threshold: 0.15,
            bias_init: 1.5,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Per-layer gate
// ═══════════════════════════════════════════════════════════════════════════

/// A single learned gate for one backbone layer.
///
/// Computes: gate = σ(W · h + b), output = h ⊙ gate
struct LayerGate {
    /// Gate weights: (d_model,) — diagonal gate (element-wise).
    weight: Vec<f32>,
    /// Gate bias: (d_model,).
    bias: Vec<f32>,
    d_model: usize,
}

impl LayerGate {
    fn new(d_model: usize, bias_init: f32) -> Self {
        // Weights are zero-initialized so the bias alone controls the initial
        // gate state. During training, the weights learn input-dependent gating.
        // This is standard practice for gating mechanisms (cf. GRU/LSTM gate init).
        Self {
            weight: vec![0.0f32; d_model],
            bias: vec![bias_init; d_model],
            d_model,
        }
    }

    /// Apply multiplicative gating to hidden states.
    ///
    /// # Arguments
    /// - `hidden`: shape (n, d_model) — pre-mixer hidden states
    /// - `n`: number of token positions (batch * seq_len)
    ///
    /// # Returns
    /// (gated_hidden, avg_gate_activation)
    fn forward(&self, hidden: &[f32], n: usize) -> (Vec<f32>, f32) {
        let dm = self.d_model;
        let mut out = vec![0.0f32; n * dm];
        let mut gate_sum = 0.0f32;

        for i in 0..n {
            for d in 0..dm {
                let idx = i * dm + d;
                // gate_score = σ(w_d * h_d + b_d)
                let pre_act = self.weight[d] * hidden[idx] + self.bias[d];
                let gate = sigmoid(pre_act);
                // Multiplicative gating: h_gated = h * gate
                out[idx] = hidden[idx] * gate;
                gate_sum += gate;
            }
        }

        let avg_gate = if n * dm > 0 {
            gate_sum / (n * dm) as f32
        } else {
            1.0
        };

        (out, avg_gate)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ANGN
// ═══════════════════════════════════════════════════════════════════════════

/// Adaptive Neural Gating Network.
///
/// Contains one learned gate per backbone layer. Each gate independently
/// decides how much of the input signal to pass through to the Mamba-3
/// mixer at that layer.
///
/// Tracks running EMA of gate activations and detects context resets
/// when the average gate drops below a configurable threshold.
pub struct AdaptiveNeuralGate {
    pub config: ANGNConfig,
    /// One gate per backbone layer.
    gates: Vec<LayerGate>,
    /// EMA-smoothed average gate activation (across all layers).
    ema_activation: f32,
    /// Number of times gating has been applied.
    step_count: usize,
    /// Whether a context reset was detected on the last forward pass.
    context_reset_detected: bool,
    /// Per-layer gate activations from the most recent forward pass.
    layer_activations: Vec<f32>,
}

impl AdaptiveNeuralGate {
    /// Create a new ANGN with `n_layers` gates.
    ///
    /// If `config.n_gates` is 0 or > `n_layers`, creates one gate per layer.
    /// Otherwise creates exactly `config.n_gates` gates (extras are identity).
    pub fn new(config: ANGNConfig, n_layers: usize) -> Self {
        let n_gates = if config.n_gates == 0 || config.n_gates > n_layers {
            n_layers
        } else {
            config.n_gates
        };

        let gates = (0..n_gates)
            .map(|_| LayerGate::new(config.d_model, config.bias_init))
            .collect();

        Self {
            config,
            gates,
            ema_activation: 1.0, // Start fully open
            step_count: 0,
            context_reset_detected: false,
            layer_activations: vec![1.0; n_gates],
        }
    }

    /// Apply gating to hidden states before a specific backbone layer.
    ///
    /// # Arguments
    /// - `hidden`: shape (n, d_model) — hidden states before mixer
    /// - `layer_idx`: which backbone layer this is (0-indexed)
    /// - `n`: number of positions (batch * seq_len)
    ///
    /// # Returns
    /// Gated hidden states, same shape as input.
    /// If `layer_idx` has no gate (beyond n_gates), returns input unchanged.
    pub fn gate_layer(&mut self, hidden: &[f32], layer_idx: usize, n: usize) -> Vec<f32> {
        if !self.config.enabled || layer_idx >= self.gates.len() {
            return hidden.to_vec();
        }

        let (gated, avg_activation) = self.gates[layer_idx].forward(hidden, n);

        // Track per-layer activation
        if layer_idx < self.layer_activations.len() {
            self.layer_activations[layer_idx] = avg_activation;
        }

        // Update EMA on the last gated layer
        if layer_idx == self.gates.len() - 1 {
            self.update_ema();
        }

        gated
    }

    /// Update the EMA of average gate activation across all layers.
    fn update_ema(&mut self) {
        let n = self.layer_activations.len();
        if n == 0 {
            return;
        }

        let avg: f32 = self.layer_activations.iter().sum::<f32>() / n as f32;
        let alpha = self.config.ema_smoothing;
        self.ema_activation = alpha * avg + (1.0 - alpha) * self.ema_activation;
        self.step_count += 1;

        // Detect context reset
        self.context_reset_detected = self.ema_activation < self.config.context_reset_threshold;
    }

    /// Current EMA-smoothed gate activation (0..1).
    /// Near 1.0 = most input is passing through.
    /// Near 0.0 = gate is mostly closed (noise rejection or context switch).
    pub fn ema_activation(&self) -> f32 {
        self.ema_activation
    }

    /// Whether a context reset was detected (avg gate below threshold).
    /// This can be used by the agent to clear episodic memory or reset state.
    pub fn context_reset_detected(&self) -> bool {
        self.context_reset_detected
    }

    /// Per-layer gate activations from the most recent pass.
    pub fn layer_activations(&self) -> &[f32] {
        &self.layer_activations
    }

    /// Number of gated layers.
    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    /// Total forward passes (counted at the last gate layer).
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Reset EMA and step counter (e.g., after explicit context switch).
    pub fn reset(&mut self) {
        self.ema_activation = 1.0;
        self.step_count = 0;
        self.context_reset_detected = false;
        for a in &mut self.layer_activations {
            *a = 1.0;
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angn_creation() {
        let config = ANGNConfig::tiny();
        let angn = AdaptiveNeuralGate::new(config, 4);
        assert_eq!(angn.num_gates(), 2);
        assert_eq!(angn.ema_activation(), 1.0);
        assert!(!angn.context_reset_detected());
    }

    #[test]
    fn test_angn_zero_gates_uses_all_layers() {
        let mut config = ANGNConfig::tiny();
        config.n_gates = 0;
        let angn = AdaptiveNeuralGate::new(config, 6);
        assert_eq!(angn.num_gates(), 6);
    }

    #[test]
    fn test_angn_disabled_passthrough() {
        let mut config = ANGNConfig::tiny();
        config.enabled = false;
        let mut angn = AdaptiveNeuralGate::new(config, 2);

        let hidden = vec![1.0f32, 2.0, 3.0, 4.0]; // n=1, d_model=4... wait, d_model is 64
        // Use d_model=64
        let hidden = vec![1.0f32; 2 * 64]; // n=2
        let out = angn.gate_layer(&hidden, 0, 2);
        assert_eq!(out, hidden); // Disabled → identity
    }

    #[test]
    fn test_angn_gating_reduces_magnitude() {
        let config = ANGNConfig {
            enabled: true,
            d_model: 4,
            n_gates: 2,
            ema_smoothing: 0.5,
            context_reset_threshold: 0.2,
            bias_init: -5.0, // Very negative bias → gate ≈ 0
        };
        let mut angn = AdaptiveNeuralGate::new(config, 2);

        let hidden = vec![10.0f32; 2 * 4]; // n=2, d_model=4
        let gated = angn.gate_layer(&hidden, 0, 2);

        // With bias_init = -5.0, gate ≈ σ(-5) ≈ 0.007 → output ≈ 0.07
        let max_val = gated.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_val < 1.0, "gated output should be heavily attenuated, got {}", max_val);
    }

    #[test]
    fn test_angn_positive_bias_passthrough() {
        let config = ANGNConfig {
            enabled: true,
            d_model: 4,
            n_gates: 1,
            ema_smoothing: 0.5,
            context_reset_threshold: 0.2,
            bias_init: 10.0, // Very positive bias → gate ≈ 1
        };
        let mut angn = AdaptiveNeuralGate::new(config, 1);

        let hidden = vec![5.0f32; 1 * 4]; // n=1, d_model=4
        let gated = angn.gate_layer(&hidden, 0, 1);

        // With bias_init = 10.0, gate ≈ σ(10) ≈ 1.0 → output ≈ input
        for (g, h) in gated.iter().zip(hidden.iter()) {
            assert!((g - h).abs() < 0.5, "expected near-passthrough, got {} vs {}", g, h);
        }
    }

    #[test]
    fn test_angn_output_shape() {
        let config = ANGNConfig {
            enabled: true,
            d_model: 8,
            n_gates: 3,
            ema_smoothing: 0.3,
            context_reset_threshold: 0.1,
            bias_init: 1.0,
        };
        let mut angn = AdaptiveNeuralGate::new(config, 3);

        let n = 4; // batch*seq
        let hidden = vec![0.5f32; n * 8];

        for layer in 0..3 {
            let out = angn.gate_layer(&hidden, layer, n);
            assert_eq!(out.len(), hidden.len());
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_angn_ema_tracking() {
        let config = ANGNConfig {
            enabled: true,
            d_model: 4,
            n_gates: 2,
            ema_smoothing: 1.0, // Instant tracking
            context_reset_threshold: 0.2,
            bias_init: -10.0, // Gates nearly closed
        };
        let mut angn = AdaptiveNeuralGate::new(config, 2);

        let hidden = vec![1.0f32; 2 * 4];

        // Gate both layers
        angn.gate_layer(&hidden, 0, 2);
        angn.gate_layer(&hidden, 1, 2); // EMA updates on last gate

        // With very negative bias, activation should be very low
        assert!(angn.ema_activation() < 0.1,
            "expected low activation with bias=-10, got {}", angn.ema_activation());
        assert_eq!(angn.step_count(), 1);
    }

    #[test]
    fn test_angn_context_reset_detection() {
        let config = ANGNConfig {
            enabled: true,
            d_model: 4,
            n_gates: 1,
            ema_smoothing: 1.0,
            context_reset_threshold: 0.5,
            bias_init: -10.0, // Forces gate near 0
        };
        let mut angn = AdaptiveNeuralGate::new(config, 1);

        let hidden = vec![1.0f32; 1 * 4];
        angn.gate_layer(&hidden, 0, 1);

        assert!(angn.context_reset_detected(),
            "should detect context reset with closed gates");
    }

    #[test]
    fn test_angn_no_false_context_reset() {
        let config = ANGNConfig {
            enabled: true,
            d_model: 4,
            n_gates: 1,
            ema_smoothing: 1.0,
            context_reset_threshold: 0.2,
            bias_init: 10.0, // Forces gate near 1
        };
        let mut angn = AdaptiveNeuralGate::new(config, 1);

        let hidden = vec![1.0f32; 1 * 4];
        angn.gate_layer(&hidden, 0, 1);

        assert!(!angn.context_reset_detected(),
            "should not false-trigger context reset with open gates");
    }

    #[test]
    fn test_angn_beyond_n_gates_passthrough() {
        let config = ANGNConfig {
            enabled: true,
            d_model: 4,
            n_gates: 1,
            ema_smoothing: 0.5,
            context_reset_threshold: 0.2,
            bias_init: 1.0,
        };
        let mut angn = AdaptiveNeuralGate::new(config, 3);

        let hidden = vec![1.0f32; 2 * 4];
        // Layer 0 is gated, layers 1 and 2 should be passthrough
        let out1 = angn.gate_layer(&hidden, 1, 2);
        let out2 = angn.gate_layer(&hidden, 2, 2);
        assert_eq!(out1, hidden);
        assert_eq!(out2, hidden);
    }

    #[test]
    fn test_angn_reset() {
        let config = ANGNConfig {
            enabled: true,
            d_model: 4,
            n_gates: 1,
            ema_smoothing: 1.0,
            context_reset_threshold: 0.5,
            bias_init: -10.0,
        };
        let mut angn = AdaptiveNeuralGate::new(config, 1);

        let hidden = vec![1.0f32; 1 * 4];
        angn.gate_layer(&hidden, 0, 1);
        assert!(angn.context_reset_detected());

        angn.reset();
        assert!(!angn.context_reset_detected());
        assert_eq!(angn.ema_activation(), 1.0);
        assert_eq!(angn.step_count(), 0);
    }

    #[test]
    fn test_angn_layer_activations() {
        let config = ANGNConfig {
            enabled: true,
            d_model: 4,
            n_gates: 3,
            ema_smoothing: 0.5,
            context_reset_threshold: 0.1,
            bias_init: 0.0, // σ(0) = 0.5
        };
        let mut angn = AdaptiveNeuralGate::new(config, 3);

        let hidden = vec![0.0f32; 2 * 4]; // zero input
        for layer in 0..3 {
            angn.gate_layer(&hidden, layer, 2);
        }

        let acts = angn.layer_activations();
        assert_eq!(acts.len(), 3);
        // With zero input and bias=0, gate = σ(w*0 + 0) = σ(0) = 0.5
        for &a in acts {
            assert!((a - 0.5).abs() < 0.1,
                "expected activation near 0.5 with zero input and bias=0, got {}", a);
        }
    }

    #[test]
    fn test_angn_multiplicative_zeroes_noise() {
        // Key property: multiplicative gating can completely zero out noise
        let config = ANGNConfig {
            enabled: true,
            d_model: 4,
            n_gates: 1,
            ema_smoothing: 0.5,
            context_reset_threshold: 0.2,
            bias_init: -20.0, // σ(-20) ≈ 2e-9 → effectively 0
        };
        let mut angn = AdaptiveNeuralGate::new(config, 1);

        let noise = vec![1000.0f32; 2 * 4]; // Very loud noise
        let gated = angn.gate_layer(&noise, 0, 2);

        // Even 1000.0 * σ(-20) ≈ 0.000002 → effectively zero
        for &v in &gated {
            assert!(v.abs() < 0.01,
                "multiplicative gate should zero out noise, got {}", v);
        }
    }
}
