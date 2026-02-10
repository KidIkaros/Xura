//! Inference cache for Mamba autoregressive decoding.
//!
//! Manages per-layer conv_state and ssm_state for O(1) per-step generation.

use std::collections::HashMap;

/// Per-layer inference state.
#[derive(Clone)]
pub struct LayerState {
    /// Convolution rolling buffer, shape (batch, channels, kernel_size)
    pub conv_state: Vec<f32>,
    /// SSM hidden state, shape (batch, dim, d_state) for Mamba1
    /// or (batch, nheads, headdim, d_state) for Mamba2
    pub ssm_state: Vec<f32>,
}

/// Inference parameters managing state across layers and time steps.
pub struct InferenceParams {
    /// Number of tokens processed so far (0 = prefill, >0 = decode).
    pub seqlen_offset: usize,
    /// Per-layer states, keyed by layer index.
    pub states: HashMap<usize, LayerState>,
}

impl InferenceParams {
    /// Create new inference params (no state allocated yet).
    pub fn new() -> Self {
        Self {
            seqlen_offset: 0,
            states: HashMap::new(),
        }
    }

    /// Get or allocate state for a given layer.
    pub fn get_or_create_state(
        &mut self,
        layer_idx: usize,
        conv_state_size: usize,
        ssm_state_size: usize,
    ) -> &mut LayerState {
        self.states.entry(layer_idx).or_insert_with(|| LayerState {
            conv_state: vec![0.0f32; conv_state_size],
            ssm_state: vec![0.0f32; ssm_state_size],
        })
    }

    /// Get state for a layer (if it exists).
    pub fn get_state(&self, layer_idx: usize) -> Option<&LayerState> {
        self.states.get(&layer_idx)
    }

    /// Get mutable state for a layer (if it exists).
    pub fn get_state_mut(&mut self, layer_idx: usize) -> Option<&mut LayerState> {
        self.states.get_mut(&layer_idx)
    }

    /// Reset all states to zero.
    pub fn reset(&mut self) {
        self.seqlen_offset = 0;
        for state in self.states.values_mut() {
            state.conv_state.fill(0.0);
            state.ssm_state.fill(0.0);
        }
    }

    /// Advance the sequence offset by the given amount.
    pub fn advance(&mut self, n: usize) {
        self.seqlen_offset += n;
    }
}

impl Default for InferenceParams {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_params_create() {
        let mut params = InferenceParams::new();
        assert_eq!(params.seqlen_offset, 0);
        assert!(params.states.is_empty());

        let state = params.get_or_create_state(0, 10, 20);
        assert_eq!(state.conv_state.len(), 10);
        assert_eq!(state.ssm_state.len(), 20);
    }

    #[test]
    fn test_inference_params_reset() {
        let mut params = InferenceParams::new();
        let state = params.get_or_create_state(0, 4, 8);
        state.conv_state[0] = 1.0;
        state.ssm_state[0] = 2.0;
        params.seqlen_offset = 5;

        params.reset();
        assert_eq!(params.seqlen_offset, 0);
        assert_eq!(params.get_state(0).unwrap().conv_state[0], 0.0);
        assert_eq!(params.get_state(0).unwrap().ssm_state[0], 0.0);
    }
}
