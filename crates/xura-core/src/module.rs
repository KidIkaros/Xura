use std::collections::HashMap;

use crate::{Tensor, Result};

/// Base trait for all neural network modules.
pub trait Module: Send + Sync {
    /// Forward pass.
    fn forward(&self, input: &Tensor) -> Result<Tensor>;

    /// Get all trainable parameters.
    fn parameters(&self) -> Vec<&Tensor>;

    /// Get named parameters (for state_dict).
    fn named_parameters(&self) -> Vec<(&str, &Tensor)>;

    /// Set training/eval mode.
    fn train(&mut self, _mode: bool) {}

    /// Whether the module is in training mode.
    fn is_training(&self) -> bool {
        true
    }

    /// Export state dictionary.
    fn state_dict(&self) -> HashMap<String, Tensor> {
        self.named_parameters()
            .into_iter()
            .map(|(name, t)| (name.to_string(), t.clone()))
            .collect()
    }
}
