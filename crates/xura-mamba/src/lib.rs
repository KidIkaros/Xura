//! # xura-mamba
//!
//! Mamba SSM (Selective State Space Model) architecture for Xura.
//!
//! Implements both Mamba v1 and Mamba-2 from:
//! - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
//! - "Transformers are SSMs" (Dao & Gu, 2024)
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use xura_mamba::{Mamba, Mamba2, MambaLMHeadModel, MambaConfig};
//! use xura_core::Tensor;
//!
//! // Single Mamba block
//! let block = Mamba::new(64, 16, 4, 2); // d_model, d_state, d_conv, expand
//! let input = Tensor::from_f32(&vec![0.1; 2 * 8 * 64], &[2, 8, 64]);
//! let output = block.forward_train(input.as_f32_slice().unwrap(), 2, 8);
//!
//! // Full language model
//! let config = MambaConfig::tiny();
//! let model = MambaLMHeadModel::new(config);
//! let tokens = model.generate(&[0, 1, 2], 10, &Default::default());
//! ```

pub mod cache;
pub mod causal_conv1d;
pub mod config;
pub mod loader;
pub mod mamba;
pub mod mamba2;
pub mod mamba3;
pub mod mixer;
pub mod model;
pub mod norm;
pub mod selective_scan;
pub mod ssd;
pub mod ssd3;

// S4 modules
pub mod complex_utils;
pub mod dplr;
pub mod fft_conv;
pub mod hippo;
pub mod s4_block;
pub mod s4_kernel;

pub use cache::{InferenceParams, LayerState};
pub use causal_conv1d::{causal_conv1d_fn, causal_conv1d_update};
pub use config::{MambaConfig, SsmConfig};
pub use loader::{load_config, load_pretrained, LoadError};
pub use mamba::Mamba;
pub use mamba2::Mamba2;
pub use mamba3::Mamba3;
pub use mixer::{MixerLayer, MixerModel};
pub use model::{MambaLMHeadModel, SamplerConfig};
pub use norm::RMSNormGated;
pub use selective_scan::{selective_scan_ref, selective_state_update, SelectiveScanOutput};
pub use ssd::{mamba2_ssm_step, mamba_chunk_scan_combined, ChunkedScanOutput};
pub use ssd3::{mamba3_scan_combined, mamba3_ssm_step, Mamba3ScanOutput};

// S4 re-exports

pub use complex_utils::C32;
pub use fft_conv::FFTConv;
pub use s4_block::S4Block;
pub use s4_kernel::{Discretization, SSMKernelDPLR, SSMKernelDiag};
