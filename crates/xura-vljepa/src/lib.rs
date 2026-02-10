//! # xura-vljepa
//!
//! Mamba3-JEPA: State-Space World Model for Vision-Language.
//!
//! Replaces the Transformer-based Predictor (Llama-3.2-1B) and Y-Encoder
//! (EmbeddingGemma-300M) from VL-JEPA with Mamba-3 SSM blocks, creating
//! a hybrid architecture that pairs JEPA latent perception with O(N)
//! recurrent memory via complex-valued state dynamics.
//!
//! ## Architecture
//!
//! - **X-Encoder** (ViT): Frozen vision transformer for visual embeddings
//! - **Predictor** (Mamba-3): Maps visual + query → predicted target embedding
//! - **Y-Encoder** (Mamba-3): Encodes target text → target embedding
//! - **Y-Decoder** (Mamba-3 LM): Decodes predicted embedding → text
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use xura_vljepa::{Mamba3Jepa, Mamba3JepaConfig, InferenceMode, InferenceOutput};
//!
//! let config = Mamba3JepaConfig::tiny();
//! let mut model = Mamba3Jepa::new(config.clone());
//!
//! // Inference: image + query → embedding
//! let h = config.vit.image_size;
//! let image = vec![0.1f32; config.vit.in_channels * h * h];
//! let query = vec![1usize, 2, 3];
//! let output = model.infer(&image, &query, h, h, InferenceMode::Embedding);
//! ```

pub mod config;
pub mod loss;
pub mod vit;
pub mod predictor;
pub mod y_encoder;
pub mod y_decoder;
pub mod vljepa;
pub mod selective_decode;
pub mod recursion;
pub mod angn;
pub mod tools;
pub mod agent;
pub mod loader;

pub use config::{
    VitConfig,
    Mamba3PredictorConfig,
    Mamba3TextEncoderConfig,
    Mamba3DecoderConfig,
    Mamba3JepaConfig,
    RecursionConfig,
};
pub use loss::info_nce_loss;
pub use vit::VisionEncoder;
pub use predictor::Mamba3Predictor;
pub use y_encoder::Mamba3TextEncoder;
pub use y_decoder::Mamba3Decoder;
pub use vljepa::{Mamba3Jepa, TrainOutput, InferenceMode, InferenceOutput};
pub use selective_decode::{SelectiveDecoder, SelectiveDecodeConfig};
pub use loader::{load_safetensors, load_vit_weights, LoadError};
pub use recursion::{
    RecursionLayer, ConfusionMonitor, MemoryTool, LocalMemoryTool, StateInjector,
};
pub use angn::{AdaptiveNeuralGate, ANGNConfig};
pub use config::AgentConfig;
pub use tools::{Tool, ToolRegistry, ToolRequest, ToolResult, MemorySearchTool, EchoTool};
pub use agent::{Mamba3Agent, AgentInput, AgentOutput, ToolCallRecord};
