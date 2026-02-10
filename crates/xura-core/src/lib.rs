//! # xura-core
//!
//! Minimal tensor engine and module trait for Xura.
//! CPU-only, no autograd, no ops — just what Mamba and VL-JEPA need.

pub mod dtype;
pub mod device;
pub mod storage;
pub mod shape;
pub mod tensor;
pub mod error;
pub mod module;

pub use dtype::DType;
pub use device::Device;
pub use storage::Storage;
pub use shape::Shape;
pub use tensor::Tensor;
pub use error::KoreError;
pub use module::Module;

pub type Result<T> = std::result::Result<T, KoreError>;
