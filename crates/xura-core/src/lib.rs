//! # xura-core
//!
//! Minimal tensor engine and module trait for Xura.
//! CPU-only, no autograd, no ops — just what Mamba and VL-JEPA need.

pub mod device;
pub mod dtype;
pub mod error;
pub mod module;
pub mod shape;
pub mod storage;
pub mod tensor;

pub use device::Device;
pub use dtype::DType;
pub use error::XuraError;
pub use module::Module;
pub use shape::Shape;
pub use storage::Storage;
pub use tensor::Tensor;

pub type Result<T> = std::result::Result<T, XuraError>;
