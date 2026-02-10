//! Tensor operations: arithmetic, reduction, comparison.
//!
//! All operations return new tensors (functional style).
//! In-place variants are suffixed with `_` (e.g., `add_`).

pub mod arithmetic;
pub mod reduction;
pub mod comparison;
pub mod manipulation;

#[cfg(feature = "cuda")]
pub(crate) mod cuda_ops;
