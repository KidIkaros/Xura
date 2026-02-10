use thiserror::Error;

use crate::{DType, Device};

#[derive(Error, Debug)]
pub enum KoreError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Invalid axis {axis} for tensor with {ndim} dimensions")]
    InvalidAxis { axis: usize, ndim: usize },

    #[error("DType mismatch: expected {expected}, got {got}")]
    DTypeMismatch { expected: DType, got: DType },

    #[error("Device mismatch: expected {expected}, got {got}")]
    DeviceMismatch { expected: Device, got: Device },

    #[error("Cannot reshape tensor of {numel} elements into shape {shape:?}")]
    InvalidReshape { numel: usize, shape: Vec<usize> },

    #[error("Cannot broadcast shapes {a:?} and {b:?}")]
    BroadcastError { a: Vec<usize>, b: Vec<usize> },

    #[error("Index {index} out of bounds for axis {axis} with size {size}")]
    IndexOutOfBounds {
        index: usize,
        axis: usize,
        size: usize,
    },

    #[error("Matmul dimension mismatch: [{m}x{k1}] @ [{k2}x{n}]")]
    MatmulDimMismatch {
        m: usize,
        k1: usize,
        k2: usize,
        n: usize,
    },

    #[error("Operation not supported for dtype {0}")]
    UnsupportedDType(DType),

    #[error("Storage error: {0}")]
    StorageError(String),
}
