use std::fmt;

use smallvec::SmallVec;

use crate::dtype::DType;
use crate::device::Device;
use crate::error::KoreError;
use crate::shape::Shape;
use crate::storage::Storage;
use crate::Result;

/// A multi-dimensional array — the fundamental data structure in Xura.
///
/// Stripped-down version: CPU-only, no autograd, no ops.
/// Tensors support zero-copy views (reshape, transpose share storage).
#[derive(Clone)]
pub struct Tensor {
    storage: Storage,
    shape: Shape,
    strides: SmallVec<[usize; 4]>,
    offset: usize,
}

impl Tensor {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Create a tensor from f32 data with the given shape.
    pub fn from_f32(data: &[f32], shape: &[usize]) -> Self {
        let s = Shape::new(shape);
        assert_eq!(
            s.numel(),
            data.len(),
            "Shape {:?} requires {} elements, got {}",
            shape,
            s.numel(),
            data.len()
        );
        let strides = s.contiguous_strides();
        Self {
            storage: Storage::from_f32(data),
            shape: s,
            strides,
            offset: 0,
        }
    }

    /// Create a tensor from f64 data with the given shape.
    pub fn from_f64(data: &[f64], shape: &[usize]) -> Self {
        let s = Shape::new(shape);
        assert_eq!(s.numel(), data.len());
        let strides = s.contiguous_strides();
        Self {
            storage: Storage::from_f64(data),
            shape: s,
            strides,
            offset: 0,
        }
    }

    /// Create a tensor of zeros with the given shape and dtype.
    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        let s = Shape::new(shape);
        let strides = s.contiguous_strides();
        Self {
            storage: Storage::zeros(dtype, s.numel()),
            shape: s,
            strides,
            offset: 0,
        }
    }

    /// Create a tensor of ones (f32).
    pub fn ones(shape: &[usize]) -> Self {
        let s = Shape::new(shape);
        let numel = s.numel();
        let data: Vec<f32> = vec![1.0; numel];
        Self::from_f32(&data, shape)
    }

    /// Create a tensor with random values from N(0,1).
    pub fn randn(shape: &[usize]) -> Self {
        use rand::Rng;
        let s = Shape::new(shape);
        let numel = s.numel();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..numel)
            .map(|_| {
                let u1: f32 = rng.gen_range(1e-5f32..1.0f32);
                let u2: f32 = rng.gen_range(0.0f32..std::f32::consts::TAU);
                (-2.0f32 * u1.ln()).sqrt() * u2.cos()
            })
            .collect();
        Self::from_f32(&data, shape)
    }

    /// Create a tensor with random values uniformly distributed in [low, high).
    pub fn rand_uniform(shape: &[usize], low: f32, high: f32) -> Self {
        use rand::Rng;
        let s = Shape::new(shape);
        let numel = s.numel();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(low..high)).collect();
        Self::from_f32(&data, shape)
    }

    /// Create a 1-D tensor with values from `start` to `end` (exclusive).
    pub fn arange(start: f32, end: f32, step: f32) -> Self {
        assert!(step != 0.0, "arange: step must be non-zero");
        let mut data = Vec::new();
        let mut v = start;
        if step > 0.0 {
            while v < end {
                data.push(v);
                v += step;
            }
        } else {
            while v > end {
                data.push(v);
                v += step;
            }
        }
        let len = data.len();
        Self::from_f32(&data, &[len])
    }

    /// Create a scalar tensor from a single f32 value.
    pub fn scalar(value: f32) -> Self {
        Self {
            storage: Storage::from_f32(&[value]),
            shape: Shape::scalar(),
            strides: SmallVec::new(),
            offset: 0,
        }
    }

    /// Create a tensor from pre-built Storage and shape.
    pub fn from_storage(storage: Storage, shape: &[usize]) -> Self {
        let s = Shape::new(shape);
        let strides = s.contiguous_strides();
        Self {
            storage,
            shape: s,
            strides,
            offset: 0,
        }
    }

    /// Get a reference to the underlying storage.
    pub fn storage_ref(&self) -> &Storage {
        &self.storage
    }

    // =========================================================================
    // Properties
    // =========================================================================

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Whether this tensor is contiguous in memory (row-major).
    pub fn is_contiguous(&self) -> bool {
        self.strides == self.shape.contiguous_strides() && self.offset == 0
    }

    // =========================================================================
    // Data access
    // =========================================================================

    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        if !self.is_contiguous() {
            return None;
        }
        self.storage.as_f32_slice()
    }

    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        if !self.is_contiguous() {
            return None;
        }
        self.storage.as_f32_slice_mut()
    }

    pub fn get_f32(&self, flat_index: usize) -> Option<f32> {
        let slice = self.storage.as_f32_slice()?;
        let physical = self.flat_to_physical(flat_index)?;
        slice.get(physical).copied()
    }

    fn flat_to_physical(&self, flat_index: usize) -> Option<usize> {
        if self.shape.is_scalar() {
            return if flat_index == 0 {
                Some(self.offset)
            } else {
                None
            };
        }
        if flat_index >= self.numel() {
            return None;
        }
        let mut remaining = flat_index;
        let mut physical = self.offset;
        let contiguous_strides = self.shape.contiguous_strides();
        for (i, &cs) in contiguous_strides.iter().enumerate() {
            let idx = remaining / cs;
            remaining %= cs;
            physical += idx * self.strides[i];
        }
        Some(physical)
    }

    // =========================================================================
    // Shape operations (zero-copy views)
    // =========================================================================

    pub fn reshape(&self, new_shape: &[isize]) -> Result<Tensor> {
        let resolved = self.shape.resolve_reshape(new_shape).ok_or_else(|| {
            KoreError::InvalidReshape {
                numel: self.numel(),
                shape: new_shape.iter().map(|&d| d as usize).collect(),
            }
        })?;
        if !self.is_contiguous() {
            return Err(KoreError::StorageError(
                "Cannot reshape non-contiguous tensor (call .contiguous() first)".into(),
            ));
        }
        let strides = resolved.contiguous_strides();
        Ok(Tensor {
            storage: self.storage.clone(),
            shape: resolved,
            strides,
            offset: self.offset,
        })
    }

    pub fn transpose(&self) -> Result<Tensor> {
        let new_shape = self.shape.transpose().ok_or_else(|| {
            KoreError::InvalidAxis {
                axis: 0,
                ndim: self.ndim(),
            }
        })?;
        let ndim = self.ndim();
        let mut new_strides = self.strides.clone();
        new_strides.swap(ndim - 2, ndim - 1);
        Ok(Tensor {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    pub fn is_cpu(&self) -> bool {
        true
    }

    pub fn is_cuda(&self) -> bool {
        false
    }

    /// Return a contiguous copy of this tensor if it isn't already contiguous.
    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            return self.clone();
        }
        if self.dtype() == DType::F32 {
            let numel = self.numel();
            let mut data = vec![0.0f32; numel];
            for i in 0..numel {
                data[i] = self.get_f32(i)
                    .expect("contiguous: index out of bounds during copy");
            }
            Tensor::from_f32(&data, self.shape.dims())
        } else {
            self.clone()
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={}, dtype={}, device={}, contiguous={})",
            self.shape,
            self.dtype(),
            self.device(),
            self.is_contiguous(),
        )
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(data) = self.as_f32_slice() {
            if self.numel() <= 20 {
                write!(f, "tensor({:?}, shape={})", data, self.shape)
            } else {
                write!(
                    f,
                    "tensor([{:.4}, {:.4}, ..., {:.4}], shape={})",
                    data[0],
                    data[1],
                    data[self.numel() - 1],
                    self.shape
                )
            }
        } else {
            write!(f, "tensor(shape={}, dtype={})", self.shape, self.dtype())
        }
    }
}
