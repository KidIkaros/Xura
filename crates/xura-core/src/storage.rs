use std::sync::Arc;

use crate::{DType, Device, Result, XuraError};

/// Backing storage for tensor data (CPU only, no CUDA).
#[derive(Debug, Clone)]
pub struct Storage {
    data: Arc<Vec<u8>>,
    dtype: DType,
    numel: usize,
}

impl Storage {
    pub fn zeros(dtype: DType, numel: usize) -> Self {
        let nbytes = dtype.storage_bytes(numel);
        Self {
            data: Arc::new(vec![0u8; nbytes]),
            dtype,
            numel,
        }
    }

    pub fn from_bytes(dtype: DType, numel: usize, bytes: Vec<u8>) -> Result<Self> {
        let expected = dtype.storage_bytes(numel);
        if bytes.len() != expected {
            return Err(XuraError::StorageError(format!(
                "Expected {} bytes for {} elements of {}, got {}",
                expected,
                numel,
                dtype,
                bytes.len()
            )));
        }
        Ok(Self {
            data: Arc::new(bytes),
            dtype,
            numel,
        })
    }

    pub fn from_f32(data: &[f32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        Self {
            data: Arc::new(bytes),
            dtype: DType::F32,
            numel: data.len(),
        }
    }

    pub fn from_f64(data: &[f64]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        Self {
            data: Arc::new(bytes),
            dtype: DType::F64,
            numel: data.len(),
        }
    }

    pub fn from_i32(data: &[i32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        Self {
            data: Arc::new(bytes),
            dtype: DType::I32,
            numel: data.len(),
        }
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> Device {
        Device::Cpu
    }

    pub fn numel(&self) -> usize {
        self.numel
    }

    pub fn nbytes(&self) -> usize {
        self.data.len()
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        Arc::make_mut(&mut self.data).as_mut_slice()
    }

    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        if self.dtype != DType::F32 {
            return None;
        }
        Some(bytemuck::cast_slice(self.as_bytes()))
    }

    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        if self.dtype != DType::F32 {
            return None;
        }
        Some(bytemuck::cast_slice_mut(self.as_bytes_mut()))
    }

    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        if self.dtype != DType::F64 {
            return None;
        }
        Some(bytemuck::cast_slice(self.as_bytes()))
    }

    pub fn as_i32_slice(&self) -> Option<&[i32]> {
        if self.dtype != DType::I32 {
            return None;
        }
        Some(bytemuck::cast_slice(self.as_bytes()))
    }

    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.data) == 1
    }

    pub fn is_cpu(&self) -> bool {
        true
    }

    pub fn is_cuda(&self) -> bool {
        false
    }
}
