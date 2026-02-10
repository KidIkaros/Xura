use std::fmt;

/// Data types supported by Xura tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    BF16,
    F32,
    F64,
    I8,
    U8,
    I32,
    I64,
    /// Balanced ternary: {-1, 0, +1}, packed 5 trits per byte
    Ternary,
    /// Quaternary: {-3, -1, +1, +3}, packed 4 values per byte
    Quaternary,
}

impl DType {
    /// Size in bytes of a single element, or None for packed types.
    pub fn element_size(&self) -> Option<usize> {
        match self {
            DType::F16 | DType::BF16 => Some(2),
            DType::F32 => Some(4),
            DType::F64 => Some(8),
            DType::I8 | DType::U8 => Some(1),
            DType::I32 => Some(4),
            DType::I64 => Some(8),
            DType::Ternary | DType::Quaternary => None,
        }
    }

    /// Number of bytes needed to store `n` elements of this dtype.
    pub fn storage_bytes(&self, n: usize) -> usize {
        match self {
            DType::Ternary => (n + 4) / 5,
            DType::Quaternary => (n + 3) / 4,
            other => other.element_size().unwrap() * n,
        }
    }

    pub fn is_float(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }

    pub fn is_integer(&self) -> bool {
        matches!(self, DType::I8 | DType::U8 | DType::I32 | DType::I64)
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, DType::Ternary | DType::Quaternary)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::F32 => write!(f, "f32"),
            DType::F64 => write!(f, "f64"),
            DType::I8 => write!(f, "i8"),
            DType::U8 => write!(f, "u8"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
            DType::Ternary => write!(f, "ternary"),
            DType::Quaternary => write!(f, "quaternary"),
        }
    }
}
