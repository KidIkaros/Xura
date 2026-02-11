use smallvec::SmallVec;
use std::fmt;

/// Tensor shape with stack-allocated storage for ≤4 dimensions.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: SmallVec<[usize; 4]>,
}

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        Self {
            dims: SmallVec::from_slice(dims),
        }
    }

    pub fn scalar() -> Self {
        Self {
            dims: SmallVec::new(),
        }
    }

    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn dim(&self, axis: usize) -> Option<usize> {
        self.dims.get(axis).copied()
    }

    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    pub fn contiguous_strides(&self) -> SmallVec<[usize; 4]> {
        let ndim = self.dims.len();
        if ndim == 0 {
            return SmallVec::new();
        }
        let mut strides = SmallVec::from_elem(0usize, ndim);
        strides[ndim - 1] = 1;
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    pub fn resolve_reshape(&self, target: &[isize]) -> Option<Shape> {
        let numel = self.numel();
        let mut inferred_idx = None;
        let mut known_product: usize = 1;

        for (i, &d) in target.iter().enumerate() {
            if d == -1 {
                if inferred_idx.is_some() {
                    return None;
                }
                inferred_idx = Some(i);
            } else if d <= 0 {
                return None;
            } else {
                known_product = known_product.checked_mul(d as usize)?;
            }
        }

        let mut result: SmallVec<[usize; 4]> = target
            .iter()
            .map(|&d| if d == -1 { 0 } else { d as usize })
            .collect();

        if let Some(idx) = inferred_idx {
            if known_product == 0 || !numel.is_multiple_of(known_product) {
                return None;
            }
            result[idx] = numel / known_product;
        }

        let result_shape = Shape { dims: result };
        if result_shape.numel() != numel {
            return None;
        }
        Some(result_shape)
    }

    pub fn transpose(&self) -> Option<Shape> {
        if self.ndim() < 2 {
            return None;
        }
        let mut dims = self.dims.clone();
        let n = dims.len();
        dims.swap(n - 2, n - 1);
        Some(Shape { dims })
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape({:?})", self.dims.as_slice())
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape::new(dims)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape {
            dims: SmallVec::from_vec(dims),
        }
    }
}

macro_rules! impl_shape_from_array {
    ($($n:expr),*) => {
        $(
            impl From<[usize; $n]> for Shape {
                fn from(dims: [usize; $n]) -> Self {
                    Shape::new(&dims)
                }
            }
        )*
    };
}

impl_shape_from_array!(0, 1, 2, 3, 4, 5, 6);
