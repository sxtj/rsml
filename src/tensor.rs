use std::debug_assert_eq;

pub fn matvec(out: &mut [f32], w: &[f32], x: &[f32], d: usize, n: usize) {
    assert_eq!(w.len(), d * n);
    assert_eq!(x.len(), n);
    assert_eq!(out.len(), d);

    for i in 0..d {
        let row = &w[i * n..(i + 1) * n];
        let mut sum = 0.0f32;
        for j in 0..n {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}

pub fn matmul(out: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(out.len(), m * n);

    out.fill(0.0);

    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            let brow = &b[p * n..(p + 1) * n];
            let orow = &mut out[i * n..(i + 1) * n];
            for j in 0..n {
                orow[j] += aip * brow[j];
            }
        }
    }
}

pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), out.len());

    let n = x.len();
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
}

pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

pub struct TensorView<'a, T> {
    data: &'a [T],
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];

    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    strides
}

impl<'a, T> TensorView<'a, T> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

impl<T> Tensor<T> {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> &mut T {
        let flat = self.flat_index(indices);
        &mut self.data[flat]
    }

    fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.shape.len(),
            "expected {} indices, got {}",
            self.shape.len(),
            indices.len()
        );

        let mut flat_index = 0;

        for i in 0..indices.len() {
            assert!(
                indices[i] < self.shape[i],
                "index {} out of range for axis {} with size {}",
                indices[i],
                i,
                self.shape[i]
            );

            flat_index += indices[i] * self.strides[i];
        }

        flat_index
    }

    pub fn row(&self, i: usize) -> &[T] {
        assert_eq!(self.shape.len(), 2);
        let n = self.shape[1];
        &self.data[i * n..(i + 1) * n]
    }

    pub fn get(&self, indices: &[usize]) -> &T {
        let flat = self.flat_index(indices);
        &self.data[flat]
    }

    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();

        assert_eq!(
            data.len(),
            expected,
            "data length {} does not match shape {:?}",
            data.len(),
            shape
        );

        let strides = compute_strides(&shape);

        Self {
            data,
            shape,
            strides,
        }
    }
}

impl<T: Clone + Default> Tensor<T> {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Self::new(vec![T::default(); n], shape)
    }
}

#[cfg(test)]
mod tests {
    use std::assert_eq;

    use crate::tensor;

    use super::*;

    #[test]
    fn strides_are_row_major() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[5]), vec![1]);
        assert_eq!(compute_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn get_indexes_row_major() {
        let t = Tensor::new((0..6).collect(), vec![2, 3]);

        assert_eq!(*t.get(&[0, 0]), 0);
        assert_eq!(*t.get(&[0, 2]), 2);
        assert_eq!(*t.get(&[1, 0]), 3);
        assert_eq!(*t.get(&[1, 2]), 5);
    }

    #[test]
    #[should_panic(expected = "row out of bounds")]
    fn row_rejects_out_of_bounds() {
        let t = Tensor::new((0..6).collect(), vec![2, 3]);
        t.row(2);
    }

    #[test]
    #[should_panic(expected = "expected 2 indices, got 1")]
    fn get_rejects_wrong_rank() {
        Tensor::new((0..6).collect(), vec![2, 3]).get(&[1]);
    }

    #[test]
    #[should_panic(expected = "out of range for axis 1")]
    fn get_rejects_out_of_range_index() {
        Tensor::new((0..6).collect(), vec![2, 3]).get(&[0, 3]);
    }

    #[test]
    #[should_panic(expected = "does not match shape")]
    fn new_rejects_mismatched_data() {
        Tensor::new(vec![0, 1, 2], vec![2, 3]);
    }

    #[test]
    fn zeros_creates_default_tensor() {
        let t = Tensor::<i32>::zeros(vec![2, 3]);

        assert_eq!(t.as_slice(), &[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn get_mut_updates_value() {
        let mut t = Tensor::new((0..6).collect(), vec![2, 3]);

        *t.get_mut(&[1, 2]) = 42;

        assert_eq!(*t.get_mut(&[1, 2]), 42);
    }

    #[test]
    fn matvec_matches_hand_computation() {
        // W = [[1, 2], [3, 4], [5, 6]]  (3x2), x = [1, 1]
        let w = vec![1., 2., 3., 4., 5., 6.];
        let x = vec![1., 1.];
        let mut out = vec![0.0; 3];

        matvec(&mut out, &w, &x, 3, 2);
        assert_eq!(out, vec![3., 7., 11.]);
    }

    #[test]
    fn matmul_identity_is_noop() {
        let a = vec![1., 2., 3., 4., 5., 6.]; // 2x3
        let ident = vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]; // 3x3
        let mut out = vec![0.0; 6];

        matmul(&mut out, &a, &ident, 2, 3, 3);
        assert_eq!(out, a);
    }

    #[test]
    fn row_returns_correct_slice() {
        let t = Tensor::new((0..6).collect(), vec![2, 3]);

        assert_eq!(t.row(0), &[0, 1, 2]);
        assert_eq!(t.row(1), &[3, 4, 5]);
    }

    #[test]
    fn get_3d_tensor() {
        let t = Tensor::new((0..24).collect(), vec![2, 3, 4]);

        assert_eq!(*t.get(&[0, 0, 0]), 0);
        assert_eq!(*t.get(&[0, 1, 2]), 6);
        assert_eq!(*t.get(&[1, 2, 3]), 23);
    }

    #[test]
    fn meta_data_is_correct() {
        let t = Tensor::<i32>::zeros(vec![2, 3, 4]);

        assert_eq!(t.len(), 24);
        assert_eq!(t.rank(), 3);
        assert_eq!(t.shape(), &[2, 3, 4]);
    }
}
