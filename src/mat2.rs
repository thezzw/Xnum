use crate::*;
use crate::vec2::*;
use crate::mat3::*;

#[cfg(not(target_arch = "spirv"))]
use core::fmt;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Creates a 2x2 matrix from two column vectors.
#[inline(always)]
#[must_use]
pub const fn mat2(x_axis: XVec2, y_axis: XVec2) -> XMat2 {
    XMat2::from_cols(x_axis, y_axis)
}

/// A 2x2 column major matrix.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "cuda", repr(align(16)))]
#[repr(C)]
pub struct XMat2 {
    pub x_axis: XVec2,
    pub y_axis: XVec2,
}

impl XMat2 {
    /// A 2x2 matrix with all elements set to `X64::ZERO`.
    pub const ZERO: Self = Self::from_cols(XVec2::ZERO, XVec2::ZERO);

    /// A 2x2 identity matrix, where all diagonal elements are `1`, and all off-diagonal elements are `0`.
    pub const IDENTITY: Self = Self::from_cols(XVec2::X, XVec2::Y);

    /// All NAN:s.
    pub const NAN: Self = Self::from_cols(XVec2::NAN, XVec2::NAN);

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    #[must_use]
    const fn new(m00: X64, m01: X64, m10: X64, m11: X64) -> Self {
        Self {
            x_axis: XVec2::new(m00, m01),
            y_axis: XVec2::new(m10, m11),
        }
    }

    /// Creates a 2x2 matrix from two column vectors.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols(x_axis: XVec2, y_axis: XVec2) -> Self {
        Self { x_axis, y_axis }
    }

    /// Creates a 2x2 matrix from a `[X64; 4]` array stored in column major order.
    /// If your data is stored in row major you will need to `transpose` the returned
    /// matrix.
    #[inline]
    #[must_use]
    pub const fn from_cols_array(m: &[X64; 4]) -> Self {
        Self::new(m[0], m[1], m[2], m[3])
    }

    /// Creates a `[X64; 4]` array storing data in column major order.
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub const fn to_cols_array(&self) -> [X64; 4] {
        [self.x_axis.x, self.x_axis.y, self.y_axis.x, self.y_axis.y]
    }

    /// Creates a 2x2 matrix from a `[[X64; 2]; 2]` 2D array stored in column major order.
    /// If your data is in row major order you will need to `transpose` the returned
    /// matrix.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_2d(m: &[[X64; 2]; 2]) -> Self {
        Self::from_cols(XVec2::from_array(m[0]), XVec2::from_array(m[1]))
    }

    /// Creates a `[[X64; 2]; 2]` 2D array storing data in column major order.
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub const fn to_cols_array_2d(&self) -> [[X64; 2]; 2] {
        [self.x_axis.to_array(), self.y_axis.to_array()]
    }

    /// Creates a 2x2 matrix with its diagonal set to `diagonal` and all other entries set to 0.
    #[doc(alias = "scale")]
    #[inline]
    #[must_use]
    pub const fn from_diagonal(diagonal: XVec2) -> Self {
        Self::new(diagonal.x, X64::ZERO, X64::ZERO, diagonal.y)
    }

    /// Creates a 2x2 matrix containing the combining non-uniform `scale` and rotation of
    /// `angle` (in radians).
    #[inline]
    #[must_use]
    pub fn from_scale_angle(scale: XVec2, angle: X64) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self::new(cos * scale.x, sin * scale.x, -sin * scale.y, cos * scale.y)
    }

    /// Creates a 2x2 matrix containing a rotation of `angle` (in radians).
    #[inline]
    #[must_use]
    pub fn from_angle(angle: X64) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self::new(cos, sin, -sin, cos)
    }

    /// Creates a 2x2 matrix from a 3x3 matrix, discarding the 2nd row and column.
    #[inline]
    #[must_use]
    pub fn from_mat3(m: XMat3) -> Self {
        Self::from_cols(m.x_axis.truncate(), m.y_axis.truncate())
    }

    /// Creates a 2x2 matrix from the first 4 values in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 4 elements long.
    #[inline]
    #[must_use]
    pub const fn from_cols_slice(slice: &[X64]) -> Self {
        Self::new(slice[0], slice[1], slice[2], slice[3])
    }

    /// Writes the columns of `self` to the first 4 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 4 elements long.
    #[inline]
    pub fn write_cols_to_slice(self, slice: &mut [X64]) {
        slice[0] = self.x_axis.x;
        slice[1] = self.x_axis.y;
        slice[2] = self.y_axis.x;
        slice[3] = self.y_axis.y;
    }

    /// Returns the matrix column for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 1.
    #[inline]
    #[must_use]
    pub fn col(&self, index: usize) -> XVec2 {
        match index {
            0 => self.x_axis,
            1 => self.y_axis,
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns a mutable reference to the matrix column for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 1.
    #[inline]
    pub fn col_mut(&mut self, index: usize) -> &mut XVec2 {
        match index {
            0 => &mut self.x_axis,
            1 => &mut self.y_axis,
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns the matrix row for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 1.
    #[inline]
    #[must_use]
    pub fn row(&self, index: usize) -> XVec2 {
        match index {
            0 => XVec2::new(self.x_axis.x, self.y_axis.x),
            1 => XVec2::new(self.x_axis.y, self.y_axis.y),
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.x_axis.is_finite() && self.y_axis.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> bool {
        self.x_axis.is_nan() || self.y_axis.is_nan()
    }

    /// Returns the transpose of `self`.
    #[inline]
    #[must_use]
    pub fn transpose(&self) -> Self {
        Self {
            x_axis: XVec2::new(self.x_axis.x, self.y_axis.x),
            y_axis: XVec2::new(self.x_axis.y, self.y_axis.y),
        }
    }

    /// Returns the determinant of `self`.
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> X64 {
        self.x_axis.x * self.y_axis.y - self.x_axis.y * self.y_axis.x
    }

    /// Returns the inverse of `self`.
    ///
    /// If the matrix is not invertible the returned matrix will be invalid.
    ///
    /// # Panics
    ///
    /// Will panic if the determinant of `self` is zero when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        let inv_det = {
            let det = self.determinant();
            assert!(det != X64::ZERO);
            det.recip()
        };
        Self::new(
            self.y_axis.y * inv_det,
            self.x_axis.y * -inv_det,
            self.y_axis.x * -inv_det,
            self.x_axis.x * inv_det,
        )
    }

    /// Transforms a 2D vector.
    #[inline]
    #[must_use]
    pub fn mul_vec2(&self, rhs: XVec2) -> XVec2 {
        #[allow(clippy::suspicious_operation_groupings)]
        XVec2::new(
            (self.x_axis.x * rhs.x) + (self.y_axis.x * rhs.y),
            (self.x_axis.y * rhs.x) + (self.y_axis.y * rhs.y),
        )
    }

    /// Multiplies two 2x2 matrices.
    #[inline]
    #[must_use]
    pub fn mul_mat2(&self, rhs: &Self) -> Self {
        Self::from_cols(self.mul(rhs.x_axis), self.mul(rhs.y_axis))
    }

    /// Adds two 2x2 matrices.
    #[inline]
    #[must_use]
    pub fn add_mat2(&self, rhs: &Self) -> Self {
        Self::from_cols(self.x_axis.add(rhs.x_axis), self.y_axis.add(rhs.y_axis))
    }

    /// Subtracts two 2x2 matrices.
    #[inline]
    #[must_use]
    pub fn sub_mat2(&self, rhs: &Self) -> Self {
        Self::from_cols(self.x_axis.sub(rhs.x_axis), self.y_axis.sub(rhs.y_axis))
    }

    /// Multiplies a 2x2 matrix by a scalar.
    #[inline]
    #[must_use]
    pub fn mul_scalar(&self, rhs: X64) -> Self {
        Self::from_cols(self.x_axis.mul(rhs), self.y_axis.mul(rhs))
    }

    /// Returns true if the absolute difference of all elements between `self` and `rhs`
    /// is less than or equal to `max_abs_diff`.
    ///
    /// This can be used to compare if two matrices contain similar elements. It works best
    /// when comparing with a known value. The `max_abs_diff` that should be used used
    /// depends on the values being compared against.
    ///
    /// For more see
    /// [comparing floating point numbers](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/).
    #[inline]
    #[must_use]
    pub fn abs_diff_eq(&self, rhs: Self, max_abs_diff: X64) -> bool {
        self.x_axis.abs_diff_eq(rhs.x_axis, max_abs_diff)
            && self.y_axis.abs_diff_eq(rhs.y_axis, max_abs_diff)
    }
}

impl Default for XMat2 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Add<XMat2> for XMat2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.add_mat2(&rhs)
    }
}

impl AddAssign<XMat2> for XMat2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add_mat2(&rhs);
    }
}

impl Sub<XMat2> for XMat2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.sub_mat2(&rhs)
    }
}

impl SubAssign<XMat2> for XMat2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.sub_mat2(&rhs);
    }
}

impl Neg for XMat2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self::from_cols(self.x_axis.neg(), self.y_axis.neg())
    }
}

impl Mul<XMat2> for XMat2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_mat2(&rhs)
    }
}

impl MulAssign<XMat2> for XMat2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_mat2(&rhs);
    }
}

impl Mul<XVec2> for XMat2 {
    type Output = XVec2;
    #[inline]
    fn mul(self, rhs: XVec2) -> Self::Output {
        self.mul_vec2(rhs)
    }
}

impl Mul<XMat2> for X64 {
    type Output = XMat2;
    #[inline]
    fn mul(self, rhs: XMat2) -> Self::Output {
        rhs.mul_scalar(self)
    }
}

impl Mul<X64> for XMat2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: X64) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

impl MulAssign<X64> for XMat2 {
    #[inline]
    fn mul_assign(&mut self, rhs: X64) {
        *self = self.mul_scalar(rhs);
    }
}

impl Sum<Self> for XMat2 {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::ZERO, Self::add)
    }
}

impl<'a> Sum<&'a Self> for XMat2 {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::ZERO, |a, &b| Self::add(a, b))
    }
}

impl Product for XMat2 {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::IDENTITY, Self::mul)
    }
}

impl<'a> Product<&'a Self> for XMat2 {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::IDENTITY, |a, &b| Self::mul(a, b))
    }
}

impl PartialEq for XMat2 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.x_axis.eq(&rhs.x_axis) && self.y_axis.eq(&rhs.y_axis)
    }
}

#[cfg(not(target_arch = "spirv"))]
impl AsRef<[X64; 4]> for XMat2 {
    #[inline]
    fn as_ref(&self) -> &[X64; 4] {
        unsafe { &*(self as *const Self as *const [X64; 4]) }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl AsMut<[X64; 4]> for XMat2 {
    #[inline]
    fn as_mut(&mut self) -> &mut [X64; 4] {
        unsafe { &mut *(self as *mut Self as *mut [X64; 4]) }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl fmt::Debug for XMat2 {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct(stringify!(XMat2))
            .field("x_axis", &self.x_axis)
            .field("y_axis", &self.y_axis)
            .finish()
    }
}

#[cfg(not(target_arch = "spirv"))]
impl fmt::Display for XMat2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.x_axis, self.y_axis)
    }
}

