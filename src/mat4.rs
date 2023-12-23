use crate::*;
use crate::vec3::*;
use crate::vec4::*;
use crate::mat3::*;
use crate::euler::*;
use crate::quat::*;

#[cfg(not(target_arch = "spirv"))]
use core::fmt;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Creates a 4x4 matrix from four column vectors.
#[inline(always)]
#[must_use]
pub const fn mat4(x_axis: Vec4, y_axis: Vec4, z_axis: Vec4, w_axis: Vec4) -> Mat4 {
    Mat4::from_cols(x_axis, y_axis, z_axis, w_axis)
}

/// A 4x4 column major matrix.
///
/// This 4x4 matrix type features convenience methods for creating and using affine transforms and
/// perspective projections. If you are primarily dealing with 3D affine transformations
/// considering using [`DAffine3`](crate::DAffine3) which is faster than a 4x4 matrix
/// for some affine operations.
///
/// Affine transformations including 3D translation, rotation and scale can be created
/// using methods such as [`Self::from_translation()`], [`Self::from_quat()`],
/// [`Self::from_scale()`] and [`Self::from_scale_rotation_translation()`].
///
/// Orthographic projections can be created using the methods [`Self::orthographic_lh()`] for
/// left-handed coordinate systems and [`Self::orthographic_rh()`] for right-handed
/// systems. The resulting matrix is also an affine transformation.
///
/// The [`Self::transform_point3()`] and [`Self::transform_vector3()`] convenience methods
/// are provided for performing affine transformations on 3D vectors and points. These
/// multiply 3D inputs as 4D vectors with an implicit `w` value of `1` for points and `0`
/// for vectors respectively. These methods assume that `Self` contains a valid affine
/// transform.
///
/// Perspective projections can be created using methods such as
/// [`Self::perspective_lh()`], [`Self::perspective_infinite_lh()`] and
/// [`Self::perspective_infinite_reverse_lh()`] for left-handed co-ordinate systems and
/// [`Self::perspective_rh()`], [`Self::perspective_infinite_rh()`] and
/// [`Self::perspective_infinite_reverse_rh()`] for right-handed co-ordinate systems.
///
/// The resulting perspective project can be use to transform 3D vectors as points with
/// perspective correction using the [`Self::project_point3()`] convenience method.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "cuda", repr(align(16)))]
#[repr(C)]
pub struct Mat4 {
    pub x_axis: Vec4,
    pub y_axis: Vec4,
    pub z_axis: Vec4,
    pub w_axis: Vec4,
}

impl Mat4 {
    /// A 4x4 matrix with all elements set to `X64::ZERO`.
    pub const ZERO: Self = Self::from_cols(Vec4::ZERO, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO);

    /// A 4x4 identity matrix, where all diagonal elements are `1`, and all off-diagonal elements are `0`.
    pub const IDENTITY: Self = Self::from_cols(Vec4::X, Vec4::Y, Vec4::Z, Vec4::W);

    /// All NAN:s.
    pub const NAN: Self = Self::from_cols(Vec4::NAN, Vec4::NAN, Vec4::NAN, Vec4::NAN);

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    #[must_use]
    const fn new(
        m00: X64,
        m01: X64,
        m02: X64,
        m03: X64,
        m10: X64,
        m11: X64,
        m12: X64,
        m13: X64,
        m20: X64,
        m21: X64,
        m22: X64,
        m23: X64,
        m30: X64,
        m31: X64,
        m32: X64,
        m33: X64,
    ) -> Self {
        Self {
            x_axis: Vec4::new(m00, m01, m02, m03),
            y_axis: Vec4::new(m10, m11, m12, m13),
            z_axis: Vec4::new(m20, m21, m22, m23),
            w_axis: Vec4::new(m30, m31, m32, m33),
        }
    }

    /// Creates a 4x4 matrix from four column vectors.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols(x_axis: Vec4, y_axis: Vec4, z_axis: Vec4, w_axis: Vec4) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
            w_axis,
        }
    }

    /// Creates a 4x4 matrix from a `[X64; 16]` array stored in column major order.
    /// If your data is stored in row major you will need to `transpose` the returned
    /// matrix.
    #[inline]
    #[must_use]
    pub const fn from_cols_array(m: &[X64; 16]) -> Self {
        Self::new(
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13],
            m[14], m[15],
        )
    }

    /// Creates a `[X64; 16]` array storing data in column major order.
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub const fn to_cols_array(&self) -> [X64; 16] {
        [
            self.x_axis.x,
            self.x_axis.y,
            self.x_axis.z,
            self.x_axis.w,
            self.y_axis.x,
            self.y_axis.y,
            self.y_axis.z,
            self.y_axis.w,
            self.z_axis.x,
            self.z_axis.y,
            self.z_axis.z,
            self.z_axis.w,
            self.w_axis.x,
            self.w_axis.y,
            self.w_axis.z,
            self.w_axis.w,
        ]
    }

    /// Creates a 4x4 matrix from a `[[X64; 4]; 4]` 4D array stored in column major order.
    /// If your data is in row major order you will need to `transpose` the returned
    /// matrix.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_2d(m: &[[X64; 4]; 4]) -> Self {
        Self::from_cols(
            Vec4::from_array(m[0]),
            Vec4::from_array(m[1]),
            Vec4::from_array(m[2]),
            Vec4::from_array(m[3]),
        )
    }

    /// Creates a `[[X64; 4]; 4]` 4D array storing data in column major order.
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub const fn to_cols_array_2d(&self) -> [[X64; 4]; 4] {
        [
            self.x_axis.to_array(),
            self.y_axis.to_array(),
            self.z_axis.to_array(),
            self.w_axis.to_array(),
        ]
    }

    /// Creates a 4x4 matrix with its diagonal set to `diagonal` and all other entries set to 0.
    #[doc(alias = "scale")]
    #[inline]
    #[must_use]
    pub const fn from_diagonal(diagonal: Vec4) -> Self {
        Self::new(
            diagonal.x, X64::ZERO, X64::ZERO, X64::ZERO, X64::ZERO, diagonal.y, X64::ZERO, X64::ZERO, X64::ZERO, X64::ZERO, diagonal.z, X64::ZERO, X64::ZERO,
            X64::ZERO, X64::ZERO, diagonal.w,
        )
    }

    #[inline]
    #[must_use]
    fn quat_to_axes(rotation: Quat) -> (Vec4, Vec4, Vec4) {
        assert!(rotation.is_normalized());

        let (x, y, z, w) = rotation.into();
        let x2 = x + x;
        let y2 = y + y;
        let z2 = z + z;
        let xx = x * x2;
        let xy = x * y2;
        let xz = x * z2;
        let yy = y * y2;
        let yz = y * z2;
        let zz = z * z2;
        let wx = w * x2;
        let wy = w * y2;
        let wz = w * z2;

        let x_axis = Vec4::new(X64::ONE - (yy + zz), xy + wz, xz - wy, X64::ZERO);
        let y_axis = Vec4::new(xy - wz, X64::ONE - (xx + zz), yz + wx, X64::ZERO);
        let z_axis = Vec4::new(xz + wy, yz - wx, X64::ONE - (xx + yy), X64::ZERO);
        (x_axis, y_axis, z_axis)
    }

    /// Creates an affine transformation matrix from the given 3D `scale`, `rotation` and
    /// `translation`.
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    ///
    /// # Panics
    ///
    /// Will panic if `rotation` is not normalized when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn from_scale_rotation_translation(
        scale: Vec3,
        rotation: Quat,
        translation: Vec3,
    ) -> Self {
        let (x_axis, y_axis, z_axis) = Self::quat_to_axes(rotation);
        Self::from_cols(
            x_axis.mul(scale.x),
            y_axis.mul(scale.y),
            z_axis.mul(scale.z),
            Vec4::from((translation, X64::ONE)),
        )
    }

    /// Creates an affine transformation matrix from the given 3D `translation`.
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    ///
    /// # Panics
    ///
    /// Will panic if `rotation` is not normalized when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn from_rotation_translation(rotation: Quat, translation: Vec3) -> Self {
        let (x_axis, y_axis, z_axis) = Self::quat_to_axes(rotation);
        Self::from_cols(x_axis, y_axis, z_axis, Vec4::from((translation, X64::ONE)))
    }

    /// Extracts `scale`, `rotation` and `translation` from `self`. The input matrix is
    /// expected to be a 3D affine transformation matrix otherwise the output will be invalid.
    ///
    /// # Panics
    ///
    /// Will panic if the determinant of `self` is zero or if the resulting scale vector
    /// contains any zero elements when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn to_scale_rotation_translation(&self) -> (Vec3, Quat, Vec3) {
        let det = self.determinant();
        assert!(det != X64::ZERO);

        let scale = Vec3::new(
            self.x_axis.length() * det.signum(),
            self.y_axis.length(),
            self.z_axis.length(),
        );

        assert!(scale.x != 0 && scale.y != 0 && scale.z != 0);

        let inv_scale = scale.recip();

        let rotation = Quat::from_rotation_axes(
            self.x_axis.mul(inv_scale.x).truncate(),
            self.y_axis.mul(inv_scale.y).truncate(),
            self.z_axis.mul(inv_scale.z).truncate(),
        );

        let translation = self.w_axis.truncate();

        (scale, rotation, translation)
    }

    /// Creates an affine transformation matrix from the given `rotation` quaternion.
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    ///
    /// # Panics
    ///
    /// Will panic if `rotation` is not normalized when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn from_quat(rotation: Quat) -> Self {
        let (x_axis, y_axis, z_axis) = Self::quat_to_axes(rotation);
        Self::from_cols(x_axis, y_axis, z_axis, Vec4::W)
    }

    /// Creates an affine transformation matrix from the given 3x3 linear transformation
    /// matrix.
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_mat3(m: Mat3) -> Self {
        Self::from_cols(
            Vec4::from((m.x_axis, X64::ZERO)),
            Vec4::from((m.y_axis, X64::ZERO)),
            Vec4::from((m.z_axis, X64::ZERO)),
            Vec4::W,
        )
    }

    /// Creates an affine transformation matrix from the given 3D `translation`.
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_translation(translation: Vec3) -> Self {
        Self::from_cols(
            Vec4::X,
            Vec4::Y,
            Vec4::Z,
            Vec4::new(translation.x, translation.y, translation.z, X64::ONE),
        )
    }

    /// Creates an affine transformation matrix containing a 3D rotation around a normalized
    /// rotation `axis` of `angle` (in radians).
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    ///
    /// # Panics
    ///
    /// Will panic if `axis` is not normalized when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: Vec3, angle: X64) -> Self {
        assert!(axis.is_normalized());

        let (sin, cos) = angle.sin_cos();
        let axis_sin = axis.mul(sin);
        let axis_sq = axis.mul(axis);
        let omc = X64::ONE - cos;
        let xyomc = axis.x * axis.y * omc;
        let xzomc = axis.x * axis.z * omc;
        let yzomc = axis.y * axis.z * omc;
        Self::from_cols(
            Vec4::new(
                axis_sq.x * omc + cos,
                xyomc + axis_sin.z,
                xzomc - axis_sin.y,
                X64::ZERO,
            ),
            Vec4::new(
                xyomc - axis_sin.z,
                axis_sq.y * omc + cos,
                yzomc + axis_sin.x,
                X64::ZERO,
            ),
            Vec4::new(
                xzomc + axis_sin.y,
                yzomc - axis_sin.x,
                axis_sq.z * omc + cos,
                X64::ZERO,
            ),
            Vec4::W,
        )
    }

    /// Creates a affine transformation matrix containing a rotation from the given euler
    /// rotation sequence and angles (in radians).
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_euler(order: EulerRot, a: X64, b: X64, c: X64) -> Self {
        let quat = Quat::from_euler(order, a, b, c);
        Self::from_quat(quat)
    }

    /// Creates an affine transformation matrix containing a 3D rotation around the x axis of
    /// `angle` (in radians).
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_rotation_x(angle: X64) -> Self {
        let (sina, cosa) = angle.sin_cos();
        Self::from_cols(
            Vec4::X,
            Vec4::new(X64::ZERO, cosa, sina, X64::ZERO),
            Vec4::new(X64::ZERO, -sina, cosa, X64::ZERO),
            Vec4::W,
        )
    }

    /// Creates an affine transformation matrix containing a 3D rotation around the y axis of
    /// `angle` (in radians).
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_rotation_y(angle: X64) -> Self {
        let (sina, cosa) = angle.sin_cos();
        Self::from_cols(
            Vec4::new(cosa, X64::ZERO, -sina, X64::ZERO),
            Vec4::Y,
            Vec4::new(sina, X64::ZERO, cosa, X64::ZERO),
            Vec4::W,
        )
    }

    /// Creates an affine transformation matrix containing a 3D rotation around the z axis of
    /// `angle` (in radians).
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_rotation_z(angle: X64) -> Self {
        let (sina, cosa) = angle.sin_cos();
        Self::from_cols(
            Vec4::new(cosa, sina, X64::ZERO, X64::ZERO),
            Vec4::new(-sina, cosa, X64::ZERO, X64::ZERO),
            Vec4::Z,
            Vec4::W,
        )
    }

    /// Creates an affine transformation matrix containing the given 3D non-uniform `scale`.
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    ///
    /// # Panics
    ///
    /// Will panic if all elements of `scale` are zero when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn from_scale(scale: Vec3) -> Self {
        // Do not panic as long as any component is non-zero
        assert!(scale.x != 0 || scale.y != 0 || scale.z != 0);

        Self::from_cols(
            Vec4::new(scale.x, X64::ZERO, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, scale.y, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, X64::ZERO, scale.z, X64::ZERO),
            Vec4::W,
        )
    }

    /// Creates a 4x4 matrix from the first 16 values in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 16 elements long.
    #[inline]
    #[must_use]
    pub const fn from_cols_slice(slice: &[X64]) -> Self {
        Self::new(
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
            slice[8], slice[9], slice[10], slice[11], slice[12], slice[13], slice[14], slice[15],
        )
    }

    /// Writes the columns of `self` to the first 16 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 16 elements long.
    #[inline]
    pub fn write_cols_to_slice(self, slice: &mut [X64]) {
        slice[0] = self.x_axis.x;
        slice[1] = self.x_axis.y;
        slice[2] = self.x_axis.z;
        slice[3] = self.x_axis.w;
        slice[4] = self.y_axis.x;
        slice[5] = self.y_axis.y;
        slice[6] = self.y_axis.z;
        slice[7] = self.y_axis.w;
        slice[8] = self.z_axis.x;
        slice[9] = self.z_axis.y;
        slice[10] = self.z_axis.z;
        slice[11] = self.z_axis.w;
        slice[12] = self.w_axis.x;
        slice[13] = self.w_axis.y;
        slice[14] = self.w_axis.z;
        slice[15] = self.w_axis.w;
    }

    /// Returns the matrix column for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 3.
    #[inline]
    #[must_use]
    pub fn col(&self, index: usize) -> Vec4 {
        match index {
            0 => self.x_axis,
            1 => self.y_axis,
            2 => self.z_axis,
            3 => self.w_axis,
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns a mutable reference to the matrix column for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 3.
    #[inline]
    pub fn col_mut(&mut self, index: usize) -> &mut Vec4 {
        match index {
            0 => &mut self.x_axis,
            1 => &mut self.y_axis,
            2 => &mut self.z_axis,
            3 => &mut self.w_axis,
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns the matrix row for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 3.
    #[inline]
    #[must_use]
    pub fn row(&self, index: usize) -> Vec4 {
        match index {
            0 => Vec4::new(self.x_axis.x, self.y_axis.x, self.z_axis.x, self.w_axis.x),
            1 => Vec4::new(self.x_axis.y, self.y_axis.y, self.z_axis.y, self.w_axis.y),
            2 => Vec4::new(self.x_axis.z, self.y_axis.z, self.z_axis.z, self.w_axis.z),
            3 => Vec4::new(self.x_axis.w, self.y_axis.w, self.z_axis.w, self.w_axis.w),
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.x_axis.is_finite()
            && self.y_axis.is_finite()
            && self.z_axis.is_finite()
            && self.w_axis.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> bool {
        self.x_axis.is_nan() || self.y_axis.is_nan() || self.z_axis.is_nan() || self.w_axis.is_nan()
    }

    /// Returns the transpose of `self`.
    #[inline]
    #[must_use]
    pub fn transpose(&self) -> Self {
        Self {
            x_axis: Vec4::new(self.x_axis.x, self.y_axis.x, self.z_axis.x, self.w_axis.x),
            y_axis: Vec4::new(self.x_axis.y, self.y_axis.y, self.z_axis.y, self.w_axis.y),
            z_axis: Vec4::new(self.x_axis.z, self.y_axis.z, self.z_axis.z, self.w_axis.z),
            w_axis: Vec4::new(self.x_axis.w, self.y_axis.w, self.z_axis.w, self.w_axis.w),
        }
    }

    /// Returns the determinant of `self`.
    #[must_use]
    pub fn determinant(&self) -> X64 {
        let (m00, m01, m02, m03) = self.x_axis.into();
        let (m10, m11, m12, m13) = self.y_axis.into();
        let (m20, m21, m22, m23) = self.z_axis.into();
        let (m30, m31, m32, m33) = self.w_axis.into();

        let a2323 = m22 * m33 - m23 * m32;
        let a1323 = m21 * m33 - m23 * m31;
        let a1223 = m21 * m32 - m22 * m31;
        let a0323 = m20 * m33 - m23 * m30;
        let a0223 = m20 * m32 - m22 * m30;
        let a0123 = m20 * m31 - m21 * m30;

        m00 * (m11 * a2323 - m12 * a1323 + m13 * a1223)
            - m01 * (m10 * a2323 - m12 * a0323 + m13 * a0223)
            + m02 * (m10 * a1323 - m11 * a0323 + m13 * a0123)
            - m03 * (m10 * a1223 - m11 * a0223 + m12 * a0123)
    }

    /// Returns the inverse of `self`.
    ///
    /// If the matrix is not invertible the returned matrix will be invalid.
    ///
    /// # Panics
    ///
    /// Will panic if the determinant of `self` is zero when `assert` is enabled.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let (m00, m01, m02, m03) = self.x_axis.into();
        let (m10, m11, m12, m13) = self.y_axis.into();
        let (m20, m21, m22, m23) = self.z_axis.into();
        let (m30, m31, m32, m33) = self.w_axis.into();

        let coef00 = m22 * m33 - m32 * m23;
        let coef02 = m12 * m33 - m32 * m13;
        let coef03 = m12 * m23 - m22 * m13;

        let coef04 = m21 * m33 - m31 * m23;
        let coef06 = m11 * m33 - m31 * m13;
        let coef07 = m11 * m23 - m21 * m13;

        let coef08 = m21 * m32 - m31 * m22;
        let coef10 = m11 * m32 - m31 * m12;
        let coef11 = m11 * m22 - m21 * m12;

        let coef12 = m20 * m33 - m30 * m23;
        let coef14 = m10 * m33 - m30 * m13;
        let coef15 = m10 * m23 - m20 * m13;

        let coef16 = m20 * m32 - m30 * m22;
        let coef18 = m10 * m32 - m30 * m12;
        let coef19 = m10 * m22 - m20 * m12;

        let coef20 = m20 * m31 - m30 * m21;
        let coef22 = m10 * m31 - m30 * m11;
        let coef23 = m10 * m21 - m20 * m11;

        let fac0 = Vec4::new(coef00, coef00, coef02, coef03);
        let fac1 = Vec4::new(coef04, coef04, coef06, coef07);
        let fac2 = Vec4::new(coef08, coef08, coef10, coef11);
        let fac3 = Vec4::new(coef12, coef12, coef14, coef15);
        let fac4 = Vec4::new(coef16, coef16, coef18, coef19);
        let fac5 = Vec4::new(coef20, coef20, coef22, coef23);

        let vec0 = Vec4::new(m10, m00, m00, m00);
        let vec1 = Vec4::new(m11, m01, m01, m01);
        let vec2 = Vec4::new(m12, m02, m02, m02);
        let vec3 = Vec4::new(m13, m03, m03, m03);

        let inv0 = vec1.mul(fac0).sub(vec2.mul(fac1)).add(vec3.mul(fac2));
        let inv1 = vec0.mul(fac0).sub(vec2.mul(fac3)).add(vec3.mul(fac4));
        let inv2 = vec0.mul(fac1).sub(vec1.mul(fac3)).add(vec3.mul(fac5));
        let inv3 = vec0.mul(fac2).sub(vec1.mul(fac4)).add(vec2.mul(fac5));

        let sign_a = Vec4::new(X64::ONE, X64::NEG_ONE, X64::ONE, X64::NEG_ONE);
        let sign_b = Vec4::new(X64::NEG_ONE, X64::ONE, X64::NEG_ONE, X64::ONE);

        let inverse = Self::from_cols(
            inv0.mul(sign_a),
            inv1.mul(sign_b),
            inv2.mul(sign_a),
            inv3.mul(sign_b),
        );

        let col0 = Vec4::new(
            inverse.x_axis.x,
            inverse.y_axis.x,
            inverse.z_axis.x,
            inverse.w_axis.x,
        );

        let dot0 = self.x_axis.mul(col0);
        let dot1 = dot0.x + dot0.y + dot0.z + dot0.w;

        assert!(dot1 != X64::ZERO);

        let rcp_det = dot1.recip();
        inverse.mul(rcp_det)
    }

    /// Creates a left-handed view matrix using a camera position, an up direction, and a facing
    /// direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    #[inline]
    #[must_use]
    pub fn look_to_lh(eye: Vec3, dir: Vec3, up: Vec3) -> Self {
        Self::look_to_rh(eye, -dir, up)
    }

    /// Creates a right-handed view matrix using a camera position, an up direction, and a facing
    /// direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    #[inline]
    #[must_use]
    pub fn look_to_rh(eye: Vec3, dir: Vec3, up: Vec3) -> Self {
        let f = dir.normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);

        Self::from_cols(
            Vec4::new(s.x, u.x, -f.x, X64::ZERO),
            Vec4::new(s.y, u.y, -f.y, X64::ZERO),
            Vec4::new(s.z, u.z, -f.z, X64::ZERO),
            Vec4::new(-eye.dot(s), -eye.dot(u), eye.dot(f), X64::ONE),
        )
    }

    /// Creates a left-handed view matrix using a camera position, an up direction, and a focal
    /// point.
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    ///
    /// # Panics
    ///
    /// Will panic if `up` is not normalized when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn look_at_lh(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        assert!(up.is_normalized());
        Self::look_to_lh(eye, center.sub(eye), up)
    }

    /// Creates a right-handed view matrix using a camera position, an up direction, and a focal
    /// point.
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    ///
    /// # Panics
    ///
    /// Will panic if `up` is not normalized when `assert` is enabled.
    #[inline]
    pub fn look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        assert!(up.is_normalized());
        Self::look_to_rh(eye, center.sub(eye), up)
    }

    /// Creates a right-handed perspective projection matrix with [-1,1] depth range.
    /// This is the same as the OpenGL `gluPerspective` function.
    /// See <https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml>
    #[inline]
    #[must_use]
    pub fn perspective_rh_gl(
        fov_y_radians: X64,
        aspect_ratio: X64,
        z_near: X64,
        z_far: X64,
    ) -> Self {
        let inv_length = X64::ONE / (z_near - z_far);
        let f = X64::ONE / ((X64::ONE / 2) * fov_y_radians).tan();
        let a = f / aspect_ratio;
        let b = (z_near + z_far) * inv_length;
        let c = ((X64::ONE * 2) * z_near * z_far) * inv_length;
        Self::from_cols(
            Vec4::new(a, X64::ZERO, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, f, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, X64::ZERO, b, X64::NEG_ONE),
            Vec4::new(X64::ZERO, X64::ZERO, c, X64::ZERO),
        )
    }

    /// Creates a left-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// # Panics
    ///
    /// Will panic if `z_near` or `z_far` are less than or equal to zero when `assert` is
    /// enabled.
    #[inline]
    #[must_use]
    pub fn perspective_lh(fov_y_radians: X64, aspect_ratio: X64, z_near: X64, z_far: X64) -> Self {
        assert!(z_near > X64::ZERO && z_far > X64::ZERO);
        let (sin_fov, cos_fov) = ((X64::ONE / 2) * fov_y_radians).sin_cos();
        let h = cos_fov / sin_fov;
        let w = h / aspect_ratio;
        let r = z_far / (z_far - z_near);
        Self::from_cols(
            Vec4::new(w, X64::ZERO, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, h, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, X64::ZERO, r, X64::ONE),
            Vec4::new(X64::ZERO, X64::ZERO, -r * z_near, X64::ZERO),
        )
    }

    /// Creates a right-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// # Panics
    ///
    /// Will panic if `z_near` or `z_far` are less than or equal to zero when `assert` is
    /// enabled.
    #[inline]
    #[must_use]
    pub fn perspective_rh(fov_y_radians: X64, aspect_ratio: X64, z_near: X64, z_far: X64) -> Self {
        assert!(z_near > X64::ZERO && z_far > X64::ZERO);
        let (sin_fov, cos_fov) = ((X64::ONE / 2) * fov_y_radians).sin_cos();
        let h = cos_fov / sin_fov;
        let w = h / aspect_ratio;
        let r = z_far / (z_near - z_far);
        Self::from_cols(
            Vec4::new(w, X64::ZERO, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, h, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, X64::ZERO, r, X64::NEG_ONE),
            Vec4::new(X64::ZERO, X64::ZERO, r * z_near, X64::ZERO),
        )
    }

    /// Creates an infinite left-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// # Panics
    ///
    /// Will panic if `z_near` is less than or equal to zero when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn perspective_infinite_lh(fov_y_radians: X64, aspect_ratio: X64, z_near: X64) -> Self {
        assert!(z_near > X64::ZERO);
        let (sin_fov, cos_fov) = ((X64::ONE / 2) * fov_y_radians).sin_cos();
        let h = cos_fov / sin_fov;
        let w = h / aspect_ratio;
        Self::from_cols(
            Vec4::new(w, X64::ZERO, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, h, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, X64::ZERO, X64::ONE, X64::ONE),
            Vec4::new(X64::ZERO, X64::ZERO, -z_near, X64::ZERO),
        )
    }

    /// Creates an infinite left-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// # Panics
    ///
    /// Will panic if `z_near` is less than or equal to zero when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn perspective_infinite_reverse_lh(
        fov_y_radians: X64,
        aspect_ratio: X64,
        z_near: X64,
    ) -> Self {
        assert!(z_near > X64::ZERO);
        let (sin_fov, cos_fov) = ((X64::ONE / 2) * fov_y_radians).sin_cos();
        let h = cos_fov / sin_fov;
        let w = h / aspect_ratio;
        Self::from_cols(
            Vec4::new(w, X64::ZERO, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, h, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, X64::ZERO, X64::ZERO, X64::ONE),
            Vec4::new(X64::ZERO, X64::ZERO, z_near, X64::ZERO),
        )
    }

    /// Creates an infinite right-handed perspective projection matrix with
    /// `[0,1]` depth range.
    #[inline]
    #[must_use]
    pub fn perspective_infinite_rh(fov_y_radians: X64, aspect_ratio: X64, z_near: X64) -> Self {
        assert!(z_near > X64::ZERO);
        let f = X64::ONE / ((X64::ONE / 2) * fov_y_radians).tan();
        Self::from_cols(
            Vec4::new(f / aspect_ratio, X64::ZERO, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, f, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, X64::ZERO, X64::NEG_ONE, X64::NEG_ONE),
            Vec4::new(X64::ZERO, X64::ZERO, -z_near, X64::ZERO),
        )
    }

    /// Creates an infinite reverse right-handed perspective projection matrix
    /// with `[0,1]` depth range.
    #[inline]
    #[must_use]
    pub fn perspective_infinite_reverse_rh(
        fov_y_radians: X64,
        aspect_ratio: X64,
        z_near: X64,
    ) -> Self {
        assert!(z_near > X64::ZERO);
        let f = X64::ONE / ((X64::ONE / 2) * fov_y_radians).tan();
        Self::from_cols(
            Vec4::new(f / aspect_ratio, X64::ZERO, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, f, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, X64::ZERO, X64::ZERO, X64::NEG_ONE),
            Vec4::new(X64::ZERO, X64::ZERO, z_near, X64::ZERO),
        )
    }

    /// Creates a right-handed orthographic projection matrix with `[-1,1]` depth
    /// range.  This is the same as the OpenGL `glOrtho` function in OpenGL.
    /// See
    /// <https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glOrtho.xml>
    #[inline]
    #[must_use]
    pub fn orthographic_rh_gl(
        left: X64,
        right: X64,
        bottom: X64,
        top: X64,
        near: X64,
        far: X64,
    ) -> Self {
        let a = (X64::ONE * 2) / (right - left);
        let b = (X64::ONE * 2) / (top - bottom);
        let c = -(X64::ONE * 2) / (far - near);
        let tx = -(right + left) / (right - left);
        let ty = -(top + bottom) / (top - bottom);
        let tz = -(far + near) / (far - near);

        Self::from_cols(
            Vec4::new(a, X64::ZERO, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, b, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, X64::ZERO, c, X64::ZERO),
            Vec4::new(tx, ty, tz, X64::ONE),
        )
    }

    /// Creates a left-handed orthographic projection matrix with `[0,1]` depth range.
    #[inline]
    #[must_use]
    pub fn orthographic_lh(
        left: X64,
        right: X64,
        bottom: X64,
        top: X64,
        near: X64,
        far: X64,
    ) -> Self {
        let rcp_width = X64::ONE / (right - left);
        let rcp_height = X64::ONE / (top - bottom);
        let r = X64::ONE / (far - near);
        Self::from_cols(
            Vec4::new(rcp_width + rcp_width, X64::ZERO, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, rcp_height + rcp_height, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, X64::ZERO, r, X64::ZERO),
            Vec4::new(
                -(left + right) * rcp_width,
                -(top + bottom) * rcp_height,
                -r * near,
                X64::ONE,
            ),
        )
    }

    /// Creates a right-handed orthographic projection matrix with `[0,1]` depth range.
    #[inline]
    #[must_use]
    pub fn orthographic_rh(
        left: X64,
        right: X64,
        bottom: X64,
        top: X64,
        near: X64,
        far: X64,
    ) -> Self {
        let rcp_width = X64::ONE / (right - left);
        let rcp_height = X64::ONE / (top - bottom);
        let r = X64::ONE / (near - far);
        Self::from_cols(
            Vec4::new(rcp_width + rcp_width, X64::ZERO, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, rcp_height + rcp_height, X64::ZERO, X64::ZERO),
            Vec4::new(X64::ZERO, X64::ZERO, r, X64::ZERO),
            Vec4::new(
                -(left + right) * rcp_width,
                -(top + bottom) * rcp_height,
                r * near,
                X64::ONE,
            ),
        )
    }

    /// Transforms the given 3D vector as a point, applying perspective correction.
    ///
    /// This is the equivalent of multiplying the 3D vector as a 4D vector where `w` is `X64::ONE`.
    /// The perspective divide is performed meaning the resulting 3D vector is divided by `w`.
    ///
    /// This method assumes that `self` contains a projective transform.
    #[inline]
    #[must_use]
    pub fn project_point3(&self, rhs: Vec3) -> Vec3 {
        let mut res = self.x_axis.mul(rhs.x);
        res = self.y_axis.mul(rhs.y).add(res);
        res = self.z_axis.mul(rhs.z).add(res);
        res = self.w_axis.add(res);
        res = res.mul(res.wwww().recip());
        res.truncate()
    }

    /// Transforms the given 3D vector as a point.
    ///
    /// This is the equivalent of multiplying the 3D vector as a 4D vector where `w` is
    /// `X64::ONE`.
    ///
    /// This method assumes that `self` contains a valid affine transform. It does not perform
    /// a persective divide, if `self` contains a perspective transform, or if you are unsure,
    /// the [`Self::project_point3()`] method should be used instead.
    ///
    /// # Panics
    ///
    /// Will panic if the 3rd row of `self` is not `(0, 0, 0, 1)` when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn transform_point3(&self, rhs: Vec3) -> Vec3 {
        assert!(self.row(3).abs_diff_eq(Vec4::W, X64::DELTA << 3));
        let mut res = self.x_axis.mul(rhs.x);
        res = self.y_axis.mul(rhs.y).add(res);
        res = self.z_axis.mul(rhs.z).add(res);
        res = self.w_axis.add(res);
        res.truncate()
    }

    /// Transforms the give 3D vector as a direction.
    ///
    /// This is the equivalent of multiplying the 3D vector as a 4D vector where `w` is
    /// `X64::ZERO`.
    ///
    /// This method assumes that `self` contains a valid affine transform.
    ///
    /// # Panics
    ///
    /// Will panic if the 3rd row of `self` is not `(0, 0, 0, 1)` when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn transform_vector3(&self, rhs: Vec3) -> Vec3 {
        assert!(self.row(3).abs_diff_eq(Vec4::W, X64::DELTA << 3));
        let mut res = self.x_axis.mul(rhs.x);
        res = self.y_axis.mul(rhs.y).add(res);
        res = self.z_axis.mul(rhs.z).add(res);
        res.truncate()
    }

    /// Transforms a 4D vector.
    #[inline]
    #[must_use]
    pub fn mul_vec4(&self, rhs: Vec4) -> Vec4 {
        let mut res = self.x_axis.mul(rhs.x);
        res = res.add(self.y_axis.mul(rhs.y));
        res = res.add(self.z_axis.mul(rhs.z));
        res = res.add(self.w_axis.mul(rhs.w));
        res
    }

    /// Multiplies two 4x4 matrices.
    #[inline]
    #[must_use]
    pub fn mul_mat4(&self, rhs: &Self) -> Self {
        Self::from_cols(
            self.mul(rhs.x_axis),
            self.mul(rhs.y_axis),
            self.mul(rhs.z_axis),
            self.mul(rhs.w_axis),
        )
    }

    /// Adds two 4x4 matrices.
    #[inline]
    #[must_use]
    pub fn add_mat4(&self, rhs: &Self) -> Self {
        Self::from_cols(
            self.x_axis.add(rhs.x_axis),
            self.y_axis.add(rhs.y_axis),
            self.z_axis.add(rhs.z_axis),
            self.w_axis.add(rhs.w_axis),
        )
    }

    /// Subtracts two 4x4 matrices.
    #[inline]
    #[must_use]
    pub fn sub_mat4(&self, rhs: &Self) -> Self {
        Self::from_cols(
            self.x_axis.sub(rhs.x_axis),
            self.y_axis.sub(rhs.y_axis),
            self.z_axis.sub(rhs.z_axis),
            self.w_axis.sub(rhs.w_axis),
        )
    }

    /// Multiplies a 4x4 matrix by a scalar.
    #[inline]
    #[must_use]
    pub fn mul_scalar(&self, rhs: X64) -> Self {
        Self::from_cols(
            self.x_axis.mul(rhs),
            self.y_axis.mul(rhs),
            self.z_axis.mul(rhs),
            self.w_axis.mul(rhs),
        )
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
            && self.z_axis.abs_diff_eq(rhs.z_axis, max_abs_diff)
            && self.w_axis.abs_diff_eq(rhs.w_axis, max_abs_diff)
    }
}

impl Default for Mat4 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Add<Mat4> for Mat4 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.add_mat4(&rhs)
    }
}

impl AddAssign<Mat4> for Mat4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add_mat4(&rhs);
    }
}

impl Sub<Mat4> for Mat4 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.sub_mat4(&rhs)
    }
}

impl SubAssign<Mat4> for Mat4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.sub_mat4(&rhs);
    }
}

impl Neg for Mat4 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self::from_cols(
            self.x_axis.neg(),
            self.y_axis.neg(),
            self.z_axis.neg(),
            self.w_axis.neg(),
        )
    }
}

impl Mul<Mat4> for Mat4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_mat4(&rhs)
    }
}

impl MulAssign<Mat4> for Mat4 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_mat4(&rhs);
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline]
    fn mul(self, rhs: Vec4) -> Self::Output {
        self.mul_vec4(rhs)
    }
}

impl Mul<Mat4> for X64 {
    type Output = Mat4;
    #[inline]
    fn mul(self, rhs: Mat4) -> Self::Output {
        rhs.mul_scalar(self)
    }
}

impl Mul<X64> for Mat4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: X64) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

impl MulAssign<X64> for Mat4 {
    #[inline]
    fn mul_assign(&mut self, rhs: X64) {
        *self = self.mul_scalar(rhs);
    }
}

impl Sum<Self> for Mat4 {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::ZERO, Self::add)
    }
}

impl<'a> Sum<&'a Self> for Mat4 {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::ZERO, |a, &b| Self::add(a, b))
    }
}

impl Product for Mat4 {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::IDENTITY, Self::mul)
    }
}

impl<'a> Product<&'a Self> for Mat4 {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::IDENTITY, |a, &b| Self::mul(a, b))
    }
}

impl PartialEq for Mat4 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.x_axis.eq(&rhs.x_axis)
            && self.y_axis.eq(&rhs.y_axis)
            && self.z_axis.eq(&rhs.z_axis)
            && self.w_axis.eq(&rhs.w_axis)
    }
}

#[cfg(not(target_arch = "spirv"))]
impl AsRef<[X64; 16]> for Mat4 {
    #[inline]
    fn as_ref(&self) -> &[X64; 16] {
        unsafe { &*(self as *const Self as *const [X64; 16]) }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl AsMut<[X64; 16]> for Mat4 {
    #[inline]
    fn as_mut(&mut self) -> &mut [X64; 16] {
        unsafe { &mut *(self as *mut Self as *mut [X64; 16]) }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl fmt::Debug for Mat4 {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct(stringify!(Mat4))
            .field("x_axis", &self.x_axis)
            .field("y_axis", &self.y_axis)
            .field("z_axis", &self.z_axis)
            .field("w_axis", &self.w_axis)
            .finish()
    }
}

#[cfg(not(target_arch = "spirv"))]
impl fmt::Display for Mat4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}, {}, {}, {}]",
            self.x_axis, self.y_axis, self.z_axis, self.w_axis
        )
    }
}
