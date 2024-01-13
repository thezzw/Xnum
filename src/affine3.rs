use crate::*;
use crate::mat3::*;
use crate::mat4::*;
use crate::vec3::*;
use crate::quat::*;

use core::ops::{Deref, DerefMut, Mul, MulAssign};

/// A 3D affine transform, which can represent translation, rotation, scaling and shear.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct XAffine3 {
    pub matrix3: XMat3,
    pub translation: XVec3,
}

impl XAffine3 {
    /// The degenerate zero transform.
    ///
    /// This transforms any finite vector and point to zero.
    /// The zero transform is non-invertible.
    pub const ZERO: Self = Self {
        matrix3: XMat3::ZERO,
        translation: XVec3::ZERO,
    };

    /// The identity transform.
    ///
    /// Multiplying a vector with this returns the same vector.
    pub const IDENTITY: Self = Self {
        matrix3: XMat3::IDENTITY,
        translation: XVec3::ZERO,
    };

    /// All NAN:s.
    pub const NAN: Self = Self {
        matrix3: XMat3::NAN,
        translation: XVec3::NAN,
    };

    /// Creates an affine transform from three column vectors.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols(x_axis: XVec3, y_axis: XVec3, z_axis: XVec3, w_axis: XVec3) -> Self {
        Self {
            matrix3: XMat3::from_cols(x_axis, y_axis, z_axis),
            translation: w_axis,
        }
    }

    /// Creates an affine transform from a `[X64; 12]` array stored in column major order.
    #[inline]
    #[must_use]
    pub fn from_cols_array(m: &[X64; 12]) -> Self {
        Self {
            matrix3: XMat3::from_cols_slice(&m[0..9]),
            translation: XVec3::from_slice(&m[9..12]),
        }
    }

    /// Creates a `[X64; 12]` array storing data in column major order.
    #[inline]
    #[must_use]
    pub fn to_cols_array(&self) -> [X64; 12] {
        let x = &self.matrix3.x_axis;
        let y = &self.matrix3.y_axis;
        let z = &self.matrix3.z_axis;
        let w = &self.translation;
        [x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z, w.x, w.y, w.z]
    }

    /// Creates an affine transform from a `[[X64; 3]; 4]`
    /// 3D array stored in column major order.
    /// If your data is in row major order you will need to `transpose` the returned
    /// matrix.
    #[inline]
    #[must_use]
    pub fn from_cols_array_2d(m: &[[X64; 3]; 4]) -> Self {
        Self {
            matrix3: XMat3::from_cols(m[0].into(), m[1].into(), m[2].into()),
            translation: m[3].into(),
        }
    }

    /// Creates a `[[X64; 3]; 4]` 3D array storing data in
    /// column major order.
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub fn to_cols_array_2d(&self) -> [[X64; 3]; 4] {
        [
            self.matrix3.x_axis.into(),
            self.matrix3.y_axis.into(),
            self.matrix3.z_axis.into(),
            self.translation.into(),
        ]
    }

    /// Creates an affine transform from the first 12 values in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 12 elements long.
    #[inline]
    #[must_use]
    pub fn from_cols_slice(slice: &[X64]) -> Self {
        Self {
            matrix3: XMat3::from_cols_slice(&slice[0..9]),
            translation: XVec3::from_slice(&slice[9..12]),
        }
    }

    /// Writes the columns of `self` to the first 12 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 12 elements long.
    #[inline]
    pub fn write_cols_to_slice(self, slice: &mut [X64]) {
        self.matrix3.write_cols_to_slice(&mut slice[0..9]);
        self.translation.write_to_slice(&mut slice[9..12]);
    }

    /// Creates an affine transform that changes scale.
    /// Note that if any scale is zero the transform will be non-invertible.
    #[inline]
    #[must_use]
    pub fn from_scale(scale: XVec3) -> Self {
        Self {
            matrix3: XMat3::from_diagonal(scale),
            translation: XVec3::ZERO,
        }
    }
    /// Creates an affine transform from the given `rotation` quaternion.
    #[inline]
    #[must_use]
    pub fn from_quat(rotation: XQuat) -> Self {
        Self {
            matrix3: XMat3::from_quat(rotation),
            translation: XVec3::ZERO,
        }
    }

    /// Creates an affine transform containing a 3D rotation around a normalized
    /// rotation `axis` of `angle` (in radians).
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: XVec3, angle: X64) -> Self {
        Self {
            matrix3: XMat3::from_axis_angle(axis, angle),
            translation: XVec3::ZERO,
        }
    }

    /// Creates an affine transform containing a 3D rotation around the x axis of
    /// `angle` (in radians).
    #[inline]
    #[must_use]
    pub fn from_rotation_x(angle: X64) -> Self {
        Self {
            matrix3: XMat3::from_rotation_x(angle),
            translation: XVec3::ZERO,
        }
    }

    /// Creates an affine transform containing a 3D rotation around the y axis of
    /// `angle` (in radians).
    #[inline]
    #[must_use]
    pub fn from_rotation_y(angle: X64) -> Self {
        Self {
            matrix3: XMat3::from_rotation_y(angle),
            translation: XVec3::ZERO,
        }
    }

    /// Creates an affine transform containing a 3D rotation around the z axis of
    /// `angle` (in radians).
    #[inline]
    #[must_use]
    pub fn from_rotation_z(angle: X64) -> Self {
        Self {
            matrix3: XMat3::from_rotation_z(angle),
            translation: XVec3::ZERO,
        }
    }

    /// Creates an affine transformation from the given 3D `translation`.
    #[inline]
    #[must_use]
    pub fn from_translation(translation: XVec3) -> Self {
        #[allow(clippy::useless_conversion)]
        Self {
            matrix3: XMat3::IDENTITY,
            translation: translation.into(),
        }
    }

    /// Creates an affine transform from a 3x3 matrix (expressing scale, shear and
    /// rotation)
    #[inline]
    #[must_use]
    pub fn from_mat3(mat3: XMat3) -> Self {
        #[allow(clippy::useless_conversion)]
        Self {
            matrix3: mat3.into(),
            translation: XVec3::ZERO,
        }
    }

    /// Creates an affine transform from a 3x3 matrix (expressing scale, shear and rotation)
    /// and a translation vector.
    ///
    /// Equivalent to `XAffine3::from_translation(translation) * XAffine3::from_mat3(mat3)`
    #[inline]
    #[must_use]
    pub fn from_mat3_translation(mat3: XMat3, translation: XVec3) -> Self {
        #[allow(clippy::useless_conversion)]
        Self {
            matrix3: mat3.into(),
            translation: translation.into(),
        }
    }

    /// Creates an affine transform from the given 3D `scale`, `rotation` and
    /// `translation`.
    ///
    /// Equivalent to `XAffine3::from_translation(translation) *
    /// XAffine3::from_quat(rotation) * XAffine3::from_scale(scale)`
    #[inline]
    #[must_use]
    pub fn from_scale_rotation_translation(
        scale: XVec3,
        rotation: XQuat,
        translation: XVec3,
    ) -> Self {
        let rotation = XMat3::from_quat(rotation);
        #[allow(clippy::useless_conversion)]
        Self {
            matrix3: XMat3::from_cols(
                rotation.x_axis * scale.x,
                rotation.y_axis * scale.y,
                rotation.z_axis * scale.z,
            ),
            translation: translation.into(),
        }
    }

    /// Creates an affine transform from the given 3D `rotation` and `translation`.
    ///
    /// Equivalent to `XAffine3::from_translation(translation) * XAffine3::from_quat(rotation)`
    #[inline]
    #[must_use]
    pub fn from_rotation_translation(rotation: XQuat, translation: XVec3) -> Self {
        #[allow(clippy::useless_conversion)]
        Self {
            matrix3: XMat3::from_quat(rotation),
            translation: translation.into(),
        }
    }

    /// The given `XMat4` must be an affine transform,
    /// i.e. contain no perspective transform.
    #[inline]
    #[must_use]
    pub fn from_mat4(m: XMat4) -> Self {
        Self {
            matrix3: XMat3::from_cols(
                XVec3::from_vec4(m.x_axis),
                XVec3::from_vec4(m.y_axis),
                XVec3::from_vec4(m.z_axis),
            ),
            translation: XVec3::from_vec4(m.w_axis),
        }
    }

    /// Extracts `scale`, `rotation` and `translation` from `self`.
    ///
    /// The transform is expected to be non-degenerate and without shearing, or the output
    /// will be invalid.
    ///
    /// # Panics
    ///
    /// Will panic if the determinant `self.matrix3` is zero or if the resulting scale
    /// vector contains any zero elements when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn to_scale_rotation_translation(&self) -> (XVec3, XQuat, XVec3) {
        let det = self.matrix3.determinant();
        assert!(det != X64::ZERO);

        let scale = XVec3::new(
            self.matrix3.x_axis.length() * det.signum(),
            self.matrix3.y_axis.length(),
            self.matrix3.z_axis.length(),
        );

        assert!(scale.x != 0 && scale.y != 0 && scale.z != 0);

        let inv_scale = scale.recip();

        #[allow(clippy::useless_conversion)]
        let rotation = XQuat::from_mat3(&XMat3::from_cols(
            (self.matrix3.x_axis * inv_scale.x).into(),
            (self.matrix3.y_axis * inv_scale.y).into(),
            (self.matrix3.z_axis * inv_scale.z).into(),
        ));

        #[allow(clippy::useless_conversion)]
        (scale, rotation, self.translation.into())
    }

    /// Creates a left-handed view transform using a camera position, an up direction, and a facing
    /// direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    #[inline]
    #[must_use]
    pub fn look_to_lh(eye: XVec3, dir: XVec3, up: XVec3) -> Self {
        Self::look_to_rh(eye, -dir, up)
    }

    /// Creates a right-handed view transform using a camera position, an up direction, and a facing
    /// direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    #[inline]
    #[must_use]
    pub fn look_to_rh(eye: XVec3, dir: XVec3, up: XVec3) -> Self {
        let f = dir.normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);

        Self {
            matrix3: XMat3::from_cols(
                XVec3::new(s.x, u.x, -f.x),
                XVec3::new(s.y, u.y, -f.y),
                XVec3::new(s.z, u.z, -f.z),
            ),
            translation: XVec3::new(-eye.dot(s), -eye.dot(u), eye.dot(f)),
        }
    }

    /// Creates a left-handed view transform using a camera position, an up direction, and a focal
    /// point.
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    ///
    /// # Panics
    ///
    /// Will panic if `up` is not normalized when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn look_at_lh(eye: XVec3, center: XVec3, up: XVec3) -> Self {
        assert!(up.is_normalized());
        Self::look_to_lh(eye, center - eye, up)
    }

    /// Creates a right-handed view transform using a camera position, an up direction, and a focal
    /// point.
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    ///
    /// # Panics
    ///
    /// Will panic if `up` is not normalized when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn look_at_rh(eye: XVec3, center: XVec3, up: XVec3) -> Self {
        assert!(up.is_normalized());
        Self::look_to_rh(eye, center - eye, up)
    }

    /// Transforms the given 3D points, applying shear, scale, rotation and translation.
    #[inline]
    pub fn transform_point3(&self, rhs: XVec3) -> XVec3 {
        #[allow(clippy::useless_conversion)]
        ((self.matrix3.x_axis * rhs.x)
            + (self.matrix3.y_axis * rhs.y)
            + (self.matrix3.z_axis * rhs.z)
            + self.translation)
            .into()
    }

    /// Transforms the given 3D vector, applying shear, scale and rotation (but NOT
    /// translation).
    ///
    /// To also apply translation, use [`Self::transform_point3()`] instead.
    #[inline]
    #[must_use]
    pub fn transform_vector3(&self, rhs: XVec3) -> XVec3 {
        #[allow(clippy::useless_conversion)]
        ((self.matrix3.x_axis * rhs.x)
            + (self.matrix3.y_axis * rhs.y)
            + (self.matrix3.z_axis * rhs.z))
            .into()
    }

    /// Returns `true` if, and only if, all elements are finite.
    ///
    /// If any element is either `NaN`, positive or negative infinity, this will return
    /// `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.matrix3.is_finite() && self.translation.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> bool {
        self.matrix3.is_nan() || self.translation.is_nan()
    }

    /// Returns true if the absolute difference of all elements between `self` and `rhs`
    /// is less than or equal to `max_abs_diff`.
    ///
    /// This can be used to compare if two 3x4 matrices contain similar elements. It works
    /// best when comparing with a known value. The `max_abs_diff` that should be used used
    /// depends on the values being compared against.
    ///
    /// For more see
    /// [comparing floating point numbers](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/).
    #[inline]
    #[must_use]
    pub fn abs_diff_eq(&self, rhs: Self, max_abs_diff: X64) -> bool {
        self.matrix3.abs_diff_eq(rhs.matrix3, max_abs_diff)
            && self.translation.abs_diff_eq(rhs.translation, max_abs_diff)
    }

    /// Return the inverse of this transform.
    ///
    /// Note that if the transform is not invertible the result will be invalid.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        let matrix3 = self.matrix3.inverse();
        // transform negative translation by the matrix inverse:
        let translation = -(matrix3 * self.translation);

        Self {
            matrix3,
            translation,
        }
    }
}

impl Default for XAffine3 {
    #[inline(always)]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Deref for XAffine3 {
    type Target = crate::deref::Cols4<XVec3>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const Self as *const Self::Target) }
    }
}

impl DerefMut for XAffine3 {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self as *mut Self as *mut Self::Target) }
    }
}

impl PartialEq for XAffine3 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.matrix3.eq(&rhs.matrix3) && self.translation.eq(&rhs.translation)
    }
}

#[cfg(not(target_arch = "spirv"))]
impl core::fmt::Debug for XAffine3 {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt.debug_struct(stringify!(XAffine3))
            .field("matrix3", &self.matrix3)
            .field("translation", &self.translation)
            .finish()
    }
}

#[cfg(not(target_arch = "spirv"))]
impl core::fmt::Display for XAffine3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "[{}, {}, {}, {}]",
            self.matrix3.x_axis, self.matrix3.y_axis, self.matrix3.z_axis, self.translation
        )
    }
}

impl<'a> core::iter::Product<&'a Self> for XAffine3 {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::IDENTITY, |a, &b| a * b)
    }
}

impl Mul for XAffine3 {
    type Output = XAffine3;

    #[inline]
    fn mul(self, rhs: XAffine3) -> Self::Output {
        Self {
            matrix3: self.matrix3 * rhs.matrix3,
            translation: self.matrix3 * rhs.translation + self.translation,
        }
    }
}

impl MulAssign for XAffine3 {
    #[inline]
    fn mul_assign(&mut self, rhs: XAffine3) {
        *self = self.mul(rhs);
    }
}

impl From<XAffine3> for XMat4 {
    #[inline]
    fn from(m: XAffine3) -> XMat4 {
        XMat4::from_cols(
            m.matrix3.x_axis.extend(X64::ZERO),
            m.matrix3.y_axis.extend(X64::ZERO),
            m.matrix3.z_axis.extend(X64::ZERO),
            m.translation.extend(X64::ONE),
        )
    }
}

impl Mul<XMat4> for XAffine3 {
    type Output = XMat4;

    #[inline]
    fn mul(self, rhs: XMat4) -> Self::Output {
        XMat4::from(self) * rhs
    }
}

impl Mul<XAffine3> for XMat4 {
    type Output = XMat4;

    #[inline]
    fn mul(self, rhs: XAffine3) -> Self::Output {
        self * XMat4::from(rhs)
    }
}
