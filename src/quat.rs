use crate::*;
use crate::vec2::*;
use crate::vec3::*;
use crate::vec4::*;
use crate::mat3::*;
use crate::mat4::*;
use crate::euler::*;
use crate::affine3::*;

#[cfg(not(target_arch = "spirv"))]
use core::fmt;
use core::iter::{Product, Sum};
use core::ops::{Add, Div, Mul, MulAssign, Neg, Sub};

/// Creates a quaternion from `x`, `y`, `z` and `w` values.
///
/// This should generally not be called manually unless you know what you are doing. Use
/// one of the other constructors instead such as `identity` or `from_axis_angle`.
#[inline]
#[must_use]
pub const fn quat(x: X64, y: X64, z: X64, w: X64) -> XQuat {
    XQuat::from_xyzw(x, y, z, w)
}

/// A quaternion representing an orientation.
///
/// This quaternion is intended to be of unit length but may denormalize due to
/// floating point "error creep" which can occur when successive quaternion
/// operations are applied.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch = "spirv"), repr(C))]
#[cfg_attr(target_arch = "spirv", repr(simd))]
pub struct XQuat {
    pub x: X64,
    pub y: X64,
    pub z: X64,
    pub w: X64,
}

impl XQuat {
    /// All zeros.
    const ZERO: Self = Self::from_array([X64::ZERO; 4]);

    /// The identity quaternion. Corresponds to no rotation.
    pub const IDENTITY: Self = Self::from_xyzw(X64::ZERO, X64::ZERO, X64::ZERO, X64::ONE);

    /// All NANs.
    pub const NAN: Self = Self::from_array([X64::NAN; 4]);

    /// Creates a new rotation quaternion.
    ///
    /// This should generally not be called manually unless you know what you are doing.
    /// Use one of the other constructors instead such as `identity` or `from_axis_angle`.
    ///
    /// `from_xyzw` is mostly used by unit tests and `serde` deserialization.
    ///
    /// # Preconditions
    ///
    /// This function does not check if the input is normalized, it is up to the user to
    /// provide normalized input or to normalized the resulting quaternion.
    #[inline(always)]
    #[must_use]
    pub const fn from_xyzw(x: X64, y: X64, z: X64, w: X64) -> Self {
        Self { x, y, z, w }
    }

    /// Creates a rotation quaternion from an array.
    ///
    /// # Preconditions
    ///
    /// This function does not check if the input is normalized, it is up to the user to
    /// provide normalized input or to normalized the resulting quaternion.
    #[inline]
    #[must_use]
    pub const fn from_array(a: [X64; 4]) -> Self {
        Self::from_xyzw(a[0], a[1], a[2], a[3])
    }

    /// Creates a new rotation quaternion from a 4D vector.
    ///
    /// # Preconditions
    ///
    /// This function does not check if the input is normalized, it is up to the user to
    /// provide normalized input or to normalized the resulting quaternion.
    #[inline]
    #[must_use]
    pub const fn from_vec4(v: XVec4) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
            w: v.w,
        }
    }

    /// Creates a rotation quaternion from a slice.
    ///
    /// # Preconditions
    ///
    /// This function does not check if the input is normalized, it is up to the user to
    /// provide normalized input or to normalized the resulting quaternion.
    ///
    /// # Panics
    ///
    /// Panics if `slice` length is less than 4.
    #[inline]
    #[must_use]
    pub fn from_slice(slice: &[X64]) -> Self {
        Self::from_xyzw(slice[0], slice[1], slice[2], slice[3])
    }

    /// Writes the quaternion to an unaligned slice.
    ///
    /// # Panics
    ///
    /// Panics if `slice` length is less than 4.
    #[inline]
    pub fn write_to_slice(self, slice: &mut [X64]) {
        slice[0] = self.x;
        slice[1] = self.y;
        slice[2] = self.z;
        slice[3] = self.w;
    }

    /// Create a quaternion for a normalized rotation `axis` and `angle` (in radians).
    ///
    /// The axis must be a unit vector.
    ///
    /// # Panics
    ///
    /// Will panic if `axis` is not normalized when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: XVec3, angle: X64) -> Self {
        assert!(axis.is_normalized());
        let (s, c) = (angle * (X64::ONE / 2)).sin_cos();
        let v = axis * s;
        Self::from_xyzw(v.x, v.y, v.z, c)
    }

    /// Create a quaternion that rotates `v.length()` radians around `v.normalize()`.
    ///
    /// `from_scaled_axis(XVec3::ZERO)` results in the identity quaternion.
    #[inline]
    #[must_use]
    pub fn from_scaled_axis(v: XVec3) -> Self {
        let length = v.length();
        if length == X64::ZERO {
            Self::IDENTITY
        } else {
            Self::from_axis_angle(v / length, length)
        }
    }

    /// Creates a quaternion from the `angle` (in radians) around the x axis.
    #[inline]
    #[must_use]
    pub fn from_rotation_x(angle: X64) -> Self {
        let (s, c) = (angle * (X64::ONE / 2)).sin_cos();
        Self::from_xyzw(s, X64::ZERO, X64::ZERO, c)
    }

    /// Creates a quaternion from the `angle` (in radians) around the y axis.
    #[inline]
    #[must_use]
    pub fn from_rotation_y(angle: X64) -> Self {
        let (s, c) = (angle * (X64::ONE / 2)).sin_cos();
        Self::from_xyzw(X64::ZERO, s, X64::ZERO, c)
    }

    /// Creates a quaternion from the `angle` (in radians) around the z axis.
    #[inline]
    #[must_use]
    pub fn from_rotation_z(angle: X64) -> Self {
        let (s, c) = (angle * (X64::ONE / 2)).sin_cos();
        Self::from_xyzw(X64::ZERO, X64::ZERO, s, c)
    }

    /// Creates a quaternion from the given Euler rotation sequence and the angles (in radians).
    #[inline]
    #[must_use]
    pub fn from_euler(euler: XEulerRot, a: X64, b: X64, c: X64) -> Self {
        euler.new_quat(a, b, c)
    }

    /// From the columns of a 3x3 rotation matrix.
    #[inline]
    #[must_use]
    pub(crate) fn from_rotation_axes(x_axis: XVec3, y_axis: XVec3, z_axis: XVec3) -> Self {
        // Based on https://github.com/microsoft/DirectXMath `XM$quaternionRotationMatrix`
        let (m00, m01, m02) = x_axis.into();
        let (m10, m11, m12) = y_axis.into();
        let (m20, m21, m22) = z_axis.into();
        if m22 <= X64::ZERO {
            // x^2 + y^2 >= z^2 + w^2
            let dif10 = m11 - m00;
            let omm22 = X64::ONE - m22;
            if dif10 <= X64::ZERO {
                // x^2 >= y^2
                let four_xsq = omm22 - dif10;
                let inv4x = (X64::ONE / 2) / four_xsq.sqrt();
                Self::from_xyzw(
                    four_xsq * inv4x,
                    (m01 + m10) * inv4x,
                    (m02 + m20) * inv4x,
                    (m12 - m21) * inv4x,
                )
            } else {
                // y^2 >= x^2
                let four_ysq = omm22 + dif10;
                let inv4y = (X64::ONE / 2) / four_ysq.sqrt();
                Self::from_xyzw(
                    (m01 + m10) * inv4y,
                    four_ysq * inv4y,
                    (m12 + m21) * inv4y,
                    (m20 - m02) * inv4y,
                )
            }
        } else {
            // z^2 + w^2 >= x^2 + y^2
            let sum10 = m11 + m00;
            let opm22 = X64::ONE + m22;
            if sum10 <= X64::ZERO {
                // z^2 >= w^2
                let four_zsq = opm22 - sum10;
                let inv4z = (X64::ONE / 2) / four_zsq.sqrt();
                Self::from_xyzw(
                    (m02 + m20) * inv4z,
                    (m12 + m21) * inv4z,
                    four_zsq * inv4z,
                    (m01 - m10) * inv4z,
                )
            } else {
                // w^2 >= z^2
                let four_wsq = opm22 + sum10;
                let inv4w = (X64::ONE / 2) / four_wsq.sqrt();
                Self::from_xyzw(
                    (m12 - m21) * inv4w,
                    (m20 - m02) * inv4w,
                    (m01 - m10) * inv4w,
                    four_wsq * inv4w,
                )
            }
        }
    }

    /// Creates a quaternion from a 3x3 rotation matrix.
    #[inline]
    #[must_use]
    pub fn from_mat3(mat: &XMat3) -> Self {
        Self::from_rotation_axes(mat.x_axis, mat.y_axis, mat.z_axis)
    }

    /// Creates a quaternion from a 3x3 rotation matrix inside a homogeneous 4x4 matrix.
    #[inline]
    #[must_use]
    pub fn from_mat4(mat: &XMat4) -> Self {
        Self::from_rotation_axes(
            mat.x_axis.truncate(),
            mat.y_axis.truncate(),
            mat.z_axis.truncate(),
        )
    }

    /// Gets the minimal rotation for transforming `from` to `to`.  The rotation is in the
    /// plane spanned by the two vectors.  Will rotate at most 180 degrees.
    ///
    /// The inputs must be unit vectors.
    ///
    /// `from_rotation_arc(from, to) * from ≈ to`.
    ///
    /// For near-singular cases (from≈to and from≈-to) the current implementation
    /// is only accurate to about 0.001 (for `f32`).
    ///
    /// # Panics
    ///
    /// Will panic if `from` or `to` are not normalized when `assert` is enabled.
    #[must_use]
    pub fn from_rotation_arc(from: XVec3, to: XVec3) -> Self {
        assert!(from.is_normalized());
        assert!(to.is_normalized());

        let one_minus_eps: X64 = X64::ONE - (X64::ONE * 2) * (X64::DELTA << 3);
        let dot = from.dot(to);
        if dot > one_minus_eps {
            // 0° singulary: from ≈ to
            Self::IDENTITY
        } else if dot < -one_minus_eps {
            // 180° singulary: from ≈ -to
            // half a turn = 𝛕/2 = 180°
            Self::from_axis_angle(from.any_orthonormal_vector(), X64::PI)
        } else {
            let c = from.cross(to);
            Self::from_xyzw(c.x, c.y, c.z, X64::ONE + dot).normalize()
        }
    }

    /// Gets the minimal rotation for transforming `from` to either `to` or `-to`.  This means
    /// that the resulting quaternion will rotate `from` so that it is colinear with `to`.
    ///
    /// The rotation is in the plane spanned by the two vectors.  Will rotate at most 90
    /// degrees.
    ///
    /// The inputs must be unit vectors.
    ///
    /// `to.dot(from_rotation_arc_colinear(from, to) * from).abs() ≈ 1`.
    ///
    /// # Panics
    ///
    /// Will panic if `from` or `to` are not normalized when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn from_rotation_arc_colinear(from: XVec3, to: XVec3) -> Self {
        if from.dot(to) < X64::ZERO {
            Self::from_rotation_arc(from, -to)
        } else {
            Self::from_rotation_arc(from, to)
        }
    }

    /// Gets the minimal rotation for transforming `from` to `to`.  The resulting rotation is
    /// around the z axis. Will rotate at most 180 degrees.
    ///
    /// The inputs must be unit vectors.
    ///
    /// `from_rotation_arc_2d(from, to) * from ≈ to`.
    ///
    /// For near-singular cases (from≈to and from≈-to) the current implementation
    /// is only accurate to about 0.001 (for `f32`).
    ///
    /// # Panics
    ///
    /// Will panic if `from` or `to` are not normalized when `assert` is enabled.
    #[must_use]
    pub fn from_rotation_arc_2d(from: XVec2, to: XVec2) -> Self {
        assert!(from.is_normalized());
        assert!(to.is_normalized());

        let one_minus_eps: X64 = X64::ONE - (X64::ONE * 2) * (X64::DELTA << 3);
        let dot = from.dot(to);
        if dot > one_minus_eps {
            // 0° singulary: from ≈ to
            Self::IDENTITY
        } else if dot < -one_minus_eps {
            // 180° singulary: from ≈ -to
            const COS_FRAC_PI_2: X64 = X64::ZERO;
            const SIN_FRAC_PI_2: X64 = X64::ONE;
            // rotation around z by PI radians
            Self::from_xyzw(X64::ZERO, X64::ZERO, SIN_FRAC_PI_2, COS_FRAC_PI_2)
        } else {
            // vector3 cross where z=0
            let z = from.x * to.y - to.x * from.y;
            let w = X64::ONE + dot;
            // calculate length with x=0 and y=0 to normalize
            let len_rcp = X64::ONE / (z * z + w * w).sqrt();
            Self::from_xyzw(X64::ZERO, X64::ZERO, z * len_rcp, w * len_rcp)
        }
    }

    /// Returns the rotation axis (normalized) and angle (in radians) of `self`.
    #[inline]
    #[must_use]
    pub fn to_axis_angle(self) -> (XVec3, X64) {
        let epsilon: X64 = X64::DELTA << 3;
        let v = XVec3::new(self.x, self.y, self.z);
        let length = v.length();
        if length >= epsilon {
            let angle = (X64::ONE * 2) * X64::atan2(length, self.w);
            let axis = v / length;
            (axis, angle)
        } else {
            (XVec3::X, X64::ZERO)
        }
    }

    /// Returns the rotation axis scaled by the rotation in radians.
    #[inline]
    #[must_use]
    pub fn to_scaled_axis(self) -> XVec3 {
        let (axis, angle) = self.to_axis_angle();
        axis * angle
    }

    /// Returns the rotation angles for the given euler rotation sequence.
    #[inline]
    #[must_use]
    pub fn to_euler(self, euler: XEulerRot) -> (X64, X64, X64) {
        euler.convert_quat(self)
    }

    /// `[x, y, z, w]`
    #[inline]
    #[must_use]
    pub fn to_array(&self) -> [X64; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Returns the vector part of the quaternion.
    #[inline]
    #[must_use]
    pub fn xyz(self) -> XVec3 {
        XVec3::new(self.x, self.y, self.z)
    }

    /// Returns the quaternion conjugate of `self`. For a unit quaternion the
    /// conjugate is also the inverse.
    #[inline]
    #[must_use]
    pub fn conjugate(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    /// Returns the inverse of a normalized quaternion.
    ///
    /// Typically quaternion inverse returns the conjugate of a normalized quaternion.
    /// Because `self` is assumed to already be unit length this method *does not* normalize
    /// before returning the conjugate.
    ///
    /// # Panics
    ///
    /// Will panic if `self` is not normalized when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn inverse(self) -> Self {
        assert!(self.is_normalized());
        self.conjugate()
    }

    /// Computes the dot product of `self` and `rhs`. The dot product is
    /// equal to the cosine of the angle between two quaternion rotations.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> X64 {
        XVec4::from(self).dot(XVec4::from(rhs))
    }

    /// Computes the length of `self`.
    #[doc(alias = "magnitude")]
    #[inline]
    #[must_use]
    pub fn length(self) -> X64 {
        XVec4::from(self).length()
    }

    /// Computes the squared length of `self`.
    ///
    /// This is generally faster than `length()` as it avoids a square
    /// root operation.
    #[doc(alias = "magnitude2")]
    #[inline]
    #[must_use]
    pub fn length_squared(self) -> X64 {
        XVec4::from(self).length_squared()
    }

    /// Computes `X64::ONE / length()`.
    ///
    /// For valid results, `self` must _not_ be of length zero.
    #[inline]
    #[must_use]
    pub fn length_recip(self) -> X64 {
        XVec4::from(self).length_recip()
    }

    /// Returns `self` normalized to length X64::ONE.
    ///
    /// For valid results, `self` must _not_ be of length zero.
    ///
    /// Panics
    ///
    /// Will panic if `self` is zero length when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        Self::from_vec4(XVec4::from(self).normalize())
    }

    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        XVec4::from(self).is_finite()
    }

    #[inline]
    #[must_use]
    pub fn is_nan(self) -> bool {
        XVec4::from(self).is_nan()
    }

    /// Returns whether `self` of length `X64::ONE` or not.
    ///
    /// Uses a precision threshold of `1e-6`.
    #[inline]
    #[must_use]
    pub fn is_normalized(self) -> bool {
        XVec4::from(self).is_normalized()
    }

    #[inline]
    #[must_use]
    pub fn is_near_identity(self) -> bool {
        // Based on https://github.com/nfrechette/rtm `rtm::quat_near_identity`
        let threshold_angle = 0.002_847_144_6;
        // Because of floating point precision, we cannot represent very small rotations.
        // The closest f32 to X64::ONE that is not X64::ONE itself yields:
        // 0.99999994.acos() * (X64::ONE * 2)  = 0.000690533954 rad
        //
        // An error threshold of 1.e-6 is used by default.
        // (X64::ONE - 1.e-6).acos() * (X64::ONE * 2) = 0.00284714461 rad
        // (X64::ONE - 1.e-7).acos() * (X64::ONE * 2) = 0.00097656250 rad
        //
        // We don't really care about the angle value itself, only if it's close to 0.
        // This will happen whenever quat.w is close to X64::ONE.
        // If the quat.w is close to X64::NEG_ONE, the angle will be near 2*PI which is close to
        // a negative 0 rotation. By forcing quat.w to be positive, we'll end up with
        // the shortest path.
        let positive_w_angle = (self.w.abs()) * (X64::ONE * 2).acos().1;
        positive_w_angle < threshold_angle
    }

    /// Returns the angle (in radians) for the minimal rotation
    /// for transforming this quaternion into another.
    ///
    /// Both quaternions must be normalized.
    ///
    /// # Panics
    ///
    /// Will panic if `self` or `rhs` are not normalized when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn angle_between(self, rhs: Self) -> X64 {
        assert!(self.is_normalized() && rhs.is_normalized());
        self.dot(rhs).abs().acos().1 * (X64::ONE * 2)
    }

    /// Returns true if the absolute difference of all elements between `self` and `rhs`
    /// is less than or equal to `max_abs_diff`.
    ///
    /// This can be used to compare if two quaternions contain similar elements. It works
    /// best when comparing with a known value. The `max_abs_diff` that should be used used
    /// depends on the values being compared against.
    ///
    /// For more see
    /// [comparing floating point numbers](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/).
    #[inline]
    #[must_use]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: X64) -> bool {
        XVec4::from(self).abs_diff_eq(XVec4::from(rhs), max_abs_diff)
    }

    /// Performs a linear interpolation between `self` and `rhs` based on
    /// the value `s`.
    ///
    /// When `s` is `X64::ZERO`, the result will be equal to `self`.  When `s`
    /// is `X64::ONE`, the result will be equal to `rhs`.
    ///
    /// # Panics
    ///
    /// Will panic if `self` or `end` are not normalized when `assert` is enabled.
    #[doc(alias = "mix")]
    #[inline]
    #[must_use]
    pub fn lerp(self, end: Self, s: X64) -> Self {
        assert!(self.is_normalized());
        assert!(end.is_normalized());

        let start = self;
        let dot = start.dot(end);
        let bias = if dot >= X64::ZERO { X64::ONE } else { X64::NEG_ONE };
        let interpolated = start.add(end.mul(bias).sub(start).mul(s));
        interpolated.normalize()
    }

    /// Performs a spherical linear interpolation between `self` and `end`
    /// based on the value `s`.
    ///
    /// When `s` is `X64::ZERO`, the result will be equal to `self`.  When `s`
    /// is `X64::ONE`, the result will be equal to `end`.
    ///
    /// # Panics
    ///
    /// Will panic if `self` or `end` are not normalized when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn slerp(self, mut end: Self, s: X64) -> Self {
        // http://number-none.com/product/Understanding%20Slerp,%20Then%20Not%20Using%20It/
        assert!(self.is_normalized());
        assert!(end.is_normalized());

        let dot_threshold: X64 = x64!(0.9995);

        // Note that a rotation can be represented by two quaternions: `q` and
        // `-q`. The slerp path between `q` and `end` will be different from the
        // path between `-q` and `end`. One path will take the long way around and
        // one will take the short way. In order to correct for this, the `dot`
        // product between `self` and `end` should be positive. If the `dot`
        // product is negative, slerp between `self` and `-end`.
        let mut dot = self.dot(end);
        if dot < X64::ZERO {
            end = -end;
            dot = -dot;
        }

        if dot > dot_threshold {
            // assumes lerp returns a normalized quaternion
            self.lerp(end, s)
        } else {
            let theta = dot.acos().1;

            let scale1 = (theta * (X64::ONE - s)).sin();
            let scale2 = (theta * s).sin();
            let theta_sin = theta.sin();

            self.mul(scale1).add(end.mul(scale2)).mul(X64::ONE / theta_sin)
        }
    }

    /// Multiplies a quaternion and a 3D vector, returning the rotated vector.
    ///
    /// # Panics
    ///
    /// Will panic if `self` is not normalized when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn mul_vec3(self, rhs: XVec3) -> XVec3 {
        assert!(self.is_normalized());

        let w = self.w;
        let b = XVec3::new(self.x, self.y, self.z);
        let b2 = b.dot(b);
        rhs.mul(w * w - b2)
            .add(b.mul(rhs.dot(b) * (X64::ONE * 2)))
            .add(b.cross(rhs).mul(w * (X64::ONE * 2)))
    }

    /// Multiplies two quaternions. If they each represent a rotation, the result will
    /// represent the combined rotation.
    ///
    /// Note that due to floating point rounding the result may not be perfectly normalized.
    ///
    /// # Panics
    ///
    /// Will panic if `self` or `rhs` are not normalized when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn mul_quat(self, rhs: Self) -> Self {
        assert!(self.is_normalized());
        assert!(rhs.is_normalized());

        let (x0, y0, z0, w0) = self.into();
        let (x1, y1, z1, w1) = rhs.into();
        Self::from_xyzw(
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        )
    }

    /// Creates a quaternion from a 3x3 rotation matrix inside a 3D affine transform.
    #[inline]
    #[must_use]
    pub fn from_affine3(a: &XAffine3) -> Self {
        #[allow(clippy::useless_conversion)]
        Self::from_rotation_axes(
            a.matrix3.x_axis.into(),
            a.matrix3.y_axis.into(),
            a.matrix3.z_axis.into(),
        )
    }
}

#[cfg(not(target_arch = "spirv"))]
impl fmt::Debug for XQuat {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_tuple(stringify!(XQuat))
            .field(&self.x)
            .field(&self.y)
            .field(&self.z)
            .field(&self.w)
            .finish()
    }
}

#[cfg(not(target_arch = "spirv"))]
impl fmt::Display for XQuat {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "[{}, {}, {}, {}]", self.x, self.y, self.z, self.w)
    }
}

impl Add<XQuat> for XQuat {
    type Output = Self;
    /// Adds two quaternions.
    ///
    /// The sum is not guaranteed to be normalized.
    ///
    /// Note that addition is not the same as combining the rotations represented by the
    /// two quaternions! That corresponds to multiplication.
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::from_vec4(XVec4::from(self) + XVec4::from(rhs))
    }
}

impl Sub<XQuat> for XQuat {
    type Output = Self;
    /// Subtracts the `rhs` quaternion from `self`.
    ///
    /// The difference is not guaranteed to be normalized.
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_vec4(XVec4::from(self) - XVec4::from(rhs))
    }
}

impl Mul<X64> for XQuat {
    type Output = Self;
    /// Multiplies a quaternion by a scalar value.
    ///
    /// The product is not guaranteed to be normalized.
    #[inline]
    fn mul(self, rhs: X64) -> Self {
        Self::from_vec4(XVec4::from(self) * rhs)
    }
}

impl Div<X64> for XQuat {
    type Output = Self;
    /// Divides a quaternion by a scalar value.
    /// The quotient is not guaranteed to be normalized.
    #[inline]
    fn div(self, rhs: X64) -> Self {
        Self::from_vec4(XVec4::from(self) / rhs)
    }
}

impl Mul<XQuat> for XQuat {
    type Output = Self;
    /// Multiplies two quaternions. If they each represent a rotation, the result will
    /// represent the combined rotation.
    ///
    /// Note that due to floating point rounding the result may not be perfectly
    /// normalized.
    ///
    /// # Panics
    ///
    /// Will panic if `self` or `rhs` are not normalized when `assert` is enabled.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul_quat(rhs)
    }
}

impl MulAssign<XQuat> for XQuat {
    /// Multiplies two quaternions. If they each represent a rotation, the result will
    /// represent the combined rotation.
    ///
    /// Note that due to floating point rounding the result may not be perfectly
    /// normalized.
    ///
    /// # Panics
    ///
    /// Will panic if `self` or `rhs` are not normalized when `assert` is enabled.
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_quat(rhs);
    }
}

impl Mul<XVec3> for XQuat {
    type Output = XVec3;
    /// Multiplies a quaternion and a 3D vector, returning the rotated vector.
    ///
    /// # Panics
    ///
    /// Will panic if `self` is not normalized when `assert` is enabled.
    #[inline]
    fn mul(self, rhs: XVec3) -> Self::Output {
        self.mul_vec3(rhs)
    }
}

impl Neg for XQuat {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self * X64::NEG_ONE
    }
}

impl Default for XQuat {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl PartialEq for XQuat {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        XVec4::from(*self).eq(&XVec4::from(*rhs))
    }
}

#[cfg(not(target_arch = "spirv"))]
impl AsRef<[X64; 4]> for XQuat {
    #[inline]
    fn as_ref(&self) -> &[X64; 4] {
        unsafe { &*(self as *const Self as *const [X64; 4]) }
    }
}

impl Sum<Self> for XQuat {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::ZERO, Self::add)
    }
}

impl<'a> Sum<&'a Self> for XQuat {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::ZERO, |a, &b| Self::add(a, b))
    }
}

impl Product for XQuat {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::IDENTITY, Self::mul)
    }
}

impl<'a> Product<&'a Self> for XQuat {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::IDENTITY, |a, &b| Self::mul(a, b))
    }
}

impl From<XQuat> for XVec4 {
    #[inline]
    fn from(q: XQuat) -> Self {
        Self::new(q.x, q.y, q.z, q.w)
    }
}

impl From<XQuat> for (X64, X64, X64, X64) {
    #[inline]
    fn from(q: XQuat) -> Self {
        (q.x, q.y, q.z, q.w)
    }
}

impl From<XQuat> for [X64; 4] {
    #[inline]
    fn from(q: XQuat) -> Self {
        [q.x, q.y, q.z, q.w]
    }
}
