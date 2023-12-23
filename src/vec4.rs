use crate::*;
use crate::vec2::*;
use crate::vec3::*;

#[cfg(not(target_arch = "spirv"))]
use core::fmt;
use core::iter::{Product, Sum};
use core::ops::*;

/// Creates a 4-dimensional vector.
#[inline(always)]
#[must_use]
pub const fn vec4(x: X64, y: X64, z: X64, w: X64) -> Vec4 {
    Vec4::new(x, y, z, w)
}

/// A 4-dimensional vector.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "cuda", repr(align(16)))]
#[cfg_attr(not(target_arch = "spirv"), repr(C))]
#[cfg_attr(target_arch = "spirv", repr(simd))]
pub struct Vec4 {
    pub x: X64,
    pub y: X64,
    pub z: X64,
    pub w: X64,
}

impl Vec4 {
    /// All zeroes.
    pub const ZERO: Self = Self::splat(X64::ZERO);

    /// All ones.
    pub const ONE: Self = Self::splat(X64::ONE);

    /// All negative ones.
    pub const NEG_ONE: Self = Self::splat(X64::NEG_ONE);

    /// All `X64::MIN`.
    pub const MIN: Self = Self::splat(X64::MIN);

    /// All `X64::MAX`.
    pub const MAX: Self = Self::splat(X64::MAX);

    /// All `X64::NAN`.
    pub const NAN: Self = Self::splat(X64::NAN);

    /// All `X64::INFINITY`.
    pub const INFINITY: Self = Self::splat(X64::INFINITY);

    /// All `X64::NEG_INFINITY`.
    pub const NEG_INFINITY: Self = Self::splat(X64::NEG_INFINITY);

    /// A unit vector pointing along the positive X axis.
    pub const X: Self = Self::new(X64::ONE, X64::ZERO, X64::ZERO, X64::ZERO);

    /// A unit vector pointing along the positive Y axis.
    pub const Y: Self = Self::new(X64::ZERO, X64::ONE, X64::ZERO, X64::ZERO);

    /// A unit vector pointing along the positive Z axis.
    pub const Z: Self = Self::new(X64::ZERO, X64::ZERO, X64::ONE, X64::ZERO);

    /// A unit vector pointing along the positive W axis.
    pub const W: Self = Self::new(X64::ZERO, X64::ZERO, X64::ZERO, X64::ONE);

    /// A unit vector pointing along the negative X axis.
    pub const NEG_X: Self = Self::new(X64::NEG_ONE, X64::ZERO, X64::ZERO, X64::ZERO);

    /// A unit vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self::new(X64::ZERO, X64::NEG_ONE, X64::ZERO, X64::ZERO);

    /// A unit vector pointing along the negative Z axis.
    pub const NEG_Z: Self = Self::new(X64::ZERO, X64::ZERO, X64::NEG_ONE, X64::ZERO);

    /// A unit vector pointing along the negative W axis.
    pub const NEG_W: Self = Self::new(X64::ZERO, X64::ZERO, X64::ZERO, X64::NEG_ONE);

    /// The unit axes.
    pub const AXES: [Self; 4] = [Self::X, Self::Y, Self::Z, Self::W];

    /// Creates a new vector.
    #[inline(always)]
    #[must_use]
    pub const fn new(x: X64, y: X64, z: X64, w: X64) -> Self {
        Self { x, y, z, w }
    }

    /// Creates a vector with all elements set to `v`.
    #[inline]
    #[must_use]
    pub const fn splat(v: X64) -> Self {
        Self {
            x: v,

            y: v,

            z: v,

            w: v,
        }
    }

    /// Creates a new vector from an array.
    #[inline]
    #[must_use]
    pub const fn from_array(a: [X64; 4]) -> Self {
        Self::new(a[0], a[1], a[2], a[3])
    }

    /// `[x, y, z, w]`
    #[inline]
    #[must_use]
    pub const fn to_array(&self) -> [X64; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Creates a vector from the first 4 values in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 4 elements long.
    #[inline]
    #[must_use]
    pub const fn from_slice(slice: &[X64]) -> Self {
        Self::new(slice[0], slice[1], slice[2], slice[3])
    }

    /// Writes the elements of `self` to the first 4 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 4 elements long.
    #[inline]
    pub fn write_to_slice(self, slice: &mut [X64]) {
        slice[0] = self.x;
        slice[1] = self.y;
        slice[2] = self.z;
        slice[3] = self.w;
    }

    /// Creates a 3D vector from the `x`, `y` and `z` elements of `self`, discarding `w`.
    ///
    /// Truncation to [`Vec3`] may also be performed by using [`self.xyz()`][crate::swizzles::Vec4Swizzles::xyz()].
    #[inline]
    #[must_use]
    pub fn truncate(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    /// Creates a 4D vector with all elements set to `w`.
    #[inline]
    #[must_use]
    pub fn wwww(self) -> Self {
        Self::splat(self.w)
    }

    /// Computes the dot product of `self` and `rhs`.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> X64 {
        (self.x * rhs.x) + (self.y * rhs.y) + (self.z * rhs.z) + (self.w * rhs.w)
    }

    /// Returns a vector where every component is the dot product of `self` and `rhs`.
    #[inline]
    #[must_use]
    pub fn dot_into_vec(self, rhs: Self) -> Self {
        Self::splat(self.dot(rhs))
    }

    /// Returns a vector containing the minimum values for each element of `self` and `rhs`.
    ///
    /// In other words this computes `[self.x.min(rhs.x), self.y.min(rhs.y), ..]`.
    #[inline]
    #[must_use]
    pub fn min(self, rhs: Self) -> Self {
        Self {
            x: self.x.min(rhs.x),
            y: self.y.min(rhs.y),
            z: self.z.min(rhs.z),
            w: self.w.min(rhs.w),
        }
    }

    /// Returns a vector containing the maximum values for each element of `self` and `rhs`.
    ///
    /// In other words this computes `[self.x.max(rhs.x), self.y.max(rhs.y), ..]`.
    #[inline]
    #[must_use]
    pub fn max(self, rhs: Self) -> Self {
        Self {
            x: self.x.max(rhs.x),
            y: self.y.max(rhs.y),
            z: self.z.max(rhs.z),
            w: self.w.max(rhs.w),
        }
    }

    /// Component-wise clamping of values, similar to [`X64::clamp`].
    ///
    /// Each element in `min` must be less-or-equal to the corresponding element in `max`.
    ///
    /// # Panics
    ///
    /// Will panic if `min` is greater than `max` when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        assert!(min.x.le(&max.x) && min.y.le(&max.y) && min.z.le(&max.z) && min.w.le(&max.w), "clamp: expected min <= max");
        self.max(min).min(max)
    }

    /// Returns the horizontal minimum of `self`.
    ///
    /// In other words this computes `min(x, y, ..)`.
    #[inline]
    #[must_use]
    pub fn min_element(self) -> X64 {
        self.x.min(self.y.min(self.z.min(self.w)))
    }

    /// Returns the horizontal maximum of `self`.
    ///
    /// In other words this computes `max(x, y, ..)`.
    #[inline]
    #[must_use]
    pub fn max_element(self) -> X64 {
        self.x.max(self.y.max(self.z.max(self.w)))
    }

    /// Returns a vector containing the absolute value of each element of `self`.
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
            w: self.w.abs(),
        }
    }

    /// Returns a vector with elements representing the sign of `self`.
    ///
    /// - `X64::ONE` if the number is positive, `+X64::ZERO` or `INFINITY`
    /// - `X64::NEG_ONE` if the number is negative, `-X64::ZERO` or `NEG_INFINITY`
    /// - `NAN` if the number is `NAN`
    #[inline]
    #[must_use]
    pub fn signum(self) -> Self {
        Self {
            x: self.x.signum(),
            y: self.y.signum(),
            z: self.z.signum(),
            w: self.w.signum(),
        }
    }

    /// Returns `true` if, and only if, all elements are finite.  If any element is either
    /// `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan()
    }

    /// Computes the length of `self`.
    #[doc(alias = "magnitude")]
    #[inline]
    #[must_use]
    pub fn length(self) -> X64 {
        self.dot(self).sqrt()
    }

    /// Computes the squared length of `self`.
    ///
    /// This is faster than `length()` as it avoids a square root operation.
    #[doc(alias = "magnitude2")]
    #[inline]
    #[must_use]
    pub fn length_squared(self) -> X64 {
        self.dot(self)
    }

    /// Computes `X64::ONE / length()`.
    ///
    /// For valid results, `self` must _not_ be of length zero.
    #[inline]
    #[must_use]
    pub fn length_recip(self) -> X64 {
        self.length().recip()
    }

    /// Computes the Euclidean distance between two points in space.
    #[inline]
    #[must_use]
    pub fn distance(self, rhs: Self) -> X64 {
        (self - rhs).length()
    }

    /// Compute the squared euclidean distance between two points in space.
    #[inline]
    #[must_use]
    pub fn distance_squared(self, rhs: Self) -> X64 {
        (self - rhs).length_squared()
    }

    /// Returns the element-wise quotient of [Euclidean division] of `self` by `rhs`.
    #[inline]
    #[must_use]
    pub fn div_euclid(self, rhs: Self) -> Self {
        Self::new(
            self.x.div_euclid(rhs.x),
            self.y.div_euclid(rhs.y),
            self.z.div_euclid(rhs.z),
            self.w.div_euclid(rhs.w),
        )
    }

    /// Returns the element-wise remainder of [Euclidean division] of `self` by `rhs`.
    ///
    /// [Euclidean division]: X64::rem_euclid
    #[inline]
    #[must_use]
    pub fn rem_euclid(self, rhs: Self) -> Self {
        Self::new(
            self.x.rem_euclid(rhs.x),
            self.y.rem_euclid(rhs.y),
            self.z.rem_euclid(rhs.z),
            self.w.rem_euclid(rhs.w),
        )
    }

    /// Returns `self` normalized to length X64::ONE.
    ///
    /// For valid results, `self` must _not_ be of length zero, nor very close to zero.
    ///
    /// See also [`Self::try_normalize()`] and [`Self::normalize_or_zero()`].
    ///
    /// Panics
    ///
    /// Will panic if `self` is zero length when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        #[allow(clippy::let_and_return)]
        let normalized = self.mul(self.length_recip());
        assert!(normalized.is_finite());
        normalized
    }

    /// Returns `self` normalized to length X64::ONE if possible, else returns `None`.
    ///
    /// In particular, if the input is zero (or very close to zero), or non-finite,
    /// the result of this operation will be `None`.
    ///
    /// See also [`Self::normalize_or_zero()`].
    #[inline]
    #[must_use]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp.is_finite() && rcp > X64::ZERO {
            Some(self * rcp)
        } else {
            None
        }
    }

    /// Returns `self` normalized to length X64::ONE if possible, else returns zero.
    ///
    /// In particular, if the input is zero (or very close to zero), or non-finite,
    /// the result of this operation will be zero.
    ///
    /// See also [`Self::try_normalize()`].
    #[inline]
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let rcp = self.length_recip();
        if rcp.is_finite() && rcp > X64::ZERO {
            self * rcp
        } else {
            Self::ZERO
        }
    }

    /// Returns whether `self` is length `X64::ONE` or not.
    ///
    /// Uses a precision threshold of `1e-6`.
    #[inline]
    #[must_use]
    pub fn is_normalized(self) -> bool {
        // TODO: do something with epsilon
        (self.length_squared() - X64::ONE).abs() <= X64::DELTA << 3
    }

    /// Returns the vector projection of `self` onto `rhs`.
    ///
    /// `rhs` must be of non-zero length.
    ///
    /// # Panics
    ///
    /// Will panic if `rhs` is zero length when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn project_onto(self, rhs: Self) -> Self {
        let other_len_sq_rcp = rhs.dot(rhs).recip();
        assert!(other_len_sq_rcp.is_finite());
        rhs * self.dot(rhs) * other_len_sq_rcp
    }

    /// Returns the vector rejection of `self` from `rhs`.
    ///
    /// The vector rejection is the vector perpendicular to the projection of `self` onto
    /// `rhs`, in rhs words the result of `self - self.project_onto(rhs)`.
    ///
    /// `rhs` must be of non-zero length.
    ///
    /// # Panics
    ///
    /// Will panic if `rhs` has a length of zero when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn reject_from(self, rhs: Self) -> Self {
        self - self.project_onto(rhs)
    }

    /// Returns the vector projection of `self` onto `rhs`.
    ///
    /// `rhs` must be normalized.
    ///
    /// # Panics
    ///
    /// Will panic if `rhs` is not normalized when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn project_onto_normalized(self, rhs: Self) -> Self {
        assert!(rhs.is_normalized());
        rhs * self.dot(rhs)
    }

    /// Returns the vector rejection of `self` from `rhs`.
    ///
    /// The vector rejection is the vector perpendicular to the projection of `self` onto
    /// `rhs`, in rhs words the result of `self - self.project_onto(rhs)`.
    ///
    /// `rhs` must be normalized.
    ///
    /// # Panics
    ///
    /// Will panic if `rhs` is not normalized when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn reject_from_normalized(self, rhs: Self) -> Self {
        self - self.project_onto_normalized(rhs)
    }

    /// Returns a vector containing the nearest integer to a number for each element of `self`.
    /// Round half-way cases away from X64::ZERO.
    #[inline]
    #[must_use]
    pub fn round(self) -> Self {
        Self {
            x: self.x.round(),
            y: self.y.round(),
            z: self.z.round(),
            w: self.w.round(),
        }
    }

    /// Returns a vector containing the largest integer less than or equal to a number for each
    /// element of `self`.
    #[inline]
    #[must_use]
    pub fn floor(self) -> Self {
        Self {
            x: self.x.floor(),
            y: self.y.floor(),
            z: self.z.floor(),
            w: self.w.floor(),
        }
    }

    /// Returns a vector containing the smallest integer greater than or equal to a number for
    /// each element of `self`.
    #[inline]
    #[must_use]
    pub fn ceil(self) -> Self {
        Self {
            x: self.x.ceil(),
            y: self.y.ceil(),
            z: self.z.ceil(),
            w: self.w.ceil(),
        }
    }

    /// Returns a vector containing the integer part each element of `self`. This means numbers are
    /// always truncated towards zero.
    #[inline]
    #[must_use]
    pub fn trunc(self) -> Self {
        Self {
            x: self.x.round_to_zero(),
            y: self.y.round_to_zero(),
            z: self.z.round_to_zero(),
            w: self.w.round_to_zero(),
        }
    }

    /// Returns a vector containing the fractional part of the vector, e.g. `self -
    /// self.floor()`.
    ///
    /// Note that this is fast but not precise for large numbers.
    #[inline]
    #[must_use]
    pub fn fract(self) -> Self {
        self - self.floor()
    }

    /// Returns a vector containing `e^self` (the exponential function) for each element of
    /// `self`.
    #[inline]
    #[must_use]
    pub fn exp(self) -> Self {
        Self::new(
            self.x.exp(),
            self.y.exp(),
            self.z.exp(),
            self.w.exp(),
        )
    }

    /// Returns a vector containing each element of `self` raised to the power of `n`.
    #[inline]
    #[must_use]
    pub fn powf(self, n: X64) -> Self {
        Self::new(
            self.x.powf(n),
            self.y.powf(n),
            self.z.powf(n),
            self.w.powf(n),
        )
    }

    /// Returns a vector containing the reciprocal `X64::ONE/n` of each element of `self`.
    #[inline]
    #[must_use]
    pub fn recip(self) -> Self {
        Self {
            x: X64::ONE / self.x,
            y: X64::ONE / self.y,
            z: X64::ONE / self.z,
            w: X64::ONE / self.w,
        }
    }

    /// Performs a linear interpolation between `self` and `rhs` based on the value `s`.
    ///
    /// When `s` is `X64::ZERO`, the result will be equal to `self`.  When `s` is `X64::ONE`, the result
    /// will be equal to `rhs`. When `s` is outside of range `[0, 1]`, the result is linearly
    /// extrapolated.
    #[doc(alias = "mix")]
    #[inline]
    #[must_use]
    pub fn lerp(self, rhs: Self, s: X64) -> Self {
        self + ((rhs - self) * s)
    }

    /// Calculates the midpoint between `self` and `rhs`.
    ///
    /// The midpoint is the average of, or halfway point between, two vectors.
    /// `a.midpoint(b)` should yield the same result as `a.lerp(b, 0.5)`
    /// while being slightly cheaper to compute.
    #[inline]
    pub fn midpoint(self, rhs: Self) -> Self {
        (self + rhs) * (X64::ONE / 2)
    }

    /// Returns true if the absolute difference of all elements between `self` and `rhs` is
    /// less than or equal to `max_abs_diff`.
    ///
    /// This can be used to compare if two vectors contain similar elements. It works best when
    /// comparing with a known value. The `max_abs_diff` that should be used used depends on
    /// the values being compared against.
    ///
    /// For more see
    /// [comparing floating point numbers](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/).
    #[inline]
    #[must_use]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: X64) -> bool {
        let dif = self.sub(rhs).abs();
        let gap = Self::splat(max_abs_diff);
        dif.x.le(&gap.x) && dif.y.le(&gap.y) && dif.z.le(&gap.z) && dif.w.le(&gap.w)
    }

    /// Returns a vector with a length no less than `min` and no more than `max`
    ///
    /// # Panics
    ///
    /// Will panic if `min` is greater than `max` when `assert` is enabled.
    #[inline]
    #[must_use]
    pub fn clamp_length(self, min: X64, max: X64) -> Self {
        assert!(min <= max);
        let length_sq = self.length_squared();
        if length_sq < min * min {
            min * (self / length_sq.sqrt())
        } else if length_sq > max * max {
            max * (self / length_sq.sqrt())
        } else {
            self
        }
    }

    /// Returns a vector with a length no more than `max`
    #[inline]
    #[must_use]
    pub fn clamp_length_max(self, max: X64) -> Self {
        let length_sq = self.length_squared();
        if length_sq > max * max {
            max * (self / length_sq.sqrt())
        } else {
            self
        }
    }

    /// Returns a vector with a length no less than `min`
    #[inline]
    #[must_use]
    pub fn clamp_length_min(self, min: X64) -> Self {
        let length_sq = self.length_squared();
        if length_sq < min * min {
            min * (self / length_sq.sqrt())
        } else {
            self
        }
    }

    /// Fused multiply-add. Computes `(self * a) + b` element-wise with only one rounding
    /// error, yielding a more accurate result than an unfused multiply-add.
    ///
    /// Using `mul_add` *may* be more performant than an unfused multiply-add if the target
    /// architecture has a dedicated fma CPU instruction. However, this is not always true,
    /// and will be heavily dependant on designing algorithms with specific target hardware in
    /// mind.
    #[inline]
    #[must_use]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self::new(
            self.x.mul_add(a.x, b.x),
            self.y.mul_add(a.y, b.y),
            self.z.mul_add(a.z, b.z),
            self.w.mul_add(a.w, b.w),
        )
    }
}

impl Default for Vec4 {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

impl Div<Vec4> for Vec4 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self {
            x: self.x.div(rhs.x),
            y: self.y.div(rhs.y),
            z: self.z.div(rhs.z),
            w: self.w.div(rhs.w),
        }
    }
}

impl DivAssign<Vec4> for Vec4 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.x.div_assign(rhs.x);
        self.y.div_assign(rhs.y);
        self.z.div_assign(rhs.z);
        self.w.div_assign(rhs.w);
    }
}

impl Div<X64> for Vec4 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: X64) -> Self {
        Self {
            x: self.x.div(rhs),
            y: self.y.div(rhs),
            z: self.z.div(rhs),
            w: self.w.div(rhs),
        }
    }
}

impl DivAssign<X64> for Vec4 {
    #[inline]
    fn div_assign(&mut self, rhs: X64) {
        self.x.div_assign(rhs);
        self.y.div_assign(rhs);
        self.z.div_assign(rhs);
        self.w.div_assign(rhs);
    }
}

impl Div<Vec4> for X64 {
    type Output = Vec4;
    #[inline]
    fn div(self, rhs: Vec4) -> Vec4 {
        Vec4 {
            x: self.div(rhs.x),
            y: self.div(rhs.y),
            z: self.div(rhs.z),
            w: self.div(rhs.w),
        }
    }
}

impl Mul<Vec4> for Vec4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            x: self.x.mul(rhs.x),
            y: self.y.mul(rhs.y),
            z: self.z.mul(rhs.z),
            w: self.w.mul(rhs.w),
        }
    }
}

impl MulAssign<Vec4> for Vec4 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.x.mul_assign(rhs.x);
        self.y.mul_assign(rhs.y);
        self.z.mul_assign(rhs.z);
        self.w.mul_assign(rhs.w);
    }
}

impl Mul<X64> for Vec4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: X64) -> Self {
        Self {
            x: self.x.mul(rhs),
            y: self.y.mul(rhs),
            z: self.z.mul(rhs),
            w: self.w.mul(rhs),
        }
    }
}

impl MulAssign<X64> for Vec4 {
    #[inline]
    fn mul_assign(&mut self, rhs: X64) {
        self.x.mul_assign(rhs);
        self.y.mul_assign(rhs);
        self.z.mul_assign(rhs);
        self.w.mul_assign(rhs);
    }
}

impl Mul<Vec4> for X64 {
    type Output = Vec4;
    #[inline]
    fn mul(self, rhs: Vec4) -> Vec4 {
        Vec4 {
            x: self.mul(rhs.x),
            y: self.mul(rhs.y),
            z: self.mul(rhs.z),
            w: self.mul(rhs.w),
        }
    }
}

impl Add<Vec4> for Vec4 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x.add(rhs.x),
            y: self.y.add(rhs.y),
            z: self.z.add(rhs.z),
            w: self.w.add(rhs.w),
        }
    }
}

impl AddAssign<Vec4> for Vec4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x.add_assign(rhs.x);
        self.y.add_assign(rhs.y);
        self.z.add_assign(rhs.z);
        self.w.add_assign(rhs.w);
    }
}

impl Add<X64> for Vec4 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: X64) -> Self {
        Self {
            x: self.x.add(rhs),
            y: self.y.add(rhs),
            z: self.z.add(rhs),
            w: self.w.add(rhs),
        }
    }
}

impl AddAssign<X64> for Vec4 {
    #[inline]
    fn add_assign(&mut self, rhs: X64) {
        self.x.add_assign(rhs);
        self.y.add_assign(rhs);
        self.z.add_assign(rhs);
        self.w.add_assign(rhs);
    }
}

impl Add<Vec4> for X64 {
    type Output = Vec4;
    #[inline]
    fn add(self, rhs: Vec4) -> Vec4 {
        Vec4 {
            x: self.add(rhs.x),
            y: self.add(rhs.y),
            z: self.add(rhs.z),
            w: self.add(rhs.w),
        }
    }
}

impl Sub<Vec4> for Vec4 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x.sub(rhs.x),
            y: self.y.sub(rhs.y),
            z: self.z.sub(rhs.z),
            w: self.w.sub(rhs.w),
        }
    }
}

impl SubAssign<Vec4> for Vec4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Vec4) {
        self.x.sub_assign(rhs.x);
        self.y.sub_assign(rhs.y);
        self.z.sub_assign(rhs.z);
        self.w.sub_assign(rhs.w);
    }
}

impl Sub<X64> for Vec4 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: X64) -> Self {
        Self {
            x: self.x.sub(rhs),
            y: self.y.sub(rhs),
            z: self.z.sub(rhs),
            w: self.w.sub(rhs),
        }
    }
}

impl SubAssign<X64> for Vec4 {
    #[inline]
    fn sub_assign(&mut self, rhs: X64) {
        self.x.sub_assign(rhs);
        self.y.sub_assign(rhs);
        self.z.sub_assign(rhs);
        self.w.sub_assign(rhs);
    }
}

impl Sub<Vec4> for X64 {
    type Output = Vec4;
    #[inline]
    fn sub(self, rhs: Vec4) -> Vec4 {
        Vec4 {
            x: self.sub(rhs.x),
            y: self.sub(rhs.y),
            z: self.sub(rhs.z),
            w: self.sub(rhs.w),
        }
    }
}

impl Rem<Vec4> for Vec4 {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        Self {
            x: self.x.rem(rhs.x),
            y: self.y.rem(rhs.y),
            z: self.z.rem(rhs.z),
            w: self.w.rem(rhs.w),
        }
    }
}

impl RemAssign<Vec4> for Vec4 {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        self.x.rem_assign(rhs.x);
        self.y.rem_assign(rhs.y);
        self.z.rem_assign(rhs.z);
        self.w.rem_assign(rhs.w);
    }
}

impl Rem<X64> for Vec4 {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: X64) -> Self {
        Self {
            x: self.x.rem(rhs),
            y: self.y.rem(rhs),
            z: self.z.rem(rhs),
            w: self.w.rem(rhs),
        }
    }
}

impl RemAssign<X64> for Vec4 {
    #[inline]
    fn rem_assign(&mut self, rhs: X64) {
        self.x.rem_assign(rhs);
        self.y.rem_assign(rhs);
        self.z.rem_assign(rhs);
        self.w.rem_assign(rhs);
    }
}

impl Rem<Vec4> for X64 {
    type Output = Vec4;
    #[inline]
    fn rem(self, rhs: Vec4) -> Vec4 {
        Vec4 {
            x: self.rem(rhs.x),
            y: self.rem(rhs.y),
            z: self.rem(rhs.z),
            w: self.rem(rhs.w),
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl AsRef<[X64; 4]> for Vec4 {
    #[inline]
    fn as_ref(&self) -> &[X64; 4] {
        unsafe { &*(self as *const Vec4 as *const [X64; 4]) }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl AsMut<[X64; 4]> for Vec4 {
    #[inline]
    fn as_mut(&mut self) -> &mut [X64; 4] {
        unsafe { &mut *(self as *mut Vec4 as *mut [X64; 4]) }
    }
}

impl Sum for Vec4 {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::ZERO, Self::add)
    }
}

impl<'a> Sum<&'a Self> for Vec4 {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::ZERO, |a, &b| Self::add(a, b))
    }
}

impl Product for Vec4 {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::ONE, Self::mul)
    }
}

impl<'a> Product<&'a Self> for Vec4 {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::ONE, |a, &b| Self::mul(a, b))
    }
}

impl Neg for Vec4 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: self.x.neg(),
            y: self.y.neg(),
            z: self.z.neg(),
            w: self.w.neg(),
        }
    }
}

impl Index<usize> for Vec4 {
    type Output = X64;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("index out of bounds"),
        }
    }
}

impl IndexMut<usize> for Vec4 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("index out of bounds"),
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl fmt::Display for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}, {}]", self.x, self.y, self.z, self.w)
    }
}

#[cfg(not(target_arch = "spirv"))]
impl fmt::Debug for Vec4 {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_tuple(stringify!(Vec4))
            .field(&self.x)
            .field(&self.y)
            .field(&self.z)
            .field(&self.w)
            .finish()
    }
}

impl From<[X64; 4]> for Vec4 {
    #[inline]
    fn from(a: [X64; 4]) -> Self {
        Self::new(a[0], a[1], a[2], a[3])
    }
}

impl From<Vec4> for [X64; 4] {
    #[inline]
    fn from(v: Vec4) -> Self {
        [v.x, v.y, v.z, v.w]
    }
}

impl From<(X64, X64, X64, X64)> for Vec4 {
    #[inline]
    fn from(t: (X64, X64, X64, X64)) -> Self {
        Self::new(t.0, t.1, t.2, t.3)
    }
}

impl From<Vec4> for (X64, X64, X64, X64) {
    #[inline]
    fn from(v: Vec4) -> Self {
        (v.x, v.y, v.z, v.w)
    }
}

impl From<(Vec3, X64)> for Vec4 {
    #[inline]
    fn from((v, w): (Vec3, X64)) -> Self {
        Self::new(v.x, v.y, v.z, w)
    }
}

impl From<(X64, Vec3)> for Vec4 {
    #[inline]
    fn from((x, v): (X64, Vec3)) -> Self {
        Self::new(x, v.x, v.y, v.z)
    }
}

impl From<(Vec2, X64, X64)> for Vec4 {
    #[inline]
    fn from((v, z, w): (Vec2, X64, X64)) -> Self {
        Self::new(v.x, v.y, z, w)
    }
}

impl From<(Vec2, Vec2)> for Vec4 {
    #[inline]
    fn from((v, u): (Vec2, Vec2)) -> Self {
        Self::new(v.x, v.y, u.x, u.y)
    }
}
