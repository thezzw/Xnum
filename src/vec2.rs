use crate::*;
use crate::vec3::*;

#[cfg(not(target_arch = "spirv"))]
use core::fmt;
use core::iter::{Product, Sum};
use core::ops::*;

/// Creates a 2-dimensional vector.
#[inline(always)]
#[must_use]
pub const fn vec2(x: X64, y: X64) -> Vec2 {
    Vec2::new(x, y)
}

/// A 2-dimensional vector.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "cuda", repr(align(16)))]
#[cfg_attr(not(target_arch = "spirv"), repr(C))]
#[cfg_attr(target_arch = "spirv", repr(simd))]
pub struct Vec2 {
    pub x: X64,
    pub y: X64,
}

impl Vec2 {
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
    pub const X: Self = Self::new(X64::ONE, X64::ZERO);

    /// A unit vector pointing along the positive Y axis.
    pub const Y: Self = Self::new(X64::ZERO, X64::ONE);

    /// A unit vector pointing along the negative X axis.
    pub const NEG_X: Self = Self::new(X64::NEG_ONE, X64::ZERO);

    /// A unit vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self::new(X64::ZERO, X64::NEG_ONE);

    /// The unit axes.
    pub const AXES: [Self; 2] = [Self::X, Self::Y];

    /// Creates a new vector.
    #[inline(always)]
    #[must_use]
    pub const fn new(x: X64, y: X64) -> Self {
        Self { x, y }
    }

    /// Creates a vector with all elements set to `v`.
    #[inline]
    #[must_use]
    pub const fn splat(v: X64) -> Self {
        Self { x: v, y: v }
    }

    /// Creates a new vector from an array.
    #[inline]
    #[must_use]
    pub const fn from_array(a: [X64; 2]) -> Self {
        Self::new(a[0], a[1])
    }

    /// `[x, y]`
    #[inline]
    #[must_use]
    pub const fn to_array(&self) -> [X64; 2] {
        [self.x, self.y]
    }

    /// Creates a vector from the first 2 values in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 2 elements long.
    #[inline]
    #[must_use]
    pub const fn from_slice(slice: &[X64]) -> Self {
        Self::new(slice[0], slice[1])
    }

    /// Writes the elements of `self` to the first 2 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 2 elements long.
    #[inline]
    pub fn write_to_slice(self, slice: &mut [X64]) {
        slice[0] = self.x;
        slice[1] = self.y;
    }

    /// Creates a 3D vector from `self` and the given `z` value.
    #[inline]
    #[must_use]
    pub const fn extend(self, z: X64) -> Vec3 {
        Vec3::new(self.x, self.y, z)
    }

    /// Computes the dot product of `self` and `rhs`.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> X64 {
        (self.x * rhs.x) + (self.y * rhs.y)
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
        }
    }

    /// Component-wise clamping of values, similar to [`X64::clamp`].
    ///
    /// Each element in `min` must be less-or-equal to the corresponding element in `max`.
    ///
    /// # Panics
    ///
    /// Will panic if `min` is greater than `max` when `glam_assert` is enabled.
    #[inline]
    #[must_use]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        assert!(min.x.le(&max.x) && min.y.le(&max.y), "clamp: expected min <= max");
        self.max(min).min(max)
    }

    /// Returns the horizontal minimum of `self`.
    ///
    /// In other words this computes `min(x, y, ..)`.
    #[inline]
    #[must_use]
    pub fn min_element(self) -> X64 {
        self.x.min(self.y)
    }

    /// Returns the horizontal maximum of `self`.
    ///
    /// In other words this computes `max(x, y, ..)`.
    #[inline]
    #[must_use]
    pub fn max_element(self) -> X64 {
        self.x.max(self.y)
    }

    /// Returns a vector containing the absolute value of each element of `self`.
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }

    /// Returns a vector with elements representing the sign of `self`.
    ///
    /// - `1.0` if the number is positive, `+0.0` or `INFINITY`
    /// - `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
    /// - `NAN` if the number is `NAN`
    #[inline]
    #[must_use]
    pub fn signum(self) -> Self {
        Self {
            x: self.x.signum(),
            y: self.y.signum(),
        }
    }

    /// Returns `true` if, and only if, all elements are finite.  If any element is either
    /// `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan()
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

    /// Computes `1.0 / length()`.
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
        )
    }

    /// Returns `self` normalized to length 1.0.
    ///
    /// For valid results, `self` must _not_ be of length zero, nor very close to zero.
    ///
    /// See also [`Self::try_normalize()`] and [`Self::normalize_or_zero()`].
    ///
    /// Panics
    ///
    /// Will panic if `self` is zero length when `glam_assert` is enabled.
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        #[allow(clippy::let_and_return)]
        let normalized = self.mul(self.length_recip());
        assert!(normalized.is_finite());
        normalized
    }

    /// Returns `self` normalized to length 1.0 if possible, else returns `None`.
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

    /// Returns `self` normalized to length 1.0 if possible, else returns zero.
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

    /// Returns whether `self` is length `1.0` or not.
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
    /// Will panic if `rhs` is zero length when `glam_assert` is enabled.
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
    /// Will panic if `rhs` has a length of zero when `glam_assert` is enabled.
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
    /// Will panic if `rhs` is not normalized when `glam_assert` is enabled.
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
    /// Will panic if `rhs` is not normalized when `glam_assert` is enabled.
    #[inline]
    #[must_use]
    pub fn reject_from_normalized(self, rhs: Self) -> Self {
        self - self.project_onto_normalized(rhs)
    }

    /// Returns a vector containing the nearest integer to a number for each element of `self`.
    /// Round half-way cases away from 0.0.
    #[inline]
    #[must_use]
    pub fn round(self) -> Self {
        Self {
            x: self.x.round(),
            y: self.y.round(),
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
        Self::new(self.x.exp(), self.y.exp())
    }

    /// Returns a vector containing each element of `self` raised to the power of `n`.
    #[inline]
    #[must_use]
    pub fn powf(self, n: X64) -> Self {
        Self::new(self.x.powf(n), self.y.powf(n))
    }

    /// Returns a vector containing the reciprocal `1.0/n` of each element of `self`.
    #[inline]
    #[must_use]
    pub fn recip(self) -> Self {
        Self {
            x: X64::ONE / self.x,
            y: X64::ONE / self.y,
        }
    }

    /// Performs a linear interpolation between `self` and `rhs` based on the value `s`.
    ///
    /// When `s` is `0.0`, the result will be equal to `self`.  When `s` is `1.0`, the result
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
        dif.x.le(&gap.x) && dif.y.le(&gap.y)
    }

    /// Returns a vector with a length no less than `min` and no more than `max`
    ///
    /// # Panics
    ///
    /// Will panic if `min` is greater than `max` when `glam_assert` is enabled.
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
        )
    }

    /// Creates a 2D vector containing `[angle.cos(), angle.sin()]`. This can be used in
    /// conjunction with the [`rotate()`][Self::rotate()] method, e.g.
    /// `Vec2::from_angle(PI).rotate(Vec2::Y)` will create the vector `[-1, 0]`
    /// and rotate [`Vec2::Y`] around it returning `-Vec2::Y`.
    #[inline]
    #[must_use]
    pub fn from_angle(angle: X64) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self { x: cos, y: sin }
    }

    /// Returns the angle (in radians) of this vector in the range `[-π, +π]`.
    ///
    /// The input does not need to be a unit vector however it must be non-zero.
    #[inline]
    #[must_use]
    pub fn to_angle(self) -> (X64, X64) {
        (self.y / self.x).atan()
    }

    /// Returns the angle (in radians) between `self` and `rhs` in the range `[-π, +π]`.
    ///
    /// The inputs do not need to be unit vectors however they must be non-zero.
    #[inline]
    #[must_use]
    pub fn angle_between(self, rhs: Self) -> X64 {
        let angle = (
            self.dot(rhs) / (self.length_squared() * rhs.length_squared()).sqrt()
        ).acos().1;

        angle * (self.perp_dot(rhs)).signum()
    }

    /// Returns a vector that is equal to `self` rotated by 90 degrees.
    #[inline]
    #[must_use]
    pub fn perp(self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
        }
    }

    /// The perpendicular dot product of `self` and `rhs`.
    /// Also known as the wedge product, 2D cross product, and determinant.
    #[doc(alias = "wedge")]
    #[doc(alias = "cross")]
    #[doc(alias = "determinant")]
    #[inline]
    #[must_use]
    pub fn perp_dot(self, rhs: Self) -> X64 {
        (self.x * rhs.y) - (self.y * rhs.x)
    }

    /// Returns `rhs` rotated by the angle of `self`. If `self` is normalized,
    /// then this just rotation. This is what you usually want. Otherwise,
    /// it will be like a rotation with a multiplication by `self`'s length.
    #[inline]
    #[must_use]
    pub fn rotate(self, rhs: Self) -> Self {
        Self {
            x: self.x * rhs.x - self.y * rhs.y,
            y: self.y * rhs.x + self.x * rhs.y,
        }
    }
}

impl Default for Vec2 {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

impl Div<Vec2> for Vec2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self {
            x: self.x.div(rhs.x),
            y: self.y.div(rhs.y),
        }
    }
}

impl DivAssign<Vec2> for Vec2 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.x.div_assign(rhs.x);
        self.y.div_assign(rhs.y);
    }
}

impl Div<X64> for Vec2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: X64) -> Self {
        Self {
            x: self.x.div(rhs),
            y: self.y.div(rhs),
        }
    }
}

impl DivAssign<X64> for Vec2 {
    #[inline]
    fn div_assign(&mut self, rhs: X64) {
        self.x.div_assign(rhs);
        self.y.div_assign(rhs);
    }
}

impl Div<Vec2> for X64 {
    type Output = Vec2;
    #[inline]
    fn div(self, rhs: Vec2) -> Vec2 {
        Vec2 {
            x: self.div(rhs.x),
            y: self.div(rhs.y),
        }
    }
}

impl Mul<Vec2> for Vec2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            x: self.x.mul(rhs.x),
            y: self.y.mul(rhs.y),
        }
    }
}

impl MulAssign<Vec2> for Vec2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.x.mul_assign(rhs.x);
        self.y.mul_assign(rhs.y);
    }
}

impl Mul<X64> for Vec2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: X64) -> Self {
        Self {
            x: self.x.mul(rhs),
            y: self.y.mul(rhs),
        }
    }
}

impl MulAssign<X64> for Vec2 {
    #[inline]
    fn mul_assign(&mut self, rhs: X64) {
        self.x.mul_assign(rhs);
        self.y.mul_assign(rhs);
    }
}

impl Mul<Vec2> for X64 {
    type Output = Vec2;
    #[inline]
    fn mul(self, rhs: Vec2) -> Vec2 {
        Vec2 {
            x: self.mul(rhs.x),
            y: self.mul(rhs.y),
        }
    }
}

impl Add<Vec2> for Vec2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x.add(rhs.x),
            y: self.y.add(rhs.y),
        }
    }
}

impl AddAssign<Vec2> for Vec2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x.add_assign(rhs.x);
        self.y.add_assign(rhs.y);
    }
}

impl Add<X64> for Vec2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: X64) -> Self {
        Self {
            x: self.x.add(rhs),
            y: self.y.add(rhs),
        }
    }
}

impl AddAssign<X64> for Vec2 {
    #[inline]
    fn add_assign(&mut self, rhs: X64) {
        self.x.add_assign(rhs);
        self.y.add_assign(rhs);
    }
}

impl Add<Vec2> for X64 {
    type Output = Vec2;
    #[inline]
    fn add(self, rhs: Vec2) -> Vec2 {
        Vec2 {
            x: self.add(rhs.x),
            y: self.add(rhs.y),
        }
    }
}

impl Sub<Vec2> for Vec2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x.sub(rhs.x),
            y: self.y.sub(rhs.y),
        }
    }
}

impl SubAssign<Vec2> for Vec2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Vec2) {
        self.x.sub_assign(rhs.x);
        self.y.sub_assign(rhs.y);
    }
}

impl Sub<X64> for Vec2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: X64) -> Self {
        Self {
            x: self.x.sub(rhs),
            y: self.y.sub(rhs),
        }
    }
}

impl SubAssign<X64> for Vec2 {
    #[inline]
    fn sub_assign(&mut self, rhs: X64) {
        self.x.sub_assign(rhs);
        self.y.sub_assign(rhs);
    }
}

impl Sub<Vec2> for X64 {
    type Output = Vec2;
    #[inline]
    fn sub(self, rhs: Vec2) -> Vec2 {
        Vec2 {
            x: self.sub(rhs.x),
            y: self.sub(rhs.y),
        }
    }
}

impl Rem<Vec2> for Vec2 {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        Self {
            x: self.x.rem(rhs.x),
            y: self.y.rem(rhs.y),
        }
    }
}

impl RemAssign<Vec2> for Vec2 {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        self.x.rem_assign(rhs.x);
        self.y.rem_assign(rhs.y);
    }
}

impl Rem<X64> for Vec2 {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: X64) -> Self {
        Self {
            x: self.x.rem(rhs),
            y: self.y.rem(rhs),
        }
    }
}

impl RemAssign<X64> for Vec2 {
    #[inline]
    fn rem_assign(&mut self, rhs: X64) {
        self.x.rem_assign(rhs);
        self.y.rem_assign(rhs);
    }
}

impl Rem<Vec2> for X64 {
    type Output = Vec2;
    #[inline]
    fn rem(self, rhs: Vec2) -> Vec2 {
        Vec2 {
            x: self.rem(rhs.x),
            y: self.rem(rhs.y),
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl AsRef<[X64; 2]> for Vec2 {
    #[inline]
    fn as_ref(&self) -> &[X64; 2] {
        unsafe { &*(self as *const Vec2 as *const [X64; 2]) }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl AsMut<[X64; 2]> for Vec2 {
    #[inline]
    fn as_mut(&mut self) -> &mut [X64; 2] {
        unsafe { &mut *(self as *mut Vec2 as *mut [X64; 2]) }
    }
}

impl Sum for Vec2 {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::ZERO, Self::add)
    }
}

impl<'a> Sum<&'a Self> for Vec2 {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::ZERO, |a, &b| Self::add(a, b))
    }
}

impl Product for Vec2 {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::ONE, Self::mul)
    }
}

impl<'a> Product<&'a Self> for Vec2 {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::ONE, |a, &b| Self::mul(a, b))
    }
}

impl Neg for Vec2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: self.x.neg(),
            y: self.y.neg(),
        }
    }
}

impl Index<usize> for Vec2 {
    type Output = X64;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("index out of bounds"),
        }
    }
}

impl IndexMut<usize> for Vec2 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("index out of bounds"),
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.x, self.y)
    }
}

#[cfg(not(target_arch = "spirv"))]
impl fmt::Debug for Vec2 {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_tuple(stringify!(Vec2))
            .field(&self.x)
            .field(&self.y)
            .finish()
    }
}

impl From<[X64; 2]> for Vec2 {
    #[inline]
    fn from(a: [X64; 2]) -> Self {
        Self::new(a[0], a[1])
    }
}

impl From<Vec2> for [X64; 2] {
    #[inline]
    fn from(v: Vec2) -> Self {
        [v.x, v.y]
    }
}

impl From<(X64, X64)> for Vec2 {
    #[inline]
    fn from(t: (X64, X64)) -> Self {
        Self::new(t.0, t.1)
    }
}

impl From<Vec2> for (X64, X64) {
    #[inline]
    fn from(v: Vec2) -> Self {
        (v.x, v.y)
    }
}
