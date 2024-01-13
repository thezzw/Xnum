/*
Conversion from quaternions to Euler rotation sequences.

From: http://bediyap.com/programming/convert-quaternion-to-euler-rotations/
*/

use crate::*;
use crate::quat::*;

/// Euler rotation sequences.
///
/// The angles are applied starting from the right.
/// E.g. XYZ will first apply the z-axis rotation.
///
/// YXZ can be used for yaw (y-axis), pitch (x-axis), roll (z-axis).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum XEulerRot {
    /// Intrinsic three-axis rotation ZYX
    ZYX,
    /// Intrinsic three-axis rotation ZXY
    ZXY,
    /// Intrinsic three-axis rotation YXZ
    YXZ,
    /// Intrinsic three-axis rotation YZX
    YZX,
    /// Intrinsic three-axis rotation XYZ
    XYZ,
    /// Intrinsic three-axis rotation XZY
    XZY,
}

impl Default for XEulerRot {
    /// Default `YXZ` as yaw (y-axis), pitch (x-axis), roll (z-axis).
    fn default() -> Self {
        Self::YXZ
    }
}

/// Conversion from quaternion to euler angles.
pub(crate) trait EulerFromQuaternion<Q: Copy>: Sized + Copy {
    type Output;
    /// Compute the angle of the first axis (X-x-x)
    fn first(self, q: Q) -> Self::Output;
    /// Compute then angle of the second axis (x-X-x)
    fn second(self, q: Q) -> Self::Output;
    /// Compute then angle of the third axis (x-x-X)
    fn third(self, q: Q) -> Self::Output;

    /// Compute all angles of a rotation in the notation order
    fn convert_quat(self, q: Q) -> (Self::Output, Self::Output, Self::Output);

    #[doc(hidden)]
    fn sine_theta(self, q: Q) -> Self::Output;
}

/// Conversion from euler angles to quaternion.
pub(crate) trait EulerToQuaternion<T>: Copy {
    type Output;
    /// Create the rotation quaternion for the three angles of this euler rotation sequence.
    fn new_quat(self, u: T, v: T, w: T) -> Self::Output;
}

macro_rules! impl_from_quat {
    ($t:ty, $quat:ident) => {
        impl EulerFromQuaternion<$quat> for XEulerRot {
            type Output = $t;

            fn sine_theta(self, q: $quat) -> $t {
                use XEulerRot::*;
                match self {
                    ZYX => -X64::ONE * 2 * (q.x * q.z - q.w * q.y),
                    ZXY => X64::ONE * 2 * (q.y * q.z + q.w * q.x),
                    YXZ => -X64::ONE * 2 * (q.y * q.z - q.w * q.x),
                    YZX => X64::ONE * 2 * (q.x * q.y + q.w * q.z),
                    XYZ => X64::ONE * 2 * (q.x * q.z + q.w * q.y),
                    XZY => -X64::ONE * 2 * (q.x * q.y - q.w * q.z),
                }
                .clamp(X64::NEG_ONE, X64::ONE)
            }

            fn first(self, q: $quat) -> $t {
                use XEulerRot::*;

                let sine_theta = self.sine_theta(q);
                if sine_theta.abs() > 0.99999 {
                    let scale = X64::ONE * 2 * sine_theta.signum();

                    match self {
                        ZYX => scale * X64::atan2(-q.x, q.w),
                        ZXY => scale * X64::atan2(q.y, q.w),
                        YXZ => scale * X64::atan2(-q.z, q.w),
                        YZX => scale * X64::atan2(q.x, q.w),
                        XYZ => scale * X64::atan2(q.z, q.w),
                        XZY => scale * X64::atan2(-q.y, q.w),
                    }
                } else {
                    match self {
                        ZYX => X64::atan2(
                            X64::ONE * 2 * (q.x * q.y + q.w * q.z),
                            q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z,
                        ),
                        ZXY => X64::atan2(
                            -X64::ONE * 2 * (q.x * q.y - q.w * q.z),
                            q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z,
                        ),
                        YXZ => X64::atan2(
                            X64::ONE * 2 * (q.x * q.z + q.w * q.y),
                            q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
                        ),
                        YZX => X64::atan2(
                            -X64::ONE * 2 * (q.x * q.z - q.w * q.y),
                            q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z,
                        ),
                        XYZ => X64::atan2(
                            -X64::ONE * 2 * (q.y * q.z - q.w * q.x),
                            q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
                        ),
                        XZY => X64::atan2(
                            X64::ONE * 2 * (q.y * q.z + q.w * q.x),
                            q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z,
                        ),
                    }
                }
            }

            fn second(self, q: $quat) -> $t {
                self.sine_theta(q).asin().1
            }

            fn third(self, q: $quat) -> $t {
                use XEulerRot::*;
                if self.sine_theta(q).abs() > 0.99999 {
                    X64::ZERO
                } else {
                    match self {
                        ZYX => X64::atan2(
                            X64::ONE * 2 * (q.y * q.z + q.w * q.x),
                            q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
                        ),
                        ZXY => X64::atan2(
                            -X64::ONE * 2 * (q.x * q.z - q.w * q.y),
                            q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
                        ),
                        YXZ => X64::atan2(
                            X64::ONE * 2 * (q.x * q.y + q.w * q.z),
                            q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z,
                        ),
                        YZX => X64::atan2(
                            -X64::ONE * 2 * (q.y * q.z - q.w * q.x),
                            q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z,
                        ),
                        XYZ => X64::atan2(
                            -X64::ONE * 2 * (q.x * q.y - q.w * q.z),
                            q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z,
                        ),
                        XZY => X64::atan2(
                            X64::ONE * 2 * (q.x * q.z + q.w * q.y),
                            q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z,
                        ),
                    }
                }
            }

            fn convert_quat(self, q: $quat) -> ($t, $t, $t) {
                use XEulerRot::*;

                let sine_theta = self.sine_theta(q);
                let second = sine_theta.asin().1;

                if sine_theta.abs() > 0.99999 {
                    let scale = X64::ONE * 2 * sine_theta.signum();

                    return match self {
                        ZYX => (scale * X64::atan2(-q.x, q.w), second, X64::ZERO),
                        ZXY => (scale * X64::atan2(q.y, q.w), second, X64::ZERO),
                        YXZ => (scale * X64::atan2(-q.z, q.w), second, X64::ZERO),
                        YZX => (scale * X64::atan2(q.x, q.w), second, X64::ZERO),
                        XYZ => (scale * X64::atan2(q.z, q.w), second, X64::ZERO),
                        XZY => (scale * X64::atan2(-q.y, q.w), second, X64::ZERO),
                    };
                }

                let first = match self {
                    ZYX => X64::atan2(
                        X64::ONE * 2 * (q.x * q.y + q.w * q.z),
                        q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z,
                    ),
                    ZXY => X64::atan2(
                        -X64::ONE * 2 * (q.x * q.y - q.w * q.z),
                        q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z,
                    ),
                    YXZ => X64::atan2(
                        X64::ONE * 2 * (q.x * q.z + q.w * q.y),
                        q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
                    ),
                    YZX => X64::atan2(
                        -X64::ONE * 2 * (q.x * q.z - q.w * q.y),
                        q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z,
                    ),
                    XYZ => X64::atan2(
                        -X64::ONE * 2 * (q.y * q.z - q.w * q.x),
                        q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
                    ),
                    XZY => X64::atan2(
                        X64::ONE * 2 * (q.y * q.z + q.w * q.x),
                        q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z,
                    ),
                };

                let third = match self {
                    ZYX => X64::atan2(
                        X64::ONE * 2 * (q.y * q.z + q.w * q.x),
                        q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
                    ),
                    ZXY => X64::atan2(
                        -X64::ONE * 2 * (q.x * q.z - q.w * q.y),
                        q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
                    ),
                    YXZ => X64::atan2(
                        X64::ONE * 2 * (q.x * q.y + q.w * q.z),
                        q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z,
                    ),
                    YZX => X64::atan2(
                        -X64::ONE * 2 * (q.y * q.z - q.w * q.x),
                        q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z,
                    ),
                    XYZ => X64::atan2(
                        -X64::ONE * 2 * (q.x * q.y - q.w * q.z),
                        q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z,
                    ),
                    XZY => X64::atan2(
                        X64::ONE * 2 * (q.x * q.z + q.w * q.y),
                        q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z,
                    ),
                };

                (first, second, third)
            }
        }
        // End - impl EulerFromQuaternion
    };
}

macro_rules! impl_to_quat {
    ($t:ty, $quat:ident) => {
        impl EulerToQuaternion<$t> for XEulerRot {
            type Output = $quat;
            #[inline(always)]
            fn new_quat(self, u: $t, v: $t, w: $t) -> $quat {
                use XEulerRot::*;
                #[inline(always)]
                fn rot_x(a: $t) -> $quat {
                    $quat::from_rotation_x(a)
                }
                #[inline(always)]
                fn rot_y(a: $t) -> $quat {
                    $quat::from_rotation_y(a)
                }
                #[inline(always)]
                fn rot_z(a: $t) -> $quat {
                    $quat::from_rotation_z(a)
                }
                match self {
                    ZYX => rot_z(u) * rot_y(v) * rot_x(w),
                    ZXY => rot_z(u) * rot_x(v) * rot_y(w),
                    YXZ => rot_y(u) * rot_x(v) * rot_z(w),
                    YZX => rot_y(u) * rot_z(v) * rot_x(w),
                    XYZ => rot_x(u) * rot_y(v) * rot_z(w),
                    XZY => rot_x(u) * rot_z(v) * rot_y(w),
                }
                .normalize()
            }
        }
        // End - impl EulerToQuaternion
    };
}

impl_from_quat!(X64, XQuat);
impl_to_quat!(X64, XQuat);
