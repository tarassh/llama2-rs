pub mod exp;
pub mod matmul;
pub mod rmsnorm;
pub mod softmax;

pub use exp::exp_fixed;
pub use matmul::matmul_fixed;
pub use rmsnorm::rmsnorm_fixed;
pub use softmax::softmax_fixed;

pub type FixedPoint = i32;

const SCALE_FACTOR: FixedPoint = 65_536; // 10^9 for 9 decimal places
const EPSILON: i64 = 1; // Equivalent to 1e-5 in fixed-point

pub const ONE: FixedPoint = SCALE_FACTOR; // 1.0 in fixed-point

pub trait FixedPointExt {
    fn one() -> Self;
    fn normalize(n: usize) -> Self;
    fn mul(self, other: Self) -> Self;
}

impl FixedPointExt for FixedPoint {
    fn one() -> Self {
        ONE
    }

    fn normalize(n: usize) -> Self {
        (n as FixedPoint * SCALE_FACTOR) as Self
    }

    fn mul(self, other: Self) -> Self {
        if self == Self::one() {
            return other;
        }
        if other == Self::one() {
            return self;
        }
        if self == -Self::one() {
            return -other;
        }
        if other == -Self::one() {
            return -self;
        }

        ((self as i64 * other as i64) >> 16) as FixedPoint
    }
}

/// Helper function to convert `f32` to `FixedPoint` fixed-point.
pub fn encode_fixed(value: f32) -> FixedPoint {
    (value * SCALE_FACTOR as f32) as FixedPoint
}

/// Helper function to convert `FixedPoint` fixed-point back to `f32`.
pub fn decode_fixed(value: FixedPoint) -> f32 {
    value as f32 / SCALE_FACTOR as f32
}
