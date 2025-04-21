use rayon::prelude::*;

pub type FixedPoint = i32;

const SCALE_FACTOR: FixedPoint = 65_536; // 10^9 for 9 decimal places
const EPSILON: i64 = 1; // Equivalent to 1e-5 in fixed-point

pub const ONE: FixedPoint = SCALE_FACTOR; // 1.0 in fixed-point

pub trait FixedPointExt {
    fn one() -> Self;
    fn normalize(n: usize) -> Self;
    fn mult(self, other: Self) -> Self;
}

impl FixedPointExt for FixedPoint {
    fn one() -> Self {
        ONE
    }

    fn normalize(n: usize) -> Self {
        (n as FixedPoint * SCALE_FACTOR) as Self
    }

    fn mult(self, other: Self) -> Self {
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

pub fn rmsnorm_fixed(o: &mut [FixedPoint], x: &[FixedPoint], weight: &[FixedPoint], size: usize) {
    debug_assert_eq!(o.len(), size);
    debug_assert_eq!(x.len(), size);
    debug_assert_eq!(weight.len(), size);

    let ss: i64 = x
        .iter()
        .map(|&xi| ((xi as i64 * xi as i64) >> 16))
        .sum::<i64>()
        / size as i64;

    let scale: FixedPoint = fixed_inv_sqrt((ss + EPSILON) as FixedPoint);

    for j in 0..size {
        o[j] = ((weight[j] as i64 * scale as i64 * x[j] as i64) >> 32) as FixedPoint;
    }
}

/// Compute `1 / sqrt(value)` in fixed-point (`FixedPoint`) using Newton-Raphson.
/// Assumes `value` is in fixed-point format (`SCALE_FACTOR` precision).
fn fixed_inv_sqrt(value: FixedPoint) -> FixedPoint {
    debug_assert!(value > 0, "Input must be positive");

    let mut x = SCALE_FACTOR; // initial guess (1.0)
    let mut v = value;

    while v > SCALE_FACTOR * 10 {
        v /= 4;
        x /= 2;
    }
    while v < SCALE_FACTOR / 10 {
        v *= 4;
        x *= 2;
    }

    for _ in 0..3 {
        let x_i128 = x as i128;
        let v_i128 = v as i128;
        let scale_i128 = SCALE_FACTOR as i128;

        let vx_sq = (v_i128 * x_i128 / scale_i128) * x_i128 / scale_i128;
        let three_scale_minus_vx_sq = (3 * scale_i128) - vx_sq;

        x = ((x_i128 * three_scale_minus_vx_sq / 2) / scale_i128) as FixedPoint;
    }

    x
}

/// Apply softmax normalization in-place using fixed-point `FixedPoint`
pub fn softmax_fixed(x: &mut [FixedPoint]) {
    let max_val = *x.iter().max().unwrap();
    let mut sum: i64 = 0;
    let mut exp_values: Vec<FixedPoint> = vec![0; x.len()];

    for (i, xi) in x.iter().enumerate() {
        let shifted_x = xi - max_val;
        exp_values[i] = exp_fixed(shifted_x);
        sum += exp_values[i] as i64;
    }

    for i in 0..x.len() {
        x[i] = ((exp_values[i] as i64 * SCALE_FACTOR as i64) / sum) as FixedPoint;
    }
}

/// Fixed-Point Matrix multiplication: W (d,n) @ x (n,) -> xout (d,)
/// Parallelized version using Rayon
#[cfg(feature = "parallel")]
pub fn matmul_fixed(
    xout: &mut [FixedPoint],
    x: &[FixedPoint],
    w: &[FixedPoint],
    n: usize,
    d: usize,
) {
    debug_assert_eq!(xout.len(), d);
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(w.len(), d * n);

    xout.par_iter_mut().enumerate().for_each(|(i, val)| {
        let row_start = i * n;
        *val = w[row_start..row_start + n]
            .iter()
            .zip(x.iter())
            .map(|(&w_ij, &x_j)| ((w_ij as i64 * x_j as i64) >> 16) as i64)
            .sum::<i64>() as FixedPoint;
    });
}

/// Fixed-Point Matrix multiplication: W (d,n) @ x (n,) -> xout (d,)
/// Sequential version for when parallel feature is not enabled
#[cfg(not(feature = "parallel"))]
pub fn matmul_fixed(
    xout: &mut [FixedPoint],
    x: &[FixedPoint],
    w: &[FixedPoint],
    n: usize,
    d: usize,
) {
    // Verify dimensions
    debug_assert_eq!(xout.len(), d);
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(w.len(), d * n);

    // Sequential iteration over rows
    for i in 0..d {
        let row_start = i * n;
        xout[i] = w[row_start..row_start + n]
            .iter()
            .zip(x.iter())
            .map(|(&w_ij, &x_j)| (w_ij as i128 * x_j as i128) / SCALE_FACTOR as i128) // Fixed-point multiplication
            .sum::<i128>() as FixedPoint;
    }
}

// Precomputed powers of `e` for small integers
const E_POWERS: [FixedPoint; 11] = [
    65_536,        // e^0 = 1.0
    178_145,       // e^1 = 2.718281828 * 65_536
    484_249,       // e^2 = 7.389056099 * 65_536
    1_316_325,     // e^3 = 20.085536923 * 65_536
    3_578_144,     // e^4 = 54.598150033 * 65_536
    9_726_404,     // e^5 = 148.413159103 * 65_536
    26_439_109,    // e^6 = 403.428793492 * 65_536
    71_868_952,    // e^7 = 1096.633158428 * 65_536
    195_360_063,   // e^8 = 2980.957987041 * 65_536
    531_043_712,   // e^9 = 8103.083927576 * 65_536
    1_443_526_462, // e^10 = 22026.4657948067 * 65_536
];

// Compute exp(x) in fixed-point arithmetic using Taylor series
pub fn exp_fixed(x: FixedPoint) -> FixedPoint {
    if x == 0 {
        return SCALE_FACTOR;
    }

    let mut int_part = x / SCALE_FACTOR;
    let mut frac_part = x % SCALE_FACTOR;

    // Correct handling for negative fractional parts
    if frac_part < 0 {
        frac_part += SCALE_FACTOR;
        int_part -= 1;
    }

    if frac_part == 0 && int_part.abs() < E_POWERS.len() as FixedPoint {
        return if int_part >= 0 {
            E_POWERS[int_part as usize]
        } else {
            ((SCALE_FACTOR as i64 * SCALE_FACTOR as i64) / E_POWERS[int_part.abs() as usize] as i64)
                as FixedPoint
        };
    }

    let mut result = if int_part.abs() < E_POWERS.len() as FixedPoint {
        if int_part >= 0 {
            E_POWERS[int_part as usize]
        } else {
            ((SCALE_FACTOR as i64 * SCALE_FACTOR as i64) / E_POWERS[int_part.abs() as usize] as i64)
                as FixedPoint
        }
    } else {
        let mut base = 178_145; // e^1
        let mut exp = int_part.abs();
        let mut value = SCALE_FACTOR;
        while exp > 0 {
            if exp % 2 == 1 {
                value = ((value as i64 * base as i64) >> 16) as FixedPoint;
            }
            base = ((base as i64 * base as i64) >> 16) as FixedPoint;
            exp /= 2;
        }
        if int_part > 0 {
            value
        } else {
            ((SCALE_FACTOR as i64 * SCALE_FACTOR as i64) / value as i64) as FixedPoint
        }
    };

    if frac_part != 0 {
        let frac_result = fixed_exp_chebyshev(frac_part);
        result = ((result as i64 * frac_result as i64) >> 16) as FixedPoint;
    }

    result
}

// Chebyshev coefficients for exp(x) on [-1,1] (Q16.16)
const C0: FixedPoint = 65_536; // 1.0 * 65_536
const C1: FixedPoint = 65_536; // 1.0 * 65_536
const C2: FixedPoint = 32_768; // 0.5 * 65_536
const C3: FixedPoint = 10_923; // 1/6 ≈ 0.1666667 * 65_536
const C4: FixedPoint = 2_731; // 1/24 ≈ 0.0416667 * 65_536
const C5: FixedPoint = 546; // 1/120 ≈ 0.0083333 * 65_536
const C6: FixedPoint = 91; // 1/720 ≈ 0.0013889 * 65_536
const C7: FixedPoint = 13; // 1/5040 ≈ 0.0001984 * 65_536

// Chebyshev approximation for e^x when x ∈ [-1,1]
fn fixed_exp_chebyshev(x: FixedPoint) -> FixedPoint {
    let x2 = ((x as i64 * x as i64) >> 16) as FixedPoint; // x^2
    let x3 = ((x2 as i64 * x as i64) >> 16) as FixedPoint; // x^3
    let x4 = ((x3 as i64 * x as i64) >> 16) as FixedPoint; // x^4
    let x5 = ((x4 as i64 * x as i64) >> 16) as FixedPoint; // x^5
    let x6 = ((x5 as i64 * x as i64) >> 16) as FixedPoint; // x^6
    let x7 = ((x6 as i64 * x as i64) >> 16) as FixedPoint; // x^7

    let result = C0
        + ((C1 as i64 * x as i64) >> 16) as FixedPoint
        + ((C2 as i64 * x2 as i64) >> 16) as FixedPoint
        + ((C3 as i64 * x3 as i64) >> 16) as FixedPoint
        + ((C4 as i64 * x4 as i64) >> 16) as FixedPoint
        + ((C5 as i64 * x5 as i64) >> 16) as FixedPoint
        + ((C6 as i64 * x6 as i64) >> 16) as FixedPoint
        + ((C7 as i64 * x7 as i64) >> 16) as FixedPoint;

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::matmul;
    use crate::utils::rmsnorm;
    use crate::utils::softmax;

    #[test]
    fn test_fixed_exp() {
        let test_cases = vec![
            -2.0,       // e^(-2)
            -1.0,       // e^(-1)
            0.0,        // e^0 = 1
            0.5,        // e^(0.5)
            1.0,        // e^1 = 2.718...
            2.0,        // e^2 = 7.389...
            3.0,        // e^3
            4.0,        // e^4
            5.0,        // e^5
            6.0,        // e^6
            7.0,        // e^7
            8.0,        // e^8
            9.0,        // e^9
            -0.5,       // e^(-0.5)
            -0.25,      // e^(-0.25)
            -0.75,      // e^(-0.75)
            -0.01,      // e^(-0.01)
            -0.1,       // e^(-0.1)
            -0.001,     // e^(-0.001)
            -25.824846, // e^(-25.824846)
            -20.234,    // e^(-20.234)
            -7.4335403, -8.710781, -9.919607, -10.041857, -7.064037, -8.489454, -11.895886,
            -6.072899, -11.976686, -11.888927, -8.481754, -30.182629,
            -3.14159, // e^(-3.14159)
        ];

        for &x in &test_cases {
            let x_fixed = encode_fixed(x);
            let exp_fixed = exp_fixed(x_fixed);
            let result = decode_fixed(exp_fixed);
            let expected = x.exp(); // Rust's built-in exp function

            let error = (result - expected).abs();
            let tolerance = 0.0001; // Allow a small error

            println!(
                "Testing e^{:.3}: Expected = {:.9}, Got = {:.9}, Error = {:.9}",
                x, expected, result, error
            );

            assert!(error < tolerance, "Mismatch in fixed_exp for e^{:.3}", x);
        }
    }

    #[test]
    fn test_rmsnorm_fixed_against_f32() {
        // Input vector (original floating-point values)
        let x_f32 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Weights for normalization (arbitrary test values)
        let weight_f32 = vec![0.5, 0.5, 0.5, 0.5, 0.5];

        // Convert `f32` to `FixedPoint` fixed-point
        let x_i64: Vec<FixedPoint> = x_f32.iter().map(|&v| encode_fixed(v)).collect();
        let weight_i64: Vec<FixedPoint> = weight_f32.iter().map(|&v| encode_fixed(v)).collect();
        let size = x_i64.len();

        // Output buffers
        let mut output_i64 = vec![0; size];
        let mut output_f32 = vec![0.0; size];

        // Run RMSNorm using floating-point reference implementation
        rmsnorm(&mut output_f32, &x_f32, &weight_f32, size);

        // Run RMSNorm using fixed-point implementation
        rmsnorm_fixed(&mut output_i64, &x_i64, &weight_i64, size);

        // Convert results back to `f32`
        let output_fixed_f32: Vec<f32> = output_i64.iter().map(|&v| decode_fixed(v)).collect();

        // Print results for debugging
        println!("Expected (f32): {:?}", output_f32);
        println!("Output (Fixed FixedPoint → f32): {:?}", output_fixed_f32);

        // Assert values are close within a small tolerance
        let tolerance = 1e-5;
        for (expected, actual) in output_f32.iter().zip(output_fixed_f32.iter()) {
            assert!(
                (expected - actual).abs() < tolerance,
                "Mismatch: expected={}, got={}",
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_softmax_fixed() {
        let x_f32 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Convert `f32` to `FixedPoint`
        let mut x_i64: Vec<FixedPoint> = x_f32.iter().map(|&v| encode_fixed(v)).collect();

        // Run fixed-point softmax
        softmax_fixed(&mut x_i64);

        // Convert back to `f32`
        let output_f32: Vec<f32> = x_i64.iter().map(|&v| decode_fixed(v)).collect();

        // Compute expected output using floating-point softmax
        let mut expected_f32 = x_f32.clone();
        softmax(&mut expected_f32);

        // Print results for debugging
        println!("Expected (f32): {:?}", expected_f32);
        println!("Output (Fixed FixedPoint → f32): {:?}", output_f32);

        // Compare results with a small tolerance
        let tolerance = 1e-5;
        for (expected, actual) in expected_f32.iter().zip(output_f32.iter()) {
            assert!(
                (expected - actual).abs() < tolerance,
                "Mismatch: expected={}, got={}",
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_matmul_fixed() {
        let x_f32 = vec![1.0, 2.0, 3.0];
        let w_f32 = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // 2x3 matrix
        let d = 2;
        let n = 3;

        // Convert to fixed-point
        let x_i64: Vec<FixedPoint> = x_f32.iter().map(|&v| encode_fixed(v)).collect();
        let w_i64: Vec<FixedPoint> = w_f32.iter().map(|&v| encode_fixed(v)).collect();
        let mut output_i64 = vec![0; d];
        let mut output_f32 = vec![0.0; d];

        // Run floating-point reference implementation
        matmul(&mut output_f32, &x_f32, &w_f32, n, d);

        // Run fixed-point matrix multiplication
        matmul_fixed(&mut output_i64, &x_i64, &w_i64, n, d);

        // Convert back to f32
        let output_fixed_f32: Vec<f32> = output_i64.iter().map(|&v| decode_fixed(v)).collect();

        // Print debug output
        println!("Expected (f32): {:?}", output_f32);
        println!("Output (Fixed FixedPoint → f32): {:?}", output_fixed_f32);

        // Assert values are close within a small tolerance
        let tolerance = 1e-5;
        for (expected, actual) in output_f32.iter().zip(output_fixed_f32.iter()) {
            assert!(
                (expected - actual).abs() < tolerance,
                "Mismatch: expected={}, got={}",
                expected,
                actual
            );
        }
    }
}
