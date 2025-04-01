use rayon::prelude::*;

pub type FixedPoint = i64;

const SCALE_FACTOR: FixedPoint = 1_000_000_000; // 10^9 for 9 decimal places
const EPSILON: FixedPoint = 10_000; // Equivalent to 1e-5 in fixed-point

pub const ONE: FixedPoint = SCALE_FACTOR; // 1.0 in fixed-point

pub trait FixedPointExt {
    fn one() -> Self;
    fn normalize(n: usize) -> Self;
}

impl FixedPointExt for FixedPoint {
    fn one() -> Self {
        ONE
    }

    fn normalize(n: usize) -> Self {
        (n as FixedPoint * SCALE_FACTOR) as Self
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

    // Step 1: Compute sum of squares in fixed-point
    let ss: i128 = x
        .iter()
        .map(|&xi| (xi as i128 * xi as i128) / SCALE_FACTOR as i128)
        .sum::<i128>()
        / size as i128;

    // Step 2: Compute inverse square root (Fixed-Point)
    let scale: FixedPoint = fixed_inv_sqrt(ss as FixedPoint + EPSILON);

    // Step 3: Normalize and scale
    for j in 0..size {
        o[j] = ((weight[j] as i128 * scale as i128 * x[j] as i128)
            / (SCALE_FACTOR as i128 * SCALE_FACTOR as i128 * 2)) as FixedPoint;
    }
}

/// Compute `1 / sqrt(value)` in fixed-point (`FixedPoint`) using Newton-Raphson.
/// Assumes `value` is in fixed-point format (`SCALE_FACTOR` precision).
fn fixed_inv_sqrt(value: FixedPoint) -> FixedPoint {
    debug_assert!(value > 0, "Input must be positive");

    let mut x = SCALE_FACTOR; // Initial guess (1.0 in fixed-point)
    let mut v = value;

    // Normalize `v` to prevent overflow in large numbers
    while v > SCALE_FACTOR * 10 {
        v /= 4;
        x /= 2;
    }
    while v < SCALE_FACTOR / 10 {
        v *= 4;
        x *= 2;
    }

    // Perform Newton-Raphson iterations
    for _ in 0..3 {
        x = (x * (3 * SCALE_FACTOR - ((v * x / SCALE_FACTOR) * x / SCALE_FACTOR)) / 2)
            / SCALE_FACTOR;
    }

    x
}

/// Apply softmax normalization in-place using fixed-point `FixedPoint`
pub fn softmax_fixed(x: &mut [FixedPoint]) {
    // Find max value (for numerical stability)
    let max_val = *x.iter().max().unwrap();

    // Compute exponentials in fixed-point
    let mut sum: i128 = 0;
    let mut exp_values: Vec<FixedPoint> = vec![0; x.len()];

    for (i, xi) in x.iter().enumerate() {
        let shifted_x = xi - max_val; // Scale exponent input
        exp_values[i] = exp_fixed(shifted_x);
        sum += exp_values[i] as i128;
    }

    // Normalize each value
    for i in 0..x.len() {
        x[i] = ((exp_values[i] as i128 * SCALE_FACTOR as i128) / sum) as FixedPoint;
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
    // Verify dimensions
    debug_assert_eq!(xout.len(), d);
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(w.len(), d * n);

    // Parallel iteration over rows
    xout.par_iter_mut().enumerate().for_each(|(i, val)| {
        let row_start = i * n;
        *val = w[row_start..row_start + n]
            .iter()
            .zip(x.iter())
            .map(|(&w_ij, &x_j)| (w_ij as i128 * x_j as i128) / SCALE_FACTOR as i128) // Fixed-point multiplication
            .sum::<i128>() as FixedPoint;
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
const E_POWERS: [FixedPoint; 10] = [
    1_000_000_000,     // e^0  = 1.0
    2_718_281_828,     // e^1  = 2.718...
    7_389_056_099,     // e^2  = 7.389...
    20_085_536_923,    // e^3  = 20.085...
    54_598_150_033,    // e^4  = 54.598...
    148_413_159_103,   // e^5  = 148.413...
    403_428_793_492,   // e^6  = 403.428...
    1_096_633_158_428, // e^7
    2_980_957_987_041, // e^8
    8_103_083_927_576, // e^9
];

// Compute exp(x) in fixed-point arithmetic using Taylor series
pub fn exp_fixed(x: FixedPoint) -> FixedPoint {
    if x == 0 {
        return SCALE_FACTOR;
    }

    let int_part = x / SCALE_FACTOR;
    let frac_part = x % SCALE_FACTOR;

    // Handle integer part using E_POWERS if within range
    let mut result = if int_part.abs() < E_POWERS.len() as FixedPoint {
        if int_part >= 0 {
            E_POWERS[int_part as usize]
        } else {
            (SCALE_FACTOR * SCALE_FACTOR) / E_POWERS[int_part.abs() as usize]
        }
    } else {
        // Use exponentiation-by-squaring for large integer parts (rarely happens in your range)
        let mut base = 2_718_281_828; // e^1
        let mut exp = int_part.abs();
        let mut value = SCALE_FACTOR;
        while exp > 0 {
            if exp % 2 == 1 {
                value = (value as i128 * base as i128 / SCALE_FACTOR as i128) as FixedPoint;
            }
            base = (base as i128 * base as i128 / SCALE_FACTOR as i128) as FixedPoint;
            exp /= 2;
        }
        if int_part > 0 {
            value
        } else {
            (SCALE_FACTOR * SCALE_FACTOR) / value
        }
    };

    // Apply fractional part using Chebyshev
    if frac_part != 0 {
        let frac_result = fixed_exp_chebyshev(frac_part);
        result = (result as i128 * frac_result as i128 / SCALE_FACTOR as i128) as FixedPoint;
    }

    result
}

// Chebyshev coefficients for exp(x) on [-1,1]
const C0: FixedPoint = 1_000_000_000;
const C1: FixedPoint = 1_000_000_000;
const C2: FixedPoint = 500_000_000;
const C3: FixedPoint = 166_666_667;
const C4: FixedPoint = 41_666_667;
const C5: FixedPoint = 8_333_333;
const C6: FixedPoint = 1_388_888;
const C7: FixedPoint = 198_412;

// Chebyshev approximation for e^x when x ∈ [-1,1]
fn fixed_exp_chebyshev(x: FixedPoint) -> FixedPoint {
    let x2 = (x as i128 * x as i128 / SCALE_FACTOR as i128) as FixedPoint;
    let x3 = (x2 as i128 * x as i128 / SCALE_FACTOR as i128) as FixedPoint;
    let x4 = (x3 as i128 * x as i128 / SCALE_FACTOR as i128) as FixedPoint;
    let x5 = (x4 as i128 * x as i128 / SCALE_FACTOR as i128) as FixedPoint;
    let x6 = (x5 as i128 * x as i128 / SCALE_FACTOR as i128) as FixedPoint;
    let x7 = (x6 as i128 * x as i128 / SCALE_FACTOR as i128) as FixedPoint;

    let result = C0
        + (C1 * x) / SCALE_FACTOR
        + (C2 * x2) / SCALE_FACTOR
        + (C3 * x3) / SCALE_FACTOR
        + (C4 * x4) / SCALE_FACTOR
        + (C5 * x5) / SCALE_FACTOR
        + (C6 * x6) / SCALE_FACTOR
        + (C7 * x7) / SCALE_FACTOR;

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
            5.0,        // e^5
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
