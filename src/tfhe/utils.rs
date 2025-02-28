use rayon::prelude::*;

const SCALE_FACTOR: i64 = 1_000_000_000; // 10^9 for 9 decimal places
const EPSILON: i64 = 10_000; // Equivalent to 1e-5 in fixed-point

pub fn rmsnorm_fixed(o: &mut [i64], x: &[i64], weight: &[i64], size: usize) {
    debug_assert_eq!(o.len(), size);
    debug_assert_eq!(x.len(), size);
    debug_assert_eq!(weight.len(), size);

    // Step 1: Compute sum of squares in fixed-point
    let ss: i128 = x.iter()
        .map(|&xi| (xi as i128 * xi as i128) / SCALE_FACTOR as i128)
        .sum::<i128>() / size as i128;

    // Step 2: Compute inverse square root (Fixed-Point)
    let scale: i64 = fixed_inv_sqrt(ss as i64 + EPSILON);

    // Step 3: Normalize and scale
    for j in 0..size {
        o[j] = ((weight[j] as i128 * scale as i128 * x[j] as i128) / (SCALE_FACTOR as i128 * SCALE_FACTOR as i128)) as i64;
    }
}

/// Compute inverse square root in fixed-point arithmetic
fn fixed_inv_sqrt(value: i64) -> i64 {
    let float_value = value as f64 / SCALE_FACTOR as f64;
    let inv_sqrt_f64 = 1.0 / float_value.sqrt();
    (inv_sqrt_f64 * SCALE_FACTOR as f64) as i64
}

/// Apply softmax normalization in-place using fixed-point `i64`
pub fn softmax_fixed(x: &mut [i64]) {
    // Find max value (for numerical stability)
    let max_val = *x.iter().max().unwrap();

    // Compute exponentials in fixed-point
    let mut sum: i128 = 0;
    let mut exp_values: Vec<i64> = vec![0; x.len()];
    
    for (i, xi) in x.iter().enumerate() {
        let shifted_x = xi - max_val; // Scale exponent input
        exp_values[i] = fixed_exp(shifted_x);
        sum += exp_values[i] as i128;
    }

    // Normalize each value
    for i in 0..x.len() {
        x[i] = ((exp_values[i] as i128 * SCALE_FACTOR as i128) / sum) as i64;
    }
}

/// Approximate `exp(x)` in fixed-point arithmetic
fn fixed_exp(x: i64) -> i64 {
    let float_x = x as f64 / SCALE_FACTOR as f64;
    let exp_f64 = float_x.exp();
    (exp_f64 * SCALE_FACTOR as f64) as i64
}

/// Fixed-Point Matrix multiplication: W (d,n) @ x (n,) -> xout (d,)
/// Parallelized version using Rayon
#[cfg(feature = "parallel")]
pub fn matmul_fixed(xout: &mut [i64], x: &[i64], w: &[i64], n: usize, d: usize) {
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
            .sum::<i128>() as i64;
    });
}

/// Fixed-Point Matrix multiplication: W (d,n) @ x (n,) -> xout (d,)
/// Sequential version for when parallel feature is not enabled
#[cfg(not(feature = "parallel"))]
pub fn matmul_fixed(xout: &mut [i64], x: &[i64], w: &[i64], n: usize, d: usize) {
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
            .sum::<i128>() as i64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::rmsnorm;
    use crate::utils::softmax;
    use crate::utils::matmul;

    /// Helper function to convert `f32` to `i64` fixed-point.
    fn encode_fixed(value: f32) -> i64 {
        (value * SCALE_FACTOR as f32) as i64
    }

    /// Helper function to convert `i64` fixed-point back to `f32`.
    fn decode_fixed(value: i64) -> f32 {
        value as f32 / SCALE_FACTOR as f32
    }

    #[test]
    fn test_rmsnorm_fixed_against_f32() {
        // Input vector (original floating-point values)
        let x_f32 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Weights for normalization (arbitrary test values)
        let weight_f32 = vec![0.5, 0.5, 0.5, 0.5, 0.5];

        // Convert `f32` to `i64` fixed-point
        let x_i64: Vec<i64> = x_f32.iter().map(|&v| encode_fixed(v)).collect();
        let weight_i64: Vec<i64> = weight_f32.iter().map(|&v| encode_fixed(v)).collect();
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
        println!("Output (Fixed i64 → f32): {:?}", output_fixed_f32);

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

        // Convert `f32` to `i64`
        let mut x_i64: Vec<i64> = x_f32.iter().map(|&v| encode_fixed(v)).collect();

        // Run fixed-point softmax
        softmax_fixed(&mut x_i64);

        // Convert back to `f32`
        let output_f32: Vec<f32> = x_i64.iter().map(|&v| decode_fixed(v)).collect();

        // Compute expected output using floating-point softmax
        let mut expected_f32 = x_f32.clone();
        softmax(&mut expected_f32);

        // Print results for debugging
        println!("Expected (f32): {:?}", expected_f32);
        println!("Output (Fixed i64 → f32): {:?}", output_f32);

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
        let w_f32 = vec![
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6
        ]; // 2x3 matrix
        let d = 2;
        let n = 3;

        // Convert to fixed-point
        let x_i64: Vec<i64> = x_f32.iter().map(|&v| encode_fixed(v)).collect();
        let w_i64: Vec<i64> = w_f32.iter().map(|&v| encode_fixed(v)).collect();
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
        println!("Output (Fixed i64 → f32): {:?}", output_fixed_f32);

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