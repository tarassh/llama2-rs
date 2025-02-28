
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::rmsnorm;

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
}