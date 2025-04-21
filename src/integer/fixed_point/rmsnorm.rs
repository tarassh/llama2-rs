use crate::integer::fixed_point::FixedPoint;
use crate::integer::fixed_point::EPSILON;
use crate::integer::fixed_point::SCALE_FACTOR;

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
        o[j] = ((weight[j] as i64 * scale as i64 * x[j] as i64) >> 32) as FixedPoint / 2;
    }
}

/// Compute `1 / sqrt(value)` in fixed-point (`FixedPoint`) using Newton-Raphson.
/// Assumes `value` is in fixed-point format (`SCALE_FACTOR` precision).
fn fixed_inv_sqrt(value: FixedPoint) -> FixedPoint {
    debug_assert!(value > 0, "Input must be positive");

    let mut x = SCALE_FACTOR; // initial guess: 1.0 in fixed-point
    let mut v = value;

    // Normalize input to avoid overflow or underflow
    while v > SCALE_FACTOR * 4 {
        v /= 4;
        x /= 2;
    }
    while v < SCALE_FACTOR / 4 {
        v *= 4;
        x *= 2;
    }

    // Newton-Raphson iterations
    for _ in 0..3 {
        let x_i = x as i128;
        let v_i = v as i128;

        // x_{n+1} = x_n * (3 - v * x_n^2) / 2
        let x_sq = (x_i * x_i) >> 16; // x_n^2 scaled back correctly
        let vx_sq = (v_i * x_sq) >> 16; // v * x_n^2 scaled again correctly

        let correction = (3 * SCALE_FACTOR as i128 - vx_sq) >> 1; // Divide by 2 safely

        x = ((x_i * correction) >> 16) as FixedPoint;
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integer::fixed_point::{decode_fixed, encode_fixed};
    use crate::utils::rmsnorm;

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
        let tolerance = 1e-4;
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
