use super::FixedPoint;
use rayon::prelude::*;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integer::fixed_point::{decode_fixed, encode_fixed};
    use crate::utils::matmul;

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
