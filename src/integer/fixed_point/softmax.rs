use super::exp::exp_fixed;
use crate::integer::fixed_point::FixedPoint;
use crate::integer::fixed_point::SCALE_FACTOR;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integer::fixed_point::{decode_fixed, encode_fixed};
    use crate::utils::softmax;

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
        let tolerance = 1e-4;
        for (expected, actual) in expected_f32.iter().zip(output_f32.iter()) {
            assert!(
                (expected - actual).abs() < tolerance,
                "Mismatch: expected={}, got={}",
                expected,
                actual
            );
        }
    }
}
