use super::FixedPoint;
use super::SCALE_FACTOR;

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
        if int_part < 0 {
            // For large negative integer exponents beyond precomputed E_POWERS, return 0 directly
            return 0 as FixedPoint;
        }

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
const C8: FixedPoint = 1; // 1/40320 ≈ 0.0000248 * 65_536
const C9: FixedPoint = 0; // 1/3628800 ≈ 0.0000028 * 65_536

// Chebyshev approximation for e^x when x ∈ [-1,1]
fn fixed_exp_chebyshev(x: FixedPoint) -> FixedPoint {
    let x2 = ((x as i64 * x as i64) >> 16) as FixedPoint; // x^2
    let x3 = ((x2 as i64 * x as i64) >> 16) as FixedPoint; // x^3
    let x4 = ((x3 as i64 * x as i64) >> 16) as FixedPoint; // x^4
    let x5 = ((x4 as i64 * x as i64) >> 16) as FixedPoint; // x^5
    let x6 = ((x5 as i64 * x as i64) >> 16) as FixedPoint; // x^6
    let x7 = ((x6 as i64 * x as i64) >> 16) as FixedPoint; // x^7
    let x8 = ((x7 as i64 * x as i64) >> 16) as FixedPoint; // x^8
    let x9 = ((x8 as i64 * x as i64) >> 16) as FixedPoint; // x^9

    let result = C0
        + ((C1 as i64 * x as i64) >> 16) as FixedPoint
        + ((C2 as i64 * x2 as i64) >> 16) as FixedPoint
        + ((C3 as i64 * x3 as i64) >> 16) as FixedPoint
        + ((C4 as i64 * x4 as i64) >> 16) as FixedPoint
        + ((C5 as i64 * x5 as i64) >> 16) as FixedPoint
        + ((C6 as i64 * x6 as i64) >> 16) as FixedPoint
        + ((C7 as i64 * x7 as i64) >> 16) as FixedPoint
        + ((C8 as i64 * x8 as i64) >> 16) as FixedPoint
        + ((C9 as i64 * x9 as i64) >> 16) as FixedPoint;

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integer::fixed_point::{decode_fixed, encode_fixed};

    #[test]
    fn test_fixed_exp() {
        let test_cases = vec![
            -25.824846, // e^(-25.824846)
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
            -20.234,    // e^(-20.234)
            -7.4335403, -8.710781, -9.919607, -10.041857, -7.064037, -8.489454, -11.895886,
            -6.072899, -11.976686, -11.888927, -8.481754, -30.182629,
            -3.14159, // e^(-3.14159)
        ];

        for &x in &test_cases {
            let x_fixed = encode_fixed(x);
            let exp_fixed = exp_fixed(x_fixed);
            let result = decode_fixed(exp_fixed);
            let expected = x.exp();

            let error = (result - expected).abs();
            let tolerance = 0.0001;

            println!(
                "Testing e^{:.3}: Expected = {:.9}, Got = {:.9}, Error = {:.9}",
                x, expected, result, error
            );

            assert!(
                result.signum() == expected.signum(),
                "Sign mismatch for e^{:.3}: Got sign {}, Expected sign {}",
                x,
                result.signum(),
                expected.signum()
            );

            assert!(
                error < tolerance,
                "Mismatch in fixed_exp for e^{:.3}: Expected {:.9}, Got {:.9}",
                x,
                expected,
                result
            );
        }
    }
}
