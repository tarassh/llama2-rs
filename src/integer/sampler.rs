use super::utils;
use super::utils::{FixedPoint, FixedPointExt};

#[derive(Debug, Clone, Copy)]
pub struct ProbIndex {
    pub prob: FixedPoint,
    pub index: i32,
}

#[derive(Debug)]
pub struct Sampler {
    pub vocab_size: i32,
    pub probindex: Vec<ProbIndex>, // buffer used in top-p sampling
    pub temperature: FixedPoint,
    pub topp: FixedPoint,
    pub rng_state: u64,
}

impl Sampler {
    pub fn new(vocab_size: i32, temperature: FixedPoint, topp: FixedPoint, rng_seed: u64) -> Self {
        // Initialize probindex with vocab_size elements
        let probindex = vec![ProbIndex { prob: 0, index: 0 }; vocab_size as usize];

        Sampler {
            vocab_size,
            probindex,
            temperature,
            topp,
            rng_state: rng_seed,
        }
    }

    // Return the index that has the highest probability
    fn sample_argmax(probabilities: &[FixedPoint]) -> i32 {
        let mut max_i = 0;
        let mut max_p = probabilities[0];

        for i in 1..probabilities.len() {
            if probabilities[i] > max_p {
                max_i = i;
                max_p = probabilities[i];
            }
        }

        max_i as i32
    }

    // Sample index from probabilities using a random coin flip
    // probabilities must sum to 1, coin should be in [0, 1)
    fn sample_mult(probabilities: &[FixedPoint], coin: FixedPoint) -> i32 {
        let mut cdf = 0;
        for (i, &p) in probabilities.iter().enumerate() {
            cdf += p;
            if coin < cdf {
                return i as i32;
            }
        }
        (probabilities.len() - 1) as i32 // in case of rounding errors
    }

    // Top-p (nucleus) sampling: samples from the smallest set of tokens that exceed probability topp
    fn sample_topp(
        &mut self,
        probabilities: &[FixedPoint],
        topp: FixedPoint,
        coin: FixedPoint,
    ) -> i32 {
        let n = probabilities.len();

        // Filter and collect tokens above the cutoff threshold
        let cutoff = (FixedPoint::one() - topp) / FixedPoint::normalize(n - 1);
        let mut n0 = 0;
        for (i, &prob) in probabilities.iter().enumerate() {
            if prob >= cutoff {
                self.probindex[n0] = ProbIndex {
                    index: i as i32,
                    prob,
                };
                n0 += 1;
            }
        }

        // Sort filtered probabilities in descending order
        let probindex = &mut self.probindex[..n0];
        probindex.sort_unstable_by(|a, b| {
            b.prob
                .partial_cmp(&a.prob)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find truncation point where cumulative probability exceeds topp
        let mut cumulative_prob = 0;
        let mut last_idx = n0 - 1; // Default to all elements in case of rounding errors
        for (i, pi) in probindex.iter().enumerate() {
            cumulative_prob += pi.prob;
            if cumulative_prob > topp {
                last_idx = i;
                break;
            }
        }

        // Sample from the truncated list
        let r = coin * cumulative_prob;
        let mut cdf = 0;
        for i in 0..=last_idx {
            cdf += probindex[i].prob;
            if r < cdf {
                return probindex[i].index;
            }
        }

        probindex[last_idx].index // in case of rounding errors
    }

    // xoshiro256** random number generator
    fn random_u64(&mut self) -> u64 {
        let result = self.rng_state.rotate_left(5).wrapping_mul(5).rotate_left(7);
        self.rng_state = self
            .rng_state
            .rotate_left(23)
            .wrapping_add(0x9E3779B97F4A7C15);
        result
    }

    // Generate random fixed-point number between [0, SCALE_FACTOR) as i64
    fn random_fixed(&mut self) -> FixedPoint {
        let rand_u64 = self.random_u64() >> 34; // Keep top 30 bits
        (rand_u64 as FixedPoint * FixedPoint::one()) >> 30
    }

    pub fn sample(&mut self, logits: &[FixedPoint]) -> i32 {
        if logits.len() != self.vocab_size as usize {
            return 0; // Error case
        }

        let mut logits = logits.to_vec(); // Create a mutable copy

        if self.temperature == 0 {
            // Greedy argmax sampling: take the token with the highest probability
            Self::sample_argmax(&logits)
        } else {
            // Apply the temperature to the logits
            for logit in logits.iter_mut() {
                *logit /= self.temperature;
            }

            // Apply softmax to get the probabilities for next token
            utils::softmax_fixed(&mut logits);

            // Generate random float for sampling
            let coin = self.random_fixed();

            // Choose sampling strategy
            if self.topp <= 0 || self.topp >= FixedPoint::one() {
                // Simply sample from the predicted probability distribution
                Self::sample_mult(&logits, coin)
            } else {
                // Top-p (nucleus) sampling
                self.sample_topp(&logits, self.topp, coin)
            }
        }
    }
}

// Implement comparison for ProbIndex for sorting
impl PartialEq for ProbIndex {
    fn eq(&self, other: &Self) -> bool {
        self.prob == other.prob
    }
}

impl Eq for ProbIndex {}

impl PartialOrd for ProbIndex {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.prob.partial_cmp(&self.prob) // Note: reverse ordering for descending sort
    }
}

impl Ord for ProbIndex {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}
