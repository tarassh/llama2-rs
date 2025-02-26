#[derive(Debug, Clone, Copy)]
pub struct ProbIndex {
    pub prob: f32,
    pub index: i32,
}

#[derive(Debug)]
pub struct Sampler {
    pub vocab_size: i32,
    pub probindex: Vec<ProbIndex>,  // buffer used in top-p sampling
    pub temperature: f32,
    pub topp: f32,
    pub rng_state: u64,
}

impl Sampler {
    pub fn new(vocab_size: i32, temperature: f32, topp: f32, rng_seed: u64) -> Self {
        // Initialize probindex with vocab_size elements
        let probindex = vec![
            ProbIndex {
                prob: 0.0,
                index: 0,
            };
            vocab_size as usize
        ];

        Sampler {
            vocab_size,
            probindex,
            temperature,
            topp,
            rng_state: rng_seed,
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