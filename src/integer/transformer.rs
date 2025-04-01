use super::utils;
use super::utils::FixedPoint;
use memmap2::Mmap;
use std::fs::File;
use std::io::Read;
use std::{mem, ptr};

// Configuration for the transformer architecture
#[derive(Debug, Clone)]
pub struct Config {
    pub dim: i32,        // transformer dimension
    pub hidden_dim: i32, // for ffn layers
    pub n_layers: i32,   // number of layers
    pub n_heads: i32,    // number of query heads
    pub n_kv_heads: i32, // number of key/value heads (can be < query heads because of multiquery)
    pub vocab_size: i32, // vocabulary size, usually 256 (byte-level)
    pub seq_len: i32,    // max sequence length
}

// Weights for the transformer model
#[derive(Debug)]
pub struct TransformerWeights {
    // token embedding table
    pub token_embedding_table: Vec<FixedPoint>, // (vocab_size, dim)
    // weights for rmsnorms
    pub rms_att_weight: Vec<FixedPoint>, // (layer, dim) rmsnorm weights
    pub rms_ffn_weight: Vec<FixedPoint>, // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    pub wq: Vec<FixedPoint>, // (layer, dim, n_heads * head_size)
    pub wk: Vec<FixedPoint>, // (layer, dim, n_kv_heads * head_size)
    pub wv: Vec<FixedPoint>, // (layer, dim, n_kv_heads * head_size)
    pub wo: Vec<FixedPoint>, // (layer, n_heads * head_size, dim)
    // weights for ffn
    pub w1: Vec<FixedPoint>, // (layer, hidden_dim, dim)
    pub w2: Vec<FixedPoint>, // (layer, dim, hidden_dim)
    pub w3: Vec<FixedPoint>, // (layer, hidden_dim, dim)
    // final rmsnorm
    pub rms_final_weight: Vec<FixedPoint>, // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    pub wcls: Option<Vec<FixedPoint>>,
}

#[derive(Debug)]
pub struct RunState {
    // current wave of activations
    pub x: Vec<FixedPoint>,      // activation at current time stamp (dim,)
    pub xb: Vec<FixedPoint>,     // same, but inside a residual branch (dim,)
    pub xb2: Vec<FixedPoint>,    // an additional buffer just for convenience (dim,)
    pub hb: Vec<FixedPoint>,     // buffer for hidden dimension in the ffn (hidden_dim,)
    pub hb2: Vec<FixedPoint>,    // buffer for hidden dimension in the ffn (hidden_dim,)
    pub q: Vec<FixedPoint>,      // query (dim,)
    pub att: Vec<FixedPoint>,    // buffer for scores/attention values (n_heads, seq_len)
    pub logits: Vec<FixedPoint>, // output logits
    // kv cache
    pub key_cache: Vec<FixedPoint>,   // (layer, seq_len, dim)
    pub value_cache: Vec<FixedPoint>, // (layer, seq_len, dim)
}

// Main transformer struct
#[derive(Debug)]
pub struct Transformer {
    pub config: Config, // the hyperparameters of the architecture (the blueprint)
    pub weights: TransformerWeights, // the weights of the model
    pub state: RunState, // buffers for the "wave" of activations in the forward pass
    // Memory mapping related fields
    pub file: Option<std::fs::File>, // file handle for memory mapping
    pub mmap: Option<memmap2::Mmap>, // memory mapped data

    pub rope_freqs: Vec<Vec<(FixedPoint, FixedPoint)>>, // [seq_len][dim/2]
}


impl Transformer {
    pub fn read_checkpoint(checkpoint_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Open the file
        let mut file = File::open(checkpoint_path)?;

        // Read the config header
        let mut config_bytes = vec![0u8; mem::size_of::<Config>()];
        file.read_exact(&mut config_bytes)?;

        // Convert bytes to Config struct
        let mut config: Config =
            unsafe { ptr::read_unaligned(config_bytes.as_ptr() as *const Config) };

        // Handle shared weights flag (negative vocab_size signals unshared weights)
        let shared_weights = config.vocab_size > 0;
        config.vocab_size = config.vocab_size.abs();

        // Create memory map
        let mmap = unsafe { Mmap::map(&file)? };

        // Calculate the offset to weights data
        let weights_offset = mem::size_of::<Config>();

        // Create weights from the mapped memory
        let weights = Self::memory_map_weights(&config, &mmap[weights_offset..], shared_weights)?;

        // Create run state buffers
        let state = RunState::new(&config);

        let rope_freqs = Self::precompute_rope_freqs(&config);

        Ok(Transformer {
            config: config.clone(),
            weights,
            state,
            file: Some(file),
            mmap: Some(mmap),
            rope_freqs,
        })
    }

    fn precompute_rope_freqs(config: &Config) -> Vec<Vec<(FixedPoint, FixedPoint)>> {
        let seq_len = config.seq_len as usize;
        let dim = config.dim as usize;
        let head_size = (config.dim / config.n_heads) as f32;

        let freqs: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| {
                let head_dim = (i % head_size as usize) as f32;
                1.0 / 10000f32.powf(head_dim / head_size)
            })
            .collect();

        let mut rope_freqs = vec![vec![(0, 0); freqs.len()]; seq_len];

        for pos in 0..seq_len {
            for (j, &freq) in freqs.iter().enumerate() {
                let val = pos as f32 * freq;
                rope_freqs[pos][j] = (utils::encode_fixed(val.cos()), utils::encode_fixed(val.sin()));
            }
        }

        rope_freqs
    }

    fn memory_map_weights(
        config: &Config,
        weight_data: &[u8],
        shared_weights: bool,
    ) -> Result<TransformerWeights, Box<dyn std::error::Error>> {
        // Convert the byte slice to f32 slice
        let weight_data = unsafe {
            std::slice::from_raw_parts(
                weight_data.as_ptr() as *const f32,
                weight_data.len() / mem::size_of::<f32>(),
            )
        };

        // Calculate sizes for each weight tensor
        let dim = config.dim as usize;
        let hidden_dim = config.hidden_dim as usize;
        let n_layers = config.n_layers as usize;
        let vocab_size = config.vocab_size as usize;
        let head_size = dim / config.n_heads as usize;
        let n_heads = config.n_heads as usize;
        let n_kv_heads = config.n_kv_heads as usize;
        let seq_len = config.seq_len as usize;

        // Helper to get or skip weights
        let mut offset = 0;
        let mut get_weights = |size: usize, skip: bool| {
            let slice = &weight_data[offset..offset + size];
            offset += size;
            if skip {
                vec![] // Return empty vec when skipping
            } else {
                slice.to_vec()
            }
        };

        // Get token embeddings
        let token_embedding_table = get_weights(vocab_size * dim, false);

        // Get attention weights
        let rms_att_weight = get_weights(n_layers * dim, false);

        // Get query, key, value projection weights
        let wq = get_weights(n_layers * dim * (n_heads * head_size), false);
        let wk = get_weights(n_layers * dim * (n_kv_heads * head_size), false);
        let wv = get_weights(n_layers * dim * (n_kv_heads * head_size), false);
        let wo = get_weights(n_layers * (n_heads * head_size) * dim, false);

        // Get FFN weights
        let rms_ffn_weight = get_weights(n_layers * dim, false);
        let w1 = get_weights(n_layers * dim * hidden_dim, false);
        let w2 = get_weights(n_layers * hidden_dim * dim, false);
        let w3 = get_weights(n_layers * dim * hidden_dim, false);

        // Get final normalization weights
        let rms_final_weight = get_weights(dim, false);

        // Skip RoPE frequency tables
        get_weights(seq_len * head_size / 2, true); // skip freq_cis_real
        get_weights(seq_len * head_size / 2, true); // skip freq_cis_imag

        // Create the weights structure
        let weights = TransformerWeights {
            token_embedding_table: token_embedding_table.iter().map(|&x| utils::encode_fixed(x)).collect(),
            rms_att_weight: rms_att_weight.iter().map(|&x| utils::encode_fixed(x)).collect(),
            rms_ffn_weight: rms_ffn_weight.iter().map(|&x| utils::encode_fixed(x)).collect(),
            wq: wq.iter().map(|&x| utils::encode_fixed(x)).collect(),
            wk: wk.iter().map(|&x| utils::encode_fixed(x)).collect(),
            wv: wv.iter().map(|&x| utils::encode_fixed(x)).collect(),
            wo: wo.iter().map(|&x| utils::encode_fixed(x)).collect(),
            w1: w1.iter().map(|&x| utils::encode_fixed(x)).collect(),
            w2: w2.iter().map(|&x| utils::encode_fixed(x)).collect(),
            w3: w3.iter().map(|&x| utils::encode_fixed(x)).collect(),
            rms_final_weight: rms_final_weight.iter().map(|&x| utils::encode_fixed(x)).collect(),
            wcls: if shared_weights {
                None
            } else {
                Some(get_weights(vocab_size * dim, false).iter().map(|&x| utils::encode_fixed(x)).collect())
            },
        };

        Ok(weights)
    }
}


// Helper implementation for RunState
impl RunState {
    fn new(config: &Config) -> Self {
        let dim = config.dim as usize;
        let hidden_dim = config.hidden_dim as usize;
        let n_heads = config.n_heads as usize;
        let seq_len = config.seq_len as usize;
        let n_layers = config.n_layers as usize;
        let vocab_size = config.vocab_size as usize;
        let kv_dim = (config.dim * config.n_kv_heads / config.n_heads) as usize;

        RunState {
            x: vec![0; dim],
            xb: vec![0; dim],
            xb2: vec![0; dim],
            hb: vec![0; hidden_dim],
            hb2: vec![0; hidden_dim],
            q: vec![0; dim],
            att: vec![0; n_heads * seq_len],
            logits: vec![0; vocab_size],
            key_cache: vec![0; n_layers * seq_len * kv_dim],
            value_cache: vec![0; n_layers * seq_len * kv_dim],
        }
    }
}