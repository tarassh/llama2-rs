use std::fs::File;
use std::io::Read;
use memmap2::Mmap;
use std::{ptr, mem};

// Configuration for the transformer architecture
#[derive(Debug)]
pub struct Config {
    pub dim: i32,         // transformer dimension
    pub hidden_dim: i32,  // for ffn layers
    pub n_layers: i32,    // number of layers
    pub n_heads: i32,     // number of query heads
    pub n_kv_heads: i32,  // number of key/value heads (can be < query heads because of multiquery)
    pub vocab_size: i32,  // vocabulary size, usually 256 (byte-level)
    pub seq_len: i32,     // max sequence length
}

// Weights for the transformer model
#[derive(Debug)]
pub struct TransformerWeights {
    // token embedding table
    pub token_embedding_table: Vec<f32>,    // (vocab_size, dim)
    // weights for rmsnorms
    pub rms_att_weight: Vec<f32>,  // (layer, dim) rmsnorm weights
    pub rms_ffn_weight: Vec<f32>,  // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    pub wq: Vec<f32>,  // (layer, dim, n_heads * head_size)
    pub wk: Vec<f32>,  // (layer, dim, n_kv_heads * head_size)
    pub wv: Vec<f32>,  // (layer, dim, n_kv_heads * head_size)
    pub wo: Vec<f32>,  // (layer, n_heads * head_size, dim)
    // weights for ffn
    pub w1: Vec<f32>,  // (layer, hidden_dim, dim)
    pub w2: Vec<f32>,  // (layer, dim, hidden_dim)
    pub w3: Vec<f32>,  // (layer, hidden_dim, dim)
    // final rmsnorm
    pub rms_final_weight: Vec<f32>,  // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    pub wcls: Option<Vec<f32>>,
}

// State for running the transformer
#[derive(Debug)]
pub struct RunState {
    // current wave of activations
    pub x: Vec<f32>,    // activation at current time stamp (dim,)
    pub xb: Vec<f32>,   // same, but inside a residual branch (dim,)
    pub xb2: Vec<f32>,  // an additional buffer just for convenience (dim,)
    pub hb: Vec<f32>,   // buffer for hidden dimension in the ffn (hidden_dim,)
    pub hb2: Vec<f32>,  // buffer for hidden dimension in the ffn (hidden_dim,)
    pub q: Vec<f32>,    // query (dim,)
    pub k: Vec<f32>,    // key (dim,)
    pub v: Vec<f32>,    // value (dim,)
    pub att: Vec<f32>,  // buffer for scores/attention values (n_heads, seq_len)
    pub logits: Vec<f32>, // output logits
    // kv cache
    pub key_cache: Vec<f32>,    // (layer, seq_len, dim)
    pub value_cache: Vec<f32>,  // (layer, seq_len, dim)
}

// Main transformer struct
#[derive(Debug)]
pub struct Transformer {
    pub config: Config,                  // the hyperparameters of the architecture (the blueprint)
    pub weights: TransformerWeights,     // the weights of the model
    pub state: RunState,                 // buffers for the "wave" of activations in the forward pass
    // Memory mapping related fields
    pub file: Option<std::fs::File>,     // file handle for memory mapping
    pub mmap: Option<memmap2::Mmap>,     // memory mapped data
} 

impl Transformer {

    pub fn read_checkpoint(checkpoint_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Open the file
        let mut file = File::open(checkpoint_path)?;
        
        // Read the config header
        let mut config_bytes = vec![0u8; mem::size_of::<Config>()];
        file.read_exact(&mut config_bytes)?;
        
        // Convert bytes to Config struct
        let mut config: Config = unsafe { 
            ptr::read_unaligned(config_bytes.as_ptr() as *const Config)
        };
        
        // Handle shared weights flag (negative vocab_size signals unshared weights)
        let shared_weights = config.vocab_size > 0;
        config.vocab_size = config.vocab_size.abs();

        // Create memory map
        let mmap = unsafe { Mmap::map(&file)? };
        
        // Calculate the offset to weights data
        let weights_offset = mem::size_of::<Config>();
        
        // Create weights from the mapped memory
        let weights = Self::memory_map_weights(
            &config,
            &mmap[weights_offset..],
            shared_weights
        )?;

        // Create run state buffers
        let state = RunState::new(&config);

        Ok(Transformer {
            config,
            weights,
            state,
            file: Some(file),
            mmap: Some(mmap),
        })
    }

    fn memory_map_weights(
        config: &Config,
        weight_data: &[u8],
        shared_weights: bool
    ) -> Result<TransformerWeights, Box<dyn std::error::Error>> {
        // Convert the byte slice to f32 slice
        let weight_data = unsafe {
            std::slice::from_raw_parts(
                weight_data.as_ptr() as *const f32,
                weight_data.len() / mem::size_of::<f32>()
            )
        };

        // Calculate sizes for each weight tensor
        let dim = config.dim as usize;
        let hidden_dim = config.hidden_dim as usize;
        let n_layers = config.n_layers as usize;
        let vocab_size = config.vocab_size as usize;
        
        // Helper to get a slice of the weight data
        let mut offset = 0;
        let mut get_weights = |size: usize| {
            let slice = &weight_data[offset..offset + size];
            offset += size;
            slice.to_vec()
        };

        // Create the weights structure
        let weights = TransformerWeights {
            token_embedding_table: get_weights(vocab_size * dim),
            rms_att_weight: get_weights(n_layers * dim),
            rms_ffn_weight: get_weights(n_layers * dim),
            wq: get_weights(n_layers * dim * dim),
            wk: get_weights(n_layers * dim * dim),
            wv: get_weights(n_layers * dim * dim),
            wo: get_weights(n_layers * dim * dim),
            w1: get_weights(n_layers * hidden_dim * dim),
            w2: get_weights(n_layers * dim * hidden_dim),
            w3: get_weights(n_layers * hidden_dim * dim),
            rms_final_weight: get_weights(dim),
            wcls: if shared_weights {
                None
            } else {
                Some(get_weights(vocab_size * dim))
            },
        };

        Ok(weights)
    }

    pub fn forward(&self, tokens: &[i32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let config = &self.config;
        let weights = &self.weights;
        let mut state = RunState::new(config);

        // Token embedding
        // For now, return empty logits
        Ok(vec![0.0; config.vocab_size as usize])
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

        RunState {
            x: vec![0.0; dim],
            xb: vec![0.0; dim],
            xb2: vec![0.0; dim],
            hb: vec![0.0; hidden_dim],
            hb2: vec![0.0; hidden_dim],
            q: vec![0.0; dim],
            k: vec![0.0; dim],
            v: vec![0.0; dim],
            att: vec![0.0; n_heads * seq_len],
            logits: vec![0.0; dim],
            key_cache: vec![0.0; n_layers * seq_len * dim],
            value_cache: vec![0.0; n_layers * seq_len * dim],
        }
    }
}