use std::fs::File;
use std::io::{Read, Error, ErrorKind};
use std::path::Path;

#[derive(Debug)]
pub struct TokenIndex {
    pub str_val: String,  // Changed from char* to String
    pub id: i32,
}

#[derive(Debug)]
pub struct Tokenizer {
    pub vocab: Vec<String>,          // Changed from char** to Vec<String>
    pub vocab_scores: Vec<f32>,      // Changed from float* to Vec<f32>
    pub sorted_vocab: Vec<TokenIndex>,
    pub vocab_size: i32,
    pub max_token_length: u32,
    pub byte_pieces: [u8; 512],      // Kept as fixed-size array for single-byte strings
}

impl Tokenizer {
    pub fn build_tokenizer<P: AsRef<Path>>(tokenizer_path: P, vocab_size: i32) -> Result<Self, std::io::Error> {
        // Initialize vectors with capacity
        let mut vocab = Vec::with_capacity(vocab_size as usize);
        let mut vocab_scores = Vec::with_capacity(vocab_size as usize);
        
        // Initialize byte_pieces
        let mut byte_pieces = [0u8; 512];
        for i in 0..256 {
            byte_pieces[i * 2] = i as u8;
            byte_pieces[i * 2 + 1] = 0;
        }

        // Open and read the tokenizer file
        let mut file = File::open(tokenizer_path)?;
        
        // Read max_token_length
        let mut max_token_length_bytes = [0u8; 4];
        file.read_exact(&mut max_token_length_bytes)?;
        let max_token_length = u32::from_le_bytes(max_token_length_bytes);

        // Read vocabulary and scores
        for _ in 0..vocab_size {
            // Read score
            let mut score_bytes = [0u8; 4];
            file.read_exact(&mut score_bytes)?;
            let score = f32::from_le_bytes(score_bytes);
            vocab_scores.push(score);

            // Read string length
            let mut len_bytes = [0u8; 4];
            file.read_exact(&mut len_bytes)?;
            let len = i32::from_le_bytes(len_bytes) as usize;

            // Read string data
            let mut str_bytes = vec![0u8; len];
            file.read_exact(&mut str_bytes)?;
            
            // Convert bytes to string, handling non-UTF8 data
            let token_str = match String::from_utf8(str_bytes) {
                Ok(s) => s,
                Err(e) => {
                    // Handle non-UTF8 data by using lossy conversion
                    String::from_utf8_lossy(e.as_bytes()).into_owned()
                }
            };
            
            vocab.push(token_str);
        }
        
        if vocab.len() != vocab_size as usize {
            return Err(Error::new(
                ErrorKind::Other,
                "Failed to read expected number of tokens"
            ));
        }

        Ok(Tokenizer {
            vocab,
            vocab_scores,
            sorted_vocab: Vec::new(), // initialized lazily as in C version
            vocab_size,
            max_token_length,
            byte_pieces,
        })
    }

    // Example usage:
    pub fn new_from_file<P: AsRef<Path>>(path: P, vocab_size: i32) -> Result<Self, std::io::Error> {
        Self::build_tokenizer(path, vocab_size)
    }
} 