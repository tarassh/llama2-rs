use model::Transformer;
use sampler::Sampler;
use std::time::{SystemTime, UNIX_EPOCH};
use tfhe::{prelude::*, set_server_key, FheInt32, ServerKey};
use tokenizer::Tokenizer;

use tfhe_model::TfheTransformer;
use tfhe_tokenizer::TfheTokenizer;

pub mod model;
pub mod sampler;
pub mod tokenizer;
pub mod utils;

pub mod tfhe_model;
pub mod tfhe_tokenizer;
pub mod tfhe_utils;

/// Returns the current time in milliseconds since the Unix epoch
pub fn time_in_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

pub fn generate(
    transformer: &mut Transformer,
    tokenizer: &mut Tokenizer,
    sampler: &mut Sampler,
    input_prompt: &str,
    steps: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    // Use empty string if input_prompt is empty
    let input_prompt = if input_prompt.is_empty() {
        ""
    } else {
        input_prompt
    };

    // Encode the prompt into tokens sequence
    let prompt_tokens = tokenizer.encode(input_prompt, 1, 0);
    if prompt_tokens.is_empty() {
        return Err("Expected at least 1 prompt token".into());
    }

    // Start the main loop
    let mut start_time = 0u128; // Used to time our code, only initialized after first iteration
    let mut token = prompt_tokens[0]; // Kick off with the first token in the prompt
    let mut pos = 0i32;

    while pos < steps {
        // Forward the transformer to get logits for the next token
        let logits = transformer.forward(token, pos)?;

        // Advance the state machine
        let next = if (pos as usize) < prompt_tokens.len() - 1 {
            // If we are still processing the input prompt, force the next prompt token
            prompt_tokens[pos as usize + 1]
        } else {
            // Sample the next token from the logits
            sampler.sample(&logits)
        };
        pos += 1;

        // Data-dependent terminating condition: the BOS (=1) token delimits sequences
        if next == 1 {
            break;
        }

        // Decode and print the token
        let piece = tokenizer.decode(token, next);
        print!("{}", piece);
        std::io::stdout().flush()?;
        token = next;

        // Init the timer here because the first iteration can be slower
        if start_time == 0 {
            start_time = time_in_ms();
        }
    }
    println!();

    // Report achieved tok/s (pos-1 because the timer starts after first iteration)
    if pos > 1 {
        let end_time = time_in_ms();
        eprintln!(
            "achieved tok/s: {}",
            (pos - 1) as f64 / (end_time - start_time) as f64 * 1000.0
        );
    }

    Ok(())
}

pub fn tfhe_generate(
    transformer: &mut TfheTransformer,
    tokenizer: &mut TfheTokenizer,
    sampler: &mut Sampler,
    input_prompt: &str,
    steps: i32,
    server_key: ServerKey,
) -> Result<(), Box<dyn std::error::Error>> {
    set_server_key(server_key);

    use std::io::Write;

    // Use empty string if input_prompt is empty
    let input_prompt = if input_prompt.is_empty() {
        ""
    } else {
        input_prompt
    };

    // Encode the prompt into tokens sequence
    let prompt_tokens = tokenizer.encode(input_prompt, 1, 0);
    if prompt_tokens.is_empty() {
        return Err("Expected at least 1 prompt token".into());
    }

    // Start the main loop
    let mut start_time = 0u128; // Used to time our code, only initialized after first iteration
    let mut token = prompt_tokens[0].clone(); // Kick off with the first token in the prompt
    let mut pos = 0i32;

    while pos < steps {
        // Forward the transformer to get logits for the next token
        let logits = transformer.forward(token.clone(), pos)?;

        // Advance the state machine
        let next = if (pos as usize) < prompt_tokens.len() - 1 {
            // If we are still processing the input prompt, force the next prompt token
            prompt_tokens[pos as usize + 1].clone()
        } else {
            // Sample the next token from the logits
            let next = sampler.sample(&logits);
            FheInt32::encrypt_trivial(next)
        };
        pos += 1;

        let is_end = next.eq(FheInt32::encrypt_trivial(1));
        let is_end_decrypted = is_end.try_decrypt_trivial().unwrap();

        // Data-dependent terminating condition: the BOS (=1) token delimits sequences
        if is_end_decrypted {
            break;
        }

        // Decode and print the token
        let piece = tokenizer.decode(token.clone(), next.clone());
        print!("{}", piece);
        std::io::stdout().flush()?;
        token = next;

        // Init the timer here because the first iteration can be slower
        if start_time == 0 {
            start_time = time_in_ms();
        }
    }
    println!();

    // Report achieved tok/s (pos-1 because the timer starts after first iteration)
    if pos > 1 {
        let end_time = time_in_ms();
        eprintln!(
            "achieved tok/s: {}",
            (pos - 1) as f64 / (end_time - start_time) as f64 * 1000.0
        );
    }

    Ok(())
}
