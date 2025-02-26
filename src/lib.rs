use model::Transformer;
use tokenizer::Tokenizer;
use sampler::Sampler;

pub mod model;
pub mod tokenizer;
pub mod sampler;

pub fn generate(transformer: &Transformer, tokenizer: &mut Tokenizer, sampler: &Sampler, input_prompt: &str, steps: i32) -> Result<(), Box<dyn std::error::Error>> {
    // Placeholder
    let prompt_tokens = tokenizer.encode(input_prompt, 1, 1);
    if prompt_tokens.is_empty() {
        return Err("Input prompt is empty".into());
    }

    // Placeholder
    for i in 0..prompt_tokens.len() {
        println!("prompt_tokens[{}] = {}", i, prompt_tokens[i]);
    }

    Ok(())
}