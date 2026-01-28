# llama2-rs

Rust implementation of Llama 2 inference with verifiable execution tracing.

## Features

- **Fast Inference**: Optimized Rust implementation for LLaMA 2 models
- **Parallel Execution**: Optional parallel processing with Rayon
- **Memory-Mapped Models**: Efficient model loading with memmap2
- **🔐 Verifiable Execution Trace**: Cryptographic proof of inference correctness
  - SHA-256 hashing of each generation step
  - Merkle tree construction for tamper-proof verification
  - Efficient proof generation and verification
  - See [EXECUTION_TRACE.md](EXECUTION_TRACE.md) for details

## Installation

```bash
cargo build --release
```

## Usage

### Basic Text Generation

```bash
cargo run --release -- stories42M.bin -i "Once upon a time" -n 256
```

### Parameters

- `checkpoint`: Path to model checkpoint file (required)
- `-i, --input_prompt`: Input prompt text
- `-n, --steps`: Number of tokens to generate (default: 256)
- `-t, --temperature`: Sampling temperature (default: 1.0)
- `-p, --p_value`: Top-p nucleus sampling (default: 0.9)
- `-s, --seed`: Random seed for reproducibility
- `-z, --tokenizer`: Path to tokenizer file (default: tokenizer.bin)

### Verifiable Execution Tracing

**Create a trace:**
```bash
cargo run --release -- stories42M.bin \
  -i "Once upon a time" \
  -n 100 \
  -s 42 \
  --trace execution.trace
```

**Verify an iteration:**
```bash
cargo run --release -- stories42M.bin \
  -i "Once upon a time" \
  -s 42 \
  --verify execution.trace \
  --verify-iteration 50
```

See [EXECUTION_TRACE.md](EXECUTION_TRACE.md) for comprehensive documentation.

## Features Flags

- `parallel`: Enable parallel processing (default: enabled)

Build without parallel processing:
```bash
cargo build --release --no-default-features
```

## Project Structure

```
llama2-rs/
├── src/
│   ├── main.rs           # CLI interface
│   ├── lib.rs            # Core generation logic
│   ├── model.rs          # Transformer model
│   ├── tokenizer.rs      # Text tokenization
│   ├── sampler.rs        # Token sampling strategies
│   ├── utils.rs          # Utility functions
│   └── trace/            # Execution trace module
│       ├── mod.rs        # Trace orchestration
│       ├── entry.rs      # Trace entries and hashing
│       ├── merkle.rs     # Merkle tree implementation
│       └── serialization.rs  # Binary serialization
├── Cargo.toml
├── README.md             # This file
└── EXECUTION_TRACE.md    # Detailed trace documentation
```

## Dependencies

- `clap`: Command-line argument parsing
- `tfhe`: Homomorphic encryption support
- `memmap2`: Memory-mapped file I/O
- `rayon`: Data parallelism
- `sha2`: SHA-256 cryptographic hashing
- `serde`: Serialization framework
- `bincode`: Binary encoding

## Performance

On Apple Silicon (M-series):
- **Without trace**: ~170 tok/s
- **With trace**: ~168 tok/s (<1% overhead)

Trace file sizes:
- 100 steps: ~7 KB
- 1,000 steps: ~70 KB

## Examples

### Reproducible Generation

```bash
# Generate with seed
cargo run --release -- stories42M.bin -i "Hello" -n 50 -s 123

# Regenerate with same seed (identical output)
cargo run --release -- stories42M.bin -i "Hello" -n 50 -s 123
```

### Auditable Generation

```bash
# Generate with trace
cargo run --release -- stories42M.bin \
  -i "The quick brown fox" \
  -n 100 \
  -s 42 \
  --trace audit.trace

# Verify any iteration later
cargo run --release -- stories42M.bin \
  -i "The quick brown fox" \
  -s 42 \
  --verify audit.trace \
  --verify-iteration 75
```

## Testing

```bash
# Run all tests
cargo test

# Run library tests
cargo test --lib

# Run specific module tests
cargo test trace::
```

## Verifiable Execution Use Cases

1. **AI Safety**: Prove model outputs match specific inputs
2. **Compliance**: Audit trail for regulated industries
3. **Research**: Reproduce and verify published results
4. **Distributed Systems**: Verify remote inference nodes
5. **Model Debugging**: Pinpoint exact divergence points

## Contributing

Contributions welcome! Please ensure:
- Code compiles without warnings
- All tests pass (`cargo test`)
- New features include tests
- Documentation is updated

## License

MIT License

## Acknowledgments

Based on Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c)

## Citation

If you use the verifiable execution trace feature in research, please cite:

```bibtex
@software{llama2_rs_trace,
  title = {llama2-rs: Verifiable Execution Tracing for LLM Inference},
  author = {tarassh},
  year = {2026},
  url = {https://github.com/tarassh/llama2-rs}
}
```
