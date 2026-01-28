# Verifiable Execution Trace

## Overview

This feature adds cryptographic execution tracing to llama2-rs, enabling verifiable and reproducible LLM inference. Each generation step is recorded with its cryptographic hash, and a Merkle tree provides tamper-proof verification that any iteration occurred exactly as claimed.

## Motivation

In production LLM systems, it's critical to verify that:
- Model outputs are reproducible given the same inputs and seed
- Intermediate computation steps can be audited
- Results haven't been tampered with
- Specific tokens at specific positions can be cryptographically proven

Traditional logging is insufficient because:
- Logs can be modified after the fact
- No cryptographic guarantees of authenticity
- Difficult to verify a subset without the entire log
- No efficient proof of inclusion

## How It Works

### Recording Phase

During text generation, at each iteration the system captures:
- **Position** (`pos`): Current position in the sequence
- **Token** (`token`): The token ID at this position
- **Logits Hash**: SHA-256 hash of the complete logits vector

These three values form a `TraceEntry`, which is itself hashed:

```
TraceEntry = (pos, token, logits_hash)
EntryHash = SHA256(pos || token || logits_hash)
```

### Merkle Tree Construction

After generation completes, all entry hashes become leaves of a Merkle tree:

```
                    Root Hash
                   /          \
                 /              \
               H01              H23
              /  \             /  \
            H0    H1         H2    H3
            |     |          |     |
         Entry0 Entry1    Entry2 Entry3
```

The Merkle root provides a single hash that commits to the entire execution trace.

### Verification Phase

To verify iteration N:
1. Load the trace file (contains entries + Merkle tree)
2. Replay generation from position 0 to N using the same model and seed
3. At position N, compare:
   - **Token**: Does the computed token match the trace?
   - **Logits**: Does SHA-256(logits) match the recorded hash?
4. Generate Merkle proof: Path from leaf N to root
5. Verify proof: Recompute root from leaf and siblings

If all checks pass, iteration N is cryptographically verified.

## Architecture

### Module Structure

```
src/trace/
├── mod.rs              # ExecutionTrace, TraceMetadata
├── entry.rs            # TraceEntry, hash_logits()
├── merkle.rs           # MerkleTree, MerkleProof
└── serialization.rs    # save_trace(), load_trace()
```

### Data Flow

```
┌─────────────────┐
│ Token Generation │
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Forward Pass        │
│  → Get Logits        │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Record Trace Entry  │
│  • hash_logits()     │
│  • TraceEntry::new() │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Add to Trace List   │
└────────┬─────────────┘
         │
         │ (after all iterations)
         ▼
┌──────────────────────┐
│  Finalize Trace      │
│  • Build Merkle tree │
│  • Compute root      │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Save to File        │
│  (bincode format)    │
└──────────────────────┘
```

### Verification Flow

```
┌──────────────────┐
│  Load Trace File │
└────────┬─────────┘
         │
         ▼
┌────────────────────────┐
│  Extract Target Entry  │
│  • Expected token      │
│  • Expected hash       │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Replay Generation     │
│  (pos 0 → target)      │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Compare at Target Pos │
│  • Token match?        │
│  • Hash match?         │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Generate Merkle Proof │
│  (leaf → root path)    │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Verify Proof          │
│  • Recompute root      │
│  • Match stored root?  │
└────────┬───────────────┘
         │
         ▼
    ✓ Verified
```

## Usage

### Create Execution Trace

```bash
cargo run --release -- stories42M.bin \
  -i "Once upon a time" \
  -n 100 \
  -s 42 \
  --trace execution.trace
```

**Output:**
```
Once upon a time, there was a little girl named Lily...
achieved tok/s: 168.37
Trace saved to: execution.trace
Merkle root: [f5, b6, eb, b9, 78, 5b, 94, 2b, ...]
```

### Verify Specific Iteration

Verify that iteration 50 occurred exactly as recorded:

```bash
cargo run --release -- stories42M.bin \
  -i "Once upon a time" \
  -s 42 \
  --verify execution.trace \
  --verify-iteration 50
```

**Output:**
```
Loaded trace from: execution.trace
Total entries: 100

Verifying iteration 50:
Expected position: 50
Expected token: 1234
Expected logits hash: [5f, 9a, 3c, ...]

Actual position: 50
Actual token: 1234
Actual logits hash: [5f, 9a, 3c, ...]

Merkle proof verification: VALID
Merkle root: [f5, b6, eb, b9, 78, 5b, 94, 2b, ...]

Verification SUCCESSFUL!
```

### Verification Failures

**Token Mismatch:**
```
Token mismatch at position 50
Expected: 1234
Got: 5678
```

**Logits Hash Mismatch:**
```
Logits hash mismatch at position 50
Expected: [5f, 9a, 3c, ...]
Got: [a2, 4d, 7b, ...]
```

**Invalid Merkle Proof:**
```
Merkle proof verification: INVALID
```

## Merkle Tree Deep Dive

### Why Merkle Trees?

1. **Compact Proof**: Verify any iteration with O(log N) hashes
2. **Tamper Detection**: Any modification changes the root hash
3. **Selective Disclosure**: Prove one iteration without revealing others
4. **Efficient**: Tree construction is O(N), verification is O(log N)

### Example: 4-Iteration Trace

```
Iterations:  [0]    [1]    [2]    [3]
              |      |      |      |
Hashes:      H0     H1     H2     H3
              \    /        \    /
               \  /          \  /
Level 1:       H01           H23
                 \           /
                  \         /
                   \       /
                    \     /
Level 2:            Root
```

**To verify iteration 1:**
- Need: H1 (entry hash), H0 (sibling), H23 (uncle)
- Compute: H01 = SHA256(H0 || H1)
- Compute: Root = SHA256(H01 || H23)
- Compare: Does computed Root == stored Root?

### Proof Size

For N iterations:
- Tree depth: ⌈log₂(N)⌉
- Proof size: 32 bytes × ⌈log₂(N)⌉

Examples:
- 100 iterations: ~7 hashes = 224 bytes
- 1,000 iterations: ~10 hashes = 320 bytes
- 10,000 iterations: ~14 hashes = 448 bytes

## File Format

Traces are saved in compact binary format using `bincode`:

```rust
ExecutionTrace {
    metadata: TraceMetadata {
        model_path: String,
        prompt: String,
        steps: usize,
    },
    entries: Vec<TraceEntry>,  // All recorded entries
    merkle_tree: MerkleTree,   // Complete tree structure
}
```

### File Size

Approximate size per iteration:
- TraceEntry: ~40 bytes (4 + 4 + 32)
- Merkle leaf: 32 bytes
- Total: ~72 bytes per iteration

Examples:
- 100 steps: ~7 KB
- 1,000 steps: ~70 KB
- 10,000 steps: ~700 KB

## Performance Impact

Benchmarks on M-series Mac:

| Metric | Without Trace | With Trace | Overhead |
|--------|--------------|------------|----------|
| tok/s | 169.8 | 168.4 | <1% |
| Memory per step | N/A | ~72 bytes | Minimal |
| Finalization (100 steps) | N/A | <1ms | Negligible |

The overhead is dominated by SHA-256 hashing of logits (~32K floats):
- Hash computation: ~0.2ms per iteration
- Negligible compared to forward pass (~6ms)

## Security Properties

### Guarantees

1. **Integrity**: Any modification to an entry invalidates the Merkle root
2. **Non-repudiation**: Root hash commits to exact execution sequence
3. **Efficient Verification**: Verify any subset without full trace
4. **Collision Resistance**: SHA-256 provides 128-bit security

### Threat Model

**Protected Against:**
- ✓ Trace tampering (changing tokens/positions)
- ✓ Logits manipulation (hash would differ)
- ✓ Insertion/deletion of entries (changes tree structure)
- ✓ Replay attacks (entries include position)

**Not Protected Against:**
- ✗ Different model weights (verification uses local model)
- ✗ Different random seed (must use same seed)
- ✗ Non-deterministic operations (requires deterministic mode)

## Advanced Use Cases

### Audit Trail

```bash
# Generate with trace
cargo run --release -- model.bin -i "prompt" -s 123 --trace audit.trace

# Later, audit any iteration
for i in {0..99}; do
  cargo run --release -- model.bin -i "prompt" -s 123 \
    --verify audit.trace --verify-iteration $i
done
```

### Distributed Verification

1. Node A generates output + trace
2. Node A shares: (output, trace file, merkle root)
3. Node B verifies: Any iteration independently
4. Node B only needs: Model, trace file, original prompt/seed

### Selective Proof

Prove iteration 50 without revealing others:
```rust
let proof = trace.generate_proof(50)?;
// Share: proof (224 bytes) + root hash (32 bytes)
// Verifier can confirm iteration 50 without full trace
```

## Implementation Details

### Hashing

```rust
// Hash logits vector
fn hash_logits(logits: &[f32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for &value in logits {
        hasher.update(value.to_le_bytes());
    }
    hasher.finalize().into()
}

// Hash trace entry
impl TraceEntry {
    fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.pos.to_le_bytes());
        hasher.update(self.token.to_le_bytes());
        hasher.update(&self.logits_hash);
        hasher.finalize().into()
    }
}
```

### Merkle Tree Construction

```rust
fn build(&mut self) {
    let mut level = self.leaves.clone();

    while level.len() > 1 {
        let mut next_level = Vec::new();

        for i in (0..level.len()).step_by(2) {
            if i + 1 < level.len() {
                // Hash pair
                let combined = hash_pair(&level[i], &level[i + 1]);
                next_level.push(combined);
            } else {
                // Odd node: promote to next level
                next_level.push(level[i]);
            }
        }

        level = next_level;
    }

    self.root = Some(level[0]);
}
```

## Testing

Run the test suite:

```bash
# Unit tests
cargo test --lib

# Integration tests
cargo test --test '*'

# Test specific module
cargo test trace::
```

### Manual Verification

```bash
# Create trace
cargo run --release -- stories42M.bin -i "test" -n 50 -s 42 --trace test.trace

# Verify first iteration
cargo run --release -- stories42M.bin -i "test" -s 42 --verify test.trace --verify-iteration 0

# Verify middle iteration
cargo run --release -- stories42M.bin -i "test" -s 42 --verify test.trace --verify-iteration 25

# Verify last iteration
cargo run --release -- stories42M.bin -i "test" -s 42 --verify test.trace --verify-iteration 49

# Test determinism
cargo run --release -- stories42M.bin -i "test" -n 50 -s 42 --trace test2.trace
cmp test.trace test2.trace  # Should be identical
```

## Limitations

1. **Determinism Required**: Model must be deterministic for verification
2. **Floating Point**: Minor FP differences across platforms may affect hashes
3. **Model Dependency**: Verification requires same model file
4. **Memory**: Full trace kept in memory during generation
5. **Seed Required**: Must use same random seed for verification

## Future Enhancements

- [ ] Stream trace to disk during generation (reduce memory)
- [ ] Compressed trace format (zstd/gzip)
- [ ] Incremental verification (verify subset efficiently)
- [ ] Remote verification server
- [ ] JSON export for human readability
- [ ] Proof-of-work integration for anti-spam
- [ ] Multi-signature support for distributed trust

## References

- [Merkle Trees](https://en.wikipedia.org/wiki/Merkle_tree)
- [SHA-256 Specification](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf)
- [Certificate Transparency](https://certificate.transparency.dev/) - Similar Merkle tree usage
- [Bitcoin Merkle Trees](https://en.bitcoin.it/wiki/Protocol_documentation#Merkle_Trees) - Production implementation

## License

Same as llama2-rs main project.
