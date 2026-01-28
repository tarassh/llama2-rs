mod entry;
mod merkle;
mod serialization;

pub use entry::{hash_logits, TraceEntry};
pub use merkle::{MerkleProof, MerkleTree};
pub use serialization::{load_trace, save_trace};

use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMetadata {
    pub model_path: String,
    pub prompt: String,
    pub steps: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    metadata: TraceMetadata,
    entries: Vec<TraceEntry>,
    merkle_tree: MerkleTree,
}

impl ExecutionTrace {
    pub fn new(metadata: TraceMetadata) -> Self {
        Self {
            metadata,
            entries: Vec::new(),
            merkle_tree: MerkleTree::new(),
        }
    }

    pub fn record_step(&mut self, pos: i32, token: i32, logits: &[f32]) {
        let entry = TraceEntry::new(pos, token, logits);
        self.entries.push(entry);
    }

    pub fn finalize(&mut self) {
        for entry in &self.entries {
            self.merkle_tree.add_leaf(entry.hash());
        }
        self.merkle_tree.build();
    }

    pub fn root(&self) -> Option<[u8; 32]> {
        self.merkle_tree.root()
    }

    pub fn generate_proof(&self, iteration: usize) -> Option<MerkleProof> {
        self.merkle_tree.generate_proof(iteration)
    }

    pub fn get_entry(&self, iteration: usize) -> Option<&TraceEntry> {
        self.entries.get(iteration)
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn metadata(&self) -> &TraceMetadata {
        &self.metadata
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        save_trace(self, path)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        load_trace(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_trace_basic() {
        let metadata = TraceMetadata {
            model_path: "test.bin".to_string(),
            prompt: "test".to_string(),
            steps: 5,
        };

        let mut trace = ExecutionTrace::new(metadata);
        let logits = vec![1.0, 2.0, 3.0];

        for i in 0..5 {
            trace.record_step(i, i * 2, &logits);
        }

        trace.finalize();

        assert_eq!(trace.entry_count(), 5);
        assert!(trace.root().is_some());

        // Test proof generation
        for i in 0..5 {
            let proof = trace.generate_proof(i).unwrap();
            assert!(proof.verify());
        }
    }

    #[test]
    fn test_trace_retrieval() {
        let metadata = TraceMetadata {
            model_path: "test.bin".to_string(),
            prompt: "test".to_string(),
            steps: 3,
        };

        let mut trace = ExecutionTrace::new(metadata);
        let logits = vec![1.0, 2.0, 3.0];

        for i in 0..3 {
            trace.record_step(i, i * 10, &logits);
        }

        trace.finalize();

        let entry = trace.get_entry(1).unwrap();
        assert_eq!(entry.pos, 1);
        assert_eq!(entry.token, 10);
    }
}
