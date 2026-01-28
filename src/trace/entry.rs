use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEntry {
    pub pos: i32,
    pub token: i32,
    pub text: String,
    pub logits_hash: [u8; 32],
}

impl TraceEntry {
    pub fn new(pos: i32, token: i32, text: String, logits: &[f32]) -> Self {
        let logits_hash = hash_logits(logits);
        Self {
            pos,
            token,
            text,
            logits_hash,
        }
    }

    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.pos.to_le_bytes());
        hasher.update(self.token.to_le_bytes());
        hasher.update(&self.logits_hash);
        hasher.finalize().into()
    }
}

pub fn hash_logits(logits: &[f32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for &value in logits {
        hasher.update(value.to_le_bytes());
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_logits_deterministic() {
        let logits = vec![1.0, 2.0, 3.0];
        let hash1 = hash_logits(&logits);
        let hash2 = hash_logits(&logits);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_trace_entry_hash_deterministic() {
        let logits = vec![1.0, 2.0, 3.0];
        let entry1 = TraceEntry::new(0, 42, "test".to_string(), &logits);
        let entry2 = TraceEntry::new(0, 42, "test".to_string(), &logits);
        assert_eq!(entry1.hash(), entry2.hash());
    }

    #[test]
    fn test_trace_entry_text_not_in_hash() {
        let logits = vec![1.0, 2.0, 3.0];
        let entry1 = TraceEntry::new(0, 42, "hello".to_string(), &logits);
        let entry2 = TraceEntry::new(0, 42, "world".to_string(), &logits);
        // Hash should be same because text is not part of hash
        assert_eq!(entry1.hash(), entry2.hash());
        // But text should be different
        assert_ne!(entry1.text, entry2.text);
    }

    #[test]
    fn test_different_logits_different_hash() {
        let logits1 = vec![1.0, 2.0, 3.0];
        let logits2 = vec![1.0, 2.0, 3.1];
        let hash1 = hash_logits(&logits1);
        let hash2 = hash_logits(&logits2);
        assert_ne!(hash1, hash2);
    }
}
