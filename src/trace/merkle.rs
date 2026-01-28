use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleTree {
    leaves: Vec<[u8; 32]>,
    root: Option<[u8; 32]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    pub leaf_index: usize,
    pub leaf_hash: [u8; 32],
    pub siblings: Vec<([u8; 32], bool)>, // (hash, is_right)
    pub root: [u8; 32],
}

impl MerkleTree {
    pub fn new() -> Self {
        Self {
            leaves: Vec::new(),
            root: None,
        }
    }

    pub fn add_leaf(&mut self, hash: [u8; 32]) {
        self.leaves.push(hash);
    }

    pub fn build(&mut self) {
        if self.leaves.is_empty() {
            self.root = None;
            return;
        }

        if self.leaves.len() == 1 {
            self.root = Some(self.leaves[0]);
            return;
        }

        let mut level = self.leaves.clone();

        while level.len() > 1 {
            let mut next_level = Vec::new();

            for i in (0..level.len()).step_by(2) {
                if i + 1 < level.len() {
                    let combined = hash_pair(&level[i], &level[i + 1]);
                    next_level.push(combined);
                } else {
                    // Odd node, promote to next level
                    next_level.push(level[i]);
                }
            }

            level = next_level;
        }

        self.root = Some(level[0]);
    }

    pub fn root(&self) -> Option<[u8; 32]> {
        self.root
    }

    pub fn generate_proof(&self, leaf_index: usize) -> Option<MerkleProof> {
        if leaf_index >= self.leaves.len() || self.root.is_none() {
            return None;
        }

        let mut siblings = Vec::new();
        let mut level = self.leaves.clone();
        let mut index = leaf_index;

        while level.len() > 1 {
            let sibling_index = if index % 2 == 0 {
                index + 1
            } else {
                index - 1
            };

            if sibling_index < level.len() {
                let is_right = index % 2 == 0;
                siblings.push((level[sibling_index], is_right));
            }

            let mut next_level = Vec::new();
            for i in (0..level.len()).step_by(2) {
                if i + 1 < level.len() {
                    next_level.push(hash_pair(&level[i], &level[i + 1]));
                } else {
                    next_level.push(level[i]);
                }
            }

            level = next_level;
            index /= 2;
        }

        Some(MerkleProof {
            leaf_index,
            leaf_hash: self.leaves[leaf_index],
            siblings,
            root: self.root.unwrap(),
        })
    }

    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }
}

impl MerkleProof {
    pub fn verify(&self) -> bool {
        let mut current = self.leaf_hash;

        for (sibling, is_right) in &self.siblings {
            current = if *is_right {
                hash_pair(&current, sibling)
            } else {
                hash_pair(sibling, &current)
            };
        }

        current == self.root
    }
}

fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_leaf() {
        let mut tree = MerkleTree::new();
        let leaf = [1u8; 32];
        tree.add_leaf(leaf);
        tree.build();
        assert_eq!(tree.root(), Some(leaf));
    }

    #[test]
    fn test_two_leaves() {
        let mut tree = MerkleTree::new();
        let leaf1 = [1u8; 32];
        let leaf2 = [2u8; 32];
        tree.add_leaf(leaf1);
        tree.add_leaf(leaf2);
        tree.build();

        let root = tree.root().unwrap();
        let expected = hash_pair(&leaf1, &leaf2);
        assert_eq!(root, expected);
    }

    #[test]
    fn test_proof_verification() {
        let mut tree = MerkleTree::new();
        for i in 0..4 {
            tree.add_leaf([i; 32]);
        }
        tree.build();

        for i in 0..4 {
            let proof = tree.generate_proof(i).unwrap();
            assert!(proof.verify(), "Proof for leaf {} should verify", i);
        }
    }

    #[test]
    fn test_odd_number_of_leaves() {
        let mut tree = MerkleTree::new();
        for i in 0..5 {
            tree.add_leaf([i; 32]);
        }
        tree.build();

        for i in 0..5 {
            let proof = tree.generate_proof(i).unwrap();
            assert!(proof.verify(), "Proof for leaf {} should verify", i);
        }
    }
}
