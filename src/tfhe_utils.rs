use std::ops::Add;
use tfhe::prelude::*;
use tfhe::{ClientKey, FheInt32};

#[derive(Clone)]
pub struct EncryptedKeyLUT {
    keys: Vec<FheInt32>, // Encrypted keys
    values: Vec<f32>,    // Plaintext values
}

impl EncryptedKeyLUT {
    pub fn new(values: Vec<f32>, client_key: &ClientKey) -> Self {
        let keys: Vec<FheInt32> = (0..values.len())
            .map(|i| FheInt32::try_encrypt(i as i32, client_key).unwrap())
            .collect();
        Self { keys, values }
    }

    pub fn lookup(&self, encrypted_key: &FheInt32) -> f32 {
        self.keys
            .iter()
            .zip(self.values.iter())
            .map(|(lut_key, lut_value)| {
                let condition = encrypted_key.eq(lut_key);
                let condition_as_int =
                    condition.cmux(&FheInt32::encrypt_trivial(1), &FheInt32::encrypt_trivial(0));
                condition_as_int.try_decrypt_trivial::<i32>().unwrap() as f32 * lut_value
            }) // Select matching value
            .reduce(|acc, val| acc + val) // Sum ensures only the selected value is nonzero
            .unwrap()
    }

    pub fn range_lookup(&self, encrypted_n: &FheInt32, encrypted_m: &FheInt32) -> Vec<f32> {
        let mut start = encrypted_n.clone();
        let end = encrypted_m.clone();
        let mut results = vec![];
        loop {
            let is_end = start
                .eq(end.clone())
                .cmux(&FheInt32::encrypt_trivial(1), &FheInt32::encrypt_trivial(0));
            let is_end_decrypted = is_end.try_decrypt_trivial::<i32>().unwrap();
            if is_end_decrypted == 1 {
                break;
            }

            let value = self.lookup(&start);
            results.push(value);
            start = start.add(FheInt32::encrypt_trivial(1));
        }

        results
    }
}

impl std::fmt::Debug for EncryptedKeyLUT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptedKeyLUT")
            .field("keys", &format!("<{} encrypted keys>", self.keys.len()))
            .field("values", &format!("<{} values>", self.values.len()))
            .finish()
    }
}
