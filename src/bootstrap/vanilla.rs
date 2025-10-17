use crate::bootstrap::Bootstrap;
use crate::key::CloudKey;
use crate::trgsw::{blind_rotate, identity_key_switching};
use crate::trlwe::{sample_extract_index, sample_extract_index_2};
use crate::utils::Ciphertext;

/// Vanilla bootstrap implementation
///
/// This is the standard TFHE bootstrapping as described in the original papers.
/// It uses:
/// - Blind rotation with CMUX tree
/// - Sample extraction at index 0
/// - Identity key switching to convert back to level 0
///
/// This implementation prioritizes correctness and simplicity over performance.
/// Future bootstrap strategies may optimize for:
/// - GPU acceleration
/// - Batch processing
/// - Alternative blind rotation schemes
/// - Different noise management strategies
#[derive(Debug, Clone)]
pub struct VanillaBootstrap {
  _private: (),
}

impl VanillaBootstrap {
  /// Create a new vanilla bootstrap strategy
  pub fn new() -> Self {
    VanillaBootstrap { _private: () }
  }
}

impl Default for VanillaBootstrap {
  fn default() -> Self {
    Self::new()
  }
}

impl Bootstrap for VanillaBootstrap {
  fn bootstrap(&self, ctxt: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
    // Step 1: Blind rotation - homomorphically evaluate test polynomial
    // This is the most expensive operation (~50ms on modern CPUs)
    let trlwe = blind_rotate(ctxt, cloud_key);

    // Step 2: Sample extraction - convert RLWE to LWE at coefficient index 0
    // This extracts one LWE ciphertext from the RLWE ciphertext
    let tlwe_lv1 = sample_extract_index(&trlwe, 0);

    // Step 3: Key switching - convert from level 1 key back to level 0
    // This ensures the output is under the same key as the input
    identity_key_switching(&tlwe_lv1, &cloud_key.key_switching_key)
  }

  fn bootstrap_without_key_switch(&self, ctxt: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
    // Blind rotation
    let trlwe = blind_rotate(ctxt, cloud_key);

    // Sample extraction without key switching
    // Note: sample_extract_index_2 returns a TLWELv0 struct but it's in a hybrid state
    // The result is NOT directly decryptable with key_lv0 - it's meant for further
    // homomorphic operations before final key switching (see mux function for usage)
    sample_extract_index_2(&trlwe, 0)
  }

  fn name(&self) -> &str {
    "vanilla"
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::key;
  use crate::params;
  use rand::Rng;

  #[test]
  fn test_vanilla_bootstrap() {
    let mut rng = rand::thread_rng();
    let key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&key);
    let bootstrap = VanillaBootstrap::new();

    let try_num = 10;
    for _i in 0..try_num {
      let plain = rng.gen::<bool>();
      let encrypted = Ciphertext::encrypt_bool(plain, params::tlwe_lv0::ALPHA, &key.key_lv0);
      let bootstrapped = bootstrap.bootstrap(&encrypted, &cloud_key);
      let decrypted = bootstrapped.decrypt_bool(&key.key_lv0);

      assert_eq!(
        plain, decrypted,
        "Bootstrap failed: expected {}, got {}",
        plain, decrypted
      );
    }
  }

  #[test]
  fn test_vanilla_bootstrap_without_key_switch() {
    // Note: bootstrap_without_key_switch returns an intermediate ciphertext
    // that is not directly decryptable. It's meant to be used in further
    // homomorphic operations (like addition) before final key switching.
    // See the mux() function in gates.rs for a usage example.
    //
    // This test simply verifies that the function runs without errors.
    // The correctness of the operation is validated indirectly through
    // the mux() test and other tests that use this functionality.

    let mut rng = rand::thread_rng();
    let key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&key);
    let bootstrap = VanillaBootstrap::new();

    // Test that the function runs without panicking
    for _ in 0..3 {
      let plain = rng.gen::<bool>();
      let encrypted = Ciphertext::encrypt_bool(plain, params::tlwe_lv0::ALPHA, &key.key_lv0);

      // Get intermediate result - we just verify it doesn't panic
      let _intermediate = bootstrap.bootstrap_without_key_switch(&encrypted, &cloud_key);
      // Can't decrypt this directly, but we verified it doesn't crash
    }
  }

  #[test]
  fn test_bootstrap_trait() {
    let bootstrap: Box<dyn Bootstrap> = Box::new(VanillaBootstrap::new());
    assert_eq!(bootstrap.name(), "vanilla");

    let mut rng = rand::thread_rng();
    let key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&key);

    let plain = rng.gen::<bool>();
    let encrypted = Ciphertext::encrypt_bool(plain, params::tlwe_lv0::ALPHA, &key.key_lv0);
    let bootstrapped = bootstrap.bootstrap(&encrypted, &cloud_key);
    let decrypted = bootstrapped.decrypt_bool(&key.key_lv0);

    assert_eq!(plain, decrypted);
  }
}
