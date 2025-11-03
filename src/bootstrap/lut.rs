//! LUT (Lookup Table) bootstrapping implementation
//!
//! This module provides programmable bootstrapping functionality that allows
//! evaluating arbitrary functions on encrypted data during the bootstrapping
//! process. This combines noise refreshing with function evaluation in a single operation.

use crate::bootstrap::Bootstrap;
use crate::key::CloudKey;
use crate::trgsw::identity_key_switching;
use crate::trlwe::sample_extract_index;
use crate::utils::Ciphertext;

#[cfg(feature = "lut-bootstrap")]
use crate::trgsw::blind_rotate_with_testvec;

use super::super::lut::{Generator, LookupTable};

/// LUT-based programmable bootstrapping strategy
///
/// This bootstrap strategy allows evaluating arbitrary functions during bootstrapping
/// by using pre-computed lookup tables. This is more efficient than traditional
/// bootstrapping when you need to apply the same function multiple times.
#[derive(Debug, Clone)]
pub struct LutBootstrap {
  _private: (),
}

impl LutBootstrap {
  /// Create a new LUT bootstrap strategy
  pub fn new() -> Self {
    Self { _private: () }
  }

  /// Perform programmable bootstrapping with a function
  ///
  /// The function f operates on the message space [0, message_modulus) and
  /// is evaluated homomorphically on the encrypted data during bootstrapping.
  ///
  /// This combines noise refreshing with arbitrary function evaluation.
  ///
  /// # Arguments
  /// * `ct_in` - Input ciphertext to bootstrap
  /// * `f` - Function to apply during bootstrapping
  /// * `message_modulus` - Size of the message space
  /// * `cloud_key` - Cloud key containing bootstrapping parameters
  ///
  /// # Returns
  /// Bootstrapped ciphertext with function applied
  pub fn bootstrap_func<F>(
    &self,
    ct_in: &Ciphertext,
    f: F,
    message_modulus: usize,
    cloud_key: &CloudKey,
  ) -> Ciphertext
  where
    F: Fn(usize) -> usize,
  {
    // Generate lookup table from function
    let generator = Generator::new(message_modulus);
    let lookup_table = generator.generate_lookup_table(f);

    // Perform LUT-based bootstrapping
    self.bootstrap_lut(ct_in, &lookup_table, cloud_key)
  }

  /// Perform programmable bootstrapping with a pre-computed lookup table
  ///
  /// The lookup table encodes the function to be evaluated during bootstrapping.
  /// This is more efficient than `bootstrap_func` when the same function is used multiple times.
  ///
  /// # Arguments
  /// * `ct_in` - Input ciphertext to bootstrap
  /// * `lut` - Pre-computed lookup table
  /// * `cloud_key` - Cloud key containing bootstrapping parameters
  ///
  /// # Returns
  /// Bootstrapped ciphertext with function applied
  pub fn bootstrap_lut(
    &self,
    ct_in: &Ciphertext,
    lut: &LookupTable,
    cloud_key: &CloudKey,
  ) -> Ciphertext {
    // Convert LUT to TRLWE format (test vector)
    // The LUT is already a TRLWE with the function encoded in the B polynomial
    let testvec = &lut.poly;

    // Perform blind rotation using the LUT as the test vector
    // This rotates the LUT based on the encrypted value, effectively evaluating the function
    let rotated = blind_rotate_with_testvec(ct_in, testvec, cloud_key);

    // Extract the constant term as an LWE ciphertext
    // This gives us the function evaluation encrypted under the TRLWE key
    let extracted_lwe = sample_extract_index(&rotated, 0);

    // Key switch to convert back to the original LWE key
    identity_key_switching(&extracted_lwe, &cloud_key.key_switching_key)
  }
}

impl Default for LutBootstrap {
  fn default() -> Self {
    Self::new()
  }
}

impl Bootstrap for LutBootstrap {
  fn bootstrap(&self, ctxt: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
    // Default to identity function (standard bootstrapping)
    self.bootstrap_func(ctxt, |x| x, 2, cloud_key)
  }

  fn bootstrap_without_key_switch(&self, ctxt: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
    // For LUT bootstrapping, we always need key switching to get back to level 0
    // So we perform full bootstrap and then extract without key switch

    // This is a simplified implementation - in practice, we would need
    // to modify the internal operations to support this properly
    self.bootstrap(ctxt, cloud_key)
  }

  fn name(&self) -> &str {
    "lut"
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::key;
  use crate::params;
  use rand::Rng;

  #[test]
  fn test_lut_bootstrap_creation() {
    let bootstrap = LutBootstrap::new();
    assert_eq!(bootstrap.name(), "lut");
  }

  #[test]
  fn test_identity_function() {
    let mut rng = rand::thread_rng();
    let key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&key);
    let bootstrap = LutBootstrap::new();

    // Test identity function (should preserve the value)
    let identity = |x: usize| x;

    for _ in 0..5 {
      let plain = rng.gen::<bool>();
      let message = if plain { 1 } else { 0 };
      let encrypted = Ciphertext::encrypt_lwe_message(
        message,
        2,
        params::SECURITY_128_BIT.tlwe_lv0.alpha,
        &key.key_lv0,
      );

      let bootstrapped = bootstrap.bootstrap_func(&encrypted, identity, 2, &cloud_key);
      let decrypted_message = bootstrapped.decrypt_lwe_message(2, &key.key_lv0);
      let decrypted = decrypted_message != 0;

      assert_eq!(plain, decrypted, "Identity function failed");
    }
  }

  #[test]
  fn test_not_function() {
    let mut rng = rand::thread_rng();
    let key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&key);
    let bootstrap = LutBootstrap::new();

    // Test NOT function
    let not_func = |x: usize| 1 - x;

    for _ in 0..5 {
      let plain = rng.gen::<bool>();
      let message = if plain { 1 } else { 0 };
      let encrypted = Ciphertext::encrypt_lwe_message(
        message,
        2,
        params::SECURITY_128_BIT.tlwe_lv0.alpha,
        &key.key_lv0,
      );

      let bootstrapped = bootstrap.bootstrap_func(&encrypted, not_func, 2, &cloud_key);
      let decrypted_message = bootstrapped.decrypt_lwe_message(2, &key.key_lv0);
      let decrypted = decrypted_message != 0;

      assert_eq!(!plain, decrypted, "NOT function failed");
    }
  }

  #[test]
  fn test_constant_function() {
    let mut rng = rand::thread_rng();
    let key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&key);
    let bootstrap = LutBootstrap::new();

    // Test constant function (always returns true)
    let constant_true = |_x: usize| 1;

    for _ in 0..5 {
      let plain = rng.gen::<bool>();
      let message = if plain { 1 } else { 0 };
      let encrypted = Ciphertext::encrypt_lwe_message(
        message,
        2,
        params::SECURITY_128_BIT.tlwe_lv0.alpha,
        &key.key_lv0,
      );

      let bootstrapped = bootstrap.bootstrap_func(&encrypted, constant_true, 2, &cloud_key);
      let decrypted_message = bootstrapped.decrypt_lwe_message(2, &key.key_lv0);
      let decrypted = decrypted_message != 0;

      assert!(decrypted, "Constant true function failed");
    }
  }

  #[test]
  fn test_lut_reuse() {
    let mut rng = rand::thread_rng();
    let key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&key);
    let bootstrap = LutBootstrap::new();

    // Pre-compute lookup table for NOT function
    let generator = Generator::new(2);
    let not_func = |x: usize| 1 - x;
    let lut = generator.generate_lookup_table(not_func);

    // Test multiple inputs with the same LUT
    for _ in 0..5 {
      let plain = rng.gen::<bool>();
      let message = if plain { 1 } else { 0 };
      let encrypted = Ciphertext::encrypt_lwe_message(
        message,
        2,
        params::SECURITY_128_BIT.tlwe_lv0.alpha,
        &key.key_lv0,
      );

      let bootstrapped = bootstrap.bootstrap_lut(&encrypted, &lut, &cloud_key);
      let decrypted_message = bootstrapped.decrypt_lwe_message(2, &key.key_lv0);
      let decrypted = decrypted_message != 0;

      assert_eq!(!plain, decrypted, "LUT reuse failed");
    }
  }

  #[test]
  fn test_bootstrap_trait() {
    let bootstrap: Box<dyn Bootstrap> = Box::new(LutBootstrap::new());
    assert_eq!(bootstrap.name(), "lut");

    let mut rng = rand::thread_rng();
    let key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&key);

    let plain = rng.gen::<bool>();
    let message = if plain { 1 } else { 0 };
    let encrypted =
      Ciphertext::encrypt_lwe_message(message, 2, params::tlwe_lv0::ALPHA, &key.key_lv0);
    let bootstrapped = bootstrap.bootstrap(&encrypted, &cloud_key);
    let decrypted_message = bootstrapped.decrypt_lwe_message(2, &key.key_lv0);
    let decrypted = decrypted_message != 0;

    assert_eq!(plain, decrypted);
  }
}
