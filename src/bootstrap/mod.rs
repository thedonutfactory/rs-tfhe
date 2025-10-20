pub mod vanilla;

#[cfg(feature = "lut-bootstrap")]
pub mod lut;

use crate::key::CloudKey;
use crate::utils::Ciphertext;

/// Trait for bootstrapping strategies
///
/// Bootstrapping is the core operation in TFHE that refreshes a noisy ciphertext
/// by homomorphically evaluating the decryption function. Different strategies
/// can optimize for different tradeoffs (speed, accuracy, memory, hardware).
///
/// # Core Operation
/// Bootstrap takes a noisy LWE ciphertext at level 0 and returns a refreshed
/// ciphertext with reduced noise, enabling unbounded homomorphic operations.
///
/// # Typical Flow
/// 1. Blind rotation: Homomorphically evaluate a test polynomial
/// 2. Sample extraction: Convert RLWE to LWE
/// 3. Key switching: Convert back to original key
pub trait Bootstrap: Send + Sync {
  /// Perform a full bootstrap with key switching
  ///
  /// Takes a potentially noisy ciphertext and returns a refreshed one
  /// under the original encryption key.
  fn bootstrap(&self, ctxt: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext;

  /// Perform bootstrap without key switching
  ///
  /// Returns a ciphertext under the expanded RLWE key (level 1).
  /// Useful when chaining multiple operations before final key switch.
  fn bootstrap_without_key_switch(&self, ctxt: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext;

  /// Get the name of this bootstrap strategy
  fn name(&self) -> &str;
}

/// Get the default bootstrap strategy
pub fn default_bootstrap() -> Box<dyn Bootstrap> {
  Box::new(vanilla::VanillaBootstrap::new())
}
