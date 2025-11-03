//! LWE Proxy Reencryption Module
//!
//! This module implements proxy reencryption for LWE ciphertexts, allowing secure
//! transformation of ciphertexts from one secret key to another without decryption.
//!
//! # Overview
//!
//! Proxy reencryption enables a semi-trusted proxy to convert a ciphertext encrypted
//! under one key (delegator) to a ciphertext encrypted under another key (delegatee)
//! without learning the plaintext. This is useful for:
//!
//! - Secure data sharing and delegation
//! - Access control in encrypted systems
//! - Key rotation without decryption
//! - Multi-user homomorphic encryption scenarios
//!
//! # Two Modes of Operation
//!
//! ## 1. Asymmetric Mode (Recommended for delegation)
//!
//! Alice generates a reencryption key using her secret key and Bob's public key.
//! Bob never shares his secret key.
//!
//! ```rust
//! use rs_tfhe::key::SecretKey;
//! use rs_tfhe::tlwe::TLWELv0;
//! use rs_tfhe::proxy_reenc::{PublicKeyLv0, ProxyReencryptionKey, reencrypt_tlwe_lv0};
//! use rs_tfhe::params;
//!
//! // Alice and Bob generate their keys
//! let alice_key = SecretKey::new();
//! let bob_key = SecretKey::new();
//!
//! // Bob publishes his public key (safe to share)
//! let bob_public_key = PublicKeyLv0::new(&bob_key.key_lv0);
//!
//! // Alice encrypts a message
//! let message = true;
//! let alice_ct = TLWELv0::encrypt_bool(message, params::tlwe_lv0::ALPHA, &alice_key.key_lv0);
//!
//! // Alice generates reencryption key using ONLY her secret key and Bob's public key
//! let reenc_key = ProxyReencryptionKey::new_asymmetric(&alice_key.key_lv0, &bob_public_key);
//!
//! // Proxy converts (without learning plaintext)
//! let bob_ct = reencrypt_tlwe_lv0(&alice_ct, &reenc_key);
//!
//! // Bob decrypts with his secret key
//! let decrypted = bob_ct.decrypt_bool(&bob_key.key_lv0);
//! assert_eq!(decrypted, message);
//! ```
//!
//! ## 2. Symmetric Mode (For trusted scenarios)
//!
//! When both secret keys are available (e.g., single-party key rotation).
//!
//! ```rust
//! use rs_tfhe::key::SecretKey;
//! use rs_tfhe::proxy_reenc::ProxyReencryptionKey;
//!
//! let old_key = SecretKey::new();
//! let new_key = SecretKey::new();
//!
//! // Symmetric mode - requires both secret keys
//! let reenc_key = ProxyReencryptionKey::new_symmetric(&old_key.key_lv0, &new_key.key_lv0);
//! ```
//!
//! # Security
//!
//! - The proxy learns nothing about the plaintext during reencryption
//! - Reencryption keys are unidirectional (Alice→Bob ≠ Bob→Alice)
//! - In asymmetric mode, Bob's secret key is never exposed
//! - Based on the hardness of the Learning With Errors (LWE) problem

use crate::key::SecretKeyLv0;
use crate::params;
use crate::params::Torus;
use crate::tlwe::TLWELv0;
use rand::Rng;

/// LWE Public Key for asymmetric encryption
///
/// A public key consists of encryptions of zero under the secret key.
/// This allows anyone to encrypt messages without knowing the secret key.
///
/// # Security
///
/// The public key can be safely shared. It reveals no information about the
/// secret key due to the semantic security of LWE.
///
/// # Size
///
/// The public key size is proportional to the security parameter. For typical
/// parameters, this is several KB.
#[derive(Debug, Clone)]
pub struct PublicKeyLv0 {
  /// Encryptions of zero: each is a TLWELv0 encrypting 0.0
  /// These are used as a basis for public key encryption
  pub encryptions: Vec<TLWELv0>,
}

impl PublicKeyLv0 {
  /// Generate a new public key from a secret key
  ///
  /// Creates encryptions of zero that can be used for encryption without
  /// revealing the secret key.
  ///
  /// # Arguments
  ///
  /// * `secret_key` - The secret key to generate a public key for
  ///
  /// # Returns
  ///
  /// A public key that can be safely shared
  ///
  /// # Example
  ///
  /// ```rust
  /// use rs_tfhe::key::SecretKey;
  /// use rs_tfhe::proxy_reenc::PublicKeyLv0;
  ///
  /// let secret_key = SecretKey::new();
  /// let public_key = PublicKeyLv0::new(&secret_key.key_lv0);
  /// // public_key can now be shared publicly
  /// ```
  pub fn new(secret_key: &SecretKeyLv0) -> Self {
    Self::new_with_params(
      secret_key,
      params::tlwe_lv0::N * 2, // Number of encryptions (2N for security)
      params::tlwe_lv0::ALPHA,
    )
  }

  /// Generate a public key with custom parameters
  ///
  /// # Arguments
  ///
  /// * `secret_key` - The secret key
  /// * `size` - Number of zero encryptions to generate (larger = more security)
  /// * `alpha` - Noise parameter for encryptions
  ///
  /// # Returns
  ///
  /// A public key with specified parameters
  pub fn new_with_params(secret_key: &SecretKeyLv0, size: usize, alpha: f64) -> Self {
    let mut encryptions = Vec::with_capacity(size);

    // Generate encryptions of zero
    for _ in 0..size {
      encryptions.push(TLWELv0::encrypt_f64(0.0, alpha, secret_key));
    }

    PublicKeyLv0 { encryptions }
  }

  /// Encrypt a value using the public key
  ///
  /// This allows encryption without the secret key by combining
  /// the pre-computed zero encryptions.
  ///
  /// # Arguments
  ///
  /// * `plaintext` - Value to encrypt (as f64)
  /// * `alpha` - Additional noise parameter
  ///
  /// # Returns
  ///
  /// A TLWELv0 ciphertext encrypting the plaintext
  pub fn encrypt_f64(&self, plaintext: f64, alpha: f64) -> TLWELv0 {
    let mut rng = rand::thread_rng();
    let mut result = TLWELv0::new();

    // Add the plaintext to b
    let plaintext_torus = crate::utils::f64_to_torus(plaintext);
    result.p[params::tlwe_lv0::N] = plaintext_torus;

    // Randomly combine encryptions of zero
    // This maintains semantic security while encrypting the plaintext
    for enc in &self.encryptions {
      if rng.gen_bool(0.5) {
        // Add or subtract randomly
        if rng.gen_bool(0.5) {
          for i in 0..=params::tlwe_lv0::N {
            result.p[i] = result.p[i].wrapping_add(enc.p[i]);
          }
        } else {
          for i in 0..=params::tlwe_lv0::N {
            result.p[i] = result.p[i].wrapping_sub(enc.p[i]);
          }
        }
      }
    }

    // Add fresh noise
    let normal_distr = rand_distr::Normal::new(0.0, alpha).unwrap();
    let mut rng = rand::thread_rng();
    let noise = crate::utils::gaussian_f64(0.0, &normal_distr, &mut rng);
    result.p[params::tlwe_lv0::N] = result.p[params::tlwe_lv0::N].wrapping_add(noise);

    result
  }

  /// Encrypt a boolean using the public key
  ///
  /// # Arguments
  ///
  /// * `plaintext` - Boolean value to encrypt
  /// * `alpha` - Noise parameter
  ///
  /// # Returns
  ///
  /// A TLWELv0 ciphertext encrypting the boolean
  pub fn encrypt_bool(&self, plaintext: bool, alpha: f64) -> TLWELv0 {
    let p = if plaintext { 0.125 } else { -0.125 };
    self.encrypt_f64(p, alpha)
  }
}

/// Proxy reencryption key for TLWELv0 ciphertexts
///
/// This key allows converting ciphertexts encrypted under `key_from` to
/// ciphertexts encrypted under `key_to`. It uses a decomposition-based
/// approach similar to key switching in TFHE.
#[derive(Debug, Clone)]
pub struct ProxyReencryptionKey {
  /// Decomposed encryptions for key switching
  /// Structure: [BASE * T * N] where each entry encrypts
  /// `k * key_from[i] / 2^((j+1)*BASEBIT)` under key_to
  pub key_encryptions: Vec<TLWELv0>,
  /// Base for decomposition (typically 1 << BASEBIT)
  pub base: usize,
  /// Number of decomposition levels
  pub t: usize,
}

impl ProxyReencryptionKey {
  /// Generate a proxy reencryption key using asymmetric mode (RECOMMENDED)
  ///
  /// Alice generates a reencryption key using her secret key and Bob's public key.
  /// Bob never needs to share his secret key with Alice.
  ///
  /// # Arguments
  ///
  /// * `key_from` - Alice's secret key (delegator)
  /// * `public_key_to` - Bob's public key (delegatee)
  ///
  /// # Returns
  ///
  /// A proxy reencryption key from Alice to Bob
  ///
  /// # Security
  ///
  /// This is the secure way to generate a reencryption key. Bob's secret key
  /// is never exposed, only his public key is needed.
  ///
  /// # Example
  ///
  /// ```rust
  /// use rs_tfhe::key::SecretKey;
  /// use rs_tfhe::proxy_reenc::{PublicKeyLv0, ProxyReencryptionKey};
  ///
  /// let alice_key = SecretKey::new();
  /// let bob_key = SecretKey::new();
  /// let bob_public = PublicKeyLv0::new(&bob_key.key_lv0);
  ///
  /// // Alice only needs Bob's public key
  /// let reenc_key = ProxyReencryptionKey::new_asymmetric(
  ///     &alice_key.key_lv0,
  ///     &bob_public
  /// );
  /// ```
  pub fn new_asymmetric(key_from: &SecretKeyLv0, public_key_to: &PublicKeyLv0) -> Self {
    Self::new_asymmetric_with_params(
      key_from,
      public_key_to,
      params::KSK_ALPHA,
      params::trgsw_lv1::BASEBIT,
      params::trgsw_lv1::IKS_T,
    )
  }

  /// Generate a proxy reencryption key with asymmetric mode and custom parameters
  ///
  /// # Arguments
  ///
  /// * `key_from` - Delegator's secret key
  /// * `public_key_to` - Delegatee's public key
  /// * `alpha` - Noise parameter
  /// * `basebit` - Decomposition base bits
  /// * `t` - Number of decomposition levels
  ///
  /// # Returns
  ///
  /// A proxy reencryption key with custom parameters
  pub fn new_asymmetric_with_params(
    key_from: &SecretKeyLv0,
    public_key_to: &PublicKeyLv0,
    alpha: f64,
    basebit: usize,
    t: usize,
  ) -> Self {
    let base = 1 << basebit;
    let n = params::tlwe_lv0::N;
    let mut key_encryptions = vec![TLWELv0::new(); base * t * n];

    // Generate decomposed encryptions using the PUBLIC key
    for i in 0..n {
      for j in 0..t {
        for k in 0..base {
          if k == 0 {
            continue; // Skip k=0 as it contributes nothing
          }
          // Encrypt k * key_from[i] / 2^((j+1)*basebit) using Bob's PUBLIC key
          let p = ((k as u32 * key_from[i]) as f64) / ((1 << ((j + 1) * basebit)) as f64);
          let idx = (base * t * i) + (base * j) + k;
          // Use public key encryption instead of secret key
          key_encryptions[idx] = public_key_to.encrypt_f64(p, alpha);
        }
      }
    }

    ProxyReencryptionKey {
      key_encryptions,
      base,
      t,
    }
  }

  /// Generate a proxy reencryption key using symmetric mode
  ///
  /// This requires both secret keys and should only be used in trusted scenarios
  /// like single-party key rotation or when both parties trust each other.
  ///
  /// # Arguments
  ///
  /// * `key_from` - Source secret key
  /// * `key_to` - Target secret key
  ///
  /// # Returns
  ///
  /// A proxy reencryption key from source to target
  ///
  /// # Security Warning
  ///
  /// This mode requires access to both secret keys. For true delegation where
  /// Bob doesn't share his secret key, use `new_asymmetric` instead.
  ///
  /// # Example
  ///
  /// ```rust
  /// use rs_tfhe::key::SecretKey;
  /// use rs_tfhe::proxy_reenc::ProxyReencryptionKey;
  ///
  /// // Single-party key rotation scenario
  /// let old_key = SecretKey::new();
  /// let new_key = SecretKey::new();
  ///
  /// let reenc_key = ProxyReencryptionKey::new_symmetric(
  ///     &old_key.key_lv0,
  ///     &new_key.key_lv0
  /// );
  /// ```
  pub fn new_symmetric(key_from: &SecretKeyLv0, key_to: &SecretKeyLv0) -> Self {
    Self::new_symmetric_with_params(
      key_from,
      key_to,
      params::KSK_ALPHA,
      params::trgsw_lv1::BASEBIT,
      params::trgsw_lv1::IKS_T,
    )
  }

  /// Generate a proxy reencryption key in symmetric mode with custom parameters
  ///
  /// # Arguments
  ///
  /// * `key_from` - The delegator's secret key (source)
  /// * `key_to` - The delegatee's secret key (target)
  /// * `alpha` - Noise parameter for the reencryption key
  /// * `basebit` - Base bits for decomposition (e.g., 3)
  /// * `t` - Number of decomposition levels (e.g., 8)
  ///
  /// # Returns
  ///
  /// A proxy reencryption key with specified parameters
  ///
  /// # Security Warning
  ///
  /// Requires both secret keys. Use `new_asymmetric_with_params` for secure delegation.
  pub fn new_symmetric_with_params(
    key_from: &SecretKeyLv0,
    key_to: &SecretKeyLv0,
    alpha: f64,
    basebit: usize,
    t: usize,
  ) -> Self {
    let base = 1 << basebit;
    let n = params::tlwe_lv0::N;
    let mut key_encryptions = vec![TLWELv0::new(); base * t * n];

    // Generate decomposed encryptions similar to gen_key_switching_key
    for i in 0..n {
      for j in 0..t {
        for k in 0..base {
          if k == 0 {
            continue; // Skip k=0 as it contributes nothing
          }
          // Encrypt k * key_from[i] / 2^((j+1)*basebit)
          let p = ((k as u32 * key_from[i]) as f64) / ((1 << ((j + 1) * basebit)) as f64);
          let idx = (base * t * i) + (base * j) + k;
          key_encryptions[idx] = TLWELv0::encrypt_f64(p, alpha, key_to);
        }
      }
    }

    ProxyReencryptionKey {
      key_encryptions,
      base,
      t,
    }
  }
}

// Note: TLWELv1 proxy reencryption can be added in the future following the same pattern

/// Reencrypt a TLWELv0 ciphertext from one key to another
///
/// Converts a ciphertext encrypted under the source key (embedded in the
/// reencryption key) to a ciphertext encrypted under the target key.
///
/// # Arguments
///
/// * `ct_from` - Ciphertext encrypted under the source key
/// * `reenc_key` - Proxy reencryption key from source to target
///
/// # Returns
///
/// A new ciphertext encrypting the same plaintext under the target key
///
/// # Algorithm
///
/// Uses a decomposition-based approach similar to identity_key_switching:
/// 1. Start with the b value from the source ciphertext
/// 2. For each coefficient a[i] in the source:
///    - Decompose a[i] into digits in base `reenc_key.base`
///    - Subtract the corresponding pre-computed encrypted values
/// 3. Result is an encryption of the same message under the target key
///
/// # Example
///
/// ```rust
/// use rs_tfhe::key::SecretKey;
/// use rs_tfhe::tlwe::TLWELv0;
/// use rs_tfhe::proxy_reenc::{PublicKeyLv0, ProxyReencryptionKey, reencrypt_tlwe_lv0};
/// use rs_tfhe::params;
///
/// let alice_key = SecretKey::new();
/// let bob_key = SecretKey::new();
/// let bob_public_key = PublicKeyLv0::new(&bob_key.key_lv0);
///
/// let message = true;
/// let alice_ct = TLWELv0::encrypt_bool(message, params::tlwe_lv0::ALPHA, &alice_key.key_lv0);
///
/// let reenc_key = ProxyReencryptionKey::new_asymmetric(&alice_key.key_lv0, &bob_public_key);
/// let bob_ct = reencrypt_tlwe_lv0(&alice_ct, &reenc_key);
///
/// assert_eq!(bob_ct.decrypt_bool(&bob_key.key_lv0), message);
/// ```
pub fn reencrypt_tlwe_lv0(ct_from: &TLWELv0, reenc_key: &ProxyReencryptionKey) -> TLWELv0 {
  let n = params::tlwe_lv0::N;
  let basebit = if reenc_key.base.count_ones() == 1 {
    reenc_key.base.trailing_zeros() as usize
  } else {
    3 // fallback
  };
  let base = reenc_key.base;
  let t = reenc_key.t;

  let mut result = TLWELv0::new();

  // Start with the b value from the source ciphertext
  result.p[n] = ct_from.b();

  // Precision offset for rounding (similar to identity_key_switching)
  let prec_offset: Torus = 1 << (32 - (1 + basebit * t));

  // Process each coefficient of the source ciphertext
  for i in 0..n {
    // Add precision offset for rounding
    let a_bar = ct_from.p[i].wrapping_add(prec_offset);

    // Decompose into t levels
    for j in 0..t {
      // Extract the j-th digit in base `base`
      let k = (a_bar >> (32 - (j + 1) * basebit)) & ((1 << basebit) - 1);

      if k != 0 {
        // Index into the reencryption key
        let idx = (base * t * i) + (base * j) + k as usize;

        // Subtract the pre-computed encryption
        for x in 0..=n {
          result.p[x] = result.p[x].wrapping_sub(reenc_key.key_encryptions[idx].p[x]);
        }
      }
    }
  }

  result
}

// TLWELv1 reencryption functions can be added here in the future

#[cfg(test)]
mod tests {
  use super::*;
  use crate::key::SecretKey;
  use rand::Rng;

  #[test]
  fn test_public_key_encryption() {
    let secret_key = SecretKey::new();
    let public_key = PublicKeyLv0::new(&secret_key.key_lv0);

    // Test encrypting with public key and decrypting with secret key
    for &message in &[true, false] {
      let ct = public_key.encrypt_bool(message, params::tlwe_lv0::ALPHA);
      let decrypted = ct.decrypt_bool(&secret_key.key_lv0);
      assert_eq!(decrypted, message);
    }
  }

  #[test]
  fn test_public_key_multiple() {
    let secret_key = SecretKey::new();
    let public_key = PublicKeyLv0::new(&secret_key.key_lv0);

    let mut rng = rand::thread_rng();
    let mut correct = 0;
    let iterations = 100;

    for _ in 0..iterations {
      let message = rng.gen_bool(0.5);
      let ct = public_key.encrypt_bool(message, params::tlwe_lv0::ALPHA);
      if ct.decrypt_bool(&secret_key.key_lv0) == message {
        correct += 1;
      }
    }

    let accuracy = correct as f64 / iterations as f64;
    assert!(
      accuracy > 0.95,
      "Public key encryption accuracy {} is too low",
      accuracy
    );
  }

  #[test]
  fn test_proxy_reencryption_asymmetric() {
    let alice_key = SecretKey::new();
    let bob_key = SecretKey::new();

    // Bob publishes his public key
    let bob_public_key = PublicKeyLv0::new(&bob_key.key_lv0);

    // Alice generates reencryption key using Bob's PUBLIC key (not secret key!)
    let reenc_key = ProxyReencryptionKey::new_asymmetric(&alice_key.key_lv0, &bob_public_key);

    // Test both true and false
    for message in &[true, false] {
      let alice_ct = TLWELv0::encrypt_bool(*message, params::tlwe_lv0::ALPHA, &alice_key.key_lv0);

      // Verify Alice can decrypt
      assert_eq!(alice_ct.decrypt_bool(&alice_key.key_lv0), *message);

      // Reencrypt to Bob's key
      let bob_ct = reencrypt_tlwe_lv0(&alice_ct, &reenc_key);

      // Verify Bob can decrypt
      assert_eq!(bob_ct.decrypt_bool(&bob_key.key_lv0), *message);
    }
  }

  #[test]
  fn test_proxy_reencryption_symmetric() {
    let alice_key = SecretKey::new();
    let bob_key = SecretKey::new();

    // Symmetric mode - requires both secret keys
    let reenc_key = ProxyReencryptionKey::new_symmetric(&alice_key.key_lv0, &bob_key.key_lv0);

    // Test both true and false
    for message in &[true, false] {
      let alice_ct = TLWELv0::encrypt_bool(*message, params::tlwe_lv0::ALPHA, &alice_key.key_lv0);

      // Verify Alice can decrypt
      assert_eq!(alice_ct.decrypt_bool(&alice_key.key_lv0), *message);

      // Reencrypt to Bob's key
      let bob_ct = reencrypt_tlwe_lv0(&alice_ct, &reenc_key);

      // Verify Bob can decrypt
      assert_eq!(bob_ct.decrypt_bool(&bob_key.key_lv0), *message);
    }
  }

  #[test]
  fn test_proxy_reencryption_asymmetric_multiple() {
    // Test asymmetric mode with many random plaintexts
    let alice_key = SecretKey::new();
    let bob_key = SecretKey::new();
    let bob_public_key = PublicKeyLv0::new(&bob_key.key_lv0);

    let reenc_key = ProxyReencryptionKey::new_asymmetric(&alice_key.key_lv0, &bob_public_key);

    let mut rng = rand::thread_rng();
    let mut correct = 0;
    let iterations = 100;

    for _ in 0..iterations {
      let message = rng.gen_bool(0.5);
      let alice_ct = TLWELv0::encrypt_bool(message, params::tlwe_lv0::ALPHA, &alice_key.key_lv0);
      let bob_ct = reencrypt_tlwe_lv0(&alice_ct, &reenc_key);

      if bob_ct.decrypt_bool(&bob_key.key_lv0) == message {
        correct += 1;
      }
    }

    // Should have very high accuracy (allowing for some noise growth)
    let accuracy = correct as f64 / iterations as f64;
    assert!(
      accuracy > 0.90,
      "Asymmetric accuracy {} is too low",
      accuracy
    );
  }

  #[test]
  fn test_proxy_reencryption_key_generation() {
    let alice_key = SecretKey::new();
    let bob_key = SecretKey::new();

    let reenc_key = ProxyReencryptionKey::new_symmetric(&alice_key.key_lv0, &bob_key.key_lv0);

    // Verify the key has the right size
    // Should be BASE * T * N entries
    let expected_size = reenc_key.base * reenc_key.t * params::tlwe_lv0::N;
    assert_eq!(reenc_key.key_encryptions.len(), expected_size);

    // Verify structure
    assert_eq!(reenc_key.base, 1 << params::trgsw_lv1::BASEBIT);
    assert_eq!(reenc_key.t, params::trgsw_lv1::IKS_T);

    // The key should be usable for reencryption (tested in other tests)
  }

  #[test]
  fn test_proxy_reencryption_chain_asymmetric() {
    // Test Alice -> Bob -> Carol chain using asymmetric mode
    let alice_key = SecretKey::new();
    let bob_key = SecretKey::new();
    let carol_key = SecretKey::new();

    let bob_public = PublicKeyLv0::new(&bob_key.key_lv0);
    let carol_public = PublicKeyLv0::new(&carol_key.key_lv0);

    let reenc_key_ab = ProxyReencryptionKey::new_asymmetric(&alice_key.key_lv0, &bob_public);
    let reenc_key_bc = ProxyReencryptionKey::new_asymmetric(&bob_key.key_lv0, &carol_public);

    let message = true;
    let alice_ct = TLWELv0::encrypt_bool(message, params::tlwe_lv0::ALPHA, &alice_key.key_lv0);

    // Alice -> Bob
    let bob_ct = reencrypt_tlwe_lv0(&alice_ct, &reenc_key_ab);
    assert_eq!(bob_ct.decrypt_bool(&bob_key.key_lv0), message);

    // Bob -> Carol
    let carol_ct = reencrypt_tlwe_lv0(&bob_ct, &reenc_key_bc);
    assert_eq!(carol_ct.decrypt_bool(&carol_key.key_lv0), message);
  }

  #[test]
  fn test_custom_params() {
    let alice_key = SecretKey::new();
    let bob_key = SecretKey::new();

    // Use custom parameters in symmetric mode
    let custom_alpha = params::KSK_ALPHA * 0.8;
    let reenc_key = ProxyReencryptionKey::new_symmetric_with_params(
      &alice_key.key_lv0,
      &bob_key.key_lv0,
      custom_alpha,
      params::trgsw_lv1::BASEBIT,
      params::trgsw_lv1::IKS_T,
    );

    let message = true;
    let alice_ct = TLWELv0::encrypt_bool(message, params::tlwe_lv0::ALPHA, &alice_key.key_lv0);
    let bob_ct = reencrypt_tlwe_lv0(&alice_ct, &reenc_key);

    assert_eq!(bob_ct.decrypt_bool(&bob_key.key_lv0), message);
  }
}
