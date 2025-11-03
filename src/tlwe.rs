use crate::key;
use crate::params;
use crate::params::HalfTorus;
use crate::params::Torus;
use crate::params::ZERO_TORUS;
use crate::utils;
use rand::Rng;
use std::iter::Iterator;
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Debug, Copy, Clone)]
pub struct TLWELv0 {
  pub p: [Torus; params::tlwe_lv0::N + 1],
}

impl Default for TLWELv0 {
  fn default() -> Self {
    Self::new()
  }
}

impl TLWELv0 {
  pub fn new() -> TLWELv0 {
    TLWELv0 {
      p: [0; params::tlwe_lv0::N + 1],
    }
  }

  pub fn b(&self) -> Torus {
    self.p[params::tlwe_lv0::N]
  }

  pub fn b_mut(&mut self) -> &mut Torus {
    &mut self.p[params::tlwe_lv0::N]
  }

  pub fn encrypt_f64(p: f64, alpha: f64, key: &key::SecretKeyLv0) -> TLWELv0 {
    let mut rng = rand::thread_rng();
    let mut tlwe = TLWELv0::new();
    let mut inner_product: Torus = 0;

    for (i, &key) in key.iter().enumerate() {
      let rand_torus: Torus = rng.gen();
      inner_product = inner_product.wrapping_add(key * rand_torus);
      tlwe.p[i] = rand_torus;
    }

    let normal_distr = rand_distr::Normal::new(0.0, alpha).unwrap();
    let mut rng = rand::thread_rng();
    let b = utils::gaussian_f64(p, &normal_distr, &mut rng);
    *tlwe.b_mut() = inner_product.wrapping_add(b);
    tlwe
  }

  pub fn encrypt_bool(p_bool: bool, alpha: f64, key: &key::SecretKeyLv0) -> TLWELv0 {
    let p = if p_bool { 0.125 } else { -0.125 };
    Self::encrypt_f64(p, alpha, key)
  }

  pub fn decrypt_bool(&self, key: &key::SecretKeyLv0) -> bool {
    let mut inner_product: Torus = 0;
    for (i, &key) in key.iter().enumerate() {
      inner_product = inner_product.wrapping_add(self.p[i] * key);
    }

    let res_torus = (self.p[params::tlwe_lv0::N].wrapping_sub(inner_product)) as HalfTorus;
    res_torus >= 0
  }

  /// Encrypt a message using LWE message encoding for programmable bootstrapping
  ///
  /// This function encodes a message as `message * scale` where scale is `1/(2*message_modulus)`.
  /// This is the standard encoding used in programmable bootstrapping.
  ///
  /// # Arguments
  /// * `message` - Integer message to encrypt (should be in [0, message_modulus))
  /// * `message_modulus` - Size of the message space
  /// * `alpha` - Noise parameter
  /// * `key` - Secret key for encryption
  ///
  /// # Returns
  /// Encrypted TLWE ciphertext
  #[cfg(feature = "lut-bootstrap")]
  pub fn encrypt_lwe_message(
    message: usize,
    message_modulus: usize,
    alpha: f64,
    key: &key::SecretKeyLv0,
  ) -> TLWELv0 {
    // Normalize message to [0, message_modulus)
    let message = message % message_modulus;

    // Encode as message * scale where scale = 1/(2*message_modulus)
    let scale = 1.0 / (2.0 * message_modulus as f64);
    let encoded_value = message as f64 * scale;

    Self::encrypt_f64(encoded_value, alpha, key)
  }

  /// Decrypt a message using LWE message decoding for programmable bootstrapping
  ///
  /// This function decodes a message from the LWE message encoding used in programmable bootstrapping.
  ///
  /// # Arguments
  /// * `message_modulus` - Size of the message space
  /// * `key` - Secret key for decryption
  ///
  /// # Returns
  /// Decrypted integer message
  #[cfg(feature = "lut-bootstrap")]
  pub fn decrypt_lwe_message(&self, message_modulus: usize, key: &key::SecretKeyLv0) -> usize {
    let mut inner_product: Torus = 0;
    for (i, &key_val) in key.iter().enumerate() {
      inner_product = inner_product.wrapping_add(self.p[i] * key_val);
    }

    let res_torus = self.p[params::tlwe_lv0::N].wrapping_sub(inner_product);
    let res_f64 = crate::utils::torus_to_f64(res_torus);

    // Decode from message * scale where scale = 1/(2*message_modulus)
    let scale = 1.0 / (2.0 * message_modulus as f64);
    let message = (res_f64 / scale + 0.5) as usize;

    // Normalize to [0, message_modulus)
    message % message_modulus
  }
}

impl Add for &TLWELv0 {
  type Output = TLWELv0;

  fn add(self, other: &TLWELv0) -> TLWELv0 {
    let mut res = TLWELv0::new();
    for ((rref, &sval), &oval) in res.p.iter_mut().zip(self.p.iter()).zip(other.p.iter()) {
      *rref = sval.wrapping_add(oval);
    }
    res
  }
}

impl Sub for &TLWELv0 {
  type Output = TLWELv0;

  fn sub(self, other: &TLWELv0) -> TLWELv0 {
    let mut res = TLWELv0::new();
    for ((rref, &sval), &oval) in res.p.iter_mut().zip(self.p.iter()).zip(other.p.iter()) {
      *rref = sval.wrapping_sub(oval);
    }
    res
  }
}

impl Neg for TLWELv0 {
  type Output = TLWELv0;

  fn neg(self) -> TLWELv0 {
    let mut res = TLWELv0::new();
    for (rref, sval) in res.p.iter_mut().zip(self.p.iter()) {
      *rref = ZERO_TORUS.wrapping_sub(*sval);
    }

    res
  }
}

impl Mul for &TLWELv0 {
  type Output = TLWELv0;

  fn mul(self, other: &TLWELv0) -> TLWELv0 {
    let mut res = TLWELv0::new();
    for ((rref, &sval), &oval) in res.p.iter_mut().zip(self.p.iter()).zip(other.p.iter()) {
      *rref = sval.wrapping_mul(oval);
    }
    res
  }
}

pub trait AddMul<Rhs = Self> {
  /// The resulting type after applying the operation.
  type Output;

  fn add_mul(self, rhs: Rhs, multiplier: Torus) -> Self::Output;
}

impl AddMul for &TLWELv0 {
  type Output = TLWELv0;

  fn add_mul(self, other: &TLWELv0, multiplier: Torus) -> TLWELv0 {
    let mut res = TLWELv0::new();
    for ((rref, &sval), &oval) in res.p.iter_mut().zip(self.p.iter()).zip(other.p.iter()) {
      *rref = sval.wrapping_add(oval.wrapping_mul(multiplier));
    }
    res
  }
}

pub trait SubMul<Rhs = Self> {
  /// The resulting type after applying the operation.
  type Output;

  fn sub_mul(self, rhs: Rhs, multiplier: Torus) -> Self::Output;
}

impl SubMul for &TLWELv0 {
  type Output = TLWELv0;

  fn sub_mul(self, other: &TLWELv0, multiplier: Torus) -> TLWELv0 {
    let mut res = TLWELv0::new();
    for ((rref, &sval), &oval) in res.p.iter_mut().zip(self.p.iter()).zip(other.p.iter()) {
      *rref = sval.wrapping_sub(oval.wrapping_mul(multiplier));
    }
    res
  }
}

#[derive(Debug, Copy, Clone)]
pub struct TLWELv1 {
  pub p: [Torus; params::tlwe_lv1::N + 1],
}

impl Default for TLWELv1 {
  fn default() -> Self {
    Self::new()
  }
}

impl TLWELv1 {
  pub fn new() -> TLWELv1 {
    TLWELv1 {
      p: [0; params::tlwe_lv1::N + 1],
    }
  }

  pub fn b_mut(&mut self) -> &mut Torus {
    &mut self.p[params::tlwe_lv1::N]
  }

  #[cfg(test)]
  pub fn encrypt_f64(p: f64, alpha: f64, key: &key::SecretKeyLv1) -> TLWELv1 {
    use crate::params::Torus;

    let mut rng = rand::thread_rng();
    let mut tlwe = TLWELv1::new();
    let mut inner_product: Torus = 0;
    for (i, &key_val) in key.iter().enumerate() {
      let rand_torus: Torus = rng.gen();
      inner_product = inner_product.wrapping_add(key_val * rand_torus);
      tlwe.p[i] = rand_torus;
    }
    let normal_distr = rand_distr::Normal::new(0.0, alpha).unwrap();
    let mut rng = rand::thread_rng();
    let b = utils::gaussian_f64(p, &normal_distr, &mut rng);
    *tlwe.b_mut() = inner_product.wrapping_add(b);
    tlwe
  }

  #[cfg(test)]
  pub fn encrypt_bool(b: bool, alpha: f64, key: &key::SecretKeyLv1) -> TLWELv1 {
    let p = if b { 0.125 } else { -0.125 };
    Self::encrypt_f64(p, alpha, key)
  }

  #[cfg(test)]
  pub fn decrypt_bool(&self, key: &key::SecretKeyLv1) -> bool {
    let mut inner_product: Torus = 0;
    for (i, &key_val) in key.iter().enumerate() {
      inner_product = inner_product.wrapping_add(self.p[i] * key_val);
    }

    let res_torus = (self.p[key.len()].wrapping_sub(inner_product)) as HalfTorus;
    res_torus >= 0
  }
}

#[cfg(test)]
mod tests {
  use crate::key;
  use crate::params;
  use crate::tlwe::*;

  #[test]
  fn test_tlwe_enc_and_dec() {
    let mut rng = rand::thread_rng();

    let key = key::SecretKey::new();
    let key_dirty = key::SecretKey::new();

    let mut correct = 0;
    let try_num = 10000;

    for _i in 0..try_num {
      let sample = rng.gen::<bool>();
      let secret = TLWELv0::encrypt_bool(sample, params::tlwe_lv0::ALPHA, &key.key_lv0);
      let plain = secret.decrypt_bool(&key.key_lv0);
      let plain_dirty = secret.decrypt_bool(&key_dirty.key_lv0);
      assert_eq!(plain, sample);
      if plain != plain_dirty {
        correct += 1;
      }
    }

    let probability = correct as f64 / try_num as f64;
    assert!(probability - 0.50 < 0.01);
  }
}
