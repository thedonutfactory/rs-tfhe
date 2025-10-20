use crate::key::SecretKey;
use crate::params;
use crate::utils::Ciphertext;
use std::cmp::PartialEq;
use std::ops::BitXor;
use std::ops::Shl;

/// Convert a Vector of bits to a number
pub fn convert<T: PartialEq + From<u8> + BitXor<Output = T> + Shl<Output = T> + Clone>(
  bits: &[bool],
) -> T {
  bits
    .iter()
    .rev()
    .map(|a| if *a { 1 } else { 0 })
    .fold(T::from(0), |result, bit| {
      (result << T::from(1)) ^ T::from(bit)
    })
}

fn encrypt(a: bool, secret_key: &SecretKey) -> Ciphertext {
  Ciphertext::encrypt_bool(a, params::tlwe_lv0::ALPHA, &secret_key.key_lv0)
}

pub trait AsBits<T> {
  /// Represent the bits of the byte as an array of boolean values (bits).
  /// Array is in big-endian order, where MSB is the first value of the array.
  fn to_bits(self) -> Vec<bool>;

  /// Encrypt the bits of the type as an array of LweSample values (cipherbits).
  fn encrypt(self, key: &SecretKey) -> Vec<Ciphertext>;
}

impl AsBits<u8> for u8 {
  fn to_bits(self) -> Vec<bool> {
    to_bits(usize::from(self), 8).to_vec()
  }

  fn encrypt(self, key: &SecretKey) -> Vec<Ciphertext> {
    self.to_bits().iter().map(|a| encrypt(*a, key)).collect()
  }
}

impl AsBits<u16> for u16 {
  fn to_bits(self) -> Vec<bool> {
    to_bits(usize::from(self), 16).to_vec()
  }

  fn encrypt(self, key: &SecretKey) -> Vec<Ciphertext> {
    self.to_bits().iter().map(|a| encrypt(*a, key)).collect()
  }
}

impl AsBits<u32> for u32 {
  fn to_bits(self) -> Vec<bool> {
    to_bits(self as usize, 32).to_vec()
  }

  fn encrypt(self, key: &SecretKey) -> Vec<Ciphertext> {
    self.to_bits().iter().map(|a| encrypt(*a, key)).collect()
  }
}

impl AsBits<u64> for u64 {
  fn to_bits(self) -> Vec<bool> {
    to_bits(self as usize, 64).to_vec()
  }

  fn encrypt(self, key: &SecretKey) -> Vec<Ciphertext> {
    self.to_bits().iter().map(|a| encrypt(*a, key)).collect()
  }
}

pub fn to_bits(val: usize, size: usize) -> Vec<bool> {
  let mut vec = Vec::new();
  vec.push((val & 0x1) != 0);
  let base: usize = 2;
  for i in 1..size {
    let t: bool = ((val & base.pow(i as u32)) >> i) != 0;
    vec.push(t);
  }
  vec
}
