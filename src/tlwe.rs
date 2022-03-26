use crate::key;
use crate::params;
use crate::utils;
use rand::Rng;
use std::iter::Iterator;
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Debug, Copy, Clone)]
pub struct TLWELv0 {
  pub p: [u32; params::tlwe_lv0::N + 1],
}

impl TLWELv0 {
  pub fn new() -> TLWELv0 {
    TLWELv0 {
      p: [0; params::tlwe_lv0::N + 1],
    }
  }

  pub fn b(&self) -> u32 {
    self.p[params::tlwe_lv0::N]
  }

  pub fn b_mut(&mut self) -> &mut u32 {
    &mut self.p[params::tlwe_lv0::N]
  }

  pub fn encrypt_f64(p: f64, alpha: f64, key: &key::SecretKeyLv0) -> TLWELv0 {
    let mut rng = rand::thread_rng();
    let mut tlwe = TLWELv0::new();
    let mut inner_product: u32 = 0;

    for i in 0..key.len() {
      let rand_u32: u32 = rng.gen();
      inner_product = inner_product.wrapping_add(key[i] * rand_u32);
      tlwe.p[i] = rand_u32;
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
    let mut inner_product: u32 = 0;
    for i in 0..key.len() {
      inner_product = inner_product.wrapping_add(self.p[i] * key[i]);
    }

    let res_torus = (self.p[params::tlwe_lv0::N].wrapping_sub(inner_product)) as i32;
    res_torus >= 0
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
      *rref = 0u32.wrapping_sub(*sval);
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

  fn add_mul(self, rhs: Rhs, multiplier: u32) -> Self::Output;
}

impl AddMul for &TLWELv0 {
  type Output = TLWELv0;

  fn add_mul(self, other: &TLWELv0, multiplier: u32) -> TLWELv0 {
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

  fn sub_mul(self, rhs: Rhs, multiplier: u32) -> Self::Output;
}

impl SubMul for &TLWELv0 {
  type Output = TLWELv0;

  fn sub_mul(self, other: &TLWELv0, multiplier: u32) -> TLWELv0 {
    let mut res = TLWELv0::new();
    for ((rref, &sval), &oval) in res.p.iter_mut().zip(self.p.iter()).zip(other.p.iter()) {
      *rref = sval.wrapping_sub(oval.wrapping_mul(multiplier));
    }
    res
  }
}

pub struct TLWELv1 {
  pub p: [u32; params::tlwe_lv1::N + 1],
}

impl TLWELv1 {
  pub fn new() -> TLWELv1 {
    TLWELv1 {
      p: [0; params::tlwe_lv1::N + 1],
    }
  }

  pub fn b_mut(&mut self) -> &mut u32 {
    &mut self.p[params::tlwe_lv1::N]
  }

  #[cfg(test)]
  pub fn encrypt_f64(p: f64, alpha: f64, key: &key::SecretKeyLv1) -> TLWELv1 {
    let mut rng = rand::thread_rng();
    let mut tlwe = TLWELv1::new();
    let mut inner_product: u32 = 0;
    for i in 0..key.len() {
      let rand_u32: u32 = rng.gen();
      inner_product = inner_product.wrapping_add(key[i] * rand_u32);
      tlwe.p[i] = rand_u32;
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
    let mut inner_product: u32 = 0;
    for i in 0..key.len() {
      inner_product = inner_product.wrapping_add(self.p[i] * key[i]);
    }

    let res_torus = (self.p[key.len()].wrapping_sub(inner_product)) as i32;
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
