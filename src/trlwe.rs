use crate::key;
use crate::mulfft;
use crate::params;
use crate::tlwe;
use crate::utils;
use rand::Rng;
use std::convert::TryInto;

#[derive(Debug, Copy, Clone)]
pub struct TRLWELv1 {
  pub a: [u32; params::trlwe_lv1::N],
  pub b: [u32; params::trlwe_lv1::N],
}

impl TRLWELv1 {
  pub fn new() -> TRLWELv1 {
    TRLWELv1 {
      a: [0; params::trlwe_lv1::N],
      b: [0; params::trlwe_lv1::N],
    }
  }

  pub fn encrypt_f64(
    p: &Vec<f64>,
    alpha: f64,
    key: &key::SecretKeyLv1,
    plan: &mut mulfft::FFTPlan,
  ) -> TRLWELv1 {
    let mut rng = rand::thread_rng();
    let mut trlwe = TRLWELv1::new();
    trlwe.a.iter_mut().for_each(|e| *e = rng.gen());

    let normal_distr = rand_distr::Normal::new(0.0, alpha).unwrap();
    let mut rng = rand::thread_rng();
    trlwe.b = utils::gussian_f64_vec(p, &normal_distr, &mut rng)
      .try_into()
      .unwrap();
    let poly_res = plan.spqlios.poly_mul_1024(&trlwe.a, key);

    for (bref, rval) in trlwe.b.iter_mut().zip(poly_res.iter()) {
      *bref = bref.wrapping_add(*rval);
    }

    trlwe
  }

  #[allow(dead_code)]
  pub fn encrypt_bool(
    p_bool: &Vec<bool>,
    alpha: f64,
    key: &key::SecretKeyLv1,
    plan: &mut mulfft::FFTPlan,
  ) -> TRLWELv1 {
    let p_f64 = p_bool
      .iter()
      .map(|e| if *e { 0.125f64 } else { -0.125f64 })
      .collect();
    Self::encrypt_f64(&p_f64, alpha, key, plan)
  }

  #[allow(dead_code)]
  pub fn decrypt_bool(&self, key: &key::SecretKeyLv1, plan: &mut mulfft::FFTPlan) -> Vec<bool> {
    let poly_res = plan.spqlios.poly_mul_1024(&self.a, key);
    let mut res: Vec<bool> = Vec::new();
    for i in 0..self.a.len() {
      let value = (self.b[i].wrapping_sub(poly_res[i])) as i32;
      if value < 0 {
        res.push(false);
      } else {
        res.push(true);
      }
    }
    res
  }
}

#[derive(Debug, Copy, Clone)]
pub struct TRLWELv1FFT {
  pub a: [f64; params::trlwe_lv1::N],
  pub b: [f64; params::trlwe_lv1::N],
}

impl TRLWELv1FFT {
  pub fn new(trlwe: &TRLWELv1, plan: &mut mulfft::FFTPlan) -> TRLWELv1FFT {
    TRLWELv1FFT {
      a: plan.spqlios.ifft_1024(&trlwe.a),
      b: plan.spqlios.ifft_1024(&trlwe.b),
    }
  }

  pub fn new_dummy() -> TRLWELv1FFT {
    TRLWELv1FFT {
      a: [0.0f64; params::trlwe_lv1::N],
      b: [0.0f64; params::trlwe_lv1::N],
    }
  }
}

pub fn sample_extract_index(trlwe: &TRLWELv1, k: usize) -> tlwe::TLWELv1 {
  let mut res = tlwe::TLWELv1::new();

  const N: usize = params::trlwe_lv1::N;
  for i in 0..N {
    if i <= k {
      res.p[i] = trlwe.a[k - i];
    } else {
      res.p[i] = u32::MAX - trlwe.a[N + k - i];
    }
  }
  *res.b_mut() = trlwe.b[k];

  res
}

pub fn sample_extract_index_2(trlwe: &TRLWELv1, k: usize) -> tlwe::TLWELv0 {
  let mut res = tlwe::TLWELv0::new();

  const N: usize = params::tlwe_lv0::N;
  for i in 0..N {
    if i <= k {
      res.p[i] = trlwe.a[k - i];
    } else {
      res.p[i] = u32::MAX - trlwe.a[N + k - i];
    }
  }
  *res.b_mut() = trlwe.b[k];

  res
}

#[cfg(test)]
mod tests {
  use crate::key;
  use crate::mulfft;
  use crate::params;
  use crate::trlwe;
  use rand::Rng;

  #[test]
  fn test_trlwe_enc_and_dec() {
    let mut rng = rand::thread_rng();

    // Generate 1024bits secret key
    const N: usize = params::trlwe_lv1::N;
    let key = key::SecretKey::new();
    let key_dirty = key::SecretKey::new();

    let mut plan = mulfft::FFTPlan::new(N);
    let mut correct = 0;
    let try_num = 500;

    for _i in 0..try_num {
      let mut plain_text: Vec<bool> = Vec::new();

      for _j in 0..N {
        let sample: bool = rng.gen::<bool>();
        plain_text.push(sample);
      }

      let c = trlwe::TRLWELv1::encrypt_bool(
        &plain_text,
        params::trlwe_lv1::ALPHA,
        &key.key_lv1,
        &mut plan,
      );

      let dec = c.decrypt_bool(&key.key_lv1, &mut plan);
      let dec_dirty = c.decrypt_bool(&key_dirty.key_lv1, &mut plan);

      for j in 0..N {
        assert_eq!(plain_text[j], dec[j]);
        if plain_text[j] != dec_dirty[j] {
          correct += 1;
        }
      }
    }

    let probability = correct as f64 / (try_num * N) as f64;
    assert!(probability - 0.50 < 0.1);
  }

  #[test]
  fn test_sample_extract_index() {
    let mut rng = rand::thread_rng();

    // Generate 1024bits secret key
    const N: usize = params::trlwe_lv1::N;
    let key = key::SecretKey::new();
    let key_dirty = key::SecretKey::new();

    let mut plan = mulfft::FFTPlan::new(N);
    let mut correct = 0;
    let try_num = 10;

    for _i in 0..try_num {
      let mut plain_text: Vec<bool> = Vec::new();

      for _j in 0..N {
        let sample = rng.gen::<bool>();
        plain_text.push(sample);
      }

      let c = trlwe::TRLWELv1::encrypt_bool(
        &plain_text,
        params::trlwe_lv1::ALPHA,
        &key.key_lv1,
        &mut plan,
      );

      for j in 0..N {
        let tlwe = trlwe::sample_extract_index(&c, j);
        let dec = tlwe.decrypt_bool(&key.key_lv1);
        let dec_dirty = tlwe.decrypt_bool(&key_dirty.key_lv1);
        assert_eq!(plain_text[j], dec);
        if plain_text[j] != dec_dirty {
          correct += 1;
        }
      }
    }

    let probability = correct as f64 / (try_num * N) as f64;
    assert!(probability - 0.50 < 0.1);
  }
}
