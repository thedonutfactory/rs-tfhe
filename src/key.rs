use crate::params;
use crate::tlwe;
use crate::trgsw;
use crate::trlwe;
use crate::utils;
use rand::Rng;

const TRGSWLV1_N: usize = params::trgsw_lv1::N;
const TRGSWLV1_IKS_T: usize = params::trgsw_lv1::IKS_T;
const TRGSWLV1_BASE: usize = 1 << params::trgsw_lv1::BASEBIT;

pub type SecretKeyLv0 = [u32; params::tlwe_lv0::N];
pub type SecretKeyLv1 = [u32; params::tlwe_lv1::N];
pub type KeySwitchingKey = Vec<tlwe::TLWELv0>;
pub type BootstrappingKey = Vec<trgsw::TRGSWLv1FFT>;

pub struct SecretKey {
  pub key_lv0: SecretKeyLv0,
  pub key_lv1: SecretKeyLv1,
}

impl SecretKey {
  pub fn new() -> Self {
    let mut rng = rand::thread_rng();
    let mut key = SecretKey {
      key_lv0: [0u32; params::tlwe_lv0::N],
      key_lv1: [0u32; params::tlwe_lv1::N],
    };
    key
      .key_lv0
      .iter_mut()
      .for_each(|e| *e = rng.gen::<bool>() as u32);
    key
      .key_lv1
      .iter_mut()
      .for_each(|e| *e = rng.gen::<bool>() as u32);
    key
  }
}

pub struct CloudKey {
  pub decomposition_offset: u32,
  pub blind_rotate_testvec: trlwe::TRLWELv1,
  pub key_switching_key: KeySwitchingKey,
  pub bootstrapping_key: BootstrappingKey,
}

impl CloudKey {
  pub fn new(secret_key: &SecretKey) -> Self {
    CloudKey {
      decomposition_offset: gen_decomposition_offset(),
      blind_rotate_testvec: gen_testvec(),
      key_switching_key: gen_key_switching_key(secret_key),
      bootstrapping_key: gen_bootstrapping_key(secret_key),
    }
  }

  pub fn new_no_ksk() -> Self {
    CloudKey {
      decomposition_offset: gen_decomposition_offset(),
      blind_rotate_testvec: gen_testvec(),
      key_switching_key: vec![tlwe::TLWELv0::new(); TRGSWLV1_BASE * TRGSWLV1_IKS_T * TRGSWLV1_N],
      bootstrapping_key: vec![trgsw::TRGSWLv1FFT::new_dummy(); params::tlwe_lv0::N],
    }
  }
}

pub fn gen_decomposition_offset() -> u32 {
  let mut offset: u32 = 0;

  for i in 0..(params::trgsw_lv1::L as u32) {
    offset = offset
      .wrapping_add(params::trgsw_lv1::BG / 2 * (1 << (32 - (i + 1) * params::trgsw_lv1::BGBIT)));
  }

  offset
}

pub fn gen_testvec() -> trlwe::TRLWELv1 {
  let mut testvec = trlwe::TRLWELv1::new();
  let b_torus = utils::f64_to_torus(0.125);
  for i in 0..params::trgsw_lv1::N {
    testvec.a[i] = 0;
    testvec.b[i] = b_torus;
  }

  testvec
}

pub fn gen_key_switching_key(secret_key: &SecretKey) -> KeySwitchingKey {
  const BASEBIT: usize = params::trgsw_lv1::BASEBIT;
  const IKS_T: usize = params::trgsw_lv1::IKS_T;
  let mut res = vec![tlwe::TLWELv0::new(); TRGSWLV1_BASE * TRGSWLV1_IKS_T * TRGSWLV1_N];

  for i in 0..params::trgsw_lv1::N {
    for j in 0..IKS_T {
      for k in 0..TRGSWLV1_BASE {
        if k == 0 {
          continue;
        }
        let p =
          ((k as u32 * secret_key.key_lv1[i] as u32) as f64) / ((1 << ((j + 1) * BASEBIT)) as f64);
        let idx = (TRGSWLV1_BASE * TRGSWLV1_IKS_T * i) + (TRGSWLV1_BASE * j) + k;
        res[idx] = tlwe::TLWELv0::encrypt_f64(p, params::KSK_ALPHA, &secret_key.key_lv0);
      }
    }
  }

  res
}

pub fn gen_bootstrapping_key(
  secret_key: &SecretKey,
) -> BootstrappingKey {
  crate::context::FFT_PLAN.with(|plan| {
    let p = &mut plan.borrow_mut();
    let mut res = vec![trgsw::TRGSWLv1FFT::new_dummy(); params::tlwe_lv0::N];
    res
      .iter_mut()
      .zip(secret_key.key_lv0.iter())
      .for_each(|(rref, &kval)| {
        *rref = trgsw::TRGSWLv1FFT::new(
          &trgsw::TRGSWLv1::encrypt_torus(kval, params::BSK_ALPHA, &secret_key.key_lv1, p),
          p,
        )
      });
    res
  })
}
