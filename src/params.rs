pub type Torus = u32;

pub const KSK_ALPHA: f64 = tlwe_lv0::ALPHA;
pub const BSK_ALPHA: f64 = tlwe_lv1::ALPHA;

pub mod tlwe_lv0 {

  pub const N: usize = 630;
  pub const ALPHA: f64 = 3.0517578125e-05;
}

pub mod tlwe_lv1 {

  pub const N: usize = 1024;
  pub const ALPHA: f64 = 2.980_232_238_769_531_3e-8;
}

pub mod trlwe_lv1 {
  use crate::params;

  pub const N: usize = params::tlwe_lv1::N;

  #[cfg(test)]
  pub const ALPHA: f64 = params::tlwe_lv1::ALPHA;
}

pub mod trgsw_lv1 {
  use crate::params;

  pub const N: usize = params::tlwe_lv1::N;
  pub const NBIT: usize = 10;
  pub const BGBIT: u32 = 6;
  pub const BG: u32 = 1 << BGBIT;
  pub const L: usize = 3;
  pub const BASEBIT: usize = 2;
  pub const IKS_T: usize = 8;

  #[cfg(test)]
  pub const ALPHA: f64 = params::tlwe_lv1::ALPHA;
}
