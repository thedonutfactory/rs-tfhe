use crate::params::Torus;
use crate::tlwe;
use rand_distr::Distribution;

pub type Ciphertext = tlwe::TLWELv0;

pub fn f64_to_torus(d: f64) -> Torus {
  let torus = (d % 1.0) as f64 * 2u64.pow(32) as f64;
  (torus as i64) as u32
}

pub fn f64_to_torus_vec(d: &Vec<f64>) -> Vec<Torus> {
  return d.iter().map(|&e| f64_to_torus(e)).collect();
}

pub fn gaussian_torus(
  mu: Torus,
  normal_distr: &rand_distr::Normal<f64>,
  rng: &mut rand::rngs::ThreadRng,
) -> Torus {
  let sample = normal_distr.sample(rng);
  f64_to_torus(sample).wrapping_add(mu)
}

pub fn gaussian_f64(
  mu: f64,
  normal_distr: &rand_distr::Normal<f64>,
  rng: &mut rand::rngs::ThreadRng,
) -> Torus {
  let mu_torus = f64_to_torus(mu);
  gaussian_torus(mu_torus, normal_distr, rng)
}

pub fn gussian_f64_vec(
  mu: &Vec<f64>,
  normal_distr: &rand_distr::Normal<f64>,
  rng: &mut rand::rngs::ThreadRng,
) -> Vec<Torus> {
  return mu
    .iter()
    .map(|&e| gaussian_torus(f64_to_torus(e), normal_distr, rng))
    .collect();
}

#[cfg(test)]
mod tests {
  use crate::utils::*;
  use rand_distr::Normal;

  #[test]
  fn test_double_to_torust_32bit() {
    let torus = f64_to_torus_vec(&vec![3.141592]);
    assert_eq!(torus[0], 608133009);

    let torus2 = f64_to_torus_vec(&vec![2.71828]);
    assert_eq!(torus2[0], 3084989109);
  }

  #[test]
  fn test_gussian_32bit() {
    let normal = Normal::new(0.0, 0.1).unwrap();
    let mut rng = rand::thread_rng();
    let torus = gussian_torus_vec(&vec![12], &normal, &mut rng);
    assert_eq!(torus.len(), 1);

    let torus2 = gussian_torus_vec(&vec![12, 11], &normal, &mut rng);
    assert_eq!(torus2.len(), 2);
  }

  fn gussian_torus_vec(
    mu: &Vec<Torus>,
    normal_distr: &rand_distr::Normal<f64>,
    rng: &mut rand::rngs::ThreadRng,
  ) -> Vec<Torus> {
    return mu
      .iter()
      .map(|&e| gaussian_torus(e, normal_distr, rng))
      .collect();
  }
}
