use rustfft::FftPlanner;
use itertools::Itertools;
use rustfft::num_complex::Complex;
// let mut planner = FftPlanner::new();

pub struct FFTPlan {
  // pub spqlios: spqlios::Spqlios,
  pub planner: FftPlanner<f64>,
  pub n: usize,
}

impl FFTPlan {
  pub fn new(n: usize) -> FFTPlan {
    FFTPlan {
      planner: FftPlanner::new(),
      n,
    }
  }

  pub fn ifft_1024(&mut self, input: &[u32; 1024]) -> [f64; 1024] {
    let src_const_ptr = input.as_ptr() as *const _;
    //let mut res = Box::new([0f64; 1024]);
    
    //let mut planner = FftPlanner::new();
    
    let mut res: Vec<Complex<f64>> = input
    .iter()
    .rev()
    .map(|x| f64::from(*x))
    .chain(std::iter::repeat(0_f64).take(a.len() - 1))
    .map(|x| Complex::new(x, 0_f64))
    .collect();

    let fft = self.planner.plan_fft(input.len(), rustfft::FftDirection::Inverse);
    fft.process(res);
    
    let rrr = res
      .iter()
      .map(|x: &Complex<f64>| x.re)
      .collect();
    
    return rrr;
  }

  pub fn fft_1024(&mut self, input: &[f64; 1024]) -> [u32; 1024] {
    let src_const_ptr = input.as_ptr() as *const _;
    let mut res = Box::new([0u32; 1024]);
    
    //let mut planner = FftPlanner::new();
    let fft = self.planner.plan_fft(input.len(), rustfft::FftDirection::Forward);
    fft.process(res);
    *res
  }

  pub fn poly_mul_1024(&mut self, a: &[u32; 1024], b: &[u32; 1024]) -> [u32; 1024] {
  // fn poly_mul_1024(a: &Vec<i32>, b: &Vec<i32>) -> Vec<i32> {
    // Algorithm found at https://math.stackexchange.com/questions/764727/concrete-fft-polynomial-multiplication-example
    use rustfft::num_traits::Zero;
    use rustfft::num_complex::Complex;
    use rustfft::FftPlanner;
  
    // let degree = a.degree() + b.degree();
    let degree = 1024 + 1024;
    let mut p: Vec<_> = a
      .iter()
      .rev()
      .map(|x| f64::from(*x))
      .chain(std::iter::repeat(0_f64).take(b.len() - 1))
      .map(|x| Complex::new(x, 0_f64))
      .collect();
  
    let power = p.len().next_power_of_two();
    if power != p.len() {
      // Extend the polynomial to a power of 2 length
      p.extend(
        std::iter::repeat(Complex::<f64>::zero())
          .take(power - p.len())
          .collect::<Vec<Complex<f64>>>(),
      );
    }
  
    let mut q: Vec<Complex<f64>> = b
      .iter()
      .rev()
      .map(|x| f64::from(*x))
      .chain(std::iter::repeat(0_f64).take(a.len() - 1))
      .map(|x| Complex::new(x, 0_f64))
      .collect();
  
    let power = q.len().next_power_of_two();
    if power != q.len() {
      // Extend the polynomial to a power of 2 length
      q.extend(
        std::iter::repeat(Complex::zero())
          .take(power - q.len())
          .collect::<Vec<Complex<f64>>>(),
      );
    }
  
    // Create a FFT planner for a FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft(p.len(), rustfft::FftDirection::Forward);
    fft.process(&mut p);
    fft.process(&mut q);
  
    let p_len = p.len();
    let q_len = q.len();
  
    let mut r: Vec<_> = p
      .into_iter()
      .zip_eq(q.into_iter())
      .map(|(p_c, q_c)| (p_c / (p_len as f64).sqrt()) * (q_c / (q_len as f64).sqrt()))
      .collect();
  
    // Create a FFT planner for the inverse FFT
    let fft = FftPlanner::new().plan_fft(r.len(), rustfft::FftDirection::Inverse);
    fft.process(&mut r);
  
    let coefs: Vec<u32> = r
      .into_iter()
      .take(degree + 1)
      .map(|x| x.re.round() as u32)
      .rev()
      .collect();
  
    let ret_coefs = coefs.try_into()
      .unwrap_or_else(|v: Vec<u32>| panic!("Expected a Vec of length {} but it was {}", 1024, v.len()));

    ret_coefs
  }
  
}

/*
#[cfg(test)]
mod tests {
  use crate::mulfft::*;
  use crate::params;
  use rand::Rng;

  #[test]
  fn test_spqlios_fft_ifft() {
    let n = 1024;
    let mut plan = FFTPlan::new(n);
    let mut rng = rand::thread_rng();
    let mut a: Vec<u32> = vec![0u32; n];
    a.iter_mut().for_each(|e| *e = rng.gen::<u32>());

    let a_fft = plan.spqlios.ifft(&a);
    let res = plan.spqlios.fft(&a_fft);
    for i in 0..n {
      let diff = a[i] as i32 - res[i] as i32;
      assert!(diff < 2 && diff > -2);
      println!("{} {} {}", a_fft[i], a[i], res[i]);
    }
  }

  #[test]
  fn test_spqlios_poly_mul() {
    let n = 1024;
    let mut plan = FFTPlan::new(n);
    let mut rng = rand::thread_rng();
    let mut a: Vec<u32> = vec![0u32; n];
    let mut b: Vec<u32> = vec![0u32; n];
    a.iter_mut().for_each(|e| *e = rng.gen::<u32>());
    b.iter_mut()
      .for_each(|e| *e = rng.gen::<u32>() % params::trgsw_lv1::BG as u32);

    let spqlios_res = plan.spqlios.poly_mul(&a, &b);
    let res = poly_mul(&a.to_vec(), &b.to_vec());
    for i in 0..n {
      let diff = res[i] as i32 - spqlios_res[i] as i32;
      assert!(diff < 2 && diff > -2);
    }
  }

  #[test]
  fn test_spqlios_fft_ifft_1024() {
    let mut plan = FFTPlan::new(1024);
    let mut rng = rand::thread_rng();
    let mut a = [0u32; 1024];
    a.iter_mut().for_each(|e| *e = rng.gen::<u32>());

    let a_fft = plan.spqlios.ifft_1024(&a);
    let res = plan.spqlios.fft_1024(&a_fft);
    for i in 0..1024 {
      let diff = a[i] as i32 - res[i] as i32;
      assert!(diff < 2 && diff > -2);
    }
  }

  #[test]
  fn test_spqlios_poly_mul_1024() {
    let mut plan = FFTPlan::new(1024);
    let mut rng = rand::thread_rng();
    for _i in 0..100 {
      let mut a = [0u32; 1024];
      let mut b = [0u32; 1024];
      a.iter_mut().for_each(|e| *e = rng.gen::<u32>());
      b.iter_mut()
        .for_each(|e| *e = rng.gen::<u32>() % params::trgsw_lv1::BG as u32);

      let spqlios_res = plan.spqlios.poly_mul_1024(&a, &b);
      let res = poly_mul(&a.to_vec(), &b.to_vec());
      for i in 0..1024 {
        let diff = res[i] as i32 - spqlios_res[i] as i32;
        assert!(diff < 2 && diff > -2);
      }
    }
  }

  fn poly_mul(a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    let n = a.len();
    let mut res: Vec<u32> = vec![0u32; n];

    for i in 0..n {
      for j in 0..n {
        if i + j < n {
          res[i + j] = res[i + j].wrapping_add(a[i].wrapping_mul(b[j]));
        } else {
          res[i + j - n] = res[i + j - n].wrapping_sub(a[i].wrapping_mul(b[j]));
        }
      }
    }

    res
  }
}
*/