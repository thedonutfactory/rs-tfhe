use crate::fft::{DefaultFFTProcessor, FFTProcessor};
// use crate::spqlios;

pub struct FFTPlan {
  pub processor: DefaultFFTProcessor,
  pub n: usize,
}

impl FFTPlan {
  pub fn new(n: usize) -> FFTPlan {
    FFTPlan {
      processor: DefaultFFTProcessor::new(n),
      n,
    }
  }
}

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

    let a_fft = plan.processor.ifft(&a);
    let res = plan.processor.fft(&a_fft);
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

    let spqlios_res = plan.processor.poly_mul(&a, &b);
    let res = poly_mul(&a.to_vec(), &b.to_vec());
    for i in 0..n {
      let diff = res[i] as i64 - spqlios_res[i] as i64;
      assert!(
        diff < 2 && diff > -2,
        "Failed at index {}: expected={}, got={}, diff={}",
        i,
        res[i],
        spqlios_res[i],
        diff
      );
    }
  }

  #[test]
  fn test_spqlios_simple() {
    let mut plan = FFTPlan::new(1024);
    // Delta function test
    let mut a = [0u32; 1024];
    a[0] = 1000;
    let freq = plan.processor.ifft_1024(&a);
    let res = plan.processor.fft_1024(&freq);
    println!(
      "Delta: in[0]={}, out[0]={}, diff={}",
      a[0],
      res[0],
      a[0] as i64 - res[0] as i64
    );
    assert!((a[0] as i64 - res[0] as i64).abs() < 10);
  }

  #[test]
  fn test_spqlios_fft_ifft_1024() {
    let mut plan = FFTPlan::new(1024);
    let mut rng = rand::thread_rng();
    let mut a = [0u32; 1024];
    a.iter_mut().for_each(|e| *e = rng.gen::<u32>());

    let a_fft = plan.processor.ifft_1024(&a);
    let res = plan.processor.fft_1024(&a_fft);

    let mut max_diff = 0i64;
    for i in 0..1024 {
      let diff = (a[i] as i64 - res[i] as i64).abs();
      if diff > max_diff {
        max_diff = diff;
      }
    }
    println!("Max difference: {}", max_diff);

    for i in 0..1024 {
      let diff = a[i] as i32 - res[i] as i32;
      assert!(
        diff < 2 && diff > -2,
        "Failed at index {}: input={}, output={}, diff={}",
        i,
        a[i],
        res[i],
        diff
      );
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

      let spqlios_res = plan.processor.poly_mul_1024(&a, &b);
      let res = poly_mul(&a.to_vec(), &b.to_vec());
      for i in 0..1024 {
        let diff = res[i] as i64 - spqlios_res[i] as i64;
        assert!(
          diff < 2 && diff > -2,
          "Failed at index {}: expected={}, got={}, diff={}",
          i,
          res[i],
          spqlios_res[i],
          diff
        );
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
