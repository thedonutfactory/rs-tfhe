use rustfft::{algorithm::Radix4, num_complex::Complex, Fft, FftDirection, FftNum};

pub struct FFTPlan {
    pub fft_processor: Radix4<f64>,
    pub ifft_processor: Radix4<f64>,
    pub n: usize,
}

impl FFTPlan {
    pub fn new(n: usize) -> FFTPlan {
        return FFTPlan {
            fft_processor: Radix4::new(n, FftDirection::Forward),
            ifft_processor: Radix4::new(n, FftDirection::Inverse),
            n: n,
        };
    }

    fn execute_reverse_torus(&mut self, a: &Vec<u32>) -> Vec<f64> {
      let n = a.len();
	  let n_s_two = n / 2;
	  let two_n = n * 2;
      let two_pm_33: f64 = 1./ 2u64.pow(32) as f64;

      let mut complex_in_out = vec![Complex{ re: 0.0f64, im: 0.0f64 }; two_n];

      for i in 0..n {
        complex_in_out[i] = Complex::new(f64::try_from(a[i]).unwrap() * two_pm_33, 0.);
      }
      for i in 0..n {
        complex_in_out[n+i] = Complex{ re: -complex_in_out[i].re, im: 0.0f64 };
      }
      self.ifft_processor.process(&mut complex_in_out);
      
      let mut res = vec![0.0f64; n_s_two];
      for i in 0..n_s_two {
        res.push(complex_in_out[2*i+1].re);
      }
      res
    }

    fn execute_direct_torus(&mut self, a: &Vec<f64>) -> Vec<u32> {
      let n = a.len();
	  let n_s_two = n / 2;
	  let two_n = n * 2;
      let two_p32: f64 =  2u64.pow(32) as f64;
      let one_over_n: f64 = 1. / 1024. as f64;
      let mut complex_in_out = vec![Complex{ re: 0.0f64, im: 0.0f64 }; two_n];
      
      for i in 0..n {
        complex_in_out[2*i] = Complex::new(0., 0.);
      }
      for i in 0..n_s_two {
        complex_in_out[2*i+1] = Complex::new(a[i], a[i]);
      }
      for i in 0..n_s_two {
        complex_in_out[two_n-1-2*i] = Complex::new(a[i], a[i]);
      }
      self.fft_processor.process(&mut complex_in_out);

      let mut res: Vec<u32> = Vec::new();
      res.push((complex_in_out[0].re * one_over_n * two_p32) as u32);
      for i in 1..n {
        res.push(-(complex_in_out[i].re * one_over_n * two_p32) as u32);
      }
      return res;
    }

    pub fn ifft(&mut self, input: &Vec<u32>) -> Vec<f64> {
      return self.execute_reverse_torus(input);
    }

    pub fn fft(&mut self, a: &Vec<f64>) -> Vec<u32> {
      return self.execute_direct_torus(a);
    }

    pub fn ifft_1024(&mut self, input: &[u32; 1024]) -> [f64; 1024] {
        return self
            .ifft(&input.to_vec())
            .try_into()
            .unwrap_or_else(|v: Vec<f64>| {
                panic!("Expected a Vec of length {} but it was {}", 1024, v.len())
            });
    }

    pub fn fft_1024(&mut self, input: &[f64; 1024]) -> [u32; 1024] {
        return self
            .fft(&input.to_vec())
            .try_into()
            .unwrap_or_else(|v: Vec<u32>| {
                panic!("Expected a Vec of length {} but it was {}", 1024, v.len())
            });
    }

    pub fn poly_mul_1024(&mut self, a: &[u32; 1024], b: &[u32; 1024]) -> [u32; 1024] {
        return self
            .poly_mul(&a.to_vec(), &b.to_vec())
            .try_into()
            .unwrap_or_else(|v: Vec<u32>| {
                panic!("Expected a Vec of length {} but it was {}", 1024, v.len())
            });
    }

    pub fn poly_mul(&mut self, a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
        let a_ifft = self.execute_reverse_torus(a);
        let b_ifft = self.execute_reverse_torus(b);
        let mut mul = vec![0.0f64; self.n];

        let ns = self.n / 2;
        for i in 0..ns {
            let aimbim = a_ifft[i + ns] * b_ifft[i + ns];
            let arebim = a_ifft[i] * b_ifft[i + ns];
            mul[i] = a_ifft[i] * b_ifft[i] - aimbim;
            mul[i + ns] = a_ifft[i + ns] * b_ifft[i] + arebim;
        }
        return self.fft(&mul);
        //return self.execute_direct_torus(&mul);
        // return self.fft(&mul.iter().map(|x| *x as u32).collect::<Vec<u32>>());
    }
}

#[cfg(test)]
mod tests {
    use crate::mulfft::*;
    use crate::params;
    use rand::Rng;

    /*
    #[test]
    fn test_rustfft_fft_ifft() {
        let n = 1024;
        let mut plan = FFTPlan::new(n);
        let mut rng = rand::thread_rng();
        let mut a: Vec<u32> = vec![0u32; n];
        a.iter_mut().for_each(|e| *e = rng.gen::<u32>());

        let a_fft = plan.ifft(&a);
        let res = plan.fft(&a_fft);
        for i in 0..n {
            let diff = a[i] as i32 - res[i] as i32;
            assert!(diff < 2 && diff > -2);
            println!("{} {} {}", a_fft[i], a[i], res[i]);
        }
    }
    */

    #[test]
    fn test_rustfft_poly_mul() {
        let n = 1024;
        let mut plan = FFTPlan::new(n);
        let mut rng = rand::thread_rng();
        let mut a: Vec<u32> = vec![0u32; n];
        let mut b: Vec<u32> = vec![0u32; n];
        a.iter_mut().for_each(|e| *e = rng.gen::<u32>());
        b.iter_mut()
            .for_each(|e| *e = rng.gen::<u32>() % params::trgsw_lv1::BG as u32);

        let spqlios_res = plan.poly_mul(&a, &b);
        let res = poly_mul(&a.to_vec(), &b.to_vec());
        for i in 0..n {
            let diff = res[i] as i32 - spqlios_res[i] as i32;
            assert!(diff < 2 && diff > -2);
        }
    }

    /*
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
    */

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

        return res;
    }
}
