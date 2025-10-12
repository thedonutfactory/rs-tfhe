/// Compatibility wrapper for FFT operations
///
/// This module provides a unified interface that automatically selects
/// the appropriate FFT implementation based on the target architecture:
/// - x86_64: SIMD-optimized C++ (SpqliosFFT)
/// - Other: Pure Rust (RustFFTProcessor)
use crate::fft::{DefaultFFTProcessor, FFTProcessor};

pub struct Spqlios {
  processor: DefaultFFTProcessor,
}

impl Spqlios {
  pub fn new(n: usize) -> Self {
    Spqlios {
      processor: DefaultFFTProcessor::new(n),
    }
  }

  pub fn ifft(&mut self, input: &Vec<u32>) -> Vec<f64> {
    self.processor.ifft(input.as_slice())
  }

  pub fn fft(&mut self, input: &Vec<f64>) -> Vec<u32> {
    self.processor.fft(input.as_slice())
  }

  pub fn ifft_1024(&mut self, input: &[u32; 1024]) -> [f64; 1024] {
    self.processor.ifft_1024(input)
  }

  pub fn fft_1024(&mut self, input: &[f64; 1024]) -> [u32; 1024] {
    self.processor.fft_1024(input)
  }

  pub fn poly_mul_1024(&mut self, a: &[u32; 1024], b: &[u32; 1024]) -> [u32; 1024] {
    self.processor.poly_mul_1024(a, b)
  }

  #[allow(dead_code)]
  pub fn poly_mul(&mut self, a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    // Use ifft_1024/fft_1024 if size matches, otherwise use Vec variants
    if a.len() == 1024 && b.len() == 1024 {
      let a_arr: [u32; 1024] = a.as_slice().try_into().unwrap();
      let b_arr: [u32; 1024] = b.as_slice().try_into().unwrap();
      self.poly_mul_1024(&a_arr, &b_arr).to_vec()
    } else {
      let a_ifft = self.ifft(a);
      let b_ifft = self.ifft(b);
      let mut mul = vec![0.0f64; a.len()];

      let ns = a.len() / 2;
      for i in 0..ns {
        let aimbim = a_ifft[i + ns] * b_ifft[i + ns];
        let arebim = a_ifft[i] * b_ifft[i + ns];
        mul[i] = (a_ifft[i] * b_ifft[i] - aimbim) * 0.5;
        mul[i + ns] = (a_ifft[i + ns] * b_ifft[i] + arebim) * 0.5;
      }

      self.fft(&mul)
    }
  }
}
