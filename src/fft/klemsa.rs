//! Extended FFT Processor - Hybrid High-Performance Implementation
//!
//! Based on: "Fast and Error-Free Negacyclic Integer Convolution using Extended Fourier Transform"
//! by Jakub Klemsa - https://eprint.iacr.org/2021/480
//!
//! **Hybrid Approach:**
//! - Uses rustfft's NEON planner for 512-point FFT (auto-detects SIMD)
//! - Custom twisting factor application (Extended FT method)
//! - Zero-allocation hot path with pre-allocated buffers
//! - Proper scratch buffer usage (process_with_scratch)
//!
//! **Algorithm:**
//! 1. Split N=1024 polynomial into two N/2=512 halves
//! 2. Apply twisting factors (2N-th roots of unity) + convert
//! 3. Rustfft NEON-optimized 512-point FFT (process_with_scratch)
//! 4. Scale and convert output

use super::FFTProcessor;
use crate::params;
use rustfft::num_complex::Complex;
use rustfft::Fft;
use std::cell::RefCell;
use std::f64::consts::PI;
use std::sync::Arc;

/// Extended FFT processor with rustfft optimization
///
/// Uses rustfft's Radix4 with scratch buffers + custom twisting
pub struct KlemsaProcessor {
  // Pre-computed twisting factors (2N-th roots of unity)
  twisties_re: Vec<f64>,
  twisties_im: Vec<f64>,
  // rustfft's optimized N/2-point FFT (512 for N=1024)
  fft_n2_fwd: Arc<dyn Fft<f64>>,
  fft_n2_inv: Arc<dyn Fft<f64>>,
  // Pre-allocated buffers (zero-allocation hot path)
  fourier_buffer: RefCell<Vec<Complex<f64>>>,
  scratch_fwd: RefCell<Vec<Complex<f64>>>,
  scratch_inv: RefCell<Vec<Complex<f64>>>,
}

impl KlemsaProcessor {
  pub fn new(n: usize) -> Self {
    assert!(n.is_power_of_two(), "N must be power of two");
    assert!(n >= 2, "N must be at least 2");

    let n2 = n / 2;

    // Compute twisting factors: exp(i*π*k/N) for k=0..N/2-1
    let mut twisties_re = Vec::with_capacity(n2);
    let mut twisties_im = Vec::with_capacity(n2);
    let twist_unit = PI / (n as f64);
    for i in 0..n2 {
      let angle = i as f64 * twist_unit;
      let (im, re) = angle.sin_cos();
      twisties_re.push(re);
      twisties_im.push(im);
    }

    // Use rustfft's planner - auto-detects NEON (ARM), AVX (x86), or scalar
    // FftPlanner::new() already checks for SIMD features at runtime!
    use rustfft::FftPlanner;
    let mut planner = FftPlanner::new();
    let fft_n2_fwd = planner.plan_fft_forward(n2);
    let fft_n2_inv = planner.plan_fft_inverse(n2);

    // Pre-allocate scratch buffers
    let scratch_fwd_len = fft_n2_fwd.get_inplace_scratch_len();
    let scratch_inv_len = fft_n2_inv.get_inplace_scratch_len();

    KlemsaProcessor {
      twisties_re,
      twisties_im,
      fft_n2_fwd,
      fft_n2_inv,
      fourier_buffer: RefCell::new(vec![Complex::new(0.0, 0.0); n2]),
      scratch_fwd: RefCell::new(vec![Complex::new(0.0, 0.0); scratch_fwd_len]),
      scratch_inv: RefCell::new(vec![Complex::new(0.0, 0.0); scratch_inv_len]),
    }
  }
}

impl FFTProcessor for KlemsaProcessor {
  fn new(n: usize) -> Self {
    KlemsaProcessor::new(n)
  }

  fn ifft<const N: usize>(&mut self, input: &[params::Torus; N]) -> [f64; N] {
    let n2 = N / 2;

    let (input_re, input_im) = input.split_at(n2);

    // Apply twisting factors and convert (optimized for cache)
    let mut fourier = self.fourier_buffer.borrow_mut();
    for i in 0..n2 {
      let in_re = input_re[i] as i32 as f64;
      let in_im = input_im[i] as i32 as f64;
      let w_re = self.twisties_re[i];
      let w_im = self.twisties_im[i];
      fourier[i] = Complex::new(in_re * w_re - in_im * w_im, in_re * w_im + in_im * w_re);
    }

    // Use rustfft's Radix4 with scratch buffer
    let mut scratch = self.scratch_fwd.borrow_mut();
    self
      .fft_n2_fwd
      .process_with_scratch(&mut fourier, &mut scratch);

    // Scale by 2 and convert to output
    let mut result = [0.0f64; N];
    for i in 0..n2 {
      result[i] = fourier[i].re * 2.0;
      result[i + n2] = fourier[i].im * 2.0;
    }

    result
  }

  fn fft<const N: usize>(&mut self, input: &[f64; N]) -> [params::Torus; N] {
    let n2 = N / 2;

    // Convert to complex and scale
    let (input_re, input_im) = input.split_at(n2);
    let mut fourier = self.fourier_buffer.borrow_mut();
    for i in 0..n2 {
      fourier[i] = Complex::new(input_re[i] * 0.5, input_im[i] * 0.5);
    }

    // Use rustfft's Radix4 IFFT with scratch buffer
    let mut scratch = self.scratch_inv.borrow_mut();
    self
      .fft_n2_inv
      .process_with_scratch(&mut fourier, &mut scratch);

    // Apply inverse twisting and convert to u32
    let normalization = 1.0 / (n2 as f64);
    let mut result = [params::ZERO_TORUS; N];
    for i in 0..n2 {
      let w_re = self.twisties_re[i];
      let w_im = self.twisties_im[i];
      let f_re = fourier[i].re;
      let f_im = fourier[i].im;
      let tmp_re = (f_re * w_re + f_im * w_im) * normalization;
      let tmp_im = (f_im * w_re - f_re * w_im) * normalization;
      result[i] = tmp_re.round() as i64 as params::Torus;
      result[i + n2] = tmp_im.round() as i64 as params::Torus;
    }

    result
  }

  fn poly_mul<const N: usize>(
    &mut self,
    a: &[params::Torus; N],
    b: &[params::Torus; N],
  ) -> [params::Torus; N] {
    let a_fft = self.ifft::<N>(a);
    let b_fft = self.ifft::<N>(b);

    // Complex multiplication with 0.5 scaling for negacyclic
    let mut result_fft = [0.0f64; N];
    let n2 = N / 2;
    for i in 0..n2 {
      let ar = a_fft[i];
      let ai = a_fft[i + n2];
      let br = b_fft[i];
      let bi = b_fft[i + n2];

      result_fft[i] = (ar * br - ai * bi) * 0.5;
      result_fft[i + n2] = (ar * bi + ai * br) * 0.5;
    }

    self.fft::<N>(&result_fft)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::params::TORUS_SIZE;

  #[test]
  fn test_klemsa_roundtrip() {
    const N: usize = 1024;
    let mut proc = KlemsaProcessor::new(N);

    let mut input = [0; N];
    input[0] = 1 << (TORUS_SIZE - 1);
    input[5] = 1 << (TORUS_SIZE - 2);

    let freq = proc.ifft::<N>(&input);
    let output = proc.fft::<N>(&freq);

    let mut max_diff: i64 = 0;
    for i in 0..N {
      let diff = (output[i] as i64 - input[i] as i64).abs();
      max_diff = max_diff.max(diff);
    }

    println!("Klemsa roundtrip error: {}", max_diff);
    assert!(max_diff < 2, "Roundtrip error too large: {}", max_diff);
  }
}
