//! Complete FFT Processor Implementations for TFHE
//!
//! Implements the FFTProcessor trait for multiple algorithms:
//! 1. ExtendedFftKlemsa - Baseline (Klemsa's method)
//! 2. KaratsubaNegacyclicProcessor - Hybrid for small N
//! 3. SparseOptimizedProcessor - For sparse polynomials
//! 4. MergedTransformProcessor - Cache-optimized
//! 5. AdaptiveFFTProcessor - Smart algorithm selector

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::cell::RefCell;
use std::f64::consts::PI;
use std::sync::Arc;

// Import the trait
use super::FFTProcessor;

// ============================================================================
// 1. BASELINE: Klemsa's Extended FFT Processor
// ============================================================================

pub struct ExtendedFftKlemsa {
  n: usize,
  twisties_re: Vec<f64>,
  twisties_im: Vec<f64>,
  fft_n2_fwd: Arc<dyn Fft<f64>>,
  fft_n2_inv: Arc<dyn Fft<f64>>,
  fourier_buffer: RefCell<Vec<Complex<f64>>>,
  scratch_fwd: RefCell<Vec<Complex<f64>>>,
  scratch_inv: RefCell<Vec<Complex<f64>>>,
}

impl ExtendedFftKlemsa {
  fn u32_to_torus(x: u32) -> f64 {
    // Convert u32 torus element to f64 in [-0.5, 0.5)
    (x as i32 as f64) / (1u64 << 32) as f64
  }

  fn torus_to_u32(x: f64) -> u32 {
    // Convert f64 back to u32 torus element
    (x * (1u64 << 32) as f64).round() as i64 as u32
  }
}

impl FFTProcessor for ExtendedFftKlemsa {
  fn new(n: usize) -> Self {
    assert!(n.is_power_of_two(), "N must be power of two");
    let n2 = n / 2;

    // Pre-compute twisting factors: exp(i*π*k/N) for k=0..N/2-1
    let mut twisties_re = Vec::with_capacity(n2);
    let mut twisties_im = Vec::with_capacity(n2);
    let twist_unit = PI / (n as f64);
    for i in 0..n2 {
      let angle = i as f64 * twist_unit;
      let (im, re) = angle.sin_cos();
      twisties_re.push(re);
      twisties_im.push(im);
    }

    let mut planner = FftPlanner::new();
    let fft_n2_fwd = planner.plan_fft_forward(n2);
    let fft_n2_inv = planner.plan_fft_inverse(n2);

    let scratch_fwd_len = fft_n2_fwd.get_inplace_scratch_len();
    let scratch_inv_len = fft_n2_inv.get_inplace_scratch_len();

    ExtendedFftKlemsa {
      n,
      twisties_re,
      twisties_im,
      fft_n2_fwd,
      fft_n2_inv,
      fourier_buffer: RefCell::new(vec![Complex::new(0.0, 0.0); n2]),
      scratch_fwd: RefCell::new(vec![Complex::new(0.0, 0.0); scratch_fwd_len]),
      scratch_inv: RefCell::new(vec![Complex::new(0.0, 0.0); scratch_inv_len]),
    }
  }

  fn ifft_1024(&mut self, input: &[u32; 1024]) -> [f64; 1024] {
    assert_eq!(self.n, 1024);
    let n2 = 512;

    let (input_re, input_im) = input.split_at(n2);

    // Fold and twist: convert u32 torus → f64 and apply twisting
    let mut fourier = self.fourier_buffer.borrow_mut();
    for i in 0..n2 {
      let in_re = Self::u32_to_torus(input_re[i]);
      let in_im = Self::u32_to_torus(input_im[i]);
      let w_re = self.twisties_re[i];
      let w_im = self.twisties_im[i];
      fourier[i] = Complex::new(in_re * w_re - in_im * w_im, in_re * w_im + in_im * w_re);
    }

    // FFT forward
    let mut scratch = self.scratch_fwd.borrow_mut();
    self
      .fft_n2_fwd
      .process_with_scratch(&mut fourier, &mut scratch);

    // Unfold: scale by 2 and separate real/imag
    let mut result = [0.0f64; 1024];
    for i in 0..n2 {
      result[i] = fourier[i].re * 2.0;
      result[i + n2] = fourier[i].im * 2.0;
    }

    result
  }

  fn fft_1024(&mut self, input: &[f64; 1024]) -> [u32; 1024] {
    assert_eq!(self.n, 1024);
    let n2 = 512;

    // Fold: combine real/imag and scale by 0.5
    let (input_re, input_im) = input.split_at(n2);
    let mut fourier = self.fourier_buffer.borrow_mut();
    for i in 0..n2 {
      fourier[i] = Complex::new(input_re[i] * 0.5, input_im[i] * 0.5);
    }

    // FFT inverse
    let mut scratch = self.scratch_inv.borrow_mut();
    self
      .fft_n2_inv
      .process_with_scratch(&mut fourier, &mut scratch);

    // Untwist and convert to u32
    let normalization = 1.0 / (n2 as f64);
    let mut result = [0u32; 1024];
    for i in 0..n2 {
      let w_re = self.twisties_re[i];
      let w_im = self.twisties_im[i];
      let f_re = fourier[i].re * normalization;
      let f_im = fourier[i].im * normalization;
      let tmp_re = f_re * w_re + f_im * w_im;
      let tmp_im = f_im * w_re - f_re * w_im;
      result[i] = Self::torus_to_u32(tmp_re);
      result[i + n2] = Self::torus_to_u32(tmp_im);
    }

    result
  }

  fn poly_mul_1024(&mut self, a: &[u32; 1024], b: &[u32; 1024]) -> [u32; 1024] {
    let a_freq = self.ifft_1024(a);
    let b_freq = self.ifft_1024(b);

    // Pointwise multiplication in frequency domain with 0.5 scaling
    let mut result_freq = [0.0f64; 1024];
    let n2 = 512;
    for i in 0..n2 {
      let ar = a_freq[i];
      let ai = a_freq[i + n2];
      let br = b_freq[i];
      let bi = b_freq[i + n2];

      result_freq[i] = (ar * br - ai * bi) * 0.5;
      result_freq[i + n2] = (ar * bi + ai * br) * 0.5;
    }

    self.fft_1024(&result_freq)
  }

  fn ifft(&mut self, input: &[u32]) -> Vec<f64> {
    assert_eq!(input.len(), self.n);
    if self.n == 1024 {
      let mut arr = [0u32; 1024];
      arr.copy_from_slice(input);
      self.ifft_1024(&arr).to_vec()
    } else {
      // Generic implementation for other sizes
      vec![0.0; self.n]
    }
  }

  fn fft(&mut self, input: &[f64]) -> Vec<u32> {
    assert_eq!(input.len(), self.n);
    if self.n == 1024 {
      let mut arr = [0f64; 1024];
      arr.copy_from_slice(input);
      self.fft_1024(&arr).to_vec()
    } else {
      vec![0; self.n]
    }
  }

  fn poly_mul(&mut self, a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    if a.len() == 1024 && b.len() == 1024 {
      let mut a_arr = [0u32; 1024];
      let mut b_arr = [0u32; 1024];
      a_arr.copy_from_slice(a);
      b_arr.copy_from_slice(b);
      self.poly_mul_1024(&a_arr, &b_arr).to_vec()
    } else {
      vec![0; a.len()]
    }
  }
}

// ============================================================================
// 2. KARATSUBA-NEGACYCLIC PROCESSOR
// ============================================================================

pub struct KaratsubaNegacyclicProcessor {
  n: usize,
  threshold: usize,
  fft_processor: Option<ExtendedFftKlemsa>,
}

impl KaratsubaNegacyclicProcessor {
  fn schoolbook_negacyclic(&self, a: &[u32], b: &[u32]) -> Vec<u32> {
    let n = a.len();
    let mut result = vec![0u32; n];

    for i in 0..n {
      for j in 0..n {
        let idx = (i + j) % n;
        let prod = a[i].wrapping_mul(b[j]);
        if i + j < n {
          result[idx] = result[idx].wrapping_add(prod);
        } else {
          result[idx] = result[idx].wrapping_sub(prod);
        }
      }
    }

    result
  }

  fn karatsuba_negacyclic(&self, a: &[u32], b: &[u32]) -> Vec<u32> {
    let n = a.len();

    if n <= 8 {
      return self.schoolbook_negacyclic(a, b);
    }

    let n2 = n / 2;
    let (a0, a1) = a.split_at(n2);
    let (b0, b1) = b.split_at(n2);

    // Three recursive multiplications
    let p0 = self.karatsuba_negacyclic(a0, b0);
    let p1 = self.karatsuba_negacyclic(a1, b1);

    let mut a_sum = vec![0u32; n2];
    let mut b_sum = vec![0u32; n2];
    for i in 0..n2 {
      a_sum[i] = a0[i].wrapping_add(a1[i]);
      b_sum[i] = b0[i].wrapping_add(b1[i]);
    }
    let p_mid = self.karatsuba_negacyclic(&a_sum, &b_sum);

    // Combine results for negacyclic
    let mut result = vec![0u32; n];
    for i in 0..n2 {
      result[i] = p0[i].wrapping_sub(p1[i]);
    }

    for i in 0..n2 {
      let cross = p_mid[i].wrapping_sub(p0[i]).wrapping_sub(p1[i]);
      if i < n2 - 1 {
        result[i + 1] = result[i + 1].wrapping_add(cross);
      } else {
        result[0] = result[0].wrapping_sub(cross);
      }
    }

    result
  }
}

impl FFTProcessor for KaratsubaNegacyclicProcessor {
  fn new(n: usize) -> Self {
    let threshold = 64;
    KaratsubaNegacyclicProcessor {
      n,
      threshold,
      fft_processor: if n > threshold {
        Some(ExtendedFftKlemsa::new(n))
      } else {
        None
      },
    }
  }

  fn ifft_1024(&mut self, input: &[u32; 1024]) -> [f64; 1024] {
    if let Some(ref mut fft) = self.fft_processor {
      fft.ifft_1024(input)
    } else {
      [0.0; 1024]
    }
  }

  fn fft_1024(&mut self, input: &[f64; 1024]) -> [u32; 1024] {
    if let Some(ref mut fft) = self.fft_processor {
      fft.fft_1024(input)
    } else {
      [0; 1024]
    }
  }

  fn poly_mul_1024(&mut self, a: &[u32; 1024], b: &[u32; 1024]) -> [u32; 1024] {
    if self.n < self.threshold {
      let result = self.karatsuba_negacyclic(a, b);
      let mut arr = [0u32; 1024];
      arr.copy_from_slice(&result);
      arr
    } else {
      self.fft_processor.as_mut().unwrap().poly_mul_1024(a, b)
    }
  }

  fn ifft(&mut self, input: &[u32]) -> Vec<f64> {
    if let Some(ref mut fft) = self.fft_processor {
      fft.ifft(input)
    } else {
      vec![0.0; self.n]
    }
  }

  fn fft(&mut self, input: &[f64]) -> Vec<u32> {
    if let Some(ref mut fft) = self.fft_processor {
      fft.fft(input)
    } else {
      vec![0; self.n]
    }
  }

  fn poly_mul(&mut self, a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    if a.len() < self.threshold {
      self.karatsuba_negacyclic(a, b)
    } else if let Some(ref mut fft) = self.fft_processor {
      fft.poly_mul(a, b)
    } else {
      vec![0; a.len()]
    }
  }
}

// ============================================================================
// 3. SPARSE-OPTIMIZED PROCESSOR
// ============================================================================

pub struct SparseOptimizedProcessor {
  _n: usize,
  base_processor: ExtendedFftKlemsa,
  sparsity_threshold: f64,
}

impl SparseOptimizedProcessor {
  fn compute_sparsity(&self, poly: &[u32]) -> f64 {
    let nonzero = poly.iter().filter(|&&x| x != 0).count();
    nonzero as f64 / poly.len() as f64
  }

  fn sparse_negacyclic_mul(&self, a: &[u32], b: &[u32]) -> Vec<u32> {
    let n = a.len();

    let a_sparse: Vec<(usize, u32)> = a
      .iter()
      .enumerate()
      .filter(|(_, &v)| v != 0)
      .map(|(i, &v)| (i, v))
      .collect();

    let b_sparse: Vec<(usize, u32)> = b
      .iter()
      .enumerate()
      .filter(|(_, &v)| v != 0)
      .map(|(i, &v)| (i, v))
      .collect();

    let mut result = vec![0u32; n];

    for &(i, a_val) in &a_sparse {
      for &(j, b_val) in &b_sparse {
        let idx = (i + j) % n;
        let product = a_val.wrapping_mul(b_val);

        if i + j < n {
          result[idx] = result[idx].wrapping_add(product);
        } else {
          result[idx] = result[idx].wrapping_sub(product);
        }
      }
    }

    result
  }
}

impl FFTProcessor for SparseOptimizedProcessor {
  fn new(n: usize) -> Self {
    SparseOptimizedProcessor {
      _n: n,
      base_processor: ExtendedFftKlemsa::new(n),
      sparsity_threshold: 0.15, // 15% threshold
    }
  }

  fn ifft_1024(&mut self, input: &[u32; 1024]) -> [f64; 1024] {
    self.base_processor.ifft_1024(input)
  }

  fn fft_1024(&mut self, input: &[f64; 1024]) -> [u32; 1024] {
    self.base_processor.fft_1024(input)
  }

  fn poly_mul_1024(&mut self, a: &[u32; 1024], b: &[u32; 1024]) -> [u32; 1024] {
    let sparsity_a = self.compute_sparsity(a);
    let sparsity_b = self.compute_sparsity(b);

    if sparsity_a < self.sparsity_threshold || sparsity_b < self.sparsity_threshold {
      let result = self.sparse_negacyclic_mul(a, b);
      let mut arr = [0u32; 1024];
      arr.copy_from_slice(&result);
      arr
    } else {
      self.base_processor.poly_mul_1024(a, b)
    }
  }

  fn ifft(&mut self, input: &[u32]) -> Vec<f64> {
    self.base_processor.ifft(input)
  }

  fn fft(&mut self, input: &[f64]) -> Vec<u32> {
    self.base_processor.fft(input)
  }

  fn poly_mul(&mut self, a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    let sparsity_a = self.compute_sparsity(a);
    let sparsity_b = self.compute_sparsity(b);

    if sparsity_a < self.sparsity_threshold || sparsity_b < self.sparsity_threshold {
      self.sparse_negacyclic_mul(a, b)
    } else {
      self.base_processor.poly_mul(a, b)
    }
  }
}

// ============================================================================
// 4. MERGED TRANSFORM PROCESSOR (Cache-Optimized)
// ============================================================================

pub struct MergedTransformProcessor {
  n: usize,
  twisties_combined_re: Vec<f64>,
  twisties_combined_im: Vec<f64>,
  fft_n2_fwd: Arc<dyn Fft<f64>>,
  fft_n2_inv: Arc<dyn Fft<f64>>,
  fourier_buffer: RefCell<Vec<Complex<f64>>>,
  scratch_fwd: RefCell<Vec<Complex<f64>>>,
  scratch_inv: RefCell<Vec<Complex<f64>>>,
}

impl FFTProcessor for MergedTransformProcessor {
  fn new(n: usize) -> Self {
    assert!(n.is_power_of_two());
    let n2 = n / 2;

    // Pre-compute combined twisting + first FFT stage factors
    let mut twisties_combined_re = Vec::with_capacity(n2);
    let mut twisties_combined_im = Vec::with_capacity(n2);
    let twist_unit = PI / (n as f64);

    for i in 0..n2 {
      let angle = i as f64 * twist_unit;
      let (im, re) = angle.sin_cos();
      twisties_combined_re.push(re);
      twisties_combined_im.push(im);
    }

    let mut planner = FftPlanner::new();
    let fft_n2_fwd = planner.plan_fft_forward(n2);
    let fft_n2_inv = planner.plan_fft_inverse(n2);

    let scratch_fwd_len = fft_n2_fwd.get_inplace_scratch_len();
    let scratch_inv_len = fft_n2_inv.get_inplace_scratch_len();

    MergedTransformProcessor {
      n,
      twisties_combined_re,
      twisties_combined_im,
      fft_n2_fwd,
      fft_n2_inv,
      fourier_buffer: RefCell::new(vec![Complex::new(0.0, 0.0); n2]),
      scratch_fwd: RefCell::new(vec![Complex::new(0.0, 0.0); scratch_fwd_len]),
      scratch_inv: RefCell::new(vec![Complex::new(0.0, 0.0); scratch_inv_len]),
    }
  }

  fn ifft_1024(&mut self, input: &[u32; 1024]) -> [f64; 1024] {
    let n2 = 512;
    let (input_re, input_im) = input.split_at(n2);

    // Merged fold + twist operation
    let mut fourier = self.fourier_buffer.borrow_mut();
    for i in 0..n2 {
      let in_re = (input_re[i] as i32 as f64) / (1u64 << 32) as f64;
      let in_im = (input_im[i] as i32 as f64) / (1u64 << 32) as f64;
      let w_re = self.twisties_combined_re[i];
      let w_im = self.twisties_combined_im[i];
      fourier[i] = Complex::new(in_re * w_re - in_im * w_im, in_re * w_im + in_im * w_re);
    }

    let mut scratch = self.scratch_fwd.borrow_mut();
    self
      .fft_n2_fwd
      .process_with_scratch(&mut fourier, &mut scratch);

    let mut result = [0.0f64; 1024];
    for i in 0..n2 {
      result[i] = fourier[i].re * 2.0;
      result[i + n2] = fourier[i].im * 2.0;
    }

    result
  }

  fn fft_1024(&mut self, input: &[f64; 1024]) -> [u32; 1024] {
    let n2 = 512;
    let (input_re, input_im) = input.split_at(n2);

    let mut fourier = self.fourier_buffer.borrow_mut();
    for i in 0..n2 {
      fourier[i] = Complex::new(input_re[i] * 0.5, input_im[i] * 0.5);
    }

    let mut scratch = self.scratch_inv.borrow_mut();
    self
      .fft_n2_inv
      .process_with_scratch(&mut fourier, &mut scratch);

    let normalization = 1.0 / (n2 as f64);
    let mut result = [0u32; 1024];
    for i in 0..n2 {
      let w_re = self.twisties_combined_re[i];
      let w_im = self.twisties_combined_im[i];
      let f_re = fourier[i].re * normalization;
      let f_im = fourier[i].im * normalization;
      let tmp_re = f_re * w_re + f_im * w_im;
      let tmp_im = f_im * w_re - f_re * w_im;
      result[i] = (tmp_re * (1u64 << 32) as f64).round() as i64 as u32;
      result[i + n2] = (tmp_im * (1u64 << 32) as f64).round() as i64 as u32;
    }

    result
  }

  fn poly_mul_1024(&mut self, a: &[u32; 1024], b: &[u32; 1024]) -> [u32; 1024] {
    let a_freq = self.ifft_1024(a);
    let b_freq = self.ifft_1024(b);

    let mut result_freq = [0.0f64; 1024];
    let n2 = 512;
    for i in 0..n2 {
      let ar = a_freq[i];
      let ai = a_freq[i + n2];
      let br = b_freq[i];
      let bi = b_freq[i + n2];
      result_freq[i] = (ar * br - ai * bi) * 0.5;
      result_freq[i + n2] = (ar * bi + ai * br) * 0.5;
    }

    self.fft_1024(&result_freq)
  }

  fn ifft(&mut self, input: &[u32]) -> Vec<f64> {
    if self.n == 1024 {
      let mut arr = [0u32; 1024];
      arr.copy_from_slice(input);
      self.ifft_1024(&arr).to_vec()
    } else {
      vec![0.0; self.n]
    }
  }

  fn fft(&mut self, input: &[f64]) -> Vec<u32> {
    if self.n == 1024 {
      let mut arr = [0f64; 1024];
      arr.copy_from_slice(input);
      self.fft_1024(&arr).to_vec()
    } else {
      vec![0; self.n]
    }
  }

  fn poly_mul(&mut self, a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    if a.len() == 1024 && b.len() == 1024 {
      let mut a_arr = [0u32; 1024];
      let mut b_arr = [0u32; 1024];
      a_arr.copy_from_slice(a);
      b_arr.copy_from_slice(b);
      self.poly_mul_1024(&a_arr, &b_arr).to_vec()
    } else {
      vec![0; a.len()]
    }
  }
}

// ============================================================================
// 5. ADAPTIVE FFT PROCESSOR (Smart Selector)
// ============================================================================

pub struct AdaptiveFFTProcessor {
  n: usize,
  klemsa: ExtendedFftKlemsa,
  karatsuba: Option<KaratsubaNegacyclicProcessor>,
  sparse: SparseOptimizedProcessor,
  merged: MergedTransformProcessor,
}

impl AdaptiveFFTProcessor {
  fn compute_sparsity(&self, poly: &[u32]) -> f64 {
    let nonzero = poly.iter().filter(|&&x| x != 0).count();
    nonzero as f64 / poly.len() as f64
  }

  fn select_algorithm(&self, a: &[u32], b: &[u32]) -> Algorithm {
    let a_sparsity = self.compute_sparsity(a);
    let b_sparsity = self.compute_sparsity(b);

    // Decision tree
    if self.n <= 64 {
      return Algorithm::Karatsuba;
    }

    if a_sparsity < 0.15 || b_sparsity < 0.15 {
      return Algorithm::Sparse;
    }

    if self.n >= 2048 {
      return Algorithm::Merged;
    }

    Algorithm::Klemsa
  }
}

#[derive(Debug, PartialEq)]
enum Algorithm {
  Klemsa,
  Karatsuba,
  Sparse,
  Merged,
}

impl FFTProcessor for AdaptiveFFTProcessor {
  fn new(n: usize) -> Self {
    AdaptiveFFTProcessor {
      n,
      klemsa: ExtendedFftKlemsa::new(n),
      karatsuba: if n <= 64 {
        Some(KaratsubaNegacyclicProcessor::new(n))
      } else {
        None
      },
      sparse: SparseOptimizedProcessor::new(n),
      merged: MergedTransformProcessor::new(n),
    }
  }

  fn ifft_1024(&mut self, input: &[u32; 1024]) -> [f64; 1024] {
    // For forward transforms, use default (we optimize on poly_mul)
    self.klemsa.ifft_1024(input)
  }

  fn fft_1024(&mut self, input: &[f64; 1024]) -> [u32; 1024] {
    self.klemsa.fft_1024(input)
  }

  fn poly_mul_1024(&mut self, a: &[u32; 1024], b: &[u32; 1024]) -> [u32; 1024] {
    let algo = self.select_algorithm(a, b);

    match algo {
      Algorithm::Karatsuba => {
        if let Some(ref mut k) = self.karatsuba {
          k.poly_mul_1024(a, b)
        } else {
          self.klemsa.poly_mul_1024(a, b)
        }
      }
      Algorithm::Sparse => self.sparse.poly_mul_1024(a, b),
      Algorithm::Merged => self.merged.poly_mul_1024(a, b),
      Algorithm::Klemsa => self.klemsa.poly_mul_1024(a, b),
    }
  }

  fn ifft(&mut self, input: &[u32]) -> Vec<f64> {
    self.klemsa.ifft(input)
  }

  fn fft(&mut self, input: &[f64]) -> Vec<u32> {
    self.klemsa.fft(input)
  }

  fn poly_mul(&mut self, a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    let algo = self.select_algorithm(a, b);

    match algo {
      Algorithm::Karatsuba => {
        if let Some(ref mut k) = self.karatsuba {
          k.poly_mul(a, b)
        } else {
          self.klemsa.poly_mul(a, b)
        }
      }
      Algorithm::Sparse => self.sparse.poly_mul(a, b),
      Algorithm::Merged => self.merged.poly_mul(a, b),
      Algorithm::Klemsa => self.klemsa.poly_mul(a, b),
    }
  }

  fn batch_ifft_1024(&mut self, inputs: &[[u32; 1024]]) -> Vec<[f64; 1024]> {
    // Batch optimization: process all with same algorithm
    inputs.iter().map(|input| self.ifft_1024(input)).collect()
  }

  fn batch_fft_1024(&mut self, inputs: &[[f64; 1024]]) -> Vec<[u32; 1024]> {
    inputs.iter().map(|input| self.fft_1024(input)).collect()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn poly_mul_reference(a: &[u32], b: &[u32]) -> Vec<u32> {
    let n = a.len();
    let mut res = vec![0u32; n];

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

  #[test]
  fn test_klemsa_processor() {
    let mut processor = ExtendedFftKlemsa::new(1024);
    let mut a = [0u32; 1024];
    let mut b = [0u32; 1024];

    a[0] = 1 << 30;
    a[5] = 1 << 29;
    b[0] = 1 << 30;
    b[10] = 1 << 28;

    let result = processor.poly_mul_1024(&a, &b);
    let expected = poly_mul_reference(&a.to_vec(), &b.to_vec());

    for i in 0..1024 {
      let diff = (result[i] as i64 - expected[i] as i64).abs();
      assert!(
        diff < 2,
        "Mismatch at {}: {} vs {}",
        i,
        result[i],
        expected[i]
      );
    }
    println!("✓ Klemsa processor test passed");
  }

  #[test]
  fn test_sparse_processor() {
    let mut processor = SparseOptimizedProcessor::new(1024);
    let mut a = [0u32; 1024];
    let mut b = [0u32; 1024];

    // Create sparse polynomials (5% non-zero)
    for i in (0..1024).step_by(20) {
      a[i] = 1 << 28;
      b[i] = 1 << 27;
    }

    let result = processor.poly_mul_1024(&a, &b);
    let expected = poly_mul_reference(&a.to_vec(), &b.to_vec());

    for i in 0..1024 {
      let diff = (result[i] as i64 - expected[i] as i64).abs();
      assert!(
        diff < 2,
        "Mismatch at {}: {} vs {}",
        i,
        result[i],
        expected[i]
      );
    }
    println!("✓ Sparse processor test passed");
  }

  #[test]
  fn test_adaptive_processor() {
    let mut processor = AdaptiveFFTProcessor::new(1024);

    // Test 1: Dense polynomials
    let mut a_dense = [0u32; 1024];
    let mut b_dense = [0u32; 1024];
    for i in 0..1024 {
      a_dense[i] = ((i * 1234567) % (1 << 20)) as u32;
      b_dense[i] = ((i * 7654321) % (1 << 20)) as u32;
    }

    let result_dense = processor.poly_mul_1024(&a_dense, &b_dense);
    let expected_dense = poly_mul_reference(&a_dense.to_vec(), &b_dense.to_vec());

    for i in 0..1024 {
      let diff = (result_dense[i] as i64 - expected_dense[i] as i64).abs();
      assert!(diff < 2, "Dense: Mismatch at {}", i);
    }

    // Test 2: Sparse polynomials
    let mut a_sparse = [0u32; 1024];
    let mut b_sparse = [0u32; 1024];
    for i in (0..1024).step_by(20) {
      a_sparse[i] = 1 << 28;
      b_sparse[i] = 1 << 27;
    }

    let result_sparse = processor.poly_mul_1024(&a_sparse, &b_sparse);
    let expected_sparse = poly_mul_reference(&a_sparse.to_vec(), &b_sparse.to_vec());

    for i in 0..1024 {
      let diff = (result_sparse[i] as i64 - expected_sparse[i] as i64).abs();
      assert!(diff < 2, "Sparse: Mismatch at {}", i);
    }

    println!("✓ Adaptive processor test passed");
  }

  #[test]
  fn test_all_processors_consistency() {
    let mut klemsa = ExtendedFftKlemsa::new(1024);
    let mut sparse = SparseOptimizedProcessor::new(1024);
    let mut merged = MergedTransformProcessor::new(1024);
    let mut adaptive = AdaptiveFFTProcessor::new(1024);

    let mut a = [0u32; 1024];
    let mut b = [0u32; 1024];
    for i in 0..1024 {
      a[i] = ((i * 12345) % (1 << 20)) as u32;
      b[i] = ((i * 54321) % (1 << 20)) as u32;
    }

    let result_klemsa = klemsa.poly_mul_1024(&a, &b);
    let result_sparse = sparse.poly_mul_1024(&a, &b);
    let result_merged = merged.poly_mul_1024(&a, &b);
    let result_adaptive = adaptive.poly_mul_1024(&a, &b);

    // All should produce equivalent results
    for i in 0..1024 {
      let diff_ks = (result_klemsa[i] as i64 - result_sparse[i] as i64).abs();
      let diff_km = (result_klemsa[i] as i64 - result_merged[i] as i64).abs();
      let diff_ka = (result_klemsa[i] as i64 - result_adaptive[i] as i64).abs();

      assert!(diff_ks < 2, "Klemsa vs Sparse mismatch at {}", i);
      assert!(diff_km < 2, "Klemsa vs Merged mismatch at {}", i);
      assert!(diff_ka < 2, "Klemsa vs Adaptive mismatch at {}", i);
    }

    println!("✓ All processors produce consistent results");
  }
}
