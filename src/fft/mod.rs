//! FFT Processor Module for TFHE
//!
//! This module provides abstracted FFT operations for negacyclic polynomial
//! multiplication in the ring R[X]/(X^N+1), which is fundamental to TFHE.
//!
//! # Architecture
//!
//! The module uses a trait-based design with platform-specific implementations:
//!
//! - **x86_64**: `SpqliosFFT` - Hand-optimized AVX/FMA assembly (~30ms/gate)
//! - **ARM64/Other**: `RustFFTProcessor` - Pure Rust implementation (~100ms/gate)
//!
//! Both implementations are mathematically equivalent and pass identical test suites.
//!
//! # Usage
//!
//! ```ignore
//! use crate::fft::{DefaultFFTProcessor, FFTProcessor};
//!
//! let mut processor = DefaultFFTProcessor::new(1024);
//! let freq = processor.ifft_1024(&time_domain);
//! let result = processor.fft_1024(&freq);
//! ```
//!
//! # Algorithm
//!
//! The negacyclic FFT embeds an N-point negacyclic problem into a 2N-point
//! cyclic FFT, extracting only the odd frequency indices which correspond
//! to the primitive 2N-th roots of unity needed for polynomial multiplication
//! modulo X^N+1.

/// FFT Processor trait for negacyclic polynomial multiplication in R[X]/(X^N+1)
///
/// All implementations must provide mathematically equivalent operations
/// for TFHE's core polynomial arithmetic.
pub trait FFTProcessor {
  /// Create a new FFT processor for polynomials of size n
  fn new(n: usize) -> Self;

  /// Forward FFT: time domain (N values) → frequency domain (N values)
  /// Input: N torus32 values representing polynomial coefficients
  /// Output: N f64 values (N/2 complex stored as [re_0..re_N/2-1, im_0..im_N/2-1])
  fn ifft(&mut self, input: &[u32]) -> Vec<f64>;

  /// Inverse FFT: frequency domain (N values) → time domain (N values)
  /// Input: N f64 values (N/2 complex stored as [re_0..re_N/2-1, im_0..im_N/2-1])
  /// Output: N torus32 values representing polynomial coefficients
  fn fft(&mut self, input: &[f64]) -> Vec<u32>;

  /// Forward FFT for fixed-size 1024 arrays (optimized version)
  fn ifft_1024(&mut self, input: &[u32; 1024]) -> [f64; 1024];

  /// Inverse FFT for fixed-size 1024 arrays (optimized version)
  fn fft_1024(&mut self, input: &[f64; 1024]) -> [u32; 1024];

  /// Negacyclic polynomial multiplication: a(X) * b(X) mod (X^N+1)
  /// Uses FFT for O(N log N) complexity instead of O(N²)
  fn poly_mul_1024(&mut self, a: &[u32; 1024], b: &[u32; 1024]) -> [u32; 1024];

  /// Negacyclic polynomial multiplication for variable-length vectors
  /// Fallback to poly_mul_1024 for 1024-sized inputs, otherwise uses Vec variants
  fn poly_mul(&mut self, a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32>;

  /// Batch IFFT: Transform multiple polynomials at once
  /// This can be more efficient than calling ifft_1024 multiple times
  /// Input: slice of N polynomials (each 1024 elements)
  /// Output: Vec of N frequency-domain representations
  fn batch_ifft_1024(&mut self, inputs: &[[u32; 1024]]) -> Vec<[f64; 1024]> {
    // Default implementation: just loop (subclasses can optimize)
    inputs.iter().map(|input| self.ifft_1024(input)).collect()
  }

  /// Batch FFT: Transform multiple frequency-domain representations at once
  /// Input: slice of N frequency-domain arrays (each 1024 elements)
  /// Output: Vec of N time-domain polynomials
  fn batch_fft_1024(&mut self, inputs: &[[f64; 1024]]) -> Vec<[u32; 1024]> {
    // Default implementation: just loop (subclasses can optimize)
    inputs.iter().map(|input| self.fft_1024(input)).collect()
  }
}

// Platform-specific implementations
#[cfg(target_arch = "x86_64")]
mod spqlios;

// Re-export the appropriate implementation based on architecture and features
#[cfg(target_arch = "x86_64")]
pub type DefaultFFTProcessor = spqlios::SpqliosFFT;

pub mod extended_fft_processor;
#[cfg(not(target_arch = "x86_64"))]
pub type DefaultFFTProcessor = extended_fft_processor::ExtendedFftProcessor;

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

use crate::params;
use std::cell::RefCell;

thread_local!(pub static FFT_PLAN: RefCell<FFTPlan> = RefCell::new(FFTPlan::new(params::trgsw_lv1::N)));

// Future implementation ideas:
//
// pub mod cuda_fft;         // CUDA/cuFFT implementation (~10-20ms per gate, 50-100x batch)
// pub mod metal_fft;        // Apple Metal GPU implementation (for iOS/macOS GPU)
// pub mod wasm_fft;         // WebAssembly optimized version
// pub mod neon_asm_fft;     // Hand-coded ARM NEON assembly (could match x86_64)
//
// Performance estimates:
// - CUDA (cuFFT):        ~15ms per gate, 5-7x faster than ARM64
// - CUDA (custom):       ~10ms per gate, 10x faster than ARM64, batch 50-100x
// - Metal:               ~30-50ms per gate (GPU overhead on integrated GPUs)
// - Hand-coded NEON asm: ~35-50ms per gate (could approach x86_64 performance)

#[cfg(test)]
mod tests {
  use crate::fft::FFTPlan;
  use crate::fft::FFTProcessor;
  use crate::params;
  use rand::Rng;

  #[test]
  fn test_fft_ifft() {
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
  fn test_fft_poly_mul() {
    let n = 1024;
    let mut plan = FFTPlan::new(n);
    let mut rng = rand::thread_rng();
    let mut a: Vec<u32> = vec![0u32; n];
    let mut b: Vec<u32> = vec![0u32; n];
    a.iter_mut().for_each(|e| *e = rng.gen::<u32>());
    b.iter_mut()
      .for_each(|e| *e = rng.gen::<u32>() % params::trgsw_lv1::BG as u32);

    let fft_res = plan.processor.poly_mul(&a, &b);
    let res = poly_mul(&a.to_vec(), &b.to_vec());
    for i in 0..n {
      let diff = res[i] as i64 - fft_res[i] as i64;
      assert!(
        diff < 2 && diff > -2,
        "Failed at index {}: expected={}, got={}, diff={}",
        i,
        res[i],
        fft_res[i],
        diff
      );
    }
  }

  #[test]
  fn test_fft_simple() {
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
  fn test_fft_ifft_1024() {
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
  fn test_fft_poly_mul_1024() {
    let mut plan = FFTPlan::new(1024);
    let mut rng = rand::thread_rng();
    for _i in 0..100 {
      let mut a = [0u32; 1024];
      let mut b = [0u32; 1024];
      a.iter_mut().for_each(|e| *e = rng.gen::<u32>());
      b.iter_mut()
        .for_each(|e| *e = rng.gen::<u32>() % params::trgsw_lv1::BG as u32);

      let fft_res = plan.processor.poly_mul_1024(&a, &b);
      let res = poly_mul(&a.to_vec(), &b.to_vec());
      for i in 0..1024 {
        let diff = res[i] as i64 - fft_res[i] as i64;
        assert!(
          diff < 2 && diff > -2,
          "Failed at index {}: expected={}, got={}, diff={}",
          i,
          res[i],
          fft_res[i],
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

#[cfg(target_arch = "x86_64")]
#[cfg(test)]
mod tests {
  use crate::fft::FFTPlan;
  use crate::fft::FFTProcessor;
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
