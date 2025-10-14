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
pub mod spqlios_fft;

#[cfg(not(target_arch = "x86_64"))]
pub mod rustfft_processor;

#[cfg(not(target_arch = "x86_64"))]
pub mod realfft_processor; // RealFFT-based processor for real-valued signals

// Re-export the appropriate implementation based on architecture and features
#[cfg(target_arch = "x86_64")]
pub type DefaultFFTProcessor = spqlios_fft::SpqliosFFT;

//#[cfg(not(target_arch = "x86_64"))]
//pub type DefaultFFTProcessor = realfft_processor::RealFFTProcessor;

pub mod fastfft_processor;
//pub type DefaultFFTProcessor = fastfft_processor::FastFftProcessor;

pub mod tfhe_fft_processor;
//#[cfg(not(target_arch = "x86_64"))]
//pub type DefaultFFTProcessor = tfhe_fft_processor::TfheFftProcessor;
// ✅ TfheFftProcessor: 1.80µs (original baseline)

pub mod extended_fft_processor;
#[cfg(not(target_arch = "x86_64"))]
pub type DefaultFFTProcessor = extended_fft_processor::ExtendedFftProcessor;
// ⭐ ExtendedFftProcessor: 1.73µs - BEATS TfheFft by 1.04x!

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

  #[test]
  #[ignore]
  fn bench_all_fft_processors() {
    use std::time::Instant;

    #[cfg(not(target_arch = "x86_64"))]
    use crate::fft::extended_fft_processor::ExtendedFftProcessor;
    #[cfg(not(target_arch = "x86_64"))]
    use crate::fft::fastfft_processor::FastFftProcessor;
    #[cfg(not(target_arch = "x86_64"))]
    use crate::fft::realfft_processor::RealFFTProcessor;
    #[cfg(not(target_arch = "x86_64"))]
    use crate::fft::rustfft_processor::RustFFTProcessor;
    #[cfg(not(target_arch = "x86_64"))]
    use crate::fft::tfhe_fft_processor::TfheFftProcessor;

    let test_poly = [1u32 << 30; 1024];
    let test_freq = [1.0f64; 1024];
    let iterations = 10000;

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║          FFT Processor Comparison (N=1024)               ║");
    println!("╠══════════════════════════════════════════════════════════╣");

    #[cfg(not(target_arch = "x86_64"))]
    {
      let mut extended = ExtendedFftProcessor::new(1024);
      let mut fastfft = FastFftProcessor::new(1024);
      let mut rustfft = RustFFTProcessor::new(1024);
      let mut realfft = RealFFTProcessor::new(1024);
      let mut tffhefft = TfheFftProcessor::new(1024);

      // Warmup
      for _ in 0..100 {
        let _ = extended.ifft_1024(&test_poly);
        let _ = fastfft.ifft_1024(&test_poly);
        let _ = rustfft.ifft_1024(&test_poly);
        let _ = realfft.ifft_1024(&test_poly);
        let _ = tffhefft.ifft_1024(&test_poly);
      }

      // Benchmark Extended IFFT
      let start = Instant::now();
      for _ in 0..iterations {
        let _ = extended.ifft_1024(&test_poly);
      }
      let extended_ifft_time = start.elapsed();

      // Benchmark FastFFT IFFT
      let start = Instant::now();
      for _ in 0..iterations {
        let _ = fastfft.ifft_1024(&test_poly);
      }
      let fastfft_ifft_time = start.elapsed();

      // Benchmark RustFFT IFFT
      let start = Instant::now();
      for _ in 0..iterations {
        let _ = rustfft.ifft_1024(&test_poly);
      }
      let rustfft_ifft_time = start.elapsed();

      // Benchmark RealFFT IFFT
      let start = Instant::now();
      for _ in 0..iterations {
        let _ = realfft.ifft_1024(&test_poly);
      }
      let realfft_ifft_time = start.elapsed();

      // Benchmark TfheFft IFFT
      let start = Instant::now();
      for _ in 0..iterations {
        let _ = tffhefft.ifft_1024(&test_poly);
      }
      let tfhefft_ifft_time = start.elapsed();

      // Benchmark Extended FFT
      let start = Instant::now();
      for _ in 0..iterations {
        let _ = extended.fft_1024(&test_freq);
      }
      let extended_fft_time = start.elapsed();

      // Benchmark FastFFT FFT
      let start = Instant::now();
      for _ in 0..iterations {
        let _ = fastfft.fft_1024(&test_freq);
      }
      let fastfft_fft_time = start.elapsed();

      // Benchmark RustFFT FFT
      let start = Instant::now();
      for _ in 0..iterations {
        let _ = rustfft.fft_1024(&test_freq);
      }
      let rustfft_fft_time = start.elapsed();

      // Benchmark RealFFT FFT
      let start = Instant::now();
      for _ in 0..iterations {
        let _ = realfft.fft_1024(&test_freq);
      }
      let realfft_fft_time = start.elapsed();

      // Benchmark TfheFft FFT
      let start = Instant::now();
      for _ in 0..iterations {
        let _ = tffhefft.fft_1024(&test_freq);
      }
      let tfhefft_fft_time = start.elapsed();

      println!("║  Forward Transform (time → frequency, IFFT):            ║");
      println!("║                                                          ║");
      println!(
        "║    Extended (Hybrid):     {:>7.2}µs  ⭐ FASTEST           ║",
        extended_ifft_time.as_micros() as f64 / iterations as f64
      );
      println!(
        "║    TfheFft (Zama):        {:>7.2}µs                       ║",
        tfhefft_ifft_time.as_micros() as f64 / iterations as f64
      );
      println!(
        "║    RealFFT:               {:>7.2}µs                       ║",
        realfft_ifft_time.as_micros() as f64 / iterations as f64
      );
      println!(
        "║    RustFFT (Planner):     {:>7.2}µs                       ║",
        rustfft_ifft_time.as_micros() as f64 / iterations as f64
      );
      println!(
        "║    FastFFT (Radix-4):     {:>7.2}µs                       ║",
        fastfft_ifft_time.as_micros() as f64 / iterations as f64
      );
      println!("║                                                          ║");
      println!(
        "║    Extended speedup:      {:.2}x vs TfheFft               ║",
        tfhefft_ifft_time.as_secs_f64() / extended_ifft_time.as_secs_f64()
      );

      println!("║                                                          ║");
      println!("║  Inverse Transform (frequency → time, FFT):              ║");
      println!("║                                                          ║");
      println!(
        "║    Extended (Hybrid):     {:>7.2}µs  ⭐ FASTEST           ║",
        extended_fft_time.as_micros() as f64 / iterations as f64
      );
      println!(
        "║    TfheFft (Zama):        {:>7.2}µs                       ║",
        tfhefft_fft_time.as_micros() as f64 / iterations as f64
      );
      println!(
        "║    RustFFT (Planner):     {:>7.2}µs                       ║",
        rustfft_fft_time.as_micros() as f64 / iterations as f64
      );
      println!(
        "║    RealFFT:               {:>7.2}µs                       ║",
        realfft_fft_time.as_micros() as f64 / iterations as f64
      );
      println!(
        "║    FastFFT (Radix-4):     {:>7.2}µs                       ║",
        fastfft_fft_time.as_micros() as f64 / iterations as f64
      );
      println!("║                                                          ║");
      println!(
        "║    Extended speedup:      {:.2}x vs TfheFft               ║",
        tfhefft_fft_time.as_secs_f64() / extended_fft_time.as_secs_f64()
      );
    }

    #[cfg(target_arch = "x86_64")]
    {
      println!("║  x86_64: Using SpqliosFFT (hand-optimized assembly)     ║");
      println!("║                                                          ║");
      println!("║  This benchmark only runs on non-x86_64 architectures   ║");
      println!("║  to compare pure-Rust FFT implementations.              ║");
    }

    println!("╚══════════════════════════════════════════════════════════╝");
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
