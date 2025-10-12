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
}

// Platform-specific implementations
#[cfg(target_arch = "x86_64")]
pub mod spqlios_fft;

#[cfg(not(target_arch = "x86_64"))]
pub mod rustfft_processor;

// Re-export the appropriate implementation based on architecture
#[cfg(target_arch = "x86_64")]
pub type DefaultFFTProcessor = spqlios_fft::SpqliosFFT;

#[cfg(not(target_arch = "x86_64"))]
pub type DefaultFFTProcessor = rustfft_processor::RustFFTProcessor;
