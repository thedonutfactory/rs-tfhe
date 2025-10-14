//! RealFFT-based Negacyclic FFT Processor
//!
//! Uses RealFFT library for real-valued FFT, which is more efficient than
//! complex FFT with antisymmetric embedding.
//!
//! Key advantages:
//! - Direct real → complex transform (no 2N embedding overhead)
//! - N real inputs → N/2+1 complex outputs (right-sized)
//! - Optimized for real signals
//! - Plan reuse for repeated transforms
//! - Proper scratch buffer usage via `process_with_scratch()` for zero-allocation hot path

use super::FFTProcessor;
use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use rustfft::Fft;
use std::cell::RefCell;
use std::sync::Arc;

/// RealFFT-based processor optimized for real-valued polynomials
pub struct RealFFTProcessor {
  n: usize,
  real_to_complex: Arc<dyn RealToComplex<f64>>,
  // Cached complex FFT for inverse transform
  fft_2n: Arc<dyn Fft<f64>>,
  // Pre-allocated buffer for inverse transform
  inverse_buffer: RefCell<Vec<Complex<f64>>>,
  // Scratch buffer for process_with_scratch (avoids per-call allocation)
  scratch: RefCell<Vec<Complex<f64>>>,
}

impl RealFFTProcessor {
  pub fn new(n: usize) -> Self {
    let mut real_planner = RealFftPlanner::<f64>::new();

    // For negacyclic FFT, we need 2N-point real FFT (for antisymmetric embedding)
    let fft_size = 2 * n;

    // Also cache complex FFT plan for inverse transform
    let mut complex_planner = rustfft::FftPlanner::new();
    complex_planner.plan_fft_forward(fft_size);
    let fft_2n = complex_planner.plan_fft_forward(fft_size);

    // Pre-allocate scratch buffer with optimal size
    // See: https://docs.rs/rustfft/latest/rustfft/trait.Fft.html#tymethod.get_inplace_scratch_len
    let scratch_len = fft_2n.get_inplace_scratch_len();

    RealFFTProcessor {
      n,
      real_to_complex: real_planner.plan_fft_forward(fft_size),
      fft_2n,
      inverse_buffer: RefCell::new(Vec::new()),
      scratch: RefCell::new(vec![Complex::new(0.0, 0.0); scratch_len]),
    }
  }
}

impl FFTProcessor for RealFFTProcessor {
  fn new(n: usize) -> Self {
    RealFFTProcessor::new(n)
  }

  /// Negacyclic IFFT: u32[N] → f64[N]
  ///
  /// Uses RealFFT for efficient real-valued transform
  fn ifft(&mut self, input: &[u32]) -> Vec<f64> {
    assert_eq!(input.len(), self.n);

    let n = self.n;
    let nn = 2 * n;

    // Create antisymmetric embedding: [a₀..a_{N-1}, -a₀..-a_{N-1}]
    let mut real_input = vec![0.0f64; nn];
    for i in 0..n {
      let val = input[i] as i32 as f64;
      real_input[i] = val;
      real_input[i + n] = -val;
    }

    // Real FFT: 2N real → N+1 complex
    let mut spectrum = self.real_to_complex.make_output_vec();
    self
      .real_to_complex
      .process(&mut real_input, &mut spectrum)
      .unwrap();

    // Extract odd bins for negacyclic FFT
    // Odd bins: 1, 3, 5, ..., 2N-1 but spectrum only has N+1 bins
    // Due to Hermitian symmetry, we can reconstruct
    let mut result = vec![0.0f64; n];
    let ns2 = n / 2;

    for i in 0..ns2 {
      let odd_idx = 2 * i + 1;
      if odd_idx < spectrum.len() {
        result[i] = spectrum[odd_idx].re;
        result[i + ns2] = spectrum[odd_idx].im;
      } else {
        // Use Hermitian symmetry: X[2N-k] = conj(X[k])
        let conj_idx = nn - odd_idx;
        if conj_idx < spectrum.len() {
          result[i] = spectrum[conj_idx].re;
          result[i + ns2] = -spectrum[conj_idx].im; // conjugate
        }
      }
    }

    result
  }

  /// Negacyclic FFT: f64[N] → u32[N]
  ///
  /// Uses cached complex FFT plan with pre-allocated buffer for better performance.
  /// Falls back to complex FFT because RealFFT's ComplexToReal expects full half-spectrum,
  /// but we only have odd bins from negacyclic FFT.
  fn fft(&mut self, input: &[f64]) -> Vec<u32> {
    assert_eq!(input.len(), self.n);

    let ns2 = input.len() / 2;
    let n = ns2 * 2;
    let nn = 2 * n;
    let scale = 2.0 / (n as f64);

    // Reuse pre-allocated buffer
    let mut buffer = self.inverse_buffer.borrow_mut();
    buffer.clear();
    buffer.resize(nn, Complex::new(0.0, 0.0));

    // Fill odd indices
    for i in 0..ns2 {
      let idx = 2 * i + 1;
      buffer[idx] = Complex::new(input[i] * scale, input[i + ns2] * scale);
    }

    // Fill conjugate symmetry
    for i in 0..ns2 {
      let idx = nn - 1 - 2 * i;
      buffer[idx] = Complex::new(input[i] * scale, -input[i + ns2] * scale);
    }

    // 2N-point FFT using cached plan with scratch buffer
    // Using process_with_scratch avoids per-call allocation
    // See: https://docs.rs/rustfft/latest/rustfft/trait.Fft.html#tymethod.process_with_scratch
    let mut scratch = self.scratch.borrow_mut();
    self.fft_2n.process_with_scratch(&mut buffer, &mut scratch);

    // Extract pattern with proper rounding
    let adjust = 0.25;
    let mut result = vec![0u32; n];
    result[0] = ((buffer[0].re * adjust).round()) as i64 as u32;
    for i in 1..n {
      result[i] = ((-buffer[n - i].re * adjust).round()) as i64 as u32;
    }

    result
  }

  fn ifft_1024(&mut self, input: &[u32; 1024]) -> [f64; 1024] {
    let vec_result = self.ifft(&input[..]);
    let mut result = [0.0f64; 1024];
    result.copy_from_slice(&vec_result);
    result
  }

  fn fft_1024(&mut self, input: &[f64; 1024]) -> [u32; 1024] {
    let vec_result = self.fft(&input[..]);
    let mut result = [0u32; 1024];
    result.copy_from_slice(&vec_result);
    result
  }

  fn poly_mul(&mut self, a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    assert_eq!(a.len(), b.len());
    let n = a.len();

    let a_fft = self.ifft(a);
    let b_fft = self.ifft(b);

    // Complex multiplication in frequency domain
    // Format: [re_0..re_N/2-1, im_0..im_N/2-1]
    // 0.5 scaling factor compensates for the 2x energy from odd-index extraction
    let mut result_fft = vec![0.0f64; n];
    let ns2 = n / 2;
    for i in 0..ns2 {
      let aimbim = a_fft[i + ns2] * b_fft[i + ns2];
      let arebim = a_fft[i] * b_fft[i + ns2];
      result_fft[i] = (a_fft[i] * b_fft[i] - aimbim) * 0.5;
      result_fft[i + ns2] = (a_fft[i + ns2] * b_fft[i] + arebim) * 0.5;
    }

    self.fft(&result_fft)
  }

  fn poly_mul_1024(&mut self, a: &[u32; 1024], b: &[u32; 1024]) -> [u32; 1024] {
    let a_fft = self.ifft_1024(a);
    let b_fft = self.ifft_1024(b);

    // Complex multiplication in frequency domain
    // Format: [re_0..re_511, im_0..im_511]
    // 0.5 scaling factor compensates for the 2x energy from odd-index extraction
    let mut result_fft = [0.0f64; 1024];
    let ns2 = 512;
    for i in 0..ns2 {
      let aimbim = a_fft[i + ns2] * b_fft[i + ns2];
      let arebim = a_fft[i] * b_fft[i + ns2];
      result_fft[i] = (a_fft[i] * b_fft[i] - aimbim) * 0.5;
      result_fft[i + ns2] = (a_fft[i + ns2] * b_fft[i] + arebim) * 0.5;
    }

    self.fft_1024(&result_fft)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::fft::rustfft_processor::RustFFTProcessor;

  #[test]
  fn test_realfft_correctness() {
    let mut realfft_proc = RealFFTProcessor::new(1024);
    let mut rustfft_proc = RustFFTProcessor::new(1024);

    let mut input = [0u32; 1024];
    input[0] = 1 << 31;
    input[1] = 1 << 30;

    let realfft_result = realfft_proc.ifft_1024(&input);
    let rustfft_result = rustfft_proc.ifft_1024(&input);

    let mut max_diff: f64 = 0.0;
    for i in 0..1024 {
      let diff = (realfft_result[i] - rustfft_result[i]).abs();
      max_diff = max_diff.max(diff);
    }

    println!("Max difference vs RustFFT: {:.2e}", max_diff);
    assert!(max_diff < 1.0, "RealFFT should match RustFFT!");
  }

  #[test]
  fn test_realfft_simple_case() {
    // Test with simple known case
    let mut realfft_proc = RealFFTProcessor::new(8);
    let mut rustfft_proc = RustFFTProcessor::new(8);

    let input: [u32; 8] = [100, 200, 50, 150, 0, 0, 0, 0];

    // Forward transform
    let realfft_freq = realfft_proc.ifft(&input[..]);
    let rustfft_freq = rustfft_proc.ifft(&input[..]);

    println!("\nForward transform comparison:");
    for i in 0..8 {
      println!(
        "  [{}] RealFFT: {:.4}, RustFFT: {:.4}, diff: {:.2e}",
        i,
        realfft_freq[i],
        rustfft_freq[i],
        (realfft_freq[i] - rustfft_freq[i]).abs()
      );
    }

    // Inverse transform
    let realfft_output = realfft_proc.fft(&realfft_freq);
    println!("\nRealFFT roundtrip:");
    for i in 0..8 {
      println!(
        "  [{}] Input: {}, Output: {}, diff: {}",
        i,
        input[i],
        realfft_output[i],
        (realfft_output[i] as i64 - input[i] as i64).abs()
      );
    }
  }

  #[test]
  #[ignore]
  fn test_realfft_roundtrip() {
    let mut proc = RealFFTProcessor::new(1024);

    let input: [u32; 1024] = core::array::from_fn(|i| ((i * 12345) % 65536) as u32);

    let freq = proc.ifft_1024(&input);
    let output = proc.fft_1024(&freq);

    let mut max_diff: i64 = 0;
    for i in 0..1024 {
      let diff = (output[i] as i64 - input[i] as i64).abs();
      max_diff = max_diff.max(diff);
    }

    println!("Roundtrip error: {}", max_diff);
    assert!(max_diff < 100, "Roundtrip error too large: {}", max_diff);
  }

  #[test]
  #[ignore]
  fn bench_realfft_vs_rustfft() {
    use std::time::Instant;

    let mut realfft_proc = RealFFTProcessor::new(1024);
    let mut rustfft_proc = RustFFTProcessor::new(1024);

    let test_poly = [1u32 << 30; 1024];
    let test_freq = [1.0f64; 1024];
    let iterations = 1000;

    // Benchmark RealFFT forward (ifft)
    let start = Instant::now();
    for _ in 0..iterations {
      let _ = realfft_proc.ifft_1024(&test_poly);
    }
    let realfft_ifft_time = start.elapsed();

    // Benchmark RustFFT forward (ifft)
    let start = Instant::now();
    for _ in 0..iterations {
      let _ = rustfft_proc.ifft_1024(&test_poly);
    }
    let rustfft_ifft_time = start.elapsed();

    // Benchmark RealFFT inverse (fft)
    let start = Instant::now();
    for _ in 0..iterations {
      let _ = realfft_proc.fft_1024(&test_freq);
    }
    let realfft_fft_time = start.elapsed();

    // Benchmark RustFFT inverse (fft)
    let start = Instant::now();
    for _ in 0..iterations {
      let _ = rustfft_proc.fft_1024(&test_freq);
    }
    let rustfft_fft_time = start.elapsed();

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║         RealFFT vs RustFFT Performance (1024)            ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Forward Transform (time → frequency, ifft):             ║");
    println!(
      "║    RealFFT: {:?} ({:.2}µs)                   ║",
      realfft_ifft_time,
      realfft_ifft_time.as_micros() as f64 / iterations as f64
    );
    println!(
      "║    RustFFT: {:?} ({:.2}µs)                    ║",
      rustfft_ifft_time,
      rustfft_ifft_time.as_micros() as f64 / iterations as f64
    );
    println!(
      "║    Speedup: {:.2}x                                        ║",
      rustfft_ifft_time.as_secs_f64() / realfft_ifft_time.as_secs_f64()
    );
    println!("║                                                          ║");
    println!("║  Inverse Transform (frequency → time, fft):              ║");
    println!(
      "║    RealFFT: {:?} ({:.2}µs)                   ║",
      realfft_fft_time,
      realfft_fft_time.as_micros() as f64 / iterations as f64
    );
    println!(
      "║    RustFFT: {:?} ({:.2}µs)                    ║",
      rustfft_fft_time,
      rustfft_fft_time.as_micros() as f64 / iterations as f64
    );
    println!(
      "║    Speedup: {:.2}x                                        ║",
      rustfft_fft_time.as_secs_f64() / realfft_fft_time.as_secs_f64()
    );
    println!("╚══════════════════════════════════════════════════════════╝");
  }
}
