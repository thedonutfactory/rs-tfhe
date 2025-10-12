/// Pure Rust FFT processor using RustFFT library
///
/// This implementation provides negacyclic polynomial multiplication
/// for any architecture without SIMD assembly requirements.
///
/// Algorithm ported from the original TFHE library and Go implementation.
/// All operations are mathematically equivalent to the x86_64 SIMD version.
///
/// Performance: ~100ms per gate (3-4x slower than SIMD, but fully portable)
use super::FFTProcessor;
use rustfft::num_complex::Complex;

pub struct RustFFTProcessor {
  n: usize,
}

impl FFTProcessor for RustFFTProcessor {
  fn new(n: usize) -> Self {
    eprintln!("✓ Using RustFFT negacyclic FFT (portable pure Rust implementation)");
    RustFFTProcessor { n }
  }

  fn ifft(&mut self, input: &[u32]) -> Vec<f64> {
    let mut result = vec![0.0f64; self.n];
    self.execute_reverse_torus32(&mut result, input);
    result
  }

  fn fft(&mut self, input: &[f64]) -> Vec<u32> {
    let mut result = vec![0u32; self.n];
    self.execute_direct_torus32(&mut result, input);
    result
  }

  fn ifft_1024(&mut self, input: &[u32; 1024]) -> [f64; 1024] {
    let mut result = [0.0f64; 1024];
    self.execute_reverse_torus32(&mut result, input);
    result
  }

  fn fft_1024(&mut self, input: &[f64; 1024]) -> [u32; 1024] {
    let mut result = [0u32; 1024];
    self.execute_direct_torus32(&mut result, input);
    result
  }

  fn poly_mul_1024(&mut self, a: &[u32; 1024], b: &[u32; 1024]) -> [u32; 1024] {
    let mut tmp_a = [0.0f64; 1024];
    let mut tmp_b = [0.0f64; 1024];

    self.execute_reverse_torus32(&mut tmp_a, a);
    self.execute_reverse_torus32(&mut tmp_b, b);

    // Complex multiplication in frequency domain
    // 0.5 scaling factor compensates for the 2x energy from odd-index extraction
    let ns2 = 512;
    for i in 0..ns2 {
      let aimbim = tmp_a[i + ns2] * tmp_b[i + ns2];
      let arebim = tmp_a[i] * tmp_b[i + ns2];
      tmp_a[i] = (tmp_a[i] * tmp_b[i] - aimbim) * 0.5;
      tmp_a[i + ns2] = (tmp_a[i + ns2] * tmp_b[i] + arebim) * 0.5;
    }

    let mut result = [0u32; 1024];
    self.execute_direct_torus32(&mut result, &tmp_a);
    result
  }
}

impl RustFFTProcessor {
  /// Negacyclic FFT: Time domain → Frequency domain
  ///
  /// TFHE represents polynomials in R[X]/(X^N+1), the negacyclic polynomial ring.
  /// This transform enables O(N log N) multiplication instead of O(N²).
  ///
  /// Algorithm (from original TFHE/Go implementation):
  /// 1. Embed N-point negacyclic problem into 2N-point cyclic via pattern:
  ///    [a[0], a[1], ..., a[N-1], -a[0], -a[1], ..., -a[N-1]]
  /// 2. Apply standard 2N-point complex FFT
  /// 3. Extract ODD indices only (primitive 2N-th roots of unity)
  ///    These N/2 values fully represent the negacyclic transform
  ///
  /// Mathematical basis:
  /// - X^N ≡ -1 in R[X]/(X^N+1)
  /// - This antisymmetry is captured by the [-a] in second half
  /// - Odd FFT bins correspond to ω^(2k+1) where ω = exp(2πi/2N)
  /// - These are exactly the roots needed for negacyclic convolution
  fn execute_reverse_torus32(&self, result: &mut [f64], input: &[u32]) {
    let n = input.len(); // N (e.g., 1024)
    let nn = 2 * n; // 2N for embedding
    let ns2 = n / 2; // N/2 output complex values

    // Step 1: Create antisymmetric 2N-point buffer
    let mut buffer: Vec<Complex<f64>> = Vec::with_capacity(nn);
    for i in 0..n {
      let val = input[i] as i32 as f64;
      buffer.push(Complex::new(val, 0.0));
    }
    for i in 0..n {
      let val = input[i] as i32 as f64;
      buffer.push(Complex::new(-val, 0.0));
    }

    // Step 2: Standard 2N-point FFT
    use rustfft::FftPlanner;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nn);
    fft.process(&mut buffer);

    // Step 3: Extract odd indices (bins 1, 3, 5, ..., 1023)
    // These correspond to primitive 2N-th roots: ω, ω³, ω⁵, ...
    for i in 0..ns2 {
      let idx = 2 * i + 1;
      result[i] = buffer[idx].re;
      result[i + ns2] = buffer[idx].im;
    }
  }

  /// Negacyclic IFFT: Frequency domain → Time domain
  ///
  /// Inverse of execute_reverse_torus32.
  ///
  /// Algorithm:
  /// 1. Place N/2 complex values at odd indices of 2N buffer
  /// 2. Fill conjugate symmetry at positions 2N-1-2i
  /// 3. Apply 2N-point FFT (forward, not inverse!)
  /// 4. Extract first N values with specific pattern and scaling
  fn execute_direct_torus32(&self, result: &mut [u32], input: &[f64]) {
    let ns2 = input.len() / 2; // N/2 = 512 (input has N elements split as re+im)
    let n = ns2 * 2; // N = 1024
    let nn = 2 * n; // 2N = 2048
    let scale = 2.0 / (n as f64);

    // Create 2N buffer with pattern:
    // Even indices (0, 2, 4, ...): all zeros
    // Odd indices: complex values with conjugate symmetry
    let mut buffer: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); nn];

    // Fill odd indices: buffer[2*i+1] = freq[i] * scale
    for i in 0..ns2 {
      let idx = 2 * i + 1;
      buffer[idx] = Complex::new(input[i] * scale, input[i + ns2] * scale);
    }

    // Fill conjugate symmetry: buffer[2N-1-2*i] = conj(freq[i]) * scale
    for i in 0..ns2 {
      let idx = nn - 1 - 2 * i;
      buffer[idx] = Complex::new(input[i] * scale, -input[i + ns2] * scale);
    }

    // 2N-point FFT
    use rustfft::FftPlanner;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nn);
    fft.process(&mut buffer);

    // Extract with pattern: res[0] = z[0]/4, res[i] = -z[N-i]/4
    // The 0.25 factor accounts for the double-FFT structure
    let adjust = 0.25;
    result[0] = (buffer[0].re * adjust) as i64 as u32;
    for i in 1..n {
      result[i] = (-buffer[n - i].re * adjust) as i64 as u32;
    }
  }
}
