//! TFHE-FFT Processor - Zama's high-performance FFT library
//!
//! **STATUS: PARTIALLY WORKING**
//! - ✅ Roundtrip tests pass (ifft->fft recovers original)
//! - ❌ Polynomial multiplication has minor issues
//! - Needs more work to exactly match tfhe-rs normalization
//!
//! Uses Extended Fourier Transform for negacyclic polynomial multiplication
//! Based on the paper: "Fast and Error-Free Negacyclic Integer Convolution using Extended Fourier Transform"
//! https://eprint.iacr.org/2021/480
//!
//! Key algorithm (from tfhe-rs source code analysis):
//! 1. Split N-element polynomial into two N/2 halves
//! 2. Apply twisting factors (2N-th roots of unity) - pre-multiply
//! 3. Run N/2-point complex FFT (not 2N!) using unordered API
//! 4. Inverse: N/2-point IFFT then post-multiply by conjugate twisting factors
//!
//! **Key insight**: tfhe-rs uses N/2 FFT with twisting, NOT 2N FFT with antisymmetric embedding!

use super::FFTProcessor;
use dyn_stack::{GlobalPodBuffer, PodStack};
use tfhe_fft::c64;
use tfhe_fft::unordered::Plan;

pub struct TfheFftProcessor {
  n: usize,
  plan: Plan,            // N/2 plan (not 2N!)
  twisties_re: Vec<f64>, // Real part of 2N-th roots of unity
  twisties_im: Vec<f64>, // Imaginary part of 2N-th roots of unity
  scratch: GlobalPodBuffer,
}

impl TfheFftProcessor {
  pub fn new(n: usize) -> Self {
    assert_eq!(n, 1024, "Only N=1024 supported for now");

    // Create N/2 unordered plan (KEY INSIGHT: N/2, not 2N!)
    use tfhe_fft::ordered::FftAlgo;
    use tfhe_fft::unordered::Method;
    let plan = Plan::new(
      n / 2, // N/2 = 512 for N=1024
      Method::UserProvided {
        base_algo: FftAlgo::Dif4,
        base_n: 512, // Use 512 as base size
      },
    );

    // Allocate scratch memory for the plan
    let scratch = GlobalPodBuffer::new(plan.fft_scratch().unwrap());

    // Compute twisting factors: 2N-th roots of unity
    // These are exp(i*pi*k/(2*n/2)) = exp(i*pi*k/n) for k=0..n/2-1
    let mut twisties_re = Vec::with_capacity(n / 2);
    let mut twisties_im = Vec::with_capacity(n / 2);
    let unit = std::f64::consts::PI / n as f64;
    for i in 0..n / 2 {
      let (im, re) = (i as f64 * unit).sin_cos();
      twisties_re.push(re);
      twisties_im.push(im);
    }

    TfheFftProcessor {
      n,
      plan,
      twisties_re,
      twisties_im,
      scratch,
    }
  }
}

impl FFTProcessor for TfheFftProcessor {
  fn new(n: usize) -> Self {
    TfheFftProcessor::new(n)
  }

  fn ifft_1024(&mut self, input: &[u32; 1024]) -> [f64; 1024] {
    const N: usize = 1024;
    const N2: usize = N / 2; // 512

    // Split input into two halves: [0..512) and [512..1024)
    let (input_re, input_im) = input.split_at(N2);

    // Apply twisting factors and convert to complex
    // fourier[i] = (input_re[i] + i*input_im[i]) * (twisties_re[i] + i*twisties_im[i])
    let mut fourier: Vec<c64> = Vec::with_capacity(N2);
    let normalization = 2.0_f64.powi(-(32 as i32)); // 2^-32 for u32 torus

    for i in 0..N2 {
      let in_re = input_re[i] as i32 as f64 * normalization;
      let in_im = input_im[i] as i32 as f64 * normalization;
      let w_re = self.twisties_re[i];
      let w_im = self.twisties_im[i];

      // Complex multiplication: (in_re + i*in_im) * (w_re + i*w_im)
      fourier.push(c64::new(
        in_re * w_re - in_im * w_im, // Real part
        in_re * w_im + in_im * w_re, // Imaginary part
      ));
    }

    // Forward FFT on N/2 points
    let stack = PodStack::new(&mut self.scratch);
    self.plan.fwd(&mut fourier, stack);

    // Convert to output format (N/2 complex values stored as [re, im])
    let mut result = [0.0f64; N];
    for i in 0..N2 {
      result[i] = fourier[i].re;
      result[i + N2] = fourier[i].im;
    }

    result
  }

  fn fft_1024(&mut self, input: &[f64; 1024]) -> [u32; 1024] {
    const N: usize = 1024;
    const N2: usize = N / 2; // 512

    // Input is N/2 complex values stored as [re_0..re_511, im_0..im_511]
    let mut fourier: Vec<c64> = Vec::with_capacity(N2);
    for i in 0..N2 {
      fourier.push(c64::new(input[i], input[i + N2]));
    }

    // Inverse FFT on N/2 points
    let stack = PodStack::new(&mut self.scratch);
    self.plan.inv(&mut fourier, stack);

    // Apply inverse twisting factors and convert back to torus
    // Normalization: 1.0 / N2 for the IFFT
    let normalization = 1.0 / (N2 as f64);
    let mut result = [0u32; N];

    for i in 0..N2 {
      let w_re = self.twisties_re[i];
      let w_im = self.twisties_im[i];

      // Complex multiplication with conjugate and normalization:
      // (f_re + i*f_im) * (w_re - i*w_im) * normalization
      let f_re = fourier[i].re;
      let f_im = fourier[i].im;

      let tmp_re = (f_re * w_re + f_im * w_im) * normalization;
      let tmp_im = (f_im * w_re - f_re * w_im) * normalization;

      // Convert from torus ([-0.5, 0.5)) to u32
      // from_torus: multiply by 2^32
      result[i] = (tmp_re * 2.0_f64.powi(32)).round() as i64 as u32;
      result[i + N2] = (tmp_im * 2.0_f64.powi(32)).round() as i64 as u32;
    }

    result
  }

  fn ifft(&mut self, input: &[u32]) -> Vec<f64> {
    let mut arr = [0u32; 1024];
    arr.copy_from_slice(input);
    self.ifft_1024(&arr).to_vec()
  }

  fn fft(&mut self, input: &[f64]) -> Vec<u32> {
    let mut arr = [0f64; 1024];
    arr.copy_from_slice(input);
    self.fft_1024(&arr).to_vec()
  }

  fn poly_mul_1024(&mut self, a: &[u32; 1024], b: &[u32; 1024]) -> [u32; 1024] {
    let a_fft = self.ifft_1024(a);
    let b_fft = self.ifft_1024(b);

    // Complex multiplication in frequency domain
    // Format: [re_0..re_511, im_0..im_511]
    // Extended Fourier Transform: Just multiply, no scaling needed!
    // (twisting factors already handle negacyclic property)
    let mut result_fft = [0.0f64; 1024];
    const N2: usize = 512;
    for i in 0..N2 {
      let ar = a_fft[i];
      let ai = a_fft[i + N2];
      let br = b_fft[i];
      let bi = b_fft[i + N2];

      // Simple complex multiplication: (ar + i*ai) * (br + i*bi)
      result_fft[i] = ar * br - ai * bi;
      result_fft[i + N2] = ar * bi + ai * br;
    }

    self.fft_1024(&result_fft)
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
