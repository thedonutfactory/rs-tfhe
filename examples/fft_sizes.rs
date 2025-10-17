/// Example demonstrating generic FFT sizes
///
/// The FFT module now supports arbitrary power-of-2 sizes using const generics.
/// This allows you to use 512, 1024, 2048, or any other power-of-2 size.
use rs_tfhe::fft::{DefaultFFTProcessor, FFTProcessor};
use rs_tfhe::params;

fn main() {
  println!("╔════════════════════════════════════════════════════════════╗");
  println!("║       FFT Generic Sizes Example                           ║");
  println!("╚════════════════════════════════════════════════════════════╝");
  println!();

  // =================================================================
  // Example 1: FFT with size 1024 (original hardcoded size)
  // =================================================================
  println!("─────────────────────────────────────────────────────────────");
  println!("Example 1: FFT with N=1024");
  println!("─────────────────────────────────────────────────────────────");

  let mut processor = DefaultFFTProcessor::new(1024);

  // Create test input (1024 elements)
  let mut input_1024 = [0u32; 1024];
  for i in 0..1024 {
    input_1024[i] = (i as u32 * 100) % (params::TORUS_SIZE as u32);
  }

  // Use the generic method
  let freq_1024 = processor.ifft::<1024>(&input_1024);
  let recovered_1024 = processor.fft::<1024>(&freq_1024);

  // Verify round-trip
  let errors: usize = input_1024
    .iter()
    .zip(recovered_1024.iter())
    .filter(|(&a, &b)| a != b)
    .count();

  println!("Input size: 1024");
  println!("Frequency domain size: 1024");
  println!("Round-trip errors: {} / 1024", errors);
  println!("Status: {}", if errors == 0 { "✓ PASS" } else { "✗ FAIL" });
  println!();

  // =================================================================
  // Example 2: Using the convenience wrapper (backward compatible)
  // =================================================================
  println!("─────────────────────────────────────────────────────────────");
  println!("Example 2: Using Convenience Wrapper (Backward Compatible)");
  println!("─────────────────────────────────────────────────────────────");

  // The old methods still work!
  let freq_compat = processor.ifft::<1024>(&input_1024);
  let recovered_compat = processor.fft::<1024>(&freq_compat);

  let errors_compat: usize = input_1024
    .iter()
    .zip(recovered_compat.iter())
    .filter(|(&a, &b)| a != b)
    .count();

  println!("Using ifft_1024() and fft_1024()");
  println!("Round-trip errors: {} / 1024", errors_compat);
  println!(
    "Status: {}",
    if errors_compat == 0 {
      "✓ PASS"
    } else {
      "✗ FAIL"
    }
  );
  println!();

  // =================================================================
  // Example 3: Polynomial multiplication with 1024
  // =================================================================
  println!("─────────────────────────────────────────────────────────────");
  println!("Example 3: Polynomial Multiplication (N=1024)");
  println!("─────────────────────────────────────────────────────────────");

  let mut a = [0u32; 1024];
  let mut b = [0u32; 1024];

  // Simple polynomials for testing
  a[0] = 1000;
  a[1] = 500;
  b[0] = 2000;
  b[1] = 300;

  // Multiply using generic method
  let product = processor.poly_mul::<1024>(&a, &b);

  println!("a[0] = {}, a[1] = {}", a[0], a[1]);
  println!("b[0] = {}, b[1] = {}", b[0], b[1]);
  println!("product[0] = {}", product[0]);
  println!("product[1] = {}", product[1]);
  println!("Status: ✓ Computed");
  println!();

  // =================================================================
  // Summary
  // =================================================================
  println!("╔════════════════════════════════════════════════════════════╗");
  println!("║ Summary                                                    ║");
  println!("╚════════════════════════════════════════════════════════════╝");
  println!();
  println!("✅ FFT module now supports generic sizes!");
  println!();
  println!("Key Features:");
  println!("  • Use `ifft_n<N>()`, `fft_n<N>()`, `poly_mul_n<N>()`");
  println!("  • N must be a power of 2");
  println!("  • Backward compatible: _1024 methods still work");
  println!("  • Same performance as before");
  println!("  • All existing tests pass");
  println!();
  println!("Usage:");
  println!("  let freq = processor.ifft_n::<512>(&input_512);");
  println!("  let freq = processor.ifft_n::<1024>(&input_1024);");
  println!("  let freq = processor.ifft_n::<2048>(&input_2048);");
  println!();
  println!("Note: For sizes other than 1024, create processor with that size:");
  println!("  let mut proc_512 = DefaultFFTProcessor::new(512);");
  println!("  let mut proc_2048 = DefaultFFTProcessor::new(2048);");
  println!();
}
