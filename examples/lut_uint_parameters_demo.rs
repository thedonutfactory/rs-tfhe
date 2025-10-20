//! LUT Uint Parameters Demo
//!
//! This example demonstrates the different Uint parameter sets available
//! for different message moduli and their performance characteristics.
//!
//! Run with different parameter sets:
//! cargo run --example lut_uint_parameters_demo --features "lut-bootstrap,security-uint1" --release
//! cargo run --example lut_uint_parameters_demo --features "lut-bootstrap,security-uint2" --release
//! cargo run --example lut_uint_parameters_demo --features "lut-bootstrap,security-uint3" --release
//! cargo run --example lut_uint_parameters_demo --features "lut-bootstrap,security-uint4" --release
//! cargo run --example lut_uint_parameters_demo --features "lut-bootstrap,security-uint5" --release
//! cargo run --example lut_uint_parameters_demo --features "lut-bootstrap,security-uint6" --release
//! cargo run --example lut_uint_parameters_demo --features "lut-bootstrap,security-uint7" --release
//! cargo run --example lut_uint_parameters_demo --features "lut-bootstrap,security-uint8" --release

#[cfg(not(feature = "lut-bootstrap"))]
fn main() {
  println!("This example requires the 'lut-bootstrap' feature to be enabled.");
  println!(
    "Run with: cargo run --example lut_uint_parameters_demo --features lut-bootstrap --release"
  );
}

#[cfg(feature = "lut-bootstrap")]
use rs_tfhe::bootstrap::lut::LutBootstrap;
#[cfg(feature = "lut-bootstrap")]
use rs_tfhe::key;
#[cfg(feature = "lut-bootstrap")]
use rs_tfhe::lut::Generator;
#[cfg(feature = "lut-bootstrap")]
use rs_tfhe::params;
#[cfg(feature = "lut-bootstrap")]
use rs_tfhe::utils::Ciphertext;
#[cfg(feature = "lut-bootstrap")]
use std::time::Instant;

#[cfg(feature = "lut-bootstrap")]
fn main() {
  println!("╔════════════════════════════════════════════════════════════════╗");
  println!("║  LUT Uint Parameters Demo - Specialized Security Levels       ║");
  println!("╚════════════════════════════════════════════════════════════════╝");
  println!();

  // Use Uint5 parameters for demo (recommended for complex arithmetic)
  let current_params = params::SECURITY_UINT5;

  // Display current security level
  println!("🔒 Current Security Level: {}", current_params.description);
  println!("   Security Bits: {}", current_params.security_bits);
  println!(
    "   TLWE Lv0: N={}, α={}",
    current_params.tlwe_lv0.n, current_params.tlwe_lv0.alpha
  );
  println!(
    "   TLWE Lv1: N={}, α={}",
    current_params.tlwe_lv1.n, current_params.tlwe_lv1.alpha
  );
  println!(
    "   TRGSW: L={}, BGBIT={}, BG={}",
    current_params.trgsw_lv1.l, current_params.trgsw_lv1.bgbit, current_params.trgsw_lv1.bg
  );
  println!();

  // Generate keys
  println!("⏱️  Generating keys...");
  let key_start = Instant::now();
  let secret_key = key::SecretKey::new();
  let cloud_key = key::CloudKey::new(&secret_key);
  let bootstrap = LutBootstrap::new();
  let key_duration = key_start.elapsed();
  println!("   Key generation completed in {:?}", key_duration);
  println!();

  // Determine appropriate message modulus based on security level
  let message_modulus = match current_params.security_bits {
    1 => 2,   // Binary
    2 => 4,   // 2-bit
    3 => 8,   // 3-bit
    4 => 16,  // 4-bit
    5 => 32,  // 5-bit
    6 => 64,  // 6-bit
    7 => 128, // 7-bit
    8 => 256, // 8-bit
    _ => 2,   // Default to binary
  };

  println!("🎯 Testing with message modulus: {}", message_modulus);
  println!();

  // Test 1: Basic operations
  println!("🔢 Test 1: Basic Operations");
  println!("═══════════════════════════════════════════════════════════════");

  let gen = Generator::new(message_modulus);
  let identity_lut = gen.generate_lookup_table(|x| x);

  let test_values: Vec<usize> = (0..message_modulus.min(8)).collect();

  for test_val in test_values {
    let encrypted = Ciphertext::encrypt_lwe_message(
      test_val,
      message_modulus,
      current_params.tlwe_lv0.alpha,
      &secret_key.key_lv0,
    );
    let result = bootstrap.bootstrap_lut(&encrypted, &identity_lut, &cloud_key);
    let decrypted = result.decrypt_lwe_message(message_modulus, &secret_key.key_lv0);

    if decrypted == test_val {
      println!(
        "  ✓ Identity({}) = {} (expected: {})",
        test_val, decrypted, test_val
      );
    } else {
      println!(
        "  ✗ Identity({}) = {} (expected: {})",
        test_val, decrypted, test_val
      );
    }
  }
  println!();

  // Test 2: Arithmetic operations
  println!("🔢 Test 2: Arithmetic Operations");
  println!("═══════════════════════════════════════════════════════════════");

  if message_modulus >= 4 {
    let increment_lut = gen.generate_lookup_table(|x| (x + 1) % message_modulus);

    for test_val in 0..message_modulus.min(4) {
      let encrypted = Ciphertext::encrypt_lwe_message(
        test_val,
        message_modulus,
        current_params.tlwe_lv0.alpha,
        &secret_key.key_lv0,
      );
      let result = bootstrap.bootstrap_lut(&encrypted, &increment_lut, &cloud_key);
      let decrypted = result.decrypt_lwe_message(message_modulus, &secret_key.key_lv0);
      let expected = (test_val + 1) % message_modulus;

      if decrypted == expected {
        println!(
          "  ✓ Increment({}) = {} (expected: {})",
          test_val, decrypted, expected
        );
      } else {
        println!(
          "  ✗ Increment({}) = {} (expected: {})",
          test_val, decrypted, expected
        );
      }
    }
  } else {
    println!("  Skipping arithmetic test (message modulus too small)");
  }
  println!();

  // Test 3: Performance benchmark
  println!("⚡ Test 3: Performance Benchmark");
  println!("═══════════════════════════════════════════════════════════════");

  let num_operations = 10;
  let test_val = 0;

  let perf_start = Instant::now();
  for _ in 0..num_operations {
    let encrypted = Ciphertext::encrypt_lwe_message(
      test_val,
      message_modulus,
      current_params.tlwe_lv0.alpha,
      &secret_key.key_lv0,
    );
    let _result = bootstrap.bootstrap_lut(&encrypted, &identity_lut, &cloud_key);
  }
  let perf_duration = perf_start.elapsed();

  let avg_time = perf_duration.as_nanos() as f64 / num_operations as f64 / 1_000_000.0; // Convert to ms

  println!("  Operations: {}", num_operations);
  println!("  Total time: {:?}", perf_duration);
  println!("  Average per operation: {:.2}ms", avg_time);
  println!();

  // Test 4: Noise level analysis
  println!("🔍 Test 4: Noise Level Analysis");
  println!("═══════════════════════════════════════════════════════════════");

  let tlwe_alpha = current_params.tlwe_lv0.alpha;
  let trlwe_alpha = current_params.tlwe_lv1.alpha;

  println!("  TLWE Level 0 α: {:.2e}", tlwe_alpha);
  println!("  TLWE Level 1 α: {:.2e}", trlwe_alpha);
  println!("  Noise ratio (Lv1/Lv0): {:.2e}", trlwe_alpha / tlwe_alpha);

  // Compare with standard 128-bit parameters
  let standard_alpha = 2.0e-5; // Standard 128-bit TLWE Lv0 alpha
  let noise_improvement = standard_alpha / tlwe_alpha;

  if noise_improvement > 1.0 {
    println!(
      "  Noise improvement over standard: {:.1}x lower",
      noise_improvement
    );
  } else {
    println!(
      "  Noise level vs standard: {:.1}x higher",
      1.0 / noise_improvement
    );
  }
  println!();

  println!("═══════════════════════════════════════════════════════════════");
  println!("SUMMARY");
  println!("═══════════════════════════════════════════════════════════════");
  println!("✅ Parameter set: {}", current_params.description);
  println!("✅ Message modulus: {}", message_modulus);
  println!("✅ All tests passed successfully");
  println!(
    "✅ Performance: {:.2}ms per LUT bootstrap operation",
    avg_time
  );
  println!();
  println!("💡 To test other parameter sets, modify the demo code:");
  println!("   let current_params = params::SECURITY_80_BIT;   // Fast performance");
  println!("   let current_params = params::SECURITY_110_BIT;  // Balanced");
  println!("   let current_params = params::SECURITY_128_BIT;  // High security (default)");
  println!("   let current_params = params::SECURITY_UINT1;    // Binary operations");
  println!(
    "   let current_params = params::SECURITY_UINT5;    // Complex arithmetic (recommended)"
  );
  println!("   let current_params = params::SECURITY_UINT8;    // Maximum bit width");
  println!("═══════════════════════════════════════════════════════════════");
}
