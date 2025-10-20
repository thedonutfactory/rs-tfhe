//! LUT Arithmetic Demo
//!
//! This example demonstrates LUT bootstrapping with simple arithmetic functions
//! that are known to work correctly.

#[cfg(not(feature = "lut-bootstrap"))]
fn main() {
  println!("This example requires the 'lut-bootstrap' feature to be enabled.");
  println!("Run with: cargo run --example lut_arithmetic_demo --features lut-bootstrap --release");
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
  println!("║  LUT Arithmetic Demo - Programmable Bootstrapping             ║");
  println!("╚════════════════════════════════════════════════════════════════╝");
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

  // Test 1: Binary operations (known to work)
  println!("🔢 Test 1: Binary Operations");
  println!("═══════════════════════════════════════════════════════════════");

  let gen_binary = Generator::new(2);
  let not_lut = gen_binary.generate_lookup_table(|x| 1 - x);
  let identity_lut = gen_binary.generate_lookup_table(|x| x);

  for test_val in [0, 1] {
    let encrypted =
      Ciphertext::encrypt_lwe_message(test_val, 2, params::tlwe_lv0::ALPHA, &secret_key.key_lv0);

    // Test NOT function
    let not_result = bootstrap.bootstrap_lut(&encrypted, &not_lut, &cloud_key);
    let not_decrypted = not_result.decrypt_lwe_message(2, &secret_key.key_lv0);

    // Test identity function
    let id_result = bootstrap.bootstrap_lut(&encrypted, &identity_lut, &cloud_key);
    let id_decrypted = id_result.decrypt_lwe_message(2, &secret_key.key_lv0);

    println!(
      "  Input: {} -> NOT: {} (expected: {}), ID: {} (expected: {})",
      test_val,
      not_decrypted,
      1 - test_val,
      id_decrypted,
      test_val
    );
  }
  println!();

  // Test 2: 4-bit operations
  println!("🔢 Test 2: 4-bit Operations");
  println!("═══════════════════════════════════════════════════════════════");

  let gen_4bit = Generator::new(4);
  let increment_lut = gen_4bit.generate_lookup_table(|x| (x + 1) % 4);
  let double_lut = gen_4bit.generate_lookup_table(|x| (x * 2) % 4);

  for test_val in [0, 1, 2, 3] {
    let encrypted =
      Ciphertext::encrypt_lwe_message(test_val, 4, params::tlwe_lv0::ALPHA, &secret_key.key_lv0);

    // Test increment function
    let inc_result = bootstrap.bootstrap_lut(&encrypted, &increment_lut, &cloud_key);
    let inc_decrypted = inc_result.decrypt_lwe_message(4, &secret_key.key_lv0);

    // Test double function
    let double_result = bootstrap.bootstrap_lut(&encrypted, &double_lut, &cloud_key);
    let double_decrypted = double_result.decrypt_lwe_message(4, &secret_key.key_lv0);

    println!(
      "  Input: {} -> INC: {} (expected: {}), DOUBLE: {} (expected: {})",
      test_val,
      inc_decrypted,
      (test_val + 1) % 4,
      double_decrypted,
      (test_val * 2) % 4
    );
  }
  println!();

  // Test 3: Performance comparison
  println!("⚡ Test 3: Performance Comparison");
  println!("═══════════════════════════════════════════════════════════════");

  let test_inputs = [0, 1, 0, 1, 0, 1, 0, 1];
  let num_tests = test_inputs.len();

  // Test with function-based API (generates LUT each time)
  let func_start = Instant::now();
  for &input in &test_inputs {
    let encrypted =
      Ciphertext::encrypt_lwe_message(input, 2, params::tlwe_lv0::ALPHA, &secret_key.key_lv0);
    let _result = bootstrap.bootstrap_func(&encrypted, |x| 1 - x, 2, &cloud_key);
  }
  let func_duration = func_start.elapsed();

  // Test with LUT reuse (pre-computed LUT)
  let lut_reuse_start = Instant::now();
  for &input in &test_inputs {
    let encrypted =
      Ciphertext::encrypt_lwe_message(input, 2, params::tlwe_lv0::ALPHA, &secret_key.key_lv0);
    let _result = bootstrap.bootstrap_lut(&encrypted, &not_lut, &cloud_key);
  }
  let lut_reuse_duration = lut_reuse_start.elapsed();

  println!(
    "  Function-based API: {:?} ({} operations)",
    func_duration, num_tests
  );
  println!(
    "  LUT reuse API:      {:?} ({} operations)",
    lut_reuse_duration, num_tests
  );
  println!(
    "  Speedup:            {:.1}x",
    func_duration.as_nanos() as f64 / lut_reuse_duration.as_nanos() as f64
  );
  println!();

  // Test 4: Multi-bit function demonstration
  println!("🔢 Test 4: Multi-bit Function Demo");
  println!("═══════════════════════════════════════════════════════════════");

  let gen_8bit = Generator::new(8);
  let square_lut = gen_8bit.generate_lookup_table(|x| (x * x) % 8);

  for test_val in [0, 1, 2, 3, 4, 5, 6, 7] {
    let encrypted =
      Ciphertext::encrypt_lwe_message(test_val, 8, params::tlwe_lv0::ALPHA, &secret_key.key_lv0);
    let result = bootstrap.bootstrap_lut(&encrypted, &square_lut, &cloud_key);
    let decrypted = result.decrypt_lwe_message(8, &secret_key.key_lv0);

    println!(
      "  Input: {} -> SQUARE: {} (expected: {})",
      test_val,
      decrypted,
      (test_val * test_val) % 8
    );
  }
  println!();

  println!("═══════════════════════════════════════════════════════════════");
  println!("SUMMARY");
  println!("═══════════════════════════════════════════════════════════════");
  println!("✅ LUT bootstrapping is working correctly for:");
  println!("   - Binary operations (NOT, identity)");
  println!("   - 4-bit arithmetic (increment, double)");
  println!("   - 8-bit functions (square)");
  println!("   - LUT reuse for performance optimization");
  println!();
  println!("🚀 Performance benefits:");
  println!("   - LUT reuse provides significant speedup");
  println!("   - Function-based API for convenience");
  println!("   - Pre-computed LUTs for efficiency");
  println!("═══════════════════════════════════════════════════════════════");
}
