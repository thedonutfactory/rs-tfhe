//! Fast 8-bit Addition Using Programmable Bootstrapping
//!
//! This example demonstrates how LUT bootstrapping can be used to perform
//! efficient homomorphic addition by working with nibbles (4-bit chunks)
//! and using lookup tables for modular arithmetic and carry extraction.
//!
//! The key insight is that we can perform 8-bit addition using only 3 bootstraps
//! instead of the traditional 8 bootstraps needed for bit-by-bit addition.
//!
//! Run with: cargo run --example lut_add_two_numbers --features lut-bootstrap --release

#[cfg(not(feature = "lut-bootstrap"))]
fn main() {
  println!("This example requires the 'lut-bootstrap' feature to be enabled.");
  println!("Run with: cargo run --example lut_add_two_numbers --features lut-bootstrap --release");
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
  println!("║  Fast 8-bit Addition Using Programmable Bootstrapping         ║");
  println!("╚════════════════════════════════════════════════════════════════╝");
  println!();

  // Use 128-bit parameters for complex arithmetic (more stable for addition)
  let current_params = params::SECURITY_128_BIT;

  // Generate keys
  println!("⏱️  Generating keys...");
  let key_start = Instant::now();
  let secret_key = key::SecretKey::new();
  let cloud_key = key::CloudKey::new(&secret_key);
  let bootstrap = LutBootstrap::new();
  let key_duration = key_start.elapsed();
  println!("   Key generation completed in {:?}", key_duration);
  println!();

  // Inputs
  let a: u8 = 42;
  let b: u8 = 137;
  let expected: u8 = 179;

  println!("Computing: {} + {} = {} (encrypted)", a, b, expected);
  println!();

  // Step 1: Split into nibbles (4-bit chunks)
  let a_low = (a & 0x0F) as usize; // Low nibble of a (bits 0-3)
  let a_high = ((a >> 4) & 0x0F) as usize; // High nibble of a (bits 4-7)
  let b_low = (b & 0x0F) as usize; // Low nibble of b
  let b_high = ((b >> 4) & 0x0F) as usize; // High nibble of b

  println!(
    "Input A: {:3} = 0b{:04b}_{:04b} (nibbles: high={}, low={})",
    a, a_high, a_low, a_high, a_low
  );
  println!(
    "Input B: {:3} = 0b{:04b}_{:04b} (nibbles: high={}, low={})",
    b, b_high, b_low, b_high, b_low
  );
  println!();

  // Step 2: Generate lookup tables
  println!("📋 Generating lookup tables...");
  let lut_start = Instant::now();

  // Use message modulus 32 to handle full range of possible sums (0-30)
  let gen = Generator::new(32);

  // LUT for extracting low 4 bits (sum mod 16)
  let lut_sum_low = gen.generate_lookup_table(|x| x % 16);

  // LUT for extracting carry bit (1 if x >= 16, 0 otherwise)
  let lut_carry_low = gen.generate_lookup_table(|x| if x >= 16 { 1 } else { 0 });

  // LUT for extracting high sum (mod 16)
  let lut_sum_high = gen.generate_lookup_table(|x| x % 16);

  let lut_duration = lut_start.elapsed();
  println!("   LUT generation: {:?}", lut_duration);
  println!();

  // Step 3: Encrypt nibbles
  println!("🔒 Encrypting nibbles...");
  let enc_start = Instant::now();

  let ct_a_low = Ciphertext::encrypt_lwe_message(
    a_low,
    32,
    current_params.tlwe_lv0.alpha,
    &secret_key.key_lv0,
  );
  let ct_a_high = Ciphertext::encrypt_lwe_message(
    a_high,
    32,
    current_params.tlwe_lv0.alpha,
    &secret_key.key_lv0,
  );
  let ct_b_low = Ciphertext::encrypt_lwe_message(
    b_low,
    32,
    current_params.tlwe_lv0.alpha,
    &secret_key.key_lv0,
  );
  let ct_b_high = Ciphertext::encrypt_lwe_message(
    b_high,
    32,
    current_params.tlwe_lv0.alpha,
    &secret_key.key_lv0,
  );

  let enc_duration = enc_start.elapsed();
  println!("   Encrypted 4 nibbles in {:?}", enc_duration);
  println!();

  // Step 4: Homomorphic addition of low nibbles (no bootstrap needed!)
  println!("➕ Computing encrypted addition...");
  let add_start = Instant::now();

  // Add low nibbles homomorphically
  let ct_temp_low = &ct_a_low + &ct_b_low;
  println!("   Step 1: Low nibbles added (homomorphic add, no bootstrap)");

  // Step 5: Bootstrap 1 - Extract low sum (mod 16)
  let pbs1_start = Instant::now();
  let ct_sum_low = bootstrap.bootstrap_lut(&ct_temp_low, &lut_sum_low, &cloud_key);
  let pbs1_duration = pbs1_start.elapsed();
  println!(
    "   Bootstrap 1: Extract low sum (mod 16) - {:?}",
    pbs1_duration
  );

  // Step 6: Bootstrap 2 - Extract carry from low nibbles
  let pbs2_start = Instant::now();
  let ct_carry = bootstrap.bootstrap_lut(&ct_temp_low, &lut_carry_low, &cloud_key);
  let pbs2_duration = pbs2_start.elapsed();
  println!("   Bootstrap 2: Extract carry bit - {:?}", pbs2_duration);

  // Step 7: Add high nibbles + carry (homomorphic)
  let ct_temp_high = &(&ct_a_high + &ct_b_high) + &ct_carry;
  println!("   Step 2: High nibbles + carry added (homomorphic add, no bootstrap)");

  // Step 8: Bootstrap 3 - Extract high sum (mod 16)
  let pbs3_start = Instant::now();
  let ct_sum_high = bootstrap.bootstrap_lut(&ct_temp_high, &lut_sum_high, &cloud_key);
  let pbs3_duration = pbs3_start.elapsed();
  println!(
    "   Bootstrap 3: Extract high sum (mod 16) - {:?}",
    pbs3_duration
  );

  let add_duration = add_start.elapsed();
  println!();

  // Step 9: Decrypt results
  println!("🔓 Decrypting result...");
  let dec_start = Instant::now();

  let sum_low = ct_sum_low.decrypt_lwe_message(32, &secret_key.key_lv0);
  let sum_high = ct_sum_high.decrypt_lwe_message(32, &secret_key.key_lv0);
  let carry = ct_carry.decrypt_lwe_message(32, &secret_key.key_lv0);

  let dec_duration = dec_start.elapsed();
  println!("   Decrypted nibbles in {:?}", dec_duration);
  println!();

  // Debug output
  println!("🔍 Debug Information:");
  println!(
    "   Low nibble sum: {} (expected: {})",
    sum_low,
    (a_low + b_low) % 16
  );
  println!(
    "   Carry: {} (expected: {})",
    carry,
    if (a_low + b_low) >= 16 { 1 } else { 0 }
  );
  println!(
    "   High nibble sum: {} (expected: {})",
    sum_high,
    (a_high + b_high + carry) % 16
  );
  println!();

  // Step 10: Combine nibbles into final result
  let result = (sum_low | (sum_high << 4)) as u8;

  println!("═══════════════════════════════════════════════════════════════");
  println!("RESULTS");
  println!("═══════════════════════════════════════════════════════════════");
  println!("Input A:    {:3} = 0b{:04b}_{:04b}", a, a_high, a_low);
  println!("Input B:    {:3} = 0b{:04b}_{:04b}", b, b_high, b_low);
  println!(
    "Result:     {:3} = 0b{:04b}_{:04b} (nibbles: high={}, low={})",
    result, sum_high, sum_low, sum_high, sum_low
  );
  println!("Expected:   {:3}", expected);
  println!();

  if result == expected {
    println!("✅ SUCCESS! Result is correct!");
  } else {
    println!("❌ FAILURE! Expected {}, got {}", expected, result);
  }

  println!();
  println!("═══════════════════════════════════════════════════════════════");
  println!("PERFORMANCE SUMMARY");
  println!("═══════════════════════════════════════════════════════════════");
  println!("Key Generation:  {:?}", key_duration);
  println!("LUT Generation:  {:?}", lut_duration);
  println!("Encryption:      {:?} (4 nibbles)", enc_duration);
  println!("Addition:        {:?} (3 bootstraps)", add_duration);
  println!("  - Bootstrap 1: {:?} (low sum)", pbs1_duration);
  println!("  - Bootstrap 2: {:?} (carry)", pbs2_duration);
  println!("  - Bootstrap 3: {:?} (high sum)", pbs3_duration);
  println!("Decryption:      {:?}", dec_duration);
  println!();

  // Performance comparison
  println!("═══════════════════════════════════════════════════════════════");
  println!("PERFORMANCE COMPARISON");
  println!("═══════════════════════════════════════════════════════════════");
  println!("Traditional bit-by-bit addition: 8 bootstraps");
  println!("LUT-based nibble addition:       3 bootstraps");
  println!("Speedup:                         {:.1}x", 8.0 / 3.0);
  println!("═══════════════════════════════════════════════════════════════");
}
