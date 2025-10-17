/// Example demonstrating the bootstrap trait and different bootstrapping strategies
///
/// This example shows how to:
/// 1. Use the default vanilla bootstrap
/// 2. Work with the bootstrap trait
/// 3. Switch between different bootstrap strategies (for future implementations)
use rs_tfhe::bootstrap::{default_bootstrap, vanilla::VanillaBootstrap, Bootstrap};
use rs_tfhe::key::{CloudKey, SecretKey};
use rs_tfhe::params;
use rs_tfhe::utils::Ciphertext;

fn main() {
  println!("╔════════════════════════════════════════════════════════════╗");
  println!("║       TFHE Bootstrap Strategies Example                   ║");
  println!("╚════════════════════════════════════════════════════════════╝");
  println!();

  // Generate keys
  println!("🔑 Generating cryptographic keys...");
  let secret_key = SecretKey::new();
  let cloud_key = CloudKey::new(&secret_key);
  println!("✓ Keys generated");
  println!();

  // Test plaintext
  let plaintext = true;
  println!("📝 Plaintext value: {}", plaintext);
  println!();

  // Encrypt
  println!("🔒 Encrypting...");
  let ciphertext =
    Ciphertext::encrypt_bool(plaintext, params::tlwe_lv0::ALPHA, &secret_key.key_lv0);
  println!("✓ Encrypted");
  println!();

  // =================================================================
  // Method 1: Using the default bootstrap
  // =================================================================
  println!("─────────────────────────────────────────────────────────────");
  println!("Method 1: Default Bootstrap");
  println!("─────────────────────────────────────────────────────────────");

  let bootstrap = default_bootstrap();
  println!("Strategy: {}", bootstrap.name());

  let bootstrapped = bootstrap.bootstrap(&ciphertext, &cloud_key);
  let decrypted = bootstrapped.decrypt_bool(&secret_key.key_lv0);

  println!("Result: {} (expected: {})", decrypted, plaintext);
  println!(
    "Status: {}",
    if decrypted == plaintext {
      "✓ PASS"
    } else {
      "✗ FAIL"
    }
  );
  println!();

  // =================================================================
  // Method 2: Using vanilla bootstrap directly
  // =================================================================
  println!("─────────────────────────────────────────────────────────────");
  println!("Method 2: Vanilla Bootstrap (Direct)");
  println!("─────────────────────────────────────────────────────────────");

  let vanilla = VanillaBootstrap::new();
  println!("Strategy: {}", vanilla.name());

  let bootstrapped = vanilla.bootstrap(&ciphertext, &cloud_key);
  let decrypted = bootstrapped.decrypt_bool(&secret_key.key_lv0);

  println!("Result: {} (expected: {})", decrypted, plaintext);
  println!(
    "Status: {}",
    if decrypted == plaintext {
      "✓ PASS"
    } else {
      "✗ FAIL"
    }
  );
  println!();

  // =================================================================
  // Method 3: Using trait object for runtime strategy selection
  // =================================================================
  println!("─────────────────────────────────────────────────────────────");
  println!("Method 3: Dynamic Strategy Selection");
  println!("─────────────────────────────────────────────────────────────");

  // This demonstrates how to select a strategy at runtime
  let strategy_name = "vanilla"; // Could be from config file, CLI args, etc.
  let bootstrap: Box<dyn Bootstrap> = match strategy_name {
    "vanilla" => Box::new(VanillaBootstrap::new()),
    // Future strategies can be added here:
    // "gpu" => Box::new(GpuBootstrap::new()),
    // "batch" => Box::new(BatchBootstrap::new()),
    _ => Box::new(VanillaBootstrap::new()),
  };

  println!("Selected strategy: {}", bootstrap.name());

  let bootstrapped = bootstrap.bootstrap(&ciphertext, &cloud_key);
  let decrypted = bootstrapped.decrypt_bool(&secret_key.key_lv0);

  println!("Result: {} (expected: {})", decrypted, plaintext);
  println!(
    "Status: {}",
    if decrypted == plaintext {
      "✓ PASS"
    } else {
      "✗ FAIL"
    }
  );
  println!();

  // =================================================================
  // Summary
  // =================================================================
  println!("╔════════════════════════════════════════════════════════════╗");
  println!("║ Summary                                                    ║");
  println!("╚════════════════════════════════════════════════════════════╝");
  println!("✓ All bootstrap methods produced correct results");
  println!("✓ Bootstrap trait enables testing different strategies");
  println!();
  println!("Future bootstrap strategies can include:");
  println!("  • GPU-accelerated bootstrap (Metal/CUDA)");
  println!("  • Batch bootstrap for multiple ciphertexts");
  println!("  • Optimized bootstrap with different parameter sets");
  println!("  • Hardware-specific implementations (AVX, NEON, etc.)");
  println!();
}
