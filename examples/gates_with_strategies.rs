/// Example demonstrating the refactored Gates struct with configurable bootstrap strategies
///
/// This example shows:
/// 1. Creating gates with default bootstrap
/// 2. Creating gates with a specific bootstrap strategy
/// 3. Comparing performance and flexibility
/// 4. Using both struct-based and convenience function APIs
use rs_tfhe::bootstrap::vanilla::VanillaBootstrap;
use rs_tfhe::gates::{self, Gates};
use rs_tfhe::key::{CloudKey, SecretKey};
use rs_tfhe::params;
use rs_tfhe::utils::Ciphertext;

fn main() {
  println!("╔════════════════════════════════════════════════════════════╗");
  println!("║       TFHE Gates with Bootstrap Strategies                ║");
  println!("╚════════════════════════════════════════════════════════════╝");
  println!();

  // Generate keys
  println!("🔑 Generating cryptographic keys...");
  let secret_key = SecretKey::new();
  let cloud_key = CloudKey::new(&secret_key);
  println!("✓ Keys generated");
  println!();

  // Test inputs
  let a = true;
  let b = false;
  println!("📝 Test inputs: a={}, b={}", a, b);
  println!();

  // Encrypt
  println!("🔒 Encrypting inputs...");
  let ct_a = Ciphertext::encrypt_bool(a, params::tlwe_lv0::ALPHA, &secret_key.key_lv0);
  let ct_b = Ciphertext::encrypt_bool(b, params::tlwe_lv0::ALPHA, &secret_key.key_lv0);
  println!("✓ Inputs encrypted");
  println!();

  // =================================================================
  // Method 1: Using Gates struct with default bootstrap
  // =================================================================
  println!("─────────────────────────────────────────────────────────────");
  println!("Method 1: Gates Struct (Default Bootstrap)");
  println!("─────────────────────────────────────────────────────────────");

  let gates = Gates::new();
  println!("Bootstrap strategy: {}", gates.bootstrap_strategy());
  println!();

  // Perform various gate operations
  println!("Testing gates:");

  let ct_and = gates.and(&ct_a, &ct_b, &cloud_key);
  let result_and = ct_and.decrypt_bool(&secret_key.key_lv0);
  println!("  AND:  {} AND {} = {} ✓", a, b, result_and);
  assert_eq!(result_and, a & b);

  let ct_or = gates.or(&ct_a, &ct_b, &cloud_key);
  let result_or = ct_or.decrypt_bool(&secret_key.key_lv0);
  println!("  OR:   {} OR  {} = {} ✓", a, b, result_or);
  assert_eq!(result_or, a | b);

  let ct_xor = gates.xor(&ct_a, &ct_b, &cloud_key);
  let result_xor = ct_xor.decrypt_bool(&secret_key.key_lv0);
  println!("  XOR:  {} XOR {} = {} ✓", a, b, result_xor);
  assert_eq!(result_xor, a ^ b);

  let ct_not = gates.not(&ct_a);
  let result_not = ct_not.decrypt_bool(&secret_key.key_lv0);
  println!("  NOT:  NOT {} = {} ✓", a, result_not);
  assert_eq!(result_not, !a);
  println!();

  // =================================================================
  // Method 2: Gates struct with explicit bootstrap strategy
  // =================================================================
  println!("─────────────────────────────────────────────────────────────");
  println!("Method 2: Gates Struct (Explicit Strategy)");
  println!("─────────────────────────────────────────────────────────────");

  let vanilla_bootstrap = Box::new(VanillaBootstrap::new());
  let gates_vanilla = Gates::with_bootstrap(vanilla_bootstrap);
  println!("Bootstrap strategy: {}", gates_vanilla.bootstrap_strategy());
  println!();

  let ct_nand = gates_vanilla.nand(&ct_a, &ct_b, &cloud_key);
  let result_nand = ct_nand.decrypt_bool(&secret_key.key_lv0);
  println!("  NAND: {} NAND {} = {} ✓", a, b, result_nand);
  assert_eq!(result_nand, !(a & b));

  let ct_nor = gates_vanilla.nor(&ct_a, &ct_b, &cloud_key);
  let result_nor = ct_nor.decrypt_bool(&secret_key.key_lv0);
  println!("  NOR:  {} NOR  {} = {} ✓", a, b, result_nor);
  assert_eq!(result_nor, !(a | b));
  println!();

  // =================================================================
  // Method 3: Convenience free functions (backward compatible)
  // =================================================================
  println!("─────────────────────────────────────────────────────────────");
  println!("Method 3: Convenience Functions (Backward Compatible)");
  println!("─────────────────────────────────────────────────────────────");
  println!("Using free functions that create Gates internally");
  println!();

  let ct_and_free = gates::and(&ct_a, &ct_b, &cloud_key);
  let result_and_free = ct_and_free.decrypt_bool(&secret_key.key_lv0);
  println!("  AND:  {} AND {} = {} ✓", a, b, result_and_free);
  assert_eq!(result_and_free, a & b);

  let ct_or_free = gates::or(&ct_a, &ct_b, &cloud_key);
  let result_or_free = ct_or_free.decrypt_bool(&secret_key.key_lv0);
  println!("  OR:   {} OR  {} = {} ✓", a, b, result_or_free);
  assert_eq!(result_or_free, a | b);

  let ct_not_free = gates::not(&ct_a);
  let result_not_free = ct_not_free.decrypt_bool(&secret_key.key_lv0);
  println!("  NOT:  NOT {} = {} ✓", a, result_not_free);
  assert_eq!(result_not_free, !a);
  println!();

  // =================================================================
  // Method 4: MUX gate (complex operation)
  // =================================================================
  println!("─────────────────────────────────────────────────────────────");
  println!("Method 4: Complex Gates (MUX)");
  println!("─────────────────────────────────────────────────────────────");

  let c = true;
  let ct_c = Ciphertext::encrypt_bool(c, params::tlwe_lv0::ALPHA, &secret_key.key_lv0);

  let ct_mux = gates.mux(&ct_a, &ct_b, &ct_c, &cloud_key);
  let result_mux = ct_mux.decrypt_bool(&secret_key.key_lv0);
  let expected_mux = (a & b) | (!a & c);

  println!("  MUX:  {}?{}:{} = {} ✓", a, b, c, result_mux);
  println!("        (a?b:c = (a AND b) OR (NOT a AND c))");
  assert_eq!(result_mux, expected_mux);
  println!();

  // =================================================================
  // Summary
  // =================================================================
  println!("╔════════════════════════════════════════════════════════════╗");
  println!("║ Summary                                                    ║");
  println!("╚════════════════════════════════════════════════════════════╝");
  println!();
  println!("✅ Benefits of refactored Gates struct:");
  println!();
  println!("  1. Configurable Bootstrap Strategy");
  println!("     • Create gates with any bootstrap implementation");
  println!("     • Test performance of different strategies");
  println!("     • Switch strategies at runtime");
  println!();
  println!("  2. Backward Compatibility");
  println!("     • Existing code using free functions still works");
  println!("     • Free functions use default bootstrap internally");
  println!("     • No breaking changes for current users");
  println!();
  println!("  3. Better Organization");
  println!("     • Gates logically grouped in a struct");
  println!("     • Clear ownership of bootstrap strategy");
  println!("     • Easier to add new gate operations");
  println!();
  println!("  4. Future Extensibility");
  println!("     • Easy to add GPU-accelerated gates");
  println!("     • Can implement batch-optimized gates");
  println!("     • Enables per-gate strategy customization");
  println!();
  println!("All gate operations completed successfully! 🎉");
  println!();
}
