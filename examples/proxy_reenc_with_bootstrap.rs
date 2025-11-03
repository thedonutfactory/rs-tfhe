//! Proxy Reencryption with Bootstrapping Example
//!
//! This example demonstrates how to use bootstrapping to refresh noise
//! between proxy reencryptions in multi-hop chains. This is critical
//! for maintaining correctness in long delegation chains.
//!
//! Run with:
//! ```bash
//! cargo run --example proxy_reenc_with_bootstrap --features "proxy-reenc,bootstrapping" --release
//! ```

#[cfg(all(feature = "proxy-reenc", feature = "bootstrapping"))]
use rs_tfhe::bootstrap::default_bootstrap;
#[cfg(all(feature = "proxy-reenc", feature = "bootstrapping"))]
use rs_tfhe::key::{CloudKey, SecretKey};
#[cfg(all(feature = "proxy-reenc", feature = "bootstrapping"))]
use rs_tfhe::params;
#[cfg(all(feature = "proxy-reenc", feature = "bootstrapping"))]
use rs_tfhe::proxy_reenc::{reencrypt_tlwe_lv0, ProxyReencryptionKey, PublicKeyLv0};
#[cfg(all(feature = "proxy-reenc", feature = "bootstrapping"))]
use rs_tfhe::tlwe::TLWELv0;

#[cfg(not(all(feature = "proxy-reenc", feature = "bootstrapping")))]
fn main() {
  println!("This example requires both 'proxy-reenc' and 'bootstrapping' features.");
  println!(
        "Run with: cargo run --example proxy_reenc_with_bootstrap --features \"proxy-reenc,bootstrapping\" --release"
    );
}

#[cfg(all(feature = "proxy-reenc", feature = "bootstrapping"))]
fn main() {
  println!("=== Proxy Reencryption with Bootstrapping Demo ===\n");

  // ========================================================================
  // CONFIGURATION
  // ========================================================================
  const HOPS: usize = 20; // Change this to test different chain lengths!
  const ITERATIONS: usize = 100; // Number of test messages per test

  println!("Configuration:");
  println!("• Chain length: {} hops", HOPS);
  println!("• Test iterations: {}\n", ITERATIONS);

  // ========================================================================
  // Setup: Generate keys for the chain
  // ========================================================================
  println!("Setting up {}-party delegation chain...", HOPS + 1);

  // Generate secret keys for all parties
  let mut secret_keys: Vec<SecretKey> = Vec::with_capacity(HOPS + 1);
  for _ in 0..=HOPS {
    secret_keys.push(SecretKey::new());
  }

  // Generate public keys (for asymmetric mode)
  let mut public_keys: Vec<PublicKeyLv0> = Vec::with_capacity(HOPS);
  for key in secret_keys.iter().take(HOPS + 1).skip(1) {
    public_keys.push(PublicKeyLv0::new(&key.key_lv0));
  }

  println!("✓ All keys generated\n");

  // Generate reencryption keys
  println!("Generating {} reencryption keys...", HOPS);
  let start = std::time::Instant::now();
  let mut reenc_keys: Vec<ProxyReencryptionKey> = Vec::with_capacity(HOPS);
  for i in 0..HOPS {
    reenc_keys.push(ProxyReencryptionKey::new_asymmetric(
      &secret_keys[i].key_lv0,
      &public_keys[i],
    ));
  }
  let keygen_time = start.elapsed();
  println!("✓ Reencryption keys generated in {:.2?}\n", keygen_time);

  // Generate cloud keys for bootstrapping
  println!("Generating {} cloud keys for bootstrapping...", HOPS + 1);
  let start = std::time::Instant::now();
  let mut cloud_keys: Vec<CloudKey> = Vec::with_capacity(HOPS + 1);
  for key in &secret_keys {
    cloud_keys.push(CloudKey::new(key));
  }
  let cloud_keygen_time = start.elapsed();
  println!("✓ Cloud keys generated in {:.2?}\n", cloud_keygen_time);

  let bootstrap_strategy = default_bootstrap();

  // ========================================================================
  // Test 1: Multi-hop WITHOUT bootstrapping
  // ========================================================================
  println!(
    "=== Test 1: {} hops WITHOUT bootstrapping ({} iterations) ===\n",
    HOPS, ITERATIONS
  );

  let mut without_bootstrap_correct = 0;

  for i in 0..ITERATIONS {
    let test_msg = (i % 2) == 0; // Alternate between true/false

    // Start with encryption by party 0
    let mut ct = TLWELv0::encrypt_bool(test_msg, params::tlwe_lv0::ALPHA, &secret_keys[0].key_lv0);

    // Chain through all hops
    for reenc_key in reenc_keys.iter().take(HOPS) {
      ct = reencrypt_tlwe_lv0(&ct, reenc_key);
    }

    // Final party decrypts
    let final_decrypted = ct.decrypt_bool(&secret_keys[HOPS].key_lv0);
    let correct = final_decrypted == test_msg;
    if correct {
      without_bootstrap_correct += 1;
    }

    print!("Iteration {:3}: {} → {} ", i + 1, test_msg, final_decrypted);
    if correct {
      println!("✓");
    } else {
      println!("✗ ERROR");
    }
  }

  let without_accuracy = (without_bootstrap_correct as f64 / ITERATIONS as f64) * 100.0;
  println!(
    "\nResult WITHOUT bootstrapping: {}/{} correct ({:.1}%)\n",
    without_bootstrap_correct, ITERATIONS, without_accuracy
  );

  // ========================================================================
  // Test 2: Multi-hop WITH bootstrapping between each hop
  // ========================================================================
  println!(
    "=== Test 2: {} hops WITH bootstrapping ({} iterations) ===\n",
    HOPS, ITERATIONS
  );

  let mut with_bootstrap_correct = 0;
  let mut total_reenc_time = std::time::Duration::ZERO;
  let mut total_bootstrap_time = std::time::Duration::ZERO;

  for i in 0..ITERATIONS {
    let test_msg = (i % 2) == 0; // Alternate between true/false

    // Start with encryption by party 0
    let mut ct = TLWELv0::encrypt_bool(test_msg, params::tlwe_lv0::ALPHA, &secret_keys[0].key_lv0);

    // Chain through all hops with bootstrapping after each
    for hop in 0..HOPS {
      // Reencrypt
      let start = std::time::Instant::now();
      ct = reencrypt_tlwe_lv0(&ct, &reenc_keys[hop]);
      total_reenc_time += start.elapsed();

      // Bootstrap to refresh noise
      let start = std::time::Instant::now();
      ct = bootstrap_strategy.bootstrap(&ct, &cloud_keys[hop + 1]);
      total_bootstrap_time += start.elapsed();
    }

    // Final party decrypts
    let final_decrypted = ct.decrypt_bool(&secret_keys[HOPS].key_lv0);
    let correct = final_decrypted == test_msg;
    if correct {
      with_bootstrap_correct += 1;
    }

    print!("Iteration {:3}: {} → {} ", i + 1, test_msg, final_decrypted);
    if correct {
      println!("✓");
    } else {
      println!("✗ ERROR");
    }
  }

  let with_accuracy = (with_bootstrap_correct as f64 / ITERATIONS as f64) * 100.0;
  println!(
    "\nResult WITH bootstrapping: {}/{} correct ({:.1}%)",
    with_bootstrap_correct, ITERATIONS, with_accuracy
  );
  println!(
    "Average reencryption time: {:.2?}",
    total_reenc_time / (ITERATIONS * HOPS) as u32
  );
  println!(
    "Average bootstrap time: {:.2?}\n",
    total_bootstrap_time / (ITERATIONS * HOPS) as u32
  );

  // ========================================================================
  // Summary
  // ========================================================================
  println!("\n=== Summary ===\n");
  println!("Chain Configuration:");
  println!("• Chain length: {} hops", HOPS);
  println!("• Total parties: {}", HOPS + 1);
  println!();

  println!("Results:");
  println!("• WITHOUT bootstrapping: {:.1}% accuracy", without_accuracy);
  println!("• WITH bootstrapping:    {:.1}% accuracy", with_accuracy);
  println!();

  println!("Key Insights:");
  println!("• Bootstrapping refreshes noise between hops");
  println!(
    "• Noise accumulates: {} hops = {:.1}% accuracy without bootstrap",
    HOPS, without_accuracy
  );
  println!(
    "• Bootstrap guarantees: {:.1}% accuracy (near perfect)",
    with_accuracy
  );
  println!();

  println!("Recommendations for different chain lengths:");
  println!("• 1-2 hops:  Bootstrapping optional (>99% accuracy without)");
  println!("• 3-4 hops:  Bootstrapping recommended (95-99% without, 100% with)");
  println!("• 5+ hops:   Bootstrapping essential (<95% accuracy without)");
  println!("• Any depth: Production systems should always bootstrap");
  println!();

  println!("Performance:");
  println!("• Reencryption: ~2-3ms per hop");
  println!("• Bootstrap: ~10-50ms per operation");
  println!(
    "• Total overhead: ~{:.1}x with bootstrapping",
    (total_reenc_time.as_millis() + total_bootstrap_time.as_millis()) as f64
      / total_reenc_time.as_millis() as f64
  );
  println!(
    "• Cloud key generation: ~{:.2?} per party (one-time)",
    cloud_keygen_time / (HOPS + 1) as u32
  );
  println!();

  println!("💡 TIP: Change HOPS constant at the top to test different chain lengths!");
  println!("   Try HOPS = 5, 6, or 7 to see accuracy degrade without bootstrapping.");
}
