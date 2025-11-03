//! Proxy Reencryption Example
//!
//! This example demonstrates how to use LWE proxy reencryption to securely
//! delegate access to encrypted data without decryption.
//!
//! Run with:
//! ```bash
//! cargo run --example proxy_reencryption_demo --features "proxy-reenc" --release
//! ```

#[cfg(feature = "proxy-reenc")]
use rs_tfhe::key::SecretKey;
#[cfg(feature = "proxy-reenc")]
use rs_tfhe::params;
#[cfg(feature = "proxy-reenc")]
use rs_tfhe::proxy_reenc::{reencrypt_tlwe_lv0, ProxyReencryptionKey, PublicKeyLv0};
#[cfg(feature = "proxy-reenc")]
use rs_tfhe::tlwe::TLWELv0;

#[cfg(not(feature = "proxy-reenc"))]
fn main() {
  println!("This example requires the 'proxy-reenc' feature.");
  println!(
    "Run with: cargo run --example proxy_reencryption_demo --features \"proxy-reenc\" --release"
  );
}

#[cfg(feature = "proxy-reenc")]
fn main() {
  println!("=== LWE Proxy Reencryption Demo ===\n");

  // Scenario: Alice wants to share encrypted data with Bob
  // without decrypting it, using a semi-trusted proxy

  println!("1. Setting up keys for Alice and Bob...");
  let alice_key = SecretKey::new();
  let bob_key = SecretKey::new();
  println!("   ✓ Alice's secret key generated");

  // Bob publishes his public key
  let start = std::time::Instant::now();
  let bob_public_key = PublicKeyLv0::new(&bob_key.key_lv0);
  let pubkey_time = start.elapsed();
  println!("   ✓ Bob's public key generated in {:.2?}", pubkey_time);
  println!("   ✓ Bob shares his public key (safe to publish)\n");

  // Alice encrypts some data
  println!("2. Alice encrypts her data...");
  let messages = vec![true, false, true, true, false];
  let alice_ciphertexts: Vec<TLWELv0> = messages
    .iter()
    .map(|&msg| TLWELv0::encrypt_bool(msg, params::tlwe_lv0::ALPHA, &alice_key.key_lv0))
    .collect();

  println!("   Messages encrypted by Alice:");
  for (i, &msg) in messages.iter().enumerate() {
    println!("   - Message {}: {}", i + 1, msg);
  }
  println!();

  // Alice generates a proxy reencryption key using Bob's PUBLIC key
  println!("3. Alice generates a proxy reencryption key (Alice -> Bob)...");
  println!("   Using ASYMMETRIC mode - Bob's secret key is NOT needed!");
  let start = std::time::Instant::now();
  let reenc_key = ProxyReencryptionKey::new_asymmetric(&alice_key.key_lv0, &bob_public_key);
  let keygen_time = start.elapsed();
  println!("   ✓ Reencryption key generated in {:.2?}", keygen_time);
  println!("   ✓ Alice shares this key with the proxy\n");

  // Proxy reencrypts the data (without learning the plaintext)
  println!("4. Proxy converts Alice's ciphertexts to Bob's ciphertexts...");
  let start = std::time::Instant::now();
  let bob_ciphertexts: Vec<TLWELv0> = alice_ciphertexts
    .iter()
    .map(|ct| reencrypt_tlwe_lv0(ct, &reenc_key))
    .collect();
  let reenc_time = start.elapsed();
  println!(
    "   ✓ {} ciphertexts reencrypted in {:.2?}",
    bob_ciphertexts.len(),
    reenc_time
  );
  println!(
    "   ✓ Average time per reencryption: {:.2?}\n",
    reenc_time / bob_ciphertexts.len() as u32
  );

  // Bob decrypts the reencrypted data
  println!("5. Bob decrypts the reencrypted data...");
  let mut correct = 0;
  let decrypted_messages: Vec<bool> = bob_ciphertexts
    .iter()
    .map(|ct| ct.decrypt_bool(&bob_key.key_lv0))
    .collect();

  println!("   Decrypted messages:");
  for (i, (&original, &decrypted)) in messages.iter().zip(decrypted_messages.iter()).enumerate() {
    let status = if original == decrypted {
      correct += 1;
      "✓"
    } else {
      "✗"
    };
    println!(
      "   {} Message {}: {} (original: {})",
      status,
      i + 1,
      decrypted,
      original
    );
  }
  println!();

  println!("=== Results ===");
  println!(
    "Accuracy: {}/{} ({:.1}%)",
    correct,
    messages.len(),
    (correct as f64 / messages.len() as f64) * 100.0
  );
  println!();

  // Demonstrate multi-hop reencryption: Alice -> Bob -> Carol
  println!("\n=== Multi-Hop Reencryption Demo (Asymmetric) ===\n");
  println!("Demonstrating a chain: Alice -> Bob -> Carol");
  println!("Each party only needs the next party's PUBLIC key\n");

  let carol_key = SecretKey::new();
  let carol_public_key = PublicKeyLv0::new(&carol_key.key_lv0);
  println!("1. Carol's keys generated and public key published");

  let reenc_key_bc = ProxyReencryptionKey::new_asymmetric(&bob_key.key_lv0, &carol_public_key);
  println!("2. Generated reencryption key (Bob -> Carol) using Carol's PUBLIC key");

  let test_message = true;
  let alice_ct = TLWELv0::encrypt_bool(test_message, params::tlwe_lv0::ALPHA, &alice_key.key_lv0);
  println!("3. Alice encrypts message: {}", test_message);

  let bob_ct = reencrypt_tlwe_lv0(&alice_ct, &reenc_key);
  println!("4. Proxy reencrypts Alice -> Bob");
  let bob_decrypted = bob_ct.decrypt_bool(&bob_key.key_lv0);
  println!(
    "   Bob decrypts: {} {}",
    bob_decrypted,
    if bob_decrypted == test_message {
      "✓"
    } else {
      "✗"
    }
  );

  let carol_ct = reencrypt_tlwe_lv0(&bob_ct, &reenc_key_bc);
  println!("5. Proxy reencrypts Bob -> Carol");
  let carol_decrypted = carol_ct.decrypt_bool(&carol_key.key_lv0);
  println!(
    "   Carol decrypts: {} {}",
    carol_decrypted,
    if carol_decrypted == test_message {
      "✓"
    } else {
      "✗"
    }
  );

  println!();
  println!("=== Security Notes ===");
  println!("• The proxy never learns the plaintext");
  println!("• Bob's secret key is NEVER shared - only his public key is used");
  println!("• The reencryption key only works in one direction");
  println!("• Each reencryption adds a small amount of noise");
  println!("• The scheme is unidirectional (Alice->Bob key ≠ Bob->Alice key)");
  println!("• True asymmetric proxy reencryption with LWE-based public keys");

  println!("\n=== Performance Summary ===");
  println!("Bob's public key generation: {:.2?}", pubkey_time);
  println!("Reencryption key generation: {:.2?}", keygen_time);
  println!(
    "Average reencryption time: {:.2?}",
    reenc_time / bob_ciphertexts.len() as u32
  );
}
