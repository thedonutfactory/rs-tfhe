//! Simple Bootstrap Between Reencryptions Example
//!
//! Demonstrates using bootstrapping to refresh noise in a 3-hop proxy reencryption chain.
//! This is a simplified version focused on showing the key concept.
//!
//! Run with:
//! ```bash
//! cargo run --example bootstrap_between_reenc --features "proxy-reenc,bootstrapping" --release
//! ```

#[cfg(all(feature = "proxy-reenc", feature = "bootstrapping"))]
use rs_tfhe::bootstrap::default_bootstrap;
#[cfg(all(feature = "proxy-reenc", feature = "bootstrapping"))]
use rs_tfhe::key::{CloudKey, SecretKey};
#[cfg(all(feature = "proxy-reenc", feature = "bootstrapping"))]
use rs_tfhe::params;
#[cfg(all(feature = "proxy-reenc", feature = "bootstrapping"))]
use rs_tfhe::proxy_reenc::{reencrypt_tlwe_lv0, ProxyReencryptionKey};
#[cfg(all(feature = "proxy-reenc", feature = "bootstrapping"))]
use rs_tfhe::tlwe::TLWELv0;

#[cfg(not(all(feature = "proxy-reenc", feature = "bootstrapping")))]
fn main() {
    println!("This example requires both 'proxy-reenc' and 'bootstrapping' features.");
    println!(
        "Run with: cargo run --example bootstrap_between_reenc --features \"proxy-reenc,bootstrapping\" --release"
    );
}

#[cfg(all(feature = "proxy-reenc", feature = "bootstrapping"))]
fn main() {
    println!("=== Bootstrap Between Reencryptions Demo ===\n");
    println!("Showing how bootstrapping refreshes noise in multi-hop chains\n");
    
    // Using symmetric mode for faster key generation (demonstration purposes)
    println!("Setting up 3-party chain (Alice → Bob → Carol)...");
    let alice_key = SecretKey::new();
    let bob_key = SecretKey::new();
    let carol_key = SecretKey::new();
    println!("✓ Keys generated\n");

    // Generate reencryption keys (symmetric mode is faster for demo)
    println!("Generating reencryption keys (symmetric mode)...");
    let start = std::time::Instant::now();
    let reenc_ab = ProxyReencryptionKey::new_symmetric(&alice_key.key_lv0, &bob_key.key_lv0);
    let reenc_bc = ProxyReencryptionKey::new_symmetric(&bob_key.key_lv0, &carol_key.key_lv0);
    println!("✓ Reencryption keys generated in {:.2?}\n", start.elapsed());

    // Generate cloud key for Bob (needed for bootstrapping)
    println!("Generating Bob's cloud key for bootstrapping...");
    let start = std::time::Instant::now();
    let bob_cloud = CloudKey::new(&bob_key);
    println!("✓ Cloud key generated in {:.2?}\n", start.elapsed());

    let bootstrap_strategy = default_bootstrap();
    let message = true;

    println!("Original message: {}\n", message);

    // ========================================================================
    // Scenario: Alice → Bob → Carol (3 hops)
    // ========================================================================
    
    println!("=== WITHOUT Bootstrapping ===\n");
    
    let alice_ct = TLWELv0::encrypt_bool(message, params::tlwe_lv0::ALPHA, &alice_key.key_lv0);
    println!("1. Alice encrypts: {}", message);
    
    let start = std::time::Instant::now();
    let bob_ct = reencrypt_tlwe_lv0(&alice_ct, &reenc_ab);
    let t1 = start.elapsed();
    let bob_result = bob_ct.decrypt_bool(&bob_key.key_lv0);
    println!("2. Alice → Bob: {} {} ({:.2?})", 
        bob_result, if bob_result == message { "✓" } else { "✗" }, t1);
    
    let start = std::time::Instant::now();
    let carol_ct = reencrypt_tlwe_lv0(&bob_ct, &reenc_bc);
    let t2 = start.elapsed();
    let carol_result = carol_ct.decrypt_bool(&carol_key.key_lv0);
    println!("3. Bob → Carol: {} {} ({:.2?})", 
        carol_result, if carol_result == message { "✓" } else { "✗" }, t2);
    
    println!("\nFinal result: {} {}\n", 
        carol_result, if carol_result == message { "✓" } else { "✗" });

    // ========================================================================
    // WITH Bootstrapping between hops
    // ========================================================================
    
    println!("=== WITH Bootstrapping (noise refresh at Bob) ===\n");
    
    let alice_ct = TLWELv0::encrypt_bool(message, params::tlwe_lv0::ALPHA, &alice_key.key_lv0);
    println!("1. Alice encrypts: {}", message);
    
    let start = std::time::Instant::now();
    let bob_ct = reencrypt_tlwe_lv0(&alice_ct, &reenc_ab);
    let t1 = start.elapsed();
    
    // Bootstrap to refresh noise
    let start = std::time::Instant::now();
    let bob_ct_refreshed = bootstrap_strategy.bootstrap(&bob_ct, &bob_cloud);
    let bootstrap_time = start.elapsed();
    
    let bob_result = bob_ct_refreshed.decrypt_bool(&bob_key.key_lv0);
    println!("2. Alice → Bob: {} {} ({:.2?})", 
        bob_result, if bob_result == message { "✓" } else { "✗" }, t1);
    println!("   └─ Bootstrap refresh: {:.2?}", bootstrap_time);
    
    let start = std::time::Instant::now();
    let carol_ct = reencrypt_tlwe_lv0(&bob_ct_refreshed, &reenc_bc);
    let t2 = start.elapsed();
    let carol_result = carol_ct.decrypt_bool(&carol_key.key_lv0);
    println!("3. Bob → Carol: {} {} ({:.2?})", 
        carol_result, if carol_result == message { "✓" } else { "✗" }, t2);
    
    println!("\nFinal result: {} {}\n", 
        carol_result, if carol_result == message { "✓" } else { "✗" });

    // ========================================================================
    // Statistical test
    // ========================================================================
    
    println!("=== Statistical Test (50 random messages) ===\n");
    
    let mut without_correct = 0;
    let mut with_correct = 0;
    let iterations = 50;
    
    for i in 0..iterations {
        let msg = (i % 2) == 0;
        
        // Without bootstrap
        let ct = TLWELv0::encrypt_bool(msg, params::tlwe_lv0::ALPHA, &alice_key.key_lv0);
        let ct = reencrypt_tlwe_lv0(&ct, &reenc_ab);
        let ct = reencrypt_tlwe_lv0(&ct, &reenc_bc);
        if ct.decrypt_bool(&carol_key.key_lv0) == msg {
            without_correct += 1;
        }
        
        // With bootstrap
        let ct = TLWELv0::encrypt_bool(msg, params::tlwe_lv0::ALPHA, &alice_key.key_lv0);
        let ct = reencrypt_tlwe_lv0(&ct, &reenc_ab);
        let ct = bootstrap_strategy.bootstrap(&ct, &bob_cloud);
        let ct = reencrypt_tlwe_lv0(&ct, &reenc_bc);
        if ct.decrypt_bool(&carol_key.key_lv0) == msg {
            with_correct += 1;
        }
    }
    
    println!("Without bootstrapping: {}/{} correct ({:.1}%)", 
        without_correct, iterations, 
        (without_correct as f64 / iterations as f64) * 100.0);
    println!("With bootstrapping:    {}/{} correct ({:.1}%)", 
        with_correct, iterations,
        (with_correct as f64 / iterations as f64) * 100.0);

    println!("\n=== Key Takeaways ===\n");
    println!("✓ Bootstrapping refreshes noise accumulated during reencryption");
    println!("✓ Essential for maintaining accuracy in multi-hop chains (>2-3 hops)");
    println!("✓ Trade-off: ~10-50ms per bootstrap vs. potential decryption errors");
    println!("✓ Each party needs a CloudKey to bootstrap ciphertexts under their key");
    
    println!("\n=== Usage Pattern ===\n");
    println!("```rust");
    println!("// After reencryption, before passing to next party:");
    println!("let ct_reencrypted = reencrypt_tlwe_lv0(&ct, &reenc_key);");
    println!("let ct_refreshed = bootstrap.bootstrap(&ct_reencrypted, &cloud_key);");
    println!("// ct_refreshed now has low noise, safe for another hop");
    println!("```");
}

