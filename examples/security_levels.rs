/// Example demonstrating different security levels and their performance characteristics
///
/// This example shows how to compile with different security parameters:
///   cargo run --example security_levels --release                    # 128-bit (default)
///   cargo run --example security_levels --release --features security-80bit
///   cargo run --example security_levels --release --features security-110bit

use rs_tfhe::{gates, key, params, tlwe::TLWELv0};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          TFHE Security Level Demonstration                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    
    // Display current security configuration
    println!("📊 Current Configuration:");
    println!("   Security Level: {} bits", params::SECURITY_BITS);
    println!("   Description: {}", params::SECURITY_DESCRIPTION);
    println!();
    
    println!("🔧 Cryptographic Parameters:");
    println!("   TLWE Level 0:");
    println!("     - Dimension (N): {}", params::tlwe_lv0::N);
    println!("     - Noise (α): {:.2e}", params::tlwe_lv0::ALPHA);
    println!("   TLWE Level 1:");
    println!("     - Dimension (N): {}", params::tlwe_lv1::N);
    println!("     - Noise (α): {:.2e}", params::tlwe_lv1::ALPHA);
    println!("   TRGSW:");
    println!("     - Decomposition levels (L): {}", params::trgsw_lv1::L);
    println!("     - Base bits (BGBIT): {}", params::trgsw_lv1::BGBIT);
    println!("     - Key switching (IKS_T): {}", params::trgsw_lv1::IKS_T);
    println!();
    
    // Generate keys
    println!("🔑 Generating keys...");
    let start = Instant::now();
    let secret_key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&secret_key);
    let keygen_time = start.elapsed();
    println!("   Key generation time: {:.2?}", keygen_time);
    println!();
    
    // Test data
    let test_cases = vec![
        (true, true, "AND"),
        (true, false, "AND"),
        (false, true, "OR"),
        (false, false, "XOR"),
    ];
    
    println!("🧪 Running Homomorphic Operations:");
    println!();
    
    let mut total_time = std::time::Duration::ZERO;
    let mut gate_count = 0;
    
    for (i, (a, b, operation)) in test_cases.iter().enumerate() {
        println!("   Test {}: {} {} {}", i + 1, a, operation, b);
        
        // Encrypt using TLWE
        let enc_a = TLWELv0::encrypt_bool(*a, params::tlwe_lv0::ALPHA, &secret_key.key_lv0);
        let enc_b = TLWELv0::encrypt_bool(*b, params::tlwe_lv0::ALPHA, &secret_key.key_lv0);
        
        // Homomorphic operation
        let start = Instant::now();
        let enc_result = match *operation {
            "AND" => gates::and(&enc_a, &enc_b, &cloud_key),
            "OR" => gates::or(&enc_a, &enc_b, &cloud_key),
            "XOR" => gates::xor(&enc_a, &enc_b, &cloud_key),
            _ => panic!("Unknown operation"),
        };
        let gate_time = start.elapsed();
        
        // Decrypt and verify
        let result = TLWELv0::decrypt_bool(&enc_result, &secret_key.key_lv0);
        let expected = match *operation {
            "AND" => *a && *b,
            "OR" => *a || *b,
            "XOR" => *a ^ *b,
            _ => panic!("Unknown operation"),
        };
        
        let status = if result == expected { "✓" } else { "✗" };
        println!("     Result: {} (expected: {}) {} - {:.2?}", 
                 result, expected, status, gate_time);
        
        total_time += gate_time;
        gate_count += 1;
    }
    
    println!();
    println!("📈 Performance Summary:");
    println!("   Total gates: {}", gate_count);
    println!("   Total time: {:.2?}", total_time);
    println!("   Average per gate: {:.2?}", total_time / gate_count);
    println!();
    
    println!("💡 Performance Comparison (approximate):");
    println!("   80-bit:  Fastest    (~20-30% faster than default)");
    println!("   110-bit: Balanced   (original TFHE parameters)");
    println!("   128-bit: Baseline   (default, high security)");
    println!();
    
    println!("🔒 Security Recommendations:");
    println!("   • 80-bit:  Development, testing, non-critical applications");
    println!("   • 110-bit: Balanced performance/security (original TFHE)");
    println!("   • 128-bit: Production use, high security (recommended, DEFAULT)");
    println!();
    
    println!("📚 How to switch security levels:");
    println!("   cargo run --example security_levels --release                    # 128-bit (default)");
    println!("   cargo run --example security_levels --release --features security-80bit");
    println!("   cargo run --example security_levels --release --features security-110bit");
    println!();
    
    println!("✅ Security level demonstration complete!");
}

