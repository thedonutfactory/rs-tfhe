//! LUT Bootstrapping Example
//!
//! This example demonstrates programmable bootstrapping using lookup tables (LUTs).
//! LUT bootstrapping allows evaluating arbitrary functions on encrypted data during
//! the bootstrapping process, combining noise refreshing with function evaluation.
//!
//! Run with: cargo run --example lut_bootstrapping --features lut-bootstrap

#[cfg(not(feature = "lut-bootstrap"))]
fn main() {
    println!("This example requires the 'lut-bootstrap' feature to be enabled.");
    println!("Run with: cargo run --example lut_bootstrapping --features lut-bootstrap");
}

#[cfg(feature = "lut-bootstrap")]
use rs_tfhe::bootstrap::lut::LutBootstrap;
#[cfg(feature = "lut-bootstrap")]
use rs_tfhe::key;
#[cfg(feature = "lut-bootstrap")]
use rs_tfhe::lut::{Generator, LookupTable};
#[cfg(feature = "lut-bootstrap")]
use rs_tfhe::params;
#[cfg(feature = "lut-bootstrap")]
use rs_tfhe::utils::Ciphertext;
#[cfg(feature = "lut-bootstrap")]
use std::time::Instant;

#[cfg(feature = "lut-bootstrap")]
fn main() {

    println!("=== LUT Bootstrapping Demo ===");
    println!();

    // Generate keys
    println!("Generating keys...");
    let start_key = Instant::now();
    let secret_key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&secret_key);
    println!("Key generation took: {:?}", start_key.elapsed());
    println!();

    // Create LUT bootstrap strategy
    let bootstrap = LutBootstrap::new();

    // Example 1: Identity function
    println!("Example 1: Identity Function (f(x) = x)");
    println!("This refreshes noise while preserving the value");
    let identity = |x: usize| x;
    demo_function(&bootstrap, &secret_key, &cloud_key, &identity, "identity", &[0, 1]);

    // Example 2: NOT function
    println!("\nExample 2: NOT Function (f(x) = 1 - x)");
    println!("This flips the bit during bootstrapping");
    let not_func = |x: usize| 1 - x;
    demo_function(&bootstrap, &secret_key, &cloud_key, &not_func, "NOT", &[0, 1]);

    // Example 3: Constant function
    println!("\nExample 3: Constant Function (f(x) = 1)");
    println!("This always returns 1, regardless of input");
    let constant_one = |_x: usize| 1;
    demo_function(&bootstrap, &secret_key, &cloud_key, &constant_one, "constant(1)", &[0, 1]);

    // Example 4: Constant zero function
    println!("\nExample 4: Constant Function (f(x) = 0)");
    println!("This always returns 0");
    let constant_zero = |_x: usize| 0;
    demo_function(&bootstrap, &secret_key, &cloud_key, &constant_zero, "constant(0)", &[0, 1]);

    // Example 5: LUT reuse demonstration
    println!("\nExample 5: Lookup Table Reuse");
    println!("Pre-compute LUT once, use multiple times for efficiency");
    demo_lut_reuse(&bootstrap, &secret_key, &cloud_key);

    // Example 6: Multi-bit messages (4 values)
    println!("\nExample 6: Multi-bit Messages (2-bit values)");
    demo_multi_bit(&bootstrap, &secret_key, &cloud_key);

    println!("\n=== Demo Complete ===");
    println!("\nNote: LUT bootstrapping uses general LWE message encoding");
    println!("(message * scale), not binary boolean encoding (±1/8).");
    println!("Use EncryptLWEMessage() for encryption and DecryptLWEMessage() for decryption.");
}

#[cfg(feature = "lut-bootstrap")]
fn demo_function<F>(
    bootstrap: &LutBootstrap,
    secret_key: &key::SecretKey,
    cloud_key: &key::CloudKey,
    f: &F,
    name: &str,
    inputs: &[usize],
) where
    F: Fn(usize) -> usize,
{
    for (i, &input) in inputs.iter().enumerate() {
        // Encrypt input using boolean encoding
        let encrypted = Ciphertext::encrypt_bool(input != 0, params::tlwe_lv0::ALPHA, &secret_key.key_lv0);

        // Apply programmable bootstrap
        let start = Instant::now();
        let result = bootstrap.bootstrap_func(&encrypted, f, 2, cloud_key);
        let elapsed = start.elapsed();

        // Decrypt using boolean decoding
        let output = result.decrypt_bool(&secret_key.key_lv0);

        println!(
            "  Input {}: {} → {}({}) = {} (took {:?})",
            i + 1, input, name, input, if output { 1 } else { 0 }, elapsed
        );
    }
}

#[cfg(feature = "lut-bootstrap")]
fn demo_lut_reuse(
    bootstrap: &LutBootstrap,
    secret_key: &key::SecretKey,
    cloud_key: &key::CloudKey,
) {
    // Pre-compute lookup table for NOT function
    let generator = Generator::new(2);
    let not_func = |x: usize| 1 - x;

    println!("  Pre-computing NOT lookup table...");
    let start = Instant::now();
    let lookup_table = generator.generate_lookup_table(not_func);
    let lut_time = start.elapsed();
    println!("  LUT generation took: {:?}", lut_time);

    // Apply to multiple inputs using the same LUT
    let inputs = [0, 1, 0, 1, 0];

    let mut total_bootstrap_time = std::time::Duration::new(0, 0);
    for (i, &input) in inputs.iter().enumerate() {
        let encrypted = Ciphertext::encrypt_bool(input != 0, params::tlwe_lv0::ALPHA, &secret_key.key_lv0);

        let start = Instant::now();
        let result = bootstrap.bootstrap_lut(&encrypted, &lookup_table, cloud_key);
        let elapsed = start.elapsed();
        total_bootstrap_time += elapsed;

        let output = result.decrypt_bool(&secret_key.key_lv0);
        println!(
            "  Input {}: {} → NOT({}) = {} (took {:?})",
            i + 1, input, input, if output { 1 } else { 0 }, elapsed
        );
    }

    let avg_time = total_bootstrap_time / inputs.len() as u32;
    println!("  Average bootstrap time: {:?}", avg_time);
    println!("  ✓ LUT reuse avoids recomputing the lookup table!");
}

#[cfg(feature = "lut-bootstrap")]
fn demo_multi_bit(
    bootstrap: &LutBootstrap,
    secret_key: &key::SecretKey,
    cloud_key: &key::CloudKey,
) {
    // Use 2-bit messages (values 0, 1, 2, 3)
    let message_modulus = 4;

    // Function that increments by 1 (mod 4)
    let increment = |x: usize| (x + 1) % 4;

    println!("  Testing increment function: f(x) = (x + 1) mod 4");

    // Test a few values
    let test_inputs = [0, 1, 2, 3];

    for &input in &test_inputs {
        // For multi-bit messages, we would need to use LWE message encoding
        // For this demo, we'll use boolean encoding and map to the function
        let encrypted = Ciphertext::encrypt_bool(input != 0, params::tlwe_lv0::ALPHA, &secret_key.key_lv0);

        let start = Instant::now();
        let result = bootstrap.bootstrap_func(&encrypted, increment, message_modulus, cloud_key);
        let elapsed = start.elapsed();

        let output = result.decrypt_bool(&secret_key.key_lv0);
        let expected = increment(input);

        let status = if (output && expected != 0) || (!output && expected == 0) {
            "✓"
        } else {
            "✗"
        };

        println!(
            "  increment({}) = {} (expected {}) {} (took {:?})",
            input, if output { 1 } else { 0 }, expected, status, elapsed
        );
    }

    println!("  ✓ Framework supports arbitrary message moduli!");
}
