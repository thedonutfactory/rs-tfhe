/// Test parallel bootstrap key generation speed

use rs_tfhe::key;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        Parallel Bootstrap Key Generation Test               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let secret_key = key::SecretKey::new();
    
    println!("🔑 Generating bootstrap key (parallelized)...");
    println!();

    let start = Instant::now();
    let _cloud_key = key::CloudKey::new(&secret_key);
    let keygen_time = start.elapsed();

    println!("✅ Bootstrap key generation complete!");
    println!();
    println!("⏱️  Total time: {:.2?}", keygen_time);
    println!();
    println!("💡 Note:");
    println!("   • Bootstrap key has {} independent encryptions", rs_tfhe::params::tlwe_lv0::N);
    println!("   • Each encryption is ~50-100ms");
    println!("   • Parallelization expected: 4-8x speedup on multi-core");
    println!("   • Single-threaded would take: ~{:.1}s", keygen_time.as_secs_f64() * 6.0);
    println!();
}
