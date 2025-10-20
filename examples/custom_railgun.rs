//! Example demonstrating custom parallelization with Railgun
//!
//! This example shows how to use the Railgun parallelization abstraction
//! to customize parallel execution behavior in rs-tfhe.

use rs_tfhe::{gates, key, parallel, tlwe};

fn main() {
  println!("=== Custom Railgun Parallelization Example ===\n");

  // Generate keys
  println!("Generating keys...");
  let secret_key = key::SecretKey::new();

  // Example 1: Default Railgun (uses Rayon with default settings)
  println!("\n1. Using default Railgun:");
  let default_railgun = parallel::default_railgun();
  let _cloud_key1 = key::CloudKey {
    decomposition_offset: key::gen_decomposition_offset(),
    blind_rotate_testvec: key::gen_testvec(),
    key_switching_key: key::gen_key_switching_key(&secret_key),
    bootstrapping_key: key::gen_bootstrapping_key_with_railgun(&secret_key, default_railgun),
  };
  println!("  ✓ Cloud key generated with default Railgun");

  // Example 2: Custom Railgun with specific thread configuration
  println!("\n2. Using custom Railgun (4 threads, 16MB stack):");
  let custom_config = parallel::ParallelConfig {
    stack_size: Some(16 * 1024 * 1024), // 16MB per thread
    num_threads: Some(4),               // Use exactly 4 threads
  };
  let custom_railgun = parallel::rayon_railgun(custom_config);
  let cloud_key2 = key::CloudKey {
    decomposition_offset: key::gen_decomposition_offset(),
    blind_rotate_testvec: key::gen_testvec(),
    key_switching_key: key::gen_key_switching_key(&secret_key),
    bootstrapping_key: key::gen_bootstrapping_key_with_railgun(&secret_key, &custom_railgun),
  };
  println!("  ✓ Cloud key generated with custom Railgun (4 threads, 16MB stack)");

  // Example 3: Using Railgun for batch operations
  #[cfg(feature = "bootstrapping")]
  {
    println!("\n3. Batch operations with custom Railgun:");

    // Encrypt some test data
    let alpha = 0.001; // Noise parameter
    let a = tlwe::TLWELv0::encrypt_bool(true, alpha, &secret_key.key_lv0);
    let b = tlwe::TLWELv0::encrypt_bool(false, alpha, &secret_key.key_lv0);
    let c = tlwe::TLWELv0::encrypt_bool(true, alpha, &secret_key.key_lv0);
    let d = tlwe::TLWELv0::encrypt_bool(false, alpha, &secret_key.key_lv0);

    // Batch operations with custom railgun
    let inputs = vec![(a.clone(), b.clone()), (c.clone(), d.clone())];

    let railgun_for_batch = parallel::RayonRailgun::with_config(parallel::ParallelConfig {
      stack_size: Some(8 * 1024 * 1024),
      num_threads: Some(2), // Use 2 threads for batch processing
    });

    let results = gates::batch_and_with_railgun(&inputs, &cloud_key2, &railgun_for_batch);

    println!(
      "  ✓ Processed {} AND gates in batch with custom Railgun",
      results.len()
    );
    println!("  ✓ Results: {} outputs", results.len());

    // Verify results
    for (i, result) in results.iter().enumerate() {
      let decrypted = result.decrypt_bool(&secret_key.key_lv0);
      println!("    Gate {}: {}", i + 1, decrypted);
    }
  }

  // Example 4: Demonstrating different configurations for different workloads
  println!("\n4. Different configurations for different workloads:");

  // Heavy computation (large stack, more threads)
  let heavy_config = parallel::ParallelConfig {
    stack_size: Some(32 * 1024 * 1024), // 32MB for deep recursion
    num_threads: None,                  // Use all available cores
  };
  let _heavy_railgun = parallel::rayon_railgun(heavy_config);
  println!("  ✓ Heavy workload Railgun: 32MB stack, all cores");

  // Light computation (smaller stack, fewer threads)
  let light_config = parallel::ParallelConfig {
    stack_size: Some(4 * 1024 * 1024), // 4MB stack
    num_threads: Some(2),              // Just 2 threads
  };
  let _light_railgun = parallel::rayon_railgun(light_config);
  println!("  ✓ Light workload Railgun: 4MB stack, 2 threads");

  println!("\n=== Benefits of Railgun ===");
  println!("  • Zero-cost abstraction: No runtime overhead");
  println!("  • Flexibility: Easy to swap implementations (Rayon, CUDA, etc.)");
  println!("  • Configuration: Fine-tune thread count and stack size");
  println!("  • Type-safe: Compile-time checked parallelization");
  println!("\n=== Future Extensions ===");
  println!("  • GPU backends (CUDA, Metal)");
  println!("  • Distributed computing");
  println!("  • Custom thread pools");
  println!("  • Performance profiling hooks");
}
