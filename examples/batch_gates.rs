/// Example demonstrating batch gate processing with parallelization
///
/// This shows how gate-level batching achieves near-linear speedup with core count
/// by parallelizing at the right granularity (~57ms per gate).
use rs_tfhe::{gates, key, params, tlwe::TLWELv0};
use std::time::Instant;

fn main() {
  println!("╔══════════════════════════════════════════════════════════════╗");
  println!("║        Batch Gate Processing - Parallelization Demo         ║");
  println!("╚══════════════════════════════════════════════════════════════╝");
  println!();

  // Generate keys
  println!("🔑 Generating keys...");
  let secret_key = key::SecretKey::new();
  let cloud_key = key::CloudKey::new(&secret_key);
  println!("   ✓ Keys generated");
  println!();

  // Test data: 8 pairs of inputs
    let test_pairs = [
      (true, true),
      (true, false),
      (false, true),
      (false, false),
      (true, true),
      (false, false),
      (true, false),
      (false, true),
    ];

  println!("🧪 Testing with {} NAND gates", test_pairs.len());
  println!();

  // Encrypt all inputs
  println!("🔐 Encrypting inputs...");
  let encrypted_pairs: Vec<_> = test_pairs
    .iter()
    .map(|(a, b)| {
      let enc_a = TLWELv0::encrypt_bool(*a, params::tlwe_lv0::ALPHA, &secret_key.key_lv0);
      let enc_b = TLWELv0::encrypt_bool(*b, params::tlwe_lv0::ALPHA, &secret_key.key_lv0);
      (enc_a, enc_b)
    })
    .collect();
  println!("   ✓ All inputs encrypted");
  println!();

  // ========================================================================
  // TEST 1: Sequential Processing (Baseline)
  // ========================================================================
  println!("━━━ Sequential Processing (Baseline) ━━━");
  let start = Instant::now();
  let mut sequential_results = Vec::new();
  for (enc_a, enc_b) in &encrypted_pairs {
    sequential_results.push(gates::nand(enc_a, enc_b, &cloud_key));
  }
  let sequential_time = start.elapsed();
  let sequential_per_gate = sequential_time.as_millis() as f64 / test_pairs.len() as f64;

  println!("   Total time:  {:?}", sequential_time);
  println!("   Per gate:    {:.2} ms", sequential_per_gate);
  println!();

  // Decrypt and verify
  let sequential_decrypted: Vec<bool> = sequential_results
    .iter()
    .map(|c| TLWELv0::decrypt_bool(c, &secret_key.key_lv0))
    .collect();

  // ========================================================================
  // TEST 2: Batch Processing (Parallel)
  // ========================================================================
  println!("━━━ Batch Processing (Parallel) ━━━");
  let start = Instant::now();
  let batch_results = gates::batch_nand(&encrypted_pairs, &cloud_key);
  let batch_time = start.elapsed();
  let batch_per_gate = batch_time.as_millis() as f64 / test_pairs.len() as f64;

  println!("   Total time:  {:?}", batch_time);
  println!("   Per gate:    {:.2} ms", batch_per_gate);
  println!();

  // Decrypt and verify
  let batch_decrypted: Vec<bool> = batch_results
    .iter()
    .map(|c| TLWELv0::decrypt_bool(c, &secret_key.key_lv0))
    .collect();

  // ========================================================================
  // Verification
  // ========================================================================
  println!("━━━ Verification ━━━");
  let mut all_correct = true;
  for (i, ((a, b), (seq_result, batch_result))) in test_pairs
    .iter()
    .zip(sequential_decrypted.iter().zip(batch_decrypted.iter()))
    .enumerate()
  {
    let expected = !(*a && *b); // NAND
    let seq_ok = *seq_result == expected;
    let batch_ok = *batch_result == expected;
    let status = if seq_ok && batch_ok { "✓" } else { "✗" };

    println!(
      "   Gate {}: {} NAND {} = {} (expected: {}) {}",
      i + 1,
      a,
      b,
      batch_result,
      expected,
      status
    );

    all_correct = all_correct && seq_ok && batch_ok;
  }
  println!();

  // ========================================================================
  // Performance Summary
  // ========================================================================
  println!("━━━ Performance Summary ━━━");
  println!();
  println!(
    "  Sequential: {:.2} ms total ({:.2} ms/gate)",
    sequential_time.as_millis(),
    sequential_per_gate
  );
  println!(
    "  Parallel:   {:.2} ms total ({:.2} ms/gate)",
    batch_time.as_millis(),
    batch_per_gate
  );
  println!();

  let speedup = sequential_time.as_secs_f64() / batch_time.as_secs_f64();
  println!("  Speedup: {:.2}x", speedup);
  println!();

  // Estimate ideal speedup based on CPU cores
  let num_cpus = num_cpus::get();
  let ideal_speedup = num_cpus.min(test_pairs.len()) as f64;
  let efficiency = (speedup / ideal_speedup * 100.0).min(100.0);

  println!("  CPU cores:      {}", num_cpus);
  println!(
    "  Ideal speedup:  {:.1}x (for {} gates)",
    ideal_speedup,
    test_pairs.len()
  );
  println!("  Efficiency:     {:.1}%", efficiency);
  println!();

  println!("💡 Analysis:");
  if speedup >= 1.5 {
    println!(
      "   ✅ Great! Batch processing provides {:.1}x speedup",
      speedup
    );
    println!("   This is the RIGHT level of parallelization granularity!");
  } else {
    println!("   ⚠️  Speedup is modest ({:.1}x). Consider:", speedup);
    println!("      • Testing with more gates (16-32+)");
    println!("      • Checking if system has multiple cores");
    println!("      • Profiling for unexpected bottlenecks");
  }
  println!();

  if all_correct {
    println!("✅ All results correct! Batch processing works perfectly.");
  } else {
    println!("❌ Some results incorrect - need debugging!");
  }
}
