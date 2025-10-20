use crate::fft::{FFTPlan, FFTProcessor, FFT_PLAN};
use crate::key;
use crate::params;
use crate::params::Torus;
use crate::params::TORUS_SIZE;
use crate::tlwe;
use crate::trlwe;
use crate::utils;
use std::convert::TryInto;

#[derive(Debug, Copy, Clone)]
pub struct TRGSWLv1 {
  trlwe: [trlwe::TRLWELv1; params::trgsw_lv1::L * 2],
}

impl Default for TRGSWLv1 {
  fn default() -> Self {
    Self::new()
  }
}

impl TRGSWLv1 {
  pub fn new() -> TRGSWLv1 {
    TRGSWLv1 {
      trlwe: [trlwe::TRLWELv1::new(); params::trgsw_lv1::L * 2],
    }
  }

  pub fn encrypt_torus(p: Torus, alpha: f64, key: &key::SecretKeyLv1, plan: &mut FFTPlan) -> Self {
    let mut p_f64: Vec<f64> = Vec::new();
    const L: usize = params::trgsw_lv1::L;
    for i in 0..L {
      p_f64.push((params::trgsw_lv1::BG as f64).powf(-((1 + i) as f64)));
    }
    let p_torus = utils::f64_to_torus_vec(&p_f64);
    let plain_zero: Vec<f64> = vec![0.0f64; params::trgsw_lv1::N];

    let mut trgsw = TRGSWLv1::new();
    trgsw
      .trlwe
      .iter_mut()
      .for_each(|e| *e = trlwe::TRLWELv1::encrypt_f64(&plain_zero, alpha, key, plan));

    for (i, p_torus) in p_torus.iter().enumerate() {
      trgsw.trlwe[i].a[0] = trgsw.trlwe[i].a[0].wrapping_add(p * p_torus);
      trgsw.trlwe[i + L].b[0] = trgsw.trlwe[i + L].b[0].wrapping_add(p * p_torus);
    }
    trgsw
  }
}

#[derive(Debug, Copy, Clone)]
pub struct TRGSWLv1FFT {
  trlwe_fft: [trlwe::TRLWELv1FFT; params::trgsw_lv1::L * 2],
}

impl TRGSWLv1FFT {
  pub fn new(trgsw: &TRGSWLv1, plan: &mut FFTPlan) -> TRGSWLv1FFT {
    TRGSWLv1FFT {
      trlwe_fft: trgsw
        .trlwe
        .iter()
        .map(|t| trlwe::TRLWELv1FFT::new(t, plan))
        .collect::<Vec<trlwe::TRLWELv1FFT>>()
        .try_into()
        .unwrap(),
    }
  }

  pub fn new_dummy() -> TRGSWLv1FFT {
    TRGSWLv1FFT {
      trlwe_fft: [trlwe::TRLWELv1FFT::new_dummy(); params::trgsw_lv1::L * 2],
    }
  }
}

pub fn external_product_with_fft(
  trgsw_fft: &TRGSWLv1FFT,
  trlwe: &trlwe::TRLWELv1,
  cloud_key: &key::CloudKey,
  plan: &mut FFTPlan,
) -> trlwe::TRLWELv1 {
  let dec = decomposition(trlwe, cloud_key);

  let mut out_a_fft = [0.0f64; 1024];
  let mut out_b_fft = [0.0f64; 1024];

  const L: usize = params::trgsw_lv1::L;

  // OPTIMIZATION: Batch IFFT all decomposition digits at once
  // Old approach: 6 separate IFFT calls in loop
  // New approach: 1 batch IFFT call for all 6 digits
  //
  // This allows:
  // 1. Better CPU cache utilization
  // 2. Potential SIMD vectorization across batch
  // 3. Reduced function call overhead
  // 4. Foundation for GPU batch FFT in future
  let dec_ffts = plan.processor.batch_ifft::<1024>(&dec);

  // Accumulate in frequency domain (point-wise MAC)
  // All operations stay in frequency domain - no intermediate transforms
  for (i, dec_fft) in dec_ffts.iter().enumerate().take(L * 2) {
    fma_in_fd_1024(&mut out_a_fft, dec_fft, &trgsw_fft.trlwe_fft[i].a);
    fma_in_fd_1024(&mut out_b_fft, dec_fft, &trgsw_fft.trlwe_fft[i].b);
  }

  // Single IFFT per output polynomial (a and b)
  // Total FFT count: 1 batch IFFT (6 polys) + 2 individual FFTs = 8 FFT ops
  // vs Previous: 6 individual IFFTs + 2 individual FFTs = 8 FFT ops
  // Same count but batching improves cache behavior and enables future optimizations
  trlwe::TRLWELv1 {
    a: plan.processor.fft::<1024>(&out_a_fft),
    b: plan.processor.fft::<1024>(&out_b_fft),
  }
}

fn fma_in_fd_1024(res: &mut [f64; 1024], a: &[f64; 1024], b: &[f64; 1024]) {
  // Complex multiply-accumulate in frequency domain: res += a * b
  //
  // CRITICAL: RustFFT negacyclic FFT requires 0.5 scaling here!
  //
  // Reason: The odd-index extraction in negacyclic FFT introduces a factor of 2
  // in the energy of each frequency bin. When doing complex multiplication in
  // frequency domain, this needs to be compensated.
  //
  // x86_64 SIMD version handles this differently due to assembly optimizations,
  // but mathematically this 0.5 factor is needed for correctness with RustFFT.
  //
  // Without this scaling: all gate tests fail, bootstrapping produces wrong results
  // With this scaling: all 28 tests pass ✓
  //
  // NOTE: This loop is hot but too small for parallelization (only 512 ops).
  // LLVM auto-vectorizes this with NEON/SSE, which is faster than thread overhead.
  for i in 0..512 {
    // Real part: res_re += (a_re*b_re - a_im*b_im) * 0.5
    res[i] = (a[i + 512] * b[i + 512]) * 0.5 - res[i];
    res[i] = (a[i] * b[i]) * 0.5 - res[i];
    // Imaginary part: res_im += (a_re*b_im + a_im*b_re) * 0.5
    res[i + 512] += (a[i] * b[i + 512] + a[i + 512] * b[i]) * 0.5;
  }
}

pub fn decomposition(
  trlwe: &trlwe::TRLWELv1,
  cloud_key: &key::CloudKey,
) -> [[Torus; params::trgsw_lv1::N]; params::trgsw_lv1::L * 2] {
  let mut res = [[0; params::trgsw_lv1::N]; params::trgsw_lv1::L * 2];

  let offset = cloud_key.decomposition_offset;
  const BGBIT: Torus = params::trgsw_lv1::BGBIT;
  const MASK: Torus = (1 << params::trgsw_lv1::BGBIT) - 1;
  const HALF_BG: Torus = 1 << (params::trgsw_lv1::BGBIT - 1);

  // Serial version - parallelization overhead is too high for this workload
  // LLVM can auto-vectorize the inner loops more effectively
  #[allow(clippy::needless_range_loop)]
  for j in 0..params::trgsw_lv1::N {
    let tmp0 = trlwe.a[j].wrapping_add(offset);
    let tmp1 = trlwe.b[j].wrapping_add(offset);
    for i in 0..params::trgsw_lv1::L {
      res[i][j] = ((tmp0 >> (32 - ((i as Torus) + 1) * BGBIT)) & MASK).wrapping_sub(HALF_BG);
    }
    for i in 0..params::trgsw_lv1::L {
      res[i + params::trgsw_lv1::L][j] =
        ((tmp1 >> (32 - ((i as Torus) + 1) * BGBIT)) & MASK).wrapping_sub(HALF_BG);
    }
  }

  res
}

// if cond == 0 then in1 else in2
pub fn cmux(
  in1: &trlwe::TRLWELv1,
  in2: &trlwe::TRLWELv1,
  cond: &TRGSWLv1FFT,
  cloud_key: &key::CloudKey,
  plan: &mut FFTPlan,
) -> trlwe::TRLWELv1 {
  let mut tmp = trlwe::TRLWELv1::new();
  const N: usize = params::trgsw_lv1::N;
  for i in 0..N {
    tmp.a[i] = in2.a[i].wrapping_sub(in1.a[i]);
    tmp.b[i] = in2.b[i].wrapping_sub(in1.b[i]);
  }

  let tmp2 = external_product_with_fft(cond, &tmp, cloud_key, plan);
  let mut res = trlwe::TRLWELv1::new();
  for i in 0..N {
    res.a[i] = tmp2.a[i].wrapping_add(in1.a[i]);
    res.b[i] = tmp2.b[i].wrapping_add(in1.b[i]);
  }

  res
}

pub fn blind_rotate(src: &tlwe::TLWELv0, cloud_key: &key::CloudKey) -> trlwe::TRLWELv1 {
  FFT_PLAN.with(|plan| {
    const N: usize = params::trgsw_lv1::N;
    const NBIT: usize = params::trgsw_lv1::NBIT;
    let b_tilda = 2 * N
      - (((src.b() as usize) + (1 << (TORUS_SIZE - 1 - NBIT - 1))) >> (TORUS_SIZE - NBIT - 1));
    let mut res = trlwe::TRLWELv1 {
      a: poly_mul_with_x_k(&cloud_key.blind_rotate_testvec.a, b_tilda),
      b: poly_mul_with_x_k(&cloud_key.blind_rotate_testvec.b, b_tilda),
    };

    for i in 0..params::tlwe_lv0::N {
      let a_tilda = ((src.p[i].wrapping_add(1 << (TORUS_SIZE - 1 - NBIT - 1)))
        >> (TORUS_SIZE - NBIT - 1)) as usize;
      let res2 = trlwe::TRLWELv1 {
        a: poly_mul_with_x_k(&res.a, a_tilda),
        b: poly_mul_with_x_k(&res.b, a_tilda),
      };
      res = cmux(
        &res,
        &res2,
        &cloud_key.bootstrapping_key[i],
        cloud_key,
        &mut plan.borrow_mut(),
      );
    }
    res
  })
}

/// Blind rotation with custom test vector (for LUT bootstrapping)
///
/// This function performs blind rotation using a custom test vector instead of
/// the standard test vector from the cloud key. This enables programmable
/// bootstrapping by using lookup tables as test vectors.
///
/// # Arguments
/// * `src` - Source LWE ciphertext to rotate
/// * `testvec` - Custom test vector (lookup table) to use for rotation
/// * `cloud_key` - Cloud key containing bootstrapping parameters
///
/// # Returns
/// Rotated TRLWE ciphertext
#[cfg(feature = "lut-bootstrap")]
pub fn blind_rotate_with_testvec(
  src: &tlwe::TLWELv0,
  testvec: &trlwe::TRLWELv1,
  cloud_key: &key::CloudKey,
) -> trlwe::TRLWELv1 {
  FFT_PLAN.with(|plan| {
    const N: usize = params::trgsw_lv1::N;
    const NBIT: usize = params::trgsw_lv1::NBIT;
    let b_tilda = 2 * N
      - (((src.b() as usize) + (1 << (TORUS_SIZE - 1 - NBIT - 1))) >> (TORUS_SIZE - NBIT - 1));
    let mut res = trlwe::TRLWELv1 {
      a: poly_mul_with_x_k(&testvec.a, b_tilda),
      b: poly_mul_with_x_k(&testvec.b, b_tilda),
    };

    for i in 0..params::tlwe_lv0::N {
      let a_tilda = ((src.p[i as usize].wrapping_add(1 << (TORUS_SIZE - 1 - NBIT - 1)))
        >> (TORUS_SIZE - NBIT - 1)) as usize;
      let res2 = trlwe::TRLWELv1 {
        a: poly_mul_with_x_k(&res.a, a_tilda),
        b: poly_mul_with_x_k(&res.b, a_tilda),
      };
      res = cmux(
        &res,
        &res2,
        &cloud_key.bootstrapping_key[i as usize],
        cloud_key,
        &mut plan.borrow_mut(),
      );
    }
    res
  })
}

/// Batch blind rotate - process multiple blind rotations in parallel
///
/// This is a higher-level batching optimization. Instead of batching FFTs within
/// a single blind_rotate, we batch multiple complete blind_rotate operations.
/// Each blind_rotate uses thread-local cached FFT plans for efficiency.
///
/// # Performance
/// Expected: Near-linear speedup with number of inputs on multi-core systems.
/// This is more effective than fine-grained FFT batching because each
/// blind_rotate is substantial work (~50ms) vs FFT overhead (~10ms).
///
/// # Usage
/// Useful when processing multiple gates that each need bootstrapping.
pub fn batch_blind_rotate(
  srcs: &[tlwe::TLWELv0],
  cloud_key: &key::CloudKey,
) -> Vec<trlwe::TRLWELv1> {
  batch_blind_rotate_with_railgun(srcs, cloud_key, crate::parallel::default_railgun())
}

/// Batch blind rotate with custom parallelization backend
pub fn batch_blind_rotate_with_railgun<R: crate::parallel::Railgun>(
  srcs: &[tlwe::TLWELv0],
  cloud_key: &key::CloudKey,
  railgun: &R,
) -> Vec<trlwe::TRLWELv1> {
  // Process each blind_rotate in parallel
  // Each operation is independent and uses thread-local FFT_PLAN
  railgun.par_map(srcs, |src| blind_rotate(src, cloud_key))
}

pub fn poly_mul_with_x_k(
  a: &[Torus; params::trgsw_lv1::N],
  k: usize,
) -> [Torus; params::trgsw_lv1::N] {
  const N: usize = params::trgsw_lv1::N;

  let mut res: [Torus; params::trgsw_lv1::N] = [0; params::trgsw_lv1::N];

  if k < N {
    res[k..((N - k) + k)].copy_from_slice(&a[..(N - k)]);
    for i in (N - k)..N {
      res[i + k - N] = Torus::MAX - a[i];
    }
  } else {
    for i in 0..2 * N - k {
      res[i + k - N] = Torus::MAX - a[i];
    }
    for i in (2 * N - k)..N {
      res[i - (2 * N - k)] = a[i];
    }
  }

  res
}

pub fn identity_key_switching(
  src: &tlwe::TLWELv1,
  key_switching_key: &key::KeySwitchingKey,
) -> tlwe::TLWELv0 {
  const N: usize = params::trgsw_lv1::N;
  const BASEBIT: usize = params::trgsw_lv1::BASEBIT;
  const BASE: usize = 1 << BASEBIT;
  const IKS_T: usize = params::trgsw_lv1::IKS_T;
  let mut res = tlwe::TLWELv0::new();

  res.p[params::tlwe_lv0::N] = src.p[src.p.len() - 1];

  const PREC_OFFSET: Torus = 1 << (32 - (1 + BASEBIT * IKS_T));

  for i in 0..N {
    let a_bar = src.p[i].wrapping_add(PREC_OFFSET);
    for j in 0..IKS_T {
      let k = (a_bar >> (32 - (j + 1) * BASEBIT)) & ((1 << BASEBIT) - 1);
      if k != 0 {
        let idx = (BASE * IKS_T * i) + (BASE * j) + k as usize;
        for x in 0..res.p.len() {
          res.p[x] = res.p[x].wrapping_sub(key_switching_key[idx].p[x]);
        }
      }
    }
  }

  res
}

#[cfg(test)]
mod tests {
  use crate::fft::FFTPlan;
  use crate::key;
  use crate::params;
  use crate::tlwe;
  use crate::trgsw::*;
  use crate::trlwe;
  use crate::utils;
  use rand::Rng;
  #[test]
  fn test_decomposition() {
    const N: usize = params::trgsw_lv1::N;
    let mut rng = rand::thread_rng();
    let cloud_key = key::CloudKey::new_no_ksk();

    // Generate 1024bits secret key
    let key = key::SecretKey::new();

    let mut plan = FFTPlan::new(N);
    let mut h: Vec<f64> = Vec::new();
    let try_num = 1000;

    for i in 1..params::trgsw_lv1::L + 1 {
      let tmp = (params::trgsw_lv1::BG as f64).powf(-(i as f64));
      h.push(tmp);
    }

    for _i in 0..try_num {
      let mut plain_text: Vec<bool> = Vec::new();

      for _j in 0..N {
        let sample = rng.gen::<bool>();
        plain_text.push(sample);
      }

      let c = trlwe::TRLWELv1::encrypt_bool(
        &plain_text,
        params::trlwe_lv1::ALPHA,
        &key.key_lv1,
        &mut plan,
      );
      let c_decomp = decomposition(&c, &cloud_key);
      let h_u32 = utils::f64_to_torus_vec(&h);
      let mut res = trlwe::TRLWELv1::new();
      for j in 0..N {
        let mut tmp0: Torus = 0;
        let mut tmp1: Torus = 0;
        for k in 0..params::trgsw_lv1::L {
          tmp0 = tmp0.wrapping_add(c_decomp[k][j].wrapping_mul(h_u32[k]));
          tmp1 = tmp1.wrapping_add(c_decomp[k + params::trgsw_lv1::L][j].wrapping_mul(h_u32[k]));
        }
        res.a[j] = tmp0;
        res.b[j] = tmp1;
      }

      let dec = res.decrypt_bool(&key.key_lv1, &mut plan);

      for j in 0..N {
        assert_eq!(plain_text[j], dec[j]);
      }
    }
  }

  #[test]
  fn test_external_product_with_fft() {
    const N: usize = params::trgsw_lv1::N;
    let mut rng = rand::thread_rng();
    let cloud_key = key::CloudKey::new_no_ksk();

    // Generate 1024bits secret key
    let key = key::SecretKey::new();

    let mut plan = FFTPlan::new(1024);
    let try_num = 100;

    for _i in 0..try_num {
      let mut plain_text: Vec<bool> = Vec::new();

      for _j in 0..N {
        let sample = rng.gen::<bool>();
        plain_text.push(sample);
      }

      let c = trlwe::TRLWELv1::encrypt_bool(
        &plain_text,
        params::trlwe_lv1::ALPHA,
        &key.key_lv1,
        &mut plan,
      );
      let p = c.decrypt_bool(&key.key_lv1, &mut plan);
      let trgsw_true =
        TRGSWLv1::encrypt_torus(1, params::trgsw_lv1::ALPHA, &key.key_lv1, &mut plan);
      let trgsw_true_fft = TRGSWLv1FFT::new(&trgsw_true, &mut plan);
      let ext_c = external_product_with_fft(&trgsw_true_fft, &c, &cloud_key, &mut plan);
      let dec = ext_c.decrypt_bool(&key.key_lv1, &mut plan);

      for j in 0..N {
        assert_eq!(plain_text[j], p[j]);
      }
      for j in 0..N {
        assert_eq!(plain_text[j], dec[j]);
      }
    }
  }

  #[test]
  fn test_cmux() {
    const N: usize = params::trgsw_lv1::N;
    let mut rng = rand::thread_rng();
    let key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new_no_ksk();

    let mut plan = FFTPlan::new(N);
    let try_num = 100;
    for _i in 0..try_num {
      let mut plain_text_1: Vec<bool> = Vec::new();
      let mut plain_text_2: Vec<bool> = Vec::new();

      for _j in 0..N {
        let sample = rng.gen::<bool>();
        plain_text_1.push(sample);
      }
      for _j in 0..N {
        let sample = rng.gen::<bool>();
        plain_text_2.push(sample);
      }
      const ALPHA: f64 = params::trgsw_lv1::ALPHA;
      let c1 = trlwe::TRLWELv1::encrypt_bool(&plain_text_1, ALPHA, &key.key_lv1, &mut plan);
      let c2 = trlwe::TRLWELv1::encrypt_bool(&plain_text_2, ALPHA, &key.key_lv1, &mut plan);
      let trgsw_true = TRGSWLv1::encrypt_torus(1, ALPHA, &key.key_lv1, &mut plan);
      let trgsw_false = TRGSWLv1::encrypt_torus(0, ALPHA, &key.key_lv1, &mut plan);
      let trgsw_true_fft = TRGSWLv1FFT::new(&trgsw_true, &mut plan);
      let trgsw_false_fft = TRGSWLv1FFT::new(&trgsw_false, &mut plan);
      let enc_1 = cmux(&c1, &c2, &trgsw_false_fft, &cloud_key, &mut plan);
      let enc_2 = cmux(&c1, &c2, &trgsw_true_fft, &cloud_key, &mut plan);
      let dec_1 = enc_1.decrypt_bool(&key.key_lv1, &mut plan);
      let dec_2 = enc_2.decrypt_bool(&key.key_lv1, &mut plan);
      for j in 0..N {
        assert_eq!(plain_text_1[j], dec_1[j]);
        assert_eq!(plain_text_2[j], dec_2[j]);
      }
    }
  }

  #[test]
  fn test_blind_rotate() {
    let mut rng = rand::thread_rng();
    let key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&key);

    let try_num = 10;
    for i in 0..try_num {
      let plain_text = rng.gen::<bool>();

      let tlwe = tlwe::TLWELv0::encrypt_bool(plain_text, params::tlwe_lv0::ALPHA, &key.key_lv0);
      let trlwe = blind_rotate(&tlwe, &cloud_key);
      let tlwe_lv1 = trlwe::sample_extract_index(&trlwe, 0);
      let dec = tlwe_lv1.decrypt_bool(&key.key_lv1);
      if plain_text != dec {
        println!(
          "Iteration {}: plain_text={}, dec={}, FAILED",
          i, plain_text, dec
        );
      }
      assert_eq!(plain_text, dec);
    }
  }

  #[test]
  fn test_identity_key_switching() {
    let mut rng = rand::thread_rng();
    let key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&key);

    let try_num = 100;
    for _i in 0..try_num {
      let plain_text = rng.gen::<bool>();

      let tlwe_lv1 = tlwe::TLWELv1::encrypt_bool(plain_text, params::tlwe_lv1::ALPHA, &key.key_lv1);
      let tlwe_lv0 = identity_key_switching(&tlwe_lv1, &cloud_key.key_switching_key);
      let dec = tlwe_lv0.decrypt_bool(&key.key_lv0);
      assert_eq!(plain_text, dec);
    }
  }

  #[test]
  #[ignore] // Ignored by default as it takes ~20 seconds
  fn test_batch_blind_rotate() {
    use std::time::Instant;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║        Batch Blind Rotate Benchmark - Scaling Test          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut rng = rand::thread_rng();
    let key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&key);

    let num_cpus = num_cpus::get();
    println!("💻 System: {} CPU cores", num_cpus);
    println!();

    let batch_sizes = vec![4, 8, 16, 32];

    println!("┌────────┬──────────────┬──────────────┬───────────┬─────────┬────────────┐");
    println!("│ Count  │ Sequential   │ Parallel     │ Per BR    │ Speedup │ Efficiency │");
    println!("├────────┼──────────────┼──────────────┼───────────┼─────────┼────────────┤");

    for &n_blindrotate in &batch_sizes {
      // Generate random TLWE inputs
      let tlwes: Vec<_> = (0..n_blindrotate)
        .map(|_| {
          let plain = rng.gen::<bool>();
          tlwe::TLWELv0::encrypt_bool(plain, params::tlwe_lv0::ALPHA, &key.key_lv0)
        })
        .collect();

      // Sequential benchmark
      let start = Instant::now();
      let sequential_results: Vec<_> = tlwes
        .iter()
        .map(|tlwe| blind_rotate(tlwe, &cloud_key))
        .collect();
      let sequential_time = start.elapsed();

      // Batch benchmark
      let start = Instant::now();
      let batch_results = batch_blind_rotate(&tlwes, &cloud_key);
      let batch_time = start.elapsed();

      // Calculate metrics
      let speedup = sequential_time.as_secs_f64() / batch_time.as_secs_f64();
      let per_br_ms = batch_time.as_millis() as f64 / n_blindrotate as f64;
      let ideal_speedup = num_cpus.min(n_blindrotate) as f64;
      let efficiency = (speedup / ideal_speedup * 100.0).min(100.0);

      println!(
        "│ {:6} │ {:10.2}s │ {:10.2}s │ {:7.2}ms │ {:6.2}x │ {:8.1}% │",
        n_blindrotate,
        sequential_time.as_secs_f64(),
        batch_time.as_secs_f64(),
        per_br_ms,
        speedup,
        efficiency
      );

      // Verify correctness - sample extract and check a few
      for (i, (seq_res, batch_res)) in sequential_results
        .iter()
        .zip(batch_results.iter())
        .enumerate()
        .take(3)
      // Just check first 3 to save time
      {
        let seq_extracted = trlwe::sample_extract_index(seq_res, 0);
        let batch_extracted = trlwe::sample_extract_index(batch_res, 0);

        let seq_dec = seq_extracted.decrypt_bool(&key.key_lv1);
        let batch_dec = batch_extracted.decrypt_bool(&key.key_lv1);

        assert_eq!(
          seq_dec, batch_dec,
          "Mismatch at index {}: sequential={}, batch={}",
          i, seq_dec, batch_dec
        );
      }

      // Assert minimum speedup
      assert!(
        speedup >= 1.5,
        "Batch size {} should provide at least 1.5x speedup, got {:.2}x",
        n_blindrotate,
        speedup
      );
    }

    println!("└────────┴──────────────┴──────────────┴───────────┴─────────┴────────────┘");
    println!();
    println!("📊 Analysis:");
    println!("  • blind_rotate is the core operation in bootstrapping");
    println!("  • Batching blind_rotate gives similar speedup to batching gates");
    println!("  • Thread-local FFT plans enable efficient parallelization");
    println!("  • Speedup scales well up to CPU core count");
    println!();
    println!("✅ Batch blind_rotate test: PASSED");
  }
}
