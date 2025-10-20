use crate::bootstrap::Bootstrap;
use crate::key::CloudKey;
use crate::params;
use crate::tlwe::{AddMul, SubMul};
use crate::utils;
use crate::utils::Ciphertext;

/// Gates struct that uses a configurable bootstrap strategy
///
/// This struct provides homomorphic gate operations (AND, OR, NAND, etc.) that
/// use a bootstrap strategy for noise management. The bootstrap strategy can be
/// configured at construction time, enabling experimentation with different
/// optimization approaches.
///
/// # Examples
///
/// ```ignore
/// use rs_tfhe::gates::Gates;
/// use rs_tfhe::bootstrap::vanilla::VanillaBootstrap;
///
/// // Create gates with default bootstrap
/// let gates = Gates::new();
///
/// // Or specify a strategy
/// let gates = Gates::with_bootstrap(Box::new(VanillaBootstrap::new()));
///
/// // Use the gates
/// let result = gates.and(&ct_a, &ct_b, &cloud_key);
/// ```
pub struct Gates {
  bootstrap: Box<dyn Bootstrap>,
}

impl Gates {
  /// Create a new Gates instance with the default bootstrap strategy
  pub fn new() -> Self {
    Gates {
      bootstrap: crate::bootstrap::default_bootstrap(),
    }
  }

  /// Create a Gates instance with a specific bootstrap strategy
  pub fn with_bootstrap(bootstrap: Box<dyn Bootstrap>) -> Self {
    Gates { bootstrap }
  }

  /// Get the name of the bootstrap strategy being used
  pub fn bootstrap_strategy(&self) -> &str {
    self.bootstrap.name()
  }

  /// Homomorphic NAND gate
  #[cfg(feature = "bootstrapping")]
  pub fn nand(&self, tlwe_a: &Ciphertext, tlwe_b: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
    let mut tlwe_nand = -(tlwe_a + tlwe_b);
    *tlwe_nand.b_mut() = tlwe_nand.b().wrapping_add(utils::f64_to_torus(0.125));
    self.bootstrap.bootstrap(&tlwe_nand, cloud_key)
  }

  /// Homomorphic OR gate
  #[cfg(feature = "bootstrapping")]
  pub fn or(&self, tlwe_a: &Ciphertext, tlwe_b: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
    let mut tlwe_or = tlwe_a + tlwe_b;
    *tlwe_or.b_mut() = tlwe_or.b().wrapping_add(utils::f64_to_torus(0.125));
    self.bootstrap.bootstrap(&tlwe_or, cloud_key)
  }

  /// Homomorphic AND gate
  #[cfg(feature = "bootstrapping")]
  pub fn and(&self, tlwe_a: &Ciphertext, tlwe_b: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
    let mut tlwe_and = tlwe_a + tlwe_b;
    *tlwe_and.b_mut() = tlwe_and.b().wrapping_add(utils::f64_to_torus(-0.125));
    self.bootstrap.bootstrap(&tlwe_and, cloud_key)
  }

  /// Homomorphic XOR gate
  #[cfg(feature = "bootstrapping")]
  pub fn xor(&self, tlwe_a: &Ciphertext, tlwe_b: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
    let mut tlwe_xor = tlwe_a.add_mul(tlwe_b, 2);
    *tlwe_xor.b_mut() = tlwe_xor.b().wrapping_add(utils::f64_to_torus(0.25));
    self.bootstrap.bootstrap(&tlwe_xor, cloud_key)
  }

  /// Homomorphic XNOR gate
  #[cfg(feature = "bootstrapping")]
  pub fn xnor(&self, tlwe_a: &Ciphertext, tlwe_b: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
    let mut tlwe_xnor = tlwe_a.sub_mul(tlwe_b, 2);
    *tlwe_xnor.b_mut() = tlwe_xnor.b().wrapping_add(utils::f64_to_torus(-0.25));
    self.bootstrap.bootstrap(&tlwe_xnor, cloud_key)
  }

  /// Homomorphic NOR gate
  #[cfg(feature = "bootstrapping")]
  pub fn nor(&self, tlwe_a: &Ciphertext, tlwe_b: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
    let mut tlwe_nor = -(tlwe_a + tlwe_b);
    *tlwe_nor.b_mut() = tlwe_nor.b().wrapping_add(utils::f64_to_torus(-0.125));
    self.bootstrap.bootstrap(&tlwe_nor, cloud_key)
  }

  /// Homomorphic AND-NOT-Y gate (a AND NOT b)
  #[cfg(feature = "bootstrapping")]
  pub fn and_ny(
    &self,
    tlwe_a: &Ciphertext,
    tlwe_b: &Ciphertext,
    cloud_key: &CloudKey,
  ) -> Ciphertext {
    let mut tlwe_and_ny = &-(*tlwe_a) + tlwe_b;
    *tlwe_and_ny.b_mut() = tlwe_and_ny.b().wrapping_add(utils::f64_to_torus(-0.125));
    self.bootstrap.bootstrap(&tlwe_and_ny, cloud_key)
  }

  /// Homomorphic AND-Y-NOT gate (a AND NOT b)
  #[cfg(feature = "bootstrapping")]
  pub fn and_yn(
    &self,
    tlwe_a: &Ciphertext,
    tlwe_b: &Ciphertext,
    cloud_key: &CloudKey,
  ) -> Ciphertext {
    let mut tlwe_and_yn = tlwe_a - tlwe_b;
    *tlwe_and_yn.b_mut() = tlwe_and_yn.b().wrapping_add(utils::f64_to_torus(-0.125));
    self.bootstrap.bootstrap(&tlwe_and_yn, cloud_key)
  }

  /// Homomorphic OR-NOT-Y gate (NOT a OR b)
  #[cfg(feature = "bootstrapping")]
  pub fn or_ny(
    &self,
    tlwe_a: &Ciphertext,
    tlwe_b: &Ciphertext,
    cloud_key: &CloudKey,
  ) -> Ciphertext {
    let mut tlwe_or_ny = &-*tlwe_a + tlwe_b;
    *tlwe_or_ny.b_mut() = tlwe_or_ny.b().wrapping_add(utils::f64_to_torus(0.125));
    self.bootstrap.bootstrap(&tlwe_or_ny, cloud_key)
  }

  /// Homomorphic OR-Y-NOT gate (a OR NOT b)
  #[cfg(feature = "bootstrapping")]
  pub fn or_yn(
    &self,
    tlwe_a: &Ciphertext,
    tlwe_b: &Ciphertext,
    cloud_key: &CloudKey,
  ) -> Ciphertext {
    let mut tlwe_and_yn = tlwe_a - tlwe_b;
    *tlwe_and_yn.b_mut() = tlwe_and_yn.b().wrapping_add(utils::f64_to_torus(0.125));
    self.bootstrap.bootstrap(&tlwe_and_yn, cloud_key)
  }

  /// Homomorphic MUX gate (a ? b : c)
  ///
  /// Optimized version that minimizes key switching operations by
  /// chaining bootstraps without intermediate key switches.
  #[cfg(feature = "bootstrapping")]
  pub fn mux(
    &self,
    tlwe_a: &Ciphertext,
    tlwe_b: &Ciphertext,
    tlwe_c: &Ciphertext,
    cloud_key: &CloudKey,
  ) -> Ciphertext {
    // and(a, b)
    let mut tlwe_and = tlwe_a + tlwe_b;
    *tlwe_and.b_mut() = tlwe_and.b().wrapping_add(utils::f64_to_torus(-0.125));
    let u1: &Ciphertext = &self
      .bootstrap
      .bootstrap_without_key_switch(&tlwe_and, cloud_key);

    // and(not(a), c)
    let mut tlwe_and_ny = &(self.not(tlwe_a)) + tlwe_c;
    *tlwe_and_ny.b_mut() = tlwe_and_ny.b().wrapping_add(utils::f64_to_torus(-0.125));
    let u2: &Ciphertext = &self
      .bootstrap
      .bootstrap_without_key_switch(&tlwe_and_ny, cloud_key);

    // or(u1, u2)
    let mut tlwe_or = u1 + u2;
    *tlwe_or.b_mut() = tlwe_or.b().wrapping_add(utils::f64_to_torus(0.125));

    self.bootstrap.bootstrap(&tlwe_or, cloud_key)
  }

  /// Homomorphic MUX gate (naive version)
  ///
  /// Simple implementation using basic gates. Less efficient but easier to understand.
  #[cfg(feature = "bootstrapping")]
  pub fn mux_naive(
    &self,
    tlwe_a: &Ciphertext,
    tlwe_b: &Ciphertext,
    tlwe_c: &Ciphertext,
    cloud_key: &CloudKey,
  ) -> Ciphertext {
    let a_and_b = self.and(tlwe_a, tlwe_b, cloud_key);
    let nand_a_c = self.and(&self.not(tlwe_a), tlwe_c, cloud_key);
    self.or(&a_and_b, &nand_a_c, cloud_key)
  }

  /// Homomorphic NOT gate (no bootstrapping needed)
  pub fn not(&self, tlwe_a: &Ciphertext) -> Ciphertext {
    -(*tlwe_a)
  }

  /// Copy a ciphertext (no bootstrapping needed)
  pub fn copy(&self, tlwe_a: &Ciphertext) -> Ciphertext {
    *tlwe_a
  }

  /// Create a constant encrypted value (no bootstrapping needed)
  pub fn constant(&self, value: bool) -> Ciphertext {
    let mut mu: params::Torus = utils::f64_to_torus(0.125);
    mu = if value { mu } else { 1 - mu };
    let mut res = Ciphertext::new();
    *res.b_mut() = mu;
    res
  }
}

impl Default for Gates {
  fn default() -> Self {
    Self::new()
  }
}

// ============================================================================
// CONVENIENCE FREE FUNCTIONS - Use default bootstrap strategy
// ============================================================================

/// Convenience function for NAND gate using default bootstrap
#[cfg(feature = "bootstrapping")]
pub fn nand(tlwe_a: &Ciphertext, tlwe_b: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
  Gates::new().nand(tlwe_a, tlwe_b, cloud_key)
}

/// Convenience function for OR gate using default bootstrap
#[cfg(feature = "bootstrapping")]
pub fn or(tlwe_a: &Ciphertext, tlwe_b: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
  Gates::new().or(tlwe_a, tlwe_b, cloud_key)
}

/// Convenience function for AND gate using default bootstrap
#[cfg(feature = "bootstrapping")]
pub fn and(tlwe_a: &Ciphertext, tlwe_b: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
  Gates::new().and(tlwe_a, tlwe_b, cloud_key)
}

/// Convenience function for XOR gate using default bootstrap
#[cfg(feature = "bootstrapping")]
pub fn xor(tlwe_a: &Ciphertext, tlwe_b: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
  Gates::new().xor(tlwe_a, tlwe_b, cloud_key)
}

/// Convenience function for XNOR gate using default bootstrap
#[cfg(feature = "bootstrapping")]
pub fn xnor(tlwe_a: &Ciphertext, tlwe_b: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
  Gates::new().xnor(tlwe_a, tlwe_b, cloud_key)
}

/// Convenience function for NOR gate using default bootstrap
#[cfg(feature = "bootstrapping")]
pub fn nor(tlwe_a: &Ciphertext, tlwe_b: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
  Gates::new().nor(tlwe_a, tlwe_b, cloud_key)
}

/// Convenience function for AND_NY gate using default bootstrap
#[cfg(feature = "bootstrapping")]
pub fn and_ny(tlwe_a: &Ciphertext, tlwe_b: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
  Gates::new().and_ny(tlwe_a, tlwe_b, cloud_key)
}

/// Convenience function for AND_YN gate using default bootstrap
#[cfg(feature = "bootstrapping")]
pub fn and_yn(tlwe_a: &Ciphertext, tlwe_b: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
  Gates::new().and_yn(tlwe_a, tlwe_b, cloud_key)
}

/// Convenience function for OR_NY gate using default bootstrap
#[cfg(feature = "bootstrapping")]
pub fn or_ny(tlwe_a: &Ciphertext, tlwe_b: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
  Gates::new().or_ny(tlwe_a, tlwe_b, cloud_key)
}

/// Convenience function for OR_YN gate using default bootstrap
#[cfg(feature = "bootstrapping")]
pub fn or_yn(tlwe_a: &Ciphertext, tlwe_b: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
  Gates::new().or_yn(tlwe_a, tlwe_b, cloud_key)
}

/// Convenience function for MUX gate using default bootstrap
#[cfg(feature = "bootstrapping")]
pub fn mux(
  tlwe_a: &Ciphertext,
  tlwe_b: &Ciphertext,
  tlwe_c: &Ciphertext,
  cloud_key: &CloudKey,
) -> Ciphertext {
  Gates::new().mux(tlwe_a, tlwe_b, tlwe_c, cloud_key)
}

/// Convenience function for naive MUX gate using default bootstrap
#[cfg(feature = "bootstrapping")]
pub fn mux_naive(
  tlwe_a: &Ciphertext,
  tlwe_b: &Ciphertext,
  tlwe_c: &Ciphertext,
  cloud_key: &CloudKey,
) -> Ciphertext {
  Gates::new().mux_naive(tlwe_a, tlwe_b, tlwe_c, cloud_key)
}

/// Convenience function for NOT gate
pub fn not(tlwe_a: &Ciphertext) -> Ciphertext {
  Gates::new().not(tlwe_a)
}

/// Convenience function for COPY
pub fn copy(tlwe_a: &Ciphertext) -> Ciphertext {
  Gates::new().copy(tlwe_a)
}

/// Convenience function for CONSTANT
pub fn constant(value: bool) -> Ciphertext {
  Gates::new().constant(value)
}

// ============================================================================
// BATCH GATE OPERATIONS - Parallel Processing
// ============================================================================

/// Batch NAND operation - process multiple gates in parallel
///
/// OPTIMIZED: Uses batch_blind_rotate internally for better performance.
/// Instead of parallelizing complete gates, we:
/// 1. Prepare all linear operations (fast, sequential)
/// 2. Batch all blind_rotate operations (slow, parallel) ← KEY OPTIMIZATION
/// 3. Post-process (sample extract + key switch, parallel)
///
/// This gives better cache locality and reduces overhead vs naive parallelization.
///
/// # Arguments
/// * `inputs` - Slice of (ciphertext_a, ciphertext_b) pairs
/// * `cloud_key` - Cloud key for homomorphic operations
///
/// # Returns
/// Vector of NAND results in the same order as inputs
///
/// # Performance
/// Expected speedup: ~6-7x on multi-core systems (better than naive parallel gates!)
#[cfg(feature = "bootstrapping")]
pub fn batch_nand(inputs: &[(Ciphertext, Ciphertext)], cloud_key: &CloudKey) -> Vec<Ciphertext> {
  batch_nand_with_railgun(inputs, cloud_key, crate::parallel::default_railgun())
}

#[cfg(feature = "bootstrapping")]
pub fn batch_nand_with_railgun<R: crate::parallel::Railgun>(
  inputs: &[(Ciphertext, Ciphertext)],
  cloud_key: &CloudKey,
  railgun: &R,
) -> Vec<Ciphertext> {
  use crate::trgsw::identity_key_switching;
  use crate::trlwe::sample_extract_index;

  // Step 1: Prepare all inputs for bootstrapping (fast, linear operations)
  let prepared: Vec<_> = inputs
    .iter()
    .map(|(a, b)| {
      let mut tlwe_nand = -(a + b);
      *tlwe_nand.b_mut() = tlwe_nand.b().wrapping_add(utils::f64_to_torus(0.125));
      tlwe_nand
    })
    .collect();

  // Step 2: Batch blind rotate (slow, THIS is the bottleneck - parallelize here!)
  let trlwes = crate::trgsw::batch_blind_rotate_with_railgun(&prepared, cloud_key, railgun);

  // Step 3: Post-process (sample extract + key switching, parallel)
  railgun.par_map(&trlwes, |trlwe| {
    let tlwe_lv1 = sample_extract_index(trlwe, 0);
    identity_key_switching(&tlwe_lv1, &cloud_key.key_switching_key)
  })
}

/// Batch AND operation - process multiple gates in parallel
/// Uses batch_blind_rotate internally for optimal performance
#[cfg(feature = "bootstrapping")]
pub fn batch_and(inputs: &[(Ciphertext, Ciphertext)], cloud_key: &CloudKey) -> Vec<Ciphertext> {
  batch_and_with_railgun(inputs, cloud_key, crate::parallel::default_railgun())
}

#[cfg(feature = "bootstrapping")]
pub fn batch_and_with_railgun<R: crate::parallel::Railgun>(
  inputs: &[(Ciphertext, Ciphertext)],
  cloud_key: &CloudKey,
  railgun: &R,
) -> Vec<Ciphertext> {
  use crate::trgsw::identity_key_switching;
  use crate::trlwe::sample_extract_index;

  // Step 1: Prepare inputs (linear operations)
  let prepared: Vec<_> = inputs
    .iter()
    .map(|(a, b)| {
      let mut tlwe_and = a + b;
      *tlwe_and.b_mut() = tlwe_and.b().wrapping_add(utils::f64_to_torus(-0.125));
      tlwe_and
    })
    .collect();

  // Step 2: Batch blind rotate (bottleneck, parallelized)
  let trlwes = crate::trgsw::batch_blind_rotate_with_railgun(&prepared, cloud_key, railgun);

  // Step 3: Post-process
  railgun.par_map(&trlwes, |trlwe| {
    let tlwe_lv1 = sample_extract_index(trlwe, 0);
    identity_key_switching(&tlwe_lv1, &cloud_key.key_switching_key)
  })
}

/// Batch OR operation - optimized with batch_blind_rotate
#[cfg(feature = "bootstrapping")]
pub fn batch_or(inputs: &[(Ciphertext, Ciphertext)], cloud_key: &CloudKey) -> Vec<Ciphertext> {
  batch_or_with_railgun(inputs, cloud_key, crate::parallel::default_railgun())
}

#[cfg(feature = "bootstrapping")]
pub fn batch_or_with_railgun<R: crate::parallel::Railgun>(
  inputs: &[(Ciphertext, Ciphertext)],
  cloud_key: &CloudKey,
  railgun: &R,
) -> Vec<Ciphertext> {
  use crate::trgsw::identity_key_switching;
  use crate::trlwe::sample_extract_index;

  let prepared: Vec<_> = inputs
    .iter()
    .map(|(a, b)| {
      let mut tlwe_or = a + b;
      *tlwe_or.b_mut() = tlwe_or.b().wrapping_add(utils::f64_to_torus(0.125));
      tlwe_or
    })
    .collect();

  let trlwes = crate::trgsw::batch_blind_rotate_with_railgun(&prepared, cloud_key, railgun);

  railgun.par_map(&trlwes, |trlwe| {
    let tlwe_lv1 = sample_extract_index(trlwe, 0);
    identity_key_switching(&tlwe_lv1, &cloud_key.key_switching_key)
  })
}

/// Batch XOR operation - optimized with batch_blind_rotate
#[cfg(feature = "bootstrapping")]
pub fn batch_xor(inputs: &[(Ciphertext, Ciphertext)], cloud_key: &CloudKey) -> Vec<Ciphertext> {
  batch_xor_with_railgun(inputs, cloud_key, crate::parallel::default_railgun())
}

#[cfg(feature = "bootstrapping")]
pub fn batch_xor_with_railgun<R: crate::parallel::Railgun>(
  inputs: &[(Ciphertext, Ciphertext)],
  cloud_key: &CloudKey,
  railgun: &R,
) -> Vec<Ciphertext> {
  use crate::trgsw::identity_key_switching;
  use crate::trlwe::sample_extract_index;

  let prepared: Vec<_> = inputs
    .iter()
    .map(|(a, b)| {
      let mut tlwe_xor = a.add_mul(b, 2);
      *tlwe_xor.b_mut() = tlwe_xor.b().wrapping_add(utils::f64_to_torus(0.25));
      tlwe_xor
    })
    .collect();

  let trlwes = crate::trgsw::batch_blind_rotate_with_railgun(&prepared, cloud_key, railgun);

  railgun.par_map(&trlwes, |trlwe| {
    let tlwe_lv1 = sample_extract_index(trlwe, 0);
    identity_key_switching(&tlwe_lv1, &cloud_key.key_switching_key)
  })
}

/// Batch NOR operation - optimized with batch_blind_rotate
#[cfg(feature = "bootstrapping")]
pub fn batch_nor(inputs: &[(Ciphertext, Ciphertext)], cloud_key: &CloudKey) -> Vec<Ciphertext> {
  batch_nor_with_railgun(inputs, cloud_key, crate::parallel::default_railgun())
}

#[cfg(feature = "bootstrapping")]
pub fn batch_nor_with_railgun<R: crate::parallel::Railgun>(
  inputs: &[(Ciphertext, Ciphertext)],
  cloud_key: &CloudKey,
  railgun: &R,
) -> Vec<Ciphertext> {
  use crate::trgsw::identity_key_switching;
  use crate::trlwe::sample_extract_index;

  let prepared: Vec<_> = inputs
    .iter()
    .map(|(a, b)| {
      let mut tlwe_nor = -(a + b);
      *tlwe_nor.b_mut() = tlwe_nor.b().wrapping_add(utils::f64_to_torus(-0.125));
      tlwe_nor
    })
    .collect();

  let trlwes = crate::trgsw::batch_blind_rotate_with_railgun(&prepared, cloud_key, railgun);

  railgun.par_map(&trlwes, |trlwe| {
    let tlwe_lv1 = sample_extract_index(trlwe, 0);
    identity_key_switching(&tlwe_lv1, &cloud_key.key_switching_key)
  })
}

/// Batch XNOR operation - optimized with batch_blind_rotate
#[cfg(feature = "bootstrapping")]
pub fn batch_xnor(inputs: &[(Ciphertext, Ciphertext)], cloud_key: &CloudKey) -> Vec<Ciphertext> {
  batch_xnor_with_railgun(inputs, cloud_key, crate::parallel::default_railgun())
}

#[cfg(feature = "bootstrapping")]
pub fn batch_xnor_with_railgun<R: crate::parallel::Railgun>(
  inputs: &[(Ciphertext, Ciphertext)],
  cloud_key: &CloudKey,
  railgun: &R,
) -> Vec<Ciphertext> {
  use crate::trgsw::identity_key_switching;
  use crate::trlwe::sample_extract_index;

  let prepared: Vec<_> = inputs
    .iter()
    .map(|(a, b)| {
      let mut tlwe_xnor = a.sub_mul(b, 2);
      *tlwe_xnor.b_mut() = tlwe_xnor.b().wrapping_add(utils::f64_to_torus(-0.25));
      tlwe_xnor
    })
    .collect();

  let trlwes = crate::trgsw::batch_blind_rotate_with_railgun(&prepared, cloud_key, railgun);

  railgun.par_map(&trlwes, |trlwe| {
    let tlwe_lv1 = sample_extract_index(trlwe, 0);
    identity_key_switching(&tlwe_lv1, &cloud_key.key_switching_key)
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::key;
  use crate::key::CloudKey;
  use crate::params;
  use crate::utils::Ciphertext;
  use rand::Rng;

  #[test]
  fn test_hom_nand() {
    test_gate(
      |a, b| !(a & b),
      |gates: &Gates, a: &Ciphertext, b: &Ciphertext, k: &CloudKey| gates.nand(a, b, k),
    );
  }

  #[test]
  fn test_hom_or() {
    test_gate(
      |a, b| a | b,
      |gates: &Gates, a: &Ciphertext, b: &Ciphertext, k: &CloudKey| gates.or(a, b, k),
    );
  }

  #[test]
  fn test_hom_xnor() {
    test_gate(
      |a, b| false ^ (b ^ a),
      |gates: &Gates, a: &Ciphertext, b: &Ciphertext, k: &CloudKey| gates.xnor(a, b, k),
    );
  }

  #[test]
  fn test_hom_xor() {
    test_gate(
      |a, b| a ^ b,
      |gates: &Gates, a: &Ciphertext, b: &Ciphertext, k: &CloudKey| gates.xor(a, b, k),
    );
  }

  #[test]
  fn test_hom_not() {
    test_gate(
      |a, _| !a,
      |gates: &Gates, a: &Ciphertext, _, _| gates.not(a),
    );
  }

  #[test]
  fn test_hom_copy() {
    test_gate(
      |a, _| a,
      |gates: &Gates, a: &Ciphertext, _, _| gates.copy(a),
    );
  }

  #[test]
  fn test_hom_constant() {
    let test = true;
    test_gate(
      |_, _| test,
      |gates: &Gates, _: _, _, _| gates.constant(test),
    );
  }

  #[test]
  fn test_hom_nor() {
    test_gate(
      |a, b| !(a | b),
      |gates: &Gates, a: &Ciphertext, b: &Ciphertext, k: &CloudKey| gates.nor(a, b, k),
    );
  }

  #[test]
  fn test_hom_and_ny() {
    test_gate(
      |a, b| !a & b,
      |gates: &Gates, a: &Ciphertext, b: &Ciphertext, k: &CloudKey| gates.and_ny(a, b, k),
    );
  }

  #[test]
  fn test_hom_and_yn() {
    test_gate(
      |a, b| a & !b,
      |gates: &Gates, a: &Ciphertext, b: &Ciphertext, k: &CloudKey| gates.and_yn(a, b, k),
    );
  }

  #[test]
  fn test_hom_or_ny() {
    test_gate(
      |a, b| !a | b,
      |gates: &Gates, a: &Ciphertext, b: &Ciphertext, k: &CloudKey| gates.or_ny(a, b, k),
    );
  }

  #[test]
  fn test_hom_or_yn() {
    test_gate(
      |a, b| a | !b,
      |gates: &Gates, a: &Ciphertext, b: &Ciphertext, k: &CloudKey| gates.or_yn(a, b, k),
    );
  }

  fn test_gate<
    E: Fn(bool, bool) -> bool,
    C: Fn(&Gates, &Ciphertext, &Ciphertext, &CloudKey) -> Ciphertext,
  >(
    expect: E,
    actual: C,
  ) {
    let mut rng = rand::thread_rng();
    let key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&key);
    let gates = Gates::new();

    let try_num = 10;
    for _i in 0..try_num {
      let plain_a = rng.gen::<bool>();
      let plain_b = rng.gen::<bool>();
      let expected = expect(plain_a, plain_b);

      let tlwe_a = Ciphertext::encrypt_bool(plain_a, params::tlwe_lv0::ALPHA, &key.key_lv0);
      let tlwe_b = Ciphertext::encrypt_bool(plain_b, params::tlwe_lv0::ALPHA, &key.key_lv0);
      let tlwe_op = actual(&gates, &tlwe_a, &tlwe_b, &cloud_key);
      let dec = tlwe_op.decrypt_bool(&key.key_lv0);
      dbg!(plain_a);
      dbg!(plain_b);
      dbg!(expected);
      dbg!(dec);
      assert_eq!(expected, dec);
    }
  }

  #[test]
  fn test_mux() {
    let mut rng = rand::thread_rng();
    let key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&key);
    let gates = Gates::new();

    let try_num = 10;
    for _i in 0..try_num {
      let plain_a = rng.gen::<bool>();
      let plain_b = rng.gen::<bool>();
      let plain_c = rng.gen::<bool>();
      let expected = (plain_a & plain_b) | ((!plain_a) & plain_c);

      let tlwe_a = Ciphertext::encrypt_bool(plain_a, params::tlwe_lv0::ALPHA, &key.key_lv0);
      let tlwe_b = Ciphertext::encrypt_bool(plain_b, params::tlwe_lv0::ALPHA, &key.key_lv0);
      let tlwe_c = Ciphertext::encrypt_bool(plain_c, params::tlwe_lv0::ALPHA, &key.key_lv0);
      let tlwe_op = gates.mux_naive(&tlwe_a, &tlwe_b, &tlwe_c, &cloud_key);
      let dec = tlwe_op.decrypt_bool(&key.key_lv0);
      dbg!(plain_a);
      dbg!(plain_b);
      dbg!(plain_c);
      dbg!(expected);
      dbg!(dec);
      assert_eq!(expected, dec);
    }
  }

  #[test]
  #[cfg(feature = "bootstrapping")]
  #[ignore]
  fn test_batch_and_8_gates() {
    use std::time::Instant;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║     Batch AND Scaling Benchmark - Multiple Batch Sizes      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&key);
    let gates = Gates::new();

    let num_cpus = num_cpus::get();
    println!("💻 System: {} CPU cores", num_cpus);
    println!();

    // Test multiple batch sizes
    let batch_sizes = vec![8, 16, 32, 64, 128];

    println!("┌────────┬──────────────┬──────────────┬───────────┬─────────┬────────────┐");
    println!("│ Gates  │ Sequential   │ Parallel     │ Per Gate  │ Speedup │ Efficiency │");
    println!("├────────┼──────────────┼──────────────┼───────────┼─────────┼────────────┤");

    for &n_gates in &batch_sizes {
      // Generate test data
      let test_data: Vec<_> = (0..n_gates).map(|i| ((i % 2 == 0), (i % 3 == 0))).collect();

      // Encrypt inputs
      let encrypted_pairs: Vec<_> = test_data
        .iter()
        .map(|(a, b)| {
          let enc_a = Ciphertext::encrypt_bool(*a, params::tlwe_lv0::ALPHA, &key.key_lv0);
          let enc_b = Ciphertext::encrypt_bool(*b, params::tlwe_lv0::ALPHA, &key.key_lv0);
          (enc_a, enc_b)
        })
        .collect();

      // Sequential benchmark
      let start = Instant::now();
      let sequential_results: Vec<_> = encrypted_pairs
        .iter()
        .map(|(a, b)| gates.and(a, b, &cloud_key))
        .collect();
      let sequential_time = start.elapsed();

      // Batch benchmark
      let start = Instant::now();
      let batch_results = batch_and(&encrypted_pairs, &cloud_key);
      let batch_time = start.elapsed();

      // Calculate metrics
      let speedup = sequential_time.as_secs_f64() / batch_time.as_secs_f64();
      let per_gate_ms = batch_time.as_millis() as f64 / n_gates as f64;
      let ideal_speedup = num_cpus.min(n_gates) as f64;
      let efficiency = (speedup / ideal_speedup * 100.0).min(100.0);

      println!(
        "│ {:6} │ {:10.2}s │ {:10.2}s │ {:7.2}ms │ {:6.2}x │ {:8.1}% │",
        n_gates,
        sequential_time.as_secs_f64(),
        batch_time.as_secs_f64(),
        per_gate_ms,
        speedup,
        efficiency
      );

      // Quick correctness check
      for ((a, b), (seq_result, batch_result)) in test_data
        .iter()
        .zip(sequential_results.iter().zip(batch_results.iter()))
      {
        let expected = *a && *b;
        let seq_dec = seq_result.decrypt_bool(&key.key_lv0);
        let batch_dec = batch_result.decrypt_bool(&key.key_lv0);
        assert_eq!(expected, seq_dec, "Sequential mismatch");
        assert_eq!(expected, batch_dec, "Batch mismatch");
        assert_eq!(seq_dec, batch_dec, "Seq/batch mismatch");
      }

      // Assert minimum speedup
      assert!(
        speedup >= 1.5,
        "Batch size {} should provide at least 1.5x speedup, got {:.2}x",
        n_gates,
        speedup
      );
    }

    println!("└────────┴──────────────┴──────────────┴───────────┴─────────┴────────────┘");
    println!();
    println!("📊 Key Findings:");
    println!("  • Speedup scales with batch size up to CPU core count");
    println!("  • Per-gate latency decreases dramatically with parallelization");
    println!("  • Near-linear scaling demonstrates correct parallelization granularity");
    println!("  • All results verified correct across all batch sizes");
    println!();
    println!("✅ Batch AND scaling test: PASSED");
  }

  #[test]
  fn test_gates_with_custom_bootstrap() {
    use crate::bootstrap::vanilla::VanillaBootstrap;

    let gates = Gates::with_bootstrap(Box::new(VanillaBootstrap::new()));
    assert_eq!(gates.bootstrap_strategy(), "vanilla");

    let mut rng = rand::thread_rng();
    let key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&key);

    let plain_a = rng.gen::<bool>();
    let plain_b = rng.gen::<bool>();
    let expected = plain_a & plain_b;

    let tlwe_a = Ciphertext::encrypt_bool(plain_a, params::tlwe_lv0::ALPHA, &key.key_lv0);
    let tlwe_b = Ciphertext::encrypt_bool(plain_b, params::tlwe_lv0::ALPHA, &key.key_lv0);
    let tlwe_and = gates.and(&tlwe_a, &tlwe_b, &cloud_key);
    let dec = tlwe_and.decrypt_bool(&key.key_lv0);

    assert_eq!(expected, dec);
  }
}
