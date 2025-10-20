//! Rayon-based implementation of the Railgun trait
//!
//! This module provides a CPU-parallel implementation using the Rayon library.
//! It's the default parallelization backend for rs-tfhe.

use super::{ParallelConfig, Railgun};
use rayon::prelude::*;

/// Rayon-based parallelization backend
///
/// This implementation uses Rayon's work-stealing thread pool for
/// efficient CPU parallelization.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RayonRailgun {
  config: ParallelConfig,
}

impl RayonRailgun {
  /// Create a new Rayon backend with default configuration
  pub fn new() -> Self {
    Self {
      config: ParallelConfig::default(),
    }
  }

  /// Create a new Rayon backend with custom configuration
  pub fn with_config(config: ParallelConfig) -> Self {
    Self { config }
  }
}

impl Default for RayonRailgun {
  fn default() -> Self {
    Self::new()
  }
}

impl Railgun for RayonRailgun {
  fn par_map<T, U, F>(&self, input: &[T], f: F) -> Vec<U>
  where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync + Send,
  {
    input.par_iter().map(f).collect()
  }

  fn par_map_indexed<T, U, F>(&self, input: &[T], f: F) -> Vec<U>
  where
    T: Sync,
    U: Send,
    F: Fn(usize, &T) -> U + Sync + Send,
  {
    input.par_iter().enumerate().map(|(i, x)| f(i, x)).collect()
  }

  fn with_config<F, R>(&self, config: ParallelConfig, f: F) -> R
  where
    F: Fn() -> R + Send + Sync,
    R: Send,
  {
    // Build a custom thread pool with the specified configuration
    let mut builder = rayon::ThreadPoolBuilder::new();

    if let Some(stack_size) = config.stack_size {
      builder = builder.stack_size(stack_size);
    }

    if let Some(num_threads) = config.num_threads {
      builder = builder.num_threads(num_threads);
    }

    let pool = builder.build().unwrap();
    pool.install(f)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_rayon_par_map() {
    let railgun = RayonRailgun::new();
    let input = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let result = railgun.par_map(&input, |x| x * x);
    assert_eq!(result, vec![1, 4, 9, 16, 25, 36, 49, 64]);
  }

  #[test]
  fn test_rayon_large_stack() {
    let railgun = RayonRailgun::new();
    let config = ParallelConfig {
      stack_size: Some(16 * 1024 * 1024), // 16MB
      num_threads: Some(4),
    };

    let result = railgun.with_config(config, || {
      // This would require the large stack
      let data: Vec<i32> = (0..1000).collect();
      data.par_iter().map(|x| x * 2).sum::<i32>()
    });

    assert_eq!(result, 999000);
  }

  #[test]
  fn test_rayon_indexed() {
    let railgun = RayonRailgun::new();
    let input = vec!["a", "b", "c"];
    let result = railgun.par_map_indexed(&input, |i, s| format!("{}{}", i, s));
    assert_eq!(result, vec!["0a", "1b", "2c"]);
  }
}
