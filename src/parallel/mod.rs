//! Parallelization abstraction layer for rs-tfhe
//!
//! This module provides the `Railgun` trait that abstracts over different parallelization
//! backends (Rayon, CUDA, etc.). This allows the library to swap between different parallel
//! execution strategies without changing the core algorithm implementations.
//!
//! # Architecture
//!
//! The approach uses generics with a simple trait bound, making it zero-cost while still
//! allowing different implementations to be swapped at compile time or through configuration.

use std::sync::OnceLock;

pub mod rayon_impl;

pub use rayon_impl::RayonRailgun;

/// Configuration for parallel execution
#[derive(Debug, Clone)]
pub struct ParallelConfig {
  /// Stack size per thread (in bytes)
  pub stack_size: Option<usize>,
  /// Number of threads (None = automatic)
  pub num_threads: Option<usize>,
}

impl Default for ParallelConfig {
  fn default() -> Self {
    Self {
      stack_size: Some(8 * 1024 * 1024), // 8MB default for FHE operations
      num_threads: None,
    }
  }
}

/// Trait for parallelization backends
///
/// This trait abstracts over different parallelization strategies, allowing
/// the library to be portable across different execution environments.
///
/// Implementations should be zero-cost and compile-time dispatched.
pub trait Railgun: Clone + Send + Sync {
  /// Parallel map over a slice with a function
  ///
  /// # Arguments
  /// * `input` - Input slice to process
  /// * `f` - Function to apply to each element
  ///
  /// # Returns
  /// Vector of results
  fn par_map<T, U, F>(&self, input: &[T], f: F) -> Vec<U>
  where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync + Send;

  /// Parallel map over a slice with indexed function
  ///
  /// # Arguments
  /// * `input` - Input slice to process
  /// * `f` - Function that takes (index, element) and returns result
  fn par_map_indexed<T, U, F>(&self, input: &[T], f: F) -> Vec<U>
  where
    T: Sync,
    U: Send,
    F: Fn(usize, &T) -> U + Sync + Send;

  /// Execute a closure with a custom parallel configuration
  ///
  /// This allows operations that need specific thread pool settings
  /// (e.g., larger stack sizes for deep recursion).
  fn with_config<F, R>(&self, config: ParallelConfig, f: F) -> R
  where
    F: Fn() -> R + Send + Sync,
    R: Send;
}

/// Global default parallelization backend (singleton)
///
/// This is used by default throughout the library unless explicitly overridden.
/// It uses Rayon with sensible defaults for FHE operations.
///
/// # Performance
///
/// This singleton is implemented using `std::sync::OnceLock` for zero-cost access:
/// - First call: Initializes the singleton (one-time cost)
/// - Subsequent calls: Simply returns a reference (zero overhead)
/// - Thread-safe without locks after initialization
/// - No heap allocations after first call
///
/// This design eliminates any overhead from repeatedly calling `default_railgun()`,
/// making it suitable for use in hot paths and benchmarks.
static DEFAULT_RAILGUN: OnceLock<RayonRailgun> = OnceLock::new();

pub fn default_railgun() -> &'static RayonRailgun {
  DEFAULT_RAILGUN.get_or_init(|| RayonRailgun::default())
}

/// Create a custom Rayon-based parallelization backend
pub fn rayon_railgun(config: ParallelConfig) -> RayonRailgun {
  RayonRailgun::with_config(config)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_par_map() {
    let railgun = default_railgun();
    let input = vec![1, 2, 3, 4, 5];
    let result = railgun.par_map(&input, |x| x * 2);
    assert_eq!(result, vec![2, 4, 6, 8, 10]);
  }

  #[test]
  fn test_par_map_indexed() {
    let railgun = default_railgun();
    let input = vec![10, 20, 30];
    let result = railgun.par_map_indexed(&input, |i, x| i + x);
    assert_eq!(result, vec![10, 21, 32]);
  }

  #[test]
  fn test_with_config() {
    let railgun = default_railgun();
    let config = ParallelConfig {
      stack_size: Some(4 * 1024 * 1024),
      num_threads: Some(2),
    };
    let result = railgun.with_config(config, || {
      // Simulate some work
      42
    });
    assert_eq!(result, 42);
  }
}
