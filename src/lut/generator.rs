//! Lookup table generator for programmable bootstrapping
//!
//! This module provides functionality to generate lookup tables from functions
//! for use in programmable bootstrapping operations.

use crate::params::{self, Torus};

use super::encoder::Encoder;
use super::lookup_table::LookupTable;

/// Generator for creating lookup tables from functions
///
/// The generator creates lookup tables that encode arbitrary functions
/// for evaluation during programmable bootstrapping.
#[derive(Debug, Clone)]
pub struct Generator {
  /// Encoder for message space
  encoder: Encoder,
  /// Polynomial degree (N from TRGSW parameters)
  poly_degree: usize,
  /// Lookup table size (equals poly_degree for standard TFHE)
  lookup_table_size: usize,
}

impl Generator {
  /// Create a new LUT generator
  ///
  /// # Arguments
  /// * `message_modulus` - Number of possible messages (e.g., 2 for binary)
  pub fn new(message_modulus: usize) -> Self {
    let poly_degree = params::trgsw_lv1::N;
    // For standard TFHE, lookup_table_size = poly_degree (poly_extend_factor = 1)
    // Only for extended configurations is lookup_table_size > poly_degree
    let lookup_table_size = poly_degree;

    Self {
      encoder: Encoder::new(message_modulus),
      poly_degree,
      lookup_table_size,
    }
  }

  /// Create a new LUT generator with custom scale
  ///
  /// # Arguments
  /// * `message_modulus` - Number of possible messages
  /// * `scale` - Custom scaling factor for encoding
  pub fn with_scale(message_modulus: usize, scale: f64) -> Self {
    let poly_degree = params::trgsw_lv1::N;
    let lookup_table_size = poly_degree; // Standard: lookup_table_size = poly_degree

    Self {
      encoder: Encoder::with_scale(message_modulus, scale),
      poly_degree,
      lookup_table_size,
    }
  }

  /// Generate a lookup table from a function
  ///
  /// # Arguments
  /// * `f` - Function to encode (maps message index to output value)
  ///
  /// # Returns
  /// Generated lookup table
  pub fn generate_lookup_table<F>(&self, f: F) -> LookupTable
  where
    F: Fn(usize) -> usize,
  {
    let mut lut = LookupTable::new();
    self.generate_lookup_table_assign(f, &mut lut);
    lut
  }

  /// Generate a lookup table and write to the provided output
  ///
  /// This is the core implementation of lookup table generation.
  /// The algorithm follows the tfhe-go reference implementation:
  ///
  /// 1. Create raw LUT buffer (size = lookup_table_size)
  /// 2. For each message x, fill range with encoded f(x)
  /// 3. Rotate by offset
  /// 4. Negate tail
  /// 5. Store in polynomial
  ///
  /// # Arguments
  /// * `f` - Function to encode
  /// * `lut_out` - Output lookup table to write to
  pub fn generate_lookup_table_assign<F>(&self, f: F, lut_out: &mut LookupTable)
  where
    F: Fn(usize) -> usize,
  {
    let message_modulus = self.encoder.message_modulus;

    // Create raw LUT buffer (size = lookup_table_size, which equals N for standard TFHE)
    let mut lut_raw = vec![0 as Torus; self.lookup_table_size];

    // Fill each message's range with encoded output
    for x in 0..message_modulus {
      let start = div_round(x * self.lookup_table_size, message_modulus);
      let end = div_round((x + 1) * self.lookup_table_size, message_modulus);

      // Apply function to message index
      let y = f(x);

      // Encode the output: message * scale
      // Use the same encoder as the input to maintain consistency
      let encoded_y = self.encoder.encode(y);

      // Fill range
      for xx in start..end {
        lut_raw[xx] = encoded_y;
      }
    }

    // Rotate by offset
    let offset = div_round(self.lookup_table_size, 2 * message_modulus);

    // Apply rotation
    let mut rotated = vec![0 as Torus; self.lookup_table_size];
    for i in 0..self.lookup_table_size {
      let src_idx = (i + offset) % self.lookup_table_size;
      rotated[i] = lut_raw[src_idx];
    }

    // Negate tail portion
    for i in (self.lookup_table_size - offset)..self.lookup_table_size {
      rotated[i] = rotated[i].wrapping_neg();
    }

    // Store in polynomial
    // For poly_extend_factor=1: just copy all lookup_table_size coefficients
    for i in 0..self.lookup_table_size {
      lut_out.poly.b[i] = rotated[i];
      lut_out.poly.a[i] = 0;
    }
  }

  /// Generate a lookup table from a function that returns Torus values
  ///
  /// # Arguments
  /// * `f` - Function that maps message index to Torus output value
  ///
  /// # Returns
  /// Generated lookup table
  pub fn generate_lookup_table_full<F>(&self, f: F) -> LookupTable
  where
    F: Fn(usize) -> Torus,
  {
    let mut lut = LookupTable::new();
    self.generate_lookup_table_full_assign(f, &mut lut);
    lut
  }

  /// Generate a lookup table with full control over Torus values
  ///
  /// # Arguments
  /// * `f` - Function that maps message index to Torus output value
  /// * `lut_out` - Output lookup table to write to
  pub fn generate_lookup_table_full_assign<F>(&self, f: F, lut_out: &mut LookupTable)
  where
    F: Fn(usize) -> Torus,
  {
    let message_modulus = self.encoder.message_modulus;

    let mut lut_raw = vec![0 as Torus; self.lookup_table_size];

    for x in 0..message_modulus {
      let start = div_round(x * self.lookup_table_size, message_modulus);
      let end = div_round((x + 1) * self.lookup_table_size, message_modulus);

      let y = f(x);

      for i in start..end {
        lut_raw[i] = y;
      }
    }

    let offset = div_round(self.lookup_table_size, 2 * message_modulus);
    let mut rotated = vec![0 as Torus; self.lookup_table_size];
    for i in 0..self.lookup_table_size {
      let src_idx = (i + offset) % self.lookup_table_size;
      rotated[i] = lut_raw[src_idx];
    }

    for i in (self.lookup_table_size - offset)..self.lookup_table_size {
      rotated[i] = rotated[i].wrapping_neg();
    }

    for i in 0..self.lookup_table_size {
      lut_out.poly.b[i] = rotated[i];
      lut_out.poly.a[i] = 0;
    }
  }

  /// Generate a lookup table with custom message modulus and scale
  ///
  /// # Arguments
  /// * `f` - Function to encode
  /// * `message_modulus` - Custom message modulus
  /// * `scale` - Custom scale factor
  ///
  /// # Returns
  /// Generated lookup table
  pub fn generate_lookup_table_custom<F>(
    &self,
    f: F,
    message_modulus: usize,
    scale: f64,
  ) -> LookupTable
  where
    F: Fn(usize) -> usize,
  {
    let mut lut = LookupTable::new();

    // Temporarily change encoder
    let _old_encoder = self.encoder.clone();
    let mut temp_generator = self.clone();
    temp_generator.encoder = Encoder::with_scale(message_modulus, scale);

    temp_generator.generate_lookup_table_assign(f, &mut lut);

    lut
  }

  /// Switch the modulus of x from Torus (2^32) to lookup_table_size
  ///
  /// For standard TFHE with lookup_table_size=N: result in [0, N)
  ///
  /// # Arguments
  /// * `x` - Torus value to convert
  ///
  /// # Returns
  /// Converted value in [0, lookup_table_size)
  pub fn mod_switch(&self, x: Torus) -> usize {
    let scaled = (x as f64) / (u32::MAX as f64) * (self.lookup_table_size as f64);
    let result = scaled.round() as usize % self.lookup_table_size;
    result
  }

  /// Get the message modulus
  pub fn message_modulus(&self) -> usize {
    self.encoder.message_modulus
  }

  /// Get the polynomial degree
  pub fn poly_degree(&self) -> usize {
    self.poly_degree
  }

  /// Get the lookup table size
  pub fn lookup_table_size(&self) -> usize {
    self.lookup_table_size
  }
}

/// Perform integer division with rounding
///
/// # Arguments
/// * `a` - Dividend
/// * `b` - Divisor
///
/// # Returns
/// Rounded result of a / b
fn div_round(a: usize, b: usize) -> usize {
  (a + b / 2) / b
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_generator_creation() {
    let generator = Generator::new(2);
    assert_eq!(generator.message_modulus(), 2);
    assert_eq!(generator.poly_degree(), params::trgsw_lv1::N);
    assert_eq!(generator.lookup_table_size(), params::trgsw_lv1::N);
  }

  #[test]
  fn test_identity_function() {
    let generator = Generator::new(2);
    let identity = |x: usize| x;

    let lut = generator.generate_lookup_table(identity);

    // The lookup table should not be empty
    assert!(!lut.is_empty());
  }

  #[test]
  fn test_not_function() {
    let generator = Generator::new(2);
    let not_func = |x: usize| 1 - x;

    let lut = generator.generate_lookup_table(not_func);

    // The lookup table should not be empty
    assert!(!lut.is_empty());
  }

  #[test]
  fn test_constant_function() {
    let generator = Generator::new(2);
    let constant_one = |_x: usize| 1;

    let lut = generator.generate_lookup_table(constant_one);

    // The lookup table should not be empty
    assert!(!lut.is_empty());
  }

  #[test]
  fn test_4bit_function() {
    let generator = Generator::new(4);
    let increment = |x: usize| (x + 1) % 4;

    let lut = generator.generate_lookup_table(increment);

    // The lookup table should not be empty
    assert!(!lut.is_empty());
  }

  #[test]
  fn test_custom_scale() {
    let generator = Generator::with_scale(2, 0.5);
    let identity = |x: usize| x;

    let lut = generator.generate_lookup_table(identity);

    // The lookup table should not be empty
    assert!(!lut.is_empty());
  }

  #[test]
  fn test_mod_switch() {
    let generator = Generator::new(2);

    // Test some values
    let result1 = generator.mod_switch(0);
    let result2 = generator.mod_switch(u32::MAX / 2);
    let result3 = generator.mod_switch(u32::MAX);

    assert!(result1 < generator.lookup_table_size());
    assert!(result2 < generator.lookup_table_size());
    assert!(result3 < generator.lookup_table_size());
  }

  #[test]
  fn test_div_round() {
    assert_eq!(div_round(5, 2), 3); // 5/2 = 2.5 -> 3
    assert_eq!(div_round(4, 2), 2); // 4/2 = 2.0 -> 2
    assert_eq!(div_round(3, 2), 2); // 3/2 = 1.5 -> 2
    assert_eq!(div_round(1, 2), 1); // 1/2 = 0.5 -> 1
    assert_eq!(div_round(0, 2), 0); // 0/2 = 0.0 -> 0
  }
}
