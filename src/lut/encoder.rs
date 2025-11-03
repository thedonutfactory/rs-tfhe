//! Message encoding for lookup tables
//!
//! This module provides encoding and decoding functions for different message spaces
//! used in programmable bootstrapping.

use crate::params::Torus;
use crate::utils;

/// Encoder for different message spaces in programmable bootstrapping
///
/// The encoder handles the conversion between integer messages and torus values
/// used in the lookup table generation process.
#[derive(Debug, Clone)]
pub struct Encoder {
  /// Number of possible messages (e.g., 2 for binary, 4 for 2-bit)
  pub message_modulus: usize,
  /// Scaling factor for encoding
  pub scale: f64,
}

impl Encoder {
  /// Create a new encoder with the given message modulus
  ///
  /// For binary (boolean) operations, use `message_modulus=2`.
  /// The default encoding uses `1/(2*message_modulus)` to place messages in the torus.
  ///
  /// # Arguments
  /// * `message_modulus` - Number of possible messages (e.g., 2 for binary)
  pub fn new(message_modulus: usize) -> Self {
    // For TFHE, binary messages are encoded as ±1/8
    // Message 0 (false) -> -1/8 = 7/8 in unsigned representation
    // Message 1 (true) -> +1/8
    //
    // For general case with messageModulus m, we use ±1/(2m)
    // This gives us 1/4 for binary (m=2)
    let scale = 1.0 / (2.0 * message_modulus as f64);

    Self {
      message_modulus,
      scale,
    }
  }

  /// Create a new encoder with custom message modulus and scale
  ///
  /// # Arguments
  /// * `message_modulus` - Number of possible messages
  /// * `scale` - Custom scaling factor for encoding
  pub fn with_scale(message_modulus: usize, scale: f64) -> Self {
    Self {
      message_modulus,
      scale,
    }
  }

  /// Encode an integer message into a torus value
  ///
  /// The message should be in range [0, message_modulus).
  /// For TFHE bootstrapping, the encoding is: `message * scale`
  ///
  /// # Arguments
  /// * `message` - Integer message to encode
  ///
  /// # Returns
  /// Encoded torus value
  pub fn encode(&self, message: usize) -> Torus {
    // Normalize message to [0, message_modulus)
    let message = message % self.message_modulus;

    // Encode as message * scale
    let value = message as f64 * self.scale;
    utils::f64_to_torus(value)
  }

  /// Encode with a custom scale factor
  ///
  /// # Arguments
  /// * `message` - Integer message to encode
  /// * `scale` - Custom scale factor
  ///
  /// # Returns
  /// Encoded torus value
  pub fn encode_with_scale(&self, message: usize, scale: f64) -> Torus {
    let message = message % self.message_modulus;
    let value = message as f64 * scale;
    utils::f64_to_torus(value)
  }

  /// Decode a torus value back to an integer message
  ///
  /// # Arguments
  /// * `value` - Torus value to decode
  ///
  /// # Returns
  /// Decoded integer message
  pub fn decode(&self, value: Torus) -> usize {
    // Convert torus to float
    let f = utils::torus_to_f64(value);

    // Round to nearest message
    let message = (f / self.scale + 0.5) as usize;

    // Normalize to [0, message_modulus)
    message % self.message_modulus
  }

  /// Decode a torus value to a boolean (for binary messages)
  ///
  /// # Arguments
  /// * `value` - Torus value to decode
  ///
  /// # Returns
  /// Decoded boolean value
  pub fn decode_bool(&self, value: Torus) -> bool {
    self.decode(value) != 0
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_binary_encoder() {
    let encoder = Encoder::new(2);

    // Test encoding
    let encoded_0 = encoder.encode(0);
    let encoded_1 = encoder.encode(1);

    // Test decoding
    assert_eq!(encoder.decode(encoded_0), 0);
    assert_eq!(encoder.decode(encoded_1), 1);

    // Test boolean decoding
    assert!(!encoder.decode_bool(encoded_0));
    assert!(encoder.decode_bool(encoded_1));
  }

  #[test]
  fn test_4bit_encoder() {
    let encoder = Encoder::new(4);

    for i in 0..4 {
      let encoded = encoder.encode(i);
      let decoded = encoder.decode(encoded);
      assert_eq!(decoded, i);
    }
  }

  #[test]
  fn test_custom_scale() {
    let encoder = Encoder::with_scale(2, 0.5);

    let encoded_0 = encoder.encode(0);
    let encoded_1 = encoder.encode(1);

    assert_eq!(encoder.decode(encoded_0), 0);
    assert_eq!(encoder.decode(encoded_1), 1);
  }
}
