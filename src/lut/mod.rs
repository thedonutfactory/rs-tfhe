//! Lookup Table (LUT) support for programmable bootstrapping
//!
//! This module provides functionality for creating and using lookup tables
//! in programmable bootstrapping operations. LUT bootstrapping allows
//! evaluating arbitrary functions on encrypted data during the bootstrapping
//! process, combining noise refreshing with function evaluation.
//!
//! # Key Concepts
//!
//! - **Lookup Table**: A TRLWE ciphertext that encodes a function for evaluation
//! - **Programmable Bootstrapping**: Apply arbitrary functions during bootstrapping
//! - **Message Encoding**: Support for different message moduli (binary, multi-bit)
//! - **Function Evaluation**: Evaluate f(x) on encrypted x during bootstrapping
//!
//! # Example
//!
//! ```rust
//! use rs_tfhe::lut::{LookupTable, Generator};
//! use rs_tfhe::bootstrap::lut::LutBootstrap;
//! use rs_tfhe::key;
//! use rs_tfhe::utils::Ciphertext;
//! use rs_tfhe::params;
//!
//! // Create a generator for binary messages
//! let generator = Generator::new(2);
//!
//! // Define a function (e.g., NOT)
//! let not_func = |x: usize| 1 - x;
//!
//! // Generate lookup table
//! let lut = generator.generate_lookup_table(not_func);
//!
//! // Use in programmable bootstrapping
//! let bootstrap = LutBootstrap::new();
//! let secret_key = key::SecretKey::new();
//! let cloud_key = key::CloudKey::new(&secret_key);
//! let ciphertext = Ciphertext::encrypt_lwe_message(1, 2, params::tlwe_lv0::ALPHA, &secret_key.key_lv0);
//! let result = bootstrap.bootstrap_lut(&ciphertext, &lut, &cloud_key);
//! ```

pub mod encoder;
pub mod generator;
pub mod lookup_table;

pub use encoder::Encoder;
pub use generator::Generator;
pub use lookup_table::LookupTable;
