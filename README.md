# rs-tfhe: Rust TFHE Library

[![Crates.io](https://img.shields.io/crates/v/rs_tfhe.svg)](https://crates.io/crates/rs_tfhe)
[![Documentation](https://docs.rs/rs_tfhe/badge.svg)](https://docs.rs/rs_tfhe)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/thedonutfactory/rs-tfhe)
[![DAITU](https://img.shields.io/badge/AI-Assisted-blue.svg)](DAITU)

A high-performance Rust implementation of TFHE (Torus Fully Homomorphic Encryption).

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/abf6fd0f-b6bf-4e1a-97c7-4d7871e3db35" />

## Overview

rs-tfhe is a comprehensive homomorphic encryption library that enables computation on encrypted data without decryption, built in Rust for performance and safety.

> Not the language you were looking for? Check out our [go](https://github.com/thedonutfactory/go-tfhe) or [zig](https://github.com/thedonutfactory/zig-tfhe) sister projects

### Key Features

![LUT Bootstrapping](https://img.shields.io/badge/LUT--Bootstrapping-Enabled-blue.svg)
![SIMD FFT](https://img.shields.io/badge/SIMD--FFT-AVX%2FFMA-red.svg)
![Parallel Processing](https://img.shields.io/badge/Parallel-Rayon-green.svg)
![Security Levels](https://img.shields.io/badge/Security-80%2C110%2C128--bit-purple.svg)

- **Multiple Security Levels**: 80-bit, 110-bit, and 128-bit security parameters
- **Specialized Uint Parameters**: Optimized parameter sets for different message moduli (1-8 bits)
- **Homomorphic Gates**: Complete set of boolean operations (AND, OR, NAND, NOR, XOR, XNOR, NOT, MUX)
- **Fast Arithmetic**: Efficient multi-bit arithmetic operations using nibble-based addition
- **Parallel Processing**: Rayon-based parallelization for batch operations
- **Optimized FFT**: Multiple FFT implementations including SIMD optimizations
- **Feature Flags**: Modular compilation with optional features

## Installation

Add rs-tfhe to your `Cargo.toml`:

```toml
[dependencies]
rs_tfhe = "0.1.1"
```

### Feature Flags

```toml
[dependencies]
rs_tfhe = { version = "0.1.1", features = ["lut-bootstrap", "fft_fma"] }
```

Available features:
- `bootstrapping`: Enable bootstrapping operations (default)
- `lut-bootstrap`: Enable programmable bootstrapping with lookup tables
- `fft_avx`: Enable AVX-optimized FFT (x86_64 only)
- `fft_fma`: Enable FMA-optimized FFT (default)

## Quick Start

### Basic Homomorphic Operations

```rust
use rs_tfhe::key;
use rs_tfhe::gates::Gates;
use rs_tfhe::utils::Ciphertext;

// Generate keys
let secret_key = key::SecretKey::new();
let cloud_key = key::CloudKey::new(&secret_key);

// Encrypt boolean values
let ct_true = Ciphertext::encrypt(true, &secret_key.key_lv0);
let ct_false = Ciphertext::encrypt(false, &secret_key.key_lv0);

// Perform homomorphic operations
let gates = Gates::new(&cloud_key);
let result = gates.hom_and(&ct_true, &ct_false);

// Decrypt result
let decrypted = result.decrypt(&secret_key.key_lv0);
assert_eq!(decrypted, false);
```

### Programmable Bootstrapping

```rust
#[cfg(feature = "lut-bootstrap")]
use rs_tfhe::bootstrap::lut::LutBootstrap;
#[cfg(feature = "lut-bootstrap")]
use rs_tfhe::lut::Generator;

#[cfg(feature = "lut-bootstrap")]
fn programmable_bootstrap_example() {
    let secret_key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&secret_key);
    let bootstrap = LutBootstrap::new();
    
    // Encrypt a value
    let encrypted = Ciphertext::encrypt_lwe_message(5, 8, 0.0001, &secret_key.key_lv0);
    
    // Define a function to evaluate (square function)
    let square_func = |x: usize| (x * x) % 8;
    
    // Apply function during bootstrapping
    let result = bootstrap.bootstrap_func(&encrypted, square_func, 8, &cloud_key);
    
    // Decrypt result
    let decrypted = result.decrypt_lwe_message(8, &secret_key.key_lv0);
    assert_eq!(decrypted, 1); // 5^2 mod 8 = 25 mod 8 = 1
}
```

### Fast Arithmetic with LUT Bootstrapping

```rust
#[cfg(feature = "lut-bootstrap")]
fn fast_addition_example() {
    use rs_tfhe::params;
    
    // Use specialized parameters for arithmetic
    let current_params = params::SECURITY_128_BIT;
    
    let secret_key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&secret_key);
    let bootstrap = LutBootstrap::new();
    
    // Encrypt two 4-bit values
    let a = 5;
    let b = 7;
    let ct_a = Ciphertext::encrypt_lwe_message(a, 16, current_params.tlwe_lv0.alpha, &secret_key.key_lv0);
    let ct_b = Ciphertext::encrypt_lwe_message(b, 16, current_params.tlwe_lv0.alpha, &secret_key.key_lv0);
    
    // Homomorphic addition
    let ct_sum = &ct_a + &ct_b;
    
    // Extract result using LUT bootstrapping
    let mod_func = |x: usize| x % 16;
    let result = bootstrap.bootstrap_func(&ct_sum, mod_func, 16, &cloud_key);
    
    let decrypted = result.decrypt_lwe_message(16, &secret_key.key_lv0);
    assert_eq!(decrypted, (a + b) % 16);
}
```

## Architecture

### Core Components

#### Encryption Schemes
- **TLWE**: Torus Learning With Errors for level-0 ciphertexts
- **TRLWE**: Torus Ring Learning With Errors for level-1 ciphertexts
- **TRGSW**: Torus GSW for bootstrapping keys

#### Bootstrapping Strategies
- **Vanilla Bootstrap**: Traditional noise refreshing
- **LUT Bootstrap**: Programmable bootstrapping with lookup tables

#### FFT Implementations
- **Standard FFT**: Pure Rust implementation
- **SIMD FFT**: AVX/FMA optimized for x86_64
- **Real FFT**: Optimized for real-valued polynomials

### Parameter Sets

#### Standard Security Parameters
- `SECURITY_80_BIT`: 80-bit security level
- `SECURITY_110_BIT`: 110-bit security level  
- `SECURITY_128_BIT`: 128-bit security level (default)

#### Specialized Uint Parameters (requires `lut-bootstrap` feature)
- `SECURITY_UINT1`: Binary operations (messageModulus=2)
- `SECURITY_UINT2`: 2-bit arithmetic (messageModulus=4)
- `SECURITY_UINT3`: 3-bit arithmetic (messageModulus=8)
- `SECURITY_UINT4`: 4-bit arithmetic (messageModulus=16)
- `SECURITY_UINT5`: 5-bit arithmetic (messageModulus=32) - Recommended for complex operations
- `SECURITY_UINT6`: 6-bit arithmetic (messageModulus=64)
- `SECURITY_UINT7`: 7-bit arithmetic (messageModulus=128)
- `SECURITY_UINT8`: 8-bit arithmetic (messageModulus=256)

## Examples

The `examples/` directory contains comprehensive examples:

### Basic Examples
- `add_two_numbers.rs`: Simple homomorphic addition
- `gates_with_strategies.rs`: Boolean gate operations
- `security_levels.rs`: Different security parameter comparisons

### LUT Bootstrapping Examples
- `lut_bootstrapping.rs`: Complete programmable bootstrapping demo
- `lut_bootstrapping_simple.rs`: Minimal LUT example
- `lut_add_two_numbers.rs`: Fast 8-bit addition using nibble operations
- `lut_arithmetic_demo.rs`: Various arithmetic operations
- `lut_uint_parameters_demo.rs`: Parameter set comparisons

### Performance Examples
- `batch_gates.rs`: Parallel gate processing
- `custom_railgun.rs`: Custom parallelization strategies
- `fft_diagnostics.rs`: FFT performance analysis

## Performance

![Benchmarks](https://img.shields.io/badge/Benchmarks-Criterion-orange.svg)
![Speedup](https://img.shields.io/badge/Speedup-2.7x--faster-brightgreen.svg)

### Benchmarks

Run benchmarks with:

```bash
cargo bench
```

### Performance Characteristics

| Operation | Time (ms) | Notes |
|-----------|-----------|-------|
| Key Generation | ~135 | One-time setup |
| Boolean Gate | ~15 | Per gate operation |
| Bootstrap | ~15-20 | Noise refreshing |
| LUT Bootstrap | ~15-20 | Function evaluation + noise refreshing |
| 8-bit Addition | ~50 | 3 bootstraps vs 8 for bit-by-bit |

### Optimization Features

- **Parallel Processing**: Rayon-based batch operations
- **SIMD FFT**: AVX/FMA optimizations for x86_64
- **Specialized Parameters**: Optimized for specific message moduli
- **LUT Reuse**: Pre-computed lookup tables for repeated functions

## API Reference

### Core Types

#### `Ciphertext`
Main ciphertext type supporting homomorphic operations.

```rust
impl Ciphertext {
    pub fn encrypt(plaintext: bool, key: &SecretKey) -> Self;
    pub fn decrypt(&self, key: &SecretKey) -> bool;
    pub fn encrypt_lwe_message(msg: usize, modulus: usize, alpha: f64, key: &SecretKey) -> Self;
    pub fn decrypt_lwe_message(&self, modulus: usize, key: &SecretKey) -> usize;
}
```

#### `Gates`
Boolean gate operations.

```rust
impl Gates {
    pub fn hom_and(&self, a: &Ciphertext, b: &Ciphertext) -> Ciphertext;
    pub fn hom_or(&self, a: &Ciphertext, b: &Ciphertext) -> Ciphertext;
    pub fn hom_xor(&self, a: &Ciphertext, b: &Ciphertext) -> Ciphertext;
    pub fn hom_not(&self, a: &Ciphertext) -> Ciphertext;
    pub fn mux(&self, cond: &Ciphertext, a: &Ciphertext, b: &Ciphertext) -> Ciphertext;
}
```

#### `LutBootstrap` (requires `lut-bootstrap` feature)
Programmable bootstrapping with lookup tables.

```rust
impl LutBootstrap {
    pub fn bootstrap_func<F>(&self, ct: &Ciphertext, f: F, modulus: usize, key: &CloudKey) -> Ciphertext
    where F: Fn(usize) -> usize;
    
    pub fn bootstrap_lut(&self, ct: &Ciphertext, lut: &LookupTable, key: &CloudKey) -> Ciphertext;
}
```

#### `Generator` (requires `lut-bootstrap` feature)
Lookup table generation.

```rust
impl Generator {
    pub fn new(message_modulus: usize) -> Self;
    pub fn generate_lookup_table<F>(&self, f: F) -> LookupTable
    where F: Fn(usize) -> usize;
}
```

## Contributing

Contributions are welcome! Please see the existing code style and add tests for new functionality.

### Development Setup

```bash
git clone <repository>
cd rs-tfhe
cargo test
cargo test --features "lut-bootstrap"
cargo bench
```

### Running Examples

```bash
# Basic examples
cargo run --example add_two_numbers --release
cargo run --example gates_with_strategies --release

# LUT bootstrapping examples (requires feature flag)
cargo run --example lut_bootstrapping --features "lut-bootstrap" --release
cargo run --example lut_add_two_numbers --features "lut-bootstrap" --release
```

## License

This project is licensed under the same terms as the original TFHE library. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Based on the TFHE library by Ilaria Chillotti, Nicolas Gama, Mariya Georgieva, and Malika Izabachène
- Inspired by the go-tfhe implementation
- FFT optimizations from tfhe-go reference implementation
