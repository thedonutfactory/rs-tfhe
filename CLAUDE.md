# rs-tfhe: Claude's Guide

## Overview

High-performance Rust implementation of TFHE (Torus Fully Homomorphic Encryption) - the reference implementation with advanced programmable bootstrapping capabilities.

## Key Features

- **Security Levels**: 80/110/128-bit + specialized Uint1-8 parameters
- **Performance**: SIMD FFT (AVX/FMA), Rayon parallelization, 2.7x faster than alternatives
- **Operations**: Complete boolean gates, fast arithmetic, programmable bootstrapping
- **Safety**: Rust's compile-time memory safety with zero-cost abstractions

## Architecture

```
TLWE → TRLWE → TRGSW → Homomorphic Gates
```

**Core Modules**: `params`, `key`, `tlwe`, `trlwe`, `trgsw`, `gates`, `bootstrap`, `lut`, `fft`, `utils`, `bit_utils`, `parallel`

## Quick Start

```toml
[dependencies]
rs_tfhe = "0.1.1"
```

```rust
use rs_tfhe::key;
use rs_tfhe::gates::Gates;
use rs_tfhe::utils::Ciphertext;

let secret_key = key::SecretKey::new();
let cloud_key = key::CloudKey::new(&secret_key);

let ct_true = Ciphertext::encrypt(true, &secret_key.key_lv0);
let ct_false = Ciphertext::encrypt(false, &secret_key.key_lv0);

let gates = Gates::new(&cloud_key);
let result = gates.hom_and(&ct_true, &ct_false);
let decrypted = result.decrypt(&secret_key.key_lv0);
assert_eq!(decrypted, false);
```

## Examples

```bash
cargo run --example add_two_numbers --release
cargo run --example lut_bootstrapping --features "lut-bootstrap" --release
```

## Security Parameters

```rust
use rs_tfhe::params;

let security = params::SECURITY_128_BIT;  // Default: high security
// params::SECURITY_80_BIT   // Fast, for dev/testing
// params::SECURITY_UINT4    // 4-bit messages (requires lut-bootstrap)
```

## Performance

- **Gate Operation**: ~15ms per gate
- **8-bit Addition**: ~50ms (3 bootstraps vs 8 for bit-by-bit)
- **vs Alternatives**: 2.7x faster baseline performance

## Feature Flags

```toml
rs_tfhe = { version = "0.1.1", features = ["lut-bootstrap", "fft_fma"] }
```

- `bootstrapping`: Enable bootstrapping operations (default)
- `lut-bootstrap`: Programmable bootstrapping with lookup tables
- `fft_avx`: AVX-optimized FFT (x86_64 only)
- `fft_fma`: FMA-optimized FFT (default)

## Testing

```bash
cargo test
cargo test --features "lut-bootstrap"
cargo bench
```

## Status (v0.1.1)

**Implemented**: Core TFHE, gates, vanilla/LUT bootstrapping, FFT, parallelization  
**Recent**: CRT bootstrap optimizations, advanced arithmetic operations  
**Performance**: 2.7x faster than alternatives, optimized for production use

## Key Patterns

**Memory Management**:
```rust
// Rust's ownership system handles memory automatically
let cloud_key = key::CloudKey::new(&secret_key);
// No explicit cleanup needed - RAII handles it
```

**Error Handling**:
```rust
let result = gates.hom_and(&ct_a, &ct_b)?;  // Returns Result<T, E>
```

**Feature Gating**:
```rust
#[cfg(feature = "lut-bootstrap")]
use rs_tfhe::bootstrap::lut::LutBootstrap;
```

## Related

- [zig-tfhe](https://github.com/thedonutfactory/zig-tfhe) - Zig port
- [go-tfhe](https://github.com/thedonutfactory/go-tfhe) - Go implementation
- [TFHE](https://tfhe.github.io/tfhe/) - Original C++ implementation
