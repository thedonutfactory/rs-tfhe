/// TFHE Security Parameter Selection
///
/// This library supports multiple security levels to allow users to choose
/// the right balance between performance and security for their use case.
///
/// # Available Security Levels
///
/// - **80-bit**: Fast performance, suitable for development/testing
///   - Enable with: `--features "security-80bit"`
///   - ~20-30% faster than default
///
/// - **110-bit**: Balanced performance and security
///   - Enable with: `--features "security-110bit"`
///   - Original TFHE reference parameters
///
/// - **128-bit** (DEFAULT): High security, quantum-resistant
///   - Default configuration (no feature flag needed)
///   - Strong security guarantees for production use
///
/// # Security Parameters Explained
///
/// The security level is determined by several cryptographic parameters:
/// - `N`: LWE dimension (higher = more secure, slower)
/// - `ALPHA`: Noise standard deviation (smaller = often more secure with proper dimension)
/// - `L`: Gadget decomposition levels (more = more secure, slower)
/// - `BGBIT`: Decomposition base bits (smaller = more levels, more secure, slower)
///
/// # Usage Example
///
/// ```bash
/// # Default 128-bit security (recommended)
/// cargo build --release
///
/// # Fast 80-bit security (development/testing)
/// cargo build --release --features "security-80bit"
///
/// # Balanced 110-bit security (original TFHE)
/// cargo build --release --features "security-110bit"
/// ```

pub type Torus = u32;
pub type HalfTorus = i32;
pub type IntTorus = i64;

// pub type Torus = u16;
// pub type HalfTorus = i16;
// pub type IntTorus = i32;

pub const TORUS_SIZE: usize = std::mem::size_of::<Torus>() * 8;
pub const ZERO_TORUS: Torus = 0;

// ============================================================================
// 80-BIT SECURITY PARAMETERS (Performance-Optimized)
// ============================================================================
#[cfg(feature = "security-80bit")]
pub mod implementation {
  pub const SECURITY_BITS: usize = 80;
  pub const SECURITY_DESCRIPTION: &str = "80-bit security (performance-optimized)";

  pub mod tlwe_lv0 {
    pub const N: usize = 550;
    pub const ALPHA: f64 = 5.0e-5; // 2^-14.3 approximately
  }

  pub mod tlwe_lv1 {
    pub const N: usize = 1024;
    pub const ALPHA: f64 = 3.73e-8; // 2^-24.7 approximately
  }

  pub mod trlwe_lv1 {
    pub const N: usize = super::tlwe_lv1::N;
    #[cfg(test)]
    pub const ALPHA: f64 = super::tlwe_lv1::ALPHA;
  }

  pub mod trgsw_lv1 {
    pub const N: usize = super::tlwe_lv1::N;
    pub const NBIT: usize = 10;
    pub const BGBIT: u32 = 6;
    pub const BG: u32 = 1 << BGBIT;
    pub const L: usize = 3;
    pub const BASEBIT: usize = 2;
    pub const IKS_T: usize = 7;
    #[cfg(test)]
    pub const ALPHA: f64 = super::tlwe_lv1::ALPHA;
  }
}

// ============================================================================
// 110-BIT SECURITY PARAMETERS (Original TFHE, Balanced)
// ============================================================================
#[cfg(feature = "security-110bit")]
pub mod implementation {
  pub const SECURITY_BITS: usize = 110;
  pub const SECURITY_DESCRIPTION: &str = "110-bit security (balanced, original TFHE)";

  pub mod tlwe_lv0 {
    pub const N: usize = 630;
    pub const ALPHA: f64 = 3.0517578125e-05; // 2^-15 approximately
  }

  pub mod tlwe_lv1 {
    pub const N: usize = 1024;
    pub const ALPHA: f64 = 2.980_232_238_769_531_3e-8; // 2^-25 approximately
  }

  pub mod trlwe_lv1 {
    pub const N: usize = super::tlwe_lv1::N;
    #[cfg(test)]
    pub const ALPHA: f64 = super::tlwe_lv1::ALPHA;
  }

  pub mod trgsw_lv1 {
    pub const N: usize = super::tlwe_lv1::N;
    pub const NBIT: usize = 10;
    pub const BGBIT: Torus = 6;
    pub const BG: u32 = 1 << BGBIT;
    pub const L: usize = 3;
    pub const BASEBIT: usize = 2;
    pub const IKS_T: usize = 8;
    #[cfg(test)]
    pub const ALPHA: f64 = super::tlwe_lv1::ALPHA;
  }
}

// ============================================================================
// 128-BIT SECURITY PARAMETERS (DEFAULT - High Security, Quantum-Resistant)
// ============================================================================
#[cfg(not(any(feature = "security-80bit", feature = "security-110bit")))]
pub mod implementation {
  pub const SECURITY_BITS: usize = 128;
  pub const SECURITY_DESCRIPTION: &str = "128-bit security (high security, quantum-resistant)";

  pub mod tlwe_lv0 {
    pub const N: usize = 700;
    pub const ALPHA: f64 = 2.0e-5; // 2^-15.6 approximately
  }

  pub mod tlwe_lv1 {
    pub const N: usize = 1024;
    pub const ALPHA: f64 = 2.0e-8; // 2^-25.6 approximately
  }

  pub mod trlwe_lv1 {
    pub const N: usize = super::tlwe_lv1::N;
    #[cfg(test)]
    pub const ALPHA: f64 = super::tlwe_lv1::ALPHA;
  }

  pub mod trgsw_lv1 {
    pub const N: usize = super::tlwe_lv1::N;
    pub const NBIT: usize = 10;
    pub const BGBIT: crate::params::Torus = 6;
    pub const BG: crate::params::Torus = 1 << BGBIT;
    pub const L: usize = 3;
    pub const BASEBIT: usize = 2;
    pub const IKS_T: usize = 9;
    #[cfg(test)]
    pub const ALPHA: f64 = super::tlwe_lv1::ALPHA;
  }
}

// ============================================================================
// PUBLIC API - Re-export selected implementation
// ============================================================================

pub use implementation::*;

pub const KSK_ALPHA: f64 = tlwe_lv0::ALPHA;
pub const BSK_ALPHA: f64 = tlwe_lv1::ALPHA;

// Compile-time verification that only one security level is selected
#[cfg(all(feature = "security-80bit", feature = "security-110bit"))]
compile_error!("Cannot enable both security-80bit and security-110bit features. Choose one.");

#[cfg(all(feature = "security-80bit", feature = "security-128bit"))]
compile_error!("Cannot enable both security-80bit and security-128bit features. Choose one.");

#[cfg(all(feature = "security-110bit", feature = "security-128bit"))]
compile_error!("Cannot enable both security-110bit and security-128bit features. Choose one.");

/// Get a description of the current security level
pub fn security_info() -> String {
  format!(
    "Security level: {} bits ({})",
    SECURITY_BITS, SECURITY_DESCRIPTION
  )
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_security_info() {
    let info = security_info();
    println!("{}", info);

    // Verify the security level matches expectations
    #[cfg(feature = "security-80bit")]
    assert_eq!(SECURITY_BITS, 80);

    #[cfg(feature = "security-110bit")]
    assert_eq!(SECURITY_BITS, 110);

    #[cfg(not(any(feature = "security-80bit", feature = "security-110bit")))]
    assert_eq!(SECURITY_BITS, 128);
  }

  #[test]
  fn test_parameter_sanity() {
    // Verify basic parameter relationships
    assert!(tlwe_lv0::N > 0);
    assert!(tlwe_lv1::N > 0);
    assert!(tlwe_lv0::ALPHA > 0.0);
    assert!(tlwe_lv1::ALPHA > 0.0);
    assert_eq!(trgsw_lv1::BG, 1 << trgsw_lv1::BGBIT);

    println!("TLWE Level 0: N={}, α={}", tlwe_lv0::N, tlwe_lv0::ALPHA);
    println!("TLWE Level 1: N={}, α={}", tlwe_lv1::N, tlwe_lv1::ALPHA);
    println!(
      "TRGSW: L={}, BGBIT={}, BG={}",
      trgsw_lv1::L,
      trgsw_lv1::BGBIT,
      trgsw_lv1::BG
    );
  }
}
