/// TFHE Security Parameter Selection
///
/// This library supports multiple security levels to allow users to choose
/// the right balance between performance and security for their use case.
///
/// # Available Security Levels
///
/// - **80-bit**: Fast performance, suitable for development/testing
/// - **110-bit**: Balanced performance and security (original TFHE reference)
/// - **128-bit**: High security, quantum-resistant (default)
/// - **Uint1-Uint8**: Specialized parameters for different message moduli
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
/// ```rust
/// use rs_tfhe::params::SECURITY_128_BIT;
///
/// // Use 128-bit security (default)
/// let params = SECURITY_128_BIT;
/// ```
///
/// # LUT Bootstrapping Parameters
///
/// ```rust
/// #[cfg(feature = "lut-bootstrap")]
/// use rs_tfhe::params::SECURITY_UINT5;
///
/// #[cfg(feature = "lut-bootstrap")]
/// // Use Uint5 parameters for complex arithmetic
/// let params = SECURITY_UINT5;
/// ```
pub type Torus = u32;
pub type HalfTorus = i32;
pub type IntTorus = i64;

pub const TORUS_SIZE: usize = std::mem::size_of::<Torus>() * 8;
pub const ZERO_TORUS: Torus = 0;

// ============================================================================
// PARAMETER STRUCTURE
// ============================================================================

/// Security parameter set containing all TFHE parameters
#[derive(Debug, Clone, Copy)]
pub struct SecurityParams {
  pub security_bits: usize,
  pub description: &'static str,
  pub tlwe_lv0: TlweParams,
  pub tlwe_lv1: TlweParams,
  pub trlwe_lv1: TrlweParams,
  pub trgsw_lv1: TrgswParams,
}

#[derive(Debug, Clone, Copy)]
pub struct TlweParams {
  pub n: usize,
  pub alpha: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct TrlweParams {
  pub n: usize,
  pub alpha: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct TrgswParams {
  pub n: usize,
  pub nbit: usize,
  pub bgbit: u32,
  pub bg: u32,
  pub l: usize,
  pub basebit: usize,
  pub iks_t: usize,
  pub alpha: f64,
}

// ============================================================================
// SECURITY PARAMETER CONSTANTS
// ============================================================================

/// 80-bit security parameters (performance-optimized)
pub const SECURITY_80_BIT: SecurityParams = SecurityParams {
  security_bits: 80,
  description: "80-bit security (performance-optimized)",
  tlwe_lv0: TlweParams {
    n: 550,
    alpha: 5.0e-5, // 2^-14.3 approximately
  },
  tlwe_lv1: TlweParams {
    n: 1024,
    alpha: 3.73e-8, // 2^-24.7 approximately
  },
  trlwe_lv1: TrlweParams {
    n: 1024,
    alpha: 3.73e-8,
  },
  trgsw_lv1: TrgswParams {
    n: 1024,
    nbit: 10,
    bgbit: 6,
    bg: 64,
    l: 3,
    basebit: 2,
    iks_t: 7,
    alpha: 3.73e-8,
  },
};

/// 110-bit security parameters (balanced, original TFHE)
pub const SECURITY_110_BIT: SecurityParams = SecurityParams {
  security_bits: 110,
  description: "110-bit security (balanced, original TFHE)",
  tlwe_lv0: TlweParams {
    n: 630,
    alpha: 3.0517578125e-05, // 2^-15 approximately
  },
  tlwe_lv1: TlweParams {
    n: 1024,
    alpha: 2.980_232_238_769_531_3e-8, // 2^-25 approximately
  },
  trlwe_lv1: TrlweParams {
    n: 1024,
    alpha: 2.980_232_238_769_531_3e-8,
  },
  trgsw_lv1: TrgswParams {
    n: 1024,
    nbit: 10,
    bgbit: 6,
    bg: 64,
    l: 3,
    basebit: 2,
    iks_t: 8,
    alpha: 2.980_232_238_769_531_3e-8,
  },
};

/// Uint1 parameters (1-bit binary/boolean, messageModulus=2)
#[cfg(feature = "lut-bootstrap")]
pub const SECURITY_UINT1: SecurityParams = SecurityParams {
  security_bits: 1,
  description: "Uint1 parameters (1-bit binary/boolean, messageModulus=2, N=1024)",
  tlwe_lv0: TlweParams {
    n: 700,
    alpha: 2.0e-05,
  },
  tlwe_lv1: TlweParams {
    n: 1024,
    alpha: 2.0e-08,
  },
  trlwe_lv1: TrlweParams {
    n: 1024,
    alpha: 2.0e-08,
  },
  trgsw_lv1: TrgswParams {
    n: 1024,
    nbit: 10,
    bgbit: 10,
    bg: 1024,
    l: 2,
    basebit: 2,
    iks_t: 8,
    alpha: 2.0e-08,
  },
};

/// Uint2 parameters (2-bit messages, messageModulus=4)
#[cfg(feature = "lut-bootstrap")]
pub const SECURITY_UINT2: SecurityParams = SecurityParams {
  security_bits: 2,
  description: "Uint2 parameters (2-bit messages, messageModulus=4, N=1024)",
  tlwe_lv0: TlweParams {
    n: 687,
    alpha: 0.000_021_208_468_930_699_72,
  },
  tlwe_lv1: TlweParams {
    n: 1024, // Using 1024 for compatibility with hardcoded TRGSW/TRLWE
    alpha: 0.000_000_000_002_318_412_275_270_499_5,
  },
  trlwe_lv1: TrlweParams {
    n: 1024,
    alpha: 0.000_000_000_002_318_412_275_270_499_5,
  },
  trgsw_lv1: TrgswParams {
    n: 1024,
    nbit: 10,  // 1024 = 2^10
    bgbit: 18, // Base = 1 << 18
    bg: 262144,
    l: 1,
    basebit: 4, // KeySwitch base bits
    iks_t: 3,   // KeySwitch level
    alpha: 0.000_000_000_002_318_412_275_270_499_5,
  },
};

/// Uint3 parameters (3-bit messages, messageModulus=8)
#[cfg(feature = "lut-bootstrap")]
pub const SECURITY_UINT3: SecurityParams = SecurityParams {
  security_bits: 3,
  description: "Uint3 parameters (3-bit messages, messageModulus=8, N=1024)",
  tlwe_lv0: TlweParams {
    n: 820,
    alpha: 0.000_002_516_761_609_597_955_4,
  },
  tlwe_lv1: TlweParams {
    n: 1024,
    alpha: 0.000_000_000_000_000_222_044_604_925_031_3,
  },
  trlwe_lv1: TrlweParams {
    n: 1024,
    alpha: 0.000_000_000_000_000_222_044_604_925_031_3,
  },
  trgsw_lv1: TrgswParams {
    n: 1024,
    nbit: 10,  // 1024 = 2^10
    bgbit: 23, // Base = 1 << 23
    bg: 8388608,
    l: 1,
    basebit: 6, // KeySwitch base bits
    iks_t: 2,   // KeySwitch level
    alpha: 0.000_000_000_000_000_222_044_604_925_031_3,
  },
};

/// Uint4 parameters (4-bit messages, messageModulus=16)
#[cfg(feature = "lut-bootstrap")]
pub const SECURITY_UINT4: SecurityParams = SecurityParams {
  security_bits: 4,
  description: "Uint4 parameters (4-bit messages, messageModulus=16, N=1024)",
  tlwe_lv0: TlweParams {
    n: 820,
    alpha: 0.000_002_516_761_609_597_955_4,
  },
  tlwe_lv1: TlweParams {
    n: 1024,
    alpha: 0.000_000_000_000_000_222_044_604_925_031_3,
  },
  trlwe_lv1: TrlweParams {
    n: 1024,
    alpha: 0.000_000_000_000_000_222_044_604_925_031_3,
  },
  trgsw_lv1: TrgswParams {
    n: 1024,
    nbit: 10,  // 1024 = 2^10
    bgbit: 22, // Base = 1 << 22
    bg: 4194304,
    l: 1,
    basebit: 5, // KeySwitch base bits
    iks_t: 3,   // KeySwitch level
    alpha: 0.000_000_000_000_000_222_044_604_925_031_3,
  },
};

/// Uint5 parameters (5-bit messages, messageModulus=32) - Recommended for complex arithmetic
#[cfg(feature = "lut-bootstrap")]
pub const SECURITY_UINT5: SecurityParams = SecurityParams {
  security_bits: 5,
  description: "Uint5 parameters (5-bit messages, messageModulus=32, N=1024)",
  tlwe_lv0: TlweParams {
    n: 1071,
    alpha: 7.088_226_765_410_43e-8,
  },
  tlwe_lv1: TlweParams {
    n: 1024,
    alpha: 2.2204460492503131e-17,
  },
  trlwe_lv1: TrlweParams {
    n: 1024,
    alpha: 2.2204460492503131e-17,
  },
  trgsw_lv1: TrgswParams {
    n: 1024,
    nbit: 10,
    bgbit: 22,
    bg: 4194304,
    l: 1,
    basebit: 6,
    iks_t: 3,
    alpha: 2.2204460492503131e-17,
  },
};

/// Uint6 parameters (6-bit messages, messageModulus=64)
#[cfg(feature = "lut-bootstrap")]
pub const SECURITY_UINT6: SecurityParams = SecurityParams {
  security_bits: 6,
  description: "Uint6 parameters (6-bit messages, messageModulus=64, N=1024)",
  tlwe_lv0: TlweParams {
    n: 1071,
    alpha: 7.088_226_765_410_43e-8,
  },
  tlwe_lv1: TlweParams {
    n: 1024,
    alpha: 2.2204460492503131e-17,
  },
  trlwe_lv1: TrlweParams {
    n: 1024,
    alpha: 2.2204460492503131e-17,
  },
  trgsw_lv1: TrgswParams {
    n: 1024,
    nbit: 10,
    bgbit: 22,
    bg: 4194304,
    l: 1,
    basebit: 6,
    iks_t: 3,
    alpha: 2.2204460492503131e-17,
  },
};

/// Uint7 parameters (7-bit messages, messageModulus=128)
#[cfg(feature = "lut-bootstrap")]
pub const SECURITY_UINT7: SecurityParams = SecurityParams {
  security_bits: 7,
  description: "Uint7 parameters (7-bit messages, messageModulus=128, N=1024)",
  tlwe_lv0: TlweParams {
    n: 1160,
    alpha: 1.966_220_007_498_402_7e-8,
  },
  tlwe_lv1: TlweParams {
    n: 1024,
    alpha: 2.2204460492503131e-17,
  },
  trlwe_lv1: TrlweParams {
    n: 1024,
    alpha: 2.2204460492503131e-17,
  },
  trgsw_lv1: TrgswParams {
    n: 1024,
    nbit: 10,
    bgbit: 22,
    bg: 4194304,
    l: 1,
    basebit: 7,
    iks_t: 3,
    alpha: 2.2204460492503131e-17,
  },
};

/// Uint8 parameters (8-bit messages, messageModulus=256)
#[cfg(feature = "lut-bootstrap")]
pub const SECURITY_UINT8: SecurityParams = SecurityParams {
  security_bits: 8,
  description: "Uint8 parameters (8-bit messages, messageModulus=256, N=1024)",
  tlwe_lv0: TlweParams {
    n: 1160,
    alpha: 1.966_220_007_498_402_7e-8,
  },
  tlwe_lv1: TlweParams {
    n: 1024,
    alpha: 2.2204460492503131e-17,
  },
  trlwe_lv1: TrlweParams {
    n: 1024,
    alpha: 2.2204460492503131e-17,
  },
  trgsw_lv1: TrgswParams {
    n: 1024,
    nbit: 10,
    bgbit: 22,
    bg: 4194304,
    l: 1,
    basebit: 7,
    iks_t: 3,
    alpha: 2.2204460492503131e-17,
  },
};

/// 128-bit security parameters (default, high security, quantum-resistant)
pub const SECURITY_128_BIT: SecurityParams = SecurityParams {
  security_bits: 128,
  description: "128-bit security (high security, quantum-resistant)",
  tlwe_lv0: TlweParams {
    n: 700,
    alpha: 2.0e-5, // 2^-15.6 approximately
  },
  tlwe_lv1: TlweParams {
    n: 1024,
    alpha: 2.0e-8, // 2^-25.6 approximately
  },
  trlwe_lv1: TrlweParams {
    n: 1024,
    alpha: 2.0e-8,
  },
  trgsw_lv1: TrgswParams {
    n: 1024,
    nbit: 10,
    bgbit: 6,
    bg: 64,
    l: 3,
    basebit: 2,
    iks_t: 9,
    alpha: 2.0e-8,
  },
};

// ============================================================================
// DEFAULT PARAMETER SELECTION
// ============================================================================

/// Default security parameters (128-bit)
pub const DEFAULT_SECURITY: SecurityParams = SECURITY_128_BIT;

/// Get a description of the current security level
pub fn security_info(params: SecurityParams) -> String {
  format!(
    "Security level: {} bits ({})",
    params.security_bits, params.description
  )
}

// ============================================================================
// COMPATIBILITY ALIASES - For backwards compatibility with existing code
// ============================================================================

/// Compatibility module for existing code that expects the old parameter structure
pub mod implementation {
  use super::*;

  // Use 128-bit parameters as default for compatibility
  pub const SECURITY_BITS: usize = SECURITY_128_BIT.security_bits;
  pub const SECURITY_DESCRIPTION: &str = SECURITY_128_BIT.description;

  pub mod tlwe_lv0 {
    use super::super::*;
    pub const N: usize = SECURITY_128_BIT.tlwe_lv0.n;
    pub const ALPHA: f64 = SECURITY_128_BIT.tlwe_lv0.alpha;
  }

  pub mod tlwe_lv1 {
    use super::super::*;
    pub const N: usize = SECURITY_128_BIT.tlwe_lv1.n;
    pub const ALPHA: f64 = SECURITY_128_BIT.tlwe_lv1.alpha;
  }

  pub mod trlwe_lv1 {
    use super::super::*;
    pub const N: usize = SECURITY_128_BIT.trlwe_lv1.n;
    pub const ALPHA: f64 = SECURITY_128_BIT.trlwe_lv1.alpha;
  }

  pub mod trgsw_lv1 {
    use super::super::*;
    pub const N: usize = SECURITY_128_BIT.trgsw_lv1.n;
    pub const NBIT: usize = SECURITY_128_BIT.trgsw_lv1.nbit;
    pub const BGBIT: u32 = SECURITY_128_BIT.trgsw_lv1.bgbit;
    pub const BG: u32 = SECURITY_128_BIT.trgsw_lv1.bg;
    pub const L: usize = SECURITY_128_BIT.trgsw_lv1.l;
    pub const BASEBIT: usize = SECURITY_128_BIT.trgsw_lv1.basebit;
    pub const IKS_T: usize = SECURITY_128_BIT.trgsw_lv1.iks_t;
    pub const ALPHA: f64 = SECURITY_128_BIT.trgsw_lv1.alpha;
  }
}

// Re-export for backwards compatibility
pub use implementation::*;

// Additional compatibility constants
pub const KSK_ALPHA: f64 = SECURITY_128_BIT.tlwe_lv0.alpha;
pub const BSK_ALPHA: f64 = SECURITY_128_BIT.tlwe_lv1.alpha;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_security_info() {
    let info = security_info(SECURITY_128_BIT);
    println!("{}", info);
    assert!(info.contains("128"));
  }

  #[test]
  fn test_parameter_constants() {
    // Test that all constants are accessible
    assert_eq!(SECURITY_80_BIT.security_bits, 80);
    assert_eq!(SECURITY_110_BIT.security_bits, 110);
    assert_eq!(SECURITY_128_BIT.security_bits, 128);
    #[cfg(feature = "lut-bootstrap")]
    {
      assert_eq!(SECURITY_UINT1.security_bits, 1);
      assert_eq!(SECURITY_UINT5.security_bits, 5);
      assert_eq!(SECURITY_UINT8.security_bits, 8);
    }
  }
}
