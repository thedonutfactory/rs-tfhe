//! Lookup table implementation for programmable bootstrapping
//!
//! A lookup table is a TRLWE ciphertext that encodes a function for evaluation
//! during programmable bootstrapping operations.

use crate::params;
use crate::trlwe::TRLWELv1;

/// Lookup table for programmable bootstrapping
///
/// A lookup table is a TRLWE ciphertext that encodes a function
/// for programmable bootstrapping. During blind rotation, the LUT is rotated
/// based on the encrypted value, effectively evaluating the function on the
/// encrypted data.
#[derive(Debug, Clone)]
pub struct LookupTable {
    /// Polynomial encoding the function values
    pub poly: TRLWELv1,
}

impl LookupTable {
    /// Create a new lookup table
    pub fn new() -> Self {
        Self {
            poly: TRLWELv1::new(),
        }
    }

    /// Create a lookup table from an existing TRLWE polynomial
    ///
    /// # Arguments
    /// * `poly` - TRLWE polynomial containing the encoded function
    pub fn from_poly(poly: TRLWELv1) -> Self {
        Self { poly }
    }

    /// Get a reference to the underlying polynomial
    pub fn poly(&self) -> &TRLWELv1 {
        &self.poly
    }

    /// Get a mutable reference to the underlying polynomial
    pub fn poly_mut(&mut self) -> &mut TRLWELv1 {
        &mut self.poly
    }

    /// Copy values from another lookup table
    ///
    /// # Arguments
    /// * `other` - Source lookup table to copy from
    pub fn copy_from(&mut self, other: &LookupTable) {
        self.poly.a.copy_from_slice(&other.poly.a);
        self.poly.b.copy_from_slice(&other.poly.b);
    }

    /// Clear the lookup table (sets all coefficients to 0)
    pub fn clear(&mut self) {
        let n = params::trgsw_lv1::N;
        self.poly.a[..n].fill(0);
        self.poly.b[..n].fill(0);
    }

    /// Check if the lookup table is empty (all coefficients are zero)
    pub fn is_empty(&self) -> bool {
        let n = params::trgsw_lv1::N;
        self.poly.a[..n].iter().all(|&x| x == 0) && 
        self.poly.b[..n].iter().all(|&x| x == 0)
    }
}

impl Default for LookupTable {
    fn default() -> Self {
        Self::new()
    }
}

impl From<TRLWELv1> for LookupTable {
    fn from(poly: TRLWELv1) -> Self {
        Self::from_poly(poly)
    }
}

impl From<LookupTable> for TRLWELv1 {
    fn from(lut: LookupTable) -> Self {
        lut.poly
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_table_creation() {
        let lut = LookupTable::new();
        assert!(lut.is_empty());
    }

    #[test]
    fn test_lookup_table_from_poly() {
        let mut poly = TRLWELv1::new();
        poly.b[0] = 1;
        
        let lut = LookupTable::from_poly(poly);
        assert!(!lut.is_empty());
    }

    #[test]
    fn test_lookup_table_copy() {
        let mut lut1 = LookupTable::new();
        let mut lut2 = LookupTable::new();
        
        // Set some values in lut1
        lut1.poly.b[0] = 42;
        lut1.poly.b[1] = 24;
        
        // Copy to lut2
        lut2.copy_from(&lut1);
        
        assert_eq!(lut2.poly.b[0], 42);
        assert_eq!(lut2.poly.b[1], 24);
    }

    #[test]
    fn test_lookup_table_clear() {
        let mut lut = LookupTable::new();
        
        // Set some values
        lut.poly.b[0] = 42;
        lut.poly.b[1] = 24;
        assert!(!lut.is_empty());
        
        // Clear and check
        lut.clear();
        assert!(lut.is_empty());
    }

    #[test]
    fn test_lookup_table_conversions() {
        let mut poly = TRLWELv1::new();
        poly.b[0] = 123;
        
        // From TRLWELv1 to LookupTable
        let lut = LookupTable::from(poly);
        assert_eq!(lut.poly.b[0], 123);
        
        // From LookupTable to TRLWELv1
        let poly_back: TRLWELv1 = lut.into();
        assert_eq!(poly_back.b[0], 123);
    }
}
