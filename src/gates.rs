use crate::key::CloudKey;
use crate::mulfft::FFTPlan;
use crate::params::Torus;
use crate::params;
use crate::tlwe::{AddMul, SubMul};
use crate::trgsw::{blind_rotate, identity_key_switching};
use crate::trlwe::sample_extract_index;
use crate::utils;
use crate::utils::Ciphertext;

pub struct Gates {
    plan: FFTPlan
}

impl Gates {
    pub fn new() -> Self {
        Gates {
            plan: FFTPlan::new(params::trgsw_lv1::N)
        }
    }

    #[allow(dead_code)]
    pub fn hom_nand(&mut self,
            tlwe_a: &Ciphertext,
            tlwe_b: &Ciphertext,
            cloud_key: &CloudKey,
        ) -> Ciphertext {
        let mut tlwe_nand = -(tlwe_a + tlwe_b);
        *tlwe_nand.b_mut() = tlwe_nand.b().wrapping_add(utils::f64_to_torus(0.125));
        #[cfg(feature = "bootstrapping")]
        {
            self.bootstrap(&tlwe_nand, cloud_key)
        }
        #[cfg(not(feature = "bootstrapping"))]
        {
            return tlwe_nand;
        }
    }

    #[allow(dead_code)]
    pub fn hom_or(&mut self,
            tlwe_a: &Ciphertext,
            tlwe_b: &Ciphertext,
            cloud_key: &CloudKey,
        ) -> Ciphertext {
        let mut tlwe_or = tlwe_a + tlwe_b;
        *tlwe_or.b_mut() = tlwe_or.b().wrapping_add(utils::f64_to_torus(0.125));
        #[cfg(feature = "bootstrapping")]
        {
            self.bootstrap(&tlwe_or, cloud_key)
        }
        #[cfg(not(feature = "bootstrapping"))]
        {
            return tlwe_or;
        }
    }

    #[allow(dead_code)]
    pub fn hom_and(&mut self,
            tlwe_a: &Ciphertext,
            tlwe_b: &Ciphertext,
            cloud_key: &CloudKey,
        ) -> Ciphertext {
        let mut tlwe_and = tlwe_a + tlwe_b;
        *tlwe_and.b_mut() = tlwe_and.b().wrapping_add(utils::f64_to_torus(-0.125));
        #[cfg(feature = "bootstrapping")]
        {
            self.bootstrap(&tlwe_and, cloud_key)
        }
        #[cfg(not(feature = "bootstrapping"))]
        {
            return tlwe_and;
        }
    }

    #[allow(dead_code)]
    pub fn hom_xor(&mut self,
            tlwe_a: &Ciphertext,
            tlwe_b: &Ciphertext,
            cloud_key: &CloudKey,
        ) -> Ciphertext {
        let mut tlwe_xor = tlwe_a.add_mul(tlwe_b, 2);
        *tlwe_xor.b_mut() = tlwe_xor.b().wrapping_add(utils::f64_to_torus(0.25));
        #[cfg(feature = "bootstrapping")]
        {
            self.bootstrap(&tlwe_xor, cloud_key)
        }
        #[cfg(not(feature = "bootstrapping"))]
        {
            return tlwe_xor;
        }
    }

    #[allow(dead_code)]
    pub fn hom_xnor(&mut self,
            tlwe_a: &Ciphertext,
            tlwe_b: &Ciphertext,
            cloud_key: &CloudKey,
        ) -> Ciphertext {
        let mut tlwe_xnor = tlwe_a.sub_mul(tlwe_b, 2);
        *tlwe_xnor.b_mut() = tlwe_xnor.b().wrapping_add(utils::f64_to_torus(-0.25));
        #[cfg(feature = "bootstrapping")]
        {
            self.bootstrap(&tlwe_xnor, cloud_key)
        }
        #[cfg(not(feature = "bootstrapping"))]
        {
            return tlwe_xnor;
        }
    }

    #[allow(dead_code)]
    pub fn hom_nor(&mut self,
            tlwe_a: &Ciphertext,
            tlwe_b: &Ciphertext,
            cloud_key: &CloudKey,
        ) -> Ciphertext {
        let mut tlwe_nor = -(tlwe_a + tlwe_b);
        *tlwe_nor.b_mut() = tlwe_nor.b().wrapping_add(utils::f64_to_torus(-0.125));
        #[cfg(feature = "bootstrapping")]
        {
            self.bootstrap(&tlwe_nor, cloud_key)
        }
        #[cfg(not(feature = "bootstrapping"))]
        {
            return tlwe_nor;
        }
    }

    #[allow(dead_code)]
    pub fn hom_and_ny(&mut self,
            tlwe_a: &Ciphertext,
            tlwe_b: &Ciphertext,
            cloud_key: &CloudKey,
        ) -> Ciphertext {
        let mut tlwe_and_ny = &-(*tlwe_a) + tlwe_b;
        *tlwe_and_ny.b_mut() = tlwe_and_ny.b().wrapping_add(utils::f64_to_torus(-0.125));
        #[cfg(feature = "bootstrapping")]
        {
            self.bootstrap(&tlwe_and_ny, cloud_key)
        }
        #[cfg(not(feature = "bootstrapping"))]
        {
            return tlwe_and_ny;
        }
    }

    #[allow(dead_code)]
    pub fn hom_and_yn(&mut self,
            tlwe_a: &Ciphertext,
            tlwe_b: &Ciphertext,
            cloud_key: &CloudKey,
        ) -> Ciphertext {
        let mut tlwe_and_yn = tlwe_a - tlwe_b;
        *tlwe_and_yn.b_mut() = tlwe_and_yn.b().wrapping_add(utils::f64_to_torus(-0.125));
        #[cfg(feature = "bootstrapping")]
        {
            self.bootstrap(&tlwe_and_yn, cloud_key)
        }
        #[cfg(not(feature = "bootstrapping"))]
        {
            return tlwe_and_yn;
        }
    }

    #[allow(dead_code)]
    pub fn hom_or_ny(&mut self,
            tlwe_a: &Ciphertext,
            tlwe_b: &Ciphertext,
            cloud_key: &CloudKey,
        ) -> Ciphertext {
        let mut tlwe_or_ny = &-*tlwe_a + tlwe_b;
        *tlwe_or_ny.b_mut() = tlwe_or_ny.b().wrapping_add(utils::f64_to_torus(0.125));
        #[cfg(feature = "bootstrapping")]
        {
            self.bootstrap(&tlwe_or_ny, cloud_key)
        }
        #[cfg(not(feature = "bootstrapping"))]
        {
            return tlwe_or_ny;
        }
    }

    #[allow(dead_code)]
    /// Homomorphic bootstrapped Mux(a,b,c) = a?b:c = a*b + not(a)*c
    pub fn hom_mux(&mut self,
            tlwe_a: &Ciphertext,
            tlwe_b: &Ciphertext,
            tlwe_c: &Ciphertext,
        ) -> Ciphertext {

        let cloud_key_no_ksk = CloudKey::new_no_ksk();

        // and(a, b)
        let mut tlwe_and = tlwe_a + tlwe_b;
        *tlwe_and.b_mut() = tlwe_and.b().wrapping_add(utils::f64_to_torus(-0.125));
        let u1:&Ciphertext = &self.bootstrap_no_ksk(&tlwe_and, &cloud_key_no_ksk);

        // and(not(a), c) -> nand(a, c)  
        let mut tlwe_nand = -(tlwe_a + tlwe_c);
        *tlwe_nand.b_mut() = tlwe_nand.b().wrapping_add(utils::f64_to_torus(0.125));
        let u2:&Ciphertext = &self.bootstrap_no_ksk(&tlwe_nand, &cloud_key_no_ksk);
        
        // or(u1, u2)
        let mut tlwe_or = u1 + u2;
        *tlwe_or.b_mut() = tlwe_or.b().wrapping_add(utils::f64_to_torus(0.125));

        return tlwe_or; // self.bootstrap_no_ksk(&tlwe_or, &cloud_key_no_ksk);
        
        /*
        #[cfg(feature = "bootstrapping")]
        {
            self.bootstrap(&tlwe_and_yn, cloud_key)
        }
        #[cfg(not(feature = "bootstrapping"))]
        {
            return tlwe_and_yn;
        }
        */
    }

    #[allow(dead_code)]
    pub fn hom_not(&self, tlwe_a: &Ciphertext) -> Ciphertext {
        -(*tlwe_a)
    }

    #[allow(dead_code)]
    pub fn hom_copy(&self, tlwe_a: &Ciphertext) -> Ciphertext {
        *tlwe_a
    }

    #[allow(dead_code)]
    pub fn hom_constant(&self, value: bool) -> Ciphertext {
        let mut mu: Torus = utils::f64_to_torus(0.125);
        mu = if value { mu } else { 1 - mu };
        let mut res = Ciphertext::new();
        *res.b_mut() = mu;
        res
    }

    fn bootstrap(&mut self, ctxt: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
        let trlwe = blind_rotate(ctxt, cloud_key, &mut self.plan);
        let tlwe_lv1 = sample_extract_index(&trlwe, 0);

        identity_key_switching(&tlwe_lv1, &cloud_key.key_switching_key)
    }

    fn bootstrap_no_ksk(&mut self, ctxt: &Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
      let trlwe = blind_rotate(ctxt, &cloud_key, &mut self.plan);
      let tlwe_lv1 = sample_extract_index(&trlwe, 0);

      identity_key_switching(&tlwe_lv1, &cloud_key.key_switching_key)
  }
}

#[cfg(test)]
mod tests {
  use crate::gates;
  use crate::key;
  use crate::key::CloudKey;
  use crate::params;
  use crate::utils::Ciphertext;
  use rand::Rng;

  #[test]
  fn test_hom_nand() {
    let mut gates = gates::Gates::new();
    test_gate(
      |a, b| !(a & b),
      |a: &Ciphertext, b: &Ciphertext, k: &CloudKey| gates.hom_nand(a, b, k),
    );
  }

  #[test]
  fn test_hom_or() {
    let mut gates = gates::Gates::new();
    test_gate(
      |a, b| a | b,
      |a: &Ciphertext, b: &Ciphertext, k: &CloudKey| gates.hom_or(a, b, k),
    );
  }

  #[test]
  fn test_hom_xnor() {
    let mut gates = gates::Gates::new();
    test_gate(
      |a, b| false ^ (b ^ a),
      |a: &Ciphertext, b: &Ciphertext, k: &CloudKey| gates.hom_xnor(a, b, k),
    );
  }

  #[test]
  fn test_hom_xor() {
    let mut gates = gates::Gates::new();
    test_gate(
      |a, b| a ^ b,
      |a: &Ciphertext, b: &Ciphertext, k: &CloudKey| gates.hom_xor(a, b, k),
    );
  }

  #[test]
  fn test_hom_not() {
    let gates = gates::Gates::new();
    test_gate(|a, _| !a, |a: &Ciphertext, _, _| gates.hom_not(a));
  }

  #[test]
  fn test_hom_copy() {
    let gates = gates::Gates::new();
    test_gate(|a, _| a, |a: &Ciphertext, _, _| gates.hom_copy(a));
  }

  #[test]
  fn test_hom_constant() {
    let gates = gates::Gates::new();
    let test = true;
    test_gate(|_, _| test, |_: _, _, _| gates.hom_constant(test));
  }

  #[test]
  fn test_hom_nor() {
    let mut gates = gates::Gates::new();
    test_gate(
      |a, b| !(a | b),
      |a: &Ciphertext, b: &Ciphertext, k: &CloudKey| gates.hom_nor(a, b, k),
    );
  }

  #[test]
  fn test_hom_and_ny() {
    let mut gates = gates::Gates::new();
    test_gate(
      |a, b| !a & b,
      |a: &Ciphertext, b: &Ciphertext, k: &CloudKey| gates.hom_and_ny(a, b, k),
    );
  }

  #[test]
  fn test_hom_and_yn() {
    let mut gates = gates::Gates::new();
    test_gate(
      |a, b| a & !b,
      |a: &Ciphertext, b: &Ciphertext, k: &CloudKey| gates.hom_and_yn(a, b, k),
    );
  }
  #[test]
  fn test_hom_or_ny() {
    let mut gates = gates::Gates::new();
    test_gate(
      |a, b| !a | b,
      |a: &Ciphertext, b: &Ciphertext, k: &CloudKey| gates.hom_or_ny(a, b, k),
    );
  }

  #[test]
  fn test_mux() {
    let mut gates = gates::Gates::new();
    let mut rng = rand::thread_rng();
    let key = key::SecretKey::new();

    let try_num = 10;
    for _i in 0..try_num {
      let plain_a = rng.gen::<bool>();
      let plain_b = rng.gen::<bool>();
      let plain_c = rng.gen::<bool>();
      let expected = (plain_a & plain_b) | !(plain_a & plain_c);

      let tlwe_a = Ciphertext::encrypt_bool(plain_a, params::tlwe_lv0::ALPHA, &key.key_lv0);
      let tlwe_b = Ciphertext::encrypt_bool(plain_b, params::tlwe_lv0::ALPHA, &key.key_lv0);
      let tlwe_c = Ciphertext::encrypt_bool(plain_b, params::tlwe_lv0::ALPHA, &key.key_lv0);
      let tlwe_op = gates.hom_mux(&tlwe_a, &tlwe_b, &tlwe_c);
      let dec = tlwe_op.decrypt_bool(&key.key_lv0);
      dbg!(plain_a);
      dbg!(plain_b);
      dbg!(expected);
      dbg!(dec);
      assert_eq!(expected, dec);
    }
  }

  fn test_gate<
    E: Fn(bool, bool) -> bool,
    C: FnMut(&Ciphertext, &Ciphertext, &CloudKey) -> Ciphertext,
  >(
    expect: E,
    mut actual: C,
  ) {
    let mut rng = rand::thread_rng();
    let key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&key);

    let try_num = 10;
    for _i in 0..try_num {
      let plain_a = rng.gen::<bool>();
      let plain_b = rng.gen::<bool>();
      let expected = expect(plain_a, plain_b);

      let tlwe_a = Ciphertext::encrypt_bool(plain_a, params::tlwe_lv0::ALPHA, &key.key_lv0);
      let tlwe_b = Ciphertext::encrypt_bool(plain_b, params::tlwe_lv0::ALPHA, &key.key_lv0);
      let tlwe_op = actual(&tlwe_a, &tlwe_b, &cloud_key);
      let dec = tlwe_op.decrypt_bool(&key.key_lv0);
      dbg!(plain_a);
      dbg!(plain_b);
      dbg!(expected);
      dbg!(dec);
      assert_eq!(expected, dec);
    }
  }

  /*
  #[test]
  fn test_hom_nand_bench() {
      const N: usize = params::trgsw_lv1::N;
      let mut rng = rand::thread_rng();
      let mut plan = mulfft::FFTPlan::new(N);
      let key = key::SecretKey::new();
      let cloud_key = key::CloudKey::new(&key, &mut plan);

      let mut b_key: Vec<TRGSWLv1> = Vec::new();
      for i in 0..key.key_lv0.len() {
          b_key.push(TRGSWLv1::encrypt_torus(
              key.key_lv0[i],
              params::trgsw_lv1::ALPHA,
              &key.key_lv1,
              &mut plan,
          ));
      }

      let try_num = 100;
      let plain_a = rng.gen::<bool>();
      let plain_b = rng.gen::<bool>();
      let nand = !(plain_a & plain_b);

      let tlwe_a = Ciphertext::encrypt_bool(plain_a, params::tlwe_lv0::ALPHA, &key.key_lv0);
      let tlwe_b = Ciphertext::encrypt_bool(plain_b, params::tlwe_lv0::ALPHA, &key.key_lv0);
      let mut tlwe_nand = Ciphertext::new();
      println!("Started bechmark");
      let start = Instant::now();
      for _i in 0..try_num {
          tlwe_nand = gates::hom_nand(&tlwe_a, &tlwe_b, &cloud_key, &mut plan);
      }
      let end = start.elapsed();
      let exec_ms_per_gate = end.as_millis() as f64 / try_num as f64;
      println!("exec ms per gate : {} ms", exec_ms_per_gate);
      let dec = tlwe_nand.decrypt_bool(&key.key_lv0);
      dbg!(plain_a);
      dbg!(plain_b);
      dbg!(nand);
      dbg!(dec);
      assert_eq!(nand, dec);
  }
  */
}
