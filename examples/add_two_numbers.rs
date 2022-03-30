use rs_tfhe::bit_utils::{convert, AsBits};
use rs_tfhe::utils::Ciphertext;
use rs_tfhe::key::CloudKey;
use rs_tfhe::key::SecretKey;
use rs_tfhe::gates::*;
use rs_tfhe::tlwe::TLWELv0;
use rs_tfhe::params::*;

use std::time::Instant;

fn full_adder(
  server_key: &CloudKey,
  ct_a: &Ciphertext,
  ct_b: &Ciphertext,
  ct_c: &Ciphertext,
) -> (Ciphertext, Ciphertext) {
  // tlwe_nand = trgsw::hom_nand(&ct_a, &ct_b, &server_key, &mut fft_plan);
  
  let a_xor_b = xor(ct_a, ct_b, server_key);
  let a_and_b = and(ct_a, ct_b, server_key);
  let a_xor_b_and_c = and(&a_xor_b, ct_c, server_key);
  // sum = (a xor b) xor c
  let ct_sum = xor(&a_xor_b, ct_c, server_key);
  // carry = (a and b) or ((a xor b) and c)
  let ct_carry = or(&a_and_b, &a_xor_b_and_c, server_key);
  // return sum and carry
  (ct_sum, ct_carry)
}

pub fn add(
  server_key: &CloudKey,
  a: &Vec<Ciphertext>,
  b: &Vec<Ciphertext>,
  cin: Ciphertext,
) -> (Vec<Ciphertext>, Ciphertext) {
  assert_eq!(
    a.len(),
    b.len(),
    "Cannot add two numbers with different number of bits!"
  );
  let mut result = Vec::with_capacity(a.len());
  let mut carry = cin;
  for i in 0..a.len() {
    let (sum, c) = full_adder(server_key, &a[i], &b[i], &carry);
    carry = c;
    result.push(sum);
  }
  (result, carry)
}

pub fn sub(
  server_key: &CloudKey,
  a: &Vec<Ciphertext>,
  b: &Vec<Ciphertext>,
  cin: Ciphertext,
) -> (Vec<Ciphertext>, Ciphertext) {
  assert_eq!(
    a.len(),
    b.len(),
    "Cannot add two numbers with different number of bits!"
  );

  // WARNING: this function does not work as it is off by one

  let not_b = b.iter().map(not).collect::<Vec<Ciphertext>>();
  add(server_key, a, &not_b, cin)
}

fn encrypt(x: bool, secret_key: &SecretKey) -> Ciphertext {
  TLWELv0::encrypt_bool(x, tlwe_lv0::ALPHA, &secret_key.key_lv0)
}

fn decrypt(x: &Ciphertext, secret_key: &SecretKey) -> bool {
  TLWELv0::decrypt_bool(x, &secret_key.key_lv0)
}

fn main() {
  let secret_key = SecretKey::new();
  let cloud_key = CloudKey::new(&secret_key);
  // inputs
  let a: u16 = 402;
  let b: u16 = 304;

  let a_pt = a.to_bits();
  let b_pt = b.to_bits();

  // Use the client secret key to encrypt plaintext a to ciphertext a
  let c1: Vec<Ciphertext> = a_pt
    .iter()
    .map(|x| encrypt(*x, &secret_key))
    .collect::<Vec<Ciphertext>>();
  let c2: Vec<Ciphertext> = b_pt
    .iter()
    .map(|x| encrypt(*x, &secret_key))
    .collect::<Vec<Ciphertext>>();
  let cin = encrypt(false, &secret_key);

  let start = Instant::now();

  // ----------------- SERVER SIDE -----------------
  // Use the server public key to add the a and b ciphertexts
  let (c3, cin) = add(&cloud_key, &c1, &c2, cin);
  // -------------------------------------------------

  // Use the client secret key to decrypt the ciphertext of the sum
  const BITS: u16 = 16;
  const ADD_GATES_COUNT: u16 = 5;
  const NUM_OPS: u16 = 1;
  let try_num = BITS * ADD_GATES_COUNT * NUM_OPS;
  let end = start.elapsed();
  let exec_ms_per_gate = end.as_millis() as f64 / try_num as f64;
  println!("per gate: {} ms", exec_ms_per_gate);
  println!("total: {} ms", start.elapsed().as_millis());

  let r1 = c3
    .iter()
    .map(|x| decrypt(x, &secret_key))
    .collect::<Vec<bool>>();

  // Use the client secret key to decrypt the ciphertext of the carry
  let carry_pt = decrypt(&cin, &secret_key);
  println!("Carry: {:?}", carry_pt);

  // Convert Boolean tuples to integers and check result
  // Most Significant Bit in position 0, Least Significant Bit in position 
  let a = convert::<u16>(&a_pt);
  println!("A: {}", a);

  let b = convert::<u16>(&b_pt);
  println!("B: {}", b);
  let s = convert::<u16>(&r1);
  println!("sum: {}", s);
}
