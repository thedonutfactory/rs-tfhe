mod bit_utils;
mod key;
mod mulfft;
mod params;
mod spqlios;
mod tlwe;
mod trgsw;
mod trlwe;
mod utils;
mod gates;

use std::time::Instant;
use crate::utils::{Ciphertext};
use crate::bit_utils::{convert, AsBits};

fn full_adder(
    server_key: &key::CloudKey,
    ct_a: &Ciphertext,
    ct_b: &Ciphertext,
    ct_c: &Ciphertext,
    fft_plan: &mut mulfft::FFTPlan,
) -> (Ciphertext, Ciphertext) {
    // tlwe_nand = trgsw::hom_nand(&ct_a, &ct_b, &server_key, &mut fft_plan);

    let a_xor_b = gates::hom_xor(&ct_a, &ct_b, &server_key, fft_plan);
    let a_and_b = gates::hom_and(&ct_a, &ct_b, &server_key, fft_plan);
    let a_xor_b_and_c = gates::hom_and(&a_xor_b, &ct_c, &server_key, fft_plan);
    // sum = (a xor b) xor c
    let ct_sum = gates::hom_xor(&a_xor_b, &ct_c, &server_key, fft_plan);
    // carry = (a and b) or ((a xor b) and c)
    let ct_carry = gates::hom_or(&a_and_b, &a_xor_b_and_c, &server_key, fft_plan);
    // return sum and carry
    (ct_sum, ct_carry)
}

pub fn add(
    server_key: &key::CloudKey,
    a: &Vec<Ciphertext>,
    b: &Vec<Ciphertext>,
    cin: Ciphertext,
    fft_plan: &mut mulfft::FFTPlan,
) -> (Vec<Ciphertext>, Ciphertext) {
    assert_eq!(
        a.len(),
        b.len(),
        "Cannot add two numbers with different number of bits!"
    );
    let mut result = Vec::with_capacity(a.len());
    let mut carry = cin;
    for i in 0..a.len() {
        let (sum, c) = full_adder(&server_key, &a[i], &b[i], &carry, fft_plan);
        carry = c;
        result.push(sum);
    }
    return (result, carry);
}

pub fn sub(
    server_key: &key::CloudKey,
    a: &Vec<Ciphertext>,
    b: &Vec<Ciphertext>,
    cin: Ciphertext,
    fft_plan: &mut mulfft::FFTPlan,
) -> (Vec<Ciphertext>, Ciphertext) {
    assert_eq!(
        a.len(),
        b.len(),
        "Cannot add two numbers with different number of bits!"
    );
    
    // WARNING: this function does not work as it is off by one

    let not_b = b.iter().map(|x| gates::hom_not(x)).collect::<Vec<Ciphertext>>();
    return add(server_key, a, &not_b, cin, fft_plan);
}

fn encrypt(x: bool, secret_key:&key::SecretKey) -> Ciphertext {
    tlwe::TLWELv0::encrypt_bool(x, params::tlwe_lv0::ALPHA, &secret_key.key_lv0)
}

fn decrypt(x: &Ciphertext, secret_key:&key::SecretKey) -> bool {
    tlwe::TLWELv0::decrypt_bool(x, &secret_key.key_lv0)
}

fn main() {
    let mut fft_plan = mulfft::FFTPlan::new(1024);
    let secret_key = key::SecretKey::new();
    let cloud_key = key::CloudKey::new(&secret_key, &mut fft_plan);
    // inputs
    let a: u16 = 38402;
    let b: u16 = 22304;

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
    let (c3, cin) = add(&cloud_key, &c1, &c2, cin, &mut fft_plan);
    //let (c3, cin) = sub(&cloud_key, &c3, &c2, cin, &mut fft_plan);
    //let (c3, cin) = add(&cloud_key, &c3, &c2, cin, &mut fft_plan);
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
    // Most Significant Bit in position 0, Least Significant Bit in position 3

    let a = convert::<u16>(&a_pt);
    println!("A: {}", a);

    let b = convert::<u16>(&b_pt);
    println!("B: {}", b);
    let s = convert::<u16>(&r1);
    println!("sum: {}", s);

}
