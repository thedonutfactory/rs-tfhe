use rs_tfhe::bit_utils::{convert, AsBits};

fn compare_bit(a: Ciphertext, b: Ciphertext, lsb_carry: Ciphertext, cloud_key: &CloudKey) -> Ciphertext {
    let tmp = gates::hom_xnor(a, b, cloud_key);
    gates::hom_mux(tmp, lsb_carry, a)
}

fn equals(a: Ciphertext, b: Ciphertext, cloud_key: &CloudKey) -> Ciphertext {

}