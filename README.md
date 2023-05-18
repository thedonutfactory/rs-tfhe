# rs_tfhe
Rust implementation of the TFHE Homomorphic Encrypted Computation Scheme

TFHE is an open-source library for fully homomorphic encryption, distributed under the terms of the Apache 2.0 license.

The underlying scheme is described in best paper of the IACR conference Asiacrypt 2016: “Faster fully homomorphic encryption: Bootstrapping in less than 0.1 seconds”, presented by Ilaria Chillotti, Nicolas Gama, Mariya Georgieva and Malika Izabachène.

rs_tfhe is a rust library which implements a very fast gate-by-gate bootstrapping, based on [CGGI16] and [CGGI17]. The library allows to evaluate an arbitrary boolean circuit composed of binary gates, over encrypted data, without revealing any information on the data.

The library supports the homomorphic evaluation of the 10 binary gates (And, Or, Xor, Nand, Nor, etc…), as well as the negation and the Mux gate. Each binary gate takes about 13 milliseconds single-core time to evaluate, which improves [DM15] by a factor 53, and the mux gate takes about 26 CPU-ms.

Unlike other libraries, the gate-bootstrapping mode of TFHE has no restriction on the number of gates or on their composition. This allows to perform any computation over encrypted data, even if the actual function that will be applied is not yet known when the data is encrypted. The library is easy to use with either manually crafted circuits, or with the output of automated circuit generation tools.

From the user point of view, the library can:

generate a secret-keyset and a cloud-keyset. The secret keyset is private, and provides encryption/decryption abilities. The cloud-keyset can be exported to the cloud, and allows to operate over encrypted data.
With the secret keyset, the library allows to encrypt and decrypt data. The encrypted data can safely be outsourced to the cloud, in order to perform secure homomorphic computations.
With the cloud-keyset, the library can evaluate a net-list of binary gates homomorphically at a rate of about 76 gates per second per core, without decrypting its input. It suffices to provide the sequence of gates, as well as ciphertexts of the input bits. And the library computes ciphertexts of the output bits.

# Run the example

`cargo run --example add_two_numbers --release`
