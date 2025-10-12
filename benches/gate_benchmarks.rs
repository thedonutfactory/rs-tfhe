use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rs_tfhe::{gates, key, params, utils::Ciphertext};

fn bench_single_gate(c: &mut Criterion) {
  let key = key::SecretKey::new();
  let cloud_key = key::CloudKey::new(&key);

  // Prepare test ciphertexts
  let tlwe_true = Ciphertext::encrypt_bool(true, params::tlwe_lv0::ALPHA, &key.key_lv0);
  let tlwe_false = Ciphertext::encrypt_bool(false, params::tlwe_lv0::ALPHA, &key.key_lv0);

  c.bench_function("gate_nand", |b| {
    b.iter(|| {
      black_box(gates::nand(
        black_box(&tlwe_true),
        black_box(&tlwe_false),
        black_box(&cloud_key),
      ))
    })
  });
}

fn bench_all_gates(c: &mut Criterion) {
  let key = key::SecretKey::new();
  let cloud_key = key::CloudKey::new(&key);

  let tlwe_true = Ciphertext::encrypt_bool(true, params::tlwe_lv0::ALPHA, &key.key_lv0);
  let tlwe_false = Ciphertext::encrypt_bool(false, params::tlwe_lv0::ALPHA, &key.key_lv0);

  let gates: Vec<(
    &str,
    Box<dyn Fn(&Ciphertext, &Ciphertext, &key::CloudKey) -> Ciphertext>,
  )> = vec![
    ("NAND", Box::new(|a, b, k| gates::nand(a, b, k))),
    ("AND", Box::new(|a, b, k| gates::and(a, b, k))),
    ("OR", Box::new(|a, b, k| gates::or(a, b, k))),
    ("XOR", Box::new(|a, b, k| gates::xor(a, b, k))),
    ("NOR", Box::new(|a, b, k| gates::nor(a, b, k))),
    ("XNOR", Box::new(|a, b, k| gates::xnor(a, b, k))),
  ];

  let mut group = c.benchmark_group("homomorphic_gates");

  for (name, gate_fn) in gates.iter() {
    group.bench_with_input(BenchmarkId::from_parameter(name), name, |b, _| {
      b.iter(|| {
        black_box(gate_fn(
          black_box(&tlwe_true),
          black_box(&tlwe_false),
          black_box(&cloud_key),
        ))
      })
    });
  }

  group.finish();
}

fn bench_mux_gate(c: &mut Criterion) {
  let key = key::SecretKey::new();
  let cloud_key = key::CloudKey::new(&key);

  let tlwe_a = Ciphertext::encrypt_bool(true, params::tlwe_lv0::ALPHA, &key.key_lv0);
  let tlwe_b = Ciphertext::encrypt_bool(false, params::tlwe_lv0::ALPHA, &key.key_lv0);
  let tlwe_c = Ciphertext::encrypt_bool(true, params::tlwe_lv0::ALPHA, &key.key_lv0);

  c.bench_function("gate_mux", |b| {
    b.iter(|| {
      black_box(gates::mux_naive(
        black_box(&tlwe_a),
        black_box(&tlwe_b),
        black_box(&tlwe_c),
        black_box(&cloud_key),
      ))
    })
  });
}

fn bench_bootstrapping(c: &mut Criterion) {
  let key = key::SecretKey::new();
  let cloud_key = key::CloudKey::new(&key);

  let tlwe = Ciphertext::encrypt_bool(true, params::tlwe_lv0::ALPHA, &key.key_lv0);

  c.bench_function("bootstrapping", |b| {
    b.iter(|| {
      black_box(rs_tfhe::trgsw::blind_rotate(
        black_box(&tlwe),
        black_box(&cloud_key),
      ))
    })
  });
}

fn bench_fft_operations(c: &mut Criterion) {
  use rand::Rng;
  use rs_tfhe::mulfft::FFTPlan;

  let mut plan = FFTPlan::new(1024);
  let mut rng = rand::thread_rng();

  let mut input = [0u32; 1024];
  input.iter_mut().for_each(|e| *e = rng.gen::<u32>());

  let freq = plan.spqlios.ifft_1024(&input);

  let mut group = c.benchmark_group("fft_operations");

  group.bench_function("fft_forward_1024", |b| {
    b.iter(|| black_box(plan.spqlios.ifft_1024(black_box(&input))))
  });

  group.bench_function("fft_inverse_1024", |b| {
    b.iter(|| black_box(plan.spqlios.fft_1024(black_box(&freq))))
  });

  group.bench_function("poly_mul_1024", |b| {
    b.iter(|| {
      black_box(
        plan
          .spqlios
          .poly_mul_1024(black_box(&input), black_box(&input)),
      )
    })
  });

  group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_single_gate, bench_all_gates, bench_mux_gate, bench_bootstrapping, bench_fft_operations
}

criterion_main!(benches);
