extern crate cc;

fn main() {
  // Only compile SIMD code on x86_64 targets
  let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();

  if target_arch == "x86_64" {
    println!("cargo:rustc-link-lib=spqlios");
    if cfg!(feature = "fft_fma") {
      cc::Build::new()
        .cpp(true)
        .file("src/spqlios/fft_processor_spqlios.cpp")
        .file("src/spqlios/spqlios-fft-impl.cpp")
        .file("src/spqlios/spqlios-wrapper.cpp")
        .file("src/spqlios/spqlios-fft-fma.s")
        .file("src/spqlios/spqlios-ifft-fma.s")
        .flag("-std=c++17")
        .flag("-lm")
        .flag("-Ofast")
        .flag("-march=native")
        .flag("-DNDEBUG")
        .include("src")
        .compile("libspqlios.a");
    } else if cfg!(feature = "fft_avx") {
      cc::Build::new()
        .cpp(true)
        .file("src/spqlios/fft_processor_spqlios.cpp")
        .file("src/spqlios/spqlios-fft-impl.cpp")
        .file("src/spqlios/spqlios-wrapper.cpp")
        .file("src/spqlios/spqlios-fft-avx.s")
        .file("src/spqlios/spqlios-ifft-avx.s")
        .flag("-std=c++17")
        .flag("-lm")
        .flag("-Ofast")
        .flag("-march=native")
        .flag("-DNDEBUG")
        .include("src")
        .compile("libspqlios.a");
    }
  } else {
    println!("cargo:warning=SIMD FFT features (fft_fma, fft_avx) are only supported on x86_64. Skipping SIMD compilation on {}.", target_arch);
  }
}
