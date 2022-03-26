extern crate cc;

fn main() {
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
  }
  if cfg!(feature = "fft_avx") {
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
}
