fn main() {
  // The FFT is currently provided entirely by the pure-Rust `KlemsaProcessor`
  // (see `src/fft/mod.rs`). The hand-optimized spqlios C++/assembly backend is
  // not wired into the Rust code, so we intentionally do not compile it here.
  //
  // Compiling it unconditionally on x86_64 broke the Windows MSVC build (the
  // sources use GNU `__asm__ __volatile__` and POSIX `M_PI`, which `cl.exe`
  // rejects) and emitted link directives for a library that nothing references.
  //
  // The `fft_fma` / `fft_avx` feature flags are kept for API compatibility but
  // are currently no-ops until the SIMD backend is re-enabled in `fft::mod`.
  if cfg!(any(feature = "fft_fma", feature = "fft_avx")) {
    println!(
      "cargo:warning=SIMD FFT features (fft_fma, fft_avx) are not currently wired in; \
       using the pure-Rust FFT backend on all platforms."
    );
  }
}
