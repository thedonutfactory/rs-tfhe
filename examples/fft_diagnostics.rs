/// Diagnostic tool to check RustFFT configuration and options

use rustfft::{FftPlanner, Fft};
use rustfft::num_complex::Complex;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║            RustFFT Configuration Diagnostics                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut planner = FftPlanner::new();
    
    // Our key sizes
    let sizes = vec![1024, 2048];
    
    for &n in &sizes {
        println!("FFT Size: {}", n);
        
        let fft = planner.plan_fft_forward(n);
        
        // Try to get algorithm info (if available)
        println!("  Algorithm type: {:?}", std::any::type_name_of_val(&*fft));
        println!("  Length: {}", fft.len());
        println!();
        
        // Benchmark a single FFT
        let mut data = vec![Complex::new(1.0, 0.0); n];
        
        let iterations = 1000;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            fft.process(&mut data);
        }
        let elapsed = start.elapsed();
        
        println!("  Performance ({} iterations):", iterations);
        println!("    Total: {:?}", elapsed);
        println!("    Per FFT: {:.2}µs", elapsed.as_micros() as f64 / iterations as f64);
        println!();
    }
    
    println!("💡 Notes:");
    println!("  • RustFFT automatically selects the best algorithm");
    println!("  • For powers of 2, it typically uses radix-2 or radix-4");
    println!("  • Algorithm selection is based on size and CPU features");
    println!("  • Our cached plans use these optimized algorithms");
}
