use std::os::raw::c_double;
use std::os::raw::c_int;
use std::os::raw::c_uint;

pub enum SpqliosImpl {}

extern "C" {
  pub fn Spqlios_new(N: c_int) -> *mut SpqliosImpl;
  pub fn Spqlios_destructor(spqlios: *mut SpqliosImpl);
  pub fn Spqlios_ifft_lv1(spqlios: *mut SpqliosImpl, res: *mut c_double, src: *const c_uint);
  pub fn Spqlios_fft_lv1(spqlios: *mut SpqliosImpl, res: *mut c_uint, src: *const c_double);
  pub fn Spqlios_poly_mul_1024(
    spqlios: *mut SpqliosImpl,
    res: *mut c_uint,
    src_a: *const c_uint,
    src_b: *const c_uint,
  );
}

pub struct Spqlios {
  raw: *mut SpqliosImpl,
  n: usize,
}

impl Spqlios {
  pub fn new(n: usize) -> Self {
    unsafe {
      Spqlios {
        raw: Spqlios_new(n as i32),
        n,
      }
    }
  }

  pub fn ifft_1024(&mut self, input: &[u32; 1024]) -> [f64; 1024] {
    let src_const_ptr = input.as_ptr() as *const _;
    let mut res = Box::new([0.0f64; 1024]);
    let res_mut_ptr = Box::into_raw(res) as *mut _;
    unsafe {
      Spqlios_ifft_lv1(self.raw, res_mut_ptr, src_const_ptr);
    }
    res = unsafe { Box::from_raw(res_mut_ptr as *mut [f64; 1024]) };
    *res
  }

  pub fn fft_1024(&mut self, input: &[f64; 1024]) -> [u32; 1024] {
    let src_const_ptr = input.as_ptr() as *const _;
    let mut res = Box::new([0u32; 1024]);
    let res_mut_ptr = Box::into_raw(res) as *mut _;
    unsafe {
      Spqlios_fft_lv1(self.raw, res_mut_ptr, src_const_ptr);
    }
    res = unsafe { Box::from_raw(res_mut_ptr as *mut [u32; 1024]) };
    *res
  }

  pub fn poly_mul_1024(&mut self, a: &[u32; 1024], b: &[u32; 1024]) -> [u32; 1024] {
    let mut res = Box::new([0u32; 1024]);
    let res_mut_ptr = Box::into_raw(res) as *mut _;
    unsafe {
      Spqlios_poly_mul_1024(
        self.raw,
        res_mut_ptr,
        a.as_ptr() as *const _,
        b.as_ptr() as *const _,
      );
    }

    res = unsafe { Box::from_raw(res_mut_ptr as *mut [u32; 1024]) };
    *res
  }

  #[allow(dead_code)]
  pub fn ifft(&mut self, input: &Vec<u32>) -> Vec<f64> {
    let mut res: Vec<f64> = vec![0.0f64; self.n];
    unsafe {
      Spqlios_ifft_lv1(self.raw, res.as_mut_ptr(), input.as_ptr());
    }
    res
  }

  #[allow(dead_code)]
  pub fn fft(&mut self, input: &Vec<f64>) -> Vec<u32> {
    let mut res: Vec<u32> = vec![0u32; self.n];
    unsafe {
      Spqlios_fft_lv1(self.raw, res.as_mut_ptr(), input.as_ptr());
    }
    res
  }

  #[allow(dead_code)]
  pub fn poly_mul(&mut self, a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    let a_ifft = self.ifft(a);
    let b_ifft = self.ifft(b);
    let mut mul = vec![0.0f64; self.n];

    let ns = self.n / 2;
    for i in 0..ns {
      let aimbim = a_ifft[i + ns] * b_ifft[i + ns];
      let arebim = a_ifft[i] * b_ifft[i + ns];
      mul[i] = a_ifft[i] * b_ifft[i] - aimbim;
      mul[i + ns] = a_ifft[i + ns] * b_ifft[i] + arebim;
    }

    self.fft(&mul)
  }
}

impl Drop for Spqlios {
  fn drop(&mut self) {
    unsafe {
      Spqlios_destructor(self.raw);
    }
  }
}
