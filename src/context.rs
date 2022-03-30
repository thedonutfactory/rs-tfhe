use crate::mulfft::FFTPlan;
use crate::params;
use std::cell::RefCell;

thread_local!(pub static FFT_PLAN: RefCell<FFTPlan> = RefCell::new(FFTPlan::new(params::tlwe_lv1::N)));
