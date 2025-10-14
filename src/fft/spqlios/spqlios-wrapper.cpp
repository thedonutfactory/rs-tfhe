#include "spqlios-fft.h"
#include <stdio.h>
#include <array>

extern "C" {
    typedef struct {
       FFT_Processor_Spqlios impl;
    } SpqliosImpl;

    SpqliosImpl* Spqlios_new(const int32_t N) {
        FFT_Processor_Spqlios *spqlios = new FFT_Processor_Spqlios(N);
        return (SpqliosImpl *)spqlios;
    }

    void Spqlios_destructor(SpqliosImpl *si) {
        si->impl.~FFT_Processor_Spqlios();
    }

    void Spqlios_ifft_lv1(SpqliosImpl *si, double *res, const uint32_t *src) {
        si->impl.execute_reverse_torus32(res, src);
    }

    void Spqlios_fft_lv1(SpqliosImpl *si, uint32_t *res, const double *src) {
        si->impl.execute_direct_torus32(res, src);
    }

    void Spqlios_poly_mul_1024(SpqliosImpl *si, uint32_t *res, const uint32_t *src_a, const uint32_t *src_b) {
        std::array<double, 1024> tmp_a;
        std::array<double, 1024> tmp_b;
        Spqlios_ifft_lv1(si, tmp_a.data(), src_a);
        Spqlios_ifft_lv1(si, tmp_b.data(), src_b);

        for (int i=0;i<512;i++){
            double aimbim = tmp_a[i + 512] * tmp_b[i + 512];
            double arebim = tmp_a[i] * tmp_b[i + 512];
            tmp_a[i] = tmp_a[i] * tmp_b[i] - aimbim;
            tmp_a[i + 512] = tmp_a[i + 512] * tmp_b[i] + arebim;
        }

        Spqlios_fft_lv1(si, res, tmp_a.data());
    }
}