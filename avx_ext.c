#include "avx_ext.h"

#define USE_AVX2
#define USE_OMP

#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef USE_AVX2

#include <immintrin.h>

#endif

#include <malloc.h>
#include <math.h>

double predict_single_EXT(const int* inds, double* vals, int lenn, double L1, double baL2, double ialpha, double beta,
double* w, double* z, double* n, double* w_fm, double* z_fm, double* n_fm, double weight_fm, int D_fm, int bias_term, int nThreads) {
    double e = 0.0;
    double e2 = 0.0;
    if (bias_term) {
        const double wi = w[0] = -z[0] / ((beta + sqrt(n[0])) * ialpha);
        e += wi;
    }

    /*
    #ifdef USE_OMP
    #pragma omp parallel for
    #endif
    */
    for (int ii = 0; ii < lenn; ii++) {
        const int i = inds[ii];
        const double zi = z[i];
        const double sign = (zi < 0) ? -1.0 : 1.0;
        if (sign * zi > L1) {
            const double wi = (sign * L1 - zi) / (sqrt(n[i]) * ialpha + baL2);
            w[ii + 1] = wi;
            e += wi * vals[ii];
        } else w[ii + 1] = 0.0;
    }

    double wi2 = 0.0;

    int num_thread = 1;
    #ifdef USE_OMP
    if (nThreads <= 0) num_thread = omp_get_max_threads();
    else num_thread = nThreads;
    #endif

    double* acwfmk = (double*)malloc(sizeof(double) * D_fm * num_thread);
    int k;
    #ifdef USE_OMP
    #pragma omp parallel for num_threads(num_thread)
    #endif
    for (k = 0; k < D_fm * num_thread; k++) acwfmk[k] = 0.0;

    double* wi2_acc = (double*)malloc(sizeof(double) * num_thread * 4);

    #ifdef USE_OMP
    #pragma omp parallel for num_threads(num_thread)
    #endif
    for (k = 0; k < num_thread * 4; k++) wi2_acc[k] = 0.0;

    int ii;
    #ifdef USE_OMP
    #pragma omp parallel for num_threads(num_thread)
    #endif
    for (ii = 0; ii < lenn; ii++) {

        #ifdef USE_OMP
        const int iThread = omp_get_thread_num();
        #else
        const int iThread = 0;
        #endif

        double* pAcwfmk = acwfmk + iThread * D_fm;
        double* wi2_acck = wi2_acc + iThread * 4;
        const int i = inds[ii];
        double v = vals[ii];
        const int iD_fm = i * D_fm;
        int k = 0;

        double* z_fmik = z_fm + iD_fm;
        double* w_fmk = pAcwfmk;

        #ifdef USE_AVX2
        __m256d v256 = _mm256_set_pd(v, v, v, v);
        __m256d w2_256 = _mm256_loadu_pd(wi2_acck);
        while (k + 4 < D_fm) {
            __m256d d = _mm256_mul_pd(_mm256_loadu_pd(z_fmik), v256);
            __m256d w = _mm256_add_pd(_mm256_loadu_pd(w_fmk), d);
            _mm256_storeu_pd(w_fmk, w);
            d = _mm256_mul_pd(d, d);

            w2_256 = _mm256_add_pd(w2_256, d);

            k += 4;
            z_fmik+= 4;
            w_fmk+= 4;
        }
        _mm256_storeu_pd(wi2_acck, w2_256);
        #endif

        // Tail end
        double d;
        while(k < D_fm) {
            d = *z_fmik++ * v;
            pAcwfmk[k] = pAcwfmk[k++] + d;
            wi2 += (d * d);
        }
    }

    for (int k = 0; k < D_fm; k++) {
        double wfmk = 0.0;
        for (int iThread = 0; iThread < num_thread; iThread++) wfmk += acwfmk[iThread * D_fm + k];
        w_fm[k] = wfmk;
        e2 += (wfmk* wfmk);
    }

    for (int k = 0; k < num_thread * 4; k++) wi2 += wi2_acc[k];

    free(acwfmk);
    acwfmk = 0;
    free(wi2_acc);
    wi2_acc = 0;
    e2 = (e2 - wi2) * 0.5 * weight_fm;
    return e + e2;
}

void update_single_EXT(const int* inds, double* vals, int lenn, const double e, double ialpha, double* w, double* z, double* n,
double alpha_fm, const double L2_fm, double* w_fm, double* z_fm, double* n_fm, int D_fm, int bias_term, int nThreads) {

    #ifdef USE_OMP
    int num_thread;
    if (nThreads <= 0) num_thread = omp_get_max_threads();
    else num_thread = nThreads;
    #endif

    const double e_sq = e * e;

    if (bias_term) {
        const double ni = n[0];
        *z += e - ((sqrt(ni + e_sq) - sqrt(ni)) * ialpha) * w[0];
        *n += e_sq;
    }
    const double L2_fme = L2_fm / e;

    int ii;
    #ifdef USE_OMP
    #pragma omp parallel for num_threads(num_thread)
    for (ii = 0; ii < lenn; ii++) {
    #else
    for (ii = 0; ii < lenn; ii++) {
    #endif
        const int i = inds[ii];
        const double v = vals[ii];
        const double g = e * v;
        const double g2 = g * g;
        const double ni = n[i];

        z[i] += g - ((sqrt(ni + g2) - sqrt(ni)) * ialpha) * w[ii + 1];
        n[i] += g2;

        double* z_fmik = z_fm + i * D_fm;
        double* w_fmk = w_fm;
        const double lr = g* alpha_fm / (sqrt(n_fm[i]) + 1.0);
        const double reg = v - L2_fme;

        register int k = 0;
        #ifdef USE_AVX2
        __m256d reg2 = _mm256_set_pd(reg, reg, reg, reg);
        __m256d lr2 = _mm256_set_pd(lr, lr, lr, lr);

        while (k + 4 < D_fm) {
            __m256d z0 = _mm256_loadu_pd(z_fmik);
            _mm256_store_pd(z_fmik, _mm256_sub_pd(z0,
                                    _mm256_mul_pd(lr2,
                                    _mm256_sub_pd(_mm256_loadu_pd(w_fmk),
                                                  _mm256_mul_pd(z0, reg2)))));
            w_fmk+= 4;
            z_fmik+= 4;
            k+= 4;
        }
        #endif

        // Tail end
        double z0;
        while (k++ < D_fm) {
            z0 = *z_fmik;
            *z_fmik++ = z0 - lr * (*w_fmk++ - z0 * reg);
        }
        n_fm[i] += e_sq;
    }
}
