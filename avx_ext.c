
#include "avx_ext.h"

#undef USE_AVX2

#ifdef USE_AVX2
#include <immintrin.h>
#endif

#include <malloc.h>
#include <math.h>

double predict_single_EXT(const int* inds, double* vals, int lenn, double L1, double baL2, double ialpha, double beta,
double* w, double* z, double* n, double* w_fm, double* z_fm, double* n_fm, double weight_fm, int D_fm, int bias_term, int nThreads) {
    double e = 0.0;
    double e2 = 0.0;
    if (bias_term) e += *w = -*z / ((beta + sqrt(*n)) * ialpha);

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

    int num_thread = 1;

    double* acwfmk = (double*)malloc(sizeof(double) * D_fm * num_thread);

    for (int k = 0; k < D_fm * num_thread; k++) acwfmk[k] = 0.0;

    double* wi2_acc = (double*)malloc(sizeof(double) * num_thread * 4);
    
    for (int k = 0; k < num_thread * 4; k++) wi2_acc[k] = 0.0;
    
    double wi2 = 0.0;

    for (int ii = 0; ii < lenn; ii++) {
       
      const int iThread = 0;

      double* pAcwfmk = acwfmk + iThread * D_fm;
      double* wi2_acck = wi2_acc + iThread * 4;
      const int i = inds[ii];
      double v = vals[ii];
      const int iD_fm = i * D_fm;
      double* z_fmik = z_fm + iD_fm;
      double* w_fmk = pAcwfmk;

      register int k = 0;
      #ifdef USE_AVX2
      __m256d v256 = _mm256_set1_pd(v);
      __m256d w2_256 = _mm256_loadu_pd(wi2_acck);
      while (k + 3 < D_fm) {
        __m256d d = _mm256_mul_pd(_mm256_loadu_pd(z_fmik), v256);
        _mm256_storeu_pd(w_fmk, _mm256_add_pd(_mm256_loadu_pd(w_fmk), d));
        w2_256 = _mm256_add_pd(w2_256, _mm256_mul_pd(d, d));
        k += 4;
        z_fmik += 4;
        w_fmk += 4;
      }
      _mm256_storeu_pd(wi2_acck, w2_256);
      #endif

      // Tail end
      register double d;
      while (k < D_fm) {
        pAcwfmk[k++] += d = *z_fmik++ * v;
        wi2 += d*d;
      }
      
    }

    for (register int k = 0; k < D_fm;) {
        double wfmk = 0.0;
        for (register int iThread = 0; iThread < num_thread;) wfmk += acwfmk[iThread++ * D_fm + k++];
        *w_fm++ = wfmk;
        e2 += wfmk* wfmk;
    }

    for (register int k = 0; k < num_thread * 4; k++) wi2 += wi2_acc[k];

    free(acwfmk);
    free(wi2_acc);
    e2 = (e2 - wi2) * 0.5 * weight_fm;
    return e + e2;
}

void update_single_EXT(const int* inds, double* vals, int lenn, const double e, double ialpha, double* w, double* z, double* n,
double alpha_fm, const double L2_fm, double* w_fm, double* z_fm, double* n_fm, int D_fm, int bias_term, int nThreads) {

    const double e_sq = e * e;

    if (bias_term) {
        *z += e - ((sqrt(*n + e_sq) - sqrt(*n)) * ialpha) * *w;
        *n += e_sq;
    }
    const double L2_fme = L2_fm / e;

    for (int ii = 0; ii < lenn; ii++) {

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

      __m256d reg2 = _mm256_set1_pd(reg);
      __m256d lr2 = _mm256_set1_pd(lr);
      while (k + 3 < D_fm) {
        __m256d z0 = _mm256_loadu_pd(z_fmik);
        _mm256_storeu_pd(z_fmik,
          _mm256_sub_pd(z0, _mm256_mul_pd(lr2,
            _mm256_sub_pd(_mm256_loadu_pd(w_fmk),
              _mm256_mul_pd(z0, reg2)))));
        w_fmk += 4;
        z_fmik += 4;
        k += 4;
      }
      #endif

      while (k++ < D_fm) *z_fmik++ -= lr * (*w_fmk++ - *z_fmik * reg); // Tail end

      n_fm[i] += e_sq;
    }
    
}
