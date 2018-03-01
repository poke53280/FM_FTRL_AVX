
#include "SingleUpdate.h"

#define USE_AVX2
#define USE_OMP

#ifdef USE_OMP
#include <omp.h>
#endif


#ifdef USE_AVX2
#include <immintrin.h>
#endif


void update_single_EXT(const int* inds, double* vals, int lenn, const double e, double ialpha, double* w, double* z, double* n,
  double alpha_fm, const double L2_fm, double* w_fm, double* z_fm, double* n_fm, int D_fm, int bias_term, int nThreads) {

  int num_thread = 1;

#ifdef USE_OMP

  if (nThreads <= 0) {
    num_thread = omp_get_max_threads();
  }
  else {
    num_thread = nThreads;
  }

#endif

  const double e_sq = e * e;

  if (bias_term) {
    const double ni = n[0];
    z[0] += e - ((sqrt(ni + e_sq) - sqrt(ni)) * ialpha) * w[0];
    n[0] += e_sq;
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

    double * z_fmi = z_fm + i * D_fm;
    const double lr = g* alpha_fm / (sqrt(n_fm[i]) + 1.0);
    const double reg = v - L2_fme;

    int k = 0;

#ifdef USE_AVX2

    __m256d reg2 = _mm256_set_pd(reg, reg, reg, reg);
    __m256d lr2 = _mm256_set_pd(lr, lr, lr, lr);

    while (k + 3 < D_fm) {

      double* z_offset = z_fmi + k;

      const double* w_offset = w_fm + k;

      __m256d z0 = _mm256_loadu_pd(z_offset);

      __m256d w = _mm256_loadu_pd(w_offset);

      __m256d z = _mm256_mul_pd(z0, reg2);

      __m256d w2 = _mm256_sub_pd(w, z);

      __m256d w3 = _mm256_mul_pd(lr2, w2);

      __m256d res = _mm256_sub_pd(z0, w3);

      _mm256_store_pd(z_offset, res);

      k = k + 4;
    }

#endif

    // Tail end
    for (; k < D_fm; k++) {

      const double z0 = z_fmi[k];
      const double w = w_fm[k];

      const double z = z0 * reg;
      const double w2 = w - z;
      const double w3 = lr * w2;

      const double res = z0 - w3;

      z_fmi[k] = res;
    }

    n_fm[i] += e_sq;
  }
}