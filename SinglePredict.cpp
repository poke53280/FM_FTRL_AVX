

#include "SinglePredict.h"

#include <omp.h>
#include <stdio.h>
#include <cmath>
#include <immintrin.h>
#include <assert.h>

int halfer_EXT_INT(int d) {
  return d / 2;
}


double predict_single_EXT(const int* inds, double* vals, int lenn, double L1, double baL2, double ialpha, double beta,
  double* w, double* z, double* n, double* w_fm, double* z_fm, double* n_fm, double weight_fm, int D_fm, bool bias_term, int nThreads) {


  printf("Running predict_single_EXT(), at least for a while...\r\n");

  double e = 0.0;
  double e2 = 0.0;

  if (bias_term) {
    const double wi = w[0] = -z[0] / ((beta + sqrt(n[0])) * ialpha);
    e += wi;
  }

  // #pragma omp parallel for
  for (int ii = 0; ii < lenn; ii++) {

    const int i = inds[ii];
    const double zi = z[i];

    const double sign = (zi < 0) ? -1.0 : 1.0;

    if (sign * zi > L1) {
      const double wi = (sign * L1 - zi) / (sqrt(n[i]) * ialpha + baL2);

      w[ii + 1] = wi;
      e += wi * vals[ii];
    }
    else {
      w[ii + 1] = 0.0;
    }
  }

  double wi2 = 0.0;

  int num_thread;

  if (nThreads <= 0) {
    num_thread = omp_get_max_threads();
  }
  else {
    num_thread = nThreads;
  }


  printf("Running on %d threads\r\n", num_thread);

  double* acwfmk = new double[D_fm * num_thread];

#pragma omp parallel for num_threads(num_thread)
  for (int k = 0; k < D_fm * num_thread; k++) {
    acwfmk[k] = 0.0;
  }

  double* wi2_acc = new double[num_thread * 4];

#pragma omp parallel for num_threads(num_thread)
  for (int k = 0; k < num_thread * 4; k++) {
    wi2_acc[k] = 0.0;
  }


#pragma omp parallel for num_threads(num_thread)
  for (int ii = 0; ii < lenn; ii++) {

    const int iThread = omp_get_thread_num();

    assert(iThread >= 0 && iThread < num_thread);

    double*
      pAcwfmk = acwfmk + iThread * D_fm;

    double * wi2_acck = wi2_acc + iThread * 4;

    const int idx = inds[ii];
    double v = vals[ii];

    const int z_idx0 = idx * D_fm;

    int k = 0;

    __m256d v256 = _mm256_set_pd(v, v, v, v);

    while (k + 3 < D_fm) {

      const int z_idx = z_idx0 + k;

      const double * z_offset = z_fm + z_idx;

      __m256d z = _mm256_loadu_pd(z_offset);

      __m256d d = _mm256_mul_pd(z, v256);

      double * w_offset = pAcwfmk + k;

      __m256d w = _mm256_loadu_pd(w_offset);

      w = _mm256_add_pd(w, d);

      _mm256_storeu_pd(w_offset, w);

      d = _mm256_mul_pd(d, d);


      __m256d w2_256 = _mm256_loadu_pd(wi2_acck);

      w2_256 = _mm256_add_pd(w2_256, d);

      _mm256_storeu_pd(wi2_acck, w2_256);


      k = k + 4;
    }

    // Tail end
    for (; k < D_fm; k++) {

      const int z_idx = z_idx0 + k;

      double z = z_fm[z_idx];

      double d = z * v;

      pAcwfmk[k] = pAcwfmk[k] + d;

      wi2 += (d * d);
    }
  }

  for (int k = 0; k < D_fm; k++) {

    double wfmk = 0.0;

    for (int iThread = 0; iThread < num_thread; iThread++) {
      wfmk += acwfmk[iThread * D_fm + k];
    }

    w_fm[k] = wfmk;
    e2 += (wfmk* wfmk);
  }

  for (int k = 0; k < num_thread * 4; k++) {
    wi2 += wi2_acc[k];
  }


  delete[] acwfmk;
  acwfmk = nullptr;

  delete[] wi2_acc;
  wi2_acc = nullptr;

  e2 = (e2 - wi2)* 0.5 *weight_fm;
  return e + e2;

}
