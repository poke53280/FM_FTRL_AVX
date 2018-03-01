
#include "SinglePredict.h"

#define USE_AVX2
#define USE_OMP

#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef USE_AVX2

#include <immintrin.h>

#endif

#include <malloc.h>

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
    }
    else {
      w[ii + 1] = 0.0;
    }
  }

  double wi2 = 0.0;

  int num_thread = 1;


#ifdef USE_OMP
  if (nThreads <= 0) {
    num_thread = omp_get_max_threads();
  }
  else {
    num_thread = nThreads;
  }
#endif


  double* acwfmk = (double*)malloc(sizeof(double) * D_fm * num_thread);

  int k;

#ifdef USE_OMP
#pragma omp parallel for num_threads(num_thread)
#endif
  for (k = 0; k < D_fm * num_thread; k++) {
    acwfmk[k] = 0.0;
  }

  double* wi2_acc = (double*)malloc(sizeof(double) * num_thread * 4);

#ifdef USE_OMP
#pragma omp parallel for num_threads(num_thread)
#endif
  for (k = 0; k < num_thread * 4; k++) {
    wi2_acc[k] = 0.0;
  }


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
   

    double*
      pAcwfmk = acwfmk + iThread * D_fm;

    double * wi2_acck = wi2_acc + iThread * 4;

    const int idx = inds[ii];
    double v = vals[ii];

    const int z_idx0 = idx * D_fm;

    int k = 0;

#ifdef USE_AVX2

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
#endif

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


  free(acwfmk);
  acwfmk = 0;

  free(wi2_acc);
  wi2_acc = 0;

  e2 = (e2 - wi2)* 0.5 *weight_fm;
  return e + e2;

}
