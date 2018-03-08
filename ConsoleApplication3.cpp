
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <assert.h>
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <iostream>

#include "avx_ext.h"

double inv_link_f(double e, int inv_link) {
  if (inv_link == 1) {
    return 1.0 / (1.0 + exp(-fmax(fmin(e, 35.0), -35.0))); // #Sigmoid + logloss
  }
  return e;
}

void update_single_BASELINE(const int* inds, double* vals, int lenn, const double e, double ialpha, double* w, double* z, double* n,
  double alpha_fm, const double L2_fm, double* w_fm, double* z_fm, double* n_fm, int D_fm, bool bias_term) {

  const double e_sq = e * e;

  if (bias_term) {
    const double ni = n[0];
    z[0] += e - ((sqrt(ni + e_sq) - sqrt(ni)) * ialpha) * w[0];
    n[0] += e_sq;
  }

  const double L2_fme = L2_fm / e;


  // Parallell for
  for (int ii = 0; ii < lenn; ii++) {

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

    __m256d reg2 = _mm256_set_pd(reg, reg, reg, reg);
    __m256d lr2 = _mm256_set_pd(lr, lr, lr, lr);

    int k = 0;

    while (k + 3 < D_fm) {

      double* z_offset = z_fmi + k;

      const double* w_offset = w_fm + k;

      __m256d z0 = _mm256_loadu_pd(z_offset);

      __m256d w = _mm256_loadu_pd(w_offset);

      __m256d z = _mm256_mul_pd(z0, reg2);

      __m256d w2 = _mm256_sub_pd(w, z);

      __m256d w3 = _mm256_mul_pd(lr2, w2);

      __m256d res = _mm256_sub_pd(z0, w3);

      _mm256_storeu_pd(z_offset, res);

      k = k + 4;
    }

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



double predict_single_BASELINE(const int* inds, double* vals, int lenn, double L1, double baL2, double ialpha, double beta,
  double* w, double* z, double* n, double* w_fm, double* z_fm, double* n_fm, double weight_fm, int D_fm, bool bias_term) {


  double e = 0.0;
  double e2 = 0.0;

  if (bias_term) {
    const double wi = w[0] = -z[0] / ((beta + sqrt(n[0])) * ialpha);
    e += wi;
  }

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

  int num_thread = 4;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, num_thread -1);

  double* acwfmk = new double[D_fm * num_thread];

  for (int k = 0; k < D_fm * num_thread; k++) {
    acwfmk[k] = 0.0;
  }

  double* wi2_acc = new double[num_thread * 4];

  for (int k = 0; k < num_thread * 4; k++) {
    wi2_acc[k] = 0.0;
  }

  for (int ii = 0; ii < lenn; ii++) {
    
    const int iThread = dis(gen);  // 0.. num_thread -1

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


void update_single_OMP(const int* inds, double* vals, int lenn, const double e, double ialpha, double* w, double* z, double* n,
                          double alpha_fm, const double L2_fm, double* w_fm, double* z_fm, double* n_fm, int D_fm, bool bias_term, int nThreads) {
  


  int num_thread;

  if (nThreads <= 0) {
    num_thread = omp_get_max_threads();
  }
  else {
    num_thread = nThreads;
  }


  printf("Running on %d threads\r\n", num_thread);


  const double e_sq = e * e;
 
  if (bias_term) {
    const double ni = n[0];
    z[0] += e - ((sqrt(ni + e_sq) - sqrt(ni)) * ialpha) * w[0];
    n[0] += e_sq;
  }

  const double L2_fme = L2_fm / e;
  

  // Parallell for
#pragma omp parallel for num_threads(num_thread)
  for (int ii = 0; ii < lenn; ii++) {

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
    
    __m256d reg2 = _mm256_set_pd(reg, reg, reg, reg);
    __m256d lr2 = _mm256_set_pd(lr, lr, lr, lr);

    int k = 0;

    while (k  + 3 < D_fm) {

      double* z_offset = z_fmi + k;

      const double* w_offset = w_fm + k;

      __m256d z0 = _mm256_loadu_pd(z_offset);

      __m256d w  = _mm256_loadu_pd(w_offset);

      __m256d z = _mm256_mul_pd(z0, reg2);

      __m256d w2 = _mm256_sub_pd(w, z);

      __m256d w3 = _mm256_mul_pd(lr2, w2);

      __m256d res = _mm256_sub_pd(z0, w3);

      _mm256_store_pd(z_offset, res);

      k = k + 4;
    }

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

double predict_single_OMP(const int* inds, double* vals, int lenn, double L1, double baL2, double ialpha, double beta,
  double* w, double* z, double* n, double* w_fm, double* z_fm, double* n_fm, double weight_fm, int D_fm, bool bias_term, int nThreads) {


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

    double * wi2_acc_thread = wi2_acc + iThread * 4;

    const int idx = inds[ii];
    double v = vals[ii];

    const int z_idx0 = idx * D_fm;

    int k = 0;

    __m256d v256 = _mm256_set_pd(v, v, v, v);

    __m256d w2_256 = _mm256_loadu_pd(wi2_acc_thread);

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

      w2_256 = _mm256_add_pd(w2_256, d);

      k = k + 4;
    }
    
    _mm256_storeu_pd(wi2_acc_thread, w2_256);


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



int main()
{

  const int lenn = 1000000;
  const int nRun = 100;
  const int num_permutations = 400;


  const int nThreads_EXT = 1;
  const int nThreads_omp = 1;

  int D_fm = 1024;

  double L1 = 0.00001;
  double baL2 = 0.1;
  double ialpha = 0.01;
  double beta = 0.1;

  double e = 0.0001;
  double alpha_fm = 0.01;
  double L2_fm = 0.0;

  double * vals0 = new double[lenn];
  double * vals1 = new double[lenn];

  double *w0 = new double[lenn + 1];
  double *w1 = new double[lenn + 1];

  double *z0 = new double[lenn];
  double *z1 = new double[lenn];

  double *n0 = new double[lenn];
  double *n1 = new double[lenn];

  double *n_fm0 = new double[lenn];
  double *n_fm1 = new double[lenn];

  std::vector<int> idx;

  for (int ii = 0; ii < lenn; ii++) {
    idx.push_back(ii);
  }

 
  std::vector<int> *P_idx0 = new std::vector<int>[num_permutations];
  std::vector<int> *P_idx1 = new std::vector<int>[num_permutations];

  for (int i = 0; i < num_permutations; i++) {
    std::random_shuffle(idx.begin(), idx.end());
    P_idx0[i] = idx;
    P_idx1[i] = idx;
  }

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  for (int ii = 0; ii < lenn; ii++) {

    vals0[ii] = 0.3 * distribution(generator);
    vals1[ii] = vals0[ii];

    w0[ii] = 0.9 * distribution(generator);
    w1[ii] = w0[ii];
    
    z0[ii] = 0.4 * distribution(generator);
    z1[ii] = z0[ii];
    
    n0[ii] = 0.1 * distribution(generator);
    n1[ii] = n0[ii];
    
    n_fm0[ii] = 0.2 * distribution(generator);
    n_fm1[ii] = n_fm0[ii];


  }
  w0[lenn] = 0.9;
  w1[lenn] = 0.9;

  double *w_fm0 = new double[D_fm];
  double *w_fm1 = new double[D_fm];

  for (int ii = 0; ii < D_fm; ii++) {
    w_fm0[ii] = 0.0;
    w_fm1[ii] = 0.0;
  }

  double *z_fm0 = new double[D_fm * lenn];
  double *z_fm1 = new double[D_fm * lenn];

  for (int ii = 0; ii < D_fm * lenn; ii++) {
    z_fm0[ii] = 0.04;
    z_fm1[ii] = 0.04;
  }
   
  double weight_fm = 0.1;

  bool bias_term = true;

  for (int iRun = 0; iRun < nRun; iRun++) {

    int iPermutation = iRun % num_permutations;

    const std::vector<int>& pInd0 = P_idx0[iPermutation];
    const std::vector<int>& pInd1 = P_idx1[iPermutation];

    const int * inds0 = pInd0.data();
    const int * inds1 = pInd1.data();

    using namespace std::chrono;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // Writes w_fm0
    double res_base = predict_single_EXT(inds0, vals0, lenn, L1, baL2, ialpha, beta, w0, z0, n0, w_fm0, z_fm0, n_fm0, weight_fm, D_fm, bias_term, nThreads_EXT);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    // Writes w_fm0
    double res_omp = predict_single_OMP(inds1, vals1, lenn, L1, baL2, ialpha, beta, w1, z1, n1, w_fm1, z_fm1, n_fm1, weight_fm, D_fm, bias_term, nThreads_omp);

    high_resolution_clock::time_point t3 = high_resolution_clock::now();

    // Writes z_fm
    update_single_EXT(inds0, vals0, lenn, e, ialpha, w0, z0, n0, alpha_fm, L2_fm, w_fm0, z_fm0, n_fm0, D_fm, bias_term, nThreads_EXT);

    high_resolution_clock::time_point t4 = high_resolution_clock::now();

    // Writes z_fm
    update_single_OMP(inds1, vals1, lenn, e, ialpha, w1, z1, n1, alpha_fm, L2_fm, w_fm1, z_fm1, n_fm1, D_fm, bias_term, nThreads_omp);

    high_resolution_clock::time_point t5 = high_resolution_clock::now();

    duration<double> time_predict_EXT = duration_cast<duration<double>>(t2 - t1);
    duration<double> time_predict_omp      = duration_cast<duration<double>>(t3 - t2);
    duration<double> time_update_EXT  = duration_cast<duration<double>>(t4 - t3);
    duration<double> time_update_omp       = duration_cast<duration<double>>(t5 - t4);

    std::cout << "time_predict_EXT     : " << time_predict_EXT.count() << " s." << std::endl;
    std::cout << "time_predict_omp     : " << time_predict_omp.count()      << " s." << std::endl;
    std::cout << "time_update_EXT      : "  << time_update_EXT.count()  << " s." << std::endl;
    std::cout << "time_update_omp: "       << time_update_omp.count()       << " s." << std::endl;


    double diff0 = abs(res_base - res_omp);

    if (diff0 > 0.0001) {
      printf("XXXXXXXXXXXXXXXXXXXXXXXErrorXXXXXXXXXXXXXXXXXXXXXXXX, diff0 = %f\r\n", diff0);
    }

    for (int k = 0; k < D_fm; k++) {
      double wdiff = abs(w_fm0[k] - w_fm1[k]);

      double zdiff = abs(z_fm0[k] - z_fm1[k]);

      if (wdiff > 0.0001 || zdiff > 0.0001) {
        printf("XXXXXXXXXXXXXXXXXXXXXXXXErrorXXXXXXXXXXXXXXXXXXXXXXXXXXXXx at k = %d\r\n", k);
      }


    }


    printf("run complete: %d\r\n", iRun);

  }

  delete[] vals0;
  vals0 = nullptr;

  delete[] vals1;
  vals1 = nullptr;


  delete[] w0;
  w0 = nullptr;

  delete[] w1;
  w1 = nullptr;

  delete[] z0;
  z0 = nullptr;

  delete[] z1;
  z1 = nullptr;

  delete[] n0;
  n0 = nullptr;

  delete[] n1;
  n1 = nullptr;

  delete[] w_fm0;
  w_fm0 = nullptr;

  delete[] w_fm1;
  w_fm1 = nullptr;

  delete[] z_fm0;
  z_fm0 = nullptr;

  delete[] z_fm1;
  z_fm1 = nullptr;

  delete[] n_fm0;
  n_fm0 = nullptr;

  delete[] n_fm1;
  n_fm1 = nullptr;

  delete[] P_idx0;
  P_idx0 = nullptr;

  delete[] P_idx1;
  P_idx1 = nullptr;

  return 0;


}

