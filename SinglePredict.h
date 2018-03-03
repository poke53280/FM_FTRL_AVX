
#ifndef _SINGLEPREDICT_H_
#define _SINGLEPREDICT_H_

double predict_single_EXT(const int* inds, double* vals, int lenn, double L1, double baL2, double ialpha, double beta,
  double* w, double* z, double* n, double* w_fm, double* z_fm, double* n_fm, double weight_fm, int D_fm, int bias_term, int nThreads);

#endif   // _SINGLEPREDICT_H_
