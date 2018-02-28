
#ifndef _SINGLEPREDICT_H_
#define _SINGLEPREDICT_H_

#ifdef __cplusplus
extern "C" {
#endif

double predict_single_EXT(const int* inds, double* vals, int lenn, double L1, double baL2, double ialpha, double beta,
  double* w, double* z, double* n, double* w_fm, double* z_fm, double* n_fm, double weight_fm, int D_fm, bool bias_term, int nThreads);

int halfer_EXT_INT(int d);

#ifdef __cplusplus
}
#endif

#endif   // _SINGLEPREDICT_H_
