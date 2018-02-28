
#ifndef _SINGLEUPDATE_H_
#define _SINGLEUPDATE_H_

#ifdef __cplusplus
extern "C" {
#endif

  void update_single_EXT(const int* inds, double* vals, int lenn, const double e, double ialpha, double* w, double* z, double* n,
    double alpha_fm, const double L2_fm, double* w_fm, double* z_fm, double* n_fm, int D_fm, bool bias_term, int nThreads);

  int doubler_EXT(int d);

#ifdef __cplusplus
}
#endif

#endif   // _SINGLEUPDATE_H_

