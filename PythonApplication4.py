

import sys

sys.path.append('D:\\anders\\FM_FTRL_AVX\\')

import numpy.random as rnd
from hellocython import FM_FTRL_EXP

a = FM_FTRL_EXP(9)

print(a)


from wordbatch.models import FM_FTRL

b = FM_FTRL(11)




model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=sparse_merge.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                    D_fm=200, e_noise=0.0001, iters=17, inv_link="identity", threads=4)