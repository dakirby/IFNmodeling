import numpy as np


PARAM_LOWER_BOUNDS = [0.1, 0.1, 0.1]
PARAM_UPPER_BOUNDS = [1.E6, 1.E6, 1.E6,]
def antiViralActivity(pSTAT, KM):
    return np.array(100 * pSTAT / (pSTAT + KM))

def antiProliferativeActivity(pSTAT, KM1, KM2):
    H1 = 2
    H2 = 4
    return np.nan_to_num(100 * (pSTAT**H1 / (pSTAT**H1 + KM1**H1) + pSTAT**H2 / (pSTAT**H2 + KM2**H2)) / 2)
