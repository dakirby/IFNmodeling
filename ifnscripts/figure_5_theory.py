import numpy as np


PARAM_LOWER_BOUNDS = [0.1, 0.1, 0.1, 0.1, 0.1]
PARAM_UPPER_BOUNDS = [1.E6, 1.E6, 1.E6, 2, 4]
def antiViralActivity(pSTAT, KM):
    return np.array(100 * pSTAT / (pSTAT + KM))

def antiProliferativeActivity(pSTAT, KM1, KM2, H1, H2):
    # H1 = 2
    # H2 = 4
    return np.nan_to_num(100 * (1 / (1 + (KM1 / pSTAT)**H1) + 1 / (1 + (KM2 / pSTAT)**H2)) / 2)
