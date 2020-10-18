from ifnclass.ifnplot import DoseresponsePlot

from PyDREAM_SETTINGS import NITERATIONS, NCHAINS, SIM_NAME, dir_setup,\
    Mixed_Model, sf, custom_params, datalist,\
    posterior_obj, pysb_sampled_parameter_names, original_params,\
    priors_list, priors_dict

from PyDREAM_methods import DREAM_fit, posterior_IFN_summary_statistics,\
    bootstrap, _split_data

import numpy as np
import os
import pandas as pd


if __name__ == '__main__':
    train, test = _split_data(datalist, 20)
