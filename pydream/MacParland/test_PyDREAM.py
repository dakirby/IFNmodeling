from ifnclass.ifnplot import DoseresponsePlot

from PyDREAM_SETTINGS import NITERATIONS, NCHAINS, SIM_NAME, dir_setup,\
    Mixed_Model, sf, custom_params, datalist,\
    posterior_obj, pysb_sampled_parameter_names, original_params,\
    priors_list, priors_dict

from PyDREAM_methods import DREAM_fit, posterior_IFN_summary_statistics,\
    bootstrap, _split_data, _get_data_coordinates, IFN_posterior_object

import numpy as np
import os
import pandas as pd


if __name__ == '__main__':
    np.random.seed(1)
    train, test = _split_data(datalist, 20)
    print(train.data_set)
    posterior_obj = IFN_posterior_object(pysb_sampled_parameter_names,
                                         Mixed_Model, train)

    save_dir = dir_setup("PyDREAM_31-01-2021_6", False, True, False)
    bootstrap(Mixed_Model, datalist, priors_list, original_params,
              pysb_sampled_parameter_names, NITERATIONS, NCHAINS,
              SIM_NAME, save_dir, 20, 5)
