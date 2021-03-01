# PySB imports
from ifnclass.ifnfit import IfnModel
from ifnclass.ifndata import IfnData, DataAlignment
from PyDREAM_methods import IFN_posterior_object
from pydream.parameters import SampledParam

from scipy.stats import norm
import numpy as np
import os
from datetime import datetime


# -----------------------------------------------------------------------------
# Set simulation parameters
# -----------------------------------------------------------------------------
NITERATIONS = 10000
NCHAINS = 6
SIM_NAME = 'mixed_IFN'
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Model Setup
# -----------------------------------------------------------------------------
Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
sf = 1.0
custom_params = {}
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Parameters to fit:
# -----------------------------------------------------------------------------
pysb_sampled_parameter_names = ['kpa', 'kSOCSon', 'R1', 'R2', 'kd4',
                                'k_d4', 'kint_a', 'kint_b', 'krec_a2',
                                'krec_b2']

# Parameters to be sampled as unobserved random variables in DREAM:
original_params = np.log10([Mixed_Model.parameters[param] for
                            param in pysb_sampled_parameter_names])

priors_list = []
priors_dict = {}
for key in pysb_sampled_parameter_names:
    if key in ['ka1', 'ka2', 'k_a1', 'k_a2', 'R1', 'R2']:
        priors_list.append(SampledParam(norm,
                                        loc=np.log10(
                                            Mixed_Model.parameters[key]),
                                        scale=np.log10(2)))
        priors_dict.update({key: (np.log10(Mixed_Model.parameters[key]),
                                  np.log10(2))})
    else:
        priors_list.append(SampledParam(norm,
                                        loc=np.log10(
                                            Mixed_Model.parameters[key]),
                                        scale=2.0))
        priors_dict.update({key: (np.log10(
                                  Mixed_Model.parameters[key]), 2.0)})
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Preparing experimental data
# -----------------------------------------------------------------------------
mean_data = IfnData("MacParland_Extended")

# Define posterior function
posterior_obj = IFN_posterior_object(pysb_sampled_parameter_names,
                                     Mixed_Model, mean_data)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
def dir_setup(dir_name, fit_flag, bootstrap_flag, post_analysis_flag):
    """Used to set up the directory where output files will be saved during
    PyDREAM setup.
    """
    # Set up save directory
    if (fit_flag and bootstrap_flag) or\
       (post_analysis_flag and bootstrap_flag):
        raise RuntimeError("Runfile is unclear what directory to reference")

    if fit_flag:
        today = datetime.now()
        save_dir = "PyDREAM_" + today.strftime('%d-%m-%Y') + "_" +\
            str(NITERATIONS)
        os.makedirs(os.path.join(os.getcwd(), save_dir), exist_ok=True)
    elif bootstrap_flag:
        today = datetime.now()
        save_dir = "PyDREAM_" + today.strftime('%d-%m-%Y') + "_BOOTSTRAP"
        os.makedirs(os.path.join(os.getcwd(), save_dir), exist_ok=True)
    else:
        save_dir = dir_name
    return save_dir
