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
NITERATIONS = 500
ITERATION_CUTOFF = 1000
NCHAINS = 5
SIM_NAME = 'mixed_IFN'
DIR_NAME = 'PyDREAM_30-08-2021'
AFFINITY_SPECIES = 'MOUSE'  #'HUMAN'

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Model Setup
# -----------------------------------------------------------------------------
Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
if AFFINITY_SPECIES == 'MOUSE':
    NAxvolEC = 6.022E23 * 1E-5
    affinity_parameters = {'ka1': 3.75E5 / NAxvolEC,  # 2666 nM Kd -> 3.75E5 M^-1 association rate when kd1=1
                           'ka2': 6.88E6 / NAxvolEC,  # 2.18 nM -> 6.88E6 M^-1 association rate when kd2=0.015
                           'k_a1': 2.368E6 / NAxvolEC,  # 12.67 nM Kd -> 2.368E6 M^-1 association rate when k_d1=0.03
                           'k_a2': 1.195E3 / NAxvolEC}  # 1673 nM -> 1.195E3 M^-1 association rate when kd2=0.002
    Mixed_Model.set_parameters(affinity_parameters)
    Mixed_Model.set_default_parameters(Mixed_Model.parameters)
sf = 1.5
custom_params = {}
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Parameters to fit:
# -----------------------------------------------------------------------------
pysb_sampled_parameter_names = ['kpa', 'kSOCSon', 'R1*', 'R2*', 'kint_a', 'kint_b', 'krec_a1', 'krec_a2']

original_params = []
priors_list = []
priors_dict = {}
for key in pysb_sampled_parameter_names:
    # build prior
    if key in ['kd4', 'k_d4', 'R1', 'R2']:
        original_params.append(Mixed_Model.parameters[key])
        mu = np.log10(Mixed_Model.parameters[key])
        std = 0.2
        priors_list.append(SampledParam(norm, loc=mu, scale=std))
        priors_dict.update({key: (mu, std)})
    elif key in ['R1*', 'R2*']:
        # set mean prior
        original_params.append(Mixed_Model.parameters[key[:-1]])
        mu = np.log10(Mixed_Model.parameters[key[:-1]])
        std = 1.0
        priors_list.append(SampledParam(norm, loc=mu, scale=std))
        priors_dict.update({key[:-1] + '_mu*': (mu, std)})
        # set std prior
        original_params.append(0.2)
        mu = np.log10(0.2)
        std = 0.5
        priors_list.append(SampledParam(norm, loc=mu, scale=std))
        priors_dict.update({key[:-1] + '_std*': (mu, std)})
    else:
        original_params.append(Mixed_Model.parameters[key])
        mu = np.log10(Mixed_Model.parameters[key])
        std = 1.0
        priors_list.append(SampledParam(norm, loc=mu, scale=std))
        priors_dict.update({key: (mu, std)})
original_params = np.log10(original_params)
pysb_sampled_parameter_names = list(priors_dict.keys())

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Preparing experimental data
# -----------------------------------------------------------------------------
newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")
datalist = [newdata_4, newdata_3, newdata_2, newdata_1]

alignment = DataAlignment()
alignment.add_data(datalist)
alignment.align()
alignment.get_scaled_data()
mean_data = alignment.summarize_data()

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
