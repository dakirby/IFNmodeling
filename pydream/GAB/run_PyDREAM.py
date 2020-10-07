# PySB imports
from ifnclass.ifnfit import IfnModel

from PyDREAM_SETTINGS import NITERATIONS, NCHAINS, SIM_NAME
from pydream.parameters import SampledParam
from PyDREAM_methods import IFN_posterior_object, DREAM_fit

from scipy.stats import norm
import numpy as np
import os
from datetime import datetime


if __name__ == '__main__':
    # -------------------------------------------------
    # Model Setup
    # -------------------------------------------------
    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
    sf = 1.0
    custom_params = {}

    # Parameters to fit:
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
                                            scale=1.0))
            priors_dict.update({key: (np.log10(
                                      Mixed_Model.parameters[key]), 1.0)})

    posterior_obj = IFN_posterior_object(pysb_sampled_parameter_names,
                                         Mixed_Model)
    # -------------------------------------------------
    # Simulation
    # -------------------------------------------------
    today = datetime.now()
    save_dir = "PyDREAM_" + today.strftime('%d-%m-%Y') + "_" + str(NITERATIONS)
    os.makedirs(os.path.join(os.getcwd(), save_dir), exist_ok=True)

    # Save simulation parameters
    with open(os.path.join(save_dir, 'setup.txt'), 'w') as f:
        f.write('custom params:\n' + str(custom_params) + '\nscale factor = '
                + str(sf) +
                '\nParameter Vector:\n' + str(Mixed_Model.parameters) +
                '\nPrior Vector:\n' + str(priors_dict))

    DREAM_fit(model=Mixed_Model, priors_list=priors_list,
              posterior=posterior_obj.IFN_posterior,
              start_params=original_params,
              sampled_param_names=pysb_sampled_parameter_names,
              niterations=NITERATIONS,
              nchains=NCHAINS, sim_name=SIM_NAME, save_dir=save_dir)
