from ifnclass.ifnfit import DualMixedPopulation
from ifnclass.ifnmodel import EnsembleModel
import os
import pandas as pd
import numpy as np
import pickle

ENSEMBLE = False
RANDOM_ENSEMBLE = False

if ENSEMBLE:
    SCALE_FACTOR = 1.5
    PLOT_KWARGS = {'line_type': 'envelope', 'alpha': 0.2}
    with open(os.path.join(os.getcwd(), 'pydream', 'GAB','PyDREAM_SETTINGS.py'), 'r') as f:
        s = f.read()
        NCHAINS = s.split('NCHAINS =')[1].split()[0].strip()
        NITERATIONS = s.split('NITERATIONS =')[1].split()[0].strip()
        PYDREAM_DIR = s.split('DIR_NAME =')[1].split()[0].strip(' "\'') + '_' + NITERATIONS
    if RANDOM_ENSEMBLE:
        DR_KWARGS = {'num_checks': 50}
    else:
        DR_KWARGS = {'num_checks': int(NCHAINS)}

else:
    SCALE_FACTOR = 1.227
    DR_KWARGS = {'return_type': 'IfnData'}
    PLOT_KWARGS = {'line_type': 'plot', 'alpha': 1}


def load_model(model_name='Mixed_IFN_ppCompatible', ENSEMBLE=ENSEMBLE, RANDOM_ENSEMBLE=RANDOM_ENSEMBLE):
    # Dose-response method must return a Pandas DataFrame which is compatible
    # with an IfnData object

    if ENSEMBLE:
        param_file_dir = os.path.join(os.getcwd(), 'pydream', 'GAB', PYDREAM_DIR)
        param_names = np.load(param_file_dir + os.sep + 'param_names.npy')
        prior_file_name = param_file_dir + os.sep + 'init_params.pkl'
        if RANDOM_ENSEMBLE:
            param_file_name = param_file_dir + os.sep + 'mixed_IFN_samples.npy'
        else:
            param_file_name = param_file_dir + os.sep + 'mixed_IFN_ML_samples.txt'
        model = EnsembleModel(model_name, param_file_name, param_names, prior_file_name)
        DR_method = model.posterior_prediction

    else:
        initial_parameters = {'k_a1': 4.98E-14 * 1.33, 'k_a2': 8.30e-13 * 2,
                              'k_d4': 0.006 * 3.8,
                              'kpu': 0.00095,
                              'ka2': 4.98e-13 * 1.33, 'kd4': 0.3 * 2.867,
                              'kint_a': 0.000124, 'kint_b': 0.00086,
                              'krec_a1': 0.0028, 'krec_a2': 0.01,
                              'krec_b1': 0.005, 'krec_b2': 0.05}
        dual_parameters = {'kint_a': 0.00052, 'kSOCSon': 6e-07,
                           'kint_b': 0.00052,
                           'krec_a1': 0.001, 'krec_a2': 0.1,
                           'krec_b1': 0.005, 'krec_b2': 0.05}
        model = DualMixedPopulation(model_name, 0.8, 0.2)
        model.model_1.set_parameters(initial_parameters)
        model.model_1.set_parameters(dual_parameters)
        model.model_1.set_parameters({'R1': 12000.0, 'R2': 1511.1})
        model.model_2.set_parameters(initial_parameters)
        model.model_2.set_parameters(dual_parameters)
        model.model_2.set_parameters({'R1': 6755.56, 'R2': 1511.1})

        # param_file_dir = os.path.join(os.getcwd(), 'pydream', 'GAB', 'PYDREAM_07-07-2020_10000')
        # with open(param_file_dir + os.sep + 'init_params.pkl', 'wb') as f:
        #     pickle.dump(dict(model.model_1.parameters), f)
        # exit()
        DR_method = model.mixed_dose_response

    return model, DR_method
