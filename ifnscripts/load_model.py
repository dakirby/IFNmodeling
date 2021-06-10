from ifnclass.ifnfit import DualMixedPopulation
from ifnclass.ifnmodel import EnsembleModel, IfnModel
import os
import pandas as pd
import numpy as np
import pickle

# Default is to use median parameters with variance arising from R1 and R2
MODEL_TYPE = 'MEDIAN'

if MODEL_TYPE in ['ENSEMBLE', 'BOOTSTRAP']:
    SCALE_FACTOR = 1.5
    PLOT_KWARGS = {'line_type': 'envelope', 'alpha': 0.2}
    with open(os.path.join(os.getcwd(), 'pydream', 'GAB','PyDREAM_SETTINGS.py'), 'r') as f:
        s = f.read()
        NCHAINS = s.split('NCHAINS =')[1].split()[0].strip()
        NITERATIONS = s.split('NITERATIONS =')[1].split()[0].strip()
        PYDREAM_DIR = s.split('DIR_NAME =')[1].split()[0].strip(' "\'') + '_' + NITERATIONS
    if MODEL_TYPE == 'ENSEMBLE':
        DR_KWARGS = {'num_checks': 50}
    else:
        DR_KWARGS = {'num_checks': int(NCHAINS)}

else:
    DR_KWARGS = {'return_type': 'IfnData'}
    if MODEL_TYPE == 'MEDIAN':
        PLOT_KWARGS = {'line_type': 'envelope', 'alpha': 0.2}
        SCALE_FACTOR = 1.5
    elif MODEL_TYPE == 'SINGLE_CELL':
        PLOT_KWARGS = {'line_type': 'plot', 'alpha': 1}
        SCALE_FACTOR = 1.5  # 1.227
    else:
        raise ValueError('Did not recognize model type in load_model.py')


def load_model(model_name='Mixed_IFN_ppCompatible', MODEL_TYPE=MODEL_TYPE):
    # Dose-response method must return a Pandas DataFrame which is compatible
    # with an IfnData object

    if MODEL_TYPE in ['ENSEMBLE', 'BOOTSTRAP']:  # MCMC ensemble
        param_file_dir = os.path.join(os.getcwd(), 'pydream', 'GAB', PYDREAM_DIR)
        param_names = np.load(param_file_dir + os.sep + 'param_names.npy')
        prior_file_name = param_file_dir + os.sep + 'init_params.pkl'

        if MODEL_TYPE == 'ENSEMBLE':  # use MCMC samples
            param_file_name = param_file_dir + os.sep + 'mixed_IFN_samples.npy'
        else:  # use maximum posterior from each MCMC chain
            param_file_name = param_file_dir + os.sep + 'mixed_IFN_ML_samples.txt'

        model = EnsembleModel(model_name, param_file_name, param_names, prior_file_name)
        DR_method = model.posterior_prediction

    else:  # Not MCMC

        if MODEL_TYPE == 'MEDIAN':  # use distribution variables, with * at end of name
            # median parameters from MCMC, but only use variance in R for model variance
            initial_parameters = {'kSOCSon': 1.03992e-06, 'kpa': 1.e-06,
                                  'kint_a': 3.737e-05, 'kint_b': 0.0002085,
                                  'krec_a1': 0.00179, 'krec_a2': 0.00912,
                                  'krec_b1': 8.049335e-05, 'krec_b2': 0.000801,
                                  'R1_mu*': 2000., 'R1_std*': 0.190,
                                  'R2_mu*': 2023., 'R2_std*': 0.182}

            param_file_dir = os.getcwd()
            param_names = np.array(list(initial_parameters.keys()))
            prior_file_name = param_file_dir + os.sep + 'init_params_temp.pkl'
            param_file_name = param_file_dir + os.sep + 'mixed_IFN_ML_samples_temp.txt'
            # create dummy files to initiate EnsembleModel instance
            with open(prior_file_name, 'wb') as f:
                temp_model = IfnModel(model_name)
                pickle.dump(dict(temp_model.parameters), f)
            print([initial_parameters], file=open(param_file_name, 'w'))
            # initiate model and DR method
            model = EnsembleModel(model_name, param_file_name, param_names, prior_file_name, num_dist_samples=30)
            DR_method = model.posterior_prediction
            # delete dummy files
            if os.path.isfile(prior_file_name):
                os.remove(prior_file_name)
            if os.path.isfile(param_file_name):
                os.remove(param_file_name)

        elif MODEL_TYPE == 'SINGLE_CELL':  # use stagewise fit params in DualMixedPopulation
            # initial_parameters = {'k_a1': 4.98E-14 * 1.33, 'k_a2': 8.30e-13 * 2,
            #                       'k_d4': 0.006 * 3.8,
            #                       'kpu': 0.00095, 'kSOCSon': 6e-07,
            #                       'ka2': 4.98e-13 * 1.33, 'kd4': 0.3 * 2.867,
            #                       'kint_a': 0.00052, 'kint_b': 0.00052,
            #                       'krec_a1': 0.001, 'krec_a2': 0.1,
            #                       'krec_b1': 0.005, 'krec_b2': 0.05}
            # model = DualMixedPopulation(model_name, 0.8, 0.2)
            # model.model_1.set_parameters(initial_parameters)
            # model.model_1.set_parameters({'R1': 12000.0, 'R2': 1511.1})
            # model.model_2.set_parameters(initial_parameters)
            # model.model_2.set_parameters({'R1': 6755.56, 'R2': 1511.1})
            # DR_method = model.mixed_dose_response
            median_parameters = {'kSOCSon': 1.03992e-06, 'kpa': 1.e-06,
                                 'kint_a': 3.737e-05, 'kint_b': 0.0002085,
                                 'krec_a1': 0.00179, 'krec_a2': 0.00912,
                                 'krec_b1': 8.049335e-05, 'krec_b2': 0.000801,
                                 'R1': 2000., 'R2': 2023.}
            # initiate model and DR method
            model = IfnModel(model_name)
            model.set_parameters(median_parameters)
            DR_method = model.doseresponse

        else:
            raise ValueError('Did not recognize model type requested')

    return model, DR_method
