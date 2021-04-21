from PyDREAM_SETTINGS import NITERATIONS, NCHAINS, SIM_NAME, DIR_NAME,\
    ITERATION_CUTOFF, dir_setup, Mixed_Model, sf, custom_params, datalist,\
    posterior_obj, pysb_sampled_parameter_names, original_params,\
    priors_list, priors_dict

from PyDREAM_methods import DREAM_fit, posterior_IFN_summary_statistics,\
    bootstrap, plot_posterior

import numpy as np
import os
import pandas as pd
import copy
import seaborn as sns


if __name__ == '__main__':
    # -------------------------------------------------
    # Runtime control
    # -------------------------------------------------
    fit_flag = False
    post_analysis_flag = False
    bootstrap_flag = True
    save_dir = dir_setup(DIR_NAME, fit_flag, bootstrap_flag, post_analysis_flag)

    # Plotting parameters
    plot_data = True
    time_mask = [2.5, 7.5, 10.0]
    num_checks = 50  # The number of posterior samples to use in post-analysis

    # -------------------------------------------------
    # PyDREAM Fitting
    # -------------------------------------------------
    if fit_flag:
        # Save simulation parameters
        with open(os.path.join(save_dir, 'setup.txt'), 'w') as f:
            f.write('custom params:\n' + str(custom_params) +
                    '\nscale factor = '
                    + str(sf) +
                    '\nParameter Vector:\n' + str(Mixed_Model.parameters) +
                    '\nPrior Vector:\n' + str(priors_dict))

        DREAM_fit(model=Mixed_Model, priors_list=priors_list,
                  posterior=posterior_obj.IFN_posterior,
                  start_params=original_params,
                  sampled_param_names=pysb_sampled_parameter_names,
                  niterations=NITERATIONS,
                  nchains=NCHAINS, sim_name=SIM_NAME, save_dir=save_dir,
                  iteration_cutoff=ITERATION_CUTOFF,
                  verbose=False)

    # -------------------------------------------------
    # Post-fitting analysis
    # -------------------------------------------------
    if post_analysis_flag:
        try:
            posterior_param_file = save_dir + os.sep + SIM_NAME +\
                                   '_samples.npy'
            parameter_names = pd.read_csv(save_dir + os.sep +
                                          'descriptive_statistics.csv').\
                columns.values[1:]
        except FileNotFoundError:
            batch_dirs = [x[0] for x in os.walk(save_dir)][1:]
            posterior_param_file = []
            for d in batch_dirs:
                p = os.path.join(os.getcwd(), d, SIM_NAME + '_samples.npy')
                posterior_param_file.append(p)
            parameter_names = pd.read_csv(
                              os.path.join(batch_dirs[0],
                                           'descriptive_statistics.csv')).\
                columns.values[1:]

        wd = os.path.join(os.getcwd(), save_dir)
        plot_posterior(posterior_param_file, parameter_names, num_checks,
                       Mixed_Model, posterior_obj, sf, time_mask,
                       wd, plot_data=True)

    if bootstrap_flag:
        bootstrap(Mixed_Model, datalist, priors_list, original_params,
                  pysb_sampled_parameter_names, NITERATIONS, NCHAINS,
                  SIM_NAME, save_dir, 20, 5, iteration_cutoff=ITERATION_CUTOFF)
