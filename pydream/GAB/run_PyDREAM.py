# PySB imports
from ifnclass.ifnfit import IfnModel
from ifnclass.ifnplot import DoseresponsePlot
from ifnclass.ifndata import IfnData, DataAlignment

from PyDREAM_SETTINGS import NITERATIONS, NCHAINS, SIM_NAME
from pydream.parameters import SampledParam
from PyDREAM_methods import IFN_posterior_object, DREAM_fit,\
    posterior_IFN_summary_statistics, bootstrap

from scipy.stats import norm
import numpy as np
import os
from datetime import datetime
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
        save_dir = "PyDREAM_07-10-2020_4"  # change to the desired directory

    # Plotting parameters
    plot_data = True
    time_mask = [2.5, 7.5, 10.0]
    num_checks = 50  # The number of posterior samples to use in post-analysis

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

    # Align all experimental data
    newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
    newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
    newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
    newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")
    datalist = [newdata_4, newdata_3, newdata_2, newdata_1]

    alignment = DataAlignment()
    alignment.add_data(datalist)
    alignment.align()
    alignment.get_scaled_data()

    # Define posterior function
    posterior_obj = IFN_posterior_object(pysb_sampled_parameter_names,
                                         Mixed_Model, alignment)
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
                  nchains=NCHAINS, sim_name=SIM_NAME, save_dir=save_dir)

    # -------------------------------------------------
    # Post-fitting analysis
    # -------------------------------------------------
    if post_analysis_flag:
        # Preparation of parameter ensemble
        log10_parameters = np.load(save_dir + os.sep + SIM_NAME +
                                   '_samples.npy')
        parameters = np.power(10, log10_parameters)

        parameter_names = pd.read_csv(save_dir + os.sep +
                                      'descriptive_statistics.csv').\
            columns.values[1:]

        parameters_to_check = [parameters[i] for i in
                               list(np.random.randint(0, high=len(parameters),
                                                      size=num_checks))]

        # Compute posterior sample trajectories
        posterior_trajectories = []
        for p in parameters_to_check:
            param_dict = {key: value for key, value in zip(parameter_names, p)}
            pp = posterior_IFN_summary_statistics(Mixed_Model, param_dict,
                                                  parameter_names, sf)
            posterior_trajectories.append(pp)

        # Make aggregate predicitions
        mean_alpha, std_alpha, mean_beta, std_beta = \
            posterior_IFN_summary_statistics(posterior_trajectories)

        std_predictions = {'Alpha': std_alpha,
                           'Beta': std_beta}
        mean_predictions = {'Alpha': mean_alpha,
                            'Beta': mean_beta}

        mean_model = copy.deepcopy(posterior_trajectories[0])
        for s in ['Alpha', 'Beta']:
            for didx, d in enumerate(mean_model.get_doses()[s]):
                for tidx, t in enumerate(mean_model.get_times()[s]):
                    mean_model.data_set.loc[s][str(t)].loc[d] =\
                        (mean_predictions[s][didx][tidx],
                         std_predictions[s][didx][tidx])

        # Get aligned data
        mean_data = posterior_obj.aligned_data.summarize_data()

        # Plot posterior samples
        alpha_palette = sns.color_palette("deep", 6)
        beta_palette = sns.color_palette("deep", 6)

        new_fit = DoseresponsePlot((1, 2))

        # Add fits
        for idx, t in enumerate([2.5, 5.0, 7.5, 10.0, 20.0, 60.0]):
            if t not in time_mask:
                new_fit.add_trajectory(mean_model, t, 'envelope',
                                       alpha_palette[idx], (0, 0), 'Alpha',
                                       label='{} min'.format(t),
                                       linewidth=2, alpha=0.2)
                if plot_data:
                    new_fit.add_trajectory(mean_data, t, 'errorbar', 'o',
                                           (0, 0), 'Alpha',
                                           color=alpha_palette[idx])
            if t not in time_mask:
                new_fit.add_trajectory(mean_model, t, 'envelope',
                                       beta_palette[idx], (0, 1), 'Beta',
                                       label='{} min'.format(t),
                                       linewidth=2, alpha=0.2)
                if plot_data:
                    new_fit.add_trajectory(mean_data, t, 'errorbar', 'o',
                                           (0, 1), 'Beta',
                                           color=beta_palette[idx])

        # Change legend transparency
        leg = new_fit.fig.legend()
        for lh in leg.legendHandles:
            lh._legmarker.set_alpha(1)

        dr_fig, dr_axes = new_fit.show_figure()
        dr_fig.set_size_inches(14, 6)
        dr_axes[0].set_title(r'IFN$\alpha$')
        dr_axes[1].set_title(r'IFN$\beta$')

        if plot_data:
            dr_fig.savefig(os.path.join(os.getcwd(), save_dir,
                           'posterior_predictions_with_data.pdf'))
        else:
            dr_fig.savefig(os.path.join(os.getcwd(), save_dir,
                           'posterior_predictions.pdf'))

    if bootstrap_flag:
        bootstrap(Mixed_Model, datalist, priors_list, original_params,
                  pysb_sampled_parameter_names, NITERATIONS, NCHAINS,
                  SIM_NAME, save_dir, 20, 5)
