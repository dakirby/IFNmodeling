from ifnclass.ifnplot import DoseresponsePlot

from PyDREAM_SETTINGS import NITERATIONS, NCHAINS, SIM_NAME, dir_setup,\
    Mixed_Model, sf, custom_params, datalist,\
    posterior_obj, pysb_sampled_parameter_names, original_params,\
    priors_list, priors_dict

from PyDREAM_methods import DREAM_fit, posterior_IFN_summary_statistics,\
    bootstrap

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
    save_dir = dir_setup("PyDREAM_18-10-2020_4",
                         fit_flag, bootstrap_flag, post_analysis_flag)

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
