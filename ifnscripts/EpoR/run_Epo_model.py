from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from numpy import linspace, logspace, transpose
import numpy as np
import seaborn as sns
from ifnclass.ifnplot import Trajectory, DoseresponsePlot
import matplotlib.pyplot as plt
import pandas as pd
import shutil
# PyDREAM imports
from pydream.core import run_dream
from pysb.integrate import Solver
from pydream.parameters import SampledParam
from scipy.stats import norm, uniform
import os
from datetime import datetime
import inspect
from pydream.convergence import Gelman_Rubin

# Define likelihood function to generate simulated data that corresponds to experimental time points.
# This function should take as input a parameter vector (parameter values are in the same order as
# pysb_sampled_parameter_names).
# The likelihood function returns a log probability value for the parameter vector given the experimental data.
def make_likelihood(times, doses, like_ctot, model_name, param_names, response_species, dose_species):
    IC_names = ['Epo_IC', 'EMP1_IC', 'EMP33_IC']
    def likelihood(parameter_vector, sampled_parameter_names=param_names):
        # Make model
        from ifnclass.ifnmodel import IfnModel
        import numpy as np
        model = IfnModel(model_name)

        # Change model parameter values to current location in parameter space (values are in log10(value) format)
        param_dict = {pname: 10 ** pvalue for pname, pvalue in zip(sampled_parameter_names, parameter_vector)}
        model.set_parameters(param_dict)

        # Simulate experimentally measured values.

        total_simulation_data = model.doseresponse(times, response_species, dose_species, doses,
                                                  parameters={n: 0 for n in IC_names if n != dose_species},
                                                  return_type='dataframe', dataframe_labels=dose_species,
                                                  scale_factor=100.0 / model.parameters['EpoR_IC'])

        sim_data = np.array([[el[0] for el in dose] for dose in total_simulation_data.values]).flatten()

        # Calculate log probability contribution from simulated experimental values.

        logp_ctotal = np.sum(like_ctot.logpdf(sim_data))

        # If model simulation failed due to integrator errors, return a log probability of -inf.
        if np.isnan(logp_ctotal):
            logp_ctotal = -np.inf

        return logp_ctotal
    return likelihood


if __name__ == '__main__':
    # --------------------
    # Import data
    #---------------------
    Moraga_data = IfnData("2015_pSTAT5_Epo")

    # --------------------
    # Set up Model
    # --------------------
    model = IfnModel('Epo_model')

    # ---------------------------------
    # Make theory dose response curves
    # ---------------------------------
    IC_names = ['Epo_IC', 'EMP1_IC', 'EMP33_IC']
    # Make predictions
    response_species = 'T_Epo'
    dose_species = 'Epo_IC'
    times = [60.0] # min
    Epo_doses = Moraga_data.get_doses()['T_Epo'] # pM
    dr_Epo = IfnData('custom',
                     df=model.doseresponse(times, response_species, dose_species, Epo_doses,
                                           parameters={n: 0 for n in IC_names if n != dose_species},
                                           return_type='dataframe', dataframe_labels=dose_species,
                                           scale_factor=100.0/model.parameters['EpoR_IC']),
                     conditions={})

    # -------------------------------
    # Plot model dose response curves
    # -------------------------------
    palette = sns.color_palette("muted")

    Epo_plot = DoseresponsePlot((1, 1))
    # Add data
    Epo_plot.add_trajectory(Moraga_data, 60, 'errorbar', 'o', (0, 0), 'T_Epo', label='EPO', color=palette[1])
    Epo_plot.add_trajectory(Moraga_data, 60, 'errorbar', 'o', (0, 0), 'EMP_1', label='EMP-1', color=palette[2])
    Epo_plot.add_trajectory(Moraga_data, 60, 'errorbar', 'o', (0, 0), 'EMP_33', label='EMP-33', color=palette[6])
    # Add fits
    Epo_plot.add_trajectory(dr_Epo, 60.0, 'plot', palette[1], (0, 0), 'Epo_IC', label='EPO Model',
                           linewidth=2)
    Epo_plot.axes.set_title('Response at 60 min')
    Epo_plot.show_figure()

    # --------------------------------
    # Fit model to data
    # --------------------------------
    niterations = 10000
    nchains = 5
    burn_factor = 0.5
    sim_name = 'Epo'

    fit_flag = False
    if fit_flag == True:
        # Make save directory
        today = datetime.now()
        save_dir = "PyDREAM_" + today.strftime('%d-%m-%Y') + "_" + str(niterations)
        os.makedirs(os.path.join(os.getcwd(), save_dir), exist_ok=True)

        # Create scipy normal probability distributions for data likelihoods
        like_ctot = norm(loc=[el[0][0] for el in Moraga_data.get_responses()['T_Epo']],
                         scale=[el[0][1] for el in Moraga_data.get_responses()['T_Epo']])

        # Create lists of sampled pysb parameter names to use for subbing in parameter values in likelihood function.
        pysb_sampled_parameter_names = ['kpa', 'k_1_epo', 'k_2_epo']
        pysb_sampled_parameter_log10_values = np.log10(np.array([model.parameters[key] for key in pysb_sampled_parameter_names]))

        priors_list = []
        for idx, key in enumerate(pysb_sampled_parameter_names):
            priors_list.append(SampledParam(norm, loc=pysb_sampled_parameter_log10_values[idx], scale=2.0))

        # Create likelihood function
        likelihood = make_likelihood(times, Epo_doses, like_ctot, 'Epo_model', pysb_sampled_parameter_names,
                                     response_species, dose_species)

        # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
        total_iterations = niterations
        converged = False

        sampled_params, log_ps = run_dream(priors_list, likelihood, start=pysb_sampled_parameter_log10_values,
                                           niterations=niterations, nchains=nchains, multitry=False,
                                           gamma_levels=4, adapt_gamma=True, history_thin=1, model_name=sim_name,
                                           verbose=True)

        # Save sampling output (sampled parameter values and their corresponding logps).
        for chain in range(len(sampled_params)):
            np.save(os.path.join(save_dir, sim_name + str(chain) + '_' + str(total_iterations)), sampled_params[chain])
            np.save(os.path.join(save_dir, sim_name + str(chain) + '_' + str(total_iterations)), log_ps[chain])

        # Check convergence and continue sampling if not converged

        GR = Gelman_Rubin(sampled_params)
        print('At iteration: ', total_iterations, ' GR = ', GR)
        np.savetxt(os.path.join(save_dir, sim_name + str(total_iterations) + '.txt'), GR)

        old_samples = sampled_params
        if np.any(GR > 1.2):
            starts = [sampled_params[chain][-1, :] for chain in range(nchains)]
            while not converged:
                total_iterations += niterations

                sampled_params, log_ps = run_dream(priors_list, likelihood, start=starts, niterations=niterations,
                                                   nchains=nchains, multitry=False, gamma_levels=4, adapt_gamma=True,
                                                   history_thin=1, model_name=sim_name, verbose=True, restart=True)

                for chain in range(len(sampled_params)):
                    np.save(os.path.join(save_dir, sim_name + '_' + str(chain) + '_' + str(total_iterations)),
                            sampled_params[chain])
                    np.save(os.path.join(save_dir, sim_name + '_' + str(chain) + '_' + str(total_iterations)),
                            log_ps[chain])

                old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
                GR = Gelman_Rubin(old_samples)
                print('At iteration: ', total_iterations, ' GR = ', GR)
                np.savetxt(os.path.join(save_dir, sim_name + '_' + str(total_iterations) + '.txt'), GR)

                if np.all(GR < 1.2):
                    converged = True

        try:
            # Clean up PYDREAM save file output
            shutil.move(os.path.join(os.getcwd(), sim_name + '_DREAM_chain_adapted_crossoverprob.npy'), save_dir)
            shutil.move(os.path.join(os.getcwd(), sim_name + '_DREAM_chain_adapted_gammalevelprob.npy'), save_dir)
            shutil.move(os.path.join(os.getcwd(), sim_name + '_DREAM_chain_history.npy'), save_dir)

            # Plot output
            total_iterations = len(old_samples[0])
            burnin = int(total_iterations * burn_factor)
            samples = np.concatenate(list((old_samples[i][burnin:, :] for i in range(len(old_samples)))))
            np.save(os.path.join(save_dir, sim_name+'_samples'), samples)
            ndims = len(old_samples[0][0])
            colors = sns.color_palette(n_colors=ndims)
            for dim in range(ndims):
                fig = plt.figure()
                sns.distplot(samples[:, dim], color=colors[dim])
                fig.savefig(os.path.join(save_dir, sim_name + '_dimension_' + str(dim) + '_' + pysb_sampled_parameter_names[dim]+ '.pdf'))

            # Convert back to true value rather than log value
            # converted_samples = np.power(np.multiply(np.ones(np.shape(samples)), 10), samples)
            # Convert to dataframe
            df = pd.DataFrame(samples, columns=pysb_sampled_parameter_names)
            g = sns.pairplot(df)
            for i, j in zip(*np.triu_indices_from(g.axes, 1)):
                g.axes[i,j].set_visible(False)
            g.savefig(os.path.join(save_dir, 'corner_plot.pdf'))

            # Basic statistics
            mean_parameters = np.mean(samples, axis=0)
            median_parameters = np.median(samples, axis=0)
            np.save(os.path.join(save_dir, 'mean_parameters'), mean_parameters)
            np.save(os.path.join(save_dir, 'median_parameters'), median_parameters)
            df.describe().to_csv(os.path.join(save_dir, 'descriptive_statistics.csv'))

        except ImportError:
            pass
