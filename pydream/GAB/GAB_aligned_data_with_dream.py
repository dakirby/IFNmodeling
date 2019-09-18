# Pydream imports
from pydream.core import run_dream
from pysb.integrate import Solver
import numpy as np
from pydream.parameters import SampledParam
from scipy.stats import norm, uniform
import os
from datetime import datetime

import inspect
from pydream.convergence import Gelman_Rubin

# PySB imports
from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnfit import DualMixedPopulation
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


#Initialize PySB model object for running simulations.  Simulation timespan should match experimental data.
initial_parameters = {'k_a1': 4.98E-14 * 2, 'k_a2': 8.30e-13 * 2, 'k_d4': 0.006 * 3.8,
                       'kpu': 0.00095,
                       'ka2': 4.98e-13 * 2.45, 'kd4': 0.3 * 2.867,
                       'kint_a': 0.000124, 'kint_b': 0.00086,
                       'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05}
dual_parameters = {'kint_a': 0.00052, 'kSOCSon': 6e-07, 'kint_b': 0.00052, 'krec_a1': 0.001, 'krec_a2': 0.1,
                   'krec_b1': 0.005, 'krec_b2': 0.05}
scale_factor = 1.227

Mixed_Model = DualMixedPopulation('Mixed_IFN_ppCompatible', 0.8, 0.2)
Mixed_Model.model_1.set_parameters(initial_parameters)
Mixed_Model.model_1.set_parameters(dual_parameters)
Mixed_Model.model_1.set_parameters({'R1': 12000.0, 'R2': 1511.1})
Mixed_Model.model_2.set_parameters(initial_parameters)
Mixed_Model.model_2.set_parameters(dual_parameters)
Mixed_Model.model_2.set_parameters({'R1': 6755.56, 'R2': 1511.2})

tspan = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
alpha_doses = [10, 100, 300, 1000, 3000, 10000, 100000]
beta_doses = [0.2, 6, 20, 60, 200, 600, 2000]

# Load experimental data to which model will be fit.
# The "experimental data" is the TotalpSTAT trajectory for Alpha, then Beta, at each dose.
# Standard deviations are kept separate but match the same order.
newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")

alignment = DataAlignment()
alignment.add_data([newdata_4, newdata_3, newdata_2, newdata_1])
alignment.align()
alignment.get_scaled_data()
mean_data = alignment.summarize_data()
exp_data = np.array([[el[0] for el in dose] for dose in mean_data.data_set.values]).flatten()
exp_data_std = np.array([[el[1] for el in dose] for dose in mean_data.data_set.values]).flatten()

#Create scipy normal probability distributions for data likelihoods
like_ctot = norm(loc=exp_data, scale=exp_data_std)

#Create lists of sampled pysb parameter names to use for subbing in parameter values in likelihood function.
pysb_sampled_parameter_names = ['kpa', 'kSOCSon', 'kd4', 'k_d4', 'R1_1', 'R2_1', 'R1_2', 'R2_2']

#Define likelihood function to generate simulated data that corresponds to experimental time points.  
#This function should take as input a parameter vector (parameter values are in the same order as
# pysb_sampled_parameter_names).
#The function returns a log probability value for the parameter vector given the experimental data.

def likelihood(parameter_vector, sampled_parameter_names=pysb_sampled_parameter_names,
               model=Mixed_Model):

    # Change model parameter values to current location in parameter space (values are in log(value) format)

    shared_param_dict = {pname: 10 ** pvalue for pname, pvalue in zip(sampled_parameter_names, parameter_vector)
                         if pname[-2] != '_'}
    pop1_param_dict = {pname: 10 ** pvalue for pname, pvalue in zip(sampled_parameter_names, parameter_vector)
                         if pname[-2:] == '_1'}
    pop2_param_dict = {pname: 10 ** pvalue for pname, pvalue in zip(sampled_parameter_names, parameter_vector)
                         if pname[-2:] == '_2'}

    model.set_global_parameters(shared_param_dict)
    model.model_1.set_parameters(pop1_param_dict)
    model.model_2.set_parameters(pop2_param_dict)

    #Simulate experimentally measured TotalpSTAT values.
    # Alpha
    dradf = model.mixed_dose_response(tspan, 'TotalpSTAT', 'Ia', alpha_doses,
                                            parameters={'Ib': 0}, sf=scale_factor)
    drbdf = model.mixed_dose_response(tspan, 'TotalpSTAT', 'Ib', beta_doses,
                                            parameters={'Ia': 0}, sf=scale_factor)

    # Concatenate and flatten
    total_simulation_data = IfnData(name='custom', df=pd.concat([dradf, drbdf]))
    sim_data = np.array([[el[0] for el in dose] for dose in total_simulation_data.data_set.values]).flatten()

    #Calculate log probability contribution from simulated experimental values.
    
    logp_ctotal = np.sum(like_ctot.logpdf(sim_data))
    
    #If model simulation failed due to integrator errors, return a log probability of -inf.
    if np.isnan(logp_ctotal):
        logp_ctotal = -np.inf
      
    return logp_ctotal


# Add vector of PySB rate parameters to be sampled as unobserved random variables to DREAM with log normal priors.
original_params = np.log10([Mixed_Model.get_parameters()[param] for param in pysb_sampled_parameter_names])

priors_list = []
for key in pysb_sampled_parameter_names:
    priors_list.append(SampledParam(norm, loc=np.log10(Mixed_Model.get_parameters()[key]), scale=1.0))

# Set simulation parameters
niterations = 10000
converged = False
total_iterations = niterations
nchains = 5
sim_name = 'mixed_IFN'

if __name__ == '__main__':

    # Make save directory
    today = datetime.now()
    save_dir = "PyDREAM_" + today.strftime('%d-%m-%Y') + "_" + str(niterations)
    os.makedirs(os.path.join(os.getcwd(), save_dir), exist_ok=True)

    #Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    sampled_params, log_ps = run_dream(priors_list, likelihood, start=original_params,
                                       niterations=niterations, nchains=nchains, multitry=False,
                                       gamma_levels=4, adapt_gamma=True, history_thin=1, model_name=sim_name,
                                       verbose=True)
    
    #Save sampling output (sampled parameter values and their corresponding logps).
    for chain in range(len(sampled_params)):
        np.save(os.path.join(save_dir, sim_name+str(chain) + '_' + str(total_iterations)), sampled_params[chain])
        np.save(os.path.join(save_dir, sim_name+str(chain) + '_' + str(total_iterations)), log_ps[chain])

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
            np.savetxt(os.path.join(save_dir, sim_name + '_' + str(total_iterations)+'.txt'), GR)

            if np.all(GR < 1.2):
                converged = True
    
    try:
        # Plot output
        total_iterations = len(old_samples[0])
        burnin = int(total_iterations / 2)
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
