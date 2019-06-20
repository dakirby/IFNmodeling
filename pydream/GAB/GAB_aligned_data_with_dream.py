# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:58:34 2016
@author: Erin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 15:26:46 2014
@author: Erin
"""
# Pydream imports
from pydream.core import run_dream
from pysb.integrate import Solver
import numpy as np
from pydream.parameters import SampledParam
from scipy.stats import norm, uniform
import os
import inspect
from pydream.convergence import Gelman_Rubin

# PySB imports
from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
import pandas as pd

#Initialize PySB model object for running simulations.  Simulation timespan should match experimental data.
Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
# Optimal parameters for fitting mean GAB data
opt_params = {'R2': 4920, 'R1': 1200,
                'k_a1': 2.0e-13, 'k_a2': 1.328e-12, 'k_d3': 1.13e-4, 'k_d4': 0.9,
                'kSOCSon': 5e-08, 'kpu': 0.0022, 'kpa': 2.36e-06,
                'ka1': 3.3e-15, 'ka2': 1.85e-12, 'kd4': 2.0,
                'kd3': 6.52e-05,
                'kint_a': 0.0015, 'kint_b': 0.002,
                'krec_a1': 0.01, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05}
Mixed_Model.set_parameters(opt_params)
Mixed_Model.default_parameters.update(opt_params)
sf = 1.46182313424
#model = Mixed_Model.model

tspan = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
alpha_doses = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
beta_doses = [0, 0.2, 6, 20, 60, 200, 600, 2000]

print(Mixed_Model.doseresponse(tspan, 'TotalpSTAT', 'Ia', alpha_doses,
                               parameters={'Ib': 0}, scale_factor=sf, return_type='dataframe', dataframe_labels='Alpha'))
exit()

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
pysb_sampled_parameter_names = ['kpa', 'kSOCSon', 'kd4', 'k_d4', 'R1', 'R2']

#Define likelihood function to generate simulated data that corresponds to experimental time points.  
#This function should take as input a parameter vector (parameter values are in the same order as
# pysb_sampled_parameter_names).
#The function returns a log probability value for the parameter vector given the experimental data.

def likelihood(parameter_vector):

    # Change model parameter values to current location in parameter space (values are in log(value) format)

    param_dict = {pname: 10 ** pvalue for pname, pvalue in zip(pysb_sampled_parameter_names, parameter_vector)}
    Mixed_Model.set_parameters(param_dict)

    #Simulate experimentally measured TotalpSTAT values.
    # Alpha
    dfa = Mixed_Model.doseresponse(tspan, 'TotalpSTAT', 'Ia', alpha_doses,
                                          parameters={'Ib': 0}, scale_factor=sf)
    dfa = IfnData(name='custom', df=dfa, conditions={'Ib': 0})
    # Beta
    dfb = Mixed_Model.doseresponse(tspan, 'TotalpSTAT', 'Ib', beta_doses,
                                          parameters={'Ib': 0}, scale_factor=sf)
    dfb = IfnData(name='custom', df=dfb, conditions={'Ib': 0})
    # Concatenate and flatten
    total_simulation_data = IfnData(name='custom', df=pd.concat([dfa, dfb]))
    sim_data = np.concatenate((np.array([[el[0] for el in dose] for dose in total_simulation_data.data_set.values]).flatten(),
                               np.array([[el[0] for el in dose] for dose in total_simulation_data.data_set.values]).flatten()))

    #Calculate log probability contribution from simulated experimental values.
    
    logp_ctotal = np.sum(like_ctot.logpdf(sim_data))
    
    #If model simulation failed due to integrator errors, return a log probability of -inf.
    if np.isnan(logp_ctotal):
        logp_ctotal = -np.inf
      
    return logp_ctotal


# Add vector of PySB rate parameters to be sampled as unobserved random variables to DREAM with log normal priors.
  
original_params = np.log10([Mixed_Model.parameters[param] for param in pysb_sampled_parameter_names])
priors_list = []
for key in pysb_sampled_parameter_names:
    priors_list.append(SampledParam(norm, loc=np.log10(Mixed_Model.parameters[key]), scale=2.0))

# Set simulation parameters
niterations = 1000
converged = False
total_iterations = niterations
nchains = 5
sim_name = 'mixed_IFN'

if __name__ == '__main__':
    #Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    sampled_params, log_ps = run_dream(priors_list, likelihood, niterations=niterations, nchains=nchains, multitry=False, gamma_levels=4, adapt_gamma=True, history_thin=1, model_name=sim_name, verbose=True)
    
    #Save sampling output (sampled parameter values and their corresponding logps).
    for chain in range(len(sampled_params)):
        np.save(sim_name+str(chain) + '_' + str(total_iterations), sampled_params[chain])
        np.save(sim_name+str(chain) + '_' + str(total_iterations), log_ps[chain])

    # Check convergence and continue sampling if not converged

    GR = Gelman_Rubin(sampled_params)
    print('At iteration: ', total_iterations, ' GR = ', GR)
    np.savetxt(sim_name + str(total_iterations) + '.txt', GR)

    old_samples = sampled_params
    if np.any(GR > 1.8):
        starts = [sampled_params[chain][-1, :] for chain in range(nchains)]
        while not converged:
            total_iterations += niterations

            sampled_params, log_ps = run_dream(priors_list, likelihood, start=starts, niterations=niterations,
                                               nchains=nchains, multitry=False, gamma_levels=4, adapt_gamma=True,
                                               history_thin=1, model_name=sim_name, verbose=True, restart=True)

            for chain in range(len(sampled_params)):
                np.save(sim_name + '_' + str(chain) + '_' + str(total_iterations),
                            sampled_params[chain])
                np.save(sim_name + '_' + str(chain) + '_' + str(total_iterations),
                            log_ps[chain])

            old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
            GR = Gelman_Rubin(old_samples)
            print('At iteration: ', total_iterations, ' GR = ', GR)
            np.savetxt(sim_name + '_' + str(total_iterations)+'.txt', GR)

            if np.all(GR < 1.2):
                converged = True

    try:
        # Plot output
        import seaborn as sns
        from matplotlib import pyplot as plt

        total_iterations = len(old_samples[0])
        burnin = int(total_iterations / 2)
        samples = np.concatenate((old_samples[0][burnin:, :], old_samples[1][burnin:, :], old_samples[2][burnin:, :],  old_samples[3][burnin:, :], old_samples[4][burnin:, :]))

        ndims = len(old_samples[0][0])
        colors = sns.color_palette(n_colors=ndims)
        for dim in range(ndims):
            fig = plt.figure()
            sns.distplot(samples[:, dim], color=colors[dim])
            fig.savefig('PyDREAM_example_Robertson_dimension_' + str(dim))

    except ImportError:
        pass

else:
    run_kwargs = {'parameters':pysb_sampled_parameter_names, 'likelihood':likelihood, 'niterations':10000, 'nchains':nchains, 'multitry':False, 'gamma_levels':4, 'adapt_gamma':True, 'history_thin':1, 'model_name':sim_name, 'verbose':True}