# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 08:31:37 2018

@author: Duncan

MCMC Implementation Using Metropolis Hastings algorithm, for INF alpha and IFN
beta model. This code is not generalizeable at the moment. I intend to eventually
do this, but for now it is pretty much only useful to me.
"""
import os
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'MCMC_Results/')
chain_results_dir = results_dir+'Chain_Results/'


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
if not os.path.isdir(chain_results_dir):
    os.makedirs(chain_results_dir)
    
import Experimental_Data as ED
# Global data import since this script will be used exclusively on IFN data    
IFN_exps = [ED.data.loc[(ED.data.loc[:,'Dose (pM)']==10) & (ED.data.loc[:,'Interferon']=="Alpha"),['0','5','15','30','60']].values[0],
            ED.data.loc[(ED.data.loc[:,'Dose (pM)']==10) & (ED.data.loc[:,'Interferon']=="Beta"),['0','5','15','30','60']].values[0],
            ED.data.loc[(ED.data.loc[:,'Dose (pM)']==90) & (ED.data.loc[:,'Interferon']=="Alpha"),['0','5','15','30','60']].values[0],
            ED.data.loc[(ED.data.loc[:,'Dose (pM)']==90) & (ED.data.loc[:,'Interferon']=="Beta"),['0','5','15','30','60']].values[0],
            ED.data.loc[(ED.data.loc[:,'Dose (pM)']==600) & (ED.data.loc[:,'Interferon']=="Alpha"),['0','5','15','30','60']].values[0],
            ED.data.loc[(ED.data.loc[:,'Dose (pM)']==600) & (ED.data.loc[:,'Interferon']=="Beta"),['0','5','15','30','60']].values[0]]

IFN_sigmas =[ED.data.loc[(ED.data.loc[:,'Dose (pM)']==10) & (ED.data.loc[:,'Interferon']=="Alpha_std"),['0','5','15','30','60']].values[0],
             ED.data.loc[(ED.data.loc[:,'Dose (pM)']==10) & (ED.data.loc[:,'Interferon']=="Beta_std"),['0','5','15','30','60']].values[0],
             ED.data.loc[(ED.data.loc[:,'Dose (pM)']==90) & (ED.data.loc[:,'Interferon']=="Alpha_std"),['0','5','15','30','60']].values[0],
             ED.data.loc[(ED.data.loc[:,'Dose (pM)']==90) & (ED.data.loc[:,'Interferon']=="Beta_std"),['0','5','15','30','60']].values[0],
             ED.data.loc[(ED.data.loc[:,'Dose (pM)']==600) & (ED.data.loc[:,'Interferon']=="Alpha_std"),['0','5','15','30','60']].values[0],
             ED.data.loc[(ED.data.loc[:,'Dose (pM)']==600) & (ED.data.loc[:,'Interferon']=="Beta_std"),['0','5','15','30','60']].values[0]]


import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns; 
sns.set(color_codes=True)
sns.set_style("darkgrid")
import pandas as pd

from pysb.export import export
from multiprocessing import Process, Queue, JoinableQueue, cpu_count
import itertools
import time

# =============================================================================
# Takes an initial condition and generates nChains randomized initial conditions
#   from prior distributions. This is not generalized, it is designed
#   specifically for IFN alpha and IFN beta model.
# Inputs:
#   theta_0 (list) = the initial parameter vector, defining priors to draw from
#  priors_dict={'variable_name':[minval,maxval,logmean,logstd]}
#   nChains (int) = the number of unique points to start from (ie. number of chains)
# Returns:
#   theta_list (list) = list of parameter vectors    
# =============================================================================
def disperse_chains(theta_0, priors_dict, nChains):
    theta_list=[]
    for j in range(nChains):
        new_theta=[]
        for parameter in theta_0:
            if parameter[0] not in priors_dict.keys():#Can't disperse variables without priors
                new_theta.append(parameter)
            elif priors_dict[parameter[0]][2]!=None:
                new_theta.append([parameter[0],
                                  np.random.lognormal(mean=priors_dict[parameter[0]][2], 
                                  sigma=priors_dict[parameter[0]][3]),
                                  parameter[2],parameter[3]])
            else:
                new_theta.append([parameter[0],
                                  np.random.uniform(low=priors_dict[parameter[0]][0], 
                                                    high=priors_dict[parameter[0]][1]),
                                  parameter[2],parameter[3]])   
        theta_list.append(new_theta)
    return theta_list


# =============================================================================
# get_prior_logp() scores the variables against the prior distribution given    
#  variables = [['name',value],['name',value]....]
#  priors_dict={'variable_name':[minval,maxval,logmean,logstd]}
# =============================================================================
def get_prior_logp(variables, priors_dict):
    # Check bounds on parameters
    for variable in variables:
        # Make an exception for variables not in priors_dict (usually kd3, k_d3)
            name=variable[0]
            val=variable[1]
            if name in priors_dict.keys():  
                if val<priors_dict[name][0] or val>priors_dict[name][1]:
                    return 1E8
    # Otherwise calculate log probability of parameter values
    logp = 0
    for i in range(len(variables)):
        name=variables[i][0]    
        val=variables[i][1]            
        if name in priors_dict.keys() and priors_dict[name][2]!=None:
            logp += ((np.log(val)-priors_dict[name][2])/priors_dict[name][3])**2  
    return logp
    
# =============================================================================
# logp_helper() is a target function for scipy optimization of scale factor gamma
# =============================================================================
def logp_helper(gamma, alpha_parameters, beta_parameters, I_index_Alpha, I_index_Beta):
    # import models
    import ODE_system_alpha
    alpha_mod = ODE_system_alpha.Model()
    import ODE_system_beta
    beta_mod = ODE_system_beta.Model()
    # set constants
    NA = 6.022E23
    volEC = 1E-5   
    t=[0,5*60,15*60,30*60,60*60]
    all_sims=[]         
    # run simulations under experimental conditions
    alpha_parameters[I_index_Alpha] = NA*volEC*10E-12
    (_, sim) = alpha_mod.simulate(t, param_values=alpha_parameters)
    all_sims.append(sim['TotalpSTAT'])
    beta_parameters[I_index_Beta] = NA*volEC*10E-12
    (_, sim) = beta_mod.simulate(t, param_values=beta_parameters)
    all_sims.append(sim['TotalpSTAT'])
    
    alpha_parameters[I_index_Alpha] = NA*volEC*90E-12
    (_, sim) = alpha_mod.simulate(t, param_values=alpha_parameters)
    all_sims.append(sim['TotalpSTAT'])
    beta_parameters[I_index_Beta] = NA*volEC*90E-12
    (_, sim) = beta_mod.simulate(t, param_values=beta_parameters)
    all_sims.append(sim['TotalpSTAT'])

    alpha_parameters[I_index_Alpha] = NA*volEC*600E-12
    (_, sim) = alpha_mod.simulate(t, param_values=alpha_parameters)
    all_sims.append(sim['TotalpSTAT'])
    beta_parameters[I_index_Beta] = NA*volEC*600E-12
    (_, sim) = beta_mod.simulate(t, param_values=beta_parameters)
    all_sims.append(sim['TotalpSTAT'])
    logp = 0
    for i in range(len(all_sims)):
        logp += np.sum(np.square(np.divide(np.subtract(np.multiply(gamma,all_sims[i]),IFN_exps[i]),IFN_sigmas[i])))
    return logp
# =============================================================================
# get_likelihood_logp() is designed specifically for IFN alpha and IFN beta model 
#   least-squares likelihood with data used in paper
# =============================================================================
def get_likelihood_logp(fit_list):
    # import models
    import ODE_system_alpha
    alpha_mod = ODE_system_alpha.Model()
    import ODE_system_beta
    beta_mod = ODE_system_beta.Model()
    # Build parameter lists
    if 'kd4' in [el[0] for el in fit_list]:
        kd4Index=[el[0] for el in fit_list].index('kd4')
        q1 = 3.321155762205247e-14/1
        q2 = 4.98173364330787e-13/0.015
        q4 = 3.623188E-4/fit_list[kd4Index][1]
        q3 = q2*q4/q1
        kd3 = 3.623188E-4/q3 
        fit_list.insert(kd4Index+1,['kd3',kd3])               
    if 'k_d4' in [el[0] for el in fit_list]:
        k_d4Index = [el[0] for el in fit_list].index('k_d4')
        q_1 = 4.98E-14/0.03
        q_2 = 8.30e-13/0.002
        q_4 = 3.623188e-4/fit_list[k_d4Index][1]
        q_3 = q_2*q_4/q_1
        k_d3 = 3.623188e-4/q_3
        fit_list.insert(k_d4Index+1,['k_d3',k_d3])               
    alpha_parameters=[]
    beta_parameters=[]
    for p in alpha_mod.parameters:
        isInList=False
        for y in fit_list:
            if p[0]==y[0]:
                alpha_parameters.append(y[1])
                isInList=True
                break
        if isInList==False:
            alpha_parameters.append(p.value)
    for p in beta_mod.parameters:
        isInList=False
        for y in fit_list:
            if p[0]==y[0]:
                beta_parameters.append(y[1])
                isInList=True
                break
        if isInList==False:
            beta_parameters.append(p.value)
    I_index_Alpha = [el[0] for el in alpha_mod.parameters].index('I')
    I_index_Beta = [el[0] for el in beta_mod.parameters].index('I')
    # optimize choice of scale factor gamma:
    #   set initial guess for gamma at 40 to ensure we get the largest local minimum in the optimization
    opt = minimize(logp_helper,[40],args=(alpha_parameters, beta_parameters, I_index_Alpha, I_index_Beta))
    gamma = opt['x'].item()
    logp = opt['fun']
    return [logp, gamma]
    
# =============================================================================
# variables (list) = list of variables in the form [['name',value],...]
# priors_dict (dict) = dictionary defining priors for variables    
# beta, rho (floats) = values of beta and rho used in simulation
# Returns -log(probability of model)
# =============================================================================
def score_model(variables, priors_dict, beta, rho, debugging=False):
    meanR=2E3
    delR=0
    remove_meanR=False
    remove_delR=False
    for el in variables:
        if el[0]=='meanR':
            meanR=el[1]
            meanRIndex=[el[0] for el in variables].index('meanR')
            remove_meanR=True            
        elif el[0]=='delR':
            delR=el[1]
            delRIndex=[el[0] for el in variables].index('delR')
            remove_delR=True
    R1=meanR-delR/2
    R2=meanR+delR/2
    if remove_meanR==True and remove_delR==True:
        del variables[min(meanRIndex,delRIndex)]
        del variables[max(meanRIndex,delRIndex)-1]
        variables.insert(meanRIndex,['R1',R1])        
        variables.insert(meanRIndex,['R2',R2])
    elif remove_meanR==True:
        del variables[meanRIndex]
        variables.insert(meanRIndex,['R2',R2])
        variables.insert(meanRIndex,['R1',R1])
    elif remove_delR==True:
        del variables[delRIndex]
        variables.insert(delRIndex,['R2',R2])
        variables.insert(delRIndex,['R1',R1])
        
    [lk,gamma] = get_likelihood_logp(variables)
    pr = get_prior_logp(variables, priors_dict)
    if debugging==True:
        print(str(lk)+", "+str(pr))
    return [(lk/rho+pr)/beta, gamma]

# =============================================================================
# J() is the jumping distribution
# Inputs:
#   theta (list) = elements of the form ['name', current value, std dev for jumps,'distribution']
# Jumping distributions:
#   log - used for reaction rates; lognormal distributed jump lengths
#   linear - used for concentrations/species counts; normally distributed jump lengths    
#   uniform - used for data scale factor, for example; restricted uniform distributed jump lengths
# Returns:
#   new_theta (list) = same form as theta but with new values stored in second position of subarray    
# =============================================================================
def J(theta):
    new_theta=[]
    for parameter in theta:
        if parameter[3]=='log': # lognormal random walk
            new_theta.append([parameter[0],
                              np.random.lognormal(mean=np.log(parameter[1]), sigma=parameter[2]),
                              parameter[2],parameter[3]])
        elif parameter[3]=='linear': # normal random walk (restricted by prior to avoid unphysical values)
            new_theta.append([parameter[0],
                              np.random.normal(loc=parameter[1], scale=parameter[2]),
                              parameter[2],parameter[3]])
        elif parameter[3]=='uniform': # restricted uniform distributed random walk
            new_theta.append([parameter[0],
                              np.random.uniform(low=parameter[2], high=min(parameter[1]*1.4,parameter[4])),
                              parameter[2],parameter[3],parameter[4]])
    return new_theta

# =============================================================================
# get_acceptance_rate() performs 100 samples to estimate the acceptance rate with 
# current hyperparameters. Returns the acceptance rate as a percentage (eg. 24)  
# and the theta with variances that were good enough    
# =============================================================================
def get_acceptance_rate(theta, priors_dict, beta, rho):
    old_theta=theta
    [old_score, old_gamma] = score_model([[old_theta[j][0],old_theta[j][1]] for j in range(len(old_theta))], priors_dict, beta, rho)
    asymmetric_indices = [el[0] for el in enumerate(old_theta) if el[1][3]=='log']
    acceptance = 0    
    for i in range(100):
        proposal = J(old_theta)
        [new_score, new_gamma] = score_model([[proposal[j][0],proposal[j][1]] for j in range(len(proposal))], priors_dict, beta, rho)
        asymmetry_factor = 1 # log normal proposal distributions are asymmetric
        for j in asymmetric_indices:
            asymmetry_factor *= proposal[j][1]/old_theta[j][1]
        if new_score < old_score or np.random.rand() < np.exp(-(new_score-old_score))*asymmetry_factor:
        # if rand() < probability of proposed/probability of old
            old_theta=proposal
            old_score = new_score
            old_gamma = new_gamma
            acceptance += 1
    return (acceptance, old_theta, old_gamma) # = acceptance/100*100

# =============================================================================
# hyperparameter_fitting() attempts to alter the input temperature
#   to achieve a good acceptance rate during simulation
# Inputs:
#   theta_0 and beta - see MCMC() documentation
#   max_attempts (int) = the max number of attempts to get a good acceptance rate
# =============================================================================
def hyperparameter_fitting(theta_0, priors_dict, beta, rho, max_attempts):
    print("Choosing optimal temperature")
    theta = [el for el in theta_0]
    if max_attempts==0:
        return (theta_0, beta)
    # Try to find variances that give an good acceptance rate
    for attempt in range(max_attempts):
        print("Attempt {}".format(attempt+1))
        acceptance, new_theta, new_gamma = get_acceptance_rate(theta, priors_dict, beta, rho)
        
        if acceptance >= 15 and acceptance <= 50:
            print("Acceptance rate was {}%".format(acceptance))
            print("New temperature will be: "+str(beta))
            return (new_theta, beta)
        else:
            if acceptance < 15:
                print("Acceptance rate was too low: {}%".format(acceptance))
                beta = 2*beta
            if acceptance > 50:
                print("Acceptance rate was too high: {}%".format(acceptance))
                beta = 0.75*beta
    raise RuntimeError("         Failed to optimize hyperparameters.\n\
                       Please initialise with different variances or temperatures,\n\
                       or check uniform prior ranges, and try again.")
    
# =============================================================================
# plot_parameter_distributions() creates a kde plot for each parameter
# Inputs:
#     df (DataFrame or list of DataFrames) = the chain or chains that were simulated
#     title (string) = default is 'parameter_distributions.pdf'
#     save (Boolean) = whether or not to save the plot (default is True)
# =============================================================================
def plot_parameter_distributions(df, title='parameter_distributions.pdf', save=True):
    # different plot asthetics if multiple chains vs one chain
    if type(df)==list:
        k = len(df[0].columns) # total number subplots
        n = 2 # number of chart columns
        m = (k - 1) // n + 1 # number of chart rows
        fig, axes = plt.subplots(m, n, figsize=(n * 5, m * 3))
        if k % 2 == 1: # avoids extra empty subplot
            axes[-1][n-1].set_axis_off()
        palette = itertools.cycle(sns.color_palette("GnBu_d", 10)) # make chain colours nice
        for j in range(len(df)):
            for i, (name, col) in enumerate(df[j].iteritems()):
                color_code = next(palette)
                r, c = i // n, i % n
                ax = axes[r, c] # get axis object
                # determine whether or not to plot on log axis
                try:
                    if abs(int(np.log10(np.max(col)))-int(np.log10(np.min(col)))) >= 4:
                        ax.set(xscale='log', yscale='linear')
                except ValueError or OverflowError:
                    print('Some parameters were negative-valued or infinity')
                # Plot histogram with kde for chain
                sns.distplot(col, ax=ax, hist=False, kde=True, 
                     color = color_code, 
                     kde_kws={'linewidth': 4})
        fig.tight_layout() 
        if save==True:
            if title=='':
                plt.savefig(results_dir+'parameter_distributions.pdf')
            else:
                plt.savefig(results_dir+title+'.pdf')
        return [fig, axes]

    else:
        k = len(df.columns) # total number subplots
        n = 2 # number of chart columns
        m = (k - 1) // n + 1 # number of chart rows
        fig, axes = plt.subplots(m, n, figsize=(n * 5, m * 3))
        if k % 2 == 1: # avoids extra empty subplot
            axes[-1][n-1].set_axis_off()
        for i, (name, col) in enumerate(df.iteritems()):
            r, c = i // n, i % n
            ax = axes[r, c] # get axis object
            # determine whether or not to plot on log axis
            if abs(int(np.log10(np.max(col)))-int(np.log10(np.min(col)))) >= 4:
                ax.set(xscale='log', yscale='linear')
            # Plot histogram with kde for chain
            sns.distplot(col, ax=ax, hist=True, kde=True, 
                 color = 'darkblue', 
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 4})
        fig.tight_layout() 
        if save==True:
            if title=='':
                plt.savefig(results_dir+'parameter_distributions.pdf')
            else:
                plt.savefig(results_dir+title+'.pdf')
        return (fig, axes)
# =============================================================================
# Computes the Gelman-Rubin statistic for each parameter to test for convergence
# Inputs: chain_record (list of Dataframes) = list of Markov chains
# Outputs: Gelman-Rubin_Statistics.csv file summarizing the test results
# Returns: stats_list (list) = summary of test results, [['variable name', Rhat], ...]        
# =============================================================================
def gelman_rubin_convergence(chain_record):
    numChains = len(chain_record)
    column_names = list(chain_record[0].columns.values)
    stats_list = []
    n = min([chain_record[i].shape[0] for i in range(len(chain_record))])
    for variable in column_names:
        chain_mean = [chain_record[i][variable].mean() for i in range(numChains)]
        W = np.mean([(chain_record[i][variable].std())**2 for i in range(numChains)])
        B = chain_record[0].shape[0]*np.std(chain_mean, ddof=1)**2
        Var = (1-1/n)*W+B/n
        Rhat = np.sqrt(Var/W)
        stats_list.append([variable, Rhat])
    df = pd.DataFrame.from_records(stats_list, columns=['variable','GR Statistic'])
    df.to_csv(results_dir+'Gelman-Rubin_Statistics.csv')
    return stats_list

# =============================================================================
# get_parameter_distributions() takes a chain or list of chains and performs
# data formatting such as burn in, thinning, kde plots for each parameter,
# and statistical summaries
# Inputs:
#     pooled_results (list) = list of Markov chains from MCMC
#     burn_rate (float in [0,1)) = the fraction of each chain to discard as burn in
#     down_sample (int) = the stride length for using draws from the chain
# Outputs:
#     kde plot of each parameter's posterior distribution, saved as parameter_distributions.pdf
#     posterior_samples.csv = a file containing all the independent posterior samples
# Returns:
#     the combined chains' posterior samples
# =============================================================================
def get_parameter_distributions(pooled_results, burn_rate, down_sample):
    sns.palplot(sns.color_palette("GnBu_d"))
    chain_record=[] # for plotting chain histograms
    total_record=[] # to save all results to allow reanalysis
    chain_Lengths=[] # list of ints, each a chain's length; used in reanalysis
    GR_record = [] # record of chains after burn-in but before thinning, used in GR diagnostic
    for i in pooled_results:
        total_record+=i
        chain_Lengths.append(len(i))
    complete_samples = pd.DataFrame([[el[1] for el in r] for r in total_record],
                                columns=[l[0] for l in total_record[0]])
    complete_samples.to_csv(results_dir+"complete_samples.csv")
    with open(results_dir+'chain_lengths.txt', 'w') as f:
        f.write(str(chain_Lengths)[1:-1])
    first=True
    for chain in range(len(pooled_results)): #iterate over the chains
        model_record = pooled_results[chain] #current chain of interest
        # Build dataframe
        #   Account for burn in and down sampling
        sample_record = model_record[int(len(model_record)*burn_rate):-1:down_sample]
        GR_list = model_record[int(len(model_record)*burn_rate):-1]

        GR_record.append(pd.DataFrame([[el[1] for el in r] for r in GR_list],
                                columns=[l[0] for l in GR_list[0]]))
        if first==True:
            combined_samples = pd.DataFrame([[el[1] for el in r] for r in sample_record],
                                columns=[l[0] for l in sample_record[0]])
            first=False
        else: # otherwise add to existing DataFrame
            combined_samples.append(pd.DataFrame([[el[1] for el in r] for r in sample_record],
                                columns=[l[0] for l in sample_record[0]]))
        if len(pooled_results)==1: #If there was only one chain, pass DataFrame to plot function
            # Plot parameter distributions
            plot_parameter_distributions(pd.DataFrame([[el[1] for el in r] for r in sample_record],
                                columns=[l[0] for l in sample_record[0]]))
        else: # add DataFrame to list of prepared chains to plot later
            chain_record.append(pd.DataFrame([[el[1] for el in r] for r in sample_record],
                                columns=[l[0] for l in sample_record[0]]))
    # Plot multiple chains on same axes if there were multiple chains
    if len(pooled_results)>1:
            plot_parameter_distributions(chain_record)
            gelman_rubin_convergence(GR_record) #Pretty sure I should use full chains for GR-diagnostic
            
    # Save combined chains dataframe
    combined_samples.to_csv(results_dir+"posterior_samples.csv")    
    print("After thinning and burn-in, you sampled {} times from the posterior distribution".format(len(combined_samples))) 
    # Return the downsampled data frame
    return combined_samples

# =============================================================================
# plot_parameter_autocorrelations() plots the autocorrelation of each parameter
# from a given chain, to check that downsampling was sufficient to create
# independent samples from the posterior
# Inputs:
#   df (DataFrame) = the downsampled draws from posterior, typically the object 
#       returned from get_parameter_distributions()
# Outputs:
#   chain_autocorrelation.pdf (file) = the plots for the input chain, saved as a pdf
# Returns: nothing        
# =============================================================================
def plot_parameter_autocorrelations(df):
    k = len(df.columns) # total number subplots
    n = 2 # number of chart columns
    m = (k - 1) // n + 1 # number of chart rows
    fig, axes = plt.subplots(m, n, figsize=(n * 5, m * 3))
    if k % 2 == 1: # avoids extra empty subplot
        axes[-1][n-1].set_axis_off()
    for i, (name, col) in enumerate(df.iteritems()):
        r, c = i // n, i % n
        ax = axes[r, c] # get axis object
        ax.set_title(name)
        corrLen=30
        if len(col)<30: corrLen=len(col)-1
        ax.acorr(col, maxlags=corrLen)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
    fig.tight_layout()    
    plt.savefig(results_dir+'chain_autocorrelation.pdf')
   
# =============================================================================
# get_summary_statistics() computes the summary statistics for each parameter 
# from a chain, and writes this summary to a csv file titled parameter_summary.csv
# Inputs:
#   df (DataFrame) = the downsampled draws from posterior, typically the object 
#       returned from get_parameter_distributions()
# Returns: nothing
# =============================================================================
def get_summary_statistics(df):
    headers = ['name', 'mean', 'std dev', '2.5%', '25%', '75%', '97.5%']
    summary=[]
    for (name, col) in df.iteritems():
        summary.append([name, np.mean(col), np.std(col), np.percentile(col, 2.5), np.percentile(col, 25), np.percentile(col, 75), np.percentile(col, 97.5)])
    summary_df = pd.DataFrame.from_records(summary,columns=headers)
    summary_df.to_csv(results_dir+"parameter_summary.csv")


# =============================================================================
# check_proposals() allows the user to get a sense of what the random walk
# for each parameter looks like with the given values
# Inputs:
#   theta (list) = the input parameter vector
#   n (int) = the number of steps to take in the sample random walk
# =============================================================================
def check_proposals(theta, n):
    walk=[theta]
    for j in range(n):
        walk.append(J(walk[j]))
    rearrange = [[walk[i][j][1] for i in range(len(walk))] for j in range(len(theta))]
    
    k = len(theta) # total number subplots
    n = 2 # number of chart columns
    m = (k - 1) // n + 1 # number of chart rows
    fig, axes = plt.subplots(m, n, figsize=(n * 5, m * 3))
    if k % 2 == 1: # avoids extra empty subplot
        axes[-1][n-1].set_axis_off()
    for ind, w in enumerate(rearrange):
        r, c = ind // n, ind % n
        ax = axes[r, c] # get axis object
        if theta[r*n+c][3]=='log':
            ax.set(xscale='linear',yscale='log')
        ax.plot(range(len(w)), w)    
        ax.set_title(theta[r*n+c][0])
    plt.savefig(results_dir+'typical_priors_rw.pdf')

# =============================================================================
# Sanity checks for MCMC()    
# =============================================================================
def mcmcChecks(n, theta_0, beta, rho, chains, burn_rate, down_sample, max_attempts):
    if burn_rate>1:
        raise ValueError('Burn rate should be in the range [0,1)')
    if down_sample > n:
        raise ValueError('Cannot thin more than there are samples')
    if type(theta_0[0][0])!=str or type(theta_0[0][3])!=str:
        raise ValueError('Order of inputs for theta_0 seems incorrect')
    if type(chains) != int:
        raise ValueError('Type for input "chains" must be int')
    if max_attempts < 0:
        raise ValueError('max_attempts must be positive')
    lenPost = int(np.floor([chains*(n+1)*(1-burn_rate)/down_sample])[0])
    print("It's estimated this simulation will produce {} posterior samples.".format(lenPost))
    return True

# =============================================================================
# MCMC() takes an IFN model and fits it using Markov Chain Monte Carlo
# Inputs:    
#    n (int) = number of samples to collect per chain
#    theta_0 (list) = the initial guesses and jumping distribution definitions for each parameter to fit
#                       Order of theta_0 is [kpa, kSOCSon, kd4, k_d4, R1, R2]    
#                   eg. [['kpa',1E-6,0.2,'log'],['R2',2E3,250,'linear'],['kSOCSon',4,2,'uniform',40]]
#    beta (float) = effectively temperature, this factor controls the 
#                   tolerance of the probabilistic parameter search
#    rho (float) = the scale factor for Bayesian cost, to make priors more important
#    chains (int) = number of unique Markov chains to simulate    
# Optional Inputs:    
#    burn_rate (float) = initial fraction of samples to discard as 'burn in'
#                       default is to discard the first 10%    
#    down_sample (int) = step size for down sampling to reduce autocorrelation
#                       default is 1 (no down sampling unless user specifies)    
#    max_attempts (int) = the number of attempts to try and choose hyperparameters
#                           default is 6    
#    pflag (Boolean) = plot a typical random walk for each parameter after hyperparameter selection    
# =============================================================================
def mh(ID, jobs, result, countQ):
    while True:
        mGet = jobs.get()
        if mGet is None:
            break
        hyper_theta, beta, rho, n, priors_dict = mGet
        model_record=[hyper_theta]
        asymmetric_indices = [el[0] for el in enumerate(hyper_theta) if el[1][3]=='log']
        [old_score, old_gamma] = score_model([[model_record[0][j][0],model_record[0][j][1]] for j in range(len(model_record[0]))], priors_dict, beta, rho)
        gamma_list=[old_gamma]
        old_index = 0
        acceptance = 0
        attempts = 0
        # Metropolis-Hastings algorithm
        progress_bar = n/10
        while acceptance<n:
            attempts+=1
            # Monitor acceptance rate            
            if acceptance>progress_bar:
                print("{:.1f}% done".format(acceptance/n*100))
                print("Chain {} acceptance rate = {:.1f}%".format(ID, acceptance/attempts*100))
                with open(results_dir+'progress.txt','a') as f:
                    f.write("Chain {} is {:.1f}% done, currently averaging {:.1f}% acceptance.\n".format(ID, acceptance/n*100,acceptance/attempts*100))
                with open(chain_results_dir+str(ID)+'chain.txt','w') as g:
                    temp_record = []
                    for i in range(len(model_record)):
                        temp_record.append(model_record[i]+[['gamma',gamma_list[i]]])
                    g.write(str(temp_record))
                progress_bar += n/10                    
            proposal = J(model_record[old_index])
            [new_score, new_gamma] = score_model([[proposal[j][0],proposal[j][1]] for j in range(len(proposal))], priors_dict, beta, rho)
            asymmetry_factor = 1 # log normal proposal distributions are asymmetric
            for j in asymmetric_indices:
                asymmetry_factor *= proposal[j][1]/model_record[old_index][j][1]
            if new_score < old_score or np.random.rand() < np.exp(-(new_score-old_score))*asymmetry_factor:
                model_record.append(proposal)
                gamma_list.append(new_gamma)                
                old_score = new_score
                old_index += 1
                acceptance += 1
        for i in range(len(model_record)):
            model_record[i].append(['gamma',gamma_list[i]])
        result.put(model_record)
        countQ.put([ID,acceptance/attempts*100])

def MCMC(n, theta_0, priors_dict, beta, rho, chains, burn_rate=0.1, down_sample=1, max_attempts=6, 
         pflag=True, cpu=None, randomize=True):
    # Check input parameters
    mcmcChecks(n, theta_0, beta, rho, chains, burn_rate, down_sample, max_attempts)
    print("Performing MCMC Analysis")
    # Selecting optimal temperature
    hyper_theta, beta = hyperparameter_fitting(theta_0, priors_dict, beta, rho, max_attempts)
    if pflag==True:
        check_proposals(hyper_theta, 50)
    # Overdisperse chains
    if randomize==True:
        print("Dispersing chains")
        if chains > 1:
            chains_list = disperse_chains(hyper_theta, priors_dict, chains)
        else:
            chains_list = [hyper_theta]
    else:
        chains_list = [hyper_theta for i in range(chains)]
    # Sample using MCMC    
    print("Sampling from posterior distribution")    
    if chains >= cpu_count():
        NUMBER_OF_PROCESSES = cpu_count()-1
    else:
        NUMBER_OF_PROCESSES = chains
    if cpu != None: NUMBER_OF_PROCESSES = cpu # Manual override of core number selection
    print("Using {} threads".format(NUMBER_OF_PROCESSES))
    with open(results_dir+'progress.txt','w') as f: # clear previous progress report
        f.write('')
    jobs = Queue() # put jobs on queue
    result = JoinableQueue()
    countQ = JoinableQueue()
    for m in range(chains):
        jobs.put([chains_list[m],beta,rho,n,priors_dict])
    [Process(target=mh, args=(i, jobs, result, countQ)).start()
            for i in range(NUMBER_OF_PROCESSES)]
    # pull in the results from each thread
    pool_results=[]
    chain_attempts=[]
    for m in range(chains):
        r = result.get()
        pool_results.append(r)
        result.task_done()
        a = countQ.get()
        chain_attempts.append(a)
    # tell the workers there are no more jobs
    for w in range(NUMBER_OF_PROCESSES):
        jobs.put(None)
    # close all extra threads
    result.join()
    jobs.close()
    result.close()
    countQ.close()

    # Perform data analysis
    average_acceptance = np.mean([el[1] for el in chain_attempts])
    print("Average acceptance rate was {:.1f}%".format(average_acceptance))
    samples = get_parameter_distributions(pool_results, burn_rate, down_sample)
    plot_parameter_autocorrelations(samples.drop('gamma',axis=1))
    get_summary_statistics(samples.drop('gamma',axis=1))
    with open(results_dir+'simulation_summary.txt','w') as f:
        f.write('Temperature used was {}\n'.format(beta))
        f.write('Number of chains = {}\n'.format(chains))
        f.write("Average acceptance rate was {:.1f}%\n".format(average_acceptance))
        f.write("Initial conditions were\n")
        for i in chains_list:
            f.write(str(i))
            f.write("\n")



def main():
    plt.close('all')
    modelfiles = ["IFN_Models.IFN_alpha_altSOCS_ppCompatible","IFN_Models.IFN_beta_altSOCS_ppCompatible"]
# Write modelfiles
    print("Importing models")
    alpha_model = __import__(modelfiles[0],fromlist=['IFN_Models'])
    py_output = export(alpha_model.model, 'python')
    with open('ODE_system_alpha.py','w') as f:
        f.write(py_output)
    beta_model = __import__(modelfiles[1],fromlist=['IFN_Models'])
    py_output = export(beta_model.model, 'python')
    with open('ODE_system_beta.py','w') as f:
        f.write(py_output)
    p0=[['kpa',1.79E-5,0.1,'log'],['kSOCSon',1.70E-6,0.1,'log'],['kd4',0.87,0.2,'log'],
        ['k_d4',0.86,0.5,'log'],['delR',-1878,500,'linear'],['meanR',2000,300,'linear']]
    our_priors_dict={'R1':[100,12000,None,None],'R2':[100,12000,None,None],
             'kpa':[1.5E-11,0.07,np.log(1E-6),4],'kSOCSon':[1.5E-11,0.07,np.log(1E-6),4],
             'k_d4':[4E-5,0.9,np.log(0.006),1.8],'kd4':[0.002,44,np.log(0.3),1.8]}
    #   (n, theta_0, beta, rho, chains, burn_rate=0.1, down_sample=1, max_attempts=6,
    #    pflag=True, cpu=None, randomize=True)
    MCMC(500, p0, our_priors_dict, 2, 1, 3, burn_rate=0.2, down_sample=30, max_attempts=6)

    
if __name__ == '__main__':
    main()

