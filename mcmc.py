# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 08:31:37 2018

@author: Duncan

MCMC Implementation Using Metropolis Hastings algorithm
"""
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
import matplotlib.pyplot as plt
import seaborn as sns; 
sns.set(color_codes=True)
sns.set_style("darkgrid")
import pandas as pd

from pysb.export import export
from multiprocessing import Process, Queue, JoinableQueue, cpu_count
import itertools
import time

debugging = True # global value for this script

def get_prior_logp(kpa, kSOCSon, kd4, k_d4, R1, R2):
    # lognorm(std dev = 1, 0, guess at reaction rate value )
    #         
    P_kpa = np.log(1E-6)
    S_kpa = 2
    
    P_kSOCSon = np.log(1E-6)
    S_kSOCSon = 2
    
    P_kd4 = np.log(0.3)
    S_kd4 = 2
    
    P_k_d4 = np.log(0.006)
    S_k_d4 = 3
    
    P_R1 = np.log(R1) # Easy way to choose non-informative prior
    S_R1 = 1
    P_R2 = np.log(R2) # Easy way to choose non-informative prior
    S_R2 = 1
    
    theta=[kpa, kSOCSon, kd4, k_d4, R1, R2]
    P_list = [P_kpa,P_kSOCSon,P_kd4,P_k_d4,P_R1,P_R2]
    S_list = [S_kpa,S_kSOCSon,S_kd4,S_k_d4,S_R1,S_R2]
    logp = 0
    for i in range(len(theta)):
        logp += ((np.log(theta[i])-P_list[i])/S_list[i])**2
    return logp

def get_likelihood_logp(kpa,kSOCSon,kd4,k_d4,R1,R2, gamma):
    q1 = 3.321155762205247e-14/1
    q2 = 4.98173364330787e-13/0.015
    q4 = 3.623188E-4/kd4
    q3 = q2*q4/q1
    kd3 = 3.623188E-4/q3                

    q_1 = 4.98E-14/0.03
    q_2 = 8.30e-13/0.002
    q_4 = 3.623188e-4/k_d4
    q_3 = q_2*q_4/q_1
    k_d3 = 3.623188e-4/q_3
    fit_list = [['kpa',kpa],['kSOCSon',kSOCSon],['kd4',kd4],['k_d4',k_d4],
                ['R1',R1],['R2',R2],['kd3',kd3],['k_d3',k_d3]]
    
    import ODE_system_alpha
    alpha_mod = ODE_system_alpha.Model()
    import ODE_system_beta
    beta_mod = ODE_system_beta.Model()
   
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
    
    NA = 6.022E23
    volEC = 1E-5   
    t=[0,5*60,15*60,30*60,60*60]
    all_sims=[]         
    # fit this model to the data, finding the best values of kd4 and k_d4 if called for
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
        logp += np.sum(np.square(np.divide(np.subtract(all_sims[i],np.divide(IFN_exps[i],gamma)),np.divide(IFN_sigmas[i],gamma))))
    return logp
# =============================================================================
# Returns -log(probability of model)
# =============================================================================
def score_model(kpa,kSOCSon,kd4,k_d4,R1,R2, gamma, beta):
    lk = get_likelihood_logp(kpa,kSOCSon,kd4,k_d4,R1,R2, gamma)
    pr = get_prior_logp(kpa, kSOCSon, kd4, k_d4, R1, R2)
    return (lk+pr)/beta

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
        elif parameter[3]=='linear': # normal random walk restricted to be greater than cutoff
            new_theta.append([parameter[0],
                              max(np.random.normal(loc=parameter[1], scale=parameter[2]),parameter[4]),
                              parameter[2],parameter[3],parameter[4]])
        elif parameter[3]=='uniform': # restricted uniform distributed random walk
            new_theta.append([parameter[0],
                              np.random.uniform(low=parameter[2], high=min(parameter[1]*1.4,parameter[4])),
                              parameter[2],parameter[3],parameter[4]])
    return new_theta

# =============================================================================
# Performs 50 samples to estimate the acceptance rate with current hyperparameters
# Returns the acceptance rate as a percentage (eg. 24) and the theta with 
# variances that were good enough    
# =============================================================================
def get_acceptance_rate(theta, beta):
    old_theta=theta
    old_score = score_model(*[old_theta[j][1] for j in range(len(old_theta))], beta)
    acceptance = 0    
    for i in range(50):
        proposal = J(old_theta)
        new_score = score_model(*[proposal[j][1] for j in range(len(proposal))], beta)
        if np.random.rand() < np.exp(-(new_score-old_score)):
        # if rand() < probability of proposed/probability of old
            old_theta=proposal
            old_score = new_score
            acceptance += 1
    return (acceptance*2, old_theta) # = acceptance/50*100
            
def hyperparameter_fitting(n, theta_0, beta, max_attempts):
    n=int(0.1*n)
    if n<60: n=60
    theta = [el for el in theta_0]
    # Try to find variances that give an good acceptance rate
    for attempt in range(max_attempts):
        print("Attempt {}".format(attempt+1))
        acceptance, new_theta = get_acceptance_rate(theta, beta)
        
        if acceptance > 20 and acceptance < 50:
            print("Acceptance rate was {}%".format(acceptance))
            print("Initial parameter vector will be:")
            print(new_theta)
            return new_theta
        else:
            if acceptance < 20:
                print("Acceptance rate was too low")
                for parameter in range(len(theta)):
                    if theta[parameter][3] != 'uniform':
                        theta[parameter][2] = theta[parameter][2]/2
                        noise = np.random.normal(loc=0,scale=theta[parameter][2]*2) #intentionally *2
                        if theta[parameter][2]+noise > 0: theta[parameter][2] += noise
            if acceptance > 50:
                print("Acceptance rate was too high")
                for parameter in range(len(theta)):
                    if theta[parameter][3] != 'uniform':
                        theta[parameter][2] = theta[parameter][2]*2
                        noise = np.random.normal(loc=0,scale=theta[parameter][2]*2) #also intentionally *2
                        if theta[parameter][2]+noise > 0: theta[parameter][2] += noise

# =============================================================================
#             if attempt==max_attempts-1:
#                 for i in range(4):
#                     fig, ax = plt.subplots()
#                     ax.set(xscale='linear',yscale='log')
#                     ax.plot(range(len(new_theta)),[el[i][1] for el in new_theta]) 
#                 for i in range(3):
#                     fig, ax = plt.subplots()
#                     ax.set(xscale='linear',yscale='linear')
#                     ax.plot(range(len(new_theta)),[el[i+4][1] for el in new_theta]) 
# =============================================================================
    if debugging==True:
        return theta_0                
    raise RuntimeError("Failed to optimize hyperparameters.\n\
                       Please initialise with different variances\n\
                       or check uniform prior ranges, and try again.")

def plot_parameter_distributions(df, title='', save=True):
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
                if abs(int(np.log10(np.max(col)))-int(np.log10(np.min(col)))) >= 4:
                    sns.distplot(col, ax=ax, hist=False, kde=True, 
                         color = color_code, 
                         hist_kws={'edgecolor':'black','log':True},
                         kde_kws={'linewidth': 4})
                else:
                    sns.distplot(col, ax=ax, hist=False, kde=True, 
                         color = color_code, 
                         hist_kws={'edgecolor':'black'},
                         kde_kws={'linewidth': 4})
        fig.tight_layout() 
        if save==True:
            if title=='':
                plt.savefig('parameter_distributions.pdf')
            else:
                plt.savefig(title+'.pdf')
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
                sns.distplot(col, ax=ax, hist=True, kde=True, 
                     color = 'darkblue', 
                     hist_kws={'edgecolor':'black','log':True},
                     kde_kws={'linewidth': 4})
            else:
                sns.distplot(col, ax=ax, hist=True, kde=True, 
                     color = 'darkblue', 
                     hist_kws={'edgecolor':'black'},
                     kde_kws={'linewidth': 4})
        fig.tight_layout() 
        if save==True:
            if title=='':
                plt.savefig('parameter_distributions.pdf')
            else:
                plt.savefig(title+'.pdf')
        return (fig, axes)

def get_parameter_distributions(pooled_results, burn_rate, down_sample):
    sns.palplot(sns.color_palette("GnBu_d"))
    chain_record=[]
    first=True
    for chain in range(len(pooled_results)): #iterate over the chains
        model_record = pooled_results[chain] #current chain of interest
        # Build dataframe
        #   Account for burn in and down sampling
        sample_record = model_record[int(len(model_record)*burn_rate):-1:down_sample]
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
    # Save combined chains dataframe
    combined_samples.to_csv("posterior_samples.csv")    
    print("Effectively sampled {} times from posterior distribution".format(len(combined_samples))) 
    # Return the downsampled data frame
    return combined_samples

# =============================================================================
# plot_parameter_aurocorrelations() plots the aurocorrelation of each parameter
# from a given chain, to check that downsampling was sufficient to create
# independent samples from the posterior
# Inputs:
#   df (DataFrame) = the downsampled draws from posterior, typically the object 
#       returned from get_parameter_distributions()
# Outputs:
#   chain_autocorrelation.pdf (file) = the plots for the input chain, saved as a pdf
# Returns: nothing        
# =============================================================================
def plot_parameter_aurocorrelations(df):
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
        ax.acorr(col)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
    fig.tight_layout()    
    plt.savefig('chain_autocorrelation.pdf')
   
# =============================================================================
# get_summary_statistics() computes the summary statistics for each parameter 
# from a chain, and writes this summary to a csv file titled parameter_summary.csv
# Inputs:
#   df (DataFrame) = the downsampled draws from posterior, typically the object 
#       returned from get_parameter_distributions()
# Returns: nothing
# =============================================================================
def get_summary_statistics(df):
    headers = ['name', 'mean', 'std dev', '5%', '95%']
    summary=[]
    for (name, col) in df.iteritems():
        summary.append([name, np.mean(col), np.std(col), np.percentile(col, 5), np.percentile(col, 95)])
    summary_df = pd.DataFrame.from_records(summary,columns=headers)
    summary_df.to_csv("parameter_summary.csv")

# =============================================================================
# Sanity checks for MCMC()    
# =============================================================================
def mcmcChecks(n, theta_0, beta, chains, burn_rate, down_sample, max_attempts):
    if burn_rate>1:
        raise ValueError('Burn rate should be in the range [0,1)')
    if down_sample > n:
        raise ValueError('Cannot thin more than there are samples')
    if type(theta_0[0][0])!=str or type(theta_0[0][3])!=str:
        raise ValueError('Order of inputs for theta_0 seems incorrect')
    if type(chains) != int:
        raise ValueError('Type for input "chains" must be int')
    return True

# =============================================================================
# MCMC() takes an IFN model and fits it using Markov Chain Monte Carlo
# Inputs:    
#    n (int) = number of iterations to run per chain
#    theta_0 (list) = the initial guesses and jumping distribution definitions for each parameter to fit
#                       Order of theta_0 is [kpa, kSOCSon, kd4, k_d4, R1, R2, gamma]    
#                   eg. [['kpa',1E-6,0.2,'log'],['R2',2E3,250,'linear',100],['gamma',4,2,'uniform',40]]
#    beta (float) = effectively temperature, this factor controls the 
#                   tolerance of the probabilistic parameter search
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
def mh(ID, jobs, result):
    while True:
        mGet = jobs.get()
        if mGet is None:
            break
        hyper_theta, beta, n = mGet
        model_record=[hyper_theta]
        old_score = score_model(*[model_record[0][j][1] for j in range(len(model_record[0]))], beta)
        old_index = 0
        acceptance = 0
        # Metropolis-Hastings algorithm
        progress_bar = n/10
        for i in range(n):
            # Monitor acceptance rate            
            if i>progress_bar:
                progress_bar += n/10
                print("{:.1f}% done".format(i/n*100))
                print("Chain {} Acceptance rate = {:.1f}%".format(ID, acceptance/progress_bar*100))
            proposal = J(model_record[old_index])
            new_score = score_model(*[proposal[j][1] for j in range(len(proposal))], beta)
            if new_score < old_score or np.random.rand() < np.exp(-(new_score-old_score)):
                model_record.append(proposal)
                old_score = new_score
                old_index += 1
                acceptance += 1
        result.put(model_record)

def MCMC(n, theta_0, beta, chains, burn_rate=0.1, down_sample=1, max_attempts=6, pflag=True):
    # Check input parameters
    mcmcChecks(n, theta_0, beta, chains, burn_rate, down_sample, max_attempts)
    # Selecting hyperparameters
    #print("Optimizing hyperparameters")
    #hyper_theta = hyperparameter_fitting(n, theta_0, beta, max_attempts)
    #check_priors(hyper_theta, 50)
    hyper_theta=theta_0
    print("Sampling from posterior distribution")    
    if chains >= cpu_count():
        NUMBER_OF_PROCESSES = cpu_count()-1
    else:
        NUMBER_OF_PROCESSES = chains
    jobs = Queue()
    result = JoinableQueue()
    for m in range(chains):
        jobs.put([hyper_theta,beta,n])
    [Process(target=mh, args=(i, jobs, result)).start()
            for i in range(NUMBER_OF_PROCESSES)]
    # pull in the results from each thread
    pool_results=[]
    for m in range(chains):
        r = result.get()
        pool_results.append(r)
        result.task_done()
    # tell the workers there are no more jobs
    for w in range(NUMBER_OF_PROCESSES):
        jobs.put(None)
    # close all extra threads
    result.join()
    jobs.close()
    result.close()

    # Perform data analysis
    samples = get_parameter_distributions(pool_results, burn_rate, down_sample)
    plot_parameter_aurocorrelations(samples)
    get_summary_statistics(samples)
    
    
# =============================================================================
# check_priors() allows the user to get a sense of what the random walk
# for each parameter looks like with the given values
# Inputs:
#   theta (list) = the input parameter vector
#   n (int) = the number of steps to take in the sample random walk
# =============================================================================
def check_priors(theta, n):
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
    
    
def main():
    plt.close('all')
    modelfiles = ['IFN_alpha_altSOCS_ppCompatible','IFN_beta_altSOCS_ppCompatible']
# Write modelfiles
    print("Importing models")
    alpha_model = __import__(modelfiles[0])
    py_output = export(alpha_model.model, 'python')
    with open('ODE_system_alpha.py','w') as f:
        f.write(py_output)
    beta_model = __import__(modelfiles[1])
    py_output = export(beta_model.model, 'python')
    with open('ODE_system_beta.py','w') as f:
        f.write(py_output)
# =============================================================================
#     t0 = time.time()
#     t1 = time.time()
#     print("time = {}".format(t1-t0))
# =============================================================================
    print("Performing MCMC Analysis")
    p0=[['kpa',1E-6,0.1,'log'],['kSOCSon',1E-6,0.1,'log'],['kd4',0.3,0.2,'log'],
        ['k_d4',0.006,0.5,'log'],['R1',2E3,250,'linear',100],['R2',2E3,250,'linear',100],
        ['gamma',4,2,'uniform',40]]
# =============================================================================
#     p1=[['kpa', 1e-06, 0.0007, 'log'],['kSOCSon', 1.e-06, 0.002, 'log'],
#     ['kd4', 0.3, 0.0001, 'log'],['k_d4', 0.006, 0.033, 'log'],
#     ['R1', 5224, 682, 'linear', 100],['R2', 2000., 21, 'linear', 100],
#     ['gamma', 2.78, 2, 'uniform', 40]]    
# =============================================================================
    MCMC(100, p0, 75, 2, burn_rate=0.1, down_sample=1)# n, theta, beta
    
    #testChain = pd.read_csv('test_posterior_samples.csv',index_col=0)

if __name__ == '__main__':
    main()
