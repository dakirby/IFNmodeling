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


import pysb_parallel as pp
import numpy as np
import re
import matplotlib.pyplot as plt
import scipy.stats
from pysb.export import export
import time

debugging = True # global value for this script

# =============================================================================
# SSres_IFN scores the sum of square residuals for IFNa and IFNb models
# Inputs:
#     mods (list) = [modAlpha,modBeta] previously imported models
#     test_parameters (list) = list of dicts, one for alpha model and one for beta model
#                             eg. {'R1':R1b, 'R2':R2b,'IFN':600E-12,'kSOCSon':kSOCSonb,'kpa':kpab, 'k_d4':k_d4}
# =============================================================================
# =============================================================================
# def SSres_IFN(mods, test_parameters):
#     modAlpha, modBeta = mods
#     paramsAlpha, paramsBeta = test_parameters
#     all_sims = [pyplt.timecourse(modAlpha, [0,5*60,15*60,30*60,60*60], [['TotalpSTAT',"Total pSTAT"]],
#                                  suppress=True, parameters = paramsAlpha+{'IFN':10E-12*NA*volEC})["TotalpSTAT"],
#                 pyplt.timecourse(modBeta, [0,5*60,15*60,30*60,60*60], [['TotalpSTAT',"Total pSTAT"]], 
#                                  suppress=True, parameters = test_parameters[1])["TotalpSTAT"], 
#                 pyplt.timecourse(modAlpha, [0,5*60,15*60,30*60,60*60], [['TotalpSTAT',"Total pSTAT"]], 
#                                  suppress=True, parameters = test_parameters[0])["TotalpSTAT"],
#                 pyplt.timecourse(modBeta, [0,5*60,15*60,30*60,60*60], [['TotalpSTAT',"Total pSTAT"]],
#                                  suppress=True, parameters = test_parameters[1])["TotalpSTAT"],
#                 pyplt.timecourse(modAlpha, [0,5*60,15*60,30*60,60*60], [['TotalpSTAT',"Total pSTAT"]],
#                                  suppress=True, parameters = test_parameters[0])["TotalpSTAT"],
#                 pyplt.timecourse(modBeta, [0,5*60,15*60,30*60,60*60],  [['TotalpSTAT',"Total pSTAT"]],
#                                  suppress=True, parameters = test_parameters[1])["TotalpSTAT"]]
#     SSres = 0   
#     for i in range(len(IFN_exps)):
#         SSres += np.sum(np.square(np.subtract(IFN_exps[i],all_sims[i])))
#     return SSres
# =============================================================================

# =============================================================================
# J() is the jumping distribution, which generates a new parameter vector given
# the current parameter vector and the priors provided by the user
# Inputs:
#     theta_old (list) = the current parameter vector
#     priors (list) = the prior parameter vector, with 'log' distributed parameters
#                     already converted to log values
# Returns:
#     theta (list) = a proposal parameter vector
# =============================================================================
# =============================================================================
# def J(theta_old, priors):
#     for p in range(len(theta_old)):
#         if priors[p][4]=='log':
#             # generate new parameter from log-normal centered at current value
#             # with std dev same as prior for that parameter ??????????   ?????????   ?????????    ????????
#             np.random.lognormal(mean=np.log(theta_old[p][1]), sigma=9)
#         elif priors[p][4]=='linear':
#             # generate new parameter from normal distribution centered at 
#             # current value and std dev = mean
#             theta_old[p][1] = np.random.uniform(low=priors[p][2], high=priors[p][3])
#     return theta_old
# 
# =============================================================================
# =============================================================================
# prior_probability() returns the probability of a set of parameters, given 
#                     their priors
# Inputs:
#   theta (list) = the parameters to 'score'
#   priors (list) = the prior knowledge of the same parameters
# Returns:
#   p(theta_1)*p(theta_2)*p(theta_3)...
# =============================================================================
def log_prior_probability(theta, priors):
    if debugging==True:
        if [el[0] for el in theta] != [el[0] for el in priors]:
            print("Parameter order got changed!")
            print([el[0] for el in theta])
            print([el[0] for el in priors])
            return 1
    p=1
    for parameter in range(len(theta)):
        if priors[parameter][4]=='log':
            # probability of drawing theta[parameter] from the prior log-normal distribution for parameter
            p *= scipy.stats.norm(priors[parameter][1],priors[parameter][2]).pdf(np.log(theta[parameter][1]))
    return np.log(p)

def log_likelihood(theta):
    return 1

# =============================================================================
# posterior() returns value proportional to the posterior probability for 
#             a model, given data
# Inputs:
#   theta (list) = the current model parameters
#   priors (list) = the prior information on the model parameters
# Returns:
#   float = the likelihood*prior_probability     
# =============================================================================
def posterior(theta, priors):
    L = log_likelihood(theta)
    P = log_prior_probability(theta, priors)    
    return L*P
    
def mcmcChecks(priors):
    # Sanity checks:
    for i in range(len(priors)):
        if priors[i][2]>priors[i][1] or priors[i][3]<priors[i][1]:
            print("Priors should be specified as ['name',guess,lower bound, upper bound,'distribution']")
            return False
    return True

# =============================================================================
# MCMC() is a Monte Carlo Markov Chain simulation function for finding optimal
# parameter values for IFN alpha and IFN beta PySB models
#    
# models (list of strings) = alpha and beta model files to fit
# chains (int) = number of parallel markov chains to simulate (recommend m >= 2)
# n (int) = number of iterations to perform for each chain 
# priors (list of lists) = these are the model parameters to fit.
#                          Each sublist is of the form 
#                          ['name', mu_0, lower limit, upper limit, 'distribution']
#            *distribution* is a keyword which can take the following strings:
#                   'log' - use a log-normal distribution with mean mu_0 and std. dev. = (upper-lower)/1.5
#                   'linear' - use a uniform distribution over the range lower - upper
#     
# =============================================================================
# =============================================================================
# def MCMC(models, chains, n, priors):
#     # Check for coherency of arguments
#     if mcmcChecks(priors)==False:
#         return 1
#     # Generate starting modes
#     if debugging == False:
#         starting_points = pp.fit_IFN_model(models, priors, chains*10)[0:chains]
#         
#         for each in starting_points: # Reformat string key into list
#             temp = re.split("', |\], \['", each[0][3:-2])
#             each[0] = [[temp[i],float(temp[i+1])] for i in range(0,len(temp),2)]
#         starting_points = [el[0] for el in starting_points] # discard previous score for model
#     else:
#         starting_points = [[['kpa', 4.069588809647314e-06], ['kSOCSon', 3.856211599006977e-05], ['R1', 2143.339929154], ['R2', 1969.2604543664138], ['gamma', 8.0], ['kd4', 3.0], ['k_d4', 6.0]]]
#     if debugging==True:
#         print("Priors")
#         print(priors)    
#     # Translate priors to log form to avoid extra computations, since almost 
#     #   everything will be done in log-probabilities
#     for p in range(len(priors)):
#         if priors[p][4]=='log':
#             # sigma chosen so that both bounds are within 1 std dev of guess; sigma stored in priors[p][2]
#             priors[p][2] = max(abs(np.log(priors[p][1]/priors[p][2])), abs(np.log(priors[p][1]/priors[p][3])))
#             priors[p][1] = np.log(priors[p][1])  # mu
# 
#     theta_record=[]        
#     # Perform Metropolis-Hastings algorithm
#     for chain in range(chains): # eventually I will parallelize or vectorize this
#         theta_old = starting_points[chain]
#         theta_record.append([el[1] for el in theta_old])
#         r2 = posterior(theta_old,priors)
#         for iteration in range(n): # sequential loop, cannot be parallelized
#             # propose a new theta
#             theta = J(theta_old, priors)
#             r1 = posterior(theta,priors)
#             R = r1/r2
#             if np.random.rand() < R:
#                 theta_record.append([el[1] for el in theta])                
#                 theta_old = theta
#                 r2 = r1
#     fig, ax = plt.subplots()
#     ax.set(xscale="linear", yscale="log")
#     plt.scatter(range(len(theta_record)),[el[0] for el in theta_record])
#     print([el[0] for el in theta_record])
#     plt.savefig('prior.pdf')
#     return 0
# 
# =============================================================================

#
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
def score_model(kpa,kSOCSon,kd4,k_d4,R1,R2, gamma, rho):
    lk = get_likelihood_logp(kpa,kSOCSon,kd4,k_d4,R1,R2, gamma)
    pr = get_prior_logp(kpa, kSOCSon, kd4, k_d4, R1, R2)
    return lk+(pr/(rho)**2)

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
def get_acceptance_rate(theta, rho):
    old_theta=theta
    old_score = score_model(*[old_theta[j][1] for j in range(len(old_theta))], rho)
    acceptance = 0    
    for i in range(50):
        proposal = J(old_theta)
        new_score = score_model(*[proposal[j][1] for j in range(len(proposal))], rho)
        if np.random.rand() < np.exp(-(new_score-old_score)):
        # if rand() < probability of proposed/probability of old
            old_theta=proposal
            old_score = new_score
            acceptance += 1
    return (acceptance*2, old_theta) # = acceptance/50*100
            
def hyperparameter_fitting(n, theta_0, rho, max_attempts):
    n=int(0.1*n)
    if n<60: n=60
    theta = [el for el in theta_0]
    # Try to find variances that give an good acceptance rate
    for attempt in range(max_attempts):
        print("Attempt {}".format(attempt+1))
        acceptance, new_theta = get_acceptance_rate(theta, rho)
        
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
                    
    raise RuntimeError("Failed to optimize hyperparameters.\n\
                       Please initialise with different variances\n\
                       or check uniform prior ranges, and try again.")


# =============================================================================
# MCMC() takes an IFN model and fits it using Markov Chain Monte Carlo
# Inputs:    
#    n (int) = number of iterations to run per chain
#    theta_0 (list) = the initial guesses and jumping distribution definitions for each parameter to fit
#                       Order of theta_0 is [kpa, kSOCSon, kd4, k_d4, R1, R2, gamma]    
#                   eg. [['kpa',1E-6,0.2,'log'],['R2',2E3,250,'linear',100],['gamma',4,2,'uniform',40]]
#    rho (float) = the prior component of the model square is weighed by a factor 1/rho**2
# Optional Inputs:    
#    max_attempts (int) = the number of attempts to try and choose hyperparameters
#                           default is 6
#    pflag (Boolean) = whether to plot the random walks of each parameter being fit
#    sflag (Boolean) = whether to automatically adjust prior sigmas to achieve good acceptance rate
# =============================================================================
def MCMC(n, theta_0, rho, max_attempts=6, pflag=True, sflag=True):
    # Selecting hyperparameters
    print("Optimizing hyperparameters")
    hyper_theta = hyperparameter_fitting(n, theta_0, rho, max_attempts)
    # Burn-in
    hyper_theta=theta_0
    
    
    
    model_record=[hyper_theta]
    old_score = score_model(*[model_record[0][j][1] for j in range(len(model_record[0]))], rho)
    old_index = 0
    acceptance = 0
    # Metropolis-Hastings algorithm
    progress_bar = n/10
    for i in range(n):
        # Monitor acceptance rate            
        if i>progress_bar:
            progress_bar += n/10
            print("{:.1f}% done".format(i/n*100))
            print("Acceptance rate = {:.1f}%".format(acceptance/progress_bar*100))
        proposal = J(model_record[old_index])
        new_score = score_model(*[proposal[j][1] for j in range(len(proposal))], rho)
        if np.random.rand() < np.exp(-(new_score-old_score)):
        # if rand() < probability of proposed/probability of old

# =============================================================================
#                if debugging==True:
#                     lk = get_likelihood_logp(*[proposal[j][1] for j in range(len(proposal))])
#                     pr = get_prior_logp(*[proposal[j][1] for j in range(len(proposal))][0:len(proposal)-1])
# =============================================================================
            model_record.append(proposal)
            old_score = new_score
            old_index += 1
            acceptance += 1
    
    if pflag==True:
        for i in range(4):
            fig, ax = plt.subplots()
            ax.set(xscale='linear',yscale='log')
            ax.plot(range(len(model_record)),[el[i][1] for el in model_record]) 
        for i in range(3):
            fig, ax = plt.subplots()
            ax.set(xscale='linear',yscale='linear')
            ax.plot(range(len(model_record)),[el[i+4][1] for el in model_record]) 
    
    
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
    p0=[['kpa',1E-6,0.04,'log'],['kSOCSon',1E-6,0.2,'log'],['kd4',0.3,0.2,'log'],
        ['k_d4',0.006,0.5,'log'],['R1',2E3,250,'linear',100],['R2',2E3,250,'linear',100],
        ['gamma',4,1,'uniform',40]]
    MCMC(500, p0, 0.01)

if __name__ == '__main__':
    main()
