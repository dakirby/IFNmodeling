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
def J(theta_old, priors):
    for p in range(len(theta_old)):
        if priors[p][4]=='log':
            # generate new parameter from log-normal centered at current value
            # with std dev same as prior for that parameter ??????????   ?????????   ?????????    ????????
            np.random.lognormal(mean=np.log(theta_old[p][1]), sigma=9)
        elif priors[p][4]=='linear':
            # generate new parameter from normal distribution centered at 
            # current value and std dev = mean
            theta_old[p][1] = np.random.uniform(low=priors[p][2], high=priors[p][3])
    return theta_old

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
def MCMC(models, chains, n, priors):
    # Check for coherency of arguments
    if mcmcChecks(priors)==False:
        return 1
    # Generate starting modes
    if debugging == False:
        starting_points = pp.fit_IFN_model(models, priors, chains*10)[0:chains]
        
        for each in starting_points: # Reformat string key into list
            temp = re.split("', |\], \['", each[0][3:-2])
            each[0] = [[temp[i],float(temp[i+1])] for i in range(0,len(temp),2)]
        starting_points = [el[0] for el in starting_points] # discard previous score for model
    else:
        starting_points = [[['kpa', 4.069588809647314e-06], ['kSOCSon', 3.856211599006977e-05], ['R1', 2143.339929154], ['R2', 1969.2604543664138], ['gamma', 8.0], ['kd4', 3.0], ['k_d4', 6.0]]]
    if debugging==True:
        print("Priors")
        print(priors)    
    # Translate priors to log form to avoid extra computations, since almost 
    #   everything will be done in log-probabilities
    for p in range(len(priors)):
        if priors[p][4]=='log':
            # sigma chosen so that both bounds are within 1 std dev of guess; sigma stored in priors[p][2]
            priors[p][2] = max(abs(np.log(priors[p][1]/priors[p][2])), abs(np.log(priors[p][1]/priors[p][3])))
            priors[p][1] = np.log(priors[p][1])  # mu

    theta_record=[]        
    # Perform Metropolis-Hastings algorithm
    for chain in range(chains): # eventually I will parallelize or vectorize this
        theta_old = starting_points[chain]
        theta_record.append([el[1] for el in theta_old])
        r2 = posterior(theta_old,priors)
        for iteration in range(n): # sequential loop, cannot be parallelized
            # propose a new theta
            theta = J(theta_old, priors)
            r1 = posterior(theta,priors)
            R = r1/r2
            if np.random.rand() < R:
                theta_record.append([el[1] for el in theta])                
                theta_old = theta
                r2 = r1
    fig, ax = plt.subplots()
    ax.set(xscale="linear", yscale="log")
    plt.scatter(range(len(theta_record)),[el[0] for el in theta_record])
    print([el[0] for el in theta_record])
    plt.savefig('prior.pdf')
    return 0


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
    S_k_d4 = 4
    
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
    q_4 = 3.62e-4/k_d4
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
    
    import IFN_beta_altSOCS as IFNbSOCS
    from pysb.simulator import ScipyOdeSimulator

    simres = ScipyOdeSimulator(IFNbSOCS.model, tspan=t, 
                               param_values = {'R1':R1, 'R2':R2,
                                               'IFN':600E-12,'kSOCSon':kSOCSon, 
                                               'kpa':kpa, 'k_d4':k_d4},
                               compiler='python').run()
    simres = simres.all
    simres = simres['TotalpSTAT']
    print(simres)
    print(all_sims[5])

    logp = 0
    for i in range(len(all_sims)):
        logp += np.sum(np.square(np.divide(np.subtract(IFN_exps[i],np.divide(all_sims[i],gamma)),np.divide(IFN_sigmas[i],gamma))))
    return logp
    
    

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
#     p0 = [['kpa',1E-6,1E-7,5E-6,'log'],['kSOCSon',1E-6,9E-7,6e-5,'log'],
#           ['R1',2000,1000,5000,'log'],['R2',2000,1000,5000,'log'],
#           ['gamma',3.5,2,40,'linear'],['kd4',0.3,.01,7,'log'],['k_d4',0.006,0.001,7,'log']]
# =============================================================================
    p0=[['kd4',0.3,0.03,6,'log']]
    t0 = time.time()
    print(get_likelihood_logp(1.2E-6, 2E-6, 3, 6, 2000, 2000, 4))
    t1 = time.time()
    print("time = {}".format(t1-t0))
    #MCMC(modelfiles, 1, 25, p0)
# =============================================================================
#     walk = [0.3]
#     for i in range(2000):
#         walk.append(np.random.lognormal(mean=np.log(walk[i]), sigma=0.1))
# =============================================================================

if __name__ == '__main__':
    main()
