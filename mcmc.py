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
            theta_old[p][1] = np.exp(np.random.normal(np.log(theta_old[p][1]), priors[p][2]))
        elif priors[p][4]=='linear':
            # generate new parameter from normal distribution centered at 
            # current value and std dev = mean
            theta_old[p][1] = np.random.normal(theta_old[p][1], theta_old[p][1])
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
def prior_probability(theta, priors):
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
    return p

def likelihood(theta):
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
    L = likelihood(theta)
    P = prior_probability(theta, priors)    
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
# m (int) = number of parallel markov chains to simulate (recommend m >= 2)
# n (int) = number of iterations to perform for each chain 
# priors (list of lists) = these are the model parameters to fit.
#                          Each sublist is of the form 
#                          ['name', mu_0, lower limit, upper limit, 'distribution']
#            *distribution* is a keyword which can take the following strings:
#                   'log' - use a log-normal distribution with mean mu_0 and std. dev. = (upper-lower)/1.5
#                   'linear' - use a uniform distribution over the range lower - upper
#     
# =============================================================================
def MCMC(models, m, n, priors):
    # Check for coherency of arguments
    if mcmcChecks(priors)==False:
        return 1
    # Generate starting modes
    if debugging == False:
        starting_points = pp.fit_IFN_model(models, priors, m*10)[0:m]
        
        for each in starting_points: # Reformat string key into list
            temp = re.split("', |\], \['", each[0][3:-2])
            each[0] = [[temp[i],float(temp[i+1])] for i in range(0,len(temp),2)]
        starting_points = [el[0] for el in starting_points] # discard previous score for model
    else:
        starting_points = [[['kpa', 4.069588809647314e-06], ['kSOCSon', 3.856211599006977e-05], ['R1', 2143.339929154], ['R2', 1969.2604543664138], ['gamma', 8.0], ['kd4', 3.0], ['k_d4', 6.0]]]
        
    # Translate priors to log form to avoid extra computations, since almost 
    #   everything will be done in log-probabilities
    for p in range(len(priors)):
        if priors[p][4]=='log':
            # sigma chosen so that both bounds are within 1 std dev of guess; sigma stored in priors[p][2]
            priors[p][2] = max(abs(np.log(priors[p][1]/priors[p][2])), abs(np.log(priors[p][1]/priors[p][3])))
            priors[p][1] = np.log(priors[p][1])  # mu

    pVal=[]        
    # Perform Metropolis-Hastings algorithm
    for chain in range(m): # eventually I will parallelize or vectorize this
        theta_old = starting_points[chain]
        r2 = posterior(theta_old,priors)
        for iteration in range(n): # sequential loop, cannot be parallelized
            # propose a new theta
            theta = J(theta_old, priors)
            r1 = posterior(theta,priors)
            R = r1/r2
            print(R)
            if np.random.rand() < R:
                theta_old = theta
                r2 = r1
                pVal.append(theta[0][1])
    print(pVal)
    plt.figure()
    plt.scatter(range(len(pVal)),np.log(pVal))
    plt.savefig('prior.pdf')
    return 0

def main():
    plt.close('all')
    modelfiles = ['IFN_alpha_altSOCS_ppCompatible','IFN_beta_altSOCS_ppCompatible']
    p0 = [['kpa',1E-6,1E-7,5E-6,'log'],['kSOCSon',1E-6,9E-7,6e-5,'log'],
          ['R1',2000,1000,5000,'log'],['R2',2000,1000,5000,'log'],
          ['gamma',3.5,2,40,'linear'],['kd4',0.3,.01,7,'log'],['k_d4',0.006,0.001,7,'log']]
    
    MCMC(modelfiles, 1, 10, p0)

if __name__ == '__main__':
    main()
