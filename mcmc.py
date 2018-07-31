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
# Returns:
#     theta (list) = a proposal parameter vector
# =============================================================================
def J(theta_old):
    print(theta_old)
    for p in range(len(theta_old)):
        theta_old[p][1]+=np.random.rand()
    print(theta_old)
    return theta_old

def posterior(theta):
    
    return 1
def mcmcChecks(priors):
    # Sanity checks:
    for i in range(len(priors)):
        if priors[i][2]>priors[i][1] or priors[i][3]<priors[i][1]:
            print("Priors should be specified as ['name',guess,lower bound, upper bound,'distribution']")
            return False
    return True

# models (list of strings) = alpha and beta model files to fit
# m (int) = number of parallel markov chains to simulate (recommend m >= 2)
# n (int) = number of iterations to perform for each chain 
# priors (list of lists) = these are the model parameters to fit.
#                          Each sublist is of the form 
#                          ['name', mu_0, lower limit, upper limit, 'distribution']
#            *distribution* is a keyword which can take the following strings:
#                   'normal' - use a normal distribution with mean mu_0 and std. dev. = (upper-lower)/1.5
#                   'uniform' - use a uniform distribution over the range lower - upper
def MCMC(models, m, n, priors):
    # Check for coherency of arguments
    if mcmcChecks(priors)==False:
        return 1

    # Generate starting modes
    starting_points = pp.fit_IFN_model(models, priors, m*10)[0:m]
    
    for each in starting_points: # Reformat string key into list
        temp = re.split("', |\], \['", each[0][3:-2])
        each[0] = [[temp[i],float(temp[i+1])] for i in range(0,len(temp),2)]
    starting_points = [el[0] for el in starting_points] # discard previous score for model

       

    # Perform Metropolis-Hastings algorithm
    for chain in range(m): # eventually I will parallelize or vectorize this
        theta_old = starting_points[chain]
        r2 = posterior(theta_old)
        for iteration in range(n): # sequential loop, cannot be parallelized
            # propose a new theta
            theta = J(theta_old)
            r1 = posterior(theta)
            R = r1/r2
            if np.random.rand() < R:
                theta_old = theta
                r2 = r1
            
    
    return 0

def main():
    modelfiles = ['IFN_alpha_altSOCS_ppCompatible','IFN_beta_altSOCS_ppCompatible']
    inital_points = [['kpa',1E-6,1E-7,5E-6,'log'],['kSOCSon',1E-6,9E-7,6e-5,'log'],['k4',1,1,1000,'log'],
					 ['gamma',3.5,2,40,'linear'],['R1',2000,1000,5000,'linear'],['R2',2000,1000,5000,'linear']]
    MCMC(modelfiles, 2, 10, inital_points)

if __name__ == '__main__':
    main()
