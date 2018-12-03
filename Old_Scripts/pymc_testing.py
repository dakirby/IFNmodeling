# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 20:40:36 2018

@author: Duncan

Attempt to use PyMC3 library for Bayesian fitting with MCMC
"""

import theano.tensor as tt
import numpy as np
#import pickle # python3 to save results 
import matplotlib.pyplot as plt
plt.close('all')
plt.style.use('seaborn-darkgrid')
import pymc3 as pm
print('Running on PyMC3 v{}'.format(pm.__version__))
import pandas as pd
#from pymc3.step_methods import SMC
#import shutil
from pysb.simulator import ScipyOdeSimulator
import IFN_alpha_altSOCS as IFNaModelfile

def get_pSTAT(kd4):
    mod = IFNaModelfile.model
    simres = ScipyOdeSimulator(mod, tspan=[0,5,15,30,60], 
                               param_values={'kd4':kd4.random(size=1)},
                               compiler='python').run()
    timecourse = simres.all
    return timecourse['TotalpSTAT']

def two_gaussians(x, kd4):
    n = 4
    mu1 = np.ones(n) * (1. / 2)
    mu2 = -mu1    
    stdev = 0.1
    sigma = np.power(stdev, 2) * np.eye(n)
    isigma = np.linalg.inv(sigma)
    dsigma = np.linalg.det(sigma)    
    w1 = 0.1
    w2 = (1 - w1)
   
    addVar = get_pSTAT(kd4)[0]
    log_like1 = - 0.5 * n * tt.log(2 * np.pi) \
                - 0.5 * tt.log(dsigma) \
                - 0.5 * (x - mu1).T.dot(isigma).dot(x - mu1)
    log_like2 = - 0.5 * n * tt.log(2 * np.pi) \
                - 0.5 * tt.log(dsigma) \
                - 0.5 * (x - mu2).T.dot(isigma).dot(x - mu2)
    return 0.0001*(tt.log(w1 * tt.exp(log_like1) + w2 * tt.exp(log_like2))) + addVar

def main():
    n = 4
    mu1 = np.ones(n) * (1. / 2)

    with pm.Model() as ATMIP_test:
        X = pm.Uniform('X',
                       shape=n,
                       lower=-2. * np.ones_like(mu1),
                       upper=2. * np.ones_like(mu1),
                       testval=-1. * np.ones_like(mu1))
        kd4 = pm.Lognormal('kd4', mu=np.log(0.3), sd=1)
#        k_d4 = pm.Lognormal('k_d4', mu=np.log(6E-3), sd=9)
#        kSOCSon = pm.Lognormal('kSOCSon', mu=np.log(1E-6), sd = 2)
#        kpa = pm.Lognormal('kpa', mu=np.log(1E-6), sd = 2)
#        R1 = pm.Uniform('R1', lower=900, upper=5000)
#        R2 = pm.Uniform('R2', lower=900, upper=5000)
#        gamma = pm.Uniform('gamma', lower=2, upper=30)
        
        llk = pm.Potential('llk', two_gaussians(X, kd4))
    with ATMIP_test:
        trace = pm.sample(100, chains=50, step=pm.SMC())
        plt.figure()
        pm.traceplot(trace);
        plt.savefig("mc_testing.pdf")
        s = pm.stats.summary(trace)
        s.to_csv('mcmc_parameter_summary.csv')
# =============================================================================
# Working first example
# =============================================================================
# =============================================================================
#     # Initialize random number generator
#     np.random.seed(123)    
#     # True parameter values
#     alpha, sigma = 1, 1
#     beta = [1, 2.5]    
#     # Size of dataset
#     size = 100    
#     # Predictor variable
#     X1 = np.random.randn(size)
#     X2 = np.random.randn(size) * 0.2    
#     # Simulate outcome variable
#     Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma   
#     # =============================================================================
#     # fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,4))
#     # axes[0].scatter(X1, Y)
#     # axes[1].scatter(X2, Y)
#     # axes[0].set_ylabel('Y'); axes[0].set_xlabel('X1'); axes[1].set_xlabel('X2');
#     # =============================================================================
#     
#     with pm.Model() as basic_model:
#     
#         # Priors for unknown model parameters
#         kd4 = pm.Lognormal('kd4', mu=np.log(0.3), sd=6)
#         k_d4 = pm.Lognormal('k_d4', mu=np.log(6E-3), sd=9)
#         kSOCSon = pm.Lognormal('kSOCSon', mu=np.log(1E-6), sd = 2)
#         kpa = pm.Lognormal('kpa', mu=np.log(1E-6), sd = 2)
#         R1 = pm.Uniform('R1', lower=900, upper=5000)
#         R2 = pm.Uniform('R2', lower=900, upper=5000)
#         gamma = pm.Uniform('gamma', lower=2, upper=30)
#         
#         beta = pm.Normal('beta', mu=0, sd=10, shape=2)
#         sigma = pm.HalfNormal('sigma', sd=1)
#     
#         # Expected value of outcome
#         #mu = k_d4 + beta[0]*X1 + beta[1]*X2
#     
#         # Likelihood (sampling distribution) of observations
#         #Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)
#         # draw 500 posterior samples
#         trace = pm.sample(100)
#         plt.figure()
#         pm.traceplot(trace)
#         plt.savefig("mc_testing.pdf")
#         #with open('my_model.pkl', 'wb') as buff:
#         #    pickle.dump({'model': basic_model, 'trace': trace}, buff)
#         
#         # PySB Models
# # =============================================================================
# #     from pysb.simulator import ScipyOdeSimulator
# #     from IFNmodeling import IFN_alpha_altSOCS as IFNaModelfile
# #     simres = ScipyOdeSimulator(IFNaModelfile.model, tspan=[0,5,15,30,60], compiler='theano').run()
# #     timecourse = simres.all
# # =============================================================================
# 
# =============================================================================
        
if __name__ == '__main__':
    main()
        
    