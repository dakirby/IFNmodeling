# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 20:40:36 2018

@author: Duncan

Attempt to use PyMC3 library for Bayesian fitting with MCMC
"""
def main():
    import numpy as np
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.style.use('seaborn-darkgrid')
    import pickle # python3    
    import pymc3 as pm
    print('Running on PyMC3 v{}'.format(pm.__version__))

    # PySB Models
# =============================================================================
#     from pysb.simulator import ScipyOdeSimulator
#     from IFNmodeling import IFN_alpha_altSOCS as IFNaModelfile
#     simres = ScipyOdeSimulator(IFNaModelfile.model, tspan=[0,5,15,30,60], compiler='theano').run()
#     timecourse = simres.all
# =============================================================================
    
    # Initialize random number generator
    np.random.seed(123)
    
    # True parameter values
    alpha, sigma = 1, 1
    beta = [1, 2.5]
    
    # Size of dataset
    size = 100
    
    # Predictor variable
    X1 = np.random.randn(size)
    X2 = np.random.randn(size) * 0.2
    
    # Simulate outcome variable
    Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma
    
    # =============================================================================
    # fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,4))
    # axes[0].scatter(X1, Y)
    # axes[1].scatter(X2, Y)
    # axes[0].set_ylabel('Y'); axes[0].set_xlabel('X1'); axes[1].set_xlabel('X2');
    # =============================================================================
    
    basic_model = pm.Model()
    
    with basic_model:
    
        # Priors for unknown model parameters
        k_d4 = pm.Lognormal('k_d4', mu=6E-3, sd=9)
        beta = pm.Normal('beta', mu=0, sd=10, shape=2)
        sigma = pm.HalfNormal('sigma', sd=1)
        # Use pm.Lognormal('x', mu=2, sd=30)   for reaction rates
    
        # Expected value of outcome
        mu = k_d4 + beta[0]*X1 + beta[1]*X2
    
        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)
        
        # draw 500 posterior samples
        trace = pm.sample(100)
        plt.figure()
        pm.traceplot(trace)
        plt.savefig("mc_testing.pdf")
        #with open('my_model.pkl', 'wb') as buff:
        #    pickle.dump({'model': basic_model, 'trace': trace}, buff)
        
if __name__ == '__main__':
    main()
        
    