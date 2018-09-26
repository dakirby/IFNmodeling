# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 13:04:11 2018

@author: Duncan

A collection of functions for making predictions with output from MCMC
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns; 
sns.set(color_codes=True)
sns.set_style("darkgrid")
from mcmc import score_model
# =============================================================================
# import os
# script_dir = os.path.dirname(__file__)
# results_dir = os.path.join(script_dir, 'MCMC_Results/')
# chain_results_dir = results_dir+'Chain_Results/'
# if not os.path.isdir(results_dir):
#     os.makedirs(results_dir)
# if not os.path.isdir(chain_results_dir):
#     os.makedirs(chain_results_dir)
# =============================================================================
    
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

# =============================================================================
# ppc() returns a posterior predictive check of the model sampling
# modelfile: the dictionary of model parameters 
# samplesize: an integer dictating how many samples to read from the posterior samples file
#               Note: this is an input argument to bayesian_timecourse            
# Returns: corr (tuple) = (chi squared value, degrees of freedom)
# =============================================================================
def ppc(modelfile, samplesize):
    model_samples = pd.read_csv(modelfile, index_col=0)
    dof = 0
    for i in range(samplesize):
        model = model_samples.iloc[i]
        mod = model.to_dict()
        # Unpack variables
        R1Exists=False
        meanRExists=False
        for key in mod.keys():
            if key=='kpa':
                kpa=mod[key]
            elif key=='kSOCSon':
                kSOCSon=mod[key]
            elif key=='kd4':
                kd4=mod[key]
            elif key=='k_d4':
                k_d4=mod[key]
            elif key=='R1':
                R1Exists=True
                R1=mod[key]
            elif key=='R2':
                R2=mod[key]
            elif key=='meanR':
                meanRExists=True
                meanR=mod[key]
            elif key=='delR':
                delR=mod[key]
            elif key=='gamma':
                gamma=mod[key]
        if R1Exists==False:
            if meanRExists==False:
                meanR=2E3
            R1=meanR-delR/2
            R2=meanR+delR/2
    
        # Build parameter list
        import ODE_system_alpha
        alpha_mod = ODE_system_alpha.Model()
        import ODE_system_beta
        beta_mod = ODE_system_beta.Model()
        # Build parameter lists
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
        # run simulations under experimental conditions
        alpha_parameters[I_index_Alpha] = NA*volEC*10E-12
        (_, sim) = alpha_mod.simulate(t, param_values=alpha_parameters)
        all_sims.append(np.multiply(gamma,sim['TotalpSTAT']))
        beta_parameters[I_index_Beta] = NA*volEC*10E-12
        (_, sim) = beta_mod.simulate(t, param_values=beta_parameters)
        all_sims.append(np.multiply(gamma,sim['TotalpSTAT']))
        
        alpha_parameters[I_index_Alpha] = NA*volEC*90E-12
        (_, sim) = alpha_mod.simulate(t, param_values=alpha_parameters)
        all_sims.append(np.multiply(gamma,sim['TotalpSTAT']))
        beta_parameters[I_index_Beta] = NA*volEC*90E-12
        (_, sim) = beta_mod.simulate(t, param_values=beta_parameters)
        all_sims.append(np.multiply(gamma,sim['TotalpSTAT']))
    
        alpha_parameters[I_index_Alpha] = NA*volEC*600E-12
        (_, sim) = alpha_mod.simulate(t, param_values=alpha_parameters)
        all_sims.append(np.multiply(gamma,sim['TotalpSTAT']))
        beta_parameters[I_index_Beta] = NA*volEC*600E-12
        (_, sim) = beta_mod.simulate(t, param_values=beta_parameters)
        all_sims.append(np.multiply(gamma,sim['TotalpSTAT']))
    
    # Calculate degrees of freedom (ie. number of parameters in MCMC)
    dof = len(mod.keys())

    # Calculate variance for each model prediction of data point
    prediction_variances = [np.var(all_sims[i],axis=0) for i in range(len(all_sims))]
    # Calculate posterior predictive check:
    Chi2 = 0
    for i in range(len(all_sims)): #[0,2,4]:#
        Chi2 += np.sum(np.divide(np.square(np.subtract(all_sims[i],IFN_exps[i])),np.add(prediction_variances[i],IFN_sigmas[i])))
    return (Chi2,dof)
# =============================================================================
# MAP() finds the maximum a posteriori model from the posterior models listed in
# posterior_file (Input) and returns a tuple containing a 
# dictionary of the model with the lowest score and its value for gamma
# =============================================================================
def MAP(posterior_file, beta, rho, debugging=False):
    df = pd.read_csv(posterior_file,index_col=0)
    names=list(df.drop('gamma',axis=1).columns.values)
    best_score=1E8
    best_model=dict((key, 0) for key in names if key != 'gamma')
    best_gamma=1
    model=dict((key, 0) for key in names if key != 'gamma')
    for i in range(len(df)):
        for n in names:
            model.update({n:df.iloc[i][n]})
        [new_score,new_gamma] = score_model([[j,model[j]] for j in names if j!='gamma'], beta,rho)
        if new_score<best_score:
            best_score=new_score
            best_gamma=new_gamma
            best_model=model.copy()
    if debugging==True:
        print("The best model was")
        print(best_model)
        print('and gamma = '+str(best_gamma))
        print("with as score of")
        score_model([[j,best_model[j]] for j in names if j!='gamma'], beta,rho,debugging=True)
    return (best_model, best_gamma)

# =============================================================================
# MAP_timecourse() takes the MAP model from MAP() as well as gamma, dose_spec 
# species concentration, end time, and output species (Input), and returns a 
# prediction for the output species as a list
# =============================================================================
def MAP_timecourse(model, gamma, dose, dose_spec, end_time, spec):
    import ODE_system_alpha
    alpha_mod = ODE_system_alpha.Model()
    import ODE_system_beta
    beta_mod = ODE_system_beta.Model()
    key_list = []
    meanR=2E3
    for key in model:
        if key=='delR':
            key_list.append('R1')
            key_list.append('R2')
        elif key=='meanR':
            meanR=model['meanR']
        else:
            key_list.append(key)
    alpha_pvec=[]
    for p in alpha_mod.parameters:
        if p.name in key_list:
            if p.name=='R1':
                alpha_pvec.append(meanR-model['delR']/2)
            elif p.name=='R2':
                alpha_pvec.append(meanR+model['delR']/2)
            else:
                alpha_pvec.append(model[p.name])
        elif p[0]=='kd3':
            q1 = 3.321155762205247e-14/1
            q2 = 4.98173364330787e-13/0.015
            q4 = 3.623188E-4/model['kd4']
            q3 = q2*q4/q1
            kd3 = 3.623188E-4/q3   
            alpha_pvec.append(kd3)            
        else:
            alpha_pvec.append(p.value)
    beta_pvec=[]
    for p in beta_mod.parameters:
        if p[0] in key_list:
            if p.name=='R1':
                beta_pvec.append(meanR-model['delR']/2)
            elif p.name=='R2':
                beta_pvec.append(meanR+model['delR']/2)
            else:
                beta_pvec.append(model[p.name])
        elif p[0]=='k_d3':
            q_1 = 4.98E-14/0.03
            q_2 = 8.30e-13/0.002
            q_4 = 3.623188e-4/model['k_d4']
            q_3 = q_2*q_4/q_1
            k_d3 = 3.623188e-4/q_3
            beta_pvec.append(k_d3)
        else:
            beta_pvec.append(p.value)
    I_index_Alpha = [el[0] for el in alpha_mod.parameters].index(dose_spec)
    I_index_Beta = [el[0] for el in beta_mod.parameters].index(dose_spec)
    t=np.linspace(0,end_time)
    # Run simulation
    alpha_pvec[I_index_Alpha] = dose
    (_, sim) = alpha_mod.simulate(t, param_values=alpha_pvec)
    alpha_result = sim[spec]
    beta_pvec[I_index_Beta] = dose
    (_, sim) = beta_mod.simulate(t, param_values=beta_pvec)
    beta_result = sim[spec]
    return [np.multiply(gamma,alpha_result), np.multiply(gamma,beta_result)]

          
    
# =============================================================================
# bayesian_timecourse() runs a time course for each sample from posterior parameter 
# distribution, giving prediction intervals for all time points
# Inputs:
#     samplefile (str) = the name of the posterior samples file output from 
#                         Markov Chain Monte Carlo simulations
#     dose (float) = dose for the time course (IFN concentration in M)
#     end_time (int) = the end time for the simulation (in seconds)
#     sample_size (int) = the number of posterior samples to use
#     percent (int) = the percentile bounds for error in model prediction 
#                       (bounds will be 'percent' and 100-'percent') 
#     spec (string) = name of species to predict intervals for
#     beta (float) = the value of beta to use for the MAP solution    
#     rho (float) = the value of rho to use for the MAP solution    
#     suppress (Boolean) = whether or not to plot the time course (default is False)       
#     dose_species (list) = any model observable can be used; second and third list itmes are 
#                           multiplicative factors. If not needed then set to 1
#                           default is ['I' for Interferon, NA = 6.022E23, volEC = 1E-5]
#                           looks like ['I', 6.022E23, 1E-5]
#     corr_flag (Boolean) = whether or not to calculate posterior predictive check (default is False)
# Returns
#   prediction_interval (list) = pairs of two lists, corresponding to alpha then beta predictions 
#                                for each species, in order given in specList.
#                                 Each item of the form [mean, lower error, upper error]        
# =============================================================================
def bayesian_timecourse(samplefile, dose, end_time, sample_size, percent, spec, rho, beta, 
                        suppress=False, dose_species=['I', 6.022E23, 1E-5], corr_flag=False):
    # Read samples
    samples = pd.read_csv(samplefile,index_col=0)
    # Check that function inputs are valid
    (nSamples, nVars) = samples.shape
    if sample_size > nSamples:
        raise ValueError("Not enough samples in the posterior file for nPost chosen")
    # Get variables that were sampled
    variable_names = list(samples.columns.values)
    # Import models
    import ODE_system_alpha
    alpha_mod = ODE_system_alpha.Model()
    import ODE_system_beta
    beta_mod = ODE_system_beta.Model()
    # Choose 'global' best model (fits complete data optimally, not just dose-response data points)
    (best_model,best_gamma) = MAP(samplefile, beta, rho)#best_model is a dict with parameter values and best_gamma is a float
    if corr_flag==True:
        corr = ppc(samplefile,sample_size)
        print("The MAP model and the data has a Chi2 score of {}".format(corr[0]))
        pval = 1 - stats.chi2.cdf(corr[0], corr[1])
        print("A Chi2 p-value for {} degrees of freedom is {}".format(corr[1],pval))
        

    # Make best_model into a list to compare to all model later
    best_model_list = [[variable_names[i], best_model[variable_names[i]]] for i in range(nVars) if variable_names[i] != 'gamma']

    alpha_curves=[]
    beta_curves=[]
    best_curves=[]
    # Simulate each model
    found_MAP_bool=False
    for r in range(sample_size):
        MAP_bool=False #Flag to identify the MAP model
        parameter_vector = samples.iloc[r]#Pandas Series of variables and values, including gamma
        # Create list format excluding gamma:
        pList = [[variable_names[i], parameter_vector.loc[variable_names[i]]] for i in range(nVars) if variable_names[i] != 'gamma']
        # Check if this model is the MAP model:
        if best_model_list == pList:
            MAP_bool=True
            found_MAP_bool=True
        if 'meanR' in [var[0] for var in pList]:
            meanR = pList[[var[0] for var in pList].index('meanR')][1]
        else:
            meanR = 2E3
        # Convert any combined quantities
        for item in range(len(pList)):
            if pList[item][0]=='delR':
                R1=meanR-pList[item][1]/2
                R2=meanR+pList[item][1]/2
                pList=pList[0:item]+[['R1',R1],['R2',R2]]+pList[item+1:len(pList)]
        gamma = parameter_vector.loc['gamma']#of type numpy.float64
        # Maintain detailed balance
        if 'kd4' in variable_names:
            q1 = 3.321155762205247e-14/1
            q2 = 4.98173364330787e-13/0.015
            q4 = 3.623188E-4/parameter_vector.loc['kd4']
            q3 = q2*q4/q1
            kd3 = 3.623188E-4/q3                

            q_1 = 4.98E-14/0.03
            q_2 = 8.30e-13/0.002
            q_4 = 3.623188e-4/parameter_vector.loc['k_d4']
            q_3 = q_2*q_4/q_1
            k_d3 = 3.623188e-4/q_3
            pList += [['kd3', kd3],['k_d3',k_d3]]
        # Now create an ordered list of ALL model parameters to use in PySB sim
        alpha_parameters=[]
        beta_parameters=[]
        for p in alpha_mod.parameters:
            isInList=False
            for y in pList:
                if p[0]==y[0]:
                    alpha_parameters.append(y[1])
                    isInList=True
                    break
            if isInList==False:
                alpha_parameters.append(p.value)
        for p in beta_mod.parameters:
            isInList=False
            for y in pList:
                if p[0]==y[0]:
                    beta_parameters.append(y[1])
                    isInList=True
                    break
            if isInList==False:
                beta_parameters.append(p.value)
        # Prepare for dose-response curve simulation
        I_index_Alpha = [el[0] for el in alpha_mod.parameters].index(dose_species[0])
        I_index_Beta = [el[0] for el in beta_mod.parameters].index(dose_species[0])
        NA = dose_species[1] # 6.022E23
        volEC = dose_species[2] # 1E-5   
        t=np.linspace(0,end_time)
        # Run simulation for each dose
        alpha_results=[]
        beta_results=[]

        alpha_parameters[I_index_Alpha] = NA*volEC*dose
        (_, sim) = alpha_mod.simulate(t, param_values=alpha_parameters)
        alpha_results = gamma*sim[spec]
        beta_parameters[I_index_Beta] = NA*volEC*dose
        (_, sim) = beta_mod.simulate(t, param_values=beta_parameters)
        beta_results = gamma*sim[spec]

        alpha_curves.append(alpha_results)
        beta_curves.append(beta_results)
        if MAP_bool==True:
            best_curves = [alpha_results,beta_results]


    # If this was the last model and MAP wasn't looked at, include MAP:
    if found_MAP_bool==False:
        pList = best_model_list 
        if 'meanR' in [var[0] for var in pList]:
            meanR = pList[[var[0] for var in pList].index('meanR')][1]
        else:
            meanR = 2E3
        # Convert any combined quantities
        for item in range(len(pList)):
            if pList[item][0]=='delR':
                R1=meanR-pList[item][1]/2
                R2=meanR+pList[item][1]/2
                pList=pList[0:item]+[['R1',R1],['R2',R2]]+pList[item+1:len(pList)]
        gamma = parameter_vector.loc['gamma']#of type numpy.float64
        # Maintain detailed balance
        if 'kd4' in variable_names:
            q1 = 3.321155762205247e-14/1
            q2 = 4.98173364330787e-13/0.015
            q4 = 3.623188E-4/parameter_vector.loc['kd4']
            q3 = q2*q4/q1
            kd3 = 3.623188E-4/q3                

            q_1 = 4.98E-14/0.03
            q_2 = 8.30e-13/0.002
            q_4 = 3.623188e-4/parameter_vector.loc['k_d4']
            q_3 = q_2*q_4/q_1
            k_d3 = 3.623188e-4/q_3
            pList += [['kd3', kd3],['k_d3',k_d3]]
        # Now create an ordered list of ALL model parameters to use in PySB sim
        alpha_parameters=[]
        beta_parameters=[]
        for p in alpha_mod.parameters:
            isInList=False
            for y in pList:
                if p[0]==y[0]:
                    alpha_parameters.append(y[1])
                    isInList=True
                    break
            if isInList==False:
                alpha_parameters.append(p.value)
        for p in beta_mod.parameters:
            isInList=False
            for y in pList:
                if p[0]==y[0]:
                    beta_parameters.append(y[1])
                    isInList=True
                    break
            if isInList==False:
                beta_parameters.append(p.value)
        # Prepare for dose-response curve simulation
        I_index_Alpha = [el[0] for el in alpha_mod.parameters].index(dose_species[0])
        I_index_Beta = [el[0] for el in beta_mod.parameters].index(dose_species[0])
        NA = dose_species[1] # 6.022E23
        volEC = dose_species[2] # 1E-5   
        t=np.linspace(0,end_time)
        # Run simulation for each dose
        alpha_results=[]
        beta_results=[]

        alpha_parameters[I_index_Alpha] = NA*volEC*dose
        (_, sim) = alpha_mod.simulate(t, param_values=alpha_parameters)
        alpha_results = gamma*sim[spec]
        beta_parameters[I_index_Beta] = NA*volEC*dose
        (_, sim) = beta_mod.simulate(t, param_values=beta_parameters)
        beta_results = gamma*sim[spec]

        alpha_curves.append(alpha_results)
        beta_curves.append(beta_results)
        best_curves = [alpha_results,beta_results]

    
    # Prepare function return
    prediction_intervals=[]
    #   Alpha results
    upper_error_prediction = np.percentile(alpha_curves, percent, axis=0)
    lower_error_prediction = np.percentile(alpha_curves, 100-percent, axis=0)
    mean_prediction = np.mean(alpha_curves, axis=0)
    var_prediction = np.var(alpha_curves, axis=0)
    prediction_intervals.append([best_curves[0],lower_error_prediction,upper_error_prediction,
                                 mean_prediction, var_prediction, alpha_curves])

    #   Beta results
    upper_error_prediction = np.percentile(beta_curves, percent, axis=0)
    lower_error_prediction = np.percentile(beta_curves, 100-percent, axis=0)
    mean_prediction = np.mean(beta_curves, axis=0)
    var_prediction = np.var(beta_curves, axis=0)
    
    prediction_intervals.append([best_curves[1],lower_error_prediction,upper_error_prediction,
                                 mean_prediction, var_prediction, beta_curves])
    if suppress==False:
        fig, ax = plt.subplots()
        ax.plot(t, prediction_intervals[0][0], 'r')
        ax.plot(t, prediction_intervals[0][1], 'r--')
        ax.plot(t, prediction_intervals[0][2], 'r--')
        ax.plot(t, prediction_intervals[1][0], 'g')
        ax.plot(t, prediction_intervals[1][1], 'g--')
        ax.plot(t, prediction_intervals[1][2], 'g--')
        plt.show()
    return prediction_intervals

# =============================================================================
# bayesian_doseresponse() runs a dose response for each sample from posterior parameter 
# distribution, giving prediction intervals for all dose points
# Inputs:
#     samplefile (str) = the name of the posterior samples file output from 
#                         Markov Chain Monte Carlo simulations
#     dose (list) = doses for the simulation (IFN concentration in M)
#     end_time (int) = the end time for each time course (in seconds)
#     sample_size (int) = the number of posterior samples to use
#     spec (str) = name of species to predict intervals for
#     rho, beta (floats) = the values of rho and beta used in the sampling MCMC
#     percent (int) = the percentile bounds for error in model prediction 
#                       (bounds will be 'percent' and 100-'percent') 
#     suppress (Boolean) = whether or not to plot the results (default is False) 
#     dr_species (list) = any model observable can be used; second and third list itmes are 
#                           multiplicative factors. If not needed then set to 1
#                           default is ['I' for Interferon, NA = 6.022E23, volEC = 1E-5]
#                           looks like ['I', 6.022E23, 1E-5]          
# Returns
#   [alpha_responses, beta_responses] (list) = the dose response curves
#                       alpha_responses = [[mean curve, low curve, high curve] for each species]        
# =============================================================================
def bayesian_doseresponse(samplefile, doses, end_time, sample_size, percent, spec, rho,beta,
                          suppress=False, dr_species=['I', 6.022E23, 1E-5]):
    # Read samples
    samples = pd.read_csv(samplefile,index_col=0)
    # Check that function inputs are valid
    (nSamples, nVars) = samples.shape
    if sample_size > nSamples:
        print("Not enough samples in file")
        return 1
    # Get variables that were sampled
    variable_names = list(samples.columns.values)
    # Import models
    import ODE_system_alpha
    alpha_mod = ODE_system_alpha.Model()
    import ODE_system_beta
    beta_mod = ODE_system_beta.Model()
    # Choose 'global' best model (fits complete data optimally, not just dose-response data points)
    (best_model,best_gamma) = MAP(samplefile, beta, rho)#best_model is a dict with parameter values and best_gamma is a float
    # Make best_model into a list to compare to all model later
    best_model_list = [[variable_names[i], best_model[variable_names[i]]] for i in range(nVars) if variable_names[i] != 'gamma']

    alpha_curves=[]
    beta_curves=[]
    best_curves=[]
    # Simulate each model
    found_MAP_bool=False
    for r in range(sample_size):
        MAP_bool=False #Flag to identify the MAP model
        parameter_vector = samples.iloc[r]#Pandas Series of variables and values, including gamma
        # Create list format excluding gamma:
        pList = [[variable_names[i], parameter_vector.loc[variable_names[i]]] for i in range(nVars) if variable_names[i] != 'gamma']
        # Check if this model is the MAP model:
        if best_model_list == pList:
            MAP_bool=True
            found_MAP_bool=True
        if 'meanR' in [var[0] for var in pList]:
            meanR = pList[[var[0] for var in pList].index('meanR')][1]
        else:
            meanR = 2E3
        # Convert any combined quantities
        for item in range(len(pList)):
            if pList[item][0]=='delR':
                R1=meanR-pList[item][1]/2
                R2=meanR+pList[item][1]/2
                pList=pList[0:item]+[['R1',R1],['R2',R2]]+pList[item+1:len(pList)]
        gamma = parameter_vector.loc['gamma']#of type numpy.float64
        # Maintain detailed balance
        if 'kd4' in variable_names:
            q1 = 3.321155762205247e-14/1
            q2 = 4.98173364330787e-13/0.015
            q4 = 3.623188E-4/parameter_vector.loc['kd4']
            q3 = q2*q4/q1
            kd3 = 3.623188E-4/q3                

            q_1 = 4.98E-14/0.03
            q_2 = 8.30e-13/0.002
            q_4 = 3.623188e-4/parameter_vector.loc['k_d4']
            q_3 = q_2*q_4/q_1
            k_d3 = 3.623188e-4/q_3
            pList += [['kd3', kd3],['k_d3',k_d3]]
        # Now create an ordered list of ALL model parameters to use in PySB sim
        alpha_parameters=[]
        beta_parameters=[]
        for p in alpha_mod.parameters:
            isInList=False
            for y in pList:
                if p[0]==y[0]:
                    alpha_parameters.append(y[1])
                    isInList=True
                    break
            if isInList==False:
                alpha_parameters.append(p.value)
        for p in beta_mod.parameters:
            isInList=False
            for y in pList:
                if p[0]==y[0]:
                    beta_parameters.append(y[1])
                    isInList=True
                    break
            if isInList==False:
                beta_parameters.append(p.value)
        # Prepare for dose-response curve simulation
        I_index_Alpha = [el[0] for el in alpha_mod.parameters].index(dr_species[0])
        I_index_Beta = [el[0] for el in beta_mod.parameters].index(dr_species[0])
        NA = dr_species[1] # 6.022E23
        volEC = dr_species[2] # 1E-5   
        t=np.linspace(0,end_time)
        # Run simulation for each dose
        alpha_results=[]
        beta_results=[]
        for dose in doses:
            alpha_parameters[I_index_Alpha] = NA*volEC*dose
            (_, sim) = alpha_mod.simulate(t, param_values=alpha_parameters)
            alpha_results.append(gamma*sim[spec][-1])
            beta_parameters[I_index_Beta] = NA*volEC*dose
            (_, sim) = beta_mod.simulate(t, param_values=beta_parameters)
            beta_results.append(gamma*sim[spec][-1])     
        alpha_curves.append(alpha_results)
        beta_curves.append(beta_results)
        if MAP_bool==True:
            best_curves = [alpha_results,beta_results]


    # If this was the last model and MAP wasn't looked at, include MAP:
    if found_MAP_bool==False:
        pList = best_model_list
        if 'meanR' in [var[0] for var in pList]:
            meanR = pList[[var[0] for var in pList].index('meanR')][1]
        else:
            meanR = 2E3
        # Convert any combined quantities
        for item in range(len(pList)):
            if pList[item][0]=='delR':
                R1=meanR-pList[item][1]/2
                R2=meanR+pList[item][1]/2
                pList=pList[0:item]+[['R1',R1],['R2',R2]]+pList[item+1:len(pList)]
        gamma = parameter_vector.loc['gamma']#of type numpy.float64
        # Maintain detailed balance
        if 'kd4' in variable_names:
            q1 = 3.321155762205247e-14/1
            q2 = 4.98173364330787e-13/0.015
            q4 = 3.623188E-4/parameter_vector.loc['kd4']
            q3 = q2*q4/q1
            kd3 = 3.623188E-4/q3                

            q_1 = 4.98E-14/0.03
            q_2 = 8.30e-13/0.002
            q_4 = 3.623188e-4/parameter_vector.loc['k_d4']
            q_3 = q_2*q_4/q_1
            k_d3 = 3.623188e-4/q_3
            pList += [['kd3', kd3],['k_d3',k_d3]]
        # Now create an ordered list of ALL model parameters to use in PySB sim
        alpha_parameters=[]
        beta_parameters=[]
        for p in alpha_mod.parameters:
            isInList=False
            for y in pList:
                if p[0]==y[0]:
                    alpha_parameters.append(y[1])
                    isInList=True
                    break
            if isInList==False:
                alpha_parameters.append(p.value)
        for p in beta_mod.parameters:
            isInList=False
            for y in pList:
                if p[0]==y[0]:
                    beta_parameters.append(y[1])
                    isInList=True
                    break
            if isInList==False:
                beta_parameters.append(p.value)
        # Prepare for dose-response curve simulation
        I_index_Alpha = [el[0] for el in alpha_mod.parameters].index(dr_species[0])
        I_index_Beta = [el[0] for el in beta_mod.parameters].index(dr_species[0])
        NA = dr_species[1] # 6.022E23
        volEC = dr_species[2] # 1E-5   
        t=np.linspace(0,end_time)
        # Run simulation for each dose
        alpha_results=[]
        beta_results=[]
        for dose in doses:
            alpha_parameters[I_index_Alpha] = NA*volEC*dose
            (_, sim) = alpha_mod.simulate(t, param_values=alpha_parameters)
            alpha_results.append(gamma*sim[spec][-1])
            beta_parameters[I_index_Beta] = NA*volEC*dose
            (_, sim) = beta_mod.simulate(t, param_values=beta_parameters)
            beta_results.append(gamma*sim[spec][-1])     
        alpha_curves.append(alpha_results)
        beta_curves.append(beta_results)    
        best_curves = [alpha_results,beta_results]

    # Prepare function return
    prediction_intervals=[]
    #   Alpha results
    upper_error_prediction = np.percentile(alpha_curves, percent, axis=0)
    lower_error_prediction = np.percentile(alpha_curves, 100-percent, axis=0)
    prediction_intervals.append([best_curves[0],lower_error_prediction,upper_error_prediction,alpha_curves])
    
    #   Beta results
    upper_error_prediction = np.percentile(beta_curves, percent, axis=0)
    lower_error_prediction = np.percentile(beta_curves, 100-percent, axis=0)
    prediction_intervals.append([best_curves[1],lower_error_prediction,upper_error_prediction,beta_results])

    if suppress==False:
        fig, ax = plt.subplots()
        ax.set(xscale='log',yscale='linear') 
        ax.plot(doses, prediction_intervals[0][0], 'r')
        ax.plot(doses, prediction_intervals[0][1], 'r--')
        ax.plot(doses, prediction_intervals[0][2], 'r--')
        ax.plot(doses, prediction_intervals[1][0], 'g')
        ax.plot(doses, prediction_intervals[1][1], 'g--')
        ax.plot(doses, prediction_intervals[1][2], 'g--')
        plt.show()
    return prediction_intervals

