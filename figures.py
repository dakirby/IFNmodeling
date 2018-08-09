# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 12:24:57 2018

@author: Duncan

Complete script to generate figures for IFN paper
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mcmc import bayesian_timecourse, bayesian_doseresponse
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

IFN_sims = [*bayesian_timecourse('posterior_samples.csv', 10E-12, 3600, 50, 95, ['TotalpSTAT'], suppress=True),
            *bayesian_timecourse('posterior_samples.csv', 90E-12, 3600, 50, 95, ['TotalpSTAT'], suppress=True),
            *bayesian_timecourse('posterior_samples.csv', 600E-12, 3600, 50, 95, ['TotalpSTAT'], suppress=True)]
import seaborn as sns
sns.set_style("ticks")
plt.close('all')

import os
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Paper_Figures/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


fig3=False
fig4=False
fig5=False
fig6=True
gamma = 3.17


if fig3==True:
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15,8))
    plt.ion()
    matplotlib.rcParams.update({'font.size': 18})
    ax1.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)
    ax3.tick_params(labelsize=14)
    
    jitter=50 # Scale the noise used to separate alpha and beta time points
    Expt = ['0','5','15','30','60']
    ax1.set_title("10 pM Time Course \nTheory vs Experiment", fontsize=20)
    ax1.set_ylabel('pSTAT1 Relative MFI',fontsize=18)
    ax1.set_xlabel('time (s)',fontsize=18)    
    ax1.errorbar([int(el)*60+np.random.rand()*jitter for el in Expt],
                np.divide(IFN_exps[0],gamma),
                yerr = np.divide(IFN_sigmas[0],gamma),
                fmt='ro', label=r"Experiment IFN$\alpha$")
    ax1.errorbar([int(el)*60 for el in Expt],
                np.divide(IFN_exps[1],gamma),
                yerr = np.divide(IFN_sigmas[1],gamma),
                fmt='go', label=r"Experiment IFN$\beta$")
    ax1.plot(np.linspace(0,3600),IFN_sims[0][0], 'k')
    ax1.plot(np.linspace(0,3600),IFN_sims[0][1], 'k--')
    ax1.plot(np.linspace(0,3600),IFN_sims[0][2], 'k--')
    ax1.plot(np.linspace(0,3600),IFN_sims[1][0], 'k')
    ax1.plot(np.linspace(0,3600),IFN_sims[1][1], 'k:')
    ax1.plot(np.linspace(0,3600),IFN_sims[1][2], 'k:')
    
    
    
    ax2.set_title("90 pM Time Course \nTheory vs Experiment", fontsize=20)
    ax2.set_xlabel('time (s)',fontsize=18)
    ax2.errorbar([int(el)*60+np.random.rand()*jitter for el in Expt],
                np.divide(IFN_exps[2],gamma),
                yerr = np.divide(IFN_sigmas[2],gamma),
                fmt='ro', label=r"Experiment IFN$\alpha$")
    ax2.errorbar([int(el)*60 for el in Expt],
                np.divide(IFN_exps[3],gamma),
                yerr = np.divide(IFN_sigmas[3],gamma),
                fmt='go', label=r"Experiment IFN$\beta$")
    ax2.plot(np.linspace(0,3600),IFN_sims[2][0], 'k')
    ax2.plot(np.linspace(0,3600),IFN_sims[2][1], 'k--')
    ax2.plot(np.linspace(0,3600),IFN_sims[2][2], 'k--')
    ax2.plot(np.linspace(0,3600),IFN_sims[3][0], 'k')
    ax2.plot(np.linspace(0,3600),IFN_sims[3][1], 'k:')
    ax2.plot(np.linspace(0,3600),IFN_sims[3][2], 'k:')
        
            
    ax3.set_title("600 pM Time Course \nTheory vs Experiment", fontsize=20)
    ax3.set_xlabel('time (s)',fontsize=18)
    ax3.errorbar([int(el)*60+np.random.rand()*jitter for el in Expt],
                np.divide(IFN_exps[4],gamma),
                yerr = np.divide(IFN_sigmas[4],gamma),
                fmt='ro', label=r"Experiment IFN$\alpha$")
    ax3.errorbar([int(el)*60 for el in Expt],
                np.divide(IFN_exps[5],gamma),
                yerr = np.divide(IFN_sigmas[5],gamma),
                fmt='go', label=r"Experiment IFN$\beta$")
    ax3.plot(np.linspace(0,3600),IFN_sims[4][0], 'k')
    ax3.plot(np.linspace(0,3600),IFN_sims[4][1], 'k--')
    ax3.plot(np.linspace(0,3600),IFN_sims[4][2], 'k--')
    ax3.plot(np.linspace(0,3600),IFN_sims[5][0], 'k')
    ax3.plot(np.linspace(0,3600),IFN_sims[5][1], 'k:')
    ax3.plot(np.linspace(0,3600),IFN_sims[5][2], 'k:')
    
    plt.savefig(results_dir+'figure3.pdf')
    
if fig4==True:
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,8))
    matplotlib.rcParams.update({'font.size': 18})
    
    dr_curves = bayesian_doseresponse('posterior_samples.csv', np.logspace(-13,np.log10(600E-12)), 3600, 50, 95, ['TotalpSTAT'])    
    ax1.set_title("Dose Response \nTheory vs Experiment", fontsize=20)
    ax1.set_ylabel('pSTAT1 Relative MFI',fontsize=18)
    ax1.set_xlabel('IFN Dose (M)',fontsize=18)
    ax1.set(xscale='log',yscale='linear')    
    ax1.errorbar([10*1E-12,(90+1)*1E-12,600*1E-12],np.divide([IFN_exps[el][-1] for el in [0,2,4]],gamma),
                 yerr = np.divide([IFN_sigmas[el][-1] for el in [0,2,4]],gamma),
                    fmt='ro', label=r"Experiment IFN$\alpha$")
    ax1.errorbar([10*1E-12,90*1E-12,600*1E-12],np.divide([IFN_exps[el][-1] for el in [1,3,5]],gamma),
                 yerr = np.divide([IFN_sigmas[el][-1] for el in [1,3,5]],gamma),
                    fmt='go', label=r"Experiment IFN$\alpha$")
    ax1.plot(np.logspace(-13,np.log10(600E-12)), dr_curves[0][0][0], 'r')
    ax1.plot(np.logspace(-13,np.log10(600E-12)), dr_curves[0][0][1], 'r--')
    ax1.plot(np.logspace(-13,np.log10(600E-12)), dr_curves[0][0][2], 'r--')             
    ax1.plot(np.logspace(-13,np.log10(600E-12)), dr_curves[1][0][0], 'g')
    ax1.plot(np.logspace(-13,np.log10(600E-12)), dr_curves[1][0][1], 'g--')
    ax1.plot(np.logspace(-13,np.log10(600E-12)), dr_curves[1][0][2], 'g--')     

    dr60min = bayesian_doseresponse('posterior_samples.csv', np.logspace(-14,-2), 3600, 50, 95, ['TotalpSTAT'])    
    ax2.set_title("Dose Response at 60 minutes", fontsize=20)
    ax2.set_ylabel('Total pSTAT Count',fontsize=18)
    ax2.set_xlabel('IFN Dose (M)',fontsize=18)
    ax2.set(xscale='log',yscale='linear')    
    ax2.plot(np.logspace(-14,-2), dr60min[0][0][0], 'r', linewidth=2)
    #ax2.plot(np.logspace(-14,-2), dr60min[0][0][1], 'r--')
    #ax2.plot(np.logspace(-14,-2), dr60min[0][0][2], 'r--')    
    ax2.plot(np.logspace(-14,-2), dr60min[1][0][0], 'g', linewidth=2)
    #ax2.plot(np.logspace(-14,-2), dr60min[1][0][1], 'g--')
    #ax2.plot(np.logspace(-14,-2), dr60min[1][0][2], 'g--')
    
    plt.savefig(results_dir+'figure4.pdf')

if fig5==True:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,8))
    dr5min = bayesian_doseresponse('posterior_samples.csv', np.logspace(-14,-2), 5*60, 50, 95, ['TotalpSTAT'])    
    dr15min = bayesian_doseresponse('posterior_samples.csv', np.logspace(-14,-2), 15*60, 50, 95, ['TotalpSTAT'])        
    dr30min = bayesian_doseresponse('posterior_samples.csv', np.logspace(-14,-2), 30*60, 50, 95, ['TotalpSTAT'])    
    ax.set_title("Dose Response at Different Times", fontsize=20)
    ax.set_ylabel('pSTAT Normalized by Total STAT',fontsize=18)
    ax.set_xlabel('IFN Dose (M)',fontsize=18)
    ax.set(xscale='log',yscale='linear')    
    ax.plot(np.logspace(-14,-2), np.divide(dr5min[0][0][0],1E4), 'r', label=r'IFN$\alpha$ 5 min', linewidth=2)
    ax.plot(np.logspace(-14,-2), np.divide(dr5min[1][0][0],1E4), 'g', label=r'IFN$\beta$ 5 min', linewidth=2)
    ax.plot(np.logspace(-14,-2), np.divide(dr15min[0][0][0],1E4), 'r--', label=r'IFN$\alpha$ 15 min', linewidth=2)
    ax.plot(np.logspace(-14,-2), np.divide(dr15min[1][0][0],1E4), 'g--', label=r'IFN$\beta$ 15 min', linewidth=2)
    ax.plot(np.logspace(-14,-2), np.divide(dr30min[0][0][0],1E4), 'r:', label=r'IFN$\alpha$ 30 min', linewidth=2)
    ax.plot(np.logspace(-14,-2), np.divide(dr30min[1][0][0],1E4), 'g:', label=r'IFN$\beta$ 30 min', linewidth=2)
    plt.legend()
    plt.savefig(results_dir+'figure5.pdf')
    
if fig6==True:
    fig, ax = plt.subplots()
    



def rad_cell_point(samplefile, radius, end_time, sample_size, percent, specList, 
                        suppress=False, dose_species=['I', 6.022E23, 1E-5]):
    import pandas as pd
    samples = pd.read_csv(samplefile)
    (nSamples, nVars) = samples.shape
    if sample_size > nSamples:
        print("Not enough samples in file")
        return 1
    variable_names = list(samples.columns.values)

    import ODE_system_alpha
    alpha_mod = ODE_system_alpha.Model()
    import ODE_system_beta
    beta_mod = ODE_system_beta.Model()
    
    alpha_results=[]
    beta_results=[]
    for r in range(sample_size):
        parameter_vector = samples.iloc[r]
        pList = [[variable_names[i], parameter_vector.loc[variable_names[i]]] for i in range(nVars) if variable_names[i] != 'gamma']
        # Build the sample model
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
    
   
        alpha_parameters=[]
        beta_parameters=[]
        for p in alpha_mod.parameters:
            isInList=False
            for y in pList:
                if p[0]==y[0]:
                    if y[0]=='R1' or y[0]=='R2':
                        alpha_parameters.append(y[1]*(2*radius**2 + radius*(8E-6)*4)/2.76e-09)
                    alpha_parameters.append(y[1])
                    isInList=True
                    break
            if isInList==False:
                # catch S model parameter and scale it
                if p[0]=='S':
                    alpha_parameters.append(p.value*(2*radius**2 + radius*(8E-6)*4)/2.76e-09)
                else:
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
        I_index_Alpha = [el[0] for el in alpha_mod.parameters].index(dose_species[0])
        I_index_Beta = [el[0] for el in beta_mod.parameters].index(dose_species[0])
        
        NA = dose_species[1] # 1
        volEC = dose_species[2] # 1   
        t=np.linspace(0,end_time)
        # Run simulation
        alpha_parameters[I_index_Alpha] = NA*volEC*radius
        (_, sim) = alpha_mod.simulate(t, param_values=alpha_parameters)
        alpha_results.append(sim)
        beta_parameters[I_index_Beta] = NA*volEC*radius
        (_, sim) = beta_mod.simulate(t, param_values=beta_parameters)
        beta_results.append(sim)
    prediction_intervals=[]
    for spec in specList:
        tcs = [j[spec] for j in alpha_results]
        mean_prediction = np.mean(tcs, axis=0)
        upper_error_prediction = np.percentile(tcs, max(percent, 100-percent), axis=0)
        lower_error_prediction = np.percentile(tcs, min(percent, 100-percent), axis=0)
        prediction_intervals.append([mean_prediction,lower_error_prediction,upper_error_prediction])
        
        tcs = [j[spec] for j in beta_results]
        mean_prediction = np.mean(tcs, axis=0)
        upper_error_prediction = np.percentile(tcs, max(percent, 100-percent), axis=0)
        lower_error_prediction = np.percentile(tcs, min(percent, 100-percent), axis=0)
        prediction_intervals.append([mean_prediction,lower_error_prediction,upper_error_prediction])
    
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
# rad_cell_dr() runs a 'dose response' curve for each sample from posterior parameter 
# distribution, giving prediction intervals for all dose points. However, the 'dose'
# is the cell radius and then S and R model parameters scale accordingly.    
# Inputs:
#     samplefile (str) = the name of the posterior samples file output from 
#                         Markov Chain Monte Carlo simulations
#     radii (list) = radii for the simulation (measured in meters)
#     end_time (int) = the end time for each time course (in seconds)
#     sample_size (int) = the number of posterior samples to use
#     specList (list) = list of names of species to predict intervals for
#     percent (int) = the percentile bounds for error in model prediction 
#                       (bounds will be 'percent' and 100-'percent') 
#     suppress (Boolean) = whether or not to plot the results (default is False) 
#     dr_species (list) = any model observable can be used; second and third list itmes are 
#                           multiplicative factors. If not needed then set to 1
#                           default is ['rad_cell' for cell radius, 1, 1] 
#     modelfiles = ['IFN_alpha_altSOCS_ppCompatible','IFN_beta_altSOCS_ppCompatible']
# Returns
#   [alpha_responses, beta_responses] (list) = the dose response curves
#                       alpha_responses = [[mean curve, low curve, high curve] for each species]        
# =============================================================================
def rad_cell_dr(samplefile, radii, end_time, sample_size, percent, specList,
                          suppress=False, dr_species=['rad_cell', 1,1],
                          modelfiles = ['IFN_alpha_altSOCS_ppCompatible','IFN_beta_altSOCS_ppCompatible']):
    from pysb.export import export
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
    
    alpha_responses = [[] for i in range(len(specList))]
    beta_responses = [[] for i in range(len(specList))]
    for r in radii:
        courses = rad_cell_point(samplefile, r, end_time, sample_size, percent, specList, 
                                 suppress=True, dose_species=dr_species)
        # courses = [[IFNa spec 1], [IFNb spec 1], [IFNa spec 2], [IFNb spec 2]]
        #           [IFNa spec 1] = [[mean], [low], [high]]
        courses = [[l[-1] for l in s] for s in courses]
        # courses = [[mean dose, low, high]_IFNaS1, [mean dose, low, high]_IFNbS1, ...
        for i in range(len(specList)):
            alpha_responses[i].append(courses[i*2])
            beta_responses[i].append(courses[i*2+1])
    alpha_responses = [[[el[0] for el in alpha_responses[s]],[el[1] for el in alpha_responses[s]],[el[2] for el in alpha_responses[s]]] for s in range(len(alpha_responses))]
    beta_responses =  [[[el[0] for el in beta_responses[s]], [el[1] for el in beta_responses[s]], [el[2] for el in beta_responses[s]]] for s in range(len(alpha_responses))]
    return [alpha_responses, beta_responses]

