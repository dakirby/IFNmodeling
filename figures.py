# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 12:24:57 2018

@author: Duncan

Complete script to generate figures for IFN paper
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; 
import mcmc
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

IFN_sims = [*mcmc.bayesian_timecourse('posterior_samples.csv', 10E-12, 3600, 50, 95, ['TotalpSTAT'], suppress=True),
            *mcmc.bayesian_timecourse('posterior_samples.csv', 90E-12, 3600, 50, 95, ['TotalpSTAT'], suppress=True),
            *mcmc.bayesian_timecourse('posterior_samples.csv', 600E-12, 3600, 50, 95, ['TotalpSTAT'], suppress=True)]

import os
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Paper_Figures/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

plt.close('all')
sns.set_style("ticks")

fig3=False
fig4=False
fig5=True
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
    
    dr_curves = mcmc.bayesian_doseresponse('posterior_samples.csv', np.logspace(-13,np.log10(600E-12)), 3600, 50, 95, ['TotalpSTAT'])    
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

    dr60min = mcmc.bayesian_doseresponse('posterior_samples.csv', np.logspace(-14,-2), 3600, 50, 95, ['TotalpSTAT'])    
    ax2.set_title("Dose Response at 60 minutes", fontsize=20)
    ax2.set_ylabel('Total pSTAT Count',fontsize=18)
    ax2.set_xlabel('IFN Dose (M)',fontsize=18)
    ax2.set(xscale='log',yscale='linear')    
    ax2.plot(np.logspace(-14,-2), dr60min[0][0][0], 'r')
    #ax2.plot(np.logspace(-14,-2), dr60min[0][0][1], 'r--')
    #ax2.plot(np.logspace(-14,-2), dr60min[0][0][2], 'r--')    
    ax2.plot(np.logspace(-14,-2), dr60min[1][0][0], 'g')
    #ax2.plot(np.logspace(-14,-2), dr60min[1][0][1], 'g--')
    #ax2.plot(np.logspace(-14,-2), dr60min[1][0][2], 'g--')
    
    plt.savefig(results_dir+'figure4.pdf')

if fig5==True:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,8))
   

    dr60min = mcmc.bayesian_doseresponse('posterior_samples.csv', np.logspace(-14,-2), 3600, 50, 95, ['TotalpSTAT'])    
    ax2.set_title("Dose Response at 60 minutes", fontsize=20)
    ax2.set_ylabel('Total pSTAT Count',fontsize=18)
    ax2.set_xlabel('IFN Dose (M)',fontsize=18)
    ax2.set(xscale='log',yscale='linear')    
    ax2.plot(np.logspace(-14,-2), dr60min[0][0][0], 'r')
    #ax2.plot(np.logspace(-14,-2), dr60min[0][0][1], 'r--')
    #ax2.plot(np.logspace(-14,-2), dr60min[0][0][2], 'r--')    
    ax2.plot(np.logspace(-14,-2), dr60min[1][0][0], 'g')
    #ax2.plot(np.logspace(-14,-2), dr60min[1][0][1], 'g--')
    #ax2.plot(np.logspace(-14,-2), dr60min[1][0][2], 'g--')
    
    plt.savefig(results_dir+'figure4.pdf')
    

