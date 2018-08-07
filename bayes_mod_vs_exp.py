# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 08:34:29 2018

@author: Duncan

Bayesian Model vs Experiment
"""
from mcmc import bayesian_timecourse
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

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; 
import pandas as pd
from pysb.export import export

plt.close('all')
sns.set_style("ticks")

gamma = 3.17
fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3)
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
ax2.set_ylabel('pSTAT1 Relative MFI',fontsize=18)
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
ax3.set_ylabel('pSTAT1 Relative MFI',fontsize=18)
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

# Calculate mean coefficient of determination
R2_sims = [[IFN_sims[0][0][0],IFN_sims[0][0][4],IFN_sims[0][0][12],IFN_sims[0][0][25],IFN_sims[0][0][49]],
           [IFN_sims[1][0][0],IFN_sims[1][0][4],IFN_sims[1][0][12],IFN_sims[1][0][25],IFN_sims[1][0][49]],
           [IFN_sims[2][0][0],IFN_sims[2][0][4],IFN_sims[2][0][12],IFN_sims[2][0][25],IFN_sims[2][0][49]],
           [IFN_sims[3][0][0],IFN_sims[3][0][4],IFN_sims[3][0][12],IFN_sims[3][0][25],IFN_sims[3][0][49]],
           [IFN_sims[4][0][0],IFN_sims[4][0][4],IFN_sims[4][0][12],IFN_sims[4][0][25],IFN_sims[4][0][49]],
           [IFN_sims[5][0][0],IFN_sims[5][0][4],IFN_sims[5][0][12],IFN_sims[5][0][25],IFN_sims[5][0][49]]]

SSres_mean=0
SStot_mean=0
for i in range(len(IFN_exps)):
    SSres_mean += np.sum(np.square(np.subtract(np.divide(IFN_exps[i],gamma),R2_sims[i][0])))
    SStot_mean += np.sum(np.square(np.subtract(np.divide(IFN_exps[i],gamma), np.average(np.divide(IFN_exps[i],gamma)))))
R2 = 1-SSres_mean/SStot_mean
print(r"The mean coefficient of determination is approximately "+str(R2))

