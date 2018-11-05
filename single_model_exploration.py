# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 08:32:40 2018

@author: Duncan

Exploring around the MAP model
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.close('all')
from pysb.export import export
import os
script_dir = os.path.dirname(__file__)
from MCMC_plotting import MAP, MAP_timecourse, MAP_doseresponse
import Experimental_Data as ED 
posterior_filename = '/MCMC_Results-25-09-2018/posterior_samples.csv'
MAP_priors={'R1':[100,12000,None,None],'R2':[100,12000,None,None],
              'kpa':[1.5E-8,10,np.log(1),4],'kSOCSon':[1.5E-11,0.07,np.log(1E-6),4],
              'k_d4':[4E-5,0.9,np.log(0.006),1.8],'kd4':[0.002,44,np.log(0.3),1.8]}
modelfiles = ['IFN_Models.IFN_alpha_altSOCS_Internalization_ppCompatible','IFN_Models.IFN_beta_altSOCS_Internalization_ppCompatible']
alpha_model = __import__(modelfiles[0],fromlist=['IFN_Models'])
py_output = export(alpha_model.model, 'python')
with open('ODE_system_alpha.py','w') as f:
    f.write(py_output)
beta_model = __import__(modelfiles[1],fromlist=['IFN_Models'])
py_output = export(beta_model.model, 'python')
with open('ODE_system_beta.py','w') as f:
    f.write(py_output)

# =============================================================================
# (best_model, best_gamma) = MAP(script_dir+posterior_filename, MAP_priors, 8, 1)
# print("Best model was:")
# print(best_model)
# print('best_gamma: {}'.format(best_gamma))
# =============================================================================
best_model={'kpa': 8.1947067581393088e-06, 'kSOCSon': 4.6695712179079518e-06, 
            'kd4': 0.21209196665402, 'k_d4': 0.18258140531363939, 'delR': 165.7137171779523,
            'meanR': 993.75557347853999}
best_gamma=2.253870867666794

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
NA=6.022E23
volEC=1E-5
IFN_sims = [*MAP_timecourse(best_model, best_gamma, 10E-12*NA*volEC, 'I', 3600, 'TotalpSTAT'),
            *MAP_timecourse(best_model, best_gamma, 90E-12*NA*volEC, 'I', 3600, 'TotalpSTAT'),
            *MAP_timecourse(best_model, best_gamma, 600E-12*NA*volEC, 'I', 3600, 'TotalpSTAT')]
refrac=True
# Plotting
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(15,8))
plt.ion()
matplotlib.rcParams.update({'font.size': 18})
ax1.tick_params(labelsize=14)
ax2.tick_params(labelsize=14)
ax3.tick_params(labelsize=14)
ax4.tick_params(labelsize=14)
ax5.tick_params(labelsize=14)
ax6.tick_params(labelsize=14)


jitter=50 # Scale the noise used to separate alpha and beta time points
Expt = ['0','5','15','30','60']
ax1.set_title("10 pM Time Course \nTheory vs Experiment", fontsize=20)
ax1.set_ylabel('pSTAT1 Relative MFI',fontsize=18)
ax1.set_xlabel('time (s)',fontsize=18)    
ax1.errorbar([int(el)*60+np.random.rand()*jitter for el in Expt],
            IFN_exps[0],
            yerr = IFN_sigmas[0],
            fmt='ro', label=r"Experiment IFN$\alpha$")
ax4.errorbar([int(el)*60 for el in Expt],
            IFN_exps[1],
            yerr = IFN_sigmas[1],
            fmt='go', label=r"Experiment IFN$\beta$")

ax1.plot(np.linspace(0,3600,num=len(IFN_sims[0])),IFN_sims[0], 'k')
ax4.plot(np.linspace(0,3600,num=len(IFN_sims[1])),IFN_sims[1], 'k')

ax2.set_title("90 pM Time Course \nTheory vs Experiment", fontsize=20)
ax2.set_xlabel('time (s)',fontsize=18)
ax2.errorbar([int(el)*60+np.random.rand()*jitter for el in Expt],
            IFN_exps[2],
            yerr = IFN_sigmas[2],
            fmt='ro', label=r"Experiment IFN$\alpha$")
ax5.errorbar([int(el)*60 for el in Expt],
            IFN_exps[3],
            yerr = IFN_sigmas[3],
            fmt='go', label=r"Experiment IFN$\beta$")
ax2.plot(np.linspace(0,3600,num=len(IFN_sims[2])),IFN_sims[2], 'k')
ax5.plot(np.linspace(0,3600,num=len(IFN_sims[3])),IFN_sims[3], 'k')
              
ax3.set_title("600 pM Time Course \nTheory vs Experiment", fontsize=20)
ax3.set_xlabel('time (s)',fontsize=18)
ax3.errorbar([int(el)*60+np.random.rand()*jitter for el in Expt],
            IFN_exps[4],
            yerr = IFN_sigmas[4],
            fmt='ro', label=r"Experiment IFN$\alpha$")
ax6.errorbar([int(el)*60 for el in Expt],
           IFN_exps[5],
            yerr = IFN_sigmas[5],
            fmt='go', label=r"Experiment IFN$\beta$")
ax3.plot(np.linspace(0,3600,num=len(IFN_sims[4])),IFN_sims[4], 'k')
ax6.plot(np.linspace(0,3600,num=len(IFN_sims[5])),IFN_sims[5], 'k')

plt.show()

if refrac==True:
    #best_model['k_d4']=0.06
    print(best_model)
    IFN_dr = MAP_doseresponse(best_model, best_gamma, np.logspace(-14,-2,num=50)*NA*volEC, 'I', 3600, 'TotalpSTAT')
    fac1=2
    fac2=15
    norm=False
    
    kd420_model = {key: best_model[key] for key in best_model.keys()}
    try:
        kd420_model['kd4'] = best_model['kd4']*fac1
    except KeyError:
        kd420_model={'kd4':0.3*fac1}
    try:
        kd420_model['k_d4'] = best_model['k_d4']*fac1
    except KeyError:
        kd420_model.update({'k_d4':0.006*fac1})
    K420 = MAP_doseresponse(kd420_model, best_gamma, np.logspace(-14,-2,num=50)*NA*volEC, 'I', 3600, 'TotalpSTAT')
    
    kd460_model = {key: best_model[key] for key in best_model.keys()}
    try:
        kd460_model['kd4'] = best_model['kd4']*fac2
    except KeyError:
        kd460_model={'kd4':0.3*fac2}
    try:
        kd460_model['k_d4'] = best_model['k_d4']*fac2
    except KeyError:
        kd460_model.update({'k_d4':0.006*fac2})    
    K460 = MAP_doseresponse(kd460_model, best_gamma, np.logspace(-14,-2,num=50)*NA*volEC, 'I', 3600, 'TotalpSTAT')
    
    if norm==True:# Normalization
        for each in range(len(K420)):
            K420[each] = np.divide(K420[each],IFN_dr[each])    
        for each in range(len(K460)):
            K460[each] = np.divide(K460[each],IFN_dr[each])
        
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,8))
    ax.set_title("Refractoriness by Increasing K4\n (JCB 2015 measured K4*15 for pre-stim cells)", fontsize=20)
    ax.set_ylabel('pSTAT Normalized by Unprimed pSTAT Count',fontsize=18)
    ax.set_xlabel('IFN (M)',fontsize=18)
    ax.set(xscale='log',yscale='linear')
    if norm==True:
        ax.plot(np.logspace(-14,-2,num=50), IFN_dr[0], 'r', label=r'IFN$\alpha$ K4', linewidth=2)
        ax.plot(np.logspace(-14,-2,num=50), IFN_dr[1], 'g', label=r'IFN$\beta$ K4', linewidth=2)    
        ax.plot(np.logspace(-14,-2,num=50), K420[0], 'r--', label=r'IFN$\alpha$ K4*{}'.format(fac1), linewidth=2)
        ax.plot(np.logspace(-14,-2,num=50), K420[1], 'g--', label=r'IFN$\beta$ K4*{}'.format(fac1), linewidth=2)
        ax.plot(np.logspace(-14,-2,num=50), K460[0], 'r:', label=r'IFN$\alpha$ K4*{}'.format(fac2), linewidth=2)
        ax.plot(np.logspace(-14,-2,num=50), K460[1], 'g:', label=r'IFN$\beta$ K4*{}'.format(fac2), linewidth=2)
    else:
        ax.plot(np.logspace(-14,-2,num=50), K420[0], 'r--', label=r'IFN$\alpha$ K4*{}'.format(fac1), linewidth=2)
        ax.plot(np.logspace(-14,-2,num=50), K420[1], 'g--', label=r'IFN$\beta$ K4*{}'.format(fac1), linewidth=2)
        ax.plot(np.logspace(-14,-2,num=50), K460[0], 'r', label=r'IFN$\alpha$ K4*{}'.format(fac2), linewidth=2)
        ax.plot(np.logspace(-14,-2,num=50), K460[1], 'g', label=r'IFN$\beta$ K4*{}'.format(fac2), linewidth=2)

    plt.legend()
    plt.show()

