# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 12:24:57 2018

@author: Duncan

Complete script to generate figures for IFN paper
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from MCMC_plotting import bayesian_timecourse, bayesian_doseresponse
import os
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Paper_Figures/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
import seaborn as sns
sns.set_style("ticks")
plt.close('all')

import Experimental_Data as ED 
nPost=14#35
pLimit=97.5
# Best fit to Sagar's data: 25-09-2018
posterior_filename = 'MCMC_Results-25-09-2018/posterior_samples.csv'
priors_dict={'R1':[100,12000,None,None],'R2':[100,12000,None,None],
              'kpa':[1.5E-8,10,np.log(1),4],'kSOCSon':[1.5E-11,0.07,np.log(1E-6),4],
              'k_d4':[4E-5,0.9,np.log(0.006),1.8],'kd4':[0.002,44,np.log(0.3),1.8]}
modelfiles = ['IFN_Models.IFN_alpha_altSOCS_ppCompatible','IFN_Models.IFN_beta_altSOCS_ppCompatible']
# =============================================================================
# # Detailed model
# posterior_filename = 'MCMC_Results-03-10-2018/posterior_samples.csv'
# priors_dict={'R1':[100,12000,None,None],'R2':[100,12000,None,None],
#              'kpa':[1.5E-8,10,np.log(1),4],'kSOCSon':[1.5E-11,0.07,np.log(1E-6),4],
#              'k_d4':[4E-5,0.9,np.log(0.006),1.8],'kd4':[0.002,44,np.log(0.3),1.8],
#              'kSTATbinding':[1E-11,1,np.log(1E-6),4],'kloc':[1E-5,10,np.log(1.25E-3),4],
#              'kSOCSmRNA':[1E-7,10,np.log(1E-3),4],'mRNAdeg':[5E-8,10,np.log(5E-4),4],
#              'mRNAtrans':[1E-7,10,np.log(1E-3),4],'kSOCS':[5E-7,10,np.log(5E-3),4]}
# modelfiles = ['IFN_Models.IFN_detailed_model_alpha_ppCompatible','IFN_Models.IFN_detailed_model_beta_ppCompatible']
# =============================================================================
# =============================================================================
# # Internalization sim
# posterior_filename = 'MCMC_Results-15-10-2018/posterior_samples.csv' 
# priors_dict={'R1':[100,12000,None,None],'R2':[100,12000,None,None],
#              'kpa':[1.5E-9,1,np.log(1),4],'kSOCSon':[1.5E-11,0.07,np.log(1E-6),4],
#              'k_d4':[4E-5,0.9,np.log(0.006),1.8],'kd4':[0.002,44,np.log(0.3),1.8],
#              'kIntBasal_r1':[1E-7,1E-1,None,None],'kIntBasal_r2':[2E-7,2E-1,None,None],
#         'kint_IFN':[5E-7,5E-1,None,None],'krec_a1':[3E-7,3E-1,None,None],'krec_a2':[5E-6,5E0,None,None],
#         'krec_b1':[1E-7,1E-1,None,None],'krec_b2':[1E-6,1E0,None,None]}
# modelfiles = ['IFN_Models.IFN_alpha_altSOCS_Internalization_ppCompatible','IFN_Models.IFN_beta_altSOCS_Internalization_ppCompatible']
# =============================================================================
# =============================================================================
# # Limited Internalization Sim
# posterior_filename = 'MCMC_Results-03-11-2018\\Reanalysis\\posterior_sample_reanalysis.csv'
# priors_dict={'R1':[100,12000,None,None],'R2':[100,12000,None,None],
#              'kSOCSon':[1.5E-11,0.07,np.log(1E-6),4],
#              'k_d4':[4E-5,0.9,np.log(0.006),1.8],
#              'krec_a1':[3E-7,3E-1,None,None],'krec_a2':[5E-6,5E0,None,None],
#              'krec_b1':[1E-7,1E-1,None,None],'krec_b2':[1E-6,1E0,None,None]}
# modelfiles = ['IFN_Models.IFN_alpha_altSOCS_Internalization_ppCompatible','IFN_Models.IFN_beta_altSOCS_Internalization_ppCompatible']
# =============================================================================


# Make sure modelfile is up to date
# Write modelfiles
from pysb.export import export
alpha_model = __import__(modelfiles[0],fromlist=['IFN_Models'])
py_output = export(alpha_model.model, 'python')
with open('ODE_system_alpha.py','w') as f:
    f.write(py_output)
beta_model = __import__(modelfiles[1],fromlist=['IFN_Models'])
py_output = export(beta_model.model, 'python')
with open('ODE_system_beta.py','w') as f:
    f.write(py_output)
    
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

IFN_sims = [*bayesian_timecourse(posterior_filename, 10E-12, 3600, nPost, pLimit, 'TotalpSTAT',priors_dict,8,1, suppress=True),
            *bayesian_timecourse(posterior_filename, 90E-12, 3600, nPost, pLimit, 'TotalpSTAT',priors_dict,8,1, suppress=True),
            *bayesian_timecourse(posterior_filename, 600E-12, 3600, nPost, pLimit, 'TotalpSTAT',priors_dict,8,1, suppress=True)]

with open(results_dir+"IFN_sims.txt",'w') as f:
    f.write(str([IFN_sims[i][5] for i in range(len(IFN_sims))]))

fig3=False
altFig3=False
fig4=False
altfig4=True
fig5=False
fig6=False
fig7_1=False
fig7_2=False
fig8=False

if fig3==True:
    print("Fig 3")
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
                IFN_exps[0],
                yerr = IFN_sigmas[0],
                fmt='ro', label=r"Experiment IFN$\alpha$")
    ax1.errorbar([int(el)*60 for el in Expt],
                IFN_exps[1],
                yerr = IFN_sigmas[1],
                fmt='go', label=r"Experiment IFN$\beta$")

    ax1.plot(np.linspace(0,3600,num=len(IFN_sims[0][0])),IFN_sims[0][0], 'k')
    ax1.plot(np.linspace(0,3600,num=len(IFN_sims[0][1])),IFN_sims[0][1], 'k--')
    ax1.plot(np.linspace(0,3600,num=len(IFN_sims[0][2])),IFN_sims[0][2], 'k--')
    ax1.plot(np.linspace(0,3600,num=len(IFN_sims[1][0])),IFN_sims[1][0], 'k')
    ax1.plot(np.linspace(0,3600,num=len(IFN_sims[1][1])),IFN_sims[1][1], 'k:')
    ax1.plot(np.linspace(0,3600,num=len(IFN_sims[1][2])),IFN_sims[1][2], 'k:')
    
    ax2.set_title("90 pM Time Course \nTheory vs Experiment", fontsize=20)
    ax2.set_xlabel('time (s)',fontsize=18)
    ax2.errorbar([int(el)*60+np.random.rand()*jitter for el in Expt],
                IFN_exps[2],
                yerr = IFN_sigmas[2],
                fmt='ro', label=r"Experiment IFN$\alpha$")
    ax2.errorbar([int(el)*60 for el in Expt],
                IFN_exps[3],
                yerr = IFN_sigmas[3],
                fmt='go', label=r"Experiment IFN$\beta$")
    ax2.plot(np.linspace(0,3600,num=len(IFN_sims[2][0])),IFN_sims[2][0], 'k')
    ax2.plot(np.linspace(0,3600,num=len(IFN_sims[2][1])),IFN_sims[2][1], 'k--')
    ax2.plot(np.linspace(0,3600,num=len(IFN_sims[2][2])),IFN_sims[2][2], 'k--')
    ax2.plot(np.linspace(0,3600,num=len(IFN_sims[3][0])),IFN_sims[3][0], 'k')
    ax2.plot(np.linspace(0,3600,num=len(IFN_sims[3][1])),IFN_sims[3][1], 'k:')
    ax2.plot(np.linspace(0,3600,num=len(IFN_sims[3][2])),IFN_sims[3][2], 'k:')
                  
    ax3.set_title("600 pM Time Course \nTheory vs Experiment", fontsize=20)
    ax3.set_xlabel('time (s)',fontsize=18)
    ax3.errorbar([int(el)*60+np.random.rand()*jitter for el in Expt],
                IFN_exps[4],
                yerr = IFN_sigmas[4],
                fmt='ro', label=r"Experiment IFN$\alpha$")
    ax3.errorbar([int(el)*60 for el in Expt],
               IFN_exps[5],
                yerr = IFN_sigmas[5],
                fmt='go', label=r"Experiment IFN$\beta$")
    ax3.plot(np.linspace(0,3600,num=len(IFN_sims[4][0])),IFN_sims[4][0], 'k')
    ax3.plot(np.linspace(0,3600,num=len(IFN_sims[4][1])),IFN_sims[4][1], 'k--')
    ax3.plot(np.linspace(0,3600,num=len(IFN_sims[4][2])),IFN_sims[4][2], 'k--')
    ax3.plot(np.linspace(0,3600,num=len(IFN_sims[5][0])),IFN_sims[5][0], 'k')
    ax3.plot(np.linspace(0,3600,num=len(IFN_sims[5][1])),IFN_sims[5][1], 'k:')
    ax3.plot(np.linspace(0,3600,num=len(IFN_sims[5][2])),IFN_sims[5][2], 'k:')
    
    plt.savefig(results_dir+'figure3.pdf')
    plt.show()	
    
if fig4==True:
    print("Fig 4")
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,8))
    matplotlib.rcParams.update({'font.size': 18})

    dr_curves = bayesian_doseresponse(posterior_filename, np.logspace(-13,np.log10(600E-12)), 3600, nPost, pLimit, 'TotalpSTAT',priors_dict,8,1,suppress=True)    
    ax1.set_title("Dose Response \nTheory vs Experiment", fontsize=20)
    ax1.set_ylabel('pSTAT1 Relative MFI',fontsize=18)
    ax1.set_xlabel('IFN Dose (M)',fontsize=18)
    ax1.set(xscale='log',yscale='linear')    
    ax1.errorbar([10E-12,(90+1)*1E-12,600E-12],[IFN_exps[el][-1] for el in [0,2,4]],
                 yerr = [IFN_sigmas[el][-1] for el in [0,2,4]],
                    fmt='ro', label=r"Experiment IFN$\alpha$")
    ax1.errorbar([10E-12,90E-12,600E-12],[IFN_exps[el][-1] for el in [1,3,5]],
                 yerr = [IFN_sigmas[el][-1] for el in [1,3,5]],
                    fmt='go', label=r"Experiment IFN$\alpha$")
    ax1.plot(np.logspace(-13,np.log10(600E-12)), dr_curves[0][0], 'r')
    ax1.plot(np.logspace(-13,np.log10(600E-12)), dr_curves[0][1], 'r--')
    ax1.plot(np.logspace(-13,np.log10(600E-12)), dr_curves[0][2], 'r--')             
    ax1.plot(np.logspace(-13,np.log10(600E-12)), dr_curves[1][0], 'g')
    ax1.plot(np.logspace(-13,np.log10(600E-12)), dr_curves[1][1], 'g--')
    ax1.plot(np.logspace(-13,np.log10(600E-12)), dr_curves[1][2], 'g--')     

    dr60min = bayesian_doseresponse(posterior_filename, np.logspace(-14,-2), 3600, nPost, pLimit, 'TotalpSTAT',priors_dict,8,1,suppress=True)    
    ax2.set_title("Dose Response at 60 minutes", fontsize=20)
    ax2.set_ylabel('Total pSTAT Count',fontsize=18)
    ax2.set_xlabel('IFN Dose (M)',fontsize=18)
    ax2.set(xscale='log',yscale='linear')    
    ax2.plot(np.logspace(-14,-2), dr60min[0][0], 'r', linewidth=2)
    #ax2.plot(np.logspace(-14,-2), dr60min[0][1], 'r--')
    #ax2.plot(np.logspace(-14,-2), dr60min[0][2], 'r--')    
    ax2.plot(np.logspace(-14,-2), dr60min[1][0], 'g', linewidth=2)
    #ax2.plot(np.logspace(-14,-2), dr60min[1][1], 'g--')
    #ax2.plot(np.logspace(-14,-2), dr60min[1][2], 'g--')
    
    plt.savefig(results_dir+'figure4.pdf')
    plt.show()
    

if fig5==True:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,8))
    dr5min = bayesian_doseresponse(posterior_filename, np.logspace(-14,-2), 5*60, nPost,pLimit, 'TotalpSTAT',priors_dict,8,1,suppress=True)    
    dr15min = bayesian_doseresponse(posterior_filename, np.logspace(-14,-2), 15*60, nPost, pLimit, 'TotalpSTAT',priors_dict,8,1,suppress=True)        
    dr30min = bayesian_doseresponse(posterior_filename, np.logspace(-14,-2), 30*60, nPost, pLimit, 'TotalpSTAT',priors_dict,8,1,suppress=True)    
    ax.set_title("Dose Response at Different Times", fontsize=20)
    ax.set_ylabel('pSTAT Normalized by Total STAT',fontsize=18)
    ax.set_xlabel('IFN Dose (M)',fontsize=18)
    ax.set(xscale='log',yscale='linear')    
    ax.plot(np.logspace(-14,-2), np.divide(dr5min[0][0],1E4), 'r', label=r'IFN$\alpha$ 5 min', linewidth=2)
    ax.plot(np.logspace(-14,-2), np.divide(dr5min[1][0],1E4), 'g', label=r'IFN$\beta$ 5 min', linewidth=2)
    ax.plot(np.logspace(-14,-2), np.divide(dr15min[0][0],1E4), 'r--', label=r'IFN$\alpha$ 15 min', linewidth=2)
    ax.plot(np.logspace(-14,-2), np.divide(dr15min[1][0],1E4), 'g--', label=r'IFN$\beta$ 15 min', linewidth=2)
    ax.plot(np.logspace(-14,-2), np.divide(dr30min[0][0],1E4), 'r:', label=r'IFN$\alpha$ 30 min', linewidth=2)
    ax.plot(np.logspace(-14,-2), np.divide(dr30min[1][0],1E4), 'g:', label=r'IFN$\beta$ 30 min', linewidth=2)
    plt.legend()
    plt.savefig(results_dir+'figure5.pdf')
    plt.show()
    

if fig7_1==True:
    import pysbplotlib as pyplt
    import IFN_alpha_altSOCS as IFNaSOCS
    import IFN_beta_altSOCS as IFNbSOCS
    sample_size=nPost
    import pandas as pd
    samples = pd.read_csv(posterior_filename, index_col=0)
    variable_names = list(samples.columns.values)
    (nSamples, nVars) = samples.shape
    curve_population20 = [[],[]]
    curve_population60 = [[],[]]    
    for r in range(sample_size):
        parameter_vector = samples.iloc[r]    
        pList = [[variable_names[i], parameter_vector.loc[variable_names[i]]] for i in range(nVars) if variable_names[i] != 'gamma']

        k4_dictA = {}
        k420_dictA = {}
        k460_dictA = {}

        k4_dictB = {}
        k420_dictB = {}
        k460_dictB = {}
        for p in pList:
            if p[0]=='kd4':
                k4_dictA.update({p[0]:p[1]})
                k420_dictA.update({p[0]:p[1]*20})
                k460_dictA.update({p[0]:p[1]*60})                
            elif p[0]=='k_d4':
                k4_dictB.update({p[0]:p[1]})
                k420_dictB.update({p[0]:p[1]*20})
                k460_dictB.update({p[0]:p[1]*60})                
            else:
                k4_dictA.update({p[0]:p[1]})
                k420_dictA.update({p[0]:p[1]})
                k460_dictA.update({p[0]:p[1]})  
                
                k4_dictB.update({p[0]:p[1]})
                k420_dictB.update({p[0]:p[1]})
                k460_dictB.update({p[0]:p[1]})                
                
        normA = pyplt.doseresponse(IFNaSOCS, ['IFN',np.logspace(-14,-2,num=50)], np.linspace(0,3600,num=200),
                         [['TotalpSTAT',"Total pSTAT"]], parameters=k4_dictA, suppress=True)
        normB = pyplt.doseresponse(IFNbSOCS, ['IFN',np.logspace(-14,-2,num=50)], np.linspace(0,3600,num=200),
                         [['TotalpSTAT',"Total pSTAT"]], parameters=k4_dictB, suppress=True)
        A20 = pyplt.doseresponse(IFNaSOCS, ['IFN',np.logspace(-14,-2,num=50)], np.linspace(0,3600,num=200),
                         [['TotalpSTAT',"Total pSTAT"]], parameters=k420_dictA, suppress=True)
        B20 = pyplt.doseresponse(IFNbSOCS, ['IFN',np.logspace(-14,-2,num=50)], np.linspace(0,3600,num=200),
                         [['TotalpSTAT',"Total pSTAT"]], parameters=k420_dictB, suppress=True)
        A60 = pyplt.doseresponse(IFNaSOCS, ['IFN',np.logspace(-14,-2,num=50)], np.linspace(0,3600,num=200),
                         [['TotalpSTAT',"Total pSTAT"]], parameters=k460_dictA, suppress=True)
        B60 = pyplt.doseresponse(IFNbSOCS, ['IFN',np.logspace(-14,-2,num=50)], np.linspace(0,3600,num=200),
                         [['TotalpSTAT',"Total pSTAT"]], parameters=k460_dictB, suppress=True)
        
        curve_population20[0].append(np.divide(A20,normA)[0])
        curve_population60[0].append(np.divide(A60,normA)[0])
        curve_population20[1].append(np.divide(B20,normB)[0])
        curve_population60[1].append(np.divide(B60,normB)[0])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,8))
    ax.set_title("Refractoriness by Increasing K4", fontsize=20)
    ax.set_ylabel('pSTAT Normalized by Unprimed pSTAT Count',fontsize=18)
    ax.set_xlabel('IFN (M)',fontsize=18)
    ax.set(xscale='log',yscale='linear')
    ax.plot(np.logspace(-14,-2,num=50), np.mean(curve_population20[0], axis=0), 'r', label=r'IFN$\alpha$ K4*20', linewidth=2)
    ax.plot(np.logspace(-14,-2,num=50), np.mean(curve_population60[0], axis=0), 'r--', label=r'IFN$\alpha$ K4*60', linewidth=2)
    ax.plot(np.logspace(-14,-2,num=50), np.mean(curve_population20[1], axis=0), 'g', label=r'IFN$\beta$ K4*20', linewidth=2)
    ax.plot(np.logspace(-14,-2,num=50), np.mean(curve_population60[1], axis=0), 'g--', label=r'IFN$\beta$ K4*60', linewidth=2)
    plt.legend()
    plt.savefig(results_dir+'figure7_1.pdf')
    plt.show()
    
if fig7_2==True:
    import pysbplotlib as pyplt
    import IFN_alpha_altSOCS as IFNaSOCS
    import IFN_beta_altSOCS as IFNbSOCS
    sample_size=nPost
    import pandas as pd
    samples = pd.read_csv(posterior_filename, index_col=0)
    variable_names = list(samples.columns.values)
    (nSamples, nVars) = samples.shape
    curve_population15 = [[],[]]
    curve_population60 = [[],[]]    
    for r in range(sample_size):
        parameter_vector = samples.iloc[r]    
        pList = [[variable_names[i], parameter_vector.loc[variable_names[i]]] for i in range(nVars) if variable_names[i] != 'gamma']

        k4_dictA = {}
        k415_dictA = {'fracUSP18':0.6, 'USP18modfac':15}
        k460_dictA = {'fracUSP18':0.6, 'USP18modfac':60}

        k4_dictB = {}
        k415_dictB = {'fracUSP18':0.6, 'USP18modfac':15}
        k460_dictB = {'fracUSP18':0.6, 'USP18modfac':60}
        for p in pList:
            if p[0]=='kd4':
                k4_dictA.update({p[0]:p[1]})
                k415_dictA.update({p[0]:p[1]})
                k460_dictA.update({p[0]:p[1]})                
            elif p[0]=='k_d4':
                k4_dictB.update({p[0]:p[1]})
                k415_dictB.update({p[0]:p[1]})
                k460_dictB.update({p[0]:p[1]})                
            else:
                k4_dictA.update({p[0]:p[1]})
                k415_dictA.update({p[0]:p[1]})
                k460_dictA.update({p[0]:p[1]})  
                
                k4_dictB.update({p[0]:p[1]})
                k415_dictB.update({p[0]:p[1]})
                k460_dictB.update({p[0]:p[1]})                
                
        normA = pyplt.doseresponse(IFNaSOCS, ['IFN',np.logspace(-14,-2,num=50)], np.linspace(0,3600,num=200),
                         [['TotalpSTAT',"Total pSTAT"]], parameters=k4_dictA, suppress=True)
        normB = pyplt.doseresponse(IFNbSOCS, ['IFN',np.logspace(-14,-2,num=50)], np.linspace(0,3600,num=200),
                         [['TotalpSTAT',"Total pSTAT"]], parameters=k4_dictB, suppress=True)
        A15 = pyplt.doseresponse(IFNaSOCS, ['IFN',np.logspace(-14,-2,num=50)], np.linspace(0,3600,num=200),
                         [['TotalpSTAT',"Total pSTAT"]], parameters=k415_dictA, suppress=True)
        B15 = pyplt.doseresponse(IFNbSOCS, ['IFN',np.logspace(-14,-2,num=50)], np.linspace(0,3600,num=200),
                         [['TotalpSTAT',"Total pSTAT"]], parameters=k415_dictB, suppress=True)
        A60 = pyplt.doseresponse(IFNaSOCS, ['IFN',np.logspace(-14,-2,num=50)], np.linspace(0,3600,num=200),
                         [['TotalpSTAT',"Total pSTAT"]], parameters=k460_dictA, suppress=True)
        B60 = pyplt.doseresponse(IFNbSOCS, ['IFN',np.logspace(-14,-2,num=50)], np.linspace(0,3600,num=200),
                         [['TotalpSTAT',"Total pSTAT"]], parameters=k460_dictB, suppress=True)
        
        curve_population15[0].append(np.divide(A20,normA)[0])
        curve_population60[0].append(np.divide(A60,normA)[0])
        curve_population15[1].append(np.divide(B20,normB)[0])
        curve_population60[1].append(np.divide(B60,normB)[0])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,8))
    ax.set_title("Refractoriness by Modelling USP18", fontsize=20)
    ax.set_ylabel('pSTAT Normalized by Unprimed pSTAT Count',fontsize=18)
    ax.set_xlabel('IFN (M)',fontsize=18)
    ax.set(xscale='log',yscale='linear')
    ax.plot(np.logspace(-14,-2,num=50), np.mean(curve_population15[0], axis=0), 'r', label=r'IFN$\alpha$ K4*20', linewidth=2)
    ax.plot(np.logspace(-14,-2,num=50), np.mean(curve_population60[0], axis=0), 'r--', label=r'IFN$\alpha$ K4*60', linewidth=2)
    ax.plot(np.logspace(-14,-2,num=50), np.mean(curve_population15[1], axis=0), 'g', label=r'IFN$\beta$ K4*20', linewidth=2)
    ax.plot(np.logspace(-14,-2,num=50), np.mean(curve_population60[1], axis=0), 'g--', label=r'IFN$\beta$ K4*60', linewidth=2)
    plt.legend()
    plt.savefig(results_dir+'figure7_2.pdf')
    plt.show()
    
if fig8==True:
    print("Fig 8")
    import pysbplotlib as pyplt
    import IFN_alpha_altSOCS as IFNaSOCS
    import IFN_beta_altSOCS as IFNbSOCS
    sample_size=nPost
    import pandas as pd
    samples = pd.read_csv(posterior_filename, index_col=0)
    variable_names = list(samples.columns.values)
    (nSamples, nVars) = samples.shape
    dr_populationNoInt = [[],[]]
    dr_populationInt = [[],[]]  
    tc_populations = [[],[],[],[]]
    for r in range(sample_size):
        parameter_vector = samples.iloc[r]    
        pList = [[variable_names[i], parameter_vector.loc[variable_names[i]]] for i in range(nVars) if variable_names[i] != 'gamma']

        dictA = {}
        dictAInt = {'Internalization_switch':1}

        dictB = {}
        dictBInt = {'Internalization_switch':1}

        for p in pList:
            if p[0]=='kd4':
                dictA.update({p[0]:p[1]})
                dictAInt.update({p[0]:p[1]})
            elif p[0]=='k_d4':
                dictB.update({p[0]:p[1]})
                dictBInt.update({p[0]:p[1]})
            else:
                dictA.update({p[0]:p[1]})
                dictAInt.update({p[0]:p[1]})
                
                dictB.update({p[0]:p[1]})
                dictBInt.update({p[0]:p[1]})
                
        A = pyplt.doseresponse(IFNaSOCS, ['IFN',np.logspace(-14,-2,num=50)], np.linspace(0,3600,num=200),
                         [['TotalpSTAT',"Total pSTAT"]], parameters=dictA, suppress=True)
        AInt = pyplt.doseresponse(IFNaSOCS, ['IFN',np.logspace(-14,-2,num=50)], np.linspace(0,3600,num=200),
                         [['TotalpSTAT',"Total pSTAT"]], parameters=dictAInt, suppress=True)
        B = pyplt.doseresponse(IFNbSOCS, ['IFN',np.logspace(-14,-2,num=50)], np.linspace(0,3600,num=200),
                         [['TotalpSTAT',"Total pSTAT"]], parameters=dictB, suppress=True)
        BInt = pyplt.doseresponse(IFNbSOCS, ['IFN',np.logspace(-14,-2,num=50)], np.linspace(0,3600,num=200),
                         [['TotalpSTAT',"Total pSTAT"]], parameters=dictBInt, suppress=True)
        Atc = pyplt.timecourse(IFNaSOCS, np.linspace(0,3600,num=200), [['Free_R1',"Free_R1"],['Free_R1',"Free_R1"]],
                               suppress=True, parameters=dictAInt)
        Btc = pyplt.timecourse(IFNbSOCS, np.linspace(0,3600,num=200), [['Free_R1',"Free_R1"],['Free_R1',"Free_R1"]],
                               suppress=True, parameters=dictBInt)
        
        dr_populationNoInt[0].append(A)
        dr_populationNoInt[1].append(B)
        dr_populationInt[0].append(AInt)
        dr_populationInt[1].append(BInt)
        tc_populations[0].append(Atc['Free_R1'])
        tc_populations[1].append(Btc['Free_R1'])
        tc_populations[2].append(Atc['Free_R2'])
        tc_populations[3].append(Btc['Free_R2'])
        
        
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,8))
    axes[1][1].set_off()
    axes[0][0].set_title(r"IFN$\alpha$ Receptor Internalization", fontsize=20)
    axes[0][1].set_title(r"IFN$\beta$ Receptor Internalization", fontsize=20)    
    axes[1][0].set_title("pSTAT Reduced by Internalization", fontsize=20)
    axes[0][0].set_ylabel('Receptor Count',fontsize=18)
    axes[1][0].set_ylabel('pSTAT Normalized by Total STAT',fontsize=18)
    axes[0][0].set_xlabel('Time (s)',fontsize=18)
    axes[0][1].set_xlabel('Time (s)',fontsize=18)    
    axes[1][0].set_xlabel('IFN (M)', fontsize=18)
    axes[1][0].set(xscale='log',yscale='linear')

    axes[0][0].plot(np.linspace(0, 3600), np.mean(tc_populations[0], axis=0), 'xkcd:orange', label='R1', linewidth=2)
    axes[0][1].plot(np.linspace(0, 3600), np.mean(tc_populations[1], axis=0), 'xkcd:orange', label='R1', linewidth=2)
    axes[0][0].plot(np.linspace(0, 3600), np.mean(tc_populations[2], axis=0), 'xkcd:blue', label='R2', linewidth=2)
    axes[0][1].plot(np.linspace(0, 3600), np.mean(tc_populations[3], axis=0), 'xkcd:blue', label='R2', linewidth=2)

    axes[1][0].plot(np.logspace(-14,-2,num=50), np.mean(dr_populationNoInt[0], axis=0), 'r', label=r'IFN$\alpha$ No Int', linewidth=2)
    axes[1][0].plot(np.logspace(-14,-2,num=50), np.mean(dr_populationNoInt[1], axis=0), 'g', label=r'IFN$\beta$ No Int', linewidth=2)
    axes[1][0].plot(np.logspace(-14,-2,num=50), np.mean(dr_populationInt[0], axis=0), 'r--', label=r'IFN$\alpha$ Int', linewidth=2)
    axes[1][0].plot(np.logspace(-14,-2,num=50), np.mean(dr_populationInt[1], axis=0), 'g--', label=r'IFN$\beta$ Int', linewidth=2)

    plt.legend()
    plt.savefig(results_dir+'figure8.pdf')
    plt.show()

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
                    if y[0]=='R1' or y[0]=='R2': # catch 'R1' and 'R2' and scale them
                        alpha_parameters.append(y[1]*(2*radius**2 + radius*(8E-6)*4)/2.76e-09)
                    else:
                        alpha_parameters.append(y[1])
                    isInList=True
                    break
            if isInList==False:
                # catch S model parameter and scale it
                if p[0]=='S':
                    alpha_parameters.append(1E4*(8E-6)*radius**2/7.2e-15)
                elif p[0]=='I': # catch I model parameter, which should be 1.5 nM
                    alpha_parameters.append(1.5E-9*6.022E23*1E-5)
                else:
                    alpha_parameters.append(p.value)
        for p in beta_mod.parameters:
            isInList=False
            for y in pList:
                if p[0]==y[0]:
                    if y[0]=='R1' or y[0]=='R2': # catch 'R1' and 'R2' and scale them
                        beta_parameters.append(y[1]*(2*radius**2 + radius*(8E-6)*4)/2.76e-09)
                    else:
                        beta_parameters.append(y[1])
                    isInList=True
                    break
            if isInList==False:
                # catch S model parameter and scale it
                if p[0]=='S':
                    beta_parameters.append(1E4*(8E-6)*radius**2/7.2e-15)
                elif p[0]=='I': # catch I model parameter, which should be 1.5 nM
                    beta_parameters.append(1.5E-9*6.022E23*1E-5)
                else:
                    beta_parameters.append(p.value)
        I_index_Alpha = [el[0] for el in alpha_mod.parameters].index(dose_species[0])
        I_index_Beta = [el[0] for el in beta_mod.parameters].index(dose_species[0])
        
        NA = dose_species[1] # 1
        volEC = dose_species[2] # 1   
        t=np.linspace(0,end_time, num=200)
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

if fig6==True:
    fig, ax = plt.subplots()
    radCell60min = rad_cell_dr(posterior_filename, np.logspace(-8,-3), 3600, nPost, 97.5, ['TotalpSTAT'])                          
    ax.set_title("Cell Size for 60 minute Exposure", fontsize=20)
    ax.set_ylabel('pSTAT Normalized by Total STAT',fontsize=18)
    ax.set_xlabel('Cell Radius (microns)',fontsize=18)
    ax.set(xscale='log',yscale='linear')
    Snorm = np.divide(np.multiply(1E4*8E-6,np.square(np.logspace(-8,-3))),7.2e-15)
    normalized_alpha_response = np.divide(radCell60min[0][0][0],Snorm) 
    normalized_beta_response = np.divide(radCell60min[1][0][0],Snorm) 

    ax.plot(np.logspace(-2,3)[10:-2], normalized_alpha_response[10:-2], 'r', linewidth=2)
    ax.plot(np.logspace(-2,3)[10:-2], normalized_beta_response[10:-2], 'g', linewidth=2)
    
    plt.savefig(results_dir+'figure6.pdf')
    plt.show()
    
if altFig3==True:
    print("Fig 3")
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

    ax1.plot(np.linspace(0,3600,num=len(IFN_sims[0][0])),IFN_sims[0][0], 'k')
    ax1.plot(np.linspace(0,3600,num=len(IFN_sims[0][1])),IFN_sims[0][1], 'k--')
    ax1.plot(np.linspace(0,3600,num=len(IFN_sims[0][2])),IFN_sims[0][2], 'k--')
    ax4.plot(np.linspace(0,3600,num=len(IFN_sims[1][0])),IFN_sims[1][0], 'k')
    ax4.plot(np.linspace(0,3600,num=len(IFN_sims[1][1])),IFN_sims[1][1], 'k:')
    ax4.plot(np.linspace(0,3600,num=len(IFN_sims[1][2])),IFN_sims[1][2], 'k:')
    
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
    ax2.plot(np.linspace(0,3600,num=len(IFN_sims[2][0])),IFN_sims[2][0], 'k')
    ax2.plot(np.linspace(0,3600,num=len(IFN_sims[2][1])),IFN_sims[2][1], 'k--')
    ax2.plot(np.linspace(0,3600,num=len(IFN_sims[2][2])),IFN_sims[2][2], 'k--')
    ax5.plot(np.linspace(0,3600,num=len(IFN_sims[3][0])),IFN_sims[3][0], 'k')
    ax5.plot(np.linspace(0,3600,num=len(IFN_sims[3][1])),IFN_sims[3][1], 'k:')
    ax5.plot(np.linspace(0,3600,num=len(IFN_sims[3][2])),IFN_sims[3][2], 'k:')
                  
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
    ax3.plot(np.linspace(0,3600,num=len(IFN_sims[4][0])),IFN_sims[4][0], 'k')
    ax3.plot(np.linspace(0,3600,num=len(IFN_sims[4][1])),IFN_sims[4][1], 'k--')
    ax3.plot(np.linspace(0,3600,num=len(IFN_sims[4][2])),IFN_sims[4][2], 'k--')
    ax6.plot(np.linspace(0,3600,num=len(IFN_sims[5][0])),IFN_sims[5][0], 'k')
    ax6.plot(np.linspace(0,3600,num=len(IFN_sims[5][1])),IFN_sims[5][1], 'k:')
    ax6.plot(np.linspace(0,3600,num=len(IFN_sims[5][2])),IFN_sims[5][2], 'k:')
    
    plt.savefig(results_dir+'altFigure3.pdf')
    plt.show()	
    
if altfig4==True:
    for end_time_plot in [15*60,30*60,60*60]:
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,8))
        matplotlib.rcParams.update({'font.size': 18})
    
        dr_curves = bayesian_doseresponse(posterior_filename, np.logspace(-13,np.log10(600E-12)), end_time_plot, nPost, pLimit, 'TotalpSTAT',priors_dict,8,1,suppress=True)    
        ax1.set_title("Dose Response \nTheory vs Experiment", fontsize=20)
        ax1.set_ylabel('pSTAT1 Relative MFI',fontsize=18)
        ax1.set_xlabel('IFN Dose (M)',fontsize=18)
        ax1.set(xscale='log',yscale='linear')    
        ax1.errorbar([10E-12,(90+1)*1E-12,600E-12],[IFN_exps[el][-1] for el in [0,2,4]],
                     yerr = [IFN_sigmas[el][-1] for el in [0,2,4]],
                        fmt='ro', label=r"Experiment IFN$\alpha$")
        ax1.errorbar([10E-12,90E-12,600E-12],[IFN_exps[el][-1] for el in [1,3,5]],
                     yerr = [IFN_sigmas[el][-1] for el in [1,3,5]],
                        fmt='go', label=r"Experiment IFN$\alpha$")
        ax1.plot(np.logspace(-13,np.log10(600E-12)), dr_curves[0][0], 'r')
        ax1.plot(np.logspace(-13,np.log10(600E-12)), dr_curves[0][1], 'r--')
        ax1.plot(np.logspace(-13,np.log10(600E-12)), dr_curves[0][2], 'r--')             
        ax1.plot(np.logspace(-13,np.log10(600E-12)), dr_curves[1][0], 'g')
        ax1.plot(np.logspace(-13,np.log10(600E-12)), dr_curves[1][1], 'g--')
        ax1.plot(np.logspace(-13,np.log10(600E-12)), dr_curves[1][2], 'g--')     
    
        dr60min = bayesian_doseresponse(posterior_filename, np.logspace(-14,-2), end_time_plot, nPost, pLimit, 'TotalpSTAT',priors_dict,8,1,suppress=True)    
        ax2.set_title("Best Fit Dose Response at {} minutes".format(end_time_plot/60), fontsize=20)
        ax2.set_ylabel('Total pSTAT Count',fontsize=18)
        ax2.set_xlabel('IFN Dose (M)',fontsize=18)
        ax2.set(xscale='log',yscale='linear')    
        ax2.plot(np.logspace(-14,-2), dr60min[0][0], 'r', linewidth=2)
        #ax2.plot(np.logspace(-14,-2), dr60min[0][1], 'r--')
        #ax2.plot(np.logspace(-14,-2), dr60min[0][2], 'r--')    
        ax2.plot(np.logspace(-14,-2), dr60min[1][0], 'g', linewidth=2)
        #ax2.plot(np.logspace(-14,-2), dr60min[1][1], 'g--')
        #ax2.plot(np.logspace(-14,-2), dr60min[1][2], 'g--')
    
        plt.savefig(results_dir+'altfig4_{}.pdf'.format(str(end_time_plot)))
    plt.show()    
    