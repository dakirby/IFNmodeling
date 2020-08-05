# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
import os
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Paper_Figures/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
import seaborn as sns
sns.set_style("ticks")
plt.close('all')

def get_params_list(fit_list):
    import ODE_system_alpha
    alpha_mod = ODE_system_alpha.Model()
    import ODE_system_beta
    beta_mod = ODE_system_beta.Model()
    # Build parameter lists
    if 'kd4' in [el[0] for el in fit_list]:
        kd4Index=[el[0] for el in fit_list].index('kd4')
        q1 = 3.321155762205247e-14/1
        q2 = 4.98173364330787e-13/0.015
        q4 = 3.623188E-4/fit_list[kd4Index][1]
        q3 = q2*q4/q1
        kd3 = 3.623188E-4/q3 
        fit_list.insert(kd4Index+1,['kd3',kd3])               
    if 'k_d4' in [el[0] for el in fit_list]:
        k_d4Index = [el[0] for el in fit_list].index('k_d4')
        q_1 = 4.98E-14/0.03
        q_2 = 8.30e-13/0.002
        q_4 = 3.623188e-4/fit_list[k_d4Index][1]
        q_3 = q_2*q_4/q_1
        k_d3 = 3.623188e-4/q_3
        fit_list.insert(k_d4Index+1,['k_d3',k_d3])               
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
    return alpha_parameters, beta_parameters
            
import Experimental_Data as ED 
nPost=14
pLimit=97.5
#posterior_filename = 'MCMC_Results-11-09-2018/posterior_samples.csv'
posterior_filename = 'MCMC_Results-25-09-2018/posterior_samples.csv'
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
# R_distribution (list) = [[R1_value,R2_value,weight],...]
# MAP_model (list) = list of variables in the form [['name',value],...]    
def population_DRsim(R_distribution, MAP_model,gamma=1):
    NA = 6.022E23
    volEC = 1E-5   
    all_alpha_sims=[[] for i in range(len(R_distribution))]
    all_beta_sims=[[] for i in range(len(R_distribution))]   
    weighted_alpha_sim=[]
    weighted_beta_sim=[]
    subpop_index=-1
    for subpop in R_distribution:
        subpop_index+=1
        R1=subpop[0]
        R2=subpop[1]
        weight=subpop[2]
        # Build unique subpopulation parameter list
        #   parse composite variables
        varnames = [el[0] for el in MAP_model]
        if 'R1' in varnames or 'R2' in varnames:
            R1ind = varnames.index('R1')
            R2ind = varnames.index('R2')
            MAP_model[R1ind][1]=R1
            MAP_model[R2ind][1]=R2
        elif 'meanR' in varnames or 'delR' in varnames:
            if 'meanR' in varnames: 
                meanRind = varnames.index('meanR')
                del MAP_model[meanRind]
            if 'delR' in varnames: 
                delRind = varnames.index('delR')
                del MAP_model[delRind]
            MAP_model += [['R1',R1],['R2',R2]]
        #   build parameter lists
        a, b = get_params_list(MAP_model)
         
        #   simulate dose-response        
        import ODE_system_alpha
        alpha_mod = ODE_system_alpha.Model()
        import ODE_system_beta
        beta_mod = ODE_system_beta.Model()
        #   get IFN index
        alpha_lookup=[]
        for p in alpha_mod.parameters:
            alpha_lookup.append(p[0])
        IFNinda=alpha_lookup.index('I')
        beta_lookup=[]
        for p in beta_mod.parameters:
            beta_lookup.append(p[0])
        IFNindb=beta_lookup.index('I')
        # Run dose-response
        for d in np.logspace(-14,-2):
            dose = NA*volEC*d
            a[IFNinda] = dose
            b[IFNindb] = dose
            (_, sim) = alpha_mod.simulate(np.linspace(0,3600), param_values=a)
            tc=np.multiply(gamma,sim['TotalpSTAT'])
            all_alpha_sims[subpop_index].append(tc[-1]*weight)        
            (_, sim) = beta_mod.simulate(np.linspace(0,3600), param_values=b)
            tc=np.multiply(gamma,sim['TotalpSTAT'])
            all_beta_sims[subpop_index].append(tc[-1]*weight)
        if weighted_alpha_sim==[]:
            weighted_alpha_sim = all_alpha_sims[subpop_index]
            weighted_beta_sim = all_beta_sims[subpop_index]
        else:
            weighted_alpha_sim = np.add(weighted_alpha_sim,all_alpha_sims[subpop_index])
            weighted_beta_sim = np.add(weighted_beta_sim,all_beta_sims[subpop_index])
    return weighted_alpha_sim, weighted_beta_sim, all_alpha_sims, all_beta_sims
            
MAP = [['kpa',1.09e-05],['kSOCSon',5.542E-06],['kd4',0.165],['k_d4',0.043],['delR',-2837.5]]  
gammaMAP=4#2.825
# =============================================================================
# IFN_sims = population_DRsim([[200,200,0.1],[400,800,0.1],[600,800,0.1],[2000,2200,0.2],
#                              [3000,3200,0.2],[4000,5000,0.1],[8000,800,0.1],[12000,12000,0.1]],
#                             MAP,gamma=gammaMAP)
# =============================================================================
IFN_sims = population_DRsim([])


fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,8))
plt.ion()
matplotlib.rcParams.update({'font.size': 18})
ax1.tick_params(labelsize=14)
ax1.set_title("Dose Response \nMultiple Subpopulations", fontsize=20)
ax1.set_ylabel('pSTAT1 Relative MFI',fontsize=18)
ax1.set_xlabel('IFN Dose (M)',fontsize=18)
ax1.set(xscale='log',yscale='linear')
ax2.tick_params(labelsize=14)
ax2.set_title("Dose Response \nMultiple Subpopulations", fontsize=20)
ax2.set_xlabel('IFN Dose (M)',fontsize=18)
ax2.set(xscale='log',yscale='linear')
# =============================================================================
# ax1.errorbar([10E-12,(90+1)*1E-12,600E-12],[IFN_exps[el][-1] for el in [0,2,4]],
#              yerr = [IFN_sigmas[el][-1] for el in [0,2,4]],
#              fmt='ro', label=r"Experiment IFN$\alpha$")
# ax1.errorbar([10E-12,90E-12,600E-12],[IFN_exps[el][-1] for el in [1,3,5]],
#              yerr = [IFN_sigmas[el][-1] for el in [1,3,5]],
#              fmt='go', label=r"Experiment IFN$\alpha$")
# =============================================================================

ax1.plot(np.logspace(-14,-2),IFN_sims[0], 'r')
ax2.plot(np.logspace(-14,-2),IFN_sims[1], 'g')
identity=0
palette = itertools.cycle(sns.color_palette("PuOr", len(IFN_sims[2])))
for curve in IFN_sims[2]:
    color_code = next(palette)
    ax1.plot(np.logspace(-14,-2),curve, 'r--',label=str(identity)+"_a",color = color_code)
    identity+=1
    ax1.legend()
identity=0
for curve in IFN_sims[3]:
    color_code = next(palette)
    ax2.plot(np.logspace(-14,-2),curve, 'g--',label=str(identity)+"_a",color = color_code)
    identity+=1
    ax2.legend()



plt.savefig(results_dir+'Rpop.pdf')
plt.show()	

