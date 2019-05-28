# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 22:17:42 2018

@author: Duncan

Testing out split populations
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
import copy
from scipy.optimize import minimize
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

def split_plot_map_timecourse(map_model, t, dose):
    # Import models
    import ODE_system_alpha
    alpha_mod = ODE_system_alpha.Model()
    import ODE_system_beta
    beta_mod = ODE_system_beta.Model()
    variable_names = list(map_model.keys())
    nVars = len(variable_names)
    best_model_list = [[variable_names[i], map_model[variable_names[i]]] for i in range(nVars) if variable_names[i] != 'gamma']
    pList = best_model_list 
    if 'meanR' in variable_names:
        meanR = map_model['meanR']
    else:
        meanR = 2E3
    # Convert any combined quantities
    for item in range(len(pList)):
        if pList[item][0]=='delR':
            R1=meanR-pList[item][1]/2
            R2=meanR+pList[item][1]/2
            pList=pList[0:item]+[['R1',R1],['R2',R2]]+pList[item+1:len(pList)]
    gamma = map_model['gamma']
    # Maintain detailed balance
    if 'kd4' in variable_names:
        q1 = 3.321155762205247e-14/1
        q2 = 4.98173364330787e-13/0.015
        q4 = 3.623188E-4/map_model['kd4']
        q3 = q2*q4/q1
        kd3 = 3.623188E-4/q3                
    
        q_1 = 4.98E-14/0.03
        q_2 = 8.30e-13/0.002
        q_4 = 3.623188e-4/map_model['k_d4']
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
    I_index_Alpha = [el[0] for el in alpha_mod.parameters].index('I')
    I_index_Beta = [el[0] for el in beta_mod.parameters].index('I')
    NA = 6.022E23
    volEC = 1E-5   
    # Run simulation for each dose
    alpha_results=[]
    beta_results=[]
    
    alpha_parameters[I_index_Alpha] = NA*volEC*dose
    (_, sim) = alpha_mod.simulate(t, param_values=alpha_parameters)
    alpha_results = gamma*sim['TotalpSTAT']
    beta_parameters[I_index_Beta] = NA*volEC*dose
    (_, sim) = beta_mod.simulate(t, param_values=beta_parameters)
    beta_results = gamma*sim['TotalpSTAT']
    
    return [alpha_results,beta_results]

R_multiplier = 1     
map_model_sim = {'kpa': 1.2135284902663905e-05, 'kSOCSon': 1.5277346700134648e-06, 
             'kd4': 0.86270257143922802, 'k_d4': 0.69197453010925514, 
             'delR': -708.59379577612503, 'meanR': 1884.2691783436514,
             'gamma': 1.215728709237476}
exp1_pop1 = split_plot_map_timecourse(map_model_sim,np.linspace(0,3600),10E-12)
exp2_pop1 = split_plot_map_timecourse(map_model_sim,np.linspace(0,3600),90E-12)
exp3_pop1 = split_plot_map_timecourse(map_model_sim,np.linspace(0,3600),600E-12)
map_model_sim2 = {'kpa': 1.2135284902663905e-05, 'kSOCSon': 1.5277346700134648e-06, 
             'kd4': 0.86270257143922802, 'k_d4': 0.69197453010925514, 
             'delR': -708.59379577612503, 'meanR': 1884.2691783436514*R_multiplier,
             'gamma': 1.215728709237476}
exp1_pop2 = split_plot_map_timecourse(map_model_sim2,np.linspace(0,3600),10E-12)
exp2_pop2 = split_plot_map_timecourse(map_model_sim2,np.linspace(0,3600),90E-12)
exp3_pop2 = split_plot_map_timecourse(map_model_sim2,np.linspace(0,3600),600E-12)
IFN_sims = [np.add(np.multiply(0.8,exp1_pop1[0]),np.multiply(0.2,exp1_pop2[0])),
            np.add(np.multiply(0.8,exp1_pop1[1]),np.multiply(0.2,exp1_pop2[1])),
            np.add(np.multiply(0.8,exp2_pop1[0]),np.multiply(0.2,exp2_pop2[0])),
            np.add(np.multiply(0.8,exp2_pop1[1]),np.multiply(0.2,exp2_pop2[1])),
            np.add(np.multiply(0.8,exp3_pop1[0]),np.multiply(0.2,exp3_pop2[0])),
            np.add(np.multiply(0.8,exp3_pop1[1]),np.multiply(0.2,exp3_pop2[1]))]


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


plt.savefig(results_dir+'altFigure3.pdf')
plt.show()	


# Score the choice of map_1 and map_2
def score_mod(rfac,map1):
    map2 = copy.deepcopy(map1)
    map2['meanR']=map1['meanR']*rfac[0]
    s1 = split_plot_map_timecourse(map1,[0,300,900,1800,3600],10E-12)
    s2 = split_plot_map_timecourse(map1,[0,300,900,1800,3600],90E-12)
    s3 = split_plot_map_timecourse(map1,[0,300,900,1800,3600],600E-12)
    s4 = split_plot_map_timecourse(map2,[0,300,900,1800,3600],10E-12)
    s5 = split_plot_map_timecourse(map2,[0,300,900,1800,3600],90E-12)
    s6 = split_plot_map_timecourse(map2,[0,300,900,1800,3600],600E-12)
    score_sims = [np.add(np.multiply(0.8,s1[0]),np.multiply(0.2,s4[0])),
                np.add(np.multiply(0.8,s1[1]),np.multiply(0.2,s4[1])),
                np.add(np.multiply(0.8,s2[0]),np.multiply(0.2,s5[0])),
                np.add(np.multiply(0.8,s2[1]),np.multiply(0.2,s5[1])),
                np.add(np.multiply(0.8,s3[0]),np.multiply(0.2,s6[0])),
                np.add(np.multiply(0.8,s3[1]),np.multiply(0.2,s6[1]))]
    sum_square_res = 0
    for i in range(len(score_sims)):
        sum_square_res += np.sum(np.square(np.divide(np.subtract(score_sims[i],IFN_exps[i]),IFN_sigmas[i])))
    #print("Sum of square residuals is {}".format(sum_square_res))
    return sum_square_res
#score_mod(R_multiplier,map_model_sim)
opt = minimize(score_mod,R_multiplier,args=(map_model_sim))
R_multiplier_optimal = opt['x'].item()
logp = opt['fun']
print("The optimal score is {} using an R-factor = {}".format(logp,R_multiplier_optimal))

