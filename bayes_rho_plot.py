# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 20:59:56 2018

@author: Duncan

Bayesian plot
"""

import pysb_parallel as pp
import matplotlib.pyplot as plt
import Experimental_Data as ED
import numpy as np
import IFN_alpha_altSOCS as IFNaSOCS
import IFN_beta_altSOCS as IFNbSOCS
import pysbplotlib as pyplt

def main():
    plt.close('all')
    Expt = ['0','5','15','30','60']
    priors = [1E-6,1E-6,2E3,2E3,0.3,0.006]
    rho_vals=np.logspace(-3,1,num=9) # Relaxation from prior distribution
    points=[]
    for index in range(len(rho_vals)):
        rho=rho_vals[index]
        N=25 #Number of models to sample
        posteriors = pp.fit_IFN_model_Bayesian(['IFN_alpha_altSOCS_ppCompatible','IFN_beta_altSOCS_ppCompatible'], 
                         [['kpa',1E-6,1E-7,9E-6,'linear'],['kSOCSon',1E-6,1E-7,9E-6,'log'],['k4',1,0.1,100,'log'],
    					 ['gamma',3.5,2,10,'linear'],['R1',2000,100,10000,'linear'],['R2',2000,100,10000,'linear']],
                         [['kpa',1E-6],['kSOCSon',1E-6],['kd4',0.3],['k_d4',0.006],['R1',2E3],['R2',2E3]],
                         rho, N,cpu=1)
        
        kpaa = posteriors[0][1]   
        kSOCSona = posteriors[1][1]   
        R1a = posteriors[2][1]   
        R2a = posteriors[3][1]   
        gamma = posteriors[4][1]
        gammaA=gamma
        gammaB=gamma
        kd4 = posteriors[5][1]   
        
        kpab = kpaa # 
        kSOCSonb = kSOCSona #
        R1b = R1a
        R2b = R2a
        k_d4 = posteriors[6][1]   
        
    
        all_sigmas = [np.divide(ED.data.loc[(ED.data.loc[:,'Dose (pM)']==10) & (ED.data.loc[:,'Interferon']=="Alpha_std"),Expt].values[0],gammaA),
                      np.divide(ED.data.loc[(ED.data.loc[:,'Dose (pM)']==10) & (ED.data.loc[:,'Interferon']=="Beta_std"),Expt].values[0],gammaB),
                      np.divide(ED.data.loc[(ED.data.loc[:,'Dose (pM)']==90) & (ED.data.loc[:,'Interferon']=="Alpha_std"),Expt].values[0],gammaA),
                      np.divide(ED.data.loc[(ED.data.loc[:,'Dose (pM)']==90) & (ED.data.loc[:,'Interferon']=="Beta_std"),Expt].values[0],gammaB),
                      np.divide(ED.data.loc[(ED.data.loc[:,'Dose (pM)']==600) & (ED.data.loc[:,'Interferon']=="Alpha_std"),Expt].values[0],gammaA),
                      np.divide(ED.data.loc[(ED.data.loc[:,'Dose (pM)']==600) & (ED.data.loc[:,'Interferon']=="Beta_std"),Expt].values[0],gammaB)]
        
        pm10A = pyplt.timecourse(IFNaSOCS, [0,5*60,15*60,30*60,60*60], [['TotalpSTAT',"Total pSTAT"]], suppress=True,
                                       parameters = {'R1':R1a, 'R2':R2a,
                                                      'IFN':10E-12, 'kSOCSon':kSOCSona, 'kpa':kpaa, 'kd4':kd4})
        pm10B = pyplt.timecourse(IFNbSOCS, [0,5*60,15*60,30*60,60*60], [['TotalpSTAT',"Total pSTAT"]], suppress=True,
                                       parameters = {'R1':R1b, 'R2':R2b,
                                                      'IFN':10E-12,'kSOCSon':kSOCSonb, 'kpa':kpab, 'k_d4':k_d4})
        pm90A = pyplt.timecourse(IFNaSOCS, [0,5*60,15*60,30*60,60*60], [['TotalpSTAT',"Total pSTAT"]], suppress=True,
                                       parameters = {'R1':R1a, 'R2':R2a,
                                                      'IFN':90E-12, 'kSOCSon':kSOCSona, 'kpa':kpaa, 'kd4':kd4})
        pm90B = pyplt.timecourse(IFNbSOCS, [0,5*60,15*60,30*60,60*60], [['TotalpSTAT',"Total pSTAT"]], suppress=True,
                                       parameters = {'R1':R1b, 'R2':R2b,
                                                      'IFN':90E-12,'kSOCSon':kSOCSonb, 'kpa':kpab, 'k_d4':k_d4})
        pm600A = pyplt.timecourse(IFNaSOCS, [0,5*60,15*60,30*60,60*60], [['TotalpSTAT',"Total pSTAT"]], suppress=True,
                                       parameters = {'R1':R1a, 'R2':R2a,
                                                      'IFN':600E-12,'kSOCSon':kSOCSona, 'kpa':kpaa, 'kd4':kd4})
        pm600B = pyplt.timecourse(IFNbSOCS, [0,5*60,15*60,30*60,60*60],  [['TotalpSTAT',"Total pSTAT"]], suppress=True,
                                       parameters = {'R1':R1b, 'R2':R2b,
                                                      'IFN':600E-12,'kSOCSon':kSOCSonb, 'kpa':kpab, 'k_d4':k_d4})
        xdata10a = ED.data.loc[(ED.data.loc[:,'Dose (pM)']==10) & (ED.data.loc[:,'Interferon']=="Alpha"),Expt].values[0]   
        #   beta
        xdata10b = ED.data.loc[(ED.data.loc[:,'Dose (pM)']==10) & (ED.data.loc[:,'Interferon']=="Beta"),Expt].values[0]
    
        # For the 90 pM dose in Figure 3B
        # alpha
        xdata90a = ED.data.loc[(ED.data.loc[:,'Dose (pM)']==90) & (ED.data.loc[:,'Interferon']=="Alpha"),Expt].values[0]
        # beta
        xdata90b = ED.data.loc[(ED.data.loc[:,'Dose (pM)']==90) & (ED.data.loc[:,'Interferon']=="Beta"),Expt].values[0]
    
    
        # For the 600 pM dose in Figure 3B
        # alpha
        xdata600a = ED.data.loc[(ED.data.loc[:,'Dose (pM)']==600) & (ED.data.loc[:,'Interferon']=="Alpha"),Expt].values[0]
        # beta
        xdata600b = ED.data.loc[(ED.data.loc[:,'Dose (pM)']==600) & (ED.data.loc[:,'Interferon']=="Beta"),Expt].values[0]
    
        all_exps = [np.divide(xdata10a,gammaA),np.divide(xdata10b,gammaB),
                    np.divide(xdata90a,gammaA),np.divide(xdata90b,gammaB),
                    np.divide(xdata600a,gammaA),np.divide(xdata600b,gammaB)]
        all_sims = [pm10A["TotalpSTAT"], pm10B["TotalpSTAT"], 
                    pm90A["TotalpSTAT"], pm90B["TotalpSTAT"],
                    pm600A["TotalpSTAT"], pm600B["TotalpSTAT"]]
        
        xi2 = np.sum(np.square(np.subtract(np.log([kpaa,kSOCSona,R1a,R2a,kd4,k_d4]),np.log(priors))/rho)) # fitting to prior
        chi2 = 0 # fitting to data
        for i in range(len(all_exps)):
            for j in range(len(all_exps[i])):
                if all_sims[i][j]==0:
                    pass
                else:
                    chi2 += ((np.log(all_exps[i][j])-np.log(all_sims[i][j]))/all_sigmas[i][j])**2
        points.append([rho,[xi2,chi2]])
    
    # Plot
    fig, ax = plt.subplots()
    ax.set(xscale="log", yscale="linear")      
    plt.xlabel(r'$\rho$', fontsize=18)
    plt.ylabel('score', fontsize=18)
    fig.suptitle(r'Choosing $\rho$', fontsize=18)
    ax.tick_params(labelsize=14)
    ax.plot([p[0] for p in points],[p[1][0] for p in points],'bo',label=r'$\chi^2$')
    ax.plot([p[0] for p in points],[p[1][1] for p in points],'go',label=r'$\xi^2$')
    ax.plot([p[0] for p in points],[p[1][0]+p[1][1] for p in points],'k--',label=r'$\chi^2 + \xi^2$')
    fig.savefig('Bayes_rho_2.pdf')
    print([p[1][1] for p in points])
if __name__ == '__main__':
    main()
