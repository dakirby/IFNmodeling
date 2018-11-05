# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:19:51 2018

@author: Duncan

Checking degree of non-linearity
"""
from pysb.export import export
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

from pysbplotlib import doseresponse
from IFN_2D_scan import IFN_2Dscan

def main():
    data = pd.read_csv("20181031_pSTAT1_Table.csv")
    experimental_Tcell = data.loc[:,"T cells"].values.reshape(8,8)
    experimental_NKcell = data.loc[:,"NK cells"].values.reshape(8,8)
    experimental_Bcell = data.loc[:,"B cells"].values.reshape(8,8)
    
    # Experimental linear additive expectation vs response:
    Tcell_alphaDR = experimental_Tcell[:,0][::-1]
    Tcell_betaDR = experimental_Tcell[-1]
    Tcell_linear_response = np.flipud([np.add(j,Tcell_alphaDR) for j in Tcell_betaDR])
    Tcell_difference = np.subtract(np.flipud(np.fliplr(experimental_Tcell)),np.multiply(2,Tcell_linear_response))
    
    Bcell_alphaDR = experimental_Bcell[:,0][::-1]
    Bcell_betaDR = experimental_Bcell[-1]
    Bcell_linear_response = np.flipud([np.add(j,Bcell_alphaDR) for j in Bcell_betaDR])
    Bcell_difference = np.subtract(np.flipud(np.fliplr(experimental_Bcell)),np.multiply(2,Bcell_linear_response))
    
    NKcell_alphaDR = experimental_NKcell[:,0][::-1]
    NKcell_betaDR = experimental_NKcell[-1]
    NKcell_linear_response = np.flipud([np.add(j,NKcell_alphaDR) for j in NKcell_betaDR])
    NKcell_difference = np.subtract(np.flipud(np.fliplr(experimental_NKcell)),np.multiply(2,NKcell_linear_response))
    
    doses = [0,0.06,0.32,1.6,8,40,200,1000]
    f,(ax1,ax2,ax3, axcb) = plt.subplots(1,4,figsize=(15,4.5),gridspec_kw={'width_ratios':[1,1,1,0.08]})
    ax1.get_shared_y_axes().join(ax2,ax3)
    g1 = sns.heatmap(Tcell_difference,cmap="YlGnBu",cbar=False,ax=ax1)
    g1.set_ylabel('IFNb (pM)')
    g1.set_xlabel('IFNa (pM)')
    g2 = sns.heatmap(Bcell_difference,cmap="YlGnBu",cbar=False,ax=ax2)
    g2.set_ylabel('')
    g2.set_xlabel('IFNa (pM)')
    g2.set_yticks([])
    g3 = sns.heatmap(NKcell_difference,cmap="YlGnBu",ax=ax3, cbar_ax=axcb)
    g3.set_ylabel('')
    g3.set_xlabel('IFNa (pM)')
    g3.set_yticks([])  
    
    for ax in [g1,g2,g3]:
        tl = doses
        ax.set_xticklabels(tl, rotation=90)
        tly = doses[::-1]
        ax.set_yticklabels(tly, rotation=0)

    plt.show()
    
    # Simulated linear additive expectation vs response
# =============================================================================
#     doses = [0]+list(np.logspace(-2,5))
#     simulation = IFN_2Dscan("IFN_Models.Mixed_IFN_ppCompatible",
#                ["Ia",np.multiply(doses,1E-12*6.022E23*1E-5)],
#                ["Ib",np.multiply(doses,1E-12*6.022E23*1E-5)],
#                [0,300,900,1800,3600],
#                ['TotalpSTAT','pSTAT1'],
#                doseNorm=1E-12*6.022E23*1E-5,
#                suppress=True)
# 
#     nonlinear_response = [[el[-1] for el in r] for r in simulation]
#     nonlinear_response = np.flipud(nonlinear_response)
# 
#     # Comparison to linear sum of responses
#     # Alpha linear
#     from IFN_Models import IFN_alpha_altSOCS_ppCompatible
#     from IFN_Models import IFN_beta_altSOCS_ppCompatible
# 
#     alphaDR = doseresponse(IFN_alpha_altSOCS_ppCompatible, ['I', np.multiply(doses,1E-12*6.022E23*1E-5)],
#                              [0,300,900,1800,3600], [['TotalpSTAT','pSTAT']],
#                              suppress=True)[0]
#     
#     # Beta linear
#     betaDR = doseresponse(IFN_beta_altSOCS_ppCompatible, ['I', np.multiply(doses,1E-12*6.022E23*1E-5)],
#                             [0,300,900,1800,3600], [['TotalpSTAT','pSTAT']],
#                             suppress=True)[0]    
# 
#     # Check that turning off one IFN in Mixed model matches single IFN models 
#     if abs(np.sum(np.subtract(nonlinear_response[:,0][::-1],betaDR))) >1 :
#         print("Hmm... looks like the models would never match under any cicumstances.")
#         print(np.sum(np.subtract(nonlinear_response[:,0][::-1],betaDR)))
#         print(nonlinear_response[:,0][::-1])
#         print("betaDR = ")
#         print(betaDR)
#     elif abs(np.sum(np.subtract(nonlinear_response[-1],alphaDR)))>1:
#         print("Hmm... looks like the models would never match under any cicumstances.")
#         print(nonlinear_response[-1])
#         print("alphaDR = ")
#         print(alphaDR)
#     # Combination
#     linear_response = [[alphaDR[i]+betaDR[j] for i in range(len(doses))] for j in range(len(doses))]
#     linear_response = np.flipud(linear_response)
#     # Check construction of linear response
#     if abs(np.sum(np.subtract(np.diag(np.fliplr(linear_response))[::-1],np.add(alphaDR,betaDR)))) > 1:
#         print("It seems like there is something wrong in computing the linear response.")
#         print(np.diag(np.fliplr(linear_response))[::-1])
#         print(np.add(alphaDR,betaDR))    
#     # Subtract linear response
#     degree_nonlinearity = np.subtract(nonlinear_response,linear_response)
#     sns.heatmap(degree_nonlinearity, xticklabels=doses, yticklabels=doses[::-1])
#     plt.title('(Mixed IFN model response) - (Linear combination of separate IFN models)')
#     plt.xlabel('IFNa (pM)')
#     plt.ylabel('IFNb (pM)')    
#     plt.show()
# =============================================================================
if __name__ == '__main__':
    main()    
