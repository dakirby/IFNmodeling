# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 12:03:19 2018

@author: Duncan
"""
# =============================================================================
# Designed to be a series of tests for functions in pysb_parallel, to make sure 
# everything runs as it should. Note that p_timecourse, p_doseresponse, 
# p_Wtimecourse, and p_Wdoseresponse are not meant to be used as standalone
# functions, but are intended only for use in multi-threaded functions from 
# pysb_parallel. The following tests only verify that these internal functions
# are working. 
#   The main issue with these internal functions is that, due to the structure
#   of import statements required for multi-threading, only one model can be 
#   used in a script (other models aren't loaded properly; the computer loads
#   the first model used from pycache). This is not an issue with the 
#   multi-threaded functions fit_model() and p_DRparamScan(). 
# =============================================================================
simpleA = "IFN_simplified_model_alpha_ppCompatible"
simpleB = "IFN_simplified_model_beta_ppCompatible"
detailedA = "IFN_detailed_model_alpha_ppCompatible"
detailedB = "IFN_detailed_model_beta_ppCompatible"

import pysb_parallel as pp
import Experimental_Data as ED
from numpy import linspace, logspace, reshape, flipud, flip, divide
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

def main(timecourse=False, dose_response=False, wtimecourse=False, wdose_response=False,
         paramScan=False, fit=False, detailed_tc=False, detailed_dr=False, fit_lhc=False):
    plt.close('all')
    t=linspace(0,3600,num=100)
    testDose = ['I',6.022e18*logspace(-14,-2,num=50)]    
    if timecourse==True:
        tc = pp.p_timecourse(simpleB, t, [['TotalpSTAT',"Total pSTAT"]],
                             axes_labels = ['Time','pSTAT'], Norm=10000,
                             title = 'TC_test1', suppress=False)
        tc = pp.p_timecourse(simpleB, t, [['TotalpSTAT',"Total pSTAT"]],
                             axes_labels = ['Time','pSTAT'], Norm=10000,
                             title = 'TC_test2', suppress=False, parameters = [['I',3E-12]])

    if dose_response==True:    
        dr = pp.p_doseresponse(simpleB, testDose, t, 
                               [['T',"Ta"],['TotalpSTAT',"Total pSTAT"]], 
                               axes_labels = ['IFN','pSTAT'], title = 'DR_test1',
                               suppress=False, Norm=10000, dose_axis_norm=6.022e18)
        dr = pp.p_doseresponse(simpleB, testDose, t, 
                               [['T',"Ta"],['TotalpSTAT',"Total pSTAT"]], 
                               axes_labels = ['IFN','pSTAT'], title = 'DR_test2',
                               suppress=False, Norm=10000, dose_axis_norm=6.022e18,
                               parameters=[['kpa',4E-6]])
        
    if wtimecourse==True:
        pp.p_Wtimecourse([simpleB,simpleB], [0.8,0.2], [[['R1',2E3]],[['R1',8E3]]], t, 
                      [['T',"Ta"],['TotalpSTAT',"Total pSTAT"]], axes_labels = ['time','pSTAT'], 
                      title = 'wTC_test', Norm=10000,suppress=False)
        pp.p_Wtimecourse([simpleA,simpleA], [0.8,0.2], [[['R1',2E3]],[['R1',8E3]]], t, 
                      [['T',"Ta"],['TotalpSTAT',"Total pSTAT"]], axes_labels = ['time','pSTAT'], 
                      title = 'wTC_test', Norm=10000,suppress=False)
        
    if wdose_response==True:
        pWdr = pp.p_Wdoseresponse([simpleB,simpleB], [0.8,0.2], 
                                  [[['R1',2E3],['R1',8E3]],[['R1',2E3],['R1',8E3]]],
                                  testDose, t, [['T',"Ta"],['TotalpSTAT',"Total pSTAT"]], 
                                  axes_labels = ['IFN','Species'], title = 'wDR_test', 
                                  wNorm=10000, dose_axis_norm=6.022e18, suppress=True)
        print(pp.get_EC50(logspace(-14,-2,num=50),pWdr[1]))
    
    if paramScan==True:
        kpaScan = 1E-6*logspace(-2,2,num=8)
        kSSCan= 1E-6*logspace(-2,2,num=8)
        t=linspace(0,3600,num=200)
        zscan = pp.p_DRparamScan(simpleB, ['kpa',kpaScan], ['kSOCSon',kSSCan],
                                 testDose, t, [['TotalpSTAT',"Total pSTAT"]])
    if fit==True:
        kpaScan = 1E-6*logspace(-2,2,num=8)
        kSSCan= 1E-6*logspace(-2,2,num=8)
        xdata = [5*60,15*60,30*60,60*60]
        xdata = [['t',el] for el in xdata]
        ydata = ED.data.loc[(ED.data.loc[:,'Interferon']=="Beta"),['5','15','30','60']].values[0]
        uncertainty = ED.data.loc[(ED.data.loc[:,'Interferon']=="Beta_std"),['5','15','30','60']].values[0]
        NA = 6.022E23
        volEC = 1E-5
        IFN = [['I', NA*volEC*10E-12] for i in ydata]
        xdata = [[IFN[el]]+[xdata[el]] for el in range(len(xdata))]
        tc = pp.p_timecourse(simpleB, [5*60,15*60,30*60,60*60], 
                             [['TotalpSTAT',"Total pSTAT"]],suppress=True,
                             parameters=[['I',NA*volEC*10E-12]])['TotalpSTAT']
        print("ydata = "+str(tc))
        zscan = pp.fit_model(simpleB, xdata, ['TotalpSTAT',tc], ['kpa','kSOCSon'],
                             p0=[[1E-6,1E-9,1E-3,'log'],[1E-6,1E-9,1E-3,'log']],
                             sigma=uncertainty, method="bayesian")
        zscan = pp.fit_model(simpleA, xdata, ['TotalpSTAT',tc], ['kpa','kSOCSon'],
                             p0=[[1E-6,1E-9,1E-3,'log'],[1E-6,1E-9,1E-3,'log']],
                             sigma=uncertainty, method="bayesian")
    if fit_lhc==True:
        kpaScan = 1E-6*logspace(-2,2,num=8)
        kSSCan= 1E-6*logspace(-2,2,num=8)
        xdata = [5*60,15*60,30*60,60*60]
        xdata = [['t',el] for el in xdata]
        ydata = ED.data.loc[(ED.data.loc[:,'Interferon']=="Beta"),['5','15','30','60']].values[0]
        uncertainty = ED.data.loc[(ED.data.loc[:,'Interferon']=="Beta_std"),['5','15','30','60']].values[0]
        NA = 6.022E23
        volEC = 1E-5
        IFN = [['I', NA*volEC*10E-12] for i in ydata]
        xdata = [[IFN[el]]+[xdata[el]] for el in range(len(xdata))]
        tc = pp.p_timecourse(simpleB, [5*60,15*60,30*60,60*60], 
                             [['TotalpSTAT',"Total pSTAT"]],suppress=True,
                             parameters=[['I',NA*volEC*10E-12]])['TotalpSTAT']
        zscan = pp.fit_model(simpleA, xdata, ['TotalpSTAT',tc], ['ka4','kSOCSon','R1'],
                             #p0=[[3.623188E-4,3.623188E-5,3.623188E-3,'log'],[1E-6,1E-8,1E-4,'log'],[2E3,8E2,8E4,'linear']],                            
                             sigma=uncertainty, method="lhc_sampling", n=25, cpu=3)

if __name__ == '__main__':
    main(fit_lhc=True)

