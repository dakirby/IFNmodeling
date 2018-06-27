# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 12:03:19 2018

@author: Duncan
"""
# =============================================================================
# Designed to be a series of tests for functions in pysb_parallel, to make sure 
# everything runs as it should
# =============================================================================
modelfilename = "IFN_simplified_model_beta_ppCompatible"
import pysb_parallel as pp
from numpy import linspace, logspace, reshape, flipud, flip
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

def main(timecourse=False, dose_response=False, wtimecourse=False, wdose_response=False,
         paramScan=False):
    plt.close('all')
    t=linspace(0,3600,num=100)
    testDose = ['I',6.022e18*logspace(-14,-2,num=50)]    
    
    if timecourse==True:
        tc = pp.p_timecourse(modelfilename, t, [['TotalpSTAT',"Total pSTAT"]],
                             axes_labels = ['Time','pSTAT'], Norm=10000,
                             title = 'TC_test1', suppress=False)
        tc = pp.p_timecourse(modelfilename, t, [['TotalpSTAT',"Total pSTAT"]],
                             axes_labels = ['Time','pSTAT'], Norm=10000,
                             title = 'TC_test2', suppress=False, parameters = [['I',3E-12]])

    if dose_response==True:    
        dr = pp.p_doseresponse(modelfilename, testDose, t, 
                               [['T',"Ta"],['TotalpSTAT',"Total pSTAT"]], 
                               axes_labels = ['IFN','pSTAT'], title = 'DR_test1',
                               suppress=False, Norm=10000, dose_axis_norm=6.022e18)
        dr = pp.p_doseresponse(modelfilename, testDose, t, 
                               [['T',"Ta"],['TotalpSTAT',"Total pSTAT"]], 
                               axes_labels = ['IFN','pSTAT'], title = 'DR_test2',
                               suppress=False, Norm=10000, dose_axis_norm=6.022e18,
                               parameters=[['kpa',4E-6]])
        
    if wtimecourse==True:
        pp.p_Wtimecourse([modelfilename,modelfilename], [0.8,0.2], [[['R1',2E3]],[['R1',8E3]]], t, 
                      [['T',"Ta"],['TotalpSTAT',"Total pSTAT"]], axes_labels = ['time','pSTAT'], 
                      title = 'wTC_test', Norm=10000,suppress=False)
    if wdose_response==True:
        pWdr = pp.p_Wdoseresponse([modelfilename,modelfilename], [0.8,0.2], 
                                  [[['R1',2E3],['R1',8E3]],[['R1',2E3],['R1',8E3]]],
                                  testDose, t, [['T',"Ta"],['TotalpSTAT',"Total pSTAT"]], 
                                  axes_labels = ['IFN','Species'], title = 'wDR_test', 
                                  wNorm=10000, dose_axis_norm=6.022e18, suppress=True)
        print(pp.get_EC50(logspace(-14,-2,num=50),pWdr[1]))
    
    if paramScan==True:
        kpaScan = 1E-6*logspace(-2,2,num=30)
        kSSCan= 1E-6*logspace(-2,2,num=30)
        t=linspace(0,3600,num=200)
        zscan = pp.p_DRparamScan(modelfilename, ['kpa',kpaScan], ['kSOCSon',kSSCan],
                                 testDose, t, [['TotalpSTAT',"Total pSTAT"]])
	#image = [el[2] for el in zscan]
	#image = reshape(image, (len(xscan),len(yscan)))
	#print(image)
	#plt.ion()
# =============================================================================
# 	fig, ax = plt.subplots()
# 	#Build x and y axis labels
# 	xticks =  ['%.2f' % i for i in xscan]
# 	xticks = [float(i) for i in xticks]
# 	yticks = ['%.2f' % i for i in yscan]
# 	yticks = [float(i) for i in yticks]
# 	yticks = flip(yticks,0)
# 	im = sns.heatmap(flipud(image), xticklabels=xticks, yticklabels=yticks)
# 	plt.xlabel('kd4', fontsize=18)
# 	plt.ylabel('kpa', fontsize=18)
# 	fig.suptitle('2D parameter scan of EC50\n as a function of kd4 and kpa', fontsize=18)
# 	ax.tick_params(labelsize=14)
# #	plt.colorbar(im)	
# 	plt.savefig("scan.pdf")
# =============================================================================

if __name__ == '__main__':
    main(paramScan=True)

