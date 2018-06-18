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

def main():
	plt.close('all')
	t=linspace(0,3600,num=100)
	#tc = pp.p_timecourse(modelfilename, t, [['TotalpSTAT',"Total pSTAT"]],
	#                     axes_labels = ['Time','pSTAT'], Norm=10000,
	#                     title = 'Timecourse', suppress=False, parameters = [['Ia',3E9]])
	testDose = ['Ia',6.022e18*logspace(-14,-2,num=50)]
	#dr = pp.p_doseresponse(modelfilename, testDose, t, 
	#                       [['T',"Ta"],['TotalpSTAT',"Total pSTAT"]], 
	#                       axes_labels = ['IFN','pSTAT'], title = 'First p_DR',
	#                 suppress=False, Norm=10000, dose_axis_norm=6.022e18)
	#pp.p_Wtimecourse([modelfilename,modelfilename], [0.8,0.2], [[['R1',2E3]],[['R1',8E3]]], t, 
	#                 [['T',"Ta"],['TotalpSTAT',"Total pSTAT"]], axes_labels = ['time','pSTAT'], 
	#                title = 'parallel weighted timecourse', Norm=10000,suppress=False)
	#pWdr = pp.p_Wdoseresponse([modelfilename,modelfilename], [0.8,0.2], 
	#                   [[['R1',2E3],['R1',8E3]],[['R1',2E3],['R1',8E3]]],
	#                   testDose, t, [['T',"Ta"],['TotalpSTAT',"Total pSTAT"]], 
	#                 axes_labels = ['IFN','Species'], title = 'Parallel WDR', 
	#                 wNorm=10000, dose_axis_norm=6.022e18, suppress=True)
	#print(pp.get_EC50(logspace(-14,-2,num=50),dr[1]))
	xscan = 3*logspace(-4,2,num=10)
	yscan = logspace(-6,2,num=10)
	t=linspace(0,3600,num=500)
	zscan = pp.p_DRparamScan(modelfilename, ['kd4',xscan], ['kpa',yscan],
                   testDose, t, [['TotalpSTAT',"Total pSTAT"]], custom_params=[['R1',2100]],
				   Norm=10000, cpu=3)
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
    main()

