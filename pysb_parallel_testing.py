# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 12:03:19 2018

@author: Duncan
"""
# =============================================================================
# Designed to be a series of tests for functions in pysb_parallel, to make sure 
# everything runs as it should
# =============================================================================
modelfilename = "IFN_simplified_model_alpha_ppCompatible"
import pysb_parallel as pp
from numpy import linspace, logspace, reshape
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

def main():
	plt.close('all')
	t=linspace(0,3600,num=100)
	normvals = [[10000+0.0001*i for i in range(len(t))] for x in range(2)]
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
	xscan = [0.003, 0.03, 0.3, 3.0, 30.0]
	yscan = [1E-4, 1E-5, 1E-6, 1E-7, 1E-8]
	zscan = pp.p_DRparamScan(modelfilename, ['kd4',xscan], ['kpa',yscan],
                   testDose, 3600, [['TotalpSTAT',"Total pSTAT"]], custom_params=[['R1',2100]],
				   Norm=10000, cpu=4)
	image = [el[2] for el in zscan]
	image = reshape(image, (len(xscan),len(yscan)))
	print(image)
	#plt.ion()
	fig, ax = plt.subplots()
	#ax.set_xlim(xscan[0],xscan[-1])
	#ax.set_ylim(yscan[0],yscan[-1])
	im = ax.imshow(image, origin='lower')
	plt.xlabel('kd4', fontsize=18)
	plt.ylabel('kpa', fontsize=18)
	fig.suptitle('2D parameter scan of EC50\n as a function of kd4 and kpa', fontsize=18)
	ax.tick_params(labelsize=14)
	plt.colorbar(im)	
	plt.savefig("scan.pdf")

if __name__ == '__main__':
    main()

