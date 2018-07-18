# -*- coding: utf-8 -*-
"""
Created on Wed May 16 14:39:42 2018

@author: Duncan

Used to test functions in pysbplotlib.py as I develop them.
"""
import IFN_simplified_model_alpha as IFNaModel
import IFN_simplified_model_beta as IFNbModel
import IFN_detailed_model_alpha as IFNaModel_d
import IFN_detailed_model_beta as IFNbModel_d
import pysbplotlib as pyplt
import matplotlib.pyplot as plt
import numpy as np
import timeit #For profiling
plt.close('all')



# Run simulation
def main(tc=False, dr=False, multi_dr=False, multi_tc=False, 
         wtc=False, wctc=False, wdr=False, wmulti_dr=False,
         detailed_tc=False, detailed_dr=False):
    t = np.linspace(0, 3600)
    if tc==True:
        pyplt.timecourse(IFNaModel, t, [['TotalpSTAT',"Total pSTAT"]],
                         ['Time (s)','Molecules/cell'],title='IFNa Time Course')
        pyplt.timecourse(IFNbModel, t, [['T',"Tb"],['TotalpSTAT',"Total pSTAT"],
                                        ['BoundSOCS',"Bound SOCS"]],
                         ['Time (s)','Molecules/cell'],title='IFNb Time Course')
    if multi_tc==True:
        pyplt.compare_timecourse([IFNaModel, IFNbModel], 
                                 [{'IFN':10E-12},{'IFN':100E-12},{'IFN':500E-12}],
                                 t, ['TotalpSTAT',"Total pSTAT"], 
                                 ['r','r--','r:','g','g--','g:'],
                                 axes_labels=['Time (s)','Molecules/cell'],
                                 title=r"IFN$\alpha$ and IFN$\beta$ Time Course",
                                 custom_legend=[r"IFN$\alpha$ 10 pM",
                                                r"IFN$\alpha$ 100 pM",
                                                r"IFN$\alpha$ 500 pM",
                                                r"IFN$\beta$ 10 pM",
                                                r"IFN$\beta$ 100 pM",
                                                r"IFN$\beta$ 500 pM",])
    if dr==True:
        # For profiling
        tic = timeit.default_timer()
        # Plot dose-response
        pyplt.doseresponse(IFNaModel, ['IFN',np.logspace(-14,-2,num=50)], t,
                                       [['T',"Ta"],['TotalpSTAT',"Total pSTAT"]],
                         ['Dose (M)','Molecules/cell'],title='IFNa Dose-response', Norm=10000)
        pyplt.doseresponse(IFNbModel, ['IFN',np.logspace(-14,-2,num=50)], t,
                                       [['T',"Tb"],['TotalpSTAT',"Total pSTAT"]],
                         ['Dose (M)','Molecules/cell'],title='IFNb Dose-response',
                         parameters={'k_d4':200})
        toc = timeit.default_timer()
        print(toc-tic)
        
    if multi_dr==True:        
        pyplt.compare_doseresponse([IFNaModel, IFNbModel],['IFN',
                                   np.logspace(-14,-2,num=50)], [t,t],
                                   [['T',"T"],['TotalpSTAT',"Total pSTAT"]], ['r','g'],
                          ['IFN Dose (M)',r"pSTAT/$STAT_{Total}$"],
                          title='Dose-response Curves',Norm=[2000,10000])

    if wtc==True:
        pyplt.Wtimecourse([IFNaModel,IFNaModel],[0.8,0.2],
                          [{'R1':2000, 'R2':2000},{'R1':8000, 'R2':8000}],
                          t, [['TotalpSTAT',"Total pSTAT"]],
                         ['Time (s)',r"pSTAT//$pSTAT_{Total}$"],
                         title='IFNa Weighted Time Course', Norm=10000)
    if wctc==True:
        pyplt.Wcompare_timecourse([[IFNaModel,IFNaModel],[IFNbModel,IFNbModel]],
                                  [[0.8,0.2],[0.8,0.2]],
                                  [[{'R1':2000, 'R2':2000},{'R1':8000, 'R2':8000}],
                                  [{'R1':2000, 'R2':2000},{'R1':8000, 'R2':8000}]],
                                   t, ["TotalpSTAT","Total pSTAT"],
                                   ['r','g'], global_Norm=10000,
                                   axes_labels = ['time (s)',r"pSTAT/pSTAT$_{Total}$"],
                                   title = 'Weighted time course', 
                                   custom_legend=[r"IFN$\alpha$",r"IFN$\beta$"])
    if wdr==True:
        pyplt.Wdoseresponse([IFNaModel,IFNaModel],[0.8,0.2],
                            [{'R1':2000, 'R2':2000},{'R1':8000, 'R2':2000}],
                            ['IFN',np.logspace(-14,-2,num=50)], t, 
                            [['TotalpSTAT',"Total pSTAT"]],
                            axes_labels = ['IFN (M)',r"pSTAT/pSTAT$_{Total}$"],
                            title = 'Weighted dose response', wNorm=10000)

    if wmulti_dr==True:
        pyplt.Wcompare_doseresponse([[IFNaModel,IFNaModel],[IFNbModel,IFNbModel]],
                                    [[0.8,0.2],[0.8,0.2]], 
                                    ['IFN',np.logspace(-14,-2,num=50)],
                                    [[{'R1':2000, 'R2':2000},{'R1':8000, 'R2':8000}],
                                     [{'R1':2000, 'R2':2000},{'R1':8000, 'R2':8000}]], 
                                    [t,t], [['TotalpSTAT',"Total pSTAT"]],  
                                    ['r','g'], 
                                    axes_labels = ['IFN (M)',r"pSTAT/pSTAT$_{Total}$"], 
                                    title = 'Weighted dose response', Norm=10000)
    if detailed_tc==True:
        pyplt.timecourse(IFNbModel_d, t, [['TotalpSTAT',"Total pSTAT"],['BoundSOCS',"Bound SOCS"]],
                         ['Time (s)','Molecules/cell'],title='IFNb Time Course',
                         parameters={'IFN':500E-12})
        
        pyplt.compare_timecourse([IFNaModel_d, IFNbModel_d], 
                                 [{'IFN':10E-12},{'IFN':100E-12},{'IFN':500E-12}],
                                 t, [['TotalpSTAT',"pSTAT"]], 
                                 ['r','r--','r:','g','g--','g:'],
                                 axes_labels=['Time (s)','Molecules/cell'],
                                 title=r"IFN$\alpha$ and IFN$\beta$ Time Course",
                                 custom_legend=[r"IFN$\alpha$ 10 pM",
                                                r"IFN$\alpha$ 100 pM",
                                                r"IFN$\alpha$ 500 pM",
                                                r"IFN$\beta$ 10 pM",
                                                r"IFN$\beta$ 100 pM",
                                                r"IFN$\beta$ 500 pM",])

    if detailed_dr==True:
        pyplt.compare_doseresponse([IFNaModel_d, IFNaModel_d, IFNaModel_d, 
                                    IFNbModel_d, IFNbModel_d, IFNbModel_d],
                                 ['IFN',np.logspace(-14,-2,num=50)], 
                                 [np.linspace(0,900),np.linspace(0,1800),np.linspace(0,3600),
                                  np.linspace(0,900),np.linspace(0,1800),np.linspace(0,3600)],
                                 [['TotalpSTAT',"Total pSTAT"]], 
                                 ['r','r--','r:','g','g--','g:'],
                                 Norm=10000,
                                 axes_labels=['Time (s)','Fraction of pSTAT Phosphorylated'],
                                 title=r"IFN$\alpha$ and IFN$\beta$ Time Course",
                                 custom_legend=[r"IFN$\alpha$ 15 min",
                                                r"IFN$\alpha$ 30 min",
                                                r"IFN$\alpha$ 60 min",
                                                r"IFN$\beta$ 15 min",
                                                r"IFN$\beta$ 30 min",
                                                r"IFN$\beta$ 60 min",])


if __name__ == "__main__":
   main(detailed_tc=True)
   
