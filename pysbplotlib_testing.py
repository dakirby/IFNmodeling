# -*- coding: utf-8 -*-
"""
Created on Wed May 16 14:39:42 2018

@author: Duncan

Used to test functions in pysbplotlib.py as I develop them.
"""
from IFN_Models import IFN_simplified_model_alpha as IFNaModel
from IFN_Models import IFN_simplified_model_beta as IFNbModel
from IFN_Models import IFN_detailed_model_alpha as IFNaModel_d
from IFN_Models import IFN_detailed_model_beta as IFNbModel_d
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
        pyplt.compare_timecourse([IFNaModel], 
                                 [{'I':10E-12*6.022E23*1E-5},{'I':90E-12*6.022E23*1E-5},{'I':600E-12*6.022E23*1E-5}],
                                 t, ['TotalpSTAT',"Total pSTAT"], 
                                 ['r','r--','r:','g','g--','g:'],
                                 axes_labels=['Time (s)','Molecules/cell'],
                                 title=r"IFN$\alpha$ and IFN$\beta$ Time Course",
                                 custom_legend=[r"IFN$\alpha$ 10 pM",
                                                r"IFN$\alpha$ 90 pM",
                                                r"IFN$\alpha$ 600 pM"])
    if dr==True:
        # For profiling
        tic = timeit.default_timer()
        # Plot dose-response
        pyplt.doseresponse(IFNaModel, ['IFN',np.logspace(-14,-2,num=50)], t,
                                       [['T',"Ta"]],
                         ['Dose (M)','Molecules/cell'],title='IFNa Dose-response', Norm=1,
                         parameters={'kSOCSon':0})
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
        dtc = pyplt.timecourse(IFNbModel_d, t, [['TotalpSTAT',"Total pSTAT"],['BoundSOCS',"Bound SOCS"]],
                         ['Time (s)','Molecules/cell'],title='IFNb Time Course',
                         parameters={'IFN':500E-12})


        kpaa = 1    #  3.981e-07 = 80.69% BEST FIT BY CHOOSING DIFFERENT GAMMAS - Used Gamma_40 fit for beta and fit_all_new for alpha (might be wrong about alpha)
        kSOCSona = 1.000e-06 
        R1a = 2.625e+03
        R2a = 2.625e+03
        kloc = 3.219e-03
        kSOCSmRNA = 2.575e-03
        kSOCS = 1.288e-02        

        from IFN_Models import IFN_detailed_model_beta_ppCompatible as IFNbModel_dpp
        dtc_pp = pyplt.timecourse(IFNbModel_dpp, t, [['TotalpSTAT',"Total pSTAT"],['BoundSOCS',"Bound SOCS"]],
                         ['Time (s)','Molecules/cell'],title='IFNb Time Course',
                         parameters={'I':500E-12*6.022E23*1E-5})
        
        test_detailed = dtc['TotalpSTAT']-dtc_pp['TotalpSTAT']
        if all(np.square(test_detailed)) < 1:
            print("versions of detailed model are equivalent")
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
   main(dr=True)
   
