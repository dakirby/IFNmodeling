# -*- coding: utf-8 -*-
"""
Created on Wed May 16 14:39:42 2018

@author: Duncan

Used to test functions in pysbplotlib.py as I develop them.
"""
import IFN_simplified_model_alpha as IFNaModel
import IFN_simplified_model_beta as IFNbModel
import pysbplotlib as pyplt
import matplotlib.pyplot as plt
import numpy as np
import timeit #For profiling
plt.close('all')



# Run simulation
def main(tc=False, dr=False, multi_dr=False, multi_tc=False, 
         wtc=False, wctc=False, wdr=False, wmulti_dr=False,
         param_dr=False):
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
    if param_dr==True:
        pass
#       x = pyplt.mp_DR_parameter_scan([[1,7],[2,7],[3,7]],3)(3, 1, IFNaModel.model, ['IFN',np.logspace(-14,-2,num=50)],
#                                3600, ['pSTAT',"pSTAT"], 'kd4', [0.2,0.4],
#                                'kd2', [0.01,0.03], Norm=10000, parameters={'kpa':0.3})
# =============================================================================
# from multiprocessing import Process, Queue, JoinableQueue, cpu_count
# def working(id, jobs, result):
#     while True:
#         task = jobs.get()
#         if task is None:
#             break
#         print( "%d task:" % id, task)
#         result.put("%s task r" % id)
#     #result.put(None)
# def wrapper():
#     jobs = Queue()
#     result = JoinableQueue()
#     NUMBER_OF_PROCESSES = cpu_count()
#     
#     tasks = ["1","2","3","4","5"]
#     
#     for w in tasks:
#         jobs.put(w)
# 
#     [Process(target=working, args=(i, jobs, result)).start()
#             for i in range(NUMBER_OF_PROCESSES)]
#     
#     print( 'starting workers')
#     
#     for t in range(len(tasks)):
#         r = result.get()
#         print( r )
#         result.task_done()
#     
#     for w in range(NUMBER_OF_PROCESSES):
#         jobs.put(None)
# 
#     result.join()
#     jobs.close()
#     result.close()
# 
# =============================================================================
if __name__ == "__main__":
   main(wdr=True)
   
