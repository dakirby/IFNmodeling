# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 16:58:13 2018

@author: Duncan

IFN_2Dscan() produces a 2D screen of combinations of IFNa and IFNb
Input:
    IFNaDoses (list) = a list of the doses of IFNa to test (along x-axis)
    IFNbDoses (list) = a list of the doses of IFNb to test (along y-axis)
    measuredSpeciesName = the name of the species measured as a response to IFN dosage (usually pSTAT)
    t_end (int) = the time, in seconds, at which the level of pSTAT response is measured
    modelfile (str) = the name of the modelfile to use
Output:
    a pdf file containing an image of the results of the 2D scan.    
Return:
    the values used to produce the 2D scan pdf file
"""
from multiprocessing import Process, Queue, JoinableQueue, cpu_count
from pysb.export import export
from numpy import divide, subtract, abs, reshape, flipud, flip
import numpy as np
from operator import itemgetter # for sorting results after processes return
from scipy.optimize import minimize 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

from pysb_parallel import p_timecourse
# =============================================================================
# p_DRparamScan_helper() is an internal function used by p_DRparamScan(). Each 
# thread calls this function. 
# Inputs:
#     id: the thread ID
#     jobs: the arguments for p_doseresponse()
#     result: the queue to put results on to
# =============================================================================
def IFN_2Dscan_helper(id, jobs, result):
    # try to work as long as work is available
    while True:
        # get job from the queue
        task = jobs.get()
        if task is None:
            # there are no jobs
            break
        # there is a job, so do it 
        modelfile, t, spec, params = task
        #p_timecourse(modelfile, t, spec, axes_labels = ['',''], title = '',
               #Norm=None, suppress=False, parameters=None, scan=0)
        dr = p_timecourse(modelfile, t, spec, suppress=True, parameters=params, scan=1)[spec[0]]
        # put the result onto the results queue
        result.put([params[0][1], params[1][1], dr])
# =============================================================================
# IFN_heatmap() is an internal function for p_DRparamScan(). It plots the scans.
# Inputs:
#     image: the EC50 image to plot, with pixels oriented such that top left is 
#             origin (ie. as the output from image_builder())
#     param1label, param2label (str): the labels for the scanned parameters to mark x and y axes by, respectively
# =============================================================================
def IFN_heatmap(image, param1label, param2label):
    fig, ax = plt.subplots()
    fig.set_figwidth(12)
    fig.set_figheight(10)
    # Build title
    title = "{} vs {}".format(param1label,param2label)   
	 # Build x and y axis labels
    param1 = list(np.asarray([el[0] for el in image[0]]).flatten())
    xticks =  ['{:.2e}'.format(float(i)) for i in param1]
    xticks = [float(i) for i in xticks]
    param2 = list(np.asarray([el[0][1] for el in image]).flatten())
    yticks = ['{:.2e}'.format(float(i)) for i in param2]
    yticks = [float(i) for i in yticks]
    yticks = flip(yticks,0)
    # Plot image
    image = [[el[2] for el in r] for r in image]
    sns.heatmap(flipud(image), xticklabels=xticks, yticklabels=yticks)
    plt.title(title)
    plt.xlabel(param1label)
    plt.ylabel(param2label)
    # Save figure
    plt.savefig("{}.pdf".format(title))
    
# =============================================================================
# image_builder() is an internal function used by p_DRparamScan() to ensure
# results are reordered correctly after all threads return
# Inputs: 
#     results: pool_results returned by parameter scan
#     doseNorm: normalize the values of param1 and param2 for the scan (eg. convert molecules to concentration)
#     shape: the x and y axis dimensions
# Returns: 
#     response_image: responses ordered with top left item as origin 
#                 (ie. first x and y values of scan parameters)
# =============================================================================
def image_builder(results, doseNorm, shape):
    response_image = [[el[0]/doseNorm, el[1]/doseNorm, el[2][len(el[2])-1]] for el in results]
    response_image.sort(key=itemgetter(1,0))
    response_image = reshape(response_image,(shape[0],shape[1],3))
    return response_image
# =============================================================================
# IFN_2Dscan() is a 2D scan for the timecourse of a given modelfile, 
# parallelized with Python multiprocessing.
# Inputs:
#      modelfile = the name of the model file to use for the parameter scan
#      param1 = a list of the form ['parameter_name',[values for the parameter]]
#	   param2 = a list of the form ['parameter_name',[values for the parameter]]
#		** param1 and param2 are the two parameters being scanned over
#      t_list = list of times to use for each timecourse in the scan. More time points
#				*may* make numerical integration more stable.
#      spec = a two-element list of the form [name of an observable, label for observable]
#             This is meant to be the species measured to quantify cellular response    
#	   custom_params = a list of other parameters to control in the same form as param1        
#     cpu = number of cpus to use (default is cpu = multiprocessing.cpu_count()-1)   
#     doseNorm = a value to normalize the dose by for plotting, 
#               ie. convert molecules to concentrations just for plotting 
# Outputs: pdf file of the 2D scan
# Returns: A 2D array with each element containing three values: 
#			[x-axis value, y-axis value, z_value]
# =============================================================================
def IFN_2Dscan(modelfile, param1, param2, t_list, spec, custom_params=None,
                  cpu=None, doseNorm=1, suppress=False, verbose=1):
    # initialization
    jobs = Queue()
    result = JoinableQueue()
    if cpu == None or cpu >= cpu_count():
        NUMBER_OF_PROCESSES = cpu_count()-1
    else:
        NUMBER_OF_PROCESSES = cpu
    if verbose != 0: print("Using {} threads".format(NUMBER_OF_PROCESSES))
    # build task list
    params=[]
    if verbose != 0: print("Building tasks")
    if custom_params == None:
        for val1 in param1[1]:
            for val2 in param2[1]:
                params.append([[param1[0],val1],[param2[0],val2]])
    else:
        for val1 in param1[1]:
            for val2 in param2[1]:
                params.append([[param1[0],val1],[param2[0],val2]]+[c for c in custom_params])

    # Write modelfile
    imported_model = __import__(modelfile,fromlist=['IFN_Models'])
    py_output = export(imported_model.model, 'python')
    with open('ODE_system.py','w') as f:
        f.write(py_output)
				
    tasks = [[modelfile, t_list, spec, p] for p in params]
    # put jobs on the queue
    if verbose != 0: print("There are {} tasks to compute".format(len(params)))
    if verbose != 0: print("Putting tasks on the queue")
	
    for w in tasks:
        jobs.put(w)
		
    if verbose != 0: print("Computing scan")
	
    # start up the workers
    [Process(target=IFN_2Dscan_helper, args=(i, jobs, result)).start()
            for i in range(NUMBER_OF_PROCESSES)]
    
    # pull in the results from each worker
    pool_results=[]
    for t in range(len(tasks)):
        r = result.get()
        pool_results.append(r)
        result.task_done()
    # tell the workers there are no more jobs
    for w in range(NUMBER_OF_PROCESSES):
        jobs.put(None)
    # close all extra threads
    result.join()
    jobs.close()
    result.close()
    if verbose != 0: print("Done scan")
    response_image = image_builder(pool_results, doseNorm, (len(param1[1]),len(param2[1])))
    # plot heatmap if suppress==False
    if suppress==False:
        	IFN_heatmap(response_image, "response image - {}".format(param1[0]), param2[0])
    #return the scan 
    return response_image

def MSE(simulated_scan, experimental_scan):
    def score(sf, simulated_scan, experimental_scan):
        return np.sum(np.square(np.subtract(np.multiply(simulated_scan,sf), experimental_scan)))
    optimal_score = minimize(score,[40],args=(simulated_scan, experimental_scan))['fun']
    return optimal_score


# =============================================================================
# def main():
#     data = pd.read_csv("20181031_pSTAT1_Table.csv")
#     experimental_Tcell = data.drop([8]).loc[:,"T cells"].values.reshape(8,8)
#     experimental_NKcell = data.drop([8]).loc[:,"NK cells"].values.reshape(8,8)
#     experimental_Bcell = data.drop([8]).loc[:,"B cells"].values.reshape(8,8)
# 
#     def score_parameter(p,experimental_scan):
#          res = IFN_2Dscan("IFN_Models.Mixed_IFN_ppCompatible",
#                 ["Ib",np.multiply([0,0.06,0.32,1.6,8,40,200,1000],1E-12*6.022E23*1E-5)],
#                 ["Ia",np.multiply([0,0.06,0.32,1.6,8,40,200,1000],1E-12*6.022E23*1E-5)],
#                 [0,300,900,1800,3600],
#                 ['TotalpSTAT','pSTAT1'],
#                 doseNorm=6.022E23*1E-5,
#                 custom_params=[['R2',p]],
#                 suppress=True, verbose=0)
#          scan = [[el[2] for el in r] for r in res]
#          score = MSE(scan,experimental_scan)
#          return score
#      
#     Tcell_R2 = minimize(score_parameter,[2000],args=(experimental_Tcell))['x']
#     Bcell_R2 = minimize(score_parameter,[2000],args=(experimental_Bcell))['x']
#     NKcell_R2 = minimize(score_parameter,[2000],args=(experimental_NKcell))['x']
# 
#     print("The optimal value of R2 for Tcell data was {}".format(Tcell_R2))
#     print("The optimal value of R2 for Tcell data was {}".format(Bcell_R2))
#     print("The optimal value of R2 for Tcell data was {}".format(NKcell_R2))
#     
# # =============================================================================
# #     res = IFN_2Dscan("IFN_Models.Mixed_IFN_ppCompatible",
# #                ["Ib",np.multiply([0,0.36,0.32,1.6,8,40,200,1000],1E-12*6.022E23*1E-5)],
# #                ["Ia",np.multiply([0,0.36,0.32,1.6,8,40,200,1000],1E-12*6.022E23*1E-5)],
# #                [0,300,900,1800,3600],
# #                ['TotalpSTAT','pSTAT1'],
# #                doseNorm=6.022E23*1E-5,
# #                custom_params=[['R2',500]],
# #                suppress=True) 
# #     arcsinh_response = []
# #     for row in res:
# #         new_row = []
# #         for item in row:
# #             new_row.append([item[0]/1E-12,item[1]/1E-12,np.arcsinh(item[2])])
# #         arcsinh_response.append(new_row)
# #     IFN_heatmap(arcsinh_response, "R2-500 arcsinh - {}".format('Ib (pM)'), 'Ia (pM)')
# # =============================================================================
# if __name__ == '__main__':
#     main()
# =============================================================================
    