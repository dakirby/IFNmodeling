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

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
from cycler import cycler

from pysb_parallel import p_timecourse, p_doseresponse
import os
cwd = os.getcwd()
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
    sns.heatmap(flipud(image), xticklabels=xticks, yticklabels=yticks,cmap="viridis")
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
    ntimes=len(results[0][2])
    response_image = [[el[0]/doseNorm, el[1]/doseNorm, el[2][ntimes-1]] for el in results]
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
def IFN_2Dscan(modelfile, param1, param2, t_list, spec, custom_params=False,
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
    if type(custom_params) == list:
        for val1 in param1[1]:
            for val2 in param2[1]:
                params.append([[param1[0],val1],[param2[0],val2]]+[c for c in custom_params])        
    else:
        for val1 in param1[1]:
            for val2 in param2[1]:
                params.append([[param1[0],val1],[param2[0],val2]])        

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

def MSE(alpha_simulated_scan, alpha_experimental_scan, beta_simulated_scan, beta_experimental_scan,sf_flag=False):
    def score(sf, alpha_sim, alpha_exp, beta_sim, beta_exp):    
        weight_matrix = np.ones(np.shape(alpha_exp))
        for i in range(len(weight_matrix[0])):
            weight_matrix[-1][i] = 2
            weight_matrix[-2][i] = 2
            weight_matrix[-3][i] = 2
        score = np.sum(np.square(np.multiply(weight_matrix,np.subtract(np.multiply(alpha_sim,sf), alpha_exp))))+np.sum(np.square(np.multiply(weight_matrix,np.subtract(np.multiply(beta_sim,sf), beta_exp))))
        return score
    optimal_score = minimize(score,[0.9],args=(alpha_simulated_scan, alpha_experimental_scan, beta_simulated_scan, beta_experimental_scan))['fun']
    if sf_flag==True:
        return minimize(score,[0.9],args=(alpha_simulated_scan, alpha_experimental_scan, beta_simulated_scan, beta_experimental_scan))['x']
    return optimal_score

# p (list) = [['name',value]]
# experimental_scan (list) = 2D scan
# returns the mean square error
def score_parameter(p,alpha_experimental_scan,beta_experimental_scan,sf_flag=False):
    test_times =  np.multiply([2.5,5,7.5,10,20,60],60) # in minutes
    test_alpha = [0, 5, 50, 250, 500, 5000, 25000, 50000] # in pM
    test_beta = [0, 0.1, 1, 5, 10, 100, 200, 1000] # in pM
    alpha_scan = []
    beta_scan = []
    for dose in range(len(test_alpha)):
        alpha_scan.append(p_timecourse("IFN_Models.Mixed_IFN_ppCompatible", np.multiply(test_times,60),
                            [['TotalpSTAT','pSTAT1']], suppress=True, 
                            parameters=p+[['Ia',1E-12*6.022E23*1E-5*test_alpha[dose]],['Ib',0]], 
                            scan=0)['TotalpSTAT'])
        beta_scan.append(p_timecourse("IFN_Models.Mixed_IFN_ppCompatible", np.multiply(test_times,60),
                            [['TotalpSTAT','pSTAT1']], suppress=True, 
                            parameters=p+[['Ib',1E-12*6.022E23*1E-5*test_beta[dose]],['Ia',0]], 
                            scan=0)['TotalpSTAT'])
    if sf_flag==True:
         # In this case we do not find the score but rather the optimal scale factor
        print("Using weighted MSE")
        score = MSE(alpha_scan,alpha_experimental_scan,beta_scan,beta_experimental_scan,sf_flag=True)
    else:
        print("Using weighted MSE")
        score = MSE(alpha_scan,alpha_experimental_scan,beta_scan,beta_experimental_scan,)
    return score
 
# p_list (list) = [p1,p2,...] where p=['name', min_value, max_value]
# experimental_scan (list) = 2D scan
# n (int) = number of points to check for each parameter     
def stepwise_fit(p_list,alpha_experimental_scan,beta_experimental_scan,n):
    print("Beginning stepwise fit")
    final_fit=[]
    nit=len(p_list)
    initial_score=0
    for i in range(nit):
        score = 0
        best_param=[]
        for p in p_list:
            res = [score_parameter([[p[0],j]]+final_fit,alpha_experimental_scan,beta_experimental_scan) for j in np.linspace(p[1],p[2],n)]  
            best_param_value = np.linspace(p[1],p[2],n)[res.index(min(res))]
            if min(res)<score or score==0:
                if initial_score==0:
                    initial_score=min(res)
                score=min(res)
                best_param=[p[0],best_param_value]
        final_fit+=[best_param]
        p_list=[j for j in p_list if j[0] not in [el[0] for el in final_fit]]
    print("Score improved from {} after one iteration to {} after {} iterations".format(initial_score,score,nit))
    return final_fit

def plot_experimental_dr_curves(alpha_data,beta_data,times,alpha_doses,beta_doses,title):
    f, [ax1,ax2] = plt.subplots(ncols=2,figsize=(8,6))
    f.suptitle(title)
    ntimes = len(times)
    alpha_palette = sns.color_palette("Reds",ntimes)
    beta_palette = sns.color_palette("Greens",ntimes)
    ax1.set(xscale='log',yscale='linear')
    ax1.set_xlabel('IFN (pM)')
    ax1.set_ylabel('pSTAT')
    ax1.set_title('Experimental IFNa dose response curves')
    ax2.set(xscale='log',yscale='linear')
    ax2.set_xlabel('IFN (pM)')
    ax2.set_ylabel('pSTAT')
    ax2.set_title('Experimental IFNb dose response curves')
    
    for time in range(ntimes):
        alpha_dose_response = [dose[time] for dose in alpha_data]
        beta_dose_response = [dose[time] for dose in beta_data]        
        sns.lineplot(x=alpha_doses, y=alpha_dose_response, ax=ax1, legend="full", label=str(times[time])+" min", color=alpha_palette[time])
        sns.lineplot(x=beta_doses, y=beta_dose_response, ax=ax2, legend="full", label=str(times[time])+" min", color=beta_palette[time])
    
    plt.savefig(cwd+"\\Mixed_IFN_Figures\\"+title+".pdf")
         
def main():
    # Load testing data
    train_doses = [0,0.06,0.32,1.6,8,40,200,1000]
    data = pd.read_csv("20181031_pSTAT1_Table.csv")
    experimental_Tcell = data.loc[:,"T cells"].values.reshape(8,8)
    experimental_NKcell = data.loc[:,"NK cells"].values.reshape(8,8)
    experimental_Bcell = data.loc[:,"B cells"].values.reshape(8,8)
    
    # Flip the row order so it matches output of 2D scan for scoring purposes 
    #   (ie. beta increasing along a row, alpha increasing down columns)(ie. H1 (the Jaki well) at [0][0])
    experimental_Tcell = np.flipud([[el for el in row] for row in experimental_Tcell])
    experimental_NKcell = np.flipud([[el for el in row] for row in experimental_NKcell])
    experimental_Bcell = np.flipud([[el for el in row] for row in experimental_Bcell])
    # Subtract background located in H1
    experimental_Tcell = np.subtract(experimental_Tcell, experimental_Tcell[0][0])
    experimental_Bcell = np.subtract(experimental_Bcell, experimental_Bcell[0][0])
    experimental_NKcell = np.subtract(experimental_NKcell, experimental_NKcell[0][0])
    # Normalize by max response in dataset
    normfac = max([np.max([[r[0] for r in experimental_Tcell],experimental_Tcell[0]]),
                   np.max([[r[0] for r in experimental_Bcell],experimental_Bcell[0]]),
                   np.max([[r[0] for r in experimental_NKcell],experimental_NKcell[0]])])
    experimental_Tcell = np.divide(experimental_Tcell, normfac)
    experimental_Bcell = np.divide(experimental_Bcell, normfac)
    experimental_NKcell = np.divide(experimental_NKcell, normfac)
    

    # Load training data (increasing concentration down a column, increasing time along a row)
    testdata = pd.read_csv("20181113_B6_IFNs_Dose-Response.csv")
    test_times = [2.5,5,7.5,10,20,60] # in minutes
    test_doses_alpha = [0, 5, 50, 250, 500, 5000, 25000, 50000] # in pM
    test_doses_beta = [0, 0.1, 1, 5, 10, 100, 200, 1000] # in pM
    
    test_Bcell = testdata.loc[:,"Lymphocytes/B cells | Geometric Mean (Comp-APC-A)"].values.reshape(8,12)
    test_Bcell_alpha = [[concentration[time] for time in [0,2,4,6,8,10]] for concentration in test_Bcell]
    test_Bcell_beta = [[concentration[time] for time in [1,3,5,7,9,11]] for concentration in test_Bcell]

    test_Tcell = testdata.loc[:,"Lymphocytes/T cells | Geometric Mean (Comp-APC-A)"].values.reshape(8,12)
    test_Tcell_alpha = [[concentration[time] for time in [0,2,4,6,8,10]] for concentration in test_Tcell]
    test_Tcell_beta = [[concentration[time] for time in [1,3,5,7,9,11]] for concentration in test_Tcell]
    
    test_NKcell = testdata.loc[:,"Lymphocytes/NonT, nonB | Geometric Mean (Comp-APC-A)"].values.reshape(8,12)
    test_NKcell_alpha = [[concentration[time] for time in [0,2,4,6,8,10]] for concentration in test_NKcell]
    test_NKcell_beta = [[concentration[time] for time in [1,3,5,7,9,11]] for concentration in test_NKcell]
    
    # Subtract background
    test_Tcell_alpha = np.subtract(test_Tcell_alpha,np.mean(test_Tcell_alpha[0]))
    test_Tcell_beta = np.subtract(test_Tcell_beta,np.mean(test_Tcell_beta[0]))
    test_Bcell_alpha = np.subtract(test_Bcell_alpha,np.mean(test_Bcell_alpha[0]))
    test_Bcell_beta = np.subtract(test_Bcell_beta,np.mean(test_Bcell_beta[0]))
    test_NKcell_alpha = np.subtract(test_NKcell_alpha,np.mean(test_NKcell_alpha[0]))
    test_NKcell_beta = np.subtract(test_NKcell_beta,np.mean(test_NKcell_beta[0]))

    # Normalize all data by globally maximal response (test_Bcell_alpha is max)
    normfac2 = np.max([np.max(test_Tcell_alpha),np.max(test_Tcell_beta),np.max(test_Bcell_alpha),np.max(test_Bcell_beta),np.max(test_NKcell_alpha),np.max(test_NKcell_beta)])
    
    test_Tcell_alpha = np.divide(test_Tcell_alpha,normfac2)
    test_Tcell_beta = np.divide(test_Tcell_beta,normfac2)
    test_Bcell_alpha = np.divide(test_Bcell_alpha,normfac2)
    test_Bcell_beta = np.divide(test_Bcell_beta,normfac2)
    test_NKcell_alpha = np.divide(test_NKcell_alpha,normfac2)
    test_NKcell_beta = np.divide(test_NKcell_beta,normfac2)

    #plot_experimental_dr_curves(test_Tcell_alpha,test_Tcell_beta,test_times,test_doses_alpha,test_doses_beta,"Experimental_Tcell_DR")
    #plot_experimental_dr_curves(test_Bcell_alpha,test_Bcell_beta,test_times,test_doses_alpha,test_doses_beta,"Experimental_Bcell_DR")
    #plot_experimental_dr_curves(test_NKcell_alpha,test_NKcell_beta,test_times,test_doses_alpha,test_doses_beta,"Experimental_NonT-NonB_cell_DR")
# =============================================================================

# T cells
    # Fit model to mixed IFN data
    Tbest_params_list = stepwise_fit([['krec_b1',1E-5,1E-3],['krec_b2',1E-4,1E-2],['krec_a1',1E-5,1E-3],['krec_a2',1E-4,1E-2],['R2',200,12000],['R1',200,12000],['kSOCSon',1E-7,1E-4],['kpa',1E-7,1E-4]],test_Tcell_alpha,test_Tcell_beta,12)
    #Tbest_params_list = [['kpa', 9.9999999999999995e-08], ['kSOCSon', 9.9999999999999995e-08], ['R2', 200.0], ['R1', 200.0]]
    T_sf = score_parameter(Tbest_params_list,test_Tcell_alpha,test_Tcell_beta,sf_flag=True)[0]

    print("The best fit for T cells was")
    print(Tbest_params_list)
    with open('stepwise_results.txt','w') as f:
        f.write("T cell fit:\n")
        f.write(str(Tbest_params_list))
# =============================================================================
#     # Plot best fit model as arcsinh heatmap
#     res = IFN_2Dscan("IFN_Models.Mixed_IFN_ppCompatible",
#                ["Ib",np.multiply(train_doses,1E-12*6.022E23*1E-5)],
#                ["Ia",np.multiply(train_doses,1E-12*6.022E23*1E-5)],
#                [0,300,600,900,1200],
#                ['TotalpSTAT','pSTAT1'],
#                doseNorm=6.022E23*1E-5,
#                custom_params=Tbest_params_list,
#                suppress=True) 
#     
#     sf = score_parameter(Tbest_params_list,experimental_Tcell,sf_flag=True)[0]
#     res = np.multiply(res,sf)
#     arcsinh_response = []
#     for row in res:
#         new_row = []
#         for item in row:
#             new_row.append([item[0]/1E-12,item[1]/1E-12,np.arcsinh(item[2])])
#         arcsinh_response.append(new_row)
#     IFN_heatmap(arcsinh_response, cwd+"\\Mixed_IFN_Figures\\"+"Tcell Stepwise Fit - Ib (pM)", 'Ia (pM)')
# 
#     # Plot quality of fit as a heatmap
#     percent_error = np.divide(np.subtract([[el[2] for el in row] for row in res],experimental_Tcell),experimental_Tcell)
#     labelled_perecent_error = [[[res[i][j][0]/(1E-12), res[i][j][1]/(1E-12), abs(percent_error[i][j])] for j in range(8)] for i in range(8)]
#     IFN_heatmap(labelled_perecent_error, cwd+"\\Mixed_IFN_Figures\\"+"Tcell Stepwise Percent error - Ib (pM)", "Ia (pM)")
# 
# =============================================================================
    # Test best fit model on test data
    #test_alpha_dose_response = [dose[4] for dose in test_Tcell_alpha]
    #test_beta_dose_response = [dose[4] for dose in test_Tcell_beta]        
    
    min_doseA = np.log10(0.01) 
    max_doseA = np.log10(50000) 
    min_doseB = np.log10(0.01) 
    max_doseB = np.log10(50000) 
    model_doseA = np.multiply([0]+list(np.logspace(min_doseA,max_doseA,20)),6.022E23*1E-5*1E-12)
    model_doseB = np.multiply([0]+list(np.logspace(min_doseB,max_doseB,20)),6.022E23*1E-5*1E-12)
# =============================================================================
#     a_dr = np.multiply(sf,p_doseresponse("IFN_Models.Mixed_IFN_ppCompatible", ['Ia',model_doseA], 
#                                          np.multiply(test_times,60), [['TotalpSTAT','pSTAT1']],
#                                          axes_labels = ['',''], title = '', suppress=True, 
#                                          Norm=None, parameters=Tbest_params_list+[['Ib',0]], 
#                                          dose_axis_norm=False, scan=0)[0])
#     b_dr = np.multiply(sf,p_doseresponse("IFN_Models.Mixed_IFN_ppCompatible", ['Ib',model_doseB], 
#                                          np.multiply(test_times,60), [['TotalpSTAT','pSTAT1']],
#                                          axes_labels = ['',''], title = '', suppress=True, 
#                                          Norm=None, parameters=Tbest_params_list+[['Ia',0]], 
#                                          dose_axis_norm=False, scan=0)[0])
# =============================================================================
    
# =============================================================================
#     # Plot isolated IFN trajectories
#     simulated_response = [[el[2] for el in row] for row in res]
#     fig32, [[ax1,ax2],[ax3,ax4],[ax5,ax6]] = plt.subplots(nrows=3,ncols=2,figsize=(12,12))
#     for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
#         ax.set(xscale='log',yscale='linear')
#     ax5.set_xlabel(r"IFN$\alpha$ (pM)")
#     ax6.set_xlabel(r"IFN$\beta$ (pM)")
#     ax1.set_ylabel('pSTAT')
#     ax3.set_ylabel('pSTAT')
#     ax5.set_ylabel('pSTAT')
#     
#     ax1.set_title('T cell')
#     ax2.set_title('T cell')
#     ax1.scatter(train_doses[1:],[r[0] for r in experimental_Tcell[1:]],c='r')
#     ax2.scatter(train_doses[1:],experimental_Tcell[0][1:],c='g')
#     ax1.plot(train_doses[1:],[el[0] for el in simulated_response[1:]],'r')    
#     ax2.plot(train_doses[1:],simulated_response[0][1:],'g')  
#     
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#     textstr = '\n'.join([r'{}={:.2E}'.format(r[0],r[1]) for r in Tbest_params_list])
#     ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
#         verticalalignment='top', bbox=props)
#     ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14,
#         verticalalignment='top', bbox=props)
# =============================================================================
    
# =============================================================================
# B cells    
    Bbest_params_list = stepwise_fit([['krec_b1',1E-5,1E-3],['krec_b2',1E-4,1E-2],['krec_a1',1E-5,1E-3],['krec_a2',1E-4,1E-2],['R2',200,12000],['R1',200,12000],['kSOCSon',1E-7,1E-4],['kpa',1E-7,1E-4]],test_Bcell_alpha,test_Bcell_beta,12)
    #Bbest_params_list = [['kpa', 9.9999999999999995e-08], ['kSOCSon', 9.9999999999999995e-08], ['R2', 1272.7272727272727], ['R1', 1272.7272727272727]]
    print("The best fit for B cells was")
    print(Bbest_params_list)
    with open('stepwise_results.txt','a') as f:
        f.write("\nB cell fit:\n")
        f.write(str(Bbest_params_list))
    B_sf = score_parameter(Bbest_params_list,test_Bcell_alpha,test_Bcell_beta,sf_flag=True)[0]
 
# =============================================================================
#     res = IFN_2Dscan("IFN_Models.Mixed_IFN_ppCompatible",
#                ["Ib",np.multiply([0,0.06,0.32,1.6,8,40,200,1000],1E-12*6.022E23*1E-5)],
#                ["Ia",np.multiply([0,0.06,0.32,1.6,8,40,200,1000],1E-12*6.022E23*1E-5)],
#                [0,300,600,900,1200],
#                ['TotalpSTAT','pSTAT1'],
#                doseNorm=6.022E23*1E-5,
#                custom_params=Bbest_params_list,
#                suppress=True) 
#     
#     res = np.multiply(res,sf)
#     arcsinh_response = []
#     for row in res:
#         new_row = []
#         for item in row:
#             new_row.append([item[0]/1E-12,item[1]/1E-12,np.arcsinh(item[2])])
#         arcsinh_response.append(new_row)
#     IFN_heatmap(arcsinh_response, "Bcell Stepwise Fit - Ib (pM)", 'Ia (pM)')
# 
#     # Plot quality of fit
#     percent_error = np.divide(np.subtract([[el[2] for el in row] for row in res],experimental_Bcell),experimental_Bcell)
#     labelled_perecent_error = [[[res[i][j][0]/(1E-12), res[i][j][1]/(1E-12), abs(percent_error[i][j])] for j in range(8)] for i in range(8)]
#     IFN_heatmap(labelled_perecent_error, "Bcell Stepwise Percent error - Ib (pM)", "Ia (pM)") 
# =============================================================================
    
   
# Create plot of best fit model vs data
    simfig, [axes1,axes2,axes3] = plt.subplots(nrows=3,ncols=2,figsize=(8,8.5))
    simfig.suptitle("Compare fit to experimental data")
    ntimes = 3
    alpha_palette = sns.color_palette("Reds",ntimes)
    beta_palette = sns.color_palette("Greens",ntimes)
    
# Prepare all experimental data curves    
    Ttest_alpha_DR5 = [dose[1] for dose in test_Tcell_alpha]
    Ttest_beta_DR5 = [dose[1] for dose in test_Tcell_beta]            
    Ttest_alpha_DR20 = [dose[4] for dose in test_Tcell_alpha]
    Ttest_beta_DR20 = [dose[4] for dose in test_Tcell_beta]        
    Ttest_alpha_DR60 = [dose[5] for dose in test_Tcell_alpha]
    Ttest_beta_DR60 = [dose[5] for dose in test_Tcell_beta]        

    Btest_alpha_DR5 = [dose[1] for dose in test_Bcell_alpha]
    Btest_beta_DR5 = [dose[1] for dose in test_Bcell_beta]            
    Btest_alpha_DR20 = [dose[4] for dose in test_Bcell_alpha]
    Btest_beta_DR20 = [dose[4] for dose in test_Bcell_beta]        
    Btest_alpha_DR60 = [dose[5] for dose in test_Bcell_alpha]
    Btest_beta_DR60 = [dose[5] for dose in test_Bcell_beta]     
    
# Compute all necessary model simulations for smooth plotting model vs data
    Tmodel_res5 = IFN_2Dscan("IFN_Models.Mixed_IFN_ppCompatible",
               ["Ib",model_doseA],
               ["Ia",model_doseB],
               [0,120,300],
               ['TotalpSTAT','pSTAT1'],
               doseNorm=6.022E23*1E-5,
               custom_params=Tbest_params_list,
               suppress=True) 
    
    Tmodel_res20 = IFN_2Dscan("IFN_Models.Mixed_IFN_ppCompatible",
               ["Ib",model_doseA],
               ["Ia",model_doseB],
               [0,300,1200],
               ['TotalpSTAT','pSTAT1'],
               doseNorm=6.022E23*1E-5,
               custom_params=Tbest_params_list,
               suppress=True) 
    Tmodel_res60 = IFN_2Dscan("IFN_Models.Mixed_IFN_ppCompatible",
               ["Ib",model_doseA],
               ["Ia",model_doseB],
               [0,300,3600],
               ['TotalpSTAT','pSTAT1'],
               doseNorm=6.022E23*1E-5,
               custom_params=Tbest_params_list,
               suppress=True) 
    
    Bmodel_res5 = IFN_2Dscan("IFN_Models.Mixed_IFN_ppCompatible",
               ["Ib",model_doseA],
               ["Ia",model_doseB],
               [0,120,300],
               ['TotalpSTAT','pSTAT1'],
               doseNorm=6.022E23*1E-5,
               custom_params=Bbest_params_list,
               suppress=True) 
    Bmodel_res20 = IFN_2Dscan("IFN_Models.Mixed_IFN_ppCompatible",
               ["Ib",model_doseA],
               ["Ia",model_doseB],
               [0,300,1200],
               ['TotalpSTAT','pSTAT1'],
               doseNorm=6.022E23*1E-5,
               custom_params=Bbest_params_list,
               suppress=True) 
    Bmodel_res60 = IFN_2Dscan("IFN_Models.Mixed_IFN_ppCompatible",
               ["Ib",model_doseA],
               ["Ia",model_doseB],
               [0,300,3600],
               ['TotalpSTAT','pSTAT1'],
               doseNorm=6.022E23*1E-5,
               custom_params=Bbest_params_list,
               suppress=True) 
    Tmodel_sim5 = [[T_sf*el[2] for el in row] for row in Tmodel_res5]
    Tmodel_sim20 = [[T_sf*el[2] for el in row] for row in Tmodel_res20]
    Tmodel_sim60 = [[T_sf*el[2] for el in row] for row in Tmodel_res60]
    Bmodel_sim5 = [[B_sf*el[2] for el in row] for row in Bmodel_res5]
    Bmodel_sim20 = [[B_sf*el[2] for el in row] for row in Bmodel_res20]
    Bmodel_sim60 = [[B_sf*el[2] for el in row] for row in Bmodel_res60]

    # Plot T cell specific fits
    axes1[0].set(xscale='log',yscale='linear')
    axes1[0].set_xlabel('IFN (pM)')
    axes1[0].set_ylabel('pSTAT')
    axes1[0].set_title('T cells with IFNa')
    axes1[1].set(xscale='log',yscale='linear')
    axes1[1].set_xlabel('IFN (pM)')
    axes1[1].set_ylabel('pSTAT')
    axes1[1].set_title('T cells with IFNb')
    # Plot the testing data
    sns.scatterplot(x=train_doses[1:],y=[r[0] for r in experimental_Tcell[1:]], ax=axes1[0], legend="full", label="Testing Data", color=alpha_palette[2])
    sns.scatterplot(x=train_doses[1:],y=experimental_Tcell[0][1:], ax=axes1[1], legend="full", label="Testing Data", color=beta_palette[2])
    # Plot the model at 20 min
    sns.lineplot(x=np.divide(model_doseA,6.022E23*1E-5*1E-12)[1:],y=[el[0] for el in Tmodel_sim20[1:]], ax=axes1[0], legend="full",label="Model",color=alpha_palette[1])
    sns.lineplot(x=np.divide(model_doseB,6.022E23*1E-5*1E-12)[1:],y=Tmodel_sim20[0][1:],ax=axes1[1], legend="full",label="Model",color=beta_palette[1]) 
    # Plot the training data at 20 min    
    sns.scatterplot(x=test_doses_alpha[1::], y=Ttest_alpha_DR20[1::], ax=axes1[0], legend="full", label="Training Data", color=alpha_palette[0])
    sns.scatterplot(x=test_doses_beta[1::], y=Ttest_beta_DR20[1::], ax=axes1[1], legend="full", label="Training Data", color=beta_palette[0])
     
# Plot B cell specific fits
    axes2[0].set(xscale='log',yscale='linear')
    axes2[0].set_xlabel('IFN (pM)')
    axes2[0].set_ylabel('pSTAT')
    axes2[0].set_title('B cells with IFNa')
    axes2[1].set(xscale='log',yscale='linear')
    axes2[1].set_xlabel('IFN (pM)')
    axes2[1].set_ylabel('pSTAT')
    axes2[1].set_title('B cells with IFNb')
    # Plot the model at 20 min
    sns.lineplot(x=np.divide(model_doseA,6.022E23*1E-5*1E-12)[1:],y=[el[0] for el in Bmodel_sim20[1:]], ax=axes2[0], legend="full",label="Model",color=alpha_palette[1])
    sns.lineplot(x=np.divide(model_doseB,6.022E23*1E-5*1E-12)[1:],y=Bmodel_sim20[0][1:],ax=axes2[1], legend="full",label="Model",color=beta_palette[1]) 
    # Plot the testing data at 20 min    
    sns.scatterplot(x=test_doses_alpha[1::], y=Btest_alpha_DR20[1::], ax=axes2[0], legend="full", label="Training Data", color=alpha_palette[0])
    sns.scatterplot(x=test_doses_beta[1::], y=Btest_beta_DR20[1::], ax=axes2[1], legend="full", label="Training Data", color=beta_palette[0])
    # Plot the testing data at 20 min
    sns.scatterplot(x=train_doses[1:],y=[r[0] for r in experimental_Bcell[1:]], ax=axes2[0], legend="full", label="Testing Data", color=alpha_palette[2])
    sns.scatterplot(x=train_doses[1:],y=experimental_Bcell[0][1:], ax=axes2[1], legend="full", label="Testing Data", color=beta_palette[2])

# Plot 5, 20 and 60 minute simulated response and data for IFNa and IFNb
    alpha_palette = sns.color_palette("Reds",5)
    beta_palette = sns.color_palette("Greens",5)
    
    # Plot the comparison
    axes3[0].set(xscale='log',yscale='linear')
    axes3[0].set_xlabel('IFN (pM)')
    axes3[0].set_ylabel('pSTAT')
    axes3[0].set_title('T cells')
    axes3[1].set(xscale='log',yscale='linear')
    axes3[1].set_xlabel('IFN (pM)')
    axes3[1].set_ylabel('pSTAT')
    axes3[1].set_title('B cells')
    # Plot the testing data for each cell type
    sns.scatterplot(x=train_doses[1:],y=[r[0] for r in experimental_Tcell[1:]], ax=axes3[0], legend=False, label="Testing Data", color=alpha_palette[2])
    sns.scatterplot(x=train_doses[1:],y=experimental_Tcell[0][1:], ax=axes3[0], legend=False, label="Testing Data", color=beta_palette[2])
    
    sns.scatterplot(x=train_doses[1:],y=[r[0] for r in experimental_Bcell[1:]], ax=axes3[1], legend=False, label="Testing Data", color=alpha_palette[2])
    sns.scatterplot(x=train_doses[1:],y=experimental_Bcell[0][1:], ax=axes3[1], legend=False, label="Testing Data", color=beta_palette[2])
    
    # Plot the models for T cell   
    sns.lineplot(x=np.divide(model_doseA,6.022E23*1E-5*1E-12)[1:],y=[el[0] for el in Tmodel_sim5[1:]], ax=axes3[0], legend=False,label="5 min",color=alpha_palette[1])
    sns.lineplot(x=np.divide(model_doseB,6.022E23*1E-5*1E-12)[1:],y=Tmodel_sim5[0][1:],ax=axes3[0], legend=False,label="5 min",color=beta_palette[1]) 
    sns.lineplot(x=np.divide(model_doseA,6.022E23*1E-5*1E-12)[1:],y=[el[0] for el in Tmodel_sim20[1:]], ax=axes3[0], legend=False,label="20 min",color=alpha_palette[2])
    sns.lineplot(x=np.divide(model_doseB,6.022E23*1E-5*1E-12)[1:],y=Tmodel_sim20[0][1:],ax=axes3[0], legend=False,label="20 min",color=beta_palette[2]) 
    sns.lineplot(x=np.divide(model_doseA,6.022E23*1E-5*1E-12)[1:],y=[el[0] for el in Tmodel_sim60[1:]], ax=axes3[0], legend=False,label="60 min",color=alpha_palette[3])
    sns.lineplot(x=np.divide(model_doseB,6.022E23*1E-5*1E-12)[1:],y=Tmodel_sim60[0][1:],ax=axes3[0], legend=False,label="60 min",color=beta_palette[3]) 
    
    # Plot the training data for T cells
    #sns.scatterplot(x=test_doses_alpha[1::], y=Ttest_alpha_DR5[1::], ax=axes3[0], legend=False, label="Training Data 5 min", color=alpha_palette[0],marker='s')
    #sns.scatterplot(x=test_doses_beta[1::], y=Ttest_beta_DR5[1::], ax=axes3[0], legend=False, label="Training Data 5 min", color=beta_palette[0],marker='s')    
    #sns.scatterplot(x=test_doses_alpha[1::], y=Ttest_alpha_DR20[1::], ax=axes3[0], legend=False, label="Training Data 20 min", color=alpha_palette[0])
    #sns.scatterplot(x=test_doses_beta[1::], y=Ttest_beta_DR20[1::], ax=axes3[0], legend=False, label="Training Data 20 min", color=beta_palette[0])
    #sns.scatterplot(x=test_doses_alpha[1::], y=Ttest_alpha_DR60[1::], ax=axes3[0], legend=False, label="Training Data 60 min", color=alpha_palette[0],marker='v')
    #sns.scatterplot(x=test_doses_beta[1::], y=Ttest_beta_DR60[1::], ax=axes3[0], legend=False, label="Training Data 60 min", color=beta_palette[0],marker='v')

    # Plot the model sims for B cells
    sns.lineplot(x=np.divide(model_doseA,6.022E23*1E-5*1E-12)[1:],y=[el[0] for el in Bmodel_sim5[1:]], ax=axes3[1], legend=False,label="5 min",color=alpha_palette[1])
    sns.lineplot(x=np.divide(model_doseB,6.022E23*1E-5*1E-12)[1:],y=Bmodel_sim5[0][1:],ax=axes3[1], legend=False,label="5 min",color=beta_palette[1]) 
    sns.lineplot(x=np.divide(model_doseA,6.022E23*1E-5*1E-12)[1:],y=[el[0] for el in Bmodel_sim20[1:]], ax=axes3[1], legend=False,label="20 min",color=alpha_palette[2])
    sns.lineplot(x=np.divide(model_doseB,6.022E23*1E-5*1E-12)[1:],y=Bmodel_sim20[0][1:],ax=axes3[1], legend=False,label="20 min",color=beta_palette[2]) 
    sns.lineplot(x=np.divide(model_doseA,6.022E23*1E-5*1E-12)[1:],y=[el[0] for el in Bmodel_sim60[1:]], ax=axes3[1], legend=False,label="60 min",color=alpha_palette[3])
    sns.lineplot(x=np.divide(model_doseB,6.022E23*1E-5*1E-12)[1:],y=Bmodel_sim60[0][1:],ax=axes3[1], legend=False,label="60 min",color=beta_palette[3]) 

    # Plot the training data for B cells
    #sns.scatterplot(x=test_doses_alpha[1::], y=Btest_alpha_DR5[1::], ax=axes3[1], legend=False, label="Training Data 5 min", color=alpha_palette[0],marker='s')
    #sns.scatterplot(x=test_doses_beta[1::], y=Btest_beta_DR5[1::], ax=axes3[1], legend=False, label="Training Data 5 min", color=beta_palette[0],marker='s')    
    #sns.scatterplot(x=test_doses_alpha[1::], y=Btest_alpha_DR20[1::], ax=axes3[1], legend=False, label="Training Data 20 min", color=alpha_palette[0])
    #sns.scatterplot(x=test_doses_beta[1::], y=Btest_beta_DR20[1::], ax=axes3[1], legend=False, label="Training Data 20 min", color=beta_palette[0])
    #sns.scatterplot(x=test_doses_alpha[1::], y=Btest_alpha_DR60[1::], ax=axes3[1], legend=False, label="Training Data 60 min", color=alpha_palette[0],marker='v')
    #sns.scatterplot(x=test_doses_beta[1::], y=Btest_beta_DR60[1::], ax=axes3[1], legend=False, label="Training Data 60 min", color=beta_palette[0],marker='v')
    
    #axes3[0].legend(loc='center right', bbox_to_anchor=(1, 0.5))
    #axes3[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    simfig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    simfig.savefig(cwd+"\\Mixed_IFN_Figures\\"+"Out_of_sample.pdf")


    
if __name__ == '__main__':
    main()
    