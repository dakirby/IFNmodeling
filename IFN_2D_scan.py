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

def MSE(simulated_scan, experimental_scan,sf_flag=False):
    def score(sf, simulated_scan, experimental_scan):
        score = np.sum(np.square(np.subtract(np.multiply(simulated_scan,sf), experimental_scan)))
        return score
    optimal_score = minimize(score,[40],args=(simulated_scan, experimental_scan))['fun']
    if sf_flag==True:
        return minimize(score,[40],args=(simulated_scan, experimental_scan))['x']
    return optimal_score

# p (list) = [['name',value]]
# experimental_scan (list) = 2D scan
# returns the mean square error
def score_parameter(p,experimental_scan,sf_flag=False):
     res = IFN_2Dscan("IFN_Models.Mixed_IFN_ppCompatible",
            ["Ib",np.multiply([0,0.06,0.32,1.6,8,40,200,1000],1E-12*6.022E23*1E-5)],
            ["Ia",np.multiply([0,0.06,0.32,1.6,8,40,200,1000],1E-12*6.022E23*1E-5)],
            [0,300,600,900,1200],
            ['TotalpSTAT','pSTAT1'],
            doseNorm=6.022E23*1E-5,
            custom_params=p,
            suppress=True, verbose=0)
     scan = [[el[2] for el in r] for r in res]
     if sf_flag==True:
         # In this case we do not find the score but rather the optimal scale factor
         score = MSE(scan,experimental_scan,sf_flag=True)
     else:
         score = MSE(scan,experimental_scan)
     return score
 
# p_list (list) = [p1,p2,...] where p=['name', min_value, max_value]
# experimental_scan (list) = 2D scan
# n (int) = number of points to check for each parameter     
def stepwise_fit(p_list,experimental_scan,n):
    print("Beginning stepwise fit")
    final_fit=[]
    nit=len(p_list)
    initial_score=0
    for i in range(nit):
        score = 0
        best_param=[]
        for p in p_list:
            res = [score_parameter([[p[0],j]]+final_fit,experimental_scan) for j in np.linspace(p[1],p[2],n)]  
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
    # Load training data
    train_doses = [0,0.06,0.32,1.6,8,40,200,1000]
    data = pd.read_csv("20181031_pSTAT1_Table.csv")
    experimental_Tcell = data.loc[:,"T cells"].values.reshape(8,8)
    experimental_NKcell = data.loc[:,"NK cells"].values.reshape(8,8)
    experimental_Bcell = data.loc[:,"B cells"].values.reshape(8,8)
    
    # Flip the row order so it matches output of 2D scan for scoring purposes
    experimental_Tcell = np.flipud([[el for el in row] for row in experimental_Tcell])
    experimental_NKcell = np.flipud([[el for el in row] for row in experimental_NKcell])
    experimental_Bcell = np.flipud([[el for el in row] for row in experimental_Bcell])
    # Normalize by max response in dataset
    normfac = max([np.max(experimental_Tcell),np.max(experimental_Bcell),np.max(experimental_NKcell)])
    experimental_Tcell = np.divide(experimental_Tcell, normfac)
    experimental_Bcell = np.divide(experimental_Bcell, normfac)
    experimental_NKcell = np.divide(experimental_NKcell, normfac)
    # Subtract background located in H1
    experimental_Tcell = np.subtract(experimental_Tcell, experimental_Tcell[0][0])
    experimental_Bcell = np.subtract(experimental_Bcell, experimental_Bcell[0][0])
    experimental_NKcell = np.subtract(experimental_NKcell, experimental_NKcell[0][0])
    

    # Load test data
    testdata = pd.read_csv("20181113_B6_IFNs_Dose-Response.csv")
    test_times = [2.5,5,7.5,10,20,60] # in minutes
    test_doses_alpha = [0, 5, 50, 250, 500, 5000, 25000, 50000] # in pM
    test_doses_beta = [0, 0.1, 1, 5, 10, 100, 200, 1000] # in pM
    normfac2 = np.max([[el[1],el[2],el[3]] for el in testdata.values])
    
    test_Bcell = testdata.loc[:,"Lymphocytes/B cells | Geometric Mean (Comp-APC-A)"].values.reshape(8,12)
    test_Bcell_alpha = [[concentration[time] for time in [0,2,4,6,8,10]] for concentration in test_Bcell]
    test_Bcell_beta = [[concentration[time] for time in [1,3,5,7,9,11]] for concentration in test_Bcell]
    
    test_Tcell = testdata.loc[:,"Lymphocytes/T cells | Geometric Mean (Comp-APC-A)"].values.reshape(8,12)
    test_Tcell_alpha = [[concentration[time] for time in [0,2,4,6,8,10]] for concentration in test_Tcell]
    test_Tcell_beta = [[concentration[time] for time in [1,3,5,7,9,11]] for concentration in test_Tcell]
    
    test_NKcell = testdata.loc[:,"Lymphocytes/NonT, nonB | Geometric Mean (Comp-APC-A)"].values.reshape(8,12)
    test_NKcell_alpha = [[concentration[time] for time in [0,2,4,6,8,10]] for concentration in test_NKcell]
    test_NKcell_beta = [[concentration[time] for time in [1,3,5,7,9,11]] for concentration in test_NKcell]

    # Normalize all data by globally maximal response
    test_Tcell_alpha = np.divide(test_Tcell_alpha,normfac2)
    test_Tcell_beta = np.divide(test_Tcell_beta,normfac2)
    test_Bcell_alpha = np.divide(test_Bcell_alpha,normfac2)
    test_Bcell_beta = np.divide(test_Bcell_beta,normfac2)
    test_NKcell_alpha = np.divide(test_NKcell_alpha,normfac2)
    test_NKcell_beta = np.divide(test_NKcell_beta,normfac2)

    # Subtract background
    test_Tcell_alpha = np.subtract(test_Tcell_alpha,test_Tcell_alpha[0][0])
    test_Tcell_beta = np.subtract(test_Tcell_beta,test_Tcell_beta[0][0])
    test_Bcell_alpha = np.subtract(test_Bcell_alpha,test_Bcell_alpha[0][0])
    test_Bcell_beta = np.subtract(test_Bcell_beta,test_Bcell_beta[0][0])
    test_NKcell_alpha = np.subtract(test_NKcell_alpha,test_NKcell_alpha[0][0])
    test_NKcell_beta = np.subtract(test_NKcell_beta,test_NKcell_beta[0][0])


    #plot_experimental_dr_curves(test_Tcell_alpha,test_Tcell_beta,test_times,test_doses_alpha,test_doses_beta,"Experimental_Tcell_DR")
    #plot_experimental_dr_curves(test_Bcell_alpha,test_Bcell_beta,test_times,test_doses_alpha,test_doses_beta,"Experimental_Bcell_DR")
    #plot_experimental_dr_curves(test_NKcell_alpha,test_NKcell_beta,test_times,test_doses_alpha,test_doses_beta,"Experimental_NonT-NonB_cell_DR")
# =============================================================================

# T cells
    # Fit model to mixed IFN data
    #best_params_list = stepwise_fit([['R2',200,12000],['R1',200,12000],['kSOCSon',1E-7,1E-4],['kpa',1E-7,1E-4]],experimental_Tcell,12)
    best_params_list = [['R1', 200.0], ['R2', 3418.181818181818], ['kSOCSon', 9.9999999999999995e-08], ['kpa', 9.9999999999999995e-08]]
    #best_params_list = [['kpa', 0.0001], ['pS', 1000.0], ['R1', 1272.7272727272727], ['R2', 2345.4545454545455], ['kSOCSon', 9.181818181818184e-06]]
    # old values #best_params_list = [['R2', 5257.1428571428569], ['R1', 1042.8571428571429], ['kpa', 8.0714285714285719e-07], ['kSOCSon', 9.9999999999999995e-08]]

    print("The best fit for T cells was")
    print(best_params_list)
    with open('stepwise_results.txt','w') as f:
        f.write("T cell fit:\n")
        f.write(str(best_params_list))
    
    # Plot best fit model as arcsinh heatmap
    res = IFN_2Dscan("IFN_Models.Mixed_IFN_ppCompatible",
               ["Ib",np.multiply(train_doses,1E-12*6.022E23*1E-5)],
               ["Ia",np.multiply(train_doses,1E-12*6.022E23*1E-5)],
               [0,300,600,900,1200],
               ['TotalpSTAT','pSTAT1'],
               doseNorm=6.022E23*1E-5,
               custom_params=best_params_list,
               suppress=True) 
    
    sf = score_parameter(best_params_list,experimental_Tcell,sf_flag=True)[0]
    res = np.multiply(res,sf)
    arcsinh_response = []
    for row in res:
        new_row = []
        for item in row:
            new_row.append([item[0]/1E-12,item[1]/1E-12,np.arcsinh(item[2])])
        arcsinh_response.append(new_row)
    IFN_heatmap(arcsinh_response, cwd+"\\Mixed_IFN_Figures\\"+"Tcell Stepwise Fit - Ib (pM)", 'Ia (pM)')

    # Plot quality of fit as a heatmap
    percent_error = np.divide(np.subtract([[el[2] for el in row] for row in res],experimental_Tcell),experimental_Tcell)
    labelled_perecent_error = [[[res[i][j][0]/(1E-12), res[i][j][1]/(1E-12), abs(percent_error[i][j])] for j in range(8)] for i in range(8)]
    IFN_heatmap(labelled_perecent_error, cwd+"\\Mixed_IFN_Figures\\"+"Tcell Stepwise Percent error - Ib (pM)", "Ia (pM)")

    # Test best fit model on test data
    test_alpha_dose_response = [dose[4] for dose in test_Tcell_alpha]
    test_beta_dose_response = [dose[4] for dose in test_Tcell_beta]        
    
    min_doseA = np.log10(0.01) 
    max_doseA = np.log10(50000) 
    min_doseB = np.log10(0.01) 
    max_doseB = np.log10(50000) 
    model_doseA = np.multiply([0]+list(np.logspace(min_doseA,max_doseA,30)),6.022E23*1E-5*1E-12)
    model_doseB = np.multiply([0]+list(np.logspace(min_doseB,max_doseB,30)),6.022E23*1E-5*1E-12)
# =============================================================================
#     a_dr = np.multiply(sf,p_doseresponse("IFN_Models.Mixed_IFN_ppCompatible", ['Ia',model_doseA], 
#                                          np.multiply(test_times,60), [['TotalpSTAT','pSTAT1']],
#                                          axes_labels = ['',''], title = '', suppress=True, 
#                                          Norm=None, parameters=best_params_list+[['Ib',0]], 
#                                          dose_axis_norm=False, scan=0)[0])
#     b_dr = np.multiply(sf,p_doseresponse("IFN_Models.Mixed_IFN_ppCompatible", ['Ib',model_doseB], 
#                                          np.multiply(test_times,60), [['TotalpSTAT','pSTAT1']],
#                                          axes_labels = ['',''], title = '', suppress=True, 
#                                          Norm=None, parameters=best_params_list+[['Ia',0]], 
#                                          dose_axis_norm=False, scan=0)[0])
# =============================================================================

    # Plot the comparison
    simfig, [axes1,axes2,axes3] = plt.subplots(nrows=3,ncols=2,figsize=(8,8.5))
    simfig.suptitle("Compare fit to out-of-sample data")
    ntimes = 3
    alpha_palette = sns.color_palette("Reds",ntimes)
    beta_palette = sns.color_palette("Greens",ntimes)
    axes1[0].set(xscale='log',yscale='linear')
    axes1[0].set_xlabel('IFN (pM)')
    axes1[0].set_ylabel('pSTAT')
    axes1[0].set_title('T cells with IFNa')
    axes1[1].set(xscale='log',yscale='linear')
    axes1[1].set_xlabel('IFN (pM)')
    axes1[1].set_ylabel('pSTAT')
    axes1[1].set_title('T cells with IFNb')
    # Plot the training data
    sns.scatterplot(x=train_doses[1:],y=[r[0] for r in experimental_Tcell[1:]], ax=axes1[0], legend="full", label="Training Data", color=alpha_palette[0])
    sns.scatterplot(x=train_doses[1:],y=experimental_Tcell[0][1:], ax=axes1[1], legend="full", label="Training Data", color=beta_palette[0])
    # Plot the model
    model_res = IFN_2Dscan("IFN_Models.Mixed_IFN_ppCompatible",
               ["Ib",model_doseA],
               ["Ia",model_doseB],
               [0,300,600,900,1200],
               ['TotalpSTAT','pSTAT1'],
               doseNorm=6.022E23*1E-5,
               custom_params=best_params_list,
               suppress=True) 
    model_sim = [[sf*el[2] for el in row] for row in model_res]
    sns.lineplot(x=np.divide(model_doseA,6.022E23*1E-5*1E-12)[1:],y=[el[0] for el in model_sim[1:]], ax=axes1[0], legend="full",label="Model",color=alpha_palette[1])
    sns.lineplot(x=np.divide(model_doseB,6.022E23*1E-5*1E-12)[1:],y=model_sim[0][1:],ax=axes1[1], legend="full",label="Model",color=beta_palette[1]) 
    # Plot the testing data    
    data_sf=1
    sns.scatterplot(x=test_doses_alpha[1::], y=np.multiply(data_sf,test_alpha_dose_response[1::]), ax=axes1[0], legend="full", label="Testing Data", color=alpha_palette[2])
    sns.scatterplot(x=test_doses_beta[1::], y=np.multiply(data_sf,test_beta_dose_response[1::]), ax=axes1[1], legend="full", label="Testing Data", color=beta_palette[2])
    #plt.savefig(cwd+"\\Mixed_IFN_Figures\\"+"Tcell_out_of_sample.pdf")
    
    # Plot isolated IFN trajectories
    simulated_response = [[el[2] for el in row] for row in res]
    fig32, [[ax1,ax2],[ax3,ax4],[ax5,ax6]] = plt.subplots(nrows=3,ncols=2,figsize=(12,12))
    for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
        ax.set(xscale='log',yscale='linear')
    ax5.set_xlabel(r"IFN$\alpha$ (pM)")
    ax6.set_xlabel(r"IFN$\beta$ (pM)")
    ax1.set_ylabel('pSTAT')
    ax3.set_ylabel('pSTAT')
    ax5.set_ylabel('pSTAT')
    
    ax1.set_title('T cell')
    ax2.set_title('T cell')
    ax1.scatter(train_doses[1:],[r[0] for r in experimental_Tcell[1:]],c='r')
    ax2.scatter(train_doses[1:],experimental_Tcell[0][1:],c='g')
    ax1.plot(train_doses[1:],[el[0] for el in simulated_response[1:]],'r')    
    ax2.plot(train_doses[1:],simulated_response[0][1:],'g')  
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join([r'{}={:.2E}'.format(r[0],r[1]) for r in best_params_list])
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    
# =============================================================================
# B cells    
    #best_params_list = stepwise_fit([['R2',200,12000],['R1',200,12000],['kSOCSon',1E-7,1E-4],['kpa',1E-7,1E-4]],experimental_Bcell,12)
    best_params_list = [['R1', 200.0], ['R2', 2345.4545454545455], ['kSOCSon', 9.9999999999999995e-08], ['kpa', 9.9999999999999995e-08]]
    #best_params_list = [['kpa', 2.7345454545454546e-05], ['pS', 1000.0], ['R1', 200.0], ['kSOCSon', 1.8263636363636365e-05], ['R2', 2345.4545454545455]]
    # old values #[['R1', 200.0], ['R2', 2728.5714285714284], ['kpa', 8.0714285714285719e-07], ['kSOCSon', 9.9999999999999995e-08]]
    print("The best fit for B cells was")
    print(best_params_list)
    with open('stepwise_results.txt','a') as f:
        f.write("\nB cell fit:\n")
        f.write(str(best_params_list))
    
    res = IFN_2Dscan("IFN_Models.Mixed_IFN_ppCompatible",
               ["Ib",np.multiply([0,0.06,0.32,1.6,8,40,200,1000],1E-12*6.022E23*1E-5)],
               ["Ia",np.multiply([0,0.06,0.32,1.6,8,40,200,1000],1E-12*6.022E23*1E-5)],
               [0,300,600,900,1200],
               ['TotalpSTAT','pSTAT1'],
               doseNorm=6.022E23*1E-5,
               custom_params=best_params_list,
               suppress=True) 
    
    sf = score_parameter(best_params_list,experimental_Bcell,sf_flag=True)[0]
    res = np.multiply(res,sf)
    arcsinh_response = []
    for row in res:
        new_row = []
        for item in row:
            new_row.append([item[0]/1E-12,item[1]/1E-12,np.arcsinh(item[2])])
        arcsinh_response.append(new_row)
    IFN_heatmap(arcsinh_response, "Bcell Stepwise Fit - Ib (pM)", 'Ia (pM)')

    # Plot quality of fit
    percent_error = np.divide(np.subtract([[el[2] for el in row] for row in res],experimental_Bcell),experimental_Bcell)
    labelled_perecent_error = [[[res[i][j][0]/(1E-12), res[i][j][1]/(1E-12), abs(percent_error[i][j])] for j in range(8)] for i in range(8)]
    IFN_heatmap(labelled_perecent_error, "Bcell Stepwise Percent error - Ib (pM)", "Ia (pM)")

    # Test best fit model on test data
    test_alpha_dose_response = [dose[4] for dose in test_Bcell_alpha]
    test_beta_dose_response = [dose[4] for dose in test_Bcell_beta]        

    # Plot the comparison
    axes2[0].set(xscale='log',yscale='linear')
    axes2[0].set_xlabel('IFN (pM)')
    axes2[0].set_ylabel('pSTAT')
    axes2[0].set_title('B cells with IFNa')
    axes2[1].set(xscale='log',yscale='linear')
    axes2[1].set_xlabel('IFN (pM)')
    axes2[1].set_ylabel('pSTAT')
    axes2[1].set_title('B cells with IFNb')
    # Plot the training data
    sns.scatterplot(x=train_doses[1:],y=[r[0] for r in experimental_Bcell[1:]], ax=axes2[0], legend="full", label="Training Data", color=alpha_palette[0])
    sns.scatterplot(x=train_doses[1:],y=experimental_Bcell[0][1:], ax=axes2[1], legend="full", label="Training Data", color=beta_palette[0])
    # Plot the model
    model_res = IFN_2Dscan("IFN_Models.Mixed_IFN_ppCompatible",
               ["Ib",model_doseA],
               ["Ia",model_doseB],
               [0,300,600,900,1200],
               ['TotalpSTAT','pSTAT1'],
               doseNorm=6.022E23*1E-5,
               custom_params=best_params_list,
               suppress=True) 
    model_sim = [[sf*el[2] for el in row] for row in model_res]
    sns.lineplot(x=np.divide(model_doseA,6.022E23*1E-5*1E-12)[1:],y=[el[0] for el in model_sim[1:]], ax=axes2[0], legend="full",label="Model",color=alpha_palette[1])
    sns.lineplot(x=np.divide(model_doseB,6.022E23*1E-5*1E-12)[1:],y=model_sim[0][1:],ax=axes2[1], legend="full",label="Model",color=beta_palette[1]) 
    # Plot the testing data    
    data_sf=1
    sns.scatterplot(x=test_doses_alpha[1::], y=np.multiply(data_sf,test_alpha_dose_response[1::]), ax=axes2[0], legend="full", label="Testing Data", color=alpha_palette[2])
    sns.scatterplot(x=test_doses_beta[1::], y=np.multiply(data_sf,test_beta_dose_response[1::]), ax=axes2[1], legend="full", label="Testing Data", color=beta_palette[2])
    #plt.savefig(cwd+"\\Mixed_IFN_Figures\\"+"Bcell_out_of_sample.pdf")
    
    
    simulated_response = [[el[2] for el in row] for row in res]
    ax3.set_title('B cell')
    ax4.set_title('B cell')
    ax3.scatter([0.06,0.32,1.6,8,40,200,1000],[r[0] for r in experimental_Bcell[1:]],c='r')
    ax4.scatter([0.06,0.32,1.6,8,40,200,1000],experimental_Bcell[0][1:],c='g')
    ax3.plot([0.06,0.32,1.6,8,40,200,1000],[el[0] for el in simulated_response[1:]],'r')    
    ax4.plot([0.06,0.32,1.6,8,40,200,1000],simulated_response[0][1:],'g')  

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join([r'{}={:.2E}'.format(r[0],r[1]) for r in best_params_list])
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    
    simfig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    simfig.savefig(cwd+"\\Mixed_IFN_Figures\\"+"Out_of_sample.pdf")
    return 0    
# =============================================================================
# NK cells
    #best_params_list = stepwise_fit([['R2',200,12000],['R1',200,12000],['kSOCSon',1E-7,1E-4],['kpa',1E-7,1E-4]],experimental_NKcell,12)
    best_params_list = [['R1', 200.0], ['R2', 3418.181818181818], ['kSOCSon', 9.9999999999999995e-08], ['kpa', 9.9999999999999995e-08]]
    #best_params_list = [['kpa', 0.0001], ['R2', 12000.0], ['kSOCSon', 9.1818181818181835e-06], ['R1', 4490.909090909091], ['pS', 727.27272727272725]]
    # old values #best_params_list = [['R1', 200.0], ['R2', 3571.4285714285716], ['kpa', 8.0714285714285719e-07], ['kSOCSon', 9.9999999999999995e-08]]
    print("The best fit for NK cells was")
    print(best_params_list)
    with open('stepwise_results.txt','a') as f:
        f.write("\nNK cell fit:\n")
        f.write(str(best_params_list))
    
    res = IFN_2Dscan("IFN_Models.Mixed_IFN_ppCompatible",
               ["Ib",np.multiply([0,0.06,0.32,1.6,8,40,200,1000],1E-12*6.022E23*1E-5)],
               ["Ia",np.multiply([0,0.06,0.32,1.6,8,40,200,1000],1E-12*6.022E23*1E-5)],
               [0,300,600,900,1200],
               ['TotalpSTAT','pSTAT1'],
               doseNorm=6.022E23*1E-5,
               custom_params=best_params_list,
               suppress=True) 
    
    sf = score_parameter(best_params_list,experimental_NKcell,sf_flag=True)[0]
    res = np.multiply(res,sf)
    arcsinh_response = []
    for row in res:
        new_row = []
        for item in row:
            new_row.append([item[0]/1E-12,item[1]/1E-12,np.arcsinh(item[2])])
        arcsinh_response.append(new_row)
    IFN_heatmap(arcsinh_response, "NKcell Stepwise Fit - Ib (pM)", 'Ia (pM)')

    # Plot quality of fit
    percent_error = np.divide(np.subtract([[el[2] for el in row] for row in res],experimental_NKcell),experimental_NKcell)
    labelled_perecent_error = [[[res[i][j][0]/(1E-12), res[i][j][1]/(1E-12), abs(percent_error[i][j])] for j in range(8)] for i in range(8)]
    IFN_heatmap(labelled_perecent_error, "NKcell Stepwise Percent error - Ib (pM)", "Ia (pM)")

    # Test best fit model on test data
    test_alpha_dose_response = [dose[4] for dose in test_NKcell_alpha]
    test_beta_dose_response = [dose[4] for dose in test_NKcell_beta]        

    # Plot the comparison
    axes3[0].set(xscale='log',yscale='linear')
    axes3[0].set_xlabel('IFN (pM)')
    axes3[0].set_ylabel('pSTAT')
    axes3[0].set_title('NK cells with IFNa')
    axes3[1].set(xscale='log',yscale='linear')
    axes3[1].set_xlabel('IFN (pM)')
    axes3[1].set_ylabel('pSTAT')
    axes3[1].set_title('NK cells with IFNb')
    # Plot the training data
    sns.scatterplot(x=train_doses[1:],y=[r[0] for r in experimental_NKcell[1:]], ax=axes3[0], legend="full", label="Training Data", color=alpha_palette[0])
    sns.scatterplot(x=train_doses[1:],y=experimental_NKcell[0][1:], ax=axes3[1], legend="full", label="Training Data", color=beta_palette[0])
    # Plot the model
    model_res = IFN_2Dscan("IFN_Models.Mixed_IFN_ppCompatible",
               ["Ib",model_doseA],
               ["Ia",model_doseB],
               [0,300,600,900,1200],
               ['TotalpSTAT','pSTAT1'],
               doseNorm=6.022E23*1E-5,
               custom_params=best_params_list,
               suppress=True) 
    model_sim = [[sf*el[2] for el in row] for row in model_res]
    sns.lineplot(x=np.divide(model_doseA,6.022E23*1E-5*1E-12)[1:],y=[el[0] for el in model_sim[1:]], ax=axes3[0], legend="full",label="Model",color=alpha_palette[1])
    sns.lineplot(x=np.divide(model_doseB,6.022E23*1E-5*1E-12)[1:],y=model_sim[0][1:],ax=axes3[1], legend="full",label="Model",color=beta_palette[1]) 
    # Plot the testing data    
    data_sf=1
    sns.scatterplot(x=test_doses_alpha[1::], y=np.multiply(data_sf,test_alpha_dose_response[1::]), ax=axes3[0], legend="full", label="Testing Data", color=alpha_palette[2])
    sns.scatterplot(x=test_doses_beta[1::], y=np.multiply(data_sf,test_beta_dose_response[1::]), ax=axes3[1], legend="full", label="Testing Data", color=beta_palette[2])
    simfig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    simfig.savefig(cwd+"\\Mixed_IFN_Figures\\"+"Out_of_sample.pdf")
    
    

    simulated_response = [[el[2] for el in row] for row in res]
    ax5.set_title('NK cell')
    ax6.set_title('NK cell')
    ax5.scatter([0.06,0.32,1.6,8,40,200,1000],[r[0] for r in experimental_NKcell[1:]],c='r')
    ax6.scatter([0.06,0.32,1.6,8,40,200,1000],experimental_NKcell[0][1:],c='g')
    ax5.plot([0.06,0.32,1.6,8,40,200,1000],[el[0] for el in simulated_response[1:]],'r')    
    ax6.plot([0.06,0.32,1.6,8,40,200,1000],simulated_response[0][1:],'g')  
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join([r'{}={:.2E}'.format(r[0],r[1]) for r in best_params_list])
    ax5.text(0.05, 0.95, textstr, transform=ax5.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax6.text(0.05, 0.95, textstr, transform=ax6.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    
    fig32.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig32.savefig('Stepwise_dr.pdf')

    
if __name__ == '__main__':
    main()
    