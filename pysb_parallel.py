# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:19:35 2018

@author: Duncan
# =============================================================================
# NOTES:
# 1. param_values is now a list (rather than a dictionary, as in pysbplotlib)
# 2. Cannot pass IFN as a parameter. Must pass IFN*volEC*NA to the parameter 
#    Ia or Ib.
# 3. param_values must receive a vector with ALL parameter values to use
# =============================================================================

Suite of functions for performing parallelized routines such as 2D parameter
 scans: get_EC50(), p_DRparamScan()
"""
# Import statements required for the parallelized routines
from multiprocessing import Process, Queue, JoinableQueue, cpu_count
from pysb.export import export
from numpy import divide, subtract, abs, reshape, flipud, flip
import numpy as np
from operator import itemgetter # for sorting results after processes return
import itertools #for creating all combinations of lists
import re # for printing results to text files and getting values from model files

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
# =============================================================================
#  get_ODE_model() is used to get an ODE model that is isolated from PySB, 
#  allowing Python multiprocessing to be used
# Inputs:
#    modelfile = a str name for a PySB model file 

# Returns:
#    Model class object, as defined in pysb/export
# =============================================================================
def get_ODE_model(modelfile):
    imported_model = __import__(modelfile)
    py_output = export(imported_model.model, 'python')
    with open('ODE_system.py','w') as f:
        f.write(py_output)
    import ODE_system
    return ODE_system.Model()
	
# =============================================================================
# get_EC50() takes a dose-response trajectory for a single species and returns
# an estimate of the EC50.
# Inputs:
#   dose: array of the doses given
#   response: array of the responses for each dose
# Outputs: None
# Returns: the values of dose[ec50], half_max_response
# =============================================================================
def get_EC50(dose, response):
    if len(dose) != len(response):
        print("len(dose) = "+str(len(dose))+" , len(response) = "+str(len(response)))
        return 1
    half_max_response = max(response)/2
    # Find the simulation point closest to the half-max value. Be careful that
    # the selected point isn't in the right tail of the dose-response curve
    max_idx = response.index(max(response))+1
    dif = response[0:max_idx] # Avoids right tail of response curve   
    
	# This makes the half-max point near 0
    dif = abs(subtract(dif,half_max_response)).tolist()
    ec50_idx = dif.index(min(dif))
    # Return the value of dose which corresponds to half-max response and the 
	# half_max_response
    return dose[ec50_idx], response[ec50_idx]

# =============================================================================
# get_ODE_model_scan() is an internal function, only used when p_DRparamScan 
# is called. It is required because the model has to be exported before
# parallel threads are started, changing the import statements for p_timecourse
# =============================================================================
def get_ODE_model_scan():
    import ODE_system
    return ODE_system.Model()			

# =============================================================================
# p_timecourse() takes a PySB model and a set of time points and runs a 
# simulation using scipy.integrate. It can output a figure of the simulation.
# Inputs:
#     modelfile = previously imported model file
#     t = list of time points to simulate over (eg. t = linspace(20000))
#     spec = list of lists with each element being a 2-list in the form 
#             [name of an observable, label for observable]
#     axes_labels = 2-list of [x_label, y_label] (default is empty string)
#     title = string title label (default is empty string)
#      Norm = normalization factor. Default = None (ie. no normalization)
#          If Norm is a float or int then every species number will be divided by Norm
#          If Norm is a list then:
#             if len(Norm)=len(spec) then each species will have a unique normalization factor
#                if type(Norm[0]) is float or int then that factor will be a number
#                otherwise the function will assume the normalization is a 
#                trajectory over the same number of doses.      
#      suppress = a boolean determining whether or not to plot the results
#      parameters = a list of lists, with sublists of the form ['parameter name', value]
# Outputs:
#     figure of the time course
# Returns: timecourse = list indexed by observable names
# =============================================================================
def p_timecourse(modelfile, t, spec, axes_labels = ['',''], title = '',
               Norm=None, suppress=False, parameters=None, scan=0):
    # Get model
    if scan==0:
        mod = get_ODE_model(modelfile)
    elif scan==1:
        mod = get_ODE_model_scan()
    # Run simulation
    if parameters==None:
        #(species_output, observables_output)
        (_, timecourse) = mod.simulate(t)
    #Error catching
    elif type(parameters)==dict:
        print("Remember, pysb_parallel passes parameters as a list, not a dict")
        return 1
    else:
        # Build the complete list of parameters to use for the simulation, 
        # replacing the default parameters with the custom parameters
        all_parameters = []
        # list becomes immutable so build it right! That means only add parameter 
        # if it's not one of the custom parameters
        for p in mod.parameters:
            # assume the current parameter is not re-defined by the custom parameters
            isINlist=False
            # now check if it actually is in parameters
            for y in parameters:
                if y[0] == p[0]:
                    # in this case, add the custom parameter instead of the default
                    isINlist=True
                    all_parameters.append(y[1])
                    # no need to keep searching the custom parameters list
                    break
            if isINlist == False:
                # in this case, the parameter was not re-defined by the custom parameters list
                all_parameters.append(p.value)
        
        # now run the simulation
        #(species_output, observables_output)
        (_, timecourse) = mod.simulate(t, param_values=all_parameters)
        
    # Plot results
    if suppress==False:
        plt.ion()
        fig, ax = plt.subplots()
        plt.xlabel(axes_labels[0], fontsize=18)
        plt.ylabel(axes_labels[1], fontsize=18)
        fig.suptitle(title, fontsize=18)
        ax.tick_params(labelsize=14)
        if Norm==None:
            for species in spec:
                ax.plot(t, timecourse[species[0]], label=species[1], linewidth=2.0)
        elif type(Norm)==int or type(Norm)==float:
            for species in spec:
                ax.plot(t, timecourse[species[0]]/Norm, label=species[1], linewidth=2.0)
        elif type(Norm)==list:
            if type(Norm[0])==int or type(Norm[0])==float:#species-specific normalization factors
                spec_count=0
                for species in spec:
                    ax.plot(t, timecourse[species[0]]/Norm[spec_count], label=species[1], linewidth=2.0)
                    spec_count+=1
        else:#species-specific trajectory to normalize by (ie. norm-factor is a function of time)
            if len(t)!= len(Norm[0]):
                print("Expected a trajectory to normalize each species by, but len(t) != len(Norm[0])")
                return 1
            else:
                spec_count=0
                for species in spec:
                    ax.plot(t, divide(timecourse[species[0]],Norm[spec_count]),
                            label=species[1], linewidth=2.0)
                    spec_count+=1
        plt.legend()
        if title != '':
            plt.savefig(title+".pdf")
        else:
            plt.savefig("tc.pdf")
                
                
    return timecourse

# =============================================================================
# p_doseresponse() takes a model, a set of doses, and a set of time points 
# and runs a simulation using Scipy's odeint. It can output a figure of the 
# simulation.
# Inputs:
#      modelfile = (str) the name of a model to use
#      dose = list composed of ['parameter_title',[parameter values to scan]]
#      t = list of time points to simulate over (eg. t = linspace(20000))
#      spec = list of strings with each element being a 2-list in the form 
#              [name of an observable, label for observable]
#      axes_labels = 2-list of [x_label, y_label] (default is empty string)
#      title = string title label (default is empty string)
#      suppress = boolean to control whether or not to supress plotting (default is False)
#      Norm = normalization factor. Default = None (ie. no normalization)
#          If Norm is a float or int then every species number will be divided by Norm
#          If Norm is a list then:
#             if len(Norm)=len(spec) then each species will have a unique normalization factor
#                if type(Norm[0]) is float or int then that factor will be a number
#                otherwise the function will assume the normalization is a 
#                trajectory over the same number of doses.  
#      parameters = a list with elements of the form ['name',value], for custom parameters 
#					in simulation
#      dose_axis_norm = a factor to rescale the x-axis labels by, since I often
#                       want to plot concentrations but have to give doses in
#                       terms of molecules/cell    
#      Outputs:
#      figure of the dose response (unless suppress=True)
#  Returns: final_dr = final species concentrations for each dose's timecourse
# =============================================================================
def p_doseresponse(modelfile, dose, t, spec, axes_labels = ['',''], title = '',
                 suppress=False, Norm=None, parameters=False, dose_axis_norm=False,
                 scan=0):
    # Sanity check: dose parameter should not also be in parameters
    if parameters:
        for p in parameters:
            if p[0]==dose[0]:
                print("Dose parameter should not be included in parameters")
                return 1
    # Run a time course for each dose
    #For storing the final time point of each species' response at each dosasge
    #Each element in the list is the final-time dose-response curve for one species
    final_dr = [[] for i in range(len(spec))]
    dose_index=0 #Only needed for normalizing by trajectory
    for d in dose[1]:
        if parameters:
            # Add custom parameters before passing to simulation
           if dose_index == 0:
               # dose is not yet in parameters so add it
               parameters.append([dose[0],d])
           else:
               parameters[-1][1]=d
        else:
            # no custom parameters given
            parameters = [[dose[0],d]]
        temp = parameters.copy() 
        if scan==0:
            simres = p_timecourse(modelfile, t, spec, suppress=True, title=str(d), parameters = temp)
        elif scan==1:
            simres = p_timecourse(modelfile, t, spec, suppress=True, title=str(d), parameters = temp, scan=1)
        # Get the final time point for each dose
        if Norm==None:#No normalization
            for i in range(len(spec)):
                #species i .append(simres['species'][last time])
                final_dr[i].append(simres[spec[i][0]][-1])
        elif type(Norm)==int or type(Norm)==float:#Global normalization
            for i in range(len(spec)):
                final_dr[i].append(simres[spec[i][0]][-1]/Norm)
        elif type(Norm)==list:
            if type(Norm[0])==int or type(Norm[0])==float:#species-specific normalization factors
                for i in range(len(spec)):
                    final_dr[i].append(simres[spec[i][0]][-1]/Norm[i])
            else:#species-specific trajectory to normalize by (ie. norm-factor is a function of time)
                if len(dose[1])!= len(Norm[0]):
                    print("Expected a trajectory to normalize each species by, but len(dose) != len(Norm[0])")
                    return 1
                else:
                    for i in range(len(spec)):
                        final_dr[i].append(simres[spec[i][0]][-1]/Norm[i][dose_index])
        dose_index+=1
    # Plot the final dose-response
    if suppress == False:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set(xscale="log", yscale="linear")  
        plt.xlabel(axes_labels[0], fontsize=18)
        plt.ylabel(axes_labels[1], fontsize=18)
        fig.suptitle(title, fontsize=18)
        ax.tick_params(labelsize=14)
        for species in range(len(spec)):
            if dose_axis_norm:
                ax.plot(dose[1]/dose_axis_norm, final_dr[species], label=spec[species][1], linewidth=2.0)
            else:
                 ax.plot(dose[1], final_dr[species], label=spec[species][1], linewidth=2.0)
            plt.legend()
        if title != '':
            plt.savefig(title+".pdf")
        else:
            plt.savefig("dr.pdf")
    return final_dr

# =============================================================================
# p_Wtimecourse() takes a model and a set of time points and runs a simulation using 
# Scipy's odeint. It can output a figure of the simulation.
# Inputs:
#     modelfiles = a list of previously imported model files
#     weights = a list of weights; the order of weights corresponds to the 
#               order they are given in modelfiles.
#     params = a list of lists, each sublist containing the 
#               unique parameter values to use for each model file, in the same 
#               form as for p_timecourse()
#     t = list of time points to simulate over (eg. t = linspace(20000))
#     spec = list of lists with each element being a 2-list in the form 
#             [name of an observable, label for observable]
#     axes_labels = 2-list of [x_label, y_label] (default is empty string)
#     title = string title label (default is empty string)
#      Norm = normalization factor. Default = 1 (ie. no normalization)
#          If Norm is a float or int then every species number will be divided by Norm
#          If Norm is a list then:
#             if len(Norm)=len(spec) then each species will have a unique normalization factor
#                if type(Norm[0]) is float or int then that factor will be a number
#                otherwise the function will assume the normalization is a 
#                trajectory over the same number of doses.      
#      suppress = a boolean determining whether or not to plot the results
# Outputs:
#     figure of the time course
# Returns: timecourse = ScipyOdeSimulator().run().all
# =============================================================================
def p_Wtimecourse(modelfiles, weights, params, t, spec, axes_labels = ['',''], 
                title = '', Norm=None, suppress=False):
    sims = [] #Used to store the results of each modelfile's simulation
    # Run simulation
    for modelindex in range(len(modelfiles)):
        simres = p_timecourse(modelfiles[modelindex], t, spec,
                              suppress=True, parameters=params[modelindex])
        
        sims.append([simres[i[0]] for i in spec])
    # Create weighted sum of all simulations
    weightedSim={} #This will be the new dictionary of simulation results
    for species in range(len(spec)):
        for modelindex in range(len(modelfiles)):
            if spec[species][0] not in weightedSim:
                weightedSim[spec[species][0]]=weights[modelindex]*sims[modelindex][species]
            else:
                weightedSim[spec[species][0]]+=weights[modelindex]*sims[modelindex][species]
    # Plot results
    if suppress==False:
        plt.ion()
        fig, ax = plt.subplots()
        plt.xlabel(axes_labels[0], fontsize=18)
        plt.ylabel(axes_labels[1], fontsize=18)
        fig.suptitle(title, fontsize=18)
        ax.tick_params(labelsize=14)
        if Norm==None:
            for species in spec:
                ax.plot(t, weightedSim[species[0]], label=species[1], linewidth=2.0)
        elif type(Norm)==int or type(Norm)==float:
            for species in spec:
                ax.plot(t, weightedSim[species[0]]/Norm, label=species[1], linewidth=2.0)
        elif type(Norm)==list:
            if type(Norm[0])==int or type(Norm[0])==float:#species-specific normalization factors
                spec_count=0
                for species in spec:
                    ax.plot(t, weightedSim[species[0]]/Norm[spec_count], label=species[1], linewidth=2.0)
                    spec_count+=1
            else:#species-specific trajectory to normalize by (ie. norm-factor is a function of time)
                if len(t)!= len(Norm[0]):
                    print("Expected a trajectory to normalize each species by, but len(t) != len(Norm[0])")
                    return 1
                else:
                    spec_count=0
                    for species in spec:
                        ax.plot(t, divide(weightedSim[species[0]],Norm[spec_count]),
                            label=species[1], linewidth=2.0,)
                        spec_count+=1
        plt.legend()
        if title != '':
            plt.savefig(title+".pdf")
        else:
            plt.savefig("tc.pdf")
        
    return weightedSim
# =============================================================================
# p_Wdoseresponse() takes a combination of models, a set of doses, and a set of 
# time points, and runs a simulation using Scipy's odeint. It can output a
# figure of the simulation.
# Inputs:
#      modelfiles = list of model files to weight
#      weights = list of weights to use for each model file
#      parameters = a list of lists, each sublist a list of 2-item lists of
#                   the form [parameter name, parameter value]        
#      dose = list composed of ['parameter_title',[parameter values to scan]]
#      t = list of time points to simulate over (eg. t = linspace(20000))
#      spec = list of the form [name of an observable, label for observable]
#      axes_labels = 2-list of [x_label, y_label] (default is empty string)
#      title = string title label (default is empty string)
#      suppress = boolean to control whether or not to supress plotting (default is False)
#      wNorm = normalization factor. Default = 1 (ie. no normalization)
#          If Norm is a float or int then every species number will be divided by Norm
#          If Norm is a list then:
#             if len(Norm)=len(spec) then each species will have a unique normalization factor
#                if type(Norm[0]) is float or int then that factor will be a number
#                otherwise the function will assume the normalization is a 
#                trajectory over the same number of doses.   
#      dose_axis_norm = a factor to rescale the x-axis labels by, since I often
#                       want to plot concentrations but have to give doses in
#                       terms of molecules/cell        
#      Outputs:
#      figure of the dose response (unless suppress=True)
#  Returns: final_dr = list of the weighted response trajectory for each species
#    the order of the trajectories is the same as the order of the given species    
# =============================================================================
def p_Wdoseresponse(modelfiles, weights, parameters, dose, t, spec, 
                 axes_labels = ['',''], title = '', suppress=False, wNorm=None,
                 dose_axis_norm=False):
    # Run a time course for each dose
    #For storing the final time point of each species' response at each dosasge
    #Each element in the list is the final-time dose-response curve for one species
    final_dr = [[] for i in range(len(spec))]
    # Sanity check: dose parameter should not also be in parameters
    for p in parameters:
        for item in p:
            if item[0]==dose[0]:
                print("Dose parameter should not be included in parameters")
                return 1
    # Now initialize dose parameter 
    for p in parameters:
        p.append([dose[0],0.])
    # run simulations
    dose_index=0 #Only needed for normalizing by trajectory
    for d in dose[1]:
        # add dose to each list of parameters
        for c in parameters:
            c[-1][1]=d
        # run simulation
        wtraj = p_Wtimecourse(modelfiles, weights, parameters, t, spec, 
                            Norm=wNorm, suppress=True)
        # Get the final time point for each dose
        if wNorm==None:#No normalization
            for i in range(len(spec)):
                #species i .append(wtraj['species'][last time point])
                final_dr[i].append(wtraj[spec[i][0]][-1])
        elif type(wNorm)==int or type(wNorm)==float:#Global normalization
            for i in range(len(spec)):
                final_dr[i].append(wtraj[spec[i][0]][-1]/wNorm)
        elif type(wNorm)==list:
            if type(wNorm[0])==int or type(wNorm[0])==float:#species-specific normalization factors
                for i in range(len(spec)):
                    final_dr[i].append(wtraj[spec[i][0]][-1]/wNorm[i])
            else:#species-specific trajectory to normalize by (ie. norm-factor is a function of time)
                if len(dose[1])!= len(wNorm[0]):
                    print("Expected a trajectory to normalize each species by, but len(dose) != len(Norm[0])")
                    return 1
                else:
                    for i in range(len(spec)):
                        final_dr[i].append(wtraj[spec[i][0]][-1]/wNorm[i][dose_index])
        dose_index+=1

    # Plot the final dose-response
    if suppress == False:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set(xscale="log", yscale="linear")
        plt.xlabel(axes_labels[0], fontsize=18)
        plt.ylabel(axes_labels[1], fontsize=18)
        fig.suptitle(title, fontsize=18)
        ax.tick_params(labelsize=14)
        if dose_axis_norm != None:
            for species in range(len(spec)):
                ax.plot(dose[1]/dose_axis_norm, final_dr[species], label=spec[species][1], linewidth=2.0,)
                plt.legend()
        else:
            for species in range(len(spec)):
                ax.plot(dose[1], final_dr[species], label=spec[species][1], linewidth=2.0,)
                plt.legend()
        if title != '':
            plt.savefig(title+".pdf")
        else:
            plt.savefig("dr.pdf")

    return final_dr

# =============================================================================
# IFN_heatmap() is an internal function for p_DRparamScan(). It plots the scans.
# Inputs:
#     image: the EC50 image to plot, with pixels oriented such that top left is 
#             origin (ie. as the output from image_builder())
#     param1, param2: the scanned parameters to mark x and y axes by, respectively
# =============================================================================
def IFN_heatmap(image, param1, param2):
    fig, ax = plt.subplots()
    # Build title
    title = "{} vs {}".format(param1[0],param2[0])   
	 # Build x and y axis labels
    xticks =  ['{:.2e}'.format(float(i)) for i in param1[1]]
    xticks = [float(i) for i in xticks]
    yticks = ['{:.2e}'.format(float(i)) for i in param2[1]]
    yticks = [float(i) for i in yticks]
    yticks = flip(yticks,0)
    # Plot image
    sns.heatmap(flipud(image), xticklabels=xticks, yticklabels=yticks)
    plt.title(title)
    # Save figure
    plt.savefig("{}.pdf".format(title))

# =============================================================================
# p_DRparamScan_helper() is an internal function used by p_DRparamScan(). Each 
# thread calls this function. 
# Inputs:
#     id: the thread ID
#     jobs: the arguments for p_doseresponse()
#     result: the queue to put results on to
# =============================================================================
def p_DRparamScan_helper(id, jobs, result):
    # try to work as long as work is available
    while True:
        # get job from the queue
        task = jobs.get()
        if task is None:
            # there are no jobs
            break
        # there is a job, so do it
        modelfile, dose, t, spec, inNorm, params = task
        dr = p_doseresponse(modelfile, dose, t, spec, suppress=True, Norm=inNorm, 
                                parameters=params, scan=1)
        analysis = get_EC50(dose[1],dr[0])
        if analysis==1:
            print("Expected lengths of dose and response to match.")
            return [1,1,1]
        else:
            doseEC50, HMR = analysis
        # put the result onto the results queue
        result.put([params[0][1], params[1][1], doseEC50, HMR])

# =============================================================================
# image_builder() is an internal function used by p_DRparamScan() to ensure
# results are reordered correctly after all threads return
# Inputs: 
#     results: pool_results returned by parameter scan
#     doseNorm: see p_DRparamScan() documentation
#     shape: the x and y axis dimensions
# Returns: 
#     dose_image: dose EC50 ordered with top left item as origin 
#                 (ie. first x and y values of scan parameters)
#     response_image: same as dose_image but pixels are the response EC50
# =============================================================================
def image_builder(results, doseNorm, shape):
    dose_image = [[el[0], el[1], el[2]/doseNorm] for el in results]
    dose_image.sort(key=itemgetter(1,0))
    dose_image = [el[2] for el in dose_image]
    dose_image = reshape(dose_image,(shape[0],shape[1]))
    
    response_image = [[el[0], el[1], el[3]] for el in results]
    response_image.sort(key=itemgetter(1,0))
    response_image = [el[2] for el in response_image]
    response_image = reshape(response_image,(shape[0],shape[1]))
  
    return dose_image, response_image
# =============================================================================
# p_DRparamScan() is a 2D dose-response parameter scan for a given modelfile, 
# parallelized with Python multiprocessing.
#   NOTE: MULTIPROCESSING IS NOT COMPATIBLE WITH INTERACTIVE PYTHON INTERPRETERS    
# Inputs:
#      modelfile = the name of the model file to use for the parameter scan
#      weights = list of weights to use for each model file
#      param1 = a list of the form ['parameter_name',[values for the parameter]]
#	   param2 = a list of the form ['parameter_name',[values for the parameter]]
#		** param1 and param2 are the two parameters being scanned over
#      dose = list composed of ['parameter_title',[parameter values to scan]]
#      t_list = list of times to use for each timecourse in the dose-response. More time points
#				*may* make numerical integration more stable.
#      spec = a two-element list of the form [name of an observable, label for observable]
#	   custom_params = a list of other parameters to control in the same form as param1        
#      Norm = normalization factor. Default = None (ie. no normalization)
#          If Norm is a float or int then every species number will be divided by Norm
#          If Norm is a list then:
#             if len(Norm)=len(spec) then each species will have a unique normalization factor
#                if type(Norm[0]) is float or int then that factor will be a number
#                otherwise the function will assume the normalization is a 
#                trajectory over the same number of doses.   
#     cpu = number of cpus to use (relies on the user to give a reasonable number)
#           (default is cpu = multiprocessing.cpu_count())   
#     doseNorm = a value to normalize the dose by for plotting, 
#               ie. convert molecules to concentrations just for plotting 
# Outputs: None
# Returns: A 2D array with each element containing three values: 
#			[x-axis value, y-axis value, z_value]
# =============================================================================
def p_DRparamScan(modelfile, param1, param2, testDose, t_list, spec, custom_params=None,
                  Norm=None, cpu=None, suppress=False, doseNorm=1):
    # initialization
    jobs = Queue()
    result = JoinableQueue()
    if cpu == None or cpu >= cpu_count():
        NUMBER_OF_PROCESSES = cpu_count()-1
    else:
        NUMBER_OF_PROCESSES = cpu
    print("Using {} threads".format(NUMBER_OF_PROCESSES))
    # build task list
    params=[]
    print("Building tasks")
    if custom_params == None:
        for val1 in param1[1]:
            for val2 in param2[1]:
                params.append([[param1[0],val1],[param2[0],val2]])
    else:
        for val1 in param1[1]:
            for val2 in param2[1]:
                params.append([[param1[0],val1],[param2[0],val2]]+[c for c in custom_params])

    # Write modelfile
    imported_model = __import__(modelfile)
    py_output = export(imported_model.model, 'python')
    with open('ODE_system.py','w') as f:
        f.write(py_output)
				
    tasks = [[modelfile, testDose, t_list, spec, Norm, p] for p in params]
    # put jobs on the queue
    print("There are {} tasks to compute".format(len(params)))
    print("Putting tasks on the queue")
	
    for w in tasks:
        jobs.put(w)
		
    print("Computing scan")
	
    # start up the workers
    [Process(target=p_DRparamScan_helper, args=(i, jobs, result)).start()
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
    print("Done scan")
    # plot heatmap if suppress==False
    if suppress==False:
        	dose_image, response_image = image_builder(pool_results, doseNorm, (len(param1[1]),len(param2[1])))
        	IFN_heatmap(dose_image, ["dose image - {}".format(param1[0]), param1[1]], param2)
        	IFN_heatmap(response_image, ["response image - {}".format(param1[0]), param1[1]], param2)
    #return the scan 
    return pool_results

# =============================================================================
# lhc() builds an origin-centered latin hypercube for parameters to investigate
# Inputs:
#   parameters = these are the model parameters to fit
#           a list of lists, each sublist of the form ['name', lower limit, upper limit]
#   n = number of points to generate 
#   exp_params =  These are model parameters which are not being fitted but are needed
#                 to replicate experimental data (ie. one item in this list per data point)    
#               a list of lists, each sublist describing the experimental conditions 
#               which were used to generate the corresponding ydata point.
#               Each *sublist* is of the form [['name',value],['name',value],...]       
# =============================================================================
def lhc(parameters, n, exp_params):
    d = len(parameters)
    # spread function
    def spread(points):
        return sum(1./np.linalg.norm(np.subtract(points[i], points[j])) for i in range(n) for j in range(n) if i > j)

    # start with diagonal shape
    unitCube = [[i/(n-1.)]*d for i in range(n)]

    # minimize spread function by shuffling
    minspread = spread(unitCube)

    for i in range(1000):
        point1 = np.random.randint(n)
        point2 = np.random.randint(n)
        dim = np.random.randint(d)

        newlh = np.copy(unitCube)
        newlh[point1, dim], newlh[point2, dim] = newlh[point2, dim], newlh[point1, dim]
        newspread = spread(newlh)

        if newspread < minspread:
            unitCube = np.copy(newlh)
            minspread = newspread  
    # shift to be origin-centered
    for point in unitCube:
        for ind in range(d):
            point[ind]-=0.5
    # rescale in parameter ranges, add labels, and add experimental conditions
    reformatted = []
    for point in unitCube:
        point_copy = [[] for i in point]
        for dim in range(d):
            lower_bound = parameters[dim][2]
            upper_bound = parameters[dim][3]
            scale = parameters[dim][4]
            if scale=='linear':
                point_copy[dim] = [parameters[dim][0],(upper_bound-lower_bound)*point[dim]+0.5*(upper_bound+lower_bound)]
            elif scale=='log':
                logY = (np.log10(upper_bound)-np.log10(lower_bound))*point[dim]+0.5*(np.log10(upper_bound)+np.log10(lower_bound))
                point_copy[dim] = [parameters[dim][0],10**logY]
        if len(exp_params)!=0:
            for exp in exp_params:
                reformatted.append(point_copy+exp)
        else:
            reformatted.append(point_copy)
    return reformatted

# =============================================================================
# brute_parameters() builds all combinations of all parameter values to test
# Inputs:
#   parameters = list of the form 
#                [['name', [values to test]], ['name', [values to test]], ...]
#   exp_params =  a list of lists, each sublist describing the experimental conditions 
#               which were used to generate the corresponding ydata point.
#               Each *sublist* is of the form [['name',value],['name',value],...]
# Returns:
#   list of all parameter combinations to run        
# =============================================================================
def brute_parameters(parameters, exp_params):
    reformatted=[]
    # reformat list 
    for l in parameters:
        reformatted.append([[l[0],val] for val in l[1]])
    # generate all unique combinations
    testCombinations = list(itertools.product(*reformatted))
    full_fit = []
    for test in testCombinations:
        for experiment in exp_params:
            full_fit.append(list(test)+experiment)
    return full_fit

# =============================================================================
# read_parameters gets the specified model parameters and their values
# Input:
#     pList = a list of strings, each string being the name of a model parameter
#     filename = the name of the modelfile to read
# Returns:
#   vals = list of the values corresponding to the parameters; empty list if none found
# Note: this function will not fail if some parameters are not found, but will notify the user    
# =============================================================================
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
def read_parameters(pList,filename):
    count=0
    lcount=0
    vals=np.zeros(len(pList))
    with open(filename,'r') as f:
        while lcount<file_len(filename):
            if count >= len(pList):
                break
            else:
                line = f.readline().lstrip()
                lcount+=1
                #if not line: 
                #    print("EOF")
                #    break #escape if EOF
                try:
                    name = re.search('Parameter\(\'(.+?)\', ', line).group(1) # gets the parameter name
                    value = float(re.search('\',(.+?)\)', line).group(1)) # gets the parameter name
                except AttributeError:
                    name='This is not a parameter'
                if name in pList:
                    ind = pList.index(name)
                    vals[ind] = value
                    count+=1
    if count!=len(pList):
        print("Not all parameters were found")
        print(pList)
        print(vals)
        print("\n")
    return vals

# =============================================================================
# fit_helper is an internal function to allow fit_model to be parallelized
# =============================================================================
def fit_helper(id, jobs, result):
    # try to work as long as work is available
    while True:
        t_0=False #assume this is not a t=0 time point
        # get job from the queue
        task = jobs.get()
        if task is None:
            # there are no jobs
            break
        # there is a job, so do it
        # run simulation
        if len(task)==6:
            conditions, ydata, paramsList, sigma, method, p0 = task
        else:
            conditions, ydata, paramsList, method, p0 = task
            sigma=None
        params_without_time=[]
        gamma=1
        # get the time for the simulation from conditions
        for i in range(len(conditions)):
            if (conditions[len(conditions)-1-i][0]=='t') or (conditions[len(conditions)-1-i][0]=='time'):
                if conditions[len(conditions)-1-i][1] == 0:
                    t_0 = True # model is initialised with 0 pSTAT but simulations hit numerical instability if I try to simulate t=0
                else:
                    time = np.linspace(0,conditions[len(conditions)-1-i][1])
            # Some special parameters for IFN signaling                    
            elif conditions[len(conditions)-1-i][0]=='R': # 'R' is used to fit receptor number given R1 = R2 
                params_without_time.append(['R1',conditions[len(conditions)-1-i][1]])
                params_without_time.append(['R2',conditions[len(conditions)-1-i][1]])
            elif conditions[len(conditions)-1-i][0]=='gamma': # 'gamma' is used to fit the scale factor for the flow cytometry data
                gamma=conditions[len(conditions)-1-i][1]
            elif conditions[len(conditions)-1-i][0]=='kd4': #'kd4' and 'k_d4' can be fit, but detailed balance has to be enforced
                pvals = read_parameters(['ka1','kd1','ka2','kd2','ka4','ka3'],'ODE_system.py')
                q1=pvals[0]/pvals[1]
                q2=pvals[2]/pvals[3]
                q4=pvals[4]/conditions[len(conditions)-1-i][1]
                q3=q2*q4/q1
                kd3=pvals[5]/q3
                params_without_time.append(['kd3',kd3])
                params_without_time.append(conditions[len(conditions)-1-i])                
            elif conditions[len(conditions)-1-i][0]=='k_d4':
                pvals = read_parameters(['k_a1','k_d1','k_a2','k_d2','k_a4','k_a3'],'ODE_system.py')
                q1=pvals[0]/pvals[1]
                q2=pvals[2]/pvals[3]
                q4=pvals[4]/conditions[len(conditions)-1-i][1]
                q3=q2*q4/q1
                k_d3=pvals[5]/q3
                params_without_time.append(['k_d3',k_d3])
                params_without_time.append(conditions[len(conditions)-1-i])                
            else:
                params_without_time.append(conditions[len(conditions)-1-i])
        if t_0==True:
            result.put([0,conditions])
        else:
            simres = p_timecourse('', time, [ydata[0],ydata[0]], suppress=True,    # modelfile = '' because I'm using scan=1
                                      parameters=params_without_time, scan=1)[ydata[0]]
            
            # calculate residual
            if (sigma == None):
                res = (simres[-1]-ydata[1]/gamma)**2            
            else:
                # The distribution of res over models is Xi2 distributed
                res = ((simres[-1]-ydata[1]/gamma)/(sigma/gamma))**2
            # put the result onto the results queue
            result.put([res, conditions])     

# =============================================================================
# fit_model() takes a PySB model and fits an input list of parameters to 
# experimental (or otherwise) data by nonlinear least squares regression.
# Input: 
#       modelfile = the name of the model file to use for the fit
#       conditions = a list of lists, each sublist describing the experimental conditions 
#               which were used to generate the corresponding ydata point.
#               Each *sublist* is of the form [['name',value],['name',value],...]
#
#               NOTE: All data points are time course measurements. If the data
#                       point is a dose-response point, it's just a time course
#                       measurement for a specific dose and end time. It would
#                       be computationally more efficient to only run unique
#                       time courses but for now I haven't figured out how to 
#                       do that.        
#        
#       ydata = a list of lists with y-axis values of experimental data to fit to.
#               Each element of ydata should be a 2-list of the form
#                   ['corresponding model parameter name', measured value]        
#       paramsList = a list of the parameters in the model to fit
#           Note: special arguments include:
#                   'R': using this parameter will set R1 and R2 to the value of R
#                   'gamma': using this parameter will fit the scale factor for the experimental data   
#                   'k_d4' and 'kd4': using these parameterse will fit K3 and K4, maintaining detailed balance            
#       OPTIONAL ARGUMENTS:
#       n = integer
#   for method = "sampling", number of parameter combinations to try (default is n=500)
#   for method = "brute" or "bayesian", number of points to test for each parameter 
#                   (default is 5, mostly only useful for debugging) 
#       sigma = a list of uncertainties for each ydata point; equivalent to 
#               a list of ones when sigma not specified    
#       p0 = a list of lists, each sublist of the form 
#                   [initial_guess, lower_bound, upper_bound, 'linear' or 'log']
#               initial_guess = the initial guesses for parameter value
#               lower_bound, upper_bound = the minimum and maximum values to test for the parameter
#               'linear' or 'log' = strings to indicate if the values
#                       to test between bounds should be distributed linearly or logarithmically        
#        NOTE: Each value corresponds to a parameter in paramsList (ie. order MUST match)   
#                   for every experimental data point provided
#              Also, p0 must be provided if method="bayesian"
#       cpu = number of cpu's to use. If none specified, function will use n-1 
#               cores on an n-core machine
#       method = method to search parameter space; default is via brute force
#               options: 
#                   "brute" : try all parameter combinations
#                   "bayesian": bayesian inference fitting        
#                   "lhc_sampling" : sample parameter space using latin hypercube
#                                    to generate n samples for testing. 
#                                    Note: n total samples, not per parameter.
#                                    Score models based on their sum of squared residuals.       
#       rho = int or float used to weight the cost of deviating from bayesian prior
#               - a larger value for rho will make it easier to deviate from p0 prior guesses        
# Output:
#       parameters = list of optimal values for the parameters specified by paramslist
# =============================================================================
# =============================================================================
# # For debugging
# def dummy(i, jobs, result):
#     while True:
#         # get job from the queue
#         task = jobs.get()
#         if task is None:
#             # there are no jobs
#             break
#         result.put(1)     
# =============================================================================
def fit_model(modelfile, conditions, ydata, paramsList, n=5, sigma=None,
              p0=None, cpu=None, method="brute", rho=1):
# Basic sanity checks
    if len(conditions) != len(ydata[1]):
        print("Number of experimental conditions and observations do not match")
        return 1
    if type(p0)==list:
        if len(paramsList) != len(p0):
            print("Number of guesses must match number of parameters to be fit")
            return 1
    if (type(sigma)==list) and (len(sigma) != len(ydata[1])):
        print("Number of uncertainties provided does not match number of experimental observations.")
        return 1
# Write modelfile
    print("Importing model")
    imported_model = __import__(modelfile)
    py_output = export(imported_model.model, 'python')
    with open('ODE_system.py','w') as f:
        f.write(py_output)
# Set up parallelization
    jobs = Queue()
    result = JoinableQueue()
    if cpu == None or cpu >= cpu_count():
        NUMBER_OF_PROCESSES = cpu_count()-1
    else:
        NUMBER_OF_PROCESSES = cpu
    print("Using {} threads".format(NUMBER_OF_PROCESSES))
# Build list of parameter values to test
    print("Building tasks")
    if p0==None:
        vals = read_parameters(paramsList,'ODE_system.py')
        if len(vals)!=len(paramsList):
            print("Did not find all parameters in modelfile. Please supply p0 or check the parameters to fit.")
            return 1
        else:
            p0 = [[vals[i],vals[i]*0.1,vals[i]*10,'log'] for i in range(len(paramsList))]
            parameters = [[paramsList[i],vals[i],vals[i]*0.1,vals[i]*10,'log'] for i in range(len(paramsList))]
    else:
        parameters = [[paramsList[i], p0[i]] for i in range(len(paramsList))]
    if method == "brute" or method =="bayesian":
        print("Using {} points per parameter".format(n))
        print("Generating {} models to test".format(n**len(paramsList)))
        for p in parameters:
            if p[1][3]=='log':
                p[1] = np.logspace(np.log10(p[1][1]),np.log10(p[1][2]), num=n)
            elif p[1][3]=='linear':
                p[1] = np.linspace(p[1][1],p[1][2], num=n)
            else:
                print("Did not recognize specified distribution of parameter values to test")
                return 1
        tasks = brute_parameters(parameters, conditions)
    elif method == "lhc_sampling":
        print("Using latin hyper cube sampling with {} points".format(n))
        # calculate the number of values to test per parameter
        #n_p = int(n/np.math.factorial(len(parameters)))
        tasks = lhc(parameters, n, conditions)
    else:
        print("Did not recognize the method specified")
        return(1)
    # add the other arguments required for fitting to each parameter combo
    taskList = []
    for t in range(len(tasks)):
        datapoint = t % len(ydata[1])
        if (type(sigma)!=list) and (type(sigma)!=np.ndarray):#This is to get around checking truth value of an array
            if sigma == None:
                taskList.append([tasks[t], [ydata[0],ydata[1][datapoint]], paramsList, method, list(zip(paramsList,[p0[i][0] for i in range(len(p0))]))])
            else:
                print("Unsure what type of uncertainty provided in argument sigma")
        else:
            taskList.append([tasks[t], [ydata[0],ydata[1][datapoint]], paramsList, sigma[datapoint], method, list(zip(paramsList,[p0[i][0] for i in range(len(p0))]))])

# Run all combos of parameter values, calculating fit for each
    # put jobs on the queue
    print("There are {} tasks to compute".format(len(tasks)))
    print("Putting tasks on the queue")
    #for l in taskList:
    #    print(l)
    for w in taskList:
        jobs.put(w)
		
    print("Computing scan")

    # start up the workers          
    [Process(target=fit_helper, args=(i, jobs, result)).start()
            for i in range(NUMBER_OF_PROCESSES)]
    
    # pull in the results from each worker
    pool_results=[]
    for t in range(len(taskList)):
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
    print("Done scan")
# order the outputs by fit
    print("Scoring models")
    scoreboard = {}
    # create a list of all models, indexing each one
    for score, key in pool_results:
         # format key to only include parameters, not experimental conditions
         newKey=[]
         for item in key:
             if item[0] in paramsList:
                 newKey.append(item)
         newKey = str(newKey)
         scoreboard.setdefault(newKey, 0) # adds key to dictionary with value 0, unless key already exists (in which case it returns value) 
         scoreboard[newKey]+=score
    if method == "bayesian":
        # add model cost to model's score
        for key in scoreboard:
            xi2=0
            keyCopy = key
            key = key[3:-2]
            key = re.split("', |\], \['", key)
            key = [[key[i],float(key[i+1])] for i in range(0,len(key),2)]
            for parameter_k in key:
                if parameter_k[0] in paramsList:
                    index = paramsList.index(parameter_k[0])
                    xi2 += ((np.log(parameter_k[1])-np.log(p0[index][0]))**2)
                else:
                    key.remove(item)
            scoreboard[keyCopy] += xi2/(rho**2)
    # order models from smallest to largest total score
    leaderboard = [(k, scoreboard[k]) for k in sorted(scoreboard, key=scoreboard.get)]
    # write results to a file and print the best scoring model to output
    #   first clear any existing text from modelfit.txt
    f = open('modelfit.txt', 'w')
    f.close()
    with open('modelfit.txt', 'a') as outfile:
        outfile.write("# models: "+str(len(leaderboard))+"\n") 
        outfile.write("---------------------------------------------------------\n")
        header = ""
        for p in paramsList:
            header+=p+"          "
        header += "score\n"
        outfile.write(header)
        for key, score in leaderboard:
            # Example string below shows what happens before and after each line of code
            #[['kpa', 0.001], ['kSOCSon', 3.1622776601683792e-08]]
            key = key[3:-2]
            #kpa', 0.001], ['kSOCSon', 3.1622776601683792e-08
            key = re.split("', |\], \['", key)
            #["kpa" , "0.001" , "kSOCSon" , "3.1622776601683792e-08"]
            mod_p = ""
            for i in range(1,len(key),2):
                mod_p += '{:.3e}'.format(float(key[i]))+"    "
            outfile.write(mod_p+str(score)+"\n")
    print(leaderboard[0][0]+": "+str(leaderboard[0][1]))



# =============================================================================
# fit_IFN_helper() is an internal function used by fit_IFN_model() to allow 
# for multi-threaded computation of model fits. It receives the thread ID (id),
# a queue to get tasks from (jobs) and a queue to put results on (result).
# =============================================================================
def fit_IFN_helper(id, jobs, result):
    while True:
        model = jobs.get()
        if model is None:
            break
        # get key
        key = str(model)
        # parse arguments
        gamma=1
        pList = list(enumerate([el[0] for el in model]))        
        for p in pList:
            if p[1]=='gamma':
                gamma=model[p[0]][1]
                model=model[0:p[0]]+model[p[0]+1:len(model)]
                pList = list(enumerate([el[0] for el in model]))        
                break
        for p in pList:
            if p[1]=='k4':
                q1 = 3.321155762205247e-14/1
                q2 = 4.98173364330787e-13/0.015
                q4 = 3.623188E-4/(model[p[0]][1]*0.3)
                q3 = q2*q4/q1
                kd3 = 3.623188E-4/q3
                q_1 = 4.98E-14/0.03
                q_2 = 8.30e-13/0.002
                q_4 = 3.62e-4/(model[p[0]][1]*0.006)
                q_3 = q_2*q_4/q_1
                k_d3 = 2.4e-5/q_3
                alpha_parameters=model[0:p[0]]+[['kd4',model[p[0]][1]*0.3],['kd3',kd3]]+model[p[0]+1:len(model)]
                beta_parameters=model[0:p[0]]+[['k_d4',model[p[0]][1]*0.006],['k_d3',k_d3]]+model[p[0]+1:len(model)]
        # run simulation
        #   load models
        import ODE_system_alpha
        alpha_mod = ODE_system_alpha.Model()
        import ODE_system_beta
        beta_mod = ODE_system_beta.Model()
        
        # Build the complete list of parameters to use for the simulation, 
        # replacing the default parameters with the custom parameters
        all_parameters_alpha = []
        for p in alpha_mod.parameters:
            isINlist=False
            for y in alpha_parameters:
                if y[0] == p[0]:
                    isINlist=True
                    all_parameters_alpha.append(y[1])
                    break
            if isINlist == False:
                all_parameters_alpha.append(p.value)
        all_parameters_beta = []
        for p in beta_mod.parameters:
            isINlist=False
            for y in beta_parameters:
                if y[0] == p[0]:
                    isINlist=True
                    all_parameters_beta.append(y[1])
                    break
            if isINlist == False:
                all_parameters_beta.append(p.value)
        I_index_Alpha = [el[0] for el in alpha_mod.parameters].index('I')
        I_index_Beta = [el[0] for el in beta_mod.parameters].index('I')
        mod_score = 0
        #   get data and simulate it
        NA = 6.022E23
        volEC = 1E-5            
        import Experimental_Data as ED
        data = ED.data
        for r in [0,1,4,5,8,9]:
            IFN = data.iloc[r,0]*NA*volEC*1E-12 # Convert from pM to num_molecules
            all_parameters_beta
            typeIFN = data.iloc[r,1]
            experiment = np.divide(list(data.iloc[r,2:7]),gamma)
            sigma = data.iloc[r+2,2:7]
            if typeIFN == 'Alpha':
                all_parameters_alpha[I_index_Alpha]=IFN
                (_, sim) = alpha_mod.simulate([0,5*60,15*60,30*60,60*60], param_values=all_parameters_alpha)
                sim = sim['TotalpSTAT']
            elif typeIFN == 'Beta':
                (_, sim) = beta_mod.simulate([0,5*60,15*60,30*60,60*60], param_values=all_parameters_beta)
                sim = sim['TotalpSTAT']
            # Add to score
            mod_score += np.sum(np.square(np.divide(np.subtract(sim,experiment),sigma)))
        result.put([key,mod_score])
# =============================================================================
# fit_IFN_model() is specifically designed to fit IFN alpha and IFN beta models to the same parameters
# for a given set of experimental data. The structure is very similar to fit_model() described above,
# as it is multi-threaded and uses R^2 optimization to enumerate models.
# Inputs:
#     models = a list of strings, each item the name of a model file to test
#             Note: models expects 2 arguments: the first is the IFN alpha model to fit and the
#                   second is the IFN beta model to fit
#     parameters = list in the form [['name',guess, lower limit, upper limit]]
#     n = number of points to test, generated from a random latin hypercube in parameter space
#     cpu = (default is num_cores -1) specify the number of cores to use
# =============================================================================
def fit_IFN_model(models, parameters, n, cpu=None):
# Write modelfiles
    print("Importing models")
    alpha_model = __import__(models[0])
    py_output = export(alpha_model.model, 'python')
    with open('ODE_system_alpha.py','w') as f:
        f.write(py_output)
    beta_model = __import__(models[1])
    py_output = export(beta_model.model, 'python')
    with open('ODE_system_beta.py','w') as f:
        f.write(py_output)
# Generate parameters
    K4=False
    print("Building model list")
    if 'k4' in [el[0] for el in parameters]:
        K4=True
        parameters.remove(['k4'])
        models = lhc(parameters, n, [[['k4',i]] for i in np.logspace(np.log10(0.01),np.log10(100),num=30)])
    else:
        models = lhc(parameters, n, [])
    if cpu == None or cpu >= cpu_count():
        NUMBER_OF_PROCESSES = cpu_count()-1
    else:
        NUMBER_OF_PROCESSES = cpu
    print("Using {} threads".format(NUMBER_OF_PROCESSES))
    print("Computing scan")
    jobs = Queue()
    result = JoinableQueue()
    for m in models:
        jobs.put(m)    
    # start up the workers          
    [Process(target=fit_IFN_helper, args=(i, jobs, result)).start()
            for i in range(NUMBER_OF_PROCESSES)]
    # pull in the results from each worker
    pool_results=[]
    for m in range(len(models)):
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
    print("Done scan")
    # order models from smallest to largest total score
    leaderboard = sorted(pool_results, key = lambda x: x[1])
    # Write results
    f = open('modelfit_alpha_and_beta.txt', 'w')
    f.close()
    with open('modelfit_alpha_and_beta.txt', 'a') as outfile:
        outfile.write("# models: "+str(len(leaderboard))+"\n") 
        outfile.write("---------------------------------------------------------\n")
        header = ""
        for p in parameters:
            header+=p[0]+"          "
        if K4==True:
            header+="K4          "
        header += "score\n"
        outfile.write(header)
        for key, score in leaderboard:
            # Example string below shows what happens before and after each line of code
            #[['kpa', 0.001], ['kSOCSon', 3.1622776601683792e-08]]
            key = key[3:-2]
            #kpa', 0.001], ['kSOCSon', 3.1622776601683792e-08
            key = re.split("', |\], \['", key)
            #["kpa" , "0.001" , "kSOCSon" , "3.1622776601683792e-08"]
            mod_p = ""
            for i in range(1,len(key),2):
                mod_p += '{:.3e}'.format(float(key[i]))+"    "
            outfile.write(mod_p+str(score)+"\n")
    print(leaderboard[0][0]+": "+str(leaderboard[0][1]))
    
