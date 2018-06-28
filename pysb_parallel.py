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
from numpy import divide, subtract, abs, logspace, reshape, flipud, flip
from operator import itemgetter # for sorting results after processes return

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
#      parameters = a dictionary with keys='parameter name' and values to use
# Outputs:
#     figure of the time course
# Returns: timecourse = list indexed by observable names
# =============================================================================
def p_timecourse(modelfile, t, spec, axes_labels = ['',''], title = '',
               Norm=None, suppress=False, parameters=None):
    # Get model
    mod = get_ODE_model(modelfile)
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
                 suppress=False, Norm=None, parameters=False, dose_axis_norm=False):
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
        simres = p_timecourse(modelfile, t, spec, suppress=True, title=str(d), parameters = temp)
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
        dr = p_doseresponse(modelfile, dose, t, spec, suppress=True, Norm=inNorm, parameters = params)
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
    print("using {} processors".format(NUMBER_OF_PROCESSES))
    # build task list
    params=[]
    print("building tasks")
    if custom_params == None:
        for val1 in param1[1]:
            for val2 in param2[1]:
                params.append([[param1[0],val1],[param2[0],val2]])
    else:
        for val1 in param1[1]:
            for val2 in param2[1]:
                params.append([[param1[0],val1],[param2[0],val2]]+[c for c in custom_params])

    tasks = [[modelfile, testDose, t_list, spec, Norm, p] for p in params]
    # put jobs on the queue
    print("There are {} tasks to compute".format(len(params)))
    print("putting tasks on the queue")
	
    for w in tasks:
        jobs.put(w)
		
    print("computing scan")
	
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
    print("done scan")
    # plot heatmap if suppress==False
    if suppress==False:
        	dose_image, response_image = image_builder(pool_results, doseNorm, (len(param1[1]),len(param2[1])))
        	IFN_heatmap(response_image, param1, param2)
    #return the scan 
    return pool_results
	
