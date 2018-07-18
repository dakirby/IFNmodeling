# -*- coding: utf-8 -*-
"""
Created on Wed May 16 18:01:29 2018

@author: Duncan

Basic functionality for producing figures
"""
# Imports required for the basic suite and the weighted populations suite
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
import numpy as np

from pysb.simulator import ScipyOdeSimulator
# =============================================================================
# Basic suite of functions: timecourse(), compare_timecourse(), 
# doseresponse(), and compare_doseresponse()
# =============================================================================

# =============================================================================
# timecourse() takes a model and a set of time points and runs a simulation using 
# ScipyOdeSimulator. It can output a figure of the simulation.
# Inputs:
#     modelfile = previously imported model file
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
#      parameters = a dictionary with keys='parameter name' and values to use
# Outputs:
#     figure of the time course
# Returns: timecourse = ScipyOdeSimulator().run().all
# =============================================================================
def timecourse(modelfile, t, spec, axes_labels = ['',''], title = '',
               Norm=1, suppress=False, parameters=1):
    # Run simulation
    if parameters==1:
        simres = ScipyOdeSimulator(modelfile.model, tspan=t,compiler='python').run()
        timecourse = simres.all
    elif type(parameters)==dict:
        simres = ScipyOdeSimulator(modelfile.model, tspan=t, 
                                   param_values=parameters, compiler='python').run()
        timecourse = simres.all
    else:
        print("Expected parameters to be a dictionary.")
        return 1
    # Plot results
    if suppress==False:
        plt.ion()
        fig, ax = plt.subplots()
        plt.xlabel(axes_labels[0], fontsize=18)
        plt.ylabel(axes_labels[1], fontsize=18)
        fig.suptitle(title, fontsize=18)
        ax.tick_params(labelsize=14)
        if Norm==1:
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
                    ax.plot(t, np.divide(timecourse[species[0]],Norm[spec_count]),
                            label=species[1], linewidth=2.0)
                    spec_count+=1
        plt.legend()
    return timecourse

# =============================================================================
# compare_doses_timecourse() takes a model and a set of doses and runs 
# timecourse() for each dose. It outputs a figure comparing these timecourses.
# Inputs:
#      modelfiles = a previously imported model file, or list of model files
#      params_list = list of dictionaries to pass as parameters to timecourse()
#                   This list will be applied to each model in modelfiles
#      t = list of time points to simulate over (eg. t = linspace(20000))
#      spec = [name of an observable, label for observable] 
#          (only supports comparison of one observable)
#      style = list of colours/markers to use for plotting; order corresponds to 
#           [(modelfile1, params_list_1), (model_file1, params_list2),...
#           (model_file2, params_list1),...]
#      global_Norm = normalization factor. Default = 1 (ie. no normalization)
#          If Norm is a float or int then every timecourse will be divided by 
#           Norm. **Note that trajectory norms are not currently supported.
#      axes_labels = 2-list of [x_label, y_label] (default is empty string)
#      title = string title label (default is empty string)    
#      custom_legend = list of labels, order the same as style order
#                       (default is no legend)    
#  Outputs:
#      figure comparing the timecourse curves
#  Returns: Nothing
# =============================================================================
def compare_timecourse(modelfiles, params_list, t, spec, style, global_Norm=1, 
                             axes_labels = ['',''], title = '', custom_legend=False):
    trajectories = [[] for i in range(len(spec))] # to store each simulation output
    # set up plot
    plt.ion()
    fig, ax = plt.subplots()
    plt.xlabel(axes_labels[0], fontsize=18)
    plt.ylabel(axes_labels[1], fontsize=18)
    fig.suptitle(title, fontsize=18)
    ax.tick_params(labelsize=14)
    # run all simulations
    for modelfile in modelfiles:
        for params in params_list:
            tc = timecourse(modelfile, t, spec, suppress=True, parameters=params)
            for s in range(len(spec)):
                trajectories[s].append(tc[spec[s][0]])
    # plot all trajectories
    for s in range(len(spec)):
        style_index = -1    
        for trajectory in trajectories[s]:
            style_index+=1
            if global_Norm==1:
                for s in range(len(spec)):
                    ax.plot(t, trajectory, style[style_index],
                                label=spec[s][1], linewidth=2.0,)
            elif type(global_Norm)==int or type(global_Norm)==float:
                ax.plot(t, trajectory/global_Norm, style[style_index],
                            label=spec[s][1], linewidth=2.0)
            else:
                print("global_Norm must be an int or float")
    if custom_legend != False:
        ax.legend(custom_legend)

# =============================================================================
# doseresponse() takes a model, a set of doses, and a set of time points 
# and runs a simulation using ScipyOdeSimulator. It then outputs a figure
# of the simulation.
# Inputs:
#      modelfile = previously imported model file
#      dose = list composed of ['parameter_title',[parameter values to scan]]
#      t = list of time points to simulate over (eg. t = linspace(20000))
#      spec = list of strings with each element being a 2-list in the form 
#              [name of an observable, label for observable]
#      axes_labels = 2-list of [x_label, y_label] (default is empty string)
#      title = string title label (default is empty string)
#      suppress = boolean to control whether or not to supress plotting (default is False)
#      Norm = normalization factor. Default = 1 (ie. no normalization)
#          If Norm is a float or int then every species number will be divided by Norm
#          If Norm is a list then:
#             if len(Norm)=len(spec) then each species will have a unique normalization factor
#                if type(Norm[0]) is float or int then that factor will be a number
#                otherwise the function will assume the normalization is a 
#                trajectory over the same number of doses.  
#      parameters = a dictionary to pass with custom parameters for simulation        
#      Outputs:
#      figure of the dose response (unless suppress=True)
#  Returns: final_dr = final species concentrations for each dose's timecourse
# =============================================================================
def doseresponse(modelfile, dose, t, spec, axes_labels = ['',''], title = '',
                 suppress=False, Norm=1, parameters=False):
    # Run a time course for each dose
    #For storing the final time point of each species' response at each dosasge
    #Each element in the list is the final-time dose-response curve for one species
    final_dr = [[] for i in range(len(spec))]
    dose_index=0 #Only needed for normalizing by trajectory
    for d in dose[1]:
        if parameters != False:
            # Add custom parameters before passing to simulation
            parameters[dose[0]] = d
        else:
            parameters = {dose[0]:d}
        simres = ScipyOdeSimulator(modelfile.model, tspan=t, 
                                   param_values = parameters, compiler='theano').run()
        simres = simres.all
        # Get the final time point for each dose
        if Norm==1:#No normalization
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
            ax.plot(dose[1], final_dr[species], label=spec[species][1], linewidth=2.0)
            plt.legend()
    return final_dr
# =============================================================================
# compare_doseresponse() takes two models, a set of doses, and a set of time  
# points and runs each simulation using ScipyOdeSimulator. It then outputs a 
# figure comparing the two dose-response curves.
# Inputs:
#      modelfiles = a list of previously imported model files
#      dose = list composed of ['parameter_title',[parameter values to scan]]
#      time_list = list lists of time points to simulate over 
#      spec = a list of lists with elements of the form 
#               [name of an observable, label for observable] 
#     style = list of colours/markers to use for plotting; order corresponds to 
#                order that model files are given in
#      axes_labels = 2-list of [x_label, y_label] (default is empty string)
#      title = string title label (default is empty string)
#      Norm = normalization factor(s) to pass to doseresponse() 
#       *see documentation for doseresponse() for details
#       **If you want to use a single doseresponse() output, just include 
#    Norm=testNorm in arguments, where testNorm = doseresponse() for norm curve
#    customlegend = a list of labels to use if you want a customized legend
#    params_list = a list of dicts, each dict being passed to the corresponding 
#                  model file for simulation with custom parameters    
#  Outputs:
#      figures comparing the dose response curves, one figure per species given
#  Returns: Nothing
# =============================================================================
def compare_doseresponse(modelfiles, dose, time_list, spec, style, 
                         axes_labels = ['',''], title = '', Norm=1, 
                         custom_legend=False, params_list=False):
    model_traj=[]#For storing each simulation
    # Call doseresponse() for each model file
    for m in range(len(modelfiles)):
        if params_list == False:
            model_traj.append(doseresponse(modelfiles[m], dose, time_list[m], spec, 
                            axes_labels, title, True, Norm ))
        else:
             model_traj.append(doseresponse(modelfiles[m], dose, time_list[m], spec, 
                            axes_labels, title, True, Norm, parameters=params_list[m] ))

    # Now plot each species as its own figure
    for s in range(len(spec)):
        style_index=0 # just used to keep track of the colour to plot with

        plt.ion()
        fig, ax = plt.subplots()
        ax.set(xscale="log", yscale="linear")
        plt.xlabel(axes_labels[0], fontsize=18)
        plt.ylabel(axes_labels[1], fontsize=18)
        fig.suptitle(title+"\n"+spec[s][1], fontsize=18)
        ax.tick_params(labelsize=14)

        for m in model_traj:
            ax.plot(dose[1], m[s], style[style_index], linewidth=2.0)
            style_index+=1
        if custom_legend!=False:
            plt.legend(custom_legend)
        plt.show()
        
# =============================================================================
# Suite of functions for weighted populations: Wtimecourse(), 
# Wcompare_timecourse(), Wdoseresponse(), and Wcompare_doseresponse()
# =============================================================================
# =============================================================================
# Wtimecourse() takes a model and a set of time points and runs a simulation using 
# ScipyOdeSimulator. It then outputs a figure of the simulation.
# Inputs:
#     modelfiles = a list of previously imported model files
#     weights = a list of weights; the order of weights corresponds to the 
#               order they are given in modelfiles.
#     params = a list of dictionaries, each dictionary containing the 
#               unique parameter values to use for each model file, in the same 
#               order as given in modelfiles    
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
#      suppress = a boolean determining whether or not to plot the results. If false
#                   then setting Norm will have no effect        
# Outputs:
#     figure of the time course
# Returns: timecourse = ScipyOdeSimulator().run().all
# =============================================================================
def Wtimecourse(modelfiles, weights, params, t, spec, axes_labels = ['',''], 
                title = '', Norm=1, suppress=False):
    sims = [] #Used to store the results of each modelfile's simulation
    # Run simulation
    for modelindex in range(len(modelfiles)):
        simres = ScipyOdeSimulator(modelfiles[modelindex].model, tspan=t,
                                   param_values=params[modelindex],
                                   compiler='python').run()
        sims.append(simres.all)
    # Create weighted sum of all simulations
    weightedSim={} #This will be the new dictionary of simulation results
    for species in spec:
        key = species[0]
        for modelindex in range(len(modelfiles)):
            if key not in weightedSim:
                weightedSim[key]=weights[modelindex]*sims[modelindex][key]
            else:
                weightedSim[key]+=weights[modelindex]*sims[modelindex][key]
    # Plot results
    if suppress==False:
        plt.ion()
        fig, ax = plt.subplots()
        plt.xlabel(axes_labels[0], fontsize=18)
        plt.ylabel(axes_labels[1], fontsize=18)
        fig.suptitle(title, fontsize=18)
        ax.tick_params(labelsize=14)
        if Norm==1:
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
                        ax.plot(t, np.divide(weightedSim[species[0]],Norm[spec_count]),
                            label=species[1], linewidth=2.0,)
                        spec_count+=1
        plt.legend()
    return weightedSim

# =============================================================================
# Wcompare_timecourse() takes list of combinations of models, weights,
# and parameters for each model, and runs Wtimecourse() for each combo.
# It outputs a figure comparing these weighted timecourses.
# Inputs:
#      modelfiles = a list lists, each sublist composed of model files to be 
#                   included in a weighted sum    
#      weights_list = a list of lists, each sublist composed of weights for
#                   its corresponding list of models in modelfiles
#                   (ie. weights_list[0] corresponds to weights for modelfiles[0])    
#      params_list = a list of lists, each sublist being a list of dictionaries
#                    to pass as parameters to Wtimecourse().
#                   Each dict in params_list[i] will be applied to its 
#                   corresponding model in modelfiles[i]
#      t = list of time points to simulate over (eg. t = linspace(20000))
#      spec = [name of an observable, label for observable] 
#          (only supports comparison of one observable)
#      style = list of colours/markers to use for plotting; order corresponds to 
#           [(modelfile1, params_list_1), (model_file1, params_list2),...
#           (model_file2, params_list1),...]
#      global_Norm = normalization factor. Default = 1 (ie. no normalization)
#          If Norm is a float or int then every timecourse will be divided by 
#           Norm. **Note that trajectory norms are not currently supported.
#      axes_labels = 2-list of [x_label, y_label] (default is empty string)
#      title = string title label (default is empty string)    
#      custom_legend = list of labels, order the same as style order
#                       (default is no legend)    
#  Outputs:
#      figure comparing the timecourse curves
#  Returns: Nothing
# =============================================================================
def Wcompare_timecourse(modelfiles, weights_list, params_list, t, spec, style,
                        global_Norm=1, axes_labels = ['',''], title = '', 
                        custom_legend=False):
    trajectories = [] # to store each simulation output
    # run all simulations
    for combo in range(len(modelfiles)):
        trajectories.append(Wtimecourse(modelfiles[combo], weights_list[combo],
                           params_list[combo], t, [spec], Norm=global_Norm,
                           suppress=True))
    # plot all trajectories
    # set up plot
    plt.ion()
    fig, ax = plt.subplots()
    plt.xlabel(axes_labels[0], fontsize=18)
    plt.ylabel(axes_labels[1], fontsize=18)
    fig.suptitle(title, fontsize=18)
    ax.tick_params(labelsize=14)    
    style_index = -1    
    for trajectory in trajectories:
        style_index+=1
        if global_Norm==1:
            ax.plot(t, trajectory[spec[0]], style[style_index],
                        label=spec[1], linewidth=2.0,)
        elif type(global_Norm)==int or type(global_Norm)==float:
            ax.plot(t, trajectory[spec[0]]/global_Norm, style[style_index],
                        label=spec[1], linewidth=2.0)
        else:
            print("global_Norm must be an int or float")
    if custom_legend != False:
        ax.legend(custom_legend)

# =============================================================================
# Wdoseresponse() takes a combination of models, a set of doses, and a set of 
# time points, and runs a simulation using ScipyOdeSimulator. It then outputs 
# a figure of the simulation.
# Inputs:
#      modelfiles = list of model files to weight
#      weights = list of weights to use for each model file
#      parameters = a list of dictionaries, each dict unique to its corresponding
#                   model in modelfiles        
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
#      Outputs:
#      figure of the dose response (unless suppress=True)
#  Returns: final_dr = list of the weighted response trajectory for each species
# =============================================================================
def Wdoseresponse(modelfiles, weights, parameters, dose, t, spec, 
                 axes_labels = ['',''], title = '', suppress=False, wNorm=1):
    # Run a time course for each dose
    #For storing the final time point of each species' response at each dosasge
    #Each element in the list is the final-time dose-response curve for one species
    final_dr = [[] for i in range(len(spec))]
    dose_index=0 #Only needed for normalizing by trajectory
    for d in dose[1]:
        # add dose to each dict of parameters
        for c in parameters:
            c[dose[0]]=d
        # run simulation
        wtraj = Wtimecourse(modelfiles, weights, parameters, t, spec, 
                            Norm=wNorm, suppress=True)
        # remove dose from each dict of parameters
        for c in parameters:
            del c[dose[0]]
        # Get the final time point for each dose
        if wNorm==1:#No normalization
            for i in range(len(spec)):
                #species i .append(wtraj['species'][last time])
                final_dr[i].append(wtraj[spec[i][0]][-1])
        elif type(wNorm)==int or type(wNorm)==float:#Global normalization
            for i in range(len(spec)):
                final_dr[i].append(wtraj[spec[i][0]][-1]/wNorm)
        elif type(wNorm)==list:
            if type(wNorm[0])==int or type(wNorm[0])==float:#species-specific normalization factors
                for i in range(len(spec)):
                    final_dr[i].append(wtraj[spec[i][0]][-1]/wNorm[i])
            else:#species-specific trajectory to normalize by (ie. norm-factor is a function of x-axis variable)
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
        for species in range(len(spec)):
            ax.plot(dose[1], final_dr[species], label=spec[species][1], linewidth=2.0,)
            plt.legend()
    return final_dr

# =============================================================================
# Wcompare_doseresponse() takes two sets of models, a set of doses, and a set 
# of time points and runs each simulation using ScipyOdeSimulator. 
# It then outputs a figure comparing the two dose-response curves.
# Inputs:
#      modelfiles = a list of lists, each sublist being a combination of 
#                   models to contribute to a weighted trajectory
#       weights_list = a list of lists, each sublist being a list of the 
#                       weights to use for the corresponding sublist in 
#                       modelfiles (order must correspond)
#      dose = list composed of ['parameter_title',[parameter values to scan]]
#      params_list = list of lists, each sublist corresponding to a list of 
#                   dictionaries to use for the corresponding sublist in modelfiles
#      time_list = list of lists of time points to simulate over
#      spec = [[name of an observable, label for observable],...] 
#     style = list of colours/markers to use for plotting; order corresponds to 
#                order that model files are given in
#      axes_labels = 2-list of [x_label, y_label] (default is empty string)
#      title = string title label (default is empty string)
#      Norm = normalization factor(s) to pass to doseresponse() 
#       *see documentation for doseresponse() for details
#       **If you want to use a single doseresponse() output, just include 
#           Norm=testNorm in arguments, where testNorm = doseresponse() for 
#                norm curve    
#  Outputs:
#      figure comparing the dose response curves
#  Returns: Nothing
# =============================================================================
def Wcompare_doseresponse(modelfiles, weights_list, dose, params_list, time_list,
                          spec, style, axes_labels = ['',''], title = '', Norm=1):
    plt.ion()
    fig, ax = plt.subplots()
    ax.set(xscale="log", yscale="linear")
    plt.xlabel(axes_labels[0], fontsize=18)
    plt.ylabel(axes_labels[1], fontsize=18)
    fig.suptitle(title, fontsize=18)
    ax.tick_params(labelsize=14)
    style_index = 0
    # Call doseresponse() for each model file
    for m in range(len(modelfiles)):
        m_dr = Wdoseresponse(modelfiles[m], weights_list[m], params_list[m],  
                 dose, time_list[m], spec, suppress=True, wNorm=Norm)
        for species in range(len(spec)):
            ax.plot(dose[1], m_dr[species], style[style_index], 
                    label=spec[species][1], linewidth=2.0)
        style_index+=1
    plt.legend()
    plt.show()

