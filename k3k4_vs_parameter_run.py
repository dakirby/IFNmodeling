# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 12:03:19 2018

@author: Duncan
"""
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
 scans: get_EC50(), mp_DR_parameter_scan()
"""
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
 scans: get_EC50(), mp_DR_parameter_scan()
"""
# Import statements required for the parallelized routines
from multiprocessing import Process, Queue, JoinableQueue, cpu_count
from pysb.export import export
from numpy import divide, subtract, abs, logspace, linspace, reshape, flipud, flip

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

# =============================================================================
def get_ODE_model(modelfile):
    ODE_system = __import__(modelfile)
    return ODE_system.Model()

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
    plt.xlabel(param1[0])
    plt.ylabel(param2[0])
    # Save figure
    plt.savefig("{}.pdf".format(title))

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


def k3k4_DRparamScan(modelfile, typeIFN, param2, testDose, t_list, spec, 
                  custom_params=None, Norm=False, cpu=None, suppress=False, doseNorm=1):    
    # build k3 and k4 parameters, performing a sanity check at the same time
    k4scan=[]
    k3scan=[]
    if typeIFN=='alpha':
        k4scan = 0.3*logspace(-3,3,num=len(param2[1]))
        k3scan = 3E-4*logspace(-3,3,num=len(param2[1]))
    elif typeIFN=='beta':
        k4scan = 0.0006*logspace(-3,3,num=len(param2[1]))
        k3scan = 1.2e-5*logspace(-3,3,num=len(param2[1]))
    else:
        print("Expected type of interferon to be either alpha or beta")
        return 1

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
    if typeIFN=='alpha':
        if custom_params == None:
            for val1 in range(len(k4scan)):
                for val2 in param2[1]:
                    params.append([['kd3',k3scan[val1]],['kd4',k4scan[val1]],[param2[0],val2]])
        else:
            for val1 in range(len(k4scan)):
                for val2 in param2[1]:
                    params.append([['kd3',k3scan[val1]],['kd4',k4scan[val1]],[param2[0],val2]]+[c for c in custom_params])
    elif typeIFN=='beta':
        if custom_params == None:
            for val1 in range(len(k4scan)):
                for val2 in param2[1]:
                    params.append([['k_d3',k3scan[val1]],['k_d4',k4scan[val1]],[param2[0],val2]])
        else:
            for val1 in range(len(k4scan)):
                for val2 in param2[1]:
                    params.append([['k_d3',k3scan[val1]],['k_d4',k4scan[val1]],[param2[0],val2]]+[c for c in custom_params])
        
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
        dose_image = [el[2]/doseNorm for el in pool_results]
        response_image = [el[3] for el in pool_results]
        response_image = reshape(response_image, (len(k4scan),len(param2[1])))
        dose_image = reshape(dose_image,(len(k4scan),len(param2[1])))
        if typeIFN=='alpha':
            IFN_heatmap(response_image, ['Response_EC50_alpha_k4',k4scan], param2)
            IFN_heatmap(dose_image, ['Dose_EC50_alpha_k4',k4scan], param2)
        else:
            IFN_heatmap(response_image, ['Response_EC50_beta_k4',k4scan], param2)
            IFN_heatmap(dose_image, ['Dose_EC50_beta_k4',k4scan], param2)
    #return the scan 
    return pool_results
	

# =============================================================================
# Uses pysb_parallel routines to make 2D parameter scans for model sensitivity.
# The metric for model response is the EC50
# =============================================================================
alpha_modelfilename = "IFN_simplified_model_alpha_ppCompatible"
beta_modelfilename = "IFN_simplified_model_beta_ppCompatible"
import seaborn as sns
sns.set_style("ticks")

def main():
    plt.close('all')
    t=linspace(0,3600,num=100)
    testDose = ['I',6.022e18*logspace(-14,-2,num=50)]
    # kSOCS
    yscan = logspace(-6,-5,num=10)
    t=linspace(0,3600,num=500)
    # Write modelfile
    import IFN_simplified_model_alpha_ppCompatible as alpha_model
    py_output = export(alpha_model.model, 'python')
    writeFile = "ODE_system_alpha.py"
    with open(writeFile, 'w') as f:
        f.write(py_output)

    import IFN_simplified_model_beta_ppCompatible as beta_model
    py_output = export(beta_model.model, 'python')
    writeFile = "ODE_system_beta.py"
    with open(writeFile, 'w') as f:
        f.write(py_output)

    # Run scans
# =============================================================================
    k3k4_DRparamScan("ODE_system_alpha", 'alpha', ['kSOCS',yscan],
                            testDose, t, [['TotalpSTAT',"Total pSTAT"]],
                            Norm=10000, doseNorm=6.022e18)
# =============================================================================
#    k3k4_DRparamScan("ODE_system_beta", 'beta', ['kSOCS',yscan],
#                            testDose, t, [['TotalpSTAT',"Total pSTAT"]],
#                            Norm=10000)
# =============================================================================
if __name__ == '__main__':
    main()

