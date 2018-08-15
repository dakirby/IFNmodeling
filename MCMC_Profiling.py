# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 08:22:35 2018

@author: Duncan

Scaling performance for MCMC
"""
import mcmc
import time
import matplotlib.pyplot as plt
plt.close('all')


def testFunc(x):
    modelfiles = ['IFN_alpha_altSOCS_ppCompatible','IFN_beta_altSOCS_ppCompatible']
    # Write modelfiles
    alpha_model = __import__(modelfiles[0])
    py_output = export(alpha_model.model, 'python')
    with open('ODE_system_alpha.py','w') as f:
        f.write(py_output)
    beta_model = __import__(modelfiles[1])
    py_output = export(beta_model.model, 'python')
    with open('ODE_system_beta.py','w') as f:
        f.write(py_output)
    p0=[['kpa',1E-6,0.1,'log'],['kSOCSon',1E-6,0.1,'log'],['kd4',0.3,0.2,'log'],
        ['k_d4',0.006,0.5,'log'],['delR',0,500,'linear'],
        ['gamma',2,0.5,'log']]
    MCMC(500, p0, 8, 3, burn_rate=0.05, down_sample=1)

# =============================================================================




# =============================================================================
# profile creates a speed up profile for a piece of code, in this case mcmc.py
# It assumes that the first value in processes is 1
# Inputs: processes (list) = a list of int values for number of threads to test with
#                             first element must be 1 to test serial time
# Outputs: plot of the speedup 
# =============================================================================
def profile(processes):
    if processes[0]!=1:
        print("Must test serial time. Please ensure processes[0]==1")
        return 1
    times = []
    for p in processes:
        tic = time.clock()
        testFunc(p)
        toc = time.clock()
        times.append(toc - tic)
    fig, ax = plt.subplots()
    ax.plot(processes[1:], [times[0]/times[i] for i in range(1,len(times))], 'bo')
    ax.set_title('Speedup for MCMC')
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Speedup')
    plt.savefig('speedup.pdf')
    plt.show()
    
def main():
    profile([1,2,3])
main()
    
