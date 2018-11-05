# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 19:07:02 2018

@author: Duncan

Autocorrelation re-analysis
Gelman-Rubin re-analysis
Re-plot parameter autocorrelations
Plot trace
Compute pairwise correlations
Score posterior models
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import seaborn as sns; 
sns.set(color_codes=True)
sns.set_style("ticks")
import pandas as pd
import os

# Analyse how converged the chains were
    # chain_record = a list of all chain record dataframes
def gelman_rubin_reanalysis(chain_record, save_dir):
    numChains = len(chain_record)
    column_names = list(chain_record[0].columns.values)
    stats = []
    for variable in column_names:
        chain_mean = [chain_record[i][variable].mean() for i in range(numChains)]
        W = np.mean([(chain_record[i][variable].std())**2 for i in range(numChains)])
        B = chain_record[0].shape[0]*np.std(chain_mean, ddof=1)**2
        Var = (1-1/chain_record[0].shape[0])*W+B/chain_record[0].shape[0]
        Rhat = np.sqrt(Var/W)
        stats.append([variable, Rhat])
    dfGR = pd.DataFrame.from_records(stats, columns=['variable','GR Diagnostic'])
    dfGR.to_csv(save_dir+'Gelman-Rubin_Reanalysis.csv')
    return stats

# Check the autocorrelation of a chain record
    # df = dataframe of chain record
    # length = value to use for maxlags in autocorrelation function
    # ID = chain ID to use for saving the figure output
def replot_parameter_autocorrelations(df,length, ID, save_dir):
    k = len(df.columns) # total number subplots
    n = 2 # number of chart columns
    m = (k - 1) // n + 1 # number of chart rows
    fig, axes = plt.subplots(m, n, figsize=(n * 5, m * 3))
    if k % 2 == 1: # avoids extra empty subplot
        axes[-1][n-1].set_axis_off()
    for i, (name, col) in enumerate(df.iteritems()):
        r, c = i // n, i % n
        ax = axes[r, c] # get axis object
        ax.set_title(name)
        ax.acorr(col, maxlags=length)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
    fig.tight_layout()    
    plt.savefig(save_dir+'chain_'+str(ID)+'autocorrelation_reanalysis.pdf')
    plt.show()

# Plots the trace of a particular chain
    # chain = dataframe of the chain of interest
    # n_samples = the first n_samples will be plotted
def plot_trace(chain, n_samples, save_dir):
    k = len(chain.columns) # total number subplots
    n = 2 # number of chart columns
    m = (k - 1) // n + 1 # number of chart rows
    fig, axes = plt.subplots(m, n, figsize=(n * 5, m * 3))
    if k % 2 == 1: # avoids extra empty subplot
        axes[-1][n-1].set_axis_off()
    for i, (name, col) in enumerate(chain.iteritems()):
        r, c = i // n, i % n
        ax = axes[r, c] # get axis object
        # determine whether or not to plot on log axis
        try:
            if abs(int(np.log10(np.max(col)))-int(np.log10(np.min(col)))) >= 4:
                ax.set(xscale='linear', yscale='log')
        except ValueError:
            print('Some parameters were negative-valued')
        ax.plot(range(n_samples), col[0:n_samples])
        ax.set_title(name)
    plt.show()
    plt.savefig(save_dir+'traceplot.pdf')    
        
# Outputs the pairwise correlation between variables to help understand autocorrelation
    # chain = dataframe of single chain
def pairwise_correlations(chain):
    from itertools import combinations
    k = len(chain.columns) # total number variables
    varnames = chain.columns.values
    all_pairs = list(combinations(range(k),2))
    results = pd.DataFrame(index=varnames, columns=varnames)
    for pair in all_pairs:
        correl = np.corrcoef([chain.loc[:,varnames[pair[0]]].values.tolist(), chain.loc[:,varnames[pair[1]]].values.tolist()])
        results.loc[varnames[pair[0]], varnames[pair[1]]] = correl[0][1]
    return results



def resample_simulation(results_dir, burn_in, down_sample, 
                        check_convergence=False, plot_autocorr=False, check_corr=False):
    if burn_in >1 or burn_in <0:
        print("burn_in should be between 0 and 1")
        return 0
    # Find the last save point from the simulation
    chain_results_dir = results_dir+'Chain_Results/'
    if not os.path.isdir(results_dir):
        raise EnvironmentError('Could not find the simulation directory')
    if not os.path.isdir(chain_results_dir):
        raise EnvironmentError('Could not find the most recent simulation save point')
    # Make a directory to place results in
    reanalysis_dir = os.path.join(results_dir, 'Reanalysis/')
    if not os.path.isdir(reanalysis_dir):
        os.makedirs(reanalysis_dir)
    
    # Build a list of chain lengths (compatible with incomplete simulation results which have non-uniform chain lengths)
    chainLengths=[0]
    with open(results_dir+'chain_lengths.txt', 'r') as f:
        chainLengths += map(int,f.readline().split(', '))
    chainLengths.append(-1)
    chains = pd.read_csv(results_dir+'complete_samples.csv',index_col=0)
    # Read in the chain results
    chainList = []
    for i in range(0,len(chainLengths)-2):
        chainList.append(chains.iloc[sum(chainLengths[0:i+1]):sum(chainLengths[0:i+2])])
    # Thin the chains and combine to produce posterior sample
    thinned_chain_list = [chainList[j].iloc[int(burn_in*len(chainList[j])):-1:down_sample] for j in range(len(chainList))]
    thinned_chain = pd.concat(thinned_chain_list)
    print("There are now {} samples in from the posterior distribution".format(len(thinned_chain)))
    thinned_chain.to_csv(reanalysis_dir+'posterior_sample_reanalysis.csv')
    print("Wrote posterior samples to posterior_sample_reanalysis.csv in directory Reanalysis/")
    if check_convergence==True:
        print("Computing Gelman-Rubin analysis")
        gelman_rubin_reanalysis(chainList,reanalysis_dir)
        
    if plot_autocorr==True:
        print("Checking autocorrelation")
        l=min(30,len(thinned_chain))
        replot_parameter_autocorrelations(thinned_chain,l, 'combined_',reanalysis_dir)        
    
    if check_corr==True:
        print('Analysing pairwise correlations')
        print(pairwise_correlations(chainList[0]))

def sample_incomplete_sim(results_dir, burn_in, down_sample):
    # Find the last save point from the simulation
    chain_results_dir = results_dir+'Chain_Results/'
    if not os.path.isdir(results_dir):
        raise EnvironmentError('Could not find the simulation directory')
    if not os.path.isdir(chain_results_dir):
        raise EnvironmentError('Could not find the most recent simulation save point')
    # Read each chain's last save point
    import ast
    numchains=len([name for name in os.listdir(chain_results_dir) if os.path.isfile(os.path.join(chain_results_dir, name))])
    chain_lengths=[]
    thinned_chains_list=[]
    var_names=[]
    # Grab variable names to initialise dataframe
    with open(chain_results_dir+'0chain.txt','r') as f:
            l = "["+f.read().rstrip()
            l=l[1:-1]+"]"
            chain = ast.literal_eval(l)
            var_names = [var[0] for var in chain[0]]    
    complete_samples = pd.DataFrame(columns=var_names)
    for i in range(numchains):
        with open(chain_results_dir+str(i)+'chain.txt','r') as f:
            l = "["+f.read().rstrip()
            l=l[1:-1]+"]"
            chain = ast.literal_eval(l)
            numRecords=len(chain)
            variable_names = [var[0] for var in chain[0]]
            records = [[var[1] for var in sample] for sample in chain]
            df = pd.DataFrame.from_records(records,columns=variable_names)
            chain_lengths.append(len(df))
            thinned_chains_list.append(df.iloc[int(burn_in*numRecords):-1:down_sample])
            complete_samples = pd.concat([complete_samples,df])
    # Write chain_lengths.txt so that reanalysis can be performed using resample_simulation()
    with open(results_dir+'chain_lengths.txt','w') as f:
        f.write(str(chain_lengths)[1:-1])
    # Write posterior samples to file
    post = pd.concat(thinned_chains_list)
    post.to_csv(results_dir+'posterior_samples.csv')  
    print("Posterior samples written to results directory as posterior_samples.csv")
    print("{} samples were taken from the posterior distribution".format(len(post)))
    complete_samples.to_csv(results_dir+'complete_samples.csv')
    print("All samples were combined and written to complete_samples.csv in results directory")
    print("File chain_lengths.txt written to results directory to allow further reanalysis")


