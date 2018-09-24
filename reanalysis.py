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
import os
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'MCMC_Results-17-09-2018/')
chain_results_dir = results_dir+'Chain_Results/'
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
if not os.path.isdir(chain_results_dir):
    os.makedirs(chain_results_dir)

from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import seaborn as sns; 
sns.set(color_codes=True)
sns.set_style("ticks")
import pandas as pd
chainLengths=[0]
with open(results_dir+'chain_lengths.txt', 'r') as f:
    chainLengths += map(int,f.readline().split(', '))
chainLengths.append(-1)
chains = pd.read_csv(results_dir+'complete_samples.csv',index_col=0)
chainList = []
for i in range(0,len(chainLengths)-2):
    chainList.append(chains.iloc[sum(chainLengths[0:i+1]):sum(chainLengths[0:i+2])])
import mcmc

def gelman_rubin_reanalysis(chain_record):
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
    dfGR.to_csv(results_dir+'Gelman-Rubin_Reanalysis.csv')
    return stats

def replot_parameter_autocorrelations(df,length, ID):
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
    plt.savefig(results_dir+'chain_'+str(ID)+'autocorrelation_reanalysis.pdf')
    plt.show()

def plot_trace(chain, n_samples):
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
    plt.savefig(results_dir+'traceplot.pdf')

def pairwise_correlations(chain):
    k = len(chain.columns) # total number variables
    varnames = chain.columns.values
    all_pairs = list(combinations(range(k),2))
    results = pd.DataFrame(index=varnames, columns=varnames)
    for pair in all_pairs:
        correl = np.corrcoef([chain.loc[:,varnames[pair[0]]].values.tolist(), chain.loc[:,varnames[pair[1]]].values.tolist()])
        results.loc[varnames[pair[0]], varnames[pair[1]]] = correl[0][1]
    return results

def score_posterior(filename,beta,rho):
    samples = pd.read_csv(filename,index_col=0)
    s=[]
    for i in range(len(samples)):
        model = zip(samples.columns.values,samples.iloc[i])
        model_dict = {key: value for (key, value) in model}
        s.append(mcmc.score_model(model_dict['kpa'],model_dict['kSOCSon'],model_dict['kd4'],
                             model_dict['k_d4'],model_dict['delR'],model_dict['gamma'],
                             beta, rho))
    samples['score']=s
    samples = samples.sort_values('score')
    samples.to_csv(script_dir+'posterior_scores.csv')

# =============================================================================
# import ast    
# reanalysis_dir = os.path.join(results_dir, 'Reanalysis/')
# if not os.path.isdir(reanalysis_dir):
#     os.makedirs(reanalysis_dir)
# 
# with open(results_dir+'posterior_samples.csv','r') as f:
#     l = "["+f.read().rstrip().replace("]]","]],")
#     l=l[0:-1]+"]"
#     myList = ast.literal_eval(l)
#     alpha_curve = [el[0][0] for el in myList]
#     plt.figure()
#     plt.plot(range(len(alpha_curve)),alpha_curve)
# =============================================================================
# =============================================================================
# import ast    
# s = os.path.join(script_dir, 'MCMC_Results/')
# with open(s+'2chain.txt','r') as f:
#     l = f.read().rstrip()
#     myList = ast.literal_eval(l)[100:452:20]
#     post = [[el[1] for el in sample] for sample in myList]
#     postLabels = [el[0] for el in myList[0]]
#     df = pd.DataFrame.from_records(post,columns=postLabels)
#     df.to_csv(s+'chain2posterior.csv')
# =============================================================================
    
gelman_rubin_reanalysis(chainList)
thinned_chain = [chainList[j].iloc[250:-1:90] for j in range(len(chainList))]
thinned_chain = pd.concat(thinned_chain)
print(len(thinned_chain))
thinned_chain.to_csv(results_dir+'posterior_reanalysis.csv')
replot_parameter_autocorrelations(thinned_chain, 30, 'total')

#plot_trace(chainList[1], 2000)
#print(pairwise_correlations(chainList[1]))
#score_posterior(results_dir+'posterior_reanalysis.csv',1950,1)