# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 19:07:02 2018

@author: Duncan

Autocorrelation re-analysis
Gelman-Rubin re-analysis
"""
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import seaborn as sns; 
sns.set(color_codes=True)
sns.set_style("ticks")
import pandas as pd
chainLengths=[0]
with open('chain_lengths.txt', 'r') as f:
    chainLengths += map(int,f.readline().split(', ')[0:-1])
chainLengths.append(-1)
chains = pd.read_csv('complete_samples.csv',index_col=0)
chainList = []
for i in range(0,len(chainLengths)-2):
    chainList.append(chains.iloc[chainLengths[i]:chainLengths[i]+chainLengths[i+1]])

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
    dfGR.to_csv('Gelman-Rubin_Reanalysis.csv')
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
    plt.savefig('chain_'+str(ID)+'autocorrelation_reanalysis.pdf')
    plt.show()

# =============================================================================
# for j in range(len(chainList)):
#     replot_parameter_autocorrelations(chainList[j][200:-1:1500], 8, j)
# =============================================================================
#gelman_rubin_reanalysis([chainList[j][200:-1:1500] for j in range(len(chainList))])
#replot_parameter_autocorrelations(pd.concat([chainList[j][200:-1:1500] for j in range(len(chainList))]), 15, 'total')

thinned_chain = pd.concat([chainList[j][200:-1:5000] for j in range(len(chainList))])
# =============================================================================
# Rdiff=[]
# for i in range(len(thinned_chain)):
#     Rdiff.append(thinned_chain['R1'].iloc[i]-thinned_chain['R2'].iloc[i])
# =============================================================================

fig, ax = plt.subplots()
ax.set_title('R1 R2 Cross-correlation')
c=np.correlate(thinned_chain['R1'],thinned_chain['R2'])
ax.plot(range(len(c)),c)
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
