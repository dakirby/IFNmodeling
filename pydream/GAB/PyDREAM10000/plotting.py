import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


# Load data
samples = np.load('mixed_IFN_DREAM_chain_history.npy')
samples = np.reshape(samples[0:int(len(samples)/6) * 6], ( int(len(samples)/6), 6 ))

# Plot output
total_iterations = len(samples[0])
burnin = int(total_iterations / 2)

ndims = len(samples)
colors = sns.color_palette(n_colors=ndims)

# Convert back to true value rather than log value
converted_samples = np.power(np.multiply(np.ones(np.shape(samples)), 10), samples)

# Convert to dataframe
df = pd.DataFrame(converted_samples, columns=['kpa', 'kSOCSon', 'kd4', 'k_d4', 'R1', 'R2'])
g = sns.pairplot(df, )
for i, j in zip(*np.triu_indices_from(g.axes, 1)):
    g.axes[i,j].set_visible(False)
g.savefig('corner_plot.pdf')
