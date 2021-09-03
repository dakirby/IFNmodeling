"""
This script uses the IfnModel to simulate the effect on pSTAT1 response of
titrating in siRNA against IFNAR1 or IFNAR2

Author: Duncan Kirby
Date Created: 2021-08-30
"""
from ifnclass.ifndata import IfnData, DataAlignment
import load_model as lm
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
from numpy import array

DEBUG = False
if __name__ == '__main__':
    # --------------------
    # Import data
    # --------------------
    data = pd.read_csv('2011_Levin.csv')
    data_responses = [[data.loc[(data['IFN'] == 'IFN-YNS') & (data['IFNAR'] == 'IFNAR1')].values,
                       data.loc[(data['IFN'] == 'IFN-YNS') & (data['IFNAR'] == 'IFNAR2')].values],
                      [data.loc[(data['IFN'] == 'IFNa2') & (data['IFNAR'] == 'IFNAR1')].values,
                       data.loc[(data['IFN'] == 'IFNa2') & (data['IFNAR'] == 'IFNAR2')].values]]

    # --------------------
    # Set up Model
    # --------------------
    Mixed_Model, DR_method = lm.load_model(AFFINITY_SPECIES='HUMAN')
    scale_factor, DR_KWARGS, PLOT_KWARGS = lm.SCALE_FACTOR, lm.DR_KWARGS, lm.PLOT_KWARGS
    if DEBUG is True:
        Mixed_Model.num_dist_samples = 3

    # ------------------------------------------------------------
    # Make predictions following
    # Levin, Harari, & Schreiber (2011) Molc. & Cell. Bio.
    # ------------------------------------------------------------
    times = [45.]
    doses = [200]
    if DEBUG is True:
        IFNAR1_siRNA = [1., 0.5, 0.2]
        IFNAR2_siRNA = [1., 0.6, 0.4]
    else:
        IFNAR1_siRNA = np.linspace(1., 0.15, 15)  # [1., 0.85, 0.7, 0.6, 0.4, 0.2]
        IFNAR2_siRNA = np.linspace(1., 0.4, 15)  # [1., 0.9, 0.8, 0.6, 0.5, 0.4]
    siRNA_doses = [IFNAR1_siRNA, IFNAR2_siRNA]

    R_idx = [np.where(Mixed_Model.parameter_names == 'R1_mu*'),
             np.where(Mixed_Model.parameter_names == 'R2_mu*')]
    R_reference = [Mixed_Model.parameters[0][R_idx[0]][0],
                   Mixed_Model.parameters[0][R_idx[1]][0]]

    responses = [ [[], []],  [[], []] ]

    for i in [0, 1]:  # IFNAR1, then IFNAR2
        for Rsf in siRNA_doses[i]:  # Reduction in IFNARi
            # Update model with new IFNAR value
            Mixed_Model.parameters[0][R_idx[i]] = [R_reference[i] * Rsf]

            # Simulate pSTAT1 response for IFNa2 and IFNb
            dra60 = DR_method(times, 'TotalpSTAT', 'Ia', doses, parameters={'Ib': 0},
                                                    sf=scale_factor, **DR_KWARGS)
            responses[1][i].append(dra60.data_set.values.flatten())

            drb60 = DR_method(times, 'TotalpSTAT', 'Ib', doses, parameters={'Ia': 0},
                                                sf=scale_factor, **DR_KWARGS)
            responses[0][i].append(drb60.data_set.values.flatten())

        Mixed_Model.parameters[0][R_idx[i]] = [R_reference[i]]  # reset IFNARi

    # --------------------
    # Plot results
    # --------------------
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.5, 6))
    xlabels = ["IFNAR1", "IFNAR2"]
    ylabels = [r"IFN$\beta$", r"IFN$\alpha$2"]
    for i in [0, 1]:  # IFNARi index
        for IFN_idx in [0, 1]: # IFN type index
            # Prepare axis
            ax = axes[IFN_idx][i]
            ax.set_xlim(105, 0)
            ax.set_ylim(0, 120)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Format data to plot
            x = [100 * el for el in siRNA_doses[i]]
            y = [el[0][0] for el in responses[IFN_idx][i]]
            yerr = [el[0][1] for el in responses[IFN_idx][i]]
            # Normalize response to max response
            norm = max(y)
            yNorm = [100*el/norm for el in y]
            yerrNorm = [yNorm[i] * yerr[i]/y[i] for i in range(len(y))]
            y1 = [yNorm[i] + yerrNorm[i] for i in range(len(yNorm))]
            y2 = [yNorm[i] - yerrNorm[i] for i in range(len(yNorm))]
            ax.fill_between(x, y1, y2, alpha=0.2)
            ax.plot(x, yNorm)
            ax.scatter([], [], c='#1f77b4', label='Model')  # for legend
            # Data and axis labels
            ax.errorbar(data_responses[IFN_idx][i][:, 2], data_responses[IFN_idx][i][:, 3], data_responses[IFN_idx][i][:, 4], fmt='', c='k')
            ax.scatter(data_responses[IFN_idx][i][:, 2], data_responses[IFN_idx][i][:, 3], c='k', label='Data')
            ax.set_xlabel(xlabels[i] + " surface levels (%)")
            ax.set_ylabel("Relative pSTAT1 activity (%)")
            ax.set_title(ylabels[IFN_idx])
    axes[0][0].legend()
    fig.tight_layout()
    fname = os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_3', 'siRNA.pdf')
    plt.savefig(fname, )
