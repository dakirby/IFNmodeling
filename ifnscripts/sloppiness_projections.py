from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numdifftools as nd
import os

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    #ax.tick_params(top=True, bottom=False,
    #               labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="right")
             #rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

if __name__ == '__main__':
    Detailed_Model = IfnModel('Mixed_IFN_detailed')
    # Put detailed model into parameter state best fit for Ratnadeep's data (20181113_B6_IFNs_Dose_Response_Bcells)
    baseline_parameters = {'R2': 2300 * 2.5,
         'R1': 1800 * 1.8, 'k_d4': 0.06, 'kint_b': 0.0008,
         'kpu': 0.0028,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 6, 'kSOCSon': 0.9e-8, 'k_d4': 0.06,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3, 'kd4': 1.0, 'kd3': 0.001,
         'kint_a': 0.0014, 'krec_a1': 9e-03, 'krec_a2': 0.05}
    Detailed_Model.set_parameters(baseline_parameters)
    # -----------------
    # Compute Hessian
    # -----------------
    #   First create parameter vector to compute Hessian at
    parameters_to_test = list(baseline_parameters.keys())
    all_parameters = ['ka1', 'kd1', 'ka2', 'kd2', 'ka3', 'kd3', 'ka4', 'kd4', 'k_a1', 'k_d1', 'k_a2', 'k_d2', 'k_a3', 'k_d3', 'k_a4', 'k_d4',
     'kint_a', 'kint_b', 'krec_a1', 'krec_a2', 'krec_b1', 'krec_b2', 'kIntBasal_r1', 'kIntBasal_r2', 'krec_r1', 'krec_r2', 'kdeg_a', 'kdeg_b',
     'kSTATbinding', 'kSTATunbinding', 'kpa', 'kpu', 'kloc', 'kdeloc', 'mRNAdeg', 'mRNAtrans', 'kSOCS', 'SOCSdeg', 'kSOCSon', 'kSOCSoff']
    parameters_to_test.extend([el for el in all_parameters if el not in parameters_to_test])

    # Add jitter to best fit parameters to avoid numerical instability of finding Hessian at functional 0 (?)
    best_parameters = np.log([Detailed_Model.parameters[key] * np.random.uniform(0.97, 1.03) for key in parameters_to_test])

    H = pickle.load(open('results/Sloppiness/Detailed_Model_Hessian.pkl','rb'))    
    evals, evecs = np.linalg.eig(H)
    idx = evals.argsort()[::-1]   
    evals = evals[idx]
    evecs = evecs[:,idx]
    print(np.shape(evecs))
    print(np.shape(parameters_to_test))
    # Plot eigenvectors
    fig, ax = plt.subplots()
    # Plot the heatmap
    im, cbar = heatmap(evecs, parameters_to_test, ["{:.2E}".format(el) for el in evals], ax=ax,
                   cmap="viridis", cbarlabel="projections onto parameter space")
    fig.tight_layout()
    plt.show()
