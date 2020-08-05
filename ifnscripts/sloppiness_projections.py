from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import numdifftools as nd
import json
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

def discrete_cmap(N, base_cmap=None):
    # By Jake VanderPlas
    # License: BSD-style
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


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
    # -----------------
    # Plot eigenvectors
    # -----------------
    evals, evecs = np.linalg.eig(H)
    idx = evals.argsort()[::-1]   
    evals = evals[idx]
    evecs = evecs[:,idx]
    fig, ax = plt.subplots()
    # Plot the heatmap
    im, cbar = heatmap(evecs, parameters_to_test, ["{:.2E}".format(el) for el in evals], ax=ax,
                   cmap=discrete_cmap(8,base_cmap="PuOr"), cbarlabel="projections onto parameter space")
    fig.tight_layout()
    plt.show()
    # ------------------------------------
    # Get major components of eigenvectors
    # ------------------------------------
    component_list = {}
    for idx in range(5):
        coverage = 0
        components = []
        particular_evec = np.array(evecs[:,idx]).flatten()
        sorted_components = particular_evec[np.abs(particular_evec).argsort()[::-1]]
        while np.sqrt(coverage)<0.8:
            next_dimension = sorted_components[0]
            sorted_components = np.delete(sorted_components, 0)
            components.append((parameters_to_test[np.where(particular_evec == next_dimension)[0][0]], next_dimension))
            coverage += next_dimension ** 2
        component_list.update({idx:components})    
    with open(os.path.join('results','sloppiness_projections.txt'),'w') as f:
        f.write(json.dumps(component_list))
