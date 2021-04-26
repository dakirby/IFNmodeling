from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import DoseresponsePlot, TimecoursePlot
from numpy import linspace, logspace, log10, nan
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
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
    #for edge, spine in ax.spines.items():
    #    spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    #ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
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
    # ------------------------------
    # Align all data
    # ------------------------------
    newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
    newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
    newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
    newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")

    alignment = DataAlignment()
    alignment.add_data([newdata_4, newdata_3, newdata_2, newdata_1])
    alignment.align()
    alignment.get_scaled_data()
    mean_data = alignment.summarize_data()

    # -------------------------------
    # Initialize model
    # -------------------------------
    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
    Mixed_Model.set_parameters({'R2': 4920, 'R1': 1200,
                                'k_a1': 2.0e-13, 'k_a2': 1.328e-12, 'k_d3': 1.13e-4, 'k_d4': 0.9,
                                'kSOCSon': 5e-08, 'kpu': 0.0022, 'kpa': 2.36e-06,
                                'ka1': 3.3e-15, 'ka2': 1.85e-12, 'kd4': 2.0,
                                'kd3': 6.52e-05,
                                'kint_a':  0.0015, 'kint_b': 0.002,
                                'krec_a1': 0.01, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05})
    scale_factor = 1.46182313424

    # ------------------------------------------
    # Make model predictions for mixtures of IFN
    # ------------------------------------------
    times = newdata_4.get_times('Alpha')
    doses_alpha = newdata_4.get_doses('Alpha')
    doses_alpha = [0.0] + [round(el, 2) for el in list(np.logspace(np.log10(doses_alpha[0]), np.log10(doses_alpha[-1])))]
    doses_beta = newdata_4.get_doses('Beta')
    doses_beta = [0.0] + [round(el, 2) for el in list(np.logspace(np.log10(doses_beta[0]), np.log10(doses_beta[-1])))]

    model_scan = {}
    for d in doses_beta:
        dradf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', doses_alpha,
                                         parameters={'Ib': d*1E-12*6.022E23*1E-5}, return_type='dataframe', dataframe_labels='Alpha',
                                         scale_factor=scale_factor)
        model_scan.update({d: dradf})
    # -------------------------------
    # Plot Heatmaps
    # -------------------------------
    fig, axes = plt.subplots(nrows=1, ncols=len(times))
    for ax in axes:
        ax.set_xlabel(r'IFN$\alpha$ (pM)')
        ax.set_ylabel(r'IFN$\beta$ (pM)')
    fig.set_size_inches(8*6, 8)
    for t_idx, t in enumerate(times):
        axes[t_idx].set_title('{} min'.format(t))
        response_data = [[]]*len(doses_beta)
        for d_idx, d in enumerate(doses_beta):
            row = [el[0] for el in model_scan[d].loc['Alpha'].iloc[:, t_idx].values]
            response_data[len(doses_beta)-d_idx-1] = row

        im, cbar = heatmap(np.asarray(response_data), doses_beta[::-1], doses_alpha, ax=axes[t_idx],
                           cmap=discrete_cmap(12, base_cmap="BuGn"), cbarlabel="pSTAT", vmin=0, vmax=4000)
    fig.tight_layout()
    fig.show()
    plt.savefig("results/combined_IFN_model.pdf")

    # --------------------------------
    # Check how nonlinear the combo is
    # --------------------------------
    dradf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', doses_alpha, parameters={'Ib': 0},
                                     return_type='dataframe', dataframe_labels='Alpha', scale_factor=scale_factor)
    drbdf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', doses_beta, parameters={'Ia': 0},
                                     return_type='dataframe', dataframe_labels='Beta', scale_factor=scale_factor)

    fig, axes = plt.subplots(nrows=1, ncols=len(times))
    for ax in axes:
        ax.set_xlabel(r'IFN$\alpha$ (pM)')
        ax.set_ylabel(r'IFN$\beta$ (pM)')
    fig.set_size_inches(8*6, 8)
    for t_idx, t in enumerate(times):
        # Get simple addition of independent responses, in table form
        alpha_response = [el[0] for el in dradf.loc['Alpha'].iloc[:,t_idx].values]
        beta_response = np.flipud([[el[0]]*len(alpha_response) for el in drbdf.loc['Beta'].iloc[:,t_idx].values])
        alpha_response = [alpha_response for i in range(len(beta_response))]
        linear_response = np.add(alpha_response, beta_response)

        # Get model predictions
        axes[t_idx].set_title('{} min'.format(t))
        response_data = [[]]*len(doses_beta)
        for d_idx, d in enumerate(doses_beta):
            row = [el[0] for el in model_scan[d].loc['Alpha'].iloc[:, t_idx].values]
            response_data[len(doses_beta)-d_idx-1] = row
        degree_of_nonlinearity = np.log10(-np.subtract(response_data, linear_response))
        im, cbar = heatmap(np.asarray(degree_of_nonlinearity), doses_beta[::-1], doses_alpha, ax=axes[t_idx],
                           cmap=discrete_cmap(12, base_cmap="BuGn"),
                           cbarlabel="Log Negative Difference from Linear Response",
                           vmin=-1, vmax=3.5)
    fig.suptitle('Degree of Nonlinearity')
    fig.tight_layout()
    fig.show()
    plt.savefig("results/combined_IFN_nonlinearity.pdf")

    # --------------------------------------------------------
    # Check how different from the beta-only response this is
    # --------------------------------------------------------
    drbdf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', doses_beta, parameters={'Ia': 0},
                                     return_type='dataframe', dataframe_labels='Beta', scale_factor=scale_factor)

    fig, axes = plt.subplots(nrows=1, ncols=len(times))
    for ax in axes:
        ax.set_xlabel(r'IFN$\alpha$ (pM)')
        ax.set_ylabel(r'IFN$\beta$ (pM)')
    fig.set_size_inches(8*6, 8)
    for t_idx, t in enumerate(times):
        # Get simple addition of independent responses, in table form
        alpha_response = [0 for el in dradf.loc['Alpha'].iloc[:,t_idx].values]
        beta_response = np.flipud([[el[0]]*len(alpha_response) for el in drbdf.loc['Beta'].iloc[:,t_idx].values])
        alpha_response = [alpha_response for i in range(len(beta_response))]
        linear_response = np.add(alpha_response, beta_response)

        # Get model predictions
        axes[t_idx].set_title('{} min'.format(t))
        response_data = [[]]*len(doses_beta)
        for d_idx, d in enumerate(doses_beta):
            row = [el[0] for el in model_scan[d].loc['Alpha'].iloc[:, t_idx].values]
            response_data[len(doses_beta)-d_idx-1] = row
        degree_of_nonlinearity = np.subtract(response_data, linear_response)
        im, cbar = heatmap(np.asarray(degree_of_nonlinearity), doses_beta[::-1], doses_alpha, ax=axes[t_idx],
                           cmap=discrete_cmap(12, base_cmap="BuGn"),
                           cbarlabel="Difference from Beta Response",
                           vmin=-10, vmax=4000)
    fig.suptitle('Difference from Beta')
    fig.tight_layout()
    fig.show()
    plt.savefig("results/combined_IFN_diff_from_beta.pdf")