from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import seaborn as sns
import pandas as pd
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import load_model as lm
import os
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

if __name__ == '__main__':
    # Prepare output directory
    out_dir = os.path.join(os.getcwd(), 'results', 'Figures', 'Supplementary')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fname = out_dir + os.sep + 'detailed_vs_simple.pdf'

    # ---------------------------
    # Set up simple model
    # ---------------------------
    Simple_Model, DR_method = lm.load_model(MODEL_TYPE='SINGLE_CELL')

    scale_factor, DR_KWARGS, PLOT_KWARGS = 1.227, {'return_type': 'IfnData'}, {'line_type': 'plot', 'alpha': 1}

    times = [2.5, 5.0, 7.5, 10., 20., 60.]
    alpha_doses = list(logspace(1, 5.2, num=20))
    beta_doses = list(logspace(-1, 4, num=20))
    dra_s = DR_method(times, 'TotalpSTAT', 'Ia', alpha_doses,
                      parameters={'Ib': 0}, dataframe_labels='Alpha', sf=scale_factor, **DR_KWARGS)

    drb_s = DR_method(times, 'TotalpSTAT', 'Ib', beta_doses,
                      parameters={'Ia': 0}, dataframe_labels='Beta', sf=scale_factor, **DR_KWARGS)

    # -------------------------------
    # Now repeat for detailed model:
    # -------------------------------
    Detailed_Model, Detailed_DR_method = lm.load_model(model_name='Mixed_IFN_detailed',
                                                       MODEL_TYPE='SINGLE_CELL')

    # Match the simple model predictions using only unconstrained parameters:
    best_match = {'kloc': 1.25E-4, 'kdeloc': 0.6, 'kSOCSmRNA': 0.6, 'mRNAtrans': 0.8,
                  'mRNAdeg': 0,#5.00e-06,
                  'kpa': 900.0, 'kpu': 0.0020}
    # Also match shared parameters with simple model
    # best_match.update({'kSOCSon': 1.03992e-06, 'kpa': 1.e-06,
    #                    'kint_a': 3.737e-05, 'kint_b': 0.0002085,
    #                    'krec_a1': 0.00179, 'krec_a2': 0.00912,
    #                    'R1': 2000., 'R2': 2023.})
    # scale_factor = 0
    Detailed_Model.set_parameters(best_match)

    # Make detailed model predictions
    dra_d = Detailed_DR_method(times, 'TotalpSTAT', 'Ia', alpha_doses,
                               parameters={'Ib': 0}, dataframe_labels='Alpha', sf=scale_factor, **DR_KWARGS)
    drb_d = Detailed_DR_method(times, 'TotalpSTAT', 'Ib', beta_doses,
                               parameters={'Ia': 0}, dataframe_labels='Beta', sf=scale_factor, **DR_KWARGS)

    # ----------------------------------------
    # Finally, plot both models in comparison
    # ----------------------------------------
    fig = plt.figure(figsize=(6.4 * 2.5, 4.8))
    gs = gridspec.GridSpec(nrows=1, ncols=5)
    panelA = fig.add_subplot(gs[0, 0:2])
    panelB = fig.add_subplot(gs[0, 2:4])
    for ax in [panelA, panelB]:
        ax.set(xscale='log', yscale='linear')
        ax.set_xlabel('Dose (pM)')
        ax.set_ylabel('Response')
    legend_panel = fig.add_subplot(gs[0, 4])

    new_fit = DoseresponsePlot((1, 2))
    new_fit.fig = fig
    new_fit.axes = [panelA, panelB, legend_panel]

    alpha_palette = sns.color_palette("rocket_r", 6)
    beta_palette = sns.color_palette("rocket_r", 6)
    t_mask = [2.5, 7.5, 20.]
    # Add fits
    for idx, t in enumerate(times):
        if t not in t_mask:
            new_fit.add_trajectory(dra_s, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', linewidth=2.0)
            new_fit.add_trajectory(dra_d, t, 'plot', '--', (0, 0), 'Alpha', color=alpha_palette[idx], linewidth=2.0)
            new_fit.add_trajectory(drb_s, t, 'plot', beta_palette[idx], (0, 1), 'Beta', linewidth=2.0)
            new_fit.add_trajectory(drb_d, t, 'plot', '--', (0, 1), 'Beta', color=beta_palette[idx], linewidth=2.0)

    new_fit.show_figure(show_flag=False, save_flag=False)

    # formatting and legend
    for ax in [panelA, panelB]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_legend().remove()
        ax.set_ylim([1, 5000])
    panelB.set_ylabel('')
    panelA.plot([], [], 'k', label='Simple Model', linewidth=2.0)
    panelA.plot([], [], 'k--', label='Detailed Model', linewidth=2.0)
    for idx, t in enumerate(times):
        if t not in t_mask:
            panelA.scatter([], [], color=alpha_palette[idx], label='{} min'.format(t), s=30)
    panelA.legend()
    handles, labels = panelA.get_legend_handles_labels()  # get labels and handles
    legend_panel.legend(handles, labels)
    legend_panel.axis('off')
    panelA.get_legend().remove()  # now turn legend off in panel

    fig.tight_layout()
    fig.savefig(fname)
