from ifnclass.ifndata import IfnData, DataAlignment
from numpy import linspace, logspace
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from ifnclass.ifnplot import DoseresponsePlot
import matplotlib.gridspec as gridspec
import load_model as lm

PLOT_IFN = False
LOW_IFN = True

if __name__ == '__main__':
    # ----------------------
    # Set up Figure 2 layout
    # ----------------------
    Figure_2 = plt.figure(tight_layout=True, figsize=(11., 6.))
    gs = gridspec.GridSpec(nrows=2, ncols=4, height_ratios=[3, 2])

    Figure_2.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()

    # --------------------
    # Set up EC50 figures
    # --------------------
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)
    data_palette = sns.color_palette("muted", 6)
    marker_shape = ["o", "v", "s", "P", "d", "1", "x", "*"]
    dataset_names = ["20190108", "20190119", "20190121", "20190214"]

    # --------------------
    # Set up Model
    # --------------------
    Mixed_Model, DR_method = lm.load_model()
    # cell density in tissue is ~ 1E9 cells per mL, or 1E12 cells per L
    affinities = {'ka1': 2E5/(6.022E23*1E-12),
                  'ka2': 4E6/(6.022E23*1E-12),
                  'k_a1': 4E5/(6.022E23*1E-12),
                  'k_a2': 1E7/(6.022E23*1E-12),
                  'volEC': 1E-12}
    Mixed_Model.model.set_parameters(affinities)
    scale_factor, DR_KWARGS, PLOT_KWARGS = lm.SCALE_FACTOR, lm.DR_KWARGS, lm.PLOT_KWARGS

    # ---------------------------------
    # Make predictions
    times = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
    alpha_doses_20190108 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses_20190108 = [0, 0.2, 6, 20, 60, 200, 600, 2000]
    if LOW_IFN:
        doses = list(logspace(-1, 2, num=10))
    else:
        doses = list(logspace(1, 5.2, num=10))

    dfa_dict = {**{'Ib': 0}, **affinities}
    dfb_dict = {**{'Ia': 0}, **affinities}

    if PLOT_IFN:
        def plot_spec(observable, dose_spec, df_label):
            Mixed_Model.num_dist_samples = 5
            drb60 = DR_method(times, observable, dose_spec,
                                                    doses,
                                                    parameters=dfa_dict,
                                                    sf=scale_factor,
                                                    **DR_KWARGS)
            new_fit = DoseresponsePlot((1, 1))
            beta_palette = sns.color_palette("rocket_r", 6)
            for idx, t in enumerate(times):
                new_fit.add_trajectory(drb60, t, PLOT_KWARGS['line_type'],
                                       beta_palette[idx], (0, 1),
                                       df_label, label=str(t) + ' min',
                                       linewidth=2, alpha=PLOT_KWARGS['alpha'])
            fig, ax = new_fit.show_figure(show_flag=False)
            ax.set_yscale('log')

            if LOW_IFN:
                fig.savefig(os.path.join(os.getcwd(), 'results', observable+'_low_dose.pdf'))
            else:
                fig.savefig(os.path.join(os.getcwd(), 'results', observable+'.pdf'))

        plot_spec('Free_Ib', 'Ib', 'Beta')
        plot_spec('R1Ib', 'Ib', 'R1Ib')
        plot_spec('R2Ib', 'Ib', 'R2Ib')
        plot_spec('Tb', 'Ib', 'Tb')

        plot_spec('Free_Ia', 'Ia', 'Alpha')
        plot_spec('R1Ia', 'Ia', 'R1Ia')
        plot_spec('R2Ia', 'Ia', 'R2Ia')
        plot_spec('Ta', 'Ia', 'Ta')

    else:
        dra60 = DR_method(times, 'TotalpSTAT', 'Ia',
                                                doses,
                                                parameters=dfa_dict,
                                                sf=scale_factor,
                                                **DR_KWARGS)

        drb60 = DR_method(times, 'TotalpSTAT', 'Ib',
                                                doses,
                                                parameters=dfb_dict,
                                                sf=scale_factor,
                                                **DR_KWARGS)

        # ----------------------------------
        # Make model predictions for EC50
        # ----------------------------------
        time_list = list(linspace(2.5, 60, num=15))

        dfa = DR_method(time_list, 'TotalpSTAT', 'Ia',
                                              list(logspace(-3, 5)),
                                              parameters=dfa_dict,
                                              sf=scale_factor,
                                              **DR_KWARGS)

        alpha_ec_aggregate = [el[1] for el in dfa.get_ec50s()['Alpha']]
        alpha_peak_aggregate = [el[1] for el in dfa.get_max_responses()['Alpha']]

        dfb = DR_method(time_list, 'TotalpSTAT', 'Ib',
                                              list(logspace(-3, 5)),
                                              parameters=dfb_dict,
                                              sf=scale_factor,
                                              **DR_KWARGS)

        beta_ec_aggregate = [el[1] for el in dfb.get_ec50s()['Beta']]
        beta_peak_aggregate = [el[1] for el in dfb.get_max_responses()['Beta']]

        # -----------------------
        # Plot EC50 vs time
        # -----------------------
        ec50_axes_list = [Figure_2.add_subplot(gs[1, 0]),
                          Figure_2.add_subplot(gs[1, 1]),
                          Figure_2.add_subplot(gs[1, 2]),
                          Figure_2.add_subplot(gs[1, 3])]

        ec50_axes = ec50_axes_list[0:2]
        ec50_axes[0].set_xlabel("Time (min)")
        ec50_axes[1].set_xlabel("Time (min)")
        ec50_axes[0].set_title(r"EC50 vs Time for IFN$\alpha$2")
        ec50_axes[1].set_title(r"EC50 vs Time for IFN$\beta$")
        ec50_axes[0].set_ylabel("EC50 (pM)")
        ec50_axes[0].set_yscale('log')
        ec50_axes[1].set_yscale('log')
        # Add models
        ec50_axes[0].plot(time_list, alpha_ec_aggregate, label=r'IFN$\alpha$2',
                          color=alpha_palette[5], linewidth=2)
        ec50_axes[1].plot(time_list, beta_ec_aggregate, label=r'IFN$\beta$',
                          color=beta_palette[5], linewidth=2)

        # -------------------
        # Plot max response
        # -------------------
        # fig, axes = plt.subplots(nrows=1, ncols=2)
        max_response_axes = ec50_axes_list[2:4]
        max_response_axes[0].set_xlabel("Time (min)")
        max_response_axes[1].set_xlabel("Time (min)")
        max_response_axes[0].set_title(r"pSTAT$_{max}$ vs Time for IFN$\alpha$2")
        max_response_axes[1].set_title(r"pSTAT$_{max}$ vs Time for IFN$\beta$")
        max_response_axes[0].set_ylabel(r"pSTAT$_{max}$ (MFI)")

        # Add models
        max_response_axes[0].plot(time_list, alpha_peak_aggregate,
                                  color=alpha_palette[5], linewidth=2)
        max_response_axes[1].plot(time_list, beta_peak_aggregate,
                                  color=beta_palette[5], linewidth=2)

        # -------------------------------
        # Plot model dose response curves
        # -------------------------------
        alpha_palette = sns.color_palette("rocket_r", 6)
        beta_palette = sns.color_palette("rocket_r", 6)

        new_fit = DoseresponsePlot((1, 2))
        new_fit.axes = [Figure_2.add_subplot(gs[0, 0:2]),
                        Figure_2.add_subplot(gs[0, 2:4])]
        new_fit.axes[0].set_xscale('log')
        new_fit.axes[0].set_xlabel('Dose (pM)')
        new_fit.axes[0].set_ylabel('pSTAT (MFI)')
        new_fit.axes[1].set_xscale('log')
        new_fit.axes[1].set_xlabel('Dose (pM)')
        new_fit.axes[1].set_ylabel('pSTAT (MFI)')
        new_fit.fig = Figure_2

        alpha_mask = [7.5, 10.0]
        beta_mask = [7.5, 10.0]
        # Add fits
        for idx, t in enumerate(times):
            if t not in alpha_mask:
                new_fit.add_trajectory(dra60, t, PLOT_KWARGS['line_type'],
                                       alpha_palette[idx], (0, 0),
                                       'Alpha', label=str(t)+' min',
                                       linewidth=2, alpha=PLOT_KWARGS['alpha'])
            if t not in beta_mask:
                new_fit.add_trajectory(drb60, t, PLOT_KWARGS['line_type'],
                                       beta_palette[idx], (0, 1),
                                       'Beta', label=str(t) + ' min',
                                       linewidth=2, alpha=PLOT_KWARGS['alpha'])

        plt.figure(Figure_2.number)
        dr_fig, dr_axes = new_fit.show_figure(show_flag=False)

        # Format Figure_2
        # Dose response aesthetics
        if LOW_IFN:
            for ax in Figure_2.axes[4:6]:
                ax.set_yscale('log')
        else:
            for ax in Figure_2.axes[4:6]:
                ax.set_ylim((0, 6000))
        Figure_2.axes[4].set_title(r'IFN$\alpha$2')
        Figure_2.axes[5].set_title(r'IFN$\beta$')
        for direction in ['top', 'right']:
            Figure_2.axes[4].spines[direction].set_visible(False)
            Figure_2.axes[5].spines[direction].set_visible(False)

        # max pSTAT aesthetics
        for ax in Figure_2.axes[2:4]:
            ax.set_ylim([1500, 5500])

        # EC50 aesthetics
        for ax in Figure_2.axes[0:2]:
            ax.set_ylim([100, 4000])
        if LOW_IFN:
            Figure_2.savefig(os.path.join(os.getcwd(), 'results', 'tight_packing_low_dose.pdf'))
        else:
            Figure_2.savefig(os.path.join(os.getcwd(), 'results', 'tight_packing.pdf'))
