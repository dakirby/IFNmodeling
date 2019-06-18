from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnfit import DualMixedPopulation
from numpy import linspace, logspace, transpose
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from ifnclass.ifnplot import Trajectory, DoseresponsePlot
import matplotlib.gridspec as gridspec


if __name__ == '__main__':
    # ----------------------
    # Set up Figure 2 layout
    # ----------------------
    Figure_2 = plt.figure(tight_layout=True)
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
    # Parameters found by stepwise fitting GAB mean data
    initial_parameters = {'k_a1': 4.98E-14 * 2, 'k_a2': 8.30e-13 * 2, 'k_d4': 0.006 * 3.8,
                       'kpu': 0.00095,
                       'ka2': 4.98e-13 * 2.45, 'kd4': 0.3 * 2.867,
                       'kint_a': 0.000124, 'kint_b': 0.00086,
                       'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05}
    dual_parameters = {'kint_a': 0.00052, 'kSOCSon': 6e-07, 'kint_b': 0.00052, 'krec_a1': 0.001, 'krec_a2': 0.1,
                       'krec_b1': 0.005, 'krec_b2': 0.05}
    scale_factor = 1.227

    Mixed_Model = DualMixedPopulation('Mixed_IFN_ppCompatible', 0.8, 0.2)
    Mixed_Model.model_1.set_parameters(initial_parameters)
    Mixed_Model.model_1.set_parameters(dual_parameters)
    Mixed_Model.model_1.set_parameters({'R1': 12000.0, 'R2': 1511.1})
    Mixed_Model.model_2.set_parameters(initial_parameters)
    Mixed_Model.model_2.set_parameters(dual_parameters)
    Mixed_Model.model_2.set_parameters({'R1': 6755.56, 'R2': 1511.1})

    # ---------------------------------
    # Make theory dose response curves
    # ---------------------------------
    # Make predictions
    times = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
    alpha_doses_20190108 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses_20190108 = [0, 0.2, 6, 20, 60, 200, 600, 2000]

    dradf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia', list(logspace(1, 5.2)),
                                            parameters={'Ib': 0}, sf=scale_factor)
    drbdf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ib', list(logspace(-1, 4)),
                                            parameters={'Ia': 0}, sf=scale_factor)

    dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    # ----------------------------------
    # Get all data set EC50 time courses
    # ----------------------------------
    newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
    newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
    newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
    newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")

    # 20190108
    ec50_20190108 = newdata_1.get_ec50s()

    # 20190119
    ec50_20190119 = newdata_2.get_ec50s()

    # 20190121
    ec50_20190121 = newdata_3.get_ec50s()

    # 20190214
    ec50_20190214 = newdata_4.get_ec50s()

    # Aligned data, to get scale factors for each data set
    alignment = DataAlignment()
    alignment.add_data([newdata_4, newdata_3, newdata_2, newdata_1])
    alignment.align()
    alignment.get_scaled_data()
    mean_data = alignment.summarize_data()

    # ----------------------------------
    # Make model predictions for EC50
    # ----------------------------------
    time_list = list(linspace(2.5, 60, num=15))

    dfa = Mixed_Model.mixed_dose_response(time_list, 'TotalpSTAT', 'Ia', list(logspace(-3, 5)),
                                          parameters={'Ib': 0}, sf=scale_factor)
    dfa = IfnData(name='custom', df=dfa, conditions={'Ib': 0})
    alpha_ec_aggregate = [el[1] for el in dfa.get_ec50s()['Alpha']]
    alpha_peak_aggregate = [el[1] for el in dfa.get_max_responses()['Alpha']]

    dfb = Mixed_Model.mixed_dose_response(time_list, 'TotalpSTAT', 'Ib', list(logspace(-3, 5)),
                                          parameters={'Ia': 0}, sf=scale_factor)
    dfb = IfnData(name='custom', df=dfb, conditions={'Ia': 0})
    beta_ec_aggregate = [el[1] for el in dfb.get_ec50s()['Beta']]
    beta_peak_aggregate = [el[1] for el in dfb.get_max_responses()['Beta']]

    # -----------------------
    # Plot EC50 vs time
    # -----------------------
    ec50_axes_list = [Figure_2.add_subplot(gs[1, 0]), Figure_2.add_subplot(gs[1, 1]),
                      Figure_2.add_subplot(gs[1, 2]), Figure_2.add_subplot(gs[1, 3])]

    ec50_axes = ec50_axes_list[0:2]
    ec50_axes[0].set_xlabel("Time (min)")
    ec50_axes[1].set_xlabel("Time (min)")
    ec50_axes[0].set_title(r"EC50 vs Time for IFN$\alpha$")
    ec50_axes[1].set_title(r"EC50 vs Time for IFN$\beta$")
    ec50_axes[0].set_ylabel("EC50 (pM)")
    ec50_axes[0].set_yscale('log')
    ec50_axes[1].set_yscale('log')
    # Add models
    ec50_axes[0].plot(time_list, alpha_ec_aggregate, label=r'IFN$\alpha$', color=alpha_palette[5], linewidth=2)
    ec50_axes[1].plot(time_list, beta_ec_aggregate, label=r'IFN$\beta$', color=beta_palette[5], linewidth=2)
    # Add data
    for colour_idx, ec50 in enumerate([ec50_20190108, ec50_20190119, ec50_20190121, ec50_20190214]):
        ec50_axes[0].scatter([el[0] for el in ec50['Alpha']], [el[1] for el in ec50['Alpha']],
                             label=dataset_names[colour_idx],
                             color=data_palette[colour_idx], marker=marker_shape[colour_idx])
        ec50_axes[1].scatter([el[0] for el in ec50['Beta']], [el[1] for el in ec50['Beta']],
                             color=data_palette[colour_idx], marker=marker_shape[colour_idx])

    # -----------------
    # Data max response
    # -----------------
    # 20190108
    max_20190108 = newdata_1.get_max_responses()

    # 20190119
    max_20190119 = newdata_2.get_max_responses()

    # 20190121
    max_20190121 = newdata_3.get_max_responses()

    # 20190214
    max_20190214 = newdata_4.get_max_responses()

    # -------------------
    # Plot max response
    # -------------------
    # fig, axes = plt.subplots(nrows=1, ncols=2)
    ec50_axes = ec50_axes_list[2:4]
    ec50_axes[0].set_xlabel("Time (min)")
    ec50_axes[1].set_xlabel("Time (min)")
    ec50_axes[0].set_title(r"Max pSTAT vs Time for IFN$\alpha$")
    ec50_axes[1].set_title(r"Max pSTAT vs Time for IFN$\beta$")
    ec50_axes[0].set_ylabel("Max pSTAT (MFI)")

    # Add models
    ec50_axes[0].plot(time_list, alpha_peak_aggregate, color=alpha_palette[5], linewidth=2)
    ec50_axes[1].plot(time_list, beta_peak_aggregate, color=beta_palette[5], linewidth=2)
    # Add data
    for colour_idx, maxpSTAT in enumerate([max_20190108, max_20190119, max_20190121, max_20190214]):
        scale_factor = alignment.scale_factors[3 - colour_idx]
        scaled_response = [el[1] * scale_factor for el in maxpSTAT['Alpha']]
        ec50_axes[0].scatter([el[0] for el in maxpSTAT['Alpha']], scaled_response,
                             color=data_palette[colour_idx], marker=marker_shape[colour_idx])
        scaled_response = [el[1] * scale_factor for el in maxpSTAT['Beta']]
        ec50_axes[1].scatter([el[0] for el in maxpSTAT['Beta']], scaled_response,
                             color=data_palette[colour_idx], marker=marker_shape[colour_idx])

    # -------------------------------
    # Plot model dose response curves
    # -------------------------------
    alpha_palette = sns.color_palette("deep", 6)
    beta_palette = sns.color_palette("deep", 6)

    new_fit = DoseresponsePlot((1, 2))
    new_fit.axes = [Figure_2.add_subplot(gs[0, 0:2]), Figure_2.add_subplot(gs[0, 2:4])]
    new_fit.axes[0].set_xscale('log')
    new_fit.axes[0].set_xlabel('Dose (pM)')
    new_fit.axes[0].set_ylabel('pSTAT (MFI)')
    new_fit.axes[1].set_xscale('log')
    new_fit.axes[1].set_xlabel('Dose (pM)')
    new_fit.axes[1].set_ylabel('pSTAT (MFI)')
    new_fit.fig = Figure_2

    alpha_mask = [7.5]
    beta_mask = [7.5]
    # Add fits
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label=str(t)+' min',
                                   linewidth=2)
            new_fit.add_trajectory(mean_data, t, 'errorbar', 'o', (0, 0), 'Alpha', color=alpha_palette[idx])
        if t not in beta_mask:
            new_fit.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label=str(t) +' min',
                                   linewidth=2)
            new_fit.add_trajectory(mean_data, t, 'errorbar', 'o', (0, 1), 'Beta', color=beta_palette[idx])

    plt.figure(Figure_2.number)
    dr_fig, dr_axes = new_fit.show_figure()

    # Format Figure_2
    # Dose response aesthetics
    for ax in Figure_2.axes[4:6]:
        ax.set_ylim((0, 5000))
    Figure_2.axes[4].set_title(r'IFN$\alpha$')
    Figure_2.axes[5].set_title(r'IFN$\beta$')

    # max pSTAT aesthetics
    for ax in Figure_2.axes[2:4]:
        ax.set_ylim([1500, 5500])

    # EC50 aesthetics
    for ax in Figure_2.axes[0:2]:
        ax.set_ylim([1, 5000])

    Figure_2.set_size_inches(14.75, 8)
    Figure_2.tight_layout()
    Figure_2.savefig(os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_2', 'Figure_2.pdf'))
