from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from numpy import linspace, logspace, transpose
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from ifnclass.ifnplot import Trajectory, TimecoursePlot
import matplotlib.gridspec as gridspec


def MM(xdata, top, k):
    ydata = [top * x / (k + x) for x in xdata]
    return ydata


def fit_MM(doses, responses, guesses):
    top = guesses[0]
    K = guesses[1]
    results, covariance = curve_fit(MM, doses, responses, p0=[top, K])
    top = results[0]
    K = results[1]
    if K > 4E3:
        top = max(responses) * 0.5
        for i, r in enumerate(responses):
            if r > top:
                K = 10 ** ((np.log10(doses[i-1]) + np.log10(doses[i])) / 2.0)
                break
    return top, K


def get_ec50(model: IfnModel, times: list or int, dose_species: str, response_species: str, custom_parameters={},
             rflag=False):
    if type(times) == int or type(times) == float:
        dr_curve = [el[0] for el in model.doseresponse([times], response_species, dose_species, list(logspace(-3, 5)),
                                      parameters=custom_parameters, return_type='list')[response_species]]

        top, K = fit_MM(list(logspace(-3, 5)), dr_curve, [max(dr_curve), 1000])
        if rflag:
            return top, K
        else:
            return K

    elif type(times) == list:
        dr_curve = model.doseresponse(times, response_species, dose_species, list(logspace(-3, 5)),
                                      parameters=custom_parameters, return_type='list')[response_species]

        top_list = []
        K_list = []
        for t in range(len(times)):
            tslice = [el[t] for el in dr_curve]
            top, K = fit_MM(list(logspace(-3, 5)), tslice, [max(tslice), 1000])
            top_list.append(top)
            K_list.append(K)
        if rflag:
            return top_list, K_list
        else:
            return K_list

    else:
        raise TypeError("Could not identify type for variable times")

if __name__ == '__main__':
    # ----------------------
    # Set up Figure 2 layout
    # ----------------------
    Figure_2 = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(nrows=2, ncols=4, height_ratios=[3, 2])

    Figure_2.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
    Figure_2.show()

    # --------------------
    # Set up EC50 figures
    # --------------------
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)
    data_palette = sns.color_palette("muted", 6)
    marker_shape = ["o", "v", "s", "P", "d", "1", "x", "*"]
    dataset_names = ["20190108", "20190119", "20190121", "20190214"]

    # Get all data set EC50 time courses
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

    # Make model predictions
    time_list = list(linspace(2.5, 60, num=15))
    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
    # Optimal parameters for fitting mean GAB data
    Mixed_Model.set_parameters({'R2': 4920, 'R1': 1200,
                                'k_a1': 2.0e-13, 'k_a2': 1.328e-12, 'k_d3': 1.13e-4, 'k_d4': 0.9,
                                'kSOCSon': 5e-08, 'kpu': 0.0022, 'kpa': 2.36e-06,
                                'ka1': 3.3e-15, 'ka2': 1.85e-12, 'kd4': 2.0,
                                'kd3': 6.52e-05,
                                'kint_a': 0.0015, 'kint_b': 0.002,
                                'krec_a1': 0.01, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05})
    scale_factor = 1.46182313424
    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])

    alpha_peak_aggregate, alpha_ec_aggregate = get_ec50(Mixed_Model, time_list, 'Ia', 'TotalpSTAT',
                                                        custom_parameters={'Ib': 0}, rflag=True)
    beta_peak_aggregate, beta_ec_aggregate = get_ec50(Mixed_Model, time_list, 'Ib', 'TotalpSTAT',
                                                      custom_parameters={'Ia': 0}, rflag=True)
    alpha_peak_aggregate = np.multiply(alpha_peak_aggregate, scale_factor)
    beta_peak_aggregate = np.multiply(beta_peak_aggregate, scale_factor)

    # Plot EC50 vs time
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

    # -------------#
    # Max response
    # -------------#
    # 20190108
    max_20190108 = newdata_1.get_max_responses()

    # 20190119
    max_20190119 = newdata_2.get_max_responses()

    # 20190121
    max_20190121 = newdata_3.get_max_responses()

    # 20190214
    max_20190214 = newdata_4.get_max_responses()

    # Plot
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

    # ----------------------------
    # Make theory time course plot
    # ----------------------------
    alpha_doses = [10, 100, 1000, 10000]
    beta_doses = [10, 100, 1000, 10000]

    # Make predictions
    alpha_palette = sns.color_palette("Reds", 4)
    beta_palette = sns.color_palette("Greens", 4)

    alpha_time_courses = []
    for d in alpha_doses:
        alpha_time_courses.append(Mixed_Model.timecourse(list(linspace(0, 60, 30)), 'TotalpSTAT',
                                                         {'Ia': d * 6.022E23 * 1E-5 * 1E-12, 'Ib': 0},
                                                         return_type='dataframe', dataframe_labels=['Alpha', d]))
    beta_time_courses = []
    for d in beta_doses:
        beta_time_courses.append(Mixed_Model.timecourse(list(linspace(0, 60, 30)), 'TotalpSTAT',
                                                        {'Ib': d * 6.022E23 * 1E-5 * 1E-12, 'Ia': 0},
                                                        return_type='dataframe', dataframe_labels=['Beta', d]))
    # Scale simulations
    for i in range(30):
        for j in range(len(alpha_doses)):
            alpha_time_courses[j].loc['Alpha'].iloc[:, i] = alpha_time_courses[j].loc['Alpha'].iloc[:, i].apply(
                scale_data)
        for j in range(len(beta_doses)):
            beta_time_courses[j].loc['Beta'].iloc[:, i] = beta_time_courses[j].loc['Beta'].iloc[:, i].apply(scale_data)
    # Turn into IfnData objects
    alpha_IfnData_objects = []
    beta_IfnData_objects = []
    for j in range(len(alpha_doses)):
        alpha_IfnData_objects.append(IfnData('custom', df=alpha_time_courses[j], conditions={'Alpha': {'Ib': 0}}))
    for j in range(len(beta_doses)):
        beta_IfnData_objects.append(IfnData('custom', df=beta_time_courses[j], conditions={'Beta': {'Ia': 0}}))

    # Generate plot
    new_fit = TimecoursePlot((1, 2))
    new_fit.axes = [Figure_2.add_subplot(gs[0, 0:2]), Figure_2.add_subplot(gs[0, 2:4])]
    new_fit.fig = Figure_2

    # Add fits
    alpha_mask = []
    beta_mask = []
    for j, dose in enumerate(alpha_doses):
        if dose not in alpha_mask:
            new_fit.add_trajectory(alpha_IfnData_objects[j], 'plot', alpha_palette[j], (0, 0),
                                   label=str(dose)+' pM', linewidth=2)
    for j, dose in enumerate(beta_doses):
        if dose not in beta_mask:
            new_fit.add_trajectory(beta_IfnData_objects[j], 'plot', beta_palette[j], (0, 1),
                                   label=str(dose)+' pM', linewidth=2)
    #new_fit.match_axes = True
    plt.figure(Figure_2.number)
    tc_fig, tc_axes = new_fit.show_figure()

    # Format Figure_2
    # Time course aesthetics
    for ax in Figure_2.axes[4:6]:
        ax.set_ylim([0, 3500])
        ax.set_ylabel('pSTAT-1 (MFI)')
        ax.set_xlabel('Time (min)')

    # max pSTAT aesthetics
    for ax in Figure_2.axes[2:4]:
        ax.set_ylim([1500, 5500])

    # EC50 aesthetics
    #for ax in Figure_2.axes[0:2]:
    #    ax.set_ylim([10, 5000])

    Figure_2.set_size_inches(14.75, 8)
    Figure_2.tight_layout()
    Figure_2.savefig(os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_2', 'Figure_2.pdf'))
