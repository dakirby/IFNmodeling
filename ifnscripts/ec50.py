from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import DoseresponsePlot
from numpy import linspace, logspace, transpose
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def MM(xdata, top, k):
    ydata = [top * x / (k + x) for x in xdata]
    return ydata


def fit_MM(doses, responses, guesses):
    top = guesses[0]
    K = guesses[1]
    results, covariance = curve_fit(MM, doses, responses, p0=[top, K])
    top = results[0]
    K = results[1]
    if K > 1E4:
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
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

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

    # Make model predictions
    time_list = list(linspace(2.5, 60, num=15))
    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
    Mixed_Model.set_parameters({'R2': 4920, 'R1': 1200,
                                'k_a1': 2.0e-13, 'k_a2': 1.328e-12, 'k_d3': 1.13e-4, 'k_d4': 0.9,
                                'kSOCSon': 5e-08, 'kpu': 0.0022, 'kpa': 2.36e-06,
                                'ka1': 3.3e-15, 'ka2': 1.85e-12, 'kd4': 2.0,
                                'kd3': 6.52e-05,
                                'kint_a':  0.00048, 'kint_b': 0.00086,
                                'krec_a1': 0.01, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05})
    scale_factor = 1.46182313424
    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])

    alpha_peak_aggregate_a, alpha_ec_aggregate_a = get_ec50(Mixed_Model, time_list, 'Ia', 'TotalpSTAT', custom_parameters={'Ib': 0}, rflag=True)
    beta_peak_aggregate_b, beta_ec_aggregate_b = get_ec50(Mixed_Model, time_list, 'Ib', 'TotalpSTAT', custom_parameters={'Ia': 0}, rflag=True)

    # Plot EC50 vs time
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].set_xlabel("Time (s)")
    axes[1].set_xlabel("Time (s)")
    axes[0].set_title(r"EC50 vs Time for IFN$\alpha$")
    axes[1].set_title(r"EC50 vs Time for IFN$\beta$")
    axes[0].set_ylabel("EC50 (pM)")
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    # Add models
    axes[0].plot(time_list, alpha_ec_aggregate_a, label=r'IFN$\alpha$ EC50', color=alpha_palette[1])
    axes[1].plot(time_list, beta_ec_aggregate_b, label=r'IFN$\beta$ EC50', color=beta_palette[1])
    # Add data
    for colour_idx, ec50 in enumerate([ec50_20190108, ec50_20190119, ec50_20190121]):
        axes[0].scatter([el[0] for el in ec50['Alpha']], [el[1] for el in ec50['Alpha']], label='data', color=alpha_palette[colour_idx+1])
        axes[1].scatter([el[0] for el in ec50['Beta']], [el[1] for el in ec50['Beta']], label='data', color=beta_palette[colour_idx+1])
    fig.show()
    fig.savefig('results\ec50_vs_time.pdf')

    exit()
    # Plot every MM fit to data, to verify EC50 values
    dr_plot = DoseresponsePlot((6, 2))
    for idx, t in enumerate(newdata_4.get_times('Alpha')):
        dr_plot.add_trajectory(dra60, t, 'plot', alpha_model_palette[0], (idx, 0), 'Alpha',
                               label='Alpha model ' + str(t))
    for idx, t in enumerate(newdata_4.get_times('Beta')):
        dr_plot.add_trajectory(drb60, t, 'plot', beta_model_palette[0], (idx, 1), 'Beta', label='Beta model ' + str(t))

    dr_fig, dr_axes = dr_plot.show_figure(save_flag=False)
    dr_fig.set_size_inches(10.5, 7.1 * 6)