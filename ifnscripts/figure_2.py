from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from numpy import linspace, logspace, transpose
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from ifnclass.ifnplot import Trajectory, TimecoursePlot


def MM(xdata, top, n, k):
    ydata = [top * x ** n / (k ** n + x ** n) for x in xdata]
    return ydata


def fit_MM(doses, responses, guesses):
    top = guesses[0]
    n = guesses[1]
    K = guesses[2]
    results, covariance = curve_fit(MM, doses, responses, p0=[top, n, K])
    top = results[0]
    n = results[1]
    K = results[2]
    if n > 3:
        n = 3
    return top, n, K


def get_ec50(model: IfnModel, times: list or int, dose_species: str, response_species: str, custom_parameters={},
             rflag=False):
    if type(times) == int or type(times) == float:
        dr_curve = [el[0] for el in model.doseresponse([times], response_species, dose_species, list(logspace(-3, 5)),
                                      parameters=custom_parameters, return_type='list')[response_species]]

        top, n, K = fit_MM(list(logspace(-3, 5)), dr_curve, [max(dr_curve), 1, 1000])
        if rflag:
            return top, n, K
        else:
            return K

    elif type(times) == list:
        dr_curve = model.doseresponse(times, response_species, dose_species, list(logspace(-3, 5)),
                                      parameters=custom_parameters, return_type='list')[response_species]

        top_list = []
        n_list = []
        K_list = []
        for t in range(len(times)):
            tslice = [el[t] for el in dr_curve]
            top, n, K = fit_MM(list(logspace(-3, 5)), tslice, [max(tslice), 1, 1000])
            top_list.append(top)
            n_list.append(n)
            K_list.append(K)
        if rflag:
            return top_list, n_list, K_list
        else:
            return K_list

    else:
        raise TypeError("Could not identify type for variable times")


if __name__ == '__main__':
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    # Get all data set EC50 time courses
    # 20190119
    data_20190119 = IfnData("20190119_pSTAT1_IFN_Bcell")
    ec50_20190119 = data_20190119.get_ec50s()
    ec50_20190119['Alpha'][-1] = (ec50_20190119['Alpha'][-1][0], 800.)

    # 20190121
    data_20190121 = IfnData("20190121_pSTAT1_IFN_Bcell")
    ec50_20190121 = data_20190121.get_ec50s()
    #ec50_20190121['Alpha'][-1] = (ec50_20190121['Alpha'][-1][0], 800.)

    # 20190108
    data_20190108 = IfnData("20190108_pSTAT1_IFN")
    ec50_20190108 = data_20190108.get_ec50s()
    ec50_20190108['Beta'][-1] = (ec50_20190108['Beta'][-1][0], 20.)


    # Make model predictions
    time_list = list(linspace(2.5, 60, num=15))
    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
    base_parameters = {'kd4': 1.0, 'krec_a1': 3.0000000000000001e-05, 'krec_a2': 0.050000000000000003, 'krec_b2': 0.01,
         'krec_b1': 0.001, 'k_d4': 0.00059999999999999995, 'kSOCSon': 1e-08, 'kd3': 0.001,
         'k_d3': 2.3999999999999999e-06}
    Mixed_Model.set_parameters(base_parameters)

    # 20190108
    Mixed_Model.set_parameters({'R2': 5700, 'R1': 1800,
                                'k_a1': 4.98E-14 * 2, 'k_a2': 8.30e-13 * 4, 'k_d3': 2.4e-06, 'k_d4': 0.228,
                                'kSOCSon': 2e-7, 'kpu': 0.0014,
                                'ka1': 3.321155762205247e-14 * 0.1, 'ka2': 4.98173364330787e-13 * 0.5, 'kd4': 0.84,
                                'kd3': 0.001,
                                'kint_a': 0.0002, 'kint_b': 0.00048,
                                'krec_a1': 1e-04, 'krec_a2': 0.02, 'krec_b1': 0.001, 'krec_b2': 0.005})

    scale_factor = 0.260432986902
    alpha_peak20190108, alpha_n20190108, alpha_ec5020190108 = get_ec50(Mixed_Model, time_list, 'Ia', 'TotalpSTAT', custom_parameters={'Ib': 0}, rflag=True)
    beta_peak20190108, beta_n20190108, beta_ec5020190108 = get_ec50(Mixed_Model, time_list, 'Ib', 'TotalpSTAT', custom_parameters={'Ia': 0}, rflag=True)

    # 20190119
    Mixed_Model.reset_parameters()
    Mixed_Model.set_parameters(base_parameters)
    Mixed_Model.set_parameters({'R2': 5700, 'R1': 1800,
                                'k_a1': 4.98E-14 * 2, 'k_a2': 1.328e-12, 'k_d3': 2.4e-06, 'k_d4': 0.228,
                                'kSOCSon': 5e-08, 'kpu': 0.0011,
                                'ka1': 3.3e-15, 'ka2': 1.22e-12, 'kd4': 0.86,
                                'kd3': 1.74e-05,
                                'kint_a': 0.000124, 'kint_b': 0.00086,
                                'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05})

    scale_factor = 0.242052437849
    alpha_peak20190119, alpha_n20190119, alpha_ec5020190119 = get_ec50(Mixed_Model, time_list, 'Ia', 'TotalpSTAT', custom_parameters={'Ib': 0}, rflag=True)
    beta_peak20190119, beta_n20190119, beta_ec5020190119 = get_ec50(Mixed_Model, time_list, 'Ib', 'TotalpSTAT', custom_parameters={'Ia': 0}, rflag=True)

    # 20190121
    Mixed_Model.reset_parameters()
    Mixed_Model.set_parameters(base_parameters)
    Mixed_Model.set_parameters({'R2': 4140, 'R1': 4920,
                                'k_a1': 2.49e-15, 'k_a2': 1.328e-12, 'k_d3': 7.5e-06, 'k_d4': 0.06,
                                'kSOCSon': 5e-08, 'kpu': 0.0024, 'kpa': 2.08e-06,
                                'ka1': 5.3e-15, 'ka2': 1.22e-12, 'kd4': 0.86,
                                'kd3': 5.47e-05,
                                'kint_a':  0.0002, 'kint_b': 0.00086,
                                'krec_a1': 0.0001, 'krec_a2': 0.02, 'krec_b1': 0.001, 'krec_b2': 0.005})

    scale_factor = 0.2050499
    alpha_peak20190121, alpha_n20190121, alpha_ec5020190121 = get_ec50(Mixed_Model, time_list, 'Ia', 'TotalpSTAT', custom_parameters={'Ib': 0}, rflag=True)
    beta_peak20190121, beta_n20190121, beta_ec5020190121 = get_ec50(Mixed_Model, time_list, 'Ib', 'TotalpSTAT', custom_parameters={'Ia': 0}, rflag=True)

    # Plot EC50 vs Time
    ec50_vs_time_fig_obj, ec50_vs_time_axes_obj = plt.subplots(nrows=1, ncols=2)
    ec50_vs_time_axes_obj[0].set_xlabel("Time (s)")
    ec50_vs_time_axes_obj[1].set_xlabel("Time (s)")
    ec50_vs_time_axes_obj[0].set_title(r"EC50 vs Time for IFN$\alpha$")
    ec50_vs_time_axes_obj[1].set_title(r"EC50 vs Time for IFN$\beta$")
    ec50_vs_time_axes_obj[0].set_ylabel("EC50")
    # ec50_vs_time_axes_obj[0].set_yscale('log')
    # ec50_vs_time_axes_obj[1].set_yscale('log')
    # Add models
    for colour_idx, alpha_ec50 in enumerate([alpha_ec5020190108, alpha_ec5020190119, alpha_ec5020190121]):
        ec50_vs_time_axes_obj[0].plot(time_list, alpha_ec50, label=r'IFN$\alpha$ EC50', color=alpha_palette[colour_idx + 1])
    for colour_idx, beta_ec50 in enumerate([beta_ec5020190108, beta_ec5020190119, beta_ec5020190121]):
        ec50_vs_time_axes_obj[1].plot(time_list, beta_ec50, label=r'IFN$\beta$ EC50', color=beta_palette[colour_idx + 1])
    # Add data
    for colour_idx, ec50 in enumerate([ec50_20190108, ec50_20190119, ec50_20190121]):
        ec50_vs_time_axes_obj[0].scatter([el[0] for el in ec50['Alpha']], [el[1] for el in ec50['Alpha']], label='data', color=alpha_palette[colour_idx + 1])
        ec50_vs_time_axes_obj[1].scatter([el[0] for el in ec50['Beta']], [el[1] for el in ec50['Beta']], label='data', color=beta_palette[colour_idx + 1])
    ec50_vs_time_fig_obj.show()
    ec50_vs_time_fig_obj.savefig(os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_2', 'ec50_vs_time.pdf'))

    # ----------------------------
    # Make theory time course plot
    # ----------------------------
    Mixed_Model.reset_parameters()

    Mixed_Model.set_parameters({'R2': 4140, 'R1': 4920,
                                'k_a1': 2.49e-15, 'k_a2': 1.328e-12, 'k_d3': 7.5e-06, 'k_d4': 0.06,
                                'kSOCSon': 5e-08, 'kpu': 0.0024, 'kpa': 2.08e-06,
                                'ka1': 5.3e-15, 'ka2': 1.22e-12, 'kd4': 0.86,
                                'kd3': 5.47e-05,
                                'kint_a':  0.0002, 'kint_b': 0.00086,
                                'krec_a1': 0.0001, 'krec_a2': 0.02, 'krec_b1': 0.001, 'krec_b2': 0.005})
    scale_factor = 0.2050499
    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])

    alpha_doses_20190108 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses_20190108 = [0, 0.2, 6, 20, 60, 200, 600, 2000]

    # Make predictions
    alpha_palette = sns.color_palette("Reds", 8)
    beta_palette = sns.color_palette("Greens", 8)

    alpha_time_courses = []
    for d in alpha_doses_20190108:
        alpha_time_courses.append(Mixed_Model.timecourse(list(linspace(0, 60, 30)), 'TotalpSTAT',
                                                         {'Ia': d * 6.022E23 * 1E-5 * 1E-12, 'Ib': 0},
                                                         return_type='dataframe', dataframe_labels=['Alpha', d]))
    beta_time_courses = []
    for d in beta_doses_20190108:
        beta_time_courses.append(Mixed_Model.timecourse(list(linspace(0, 60, 30)), 'TotalpSTAT',
                                                        {'Ib': d * 6.022E23 * 1E-5 * 1E-12, 'Ia': 0},
                                                        return_type='dataframe', dataframe_labels=['Beta', d]))
    # Scale simulations
    for i in range(30):
        for j in range(len(alpha_doses_20190108)):
            alpha_time_courses[j].loc['Alpha'].iloc[:, i] = alpha_time_courses[j].loc['Alpha'].iloc[:, i].apply(
                scale_data)
        for j in range(len(beta_doses_20190108)):
            beta_time_courses[j].loc['Beta'].iloc[:, i] = beta_time_courses[j].loc['Beta'].iloc[:, i].apply(scale_data)
    # Turn into IfnData objects
    alpha_IfnData_objects = []
    beta_IfnData_objects = []
    for j in range(len(alpha_doses_20190108)):
        alpha_IfnData_objects.append(IfnData('custom', df=alpha_time_courses[j], conditions={'Alpha': {'Ib': 0}}))
    for j in range(len(beta_doses_20190108)):
        beta_IfnData_objects.append(IfnData('custom', df=beta_time_courses[j], conditions={'Beta': {'Ia': 0}}))
    # Generate plot
    new_fit = TimecoursePlot((1, 2))
    alpha_mask = [0, 300, 3000]
    beta_mask = [0, 60, 600]

    # Add fits
    for j, dose in enumerate(alpha_doses_20190108):
        if dose not in alpha_mask:
            new_fit.add_trajectory(alpha_IfnData_objects[j], 'plot', alpha_palette[j], (0, 0),
                                   label='Alpha ' + str(dose))
    for j, dose in enumerate(beta_doses_20190108):
        if dose not in beta_mask:
            new_fit.add_trajectory(beta_IfnData_objects[j], 'plot', beta_palette[j], (0, 1),
                                   label='Beta ' + str(dose))
    tc_fig_obj, tc_axes_obj = new_fit.save_figure(save_dir=os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_2'))

    # Combine plots into Figure 2

