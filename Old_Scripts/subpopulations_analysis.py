from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import numpy as np
import seaborn as sns
import pandas as pd
from ifnclass.ifnfit import DualMixedPopulation
from scipy.optimize import minimize
from collections import OrderedDict
import os
import json


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

    # -------------------------
    # Set up model scoring f'n
    # -------------------------
    # Parameters arbitrarily taken from best fit to GAB 20190119 data
    base_parameters = {'k_a1': 4.98E-14 * 2, 'k_a2': 1.328e-12, 'k_d3': 2.4e-06, 'k_d4': 0.228,
                       'kSOCSon': 8e-07, 'kpu': 0.0011,
                       'ka1': 3.3e-15, 'ka2': 1.22e-12, 'kd4': 0.86,
                       'kd3': 1.74e-05,
                       'kint_a': 0.000124, 'kint_b': 0.00086,
                       'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05}

    mixed_parameters = [{'R2': 570, 'R1': 180}, {'R2': 5700, 'R1': 8000} ]

    het_model = DualMixedPopulation('Mixed_IFN_ppCompatible', 0.8, 0.2)
    het_model.model_1.set_parameters(base_parameters)
    het_model.model_1.set_parameters(mixed_parameters[0])
    het_model.model_2.set_parameters(base_parameters)
    het_model.model_2.set_parameters(mixed_parameters[1])

    # ----------------------
    # Perform stepwise fit
    # ----------------------
    parameters_to_fit = {'R1__1': (200, 12000), 'R1__2': (200, 12000),
                         'R2__1': (200, 12000), 'R2__2': (200, 12000),
                         'kSOCSon': (1E-07, 1E-06),
                         'kint_a': (0.0001, 0.002), 'kint_b': (0.002, 0.0001),
                         'krec_a1': (1e-03, 1e-02), 'krec_a2': (0.1, 0.01),
                         'krec_b1': (0.005, 0.0005), 'krec_b2': (0.05, 0.005)}
    mixed_parameters = ['R1', 'R2']
    best_shared_params, best_mixed_params, best_sf = het_model.stepwise_fit(mean_data, parameters_to_fit, 10,
                                                                            mixed_parameters)

    # -------------------------
    # Simulate the best fit
    # -------------------------
    times = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
    alpha_doses = [10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses = [0.2, 6, 20, 60, 200, 600, 2000]
    best_fit_model = het_model
    best_fit_model.model_1.set_parameters(best_shared_params)
    best_fit_model.model_2.set_parameters(best_shared_params)
    best_fit_model.model_1.set_parameters(best_mixed_params[0])
    best_fit_model.model_2.set_parameters(best_mixed_params[1])

    alpha_response = best_fit_model.mixed_dose_response(times, 'TotalpSTAT', 'Ia',
                                                        np.logspace(1, 5), parameters={'Ib': 0}, sf=best_sf)
    beta_response = best_fit_model.mixed_dose_response(times, 'TotalpSTAT', 'Ib',
                                                       np.logspace(np.log10(0.2), np.log10(2000)),
                                                       parameters={'Ia': 0}, sf=best_sf)
    total_response_df = pd.concat([alpha_response, beta_response])
    total_response = IfnData('custom', df=total_response_df)

    # -------------------------
    # Plot the results
    # -------------------------
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    new_fit = DoseresponsePlot((1, 2))
    alpha_mask = [0.0, 7.5]
    beta_mask = [0.0, 7.5]
    # Add fits
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(total_response, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='Alpha ' + str(t))
        if t not in beta_mask:
            new_fit.add_trajectory(total_response, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label='Beta ' + str(t))


    # Add data
    times = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(mean_data, t, 'errorbar', 'o--', (0, 0), 'Alpha', label='', color=alpha_palette[idx])
        if t not in beta_mask:
            new_fit.add_trajectory(mean_data, t, 'errorbar', 'o--', (0, 1), 'Beta', label='', color=beta_palette[idx])

    new_fit.axes[0].set_title(r"IFN$\alpha$")
    new_fit.axes[1].set_title(r"IFN$\beta$")
    new_fit.show_figure(save_flag=False)
    print(str(best_shared_params))
    print(str(best_mixed_params[0]))
    print(str(best_mixed_params[1]))
    print(str(best_sf))
    with open(os.path.join('results','GAB_NewData','stepwise_params.txt'),'w') as f:
        f.write(json.dumps(best_shared_params))
        f.write(json.dumps(best_mixed_params[0]))
        f.write(json.dumps(best_mixed_params[1]))
        f.write(json.dumps(best_sf))

    """
    # ----------------------
    # Time Course Comparison
    # Simulate time courses
    # SUBPOPULATION 1
    Mixed_Model.set_parameters({'R2': 1000, 'R1': 1000})
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
    alpha_mask = [0]
    beta_mask = [0]

    # Add fits
    for j, dose in enumerate(alpha_doses):
        if dose not in alpha_mask:
            new_fit.add_trajectory(alpha_IfnData_objects[j], 'plot', alpha_palette[j], (0, 0),
                                   label='Alpha ' + str(dose))
    for j, dose in enumerate(beta_doses):
        if dose not in beta_mask:
            new_fit.add_trajectory(beta_IfnData_objects[j], 'plot', beta_palette[j], (0, 1),
                                   label='Beta ' + str(dose))

"""
