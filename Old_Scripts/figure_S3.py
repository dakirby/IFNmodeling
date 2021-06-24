from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from ifnclass.ifnfit import DualMixedPopulation
import numpy as np
from numpy import linspace, logspace, log10
import seaborn as sns
import pandas as pd
import os


if __name__ == '__main__':
    Sagar_data = IfnData("MacParland_Extended")
    Sagar_data.data_set.drop(labels=[4000, 8000], level=1, inplace=True)
    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')

    # ---------------------------
    # Now try to improve the fit:
    # Best low-dose fit from fit-12-12-2018
    # ---------------------------
    Mixed_Model.set_parameters({'kpu': 0.001, 'kpa': 1e-7,
                                'R2': 1742, 'R1': 1785,
                                'k_d4': 0.06, 'kd4': 0.3,
                                'k_a2': 8.3e-13 * 0.25, 'k_a1': 4.98e-14 * 0.01,
                                'ka2': 4.98173364330787e-13 * 2, 'ka1': 3.321155762205247e-14 * 1,
                                'ka4': 0.001,
                                'kSOCS': 0.01, 'kSOCSon': 2e-3, 'SOCSdeg': 0.2,
                                'kint_b': 0.0, 'kint_a': 0.04,
                                'krec_a1': 3e-05, 'krec_a2': 0.05,
                                'kdeg_a': 8E-5})

    scale_factor = 18 * 5
    # Additional fitting
    """
    stepfit25 = StepwiseFit(Mixed_Model, smooth25IfnData,
                            {'pS': (0, 200)}, n=8)
    best_parameters, scale_factor = stepfit25.fit()
    print(best_parameters)
    print(scale_factor)
    Mixed_Model = stepfit25.model
    scale_factor *= 0.25*0.8
    print(Mixed_Model.parameters)
    """
    # Additional fitting
    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])
    times = [5, 15, 30, 60]
    dradf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(1, log10(600))),
                                     parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(1, log10(11000))),
                                     parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')

    for i in range(len(times)):
        dradf.loc['Alpha'].iloc[:, i] = dradf.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf.loc['Beta'].iloc[:, i] = drbdf.loc['Beta'].iloc[:, i].apply(scale_data)

    #dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    #drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    """
    base_parameters = {'kpu': 0.0004, 'kpa': 1e-6,
                                'R2': 1742, 'R1': 1785,
                                'k_d4': 0.06, 'kd4': 0.3,
                                'k_a2': 8.3e-13 * 0.25, 'k_a1': 4.98e-14 * 0.01,
                                'ka2': 4.98173364330787e-13 * 2, 'ka1': 3.321155762205247e-14 * 1,
                                'ka4': 0.001,
                                'kSOCS': 0.01, 'kSOCSon': 2e-3, 'SOCSdeg': 0.2,
                                'kint_b': 0.0, 'kint_a': 0.04,
                                'krec_a1': 3e-05, 'krec_a2': 0.05,
                                'kdeg_a': 8E-5}
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
    
    best_shared_params, best_mixed_params, best_sf = het_model.stepwise_fit(Sagar_data, parameters_to_fit, 10,
                                                                            mixed_parameters)
    print(best_shared_params, best_mixed_params, best_sf)
    """
    """
    best_shared_params = base_parameters#{'kSOCSon': 1e-07, 'kint_a': 0.0001, 'kint_b': 0.002, 'krec_a1': 0.001, 'krec_a2': 0.1, 'krec_b1': 0.005, 'krec_b2': 0.05}
    best_mixed_params = mixed_parameters#[{'R1': 200.0, 'R2': 200.0}, {'R1': 200.0, 'R2': 200.0}]
    best_sf = 1.4 #10

    # Predictions
    best_fit_model = het_model
    best_fit_model.model_1.set_parameters(best_shared_params)
    best_fit_model.model_2.set_parameters(best_shared_params)
    best_fit_model.model_1.set_parameters(best_mixed_params[0])
    best_fit_model.model_2.set_parameters(best_mixed_params[1])

    times = [5, 15, 30, 60]

    alpha_response = best_fit_model.mixed_dose_response(times, 'TotalpSTAT', 'Ia',
                                                        np.logspace(1, 3), parameters={'Ib': 0}, sf=best_sf)
    beta_response = best_fit_model.mixed_dose_response(times, 'TotalpSTAT', 'Ib',
                                                       np.logspace(np.log10(0.2), 4),
                                                       parameters={'Ia': 0}, sf=best_sf)"""
    #total_response_df = pd.concat([alpha_response, beta_response])
    total_response_df = pd.concat([dradf, drbdf])
    total_response = IfnData('custom', df=total_response_df)

    # Plot
    new_fit = DoseresponsePlot((1, 2))
    alpha_palette = sns.color_palette("deep", 6)
    beta_palette = sns.color_palette("deep", 6)
    alpha_mask = []
    beta_mask = []
    # Add fits
    for idx, t in enumerate([el for el in times]):
        if t not in alpha_mask:
            new_fit.add_trajectory(total_response, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='Alpha '+str(t))
        if t not in beta_mask:
            new_fit.add_trajectory(total_response, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label='Beta '+str(t))
    # Add data
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(Sagar_data, t, 'errorbar', 'o', (0, 0), 'Alpha', color=alpha_palette[idx])
        if t not in beta_mask:
            new_fit.add_trajectory(Sagar_data, t, 'errorbar', 'o', (0, 1), 'Beta', color=beta_palette[idx])


    new_fit.show_figure(save_flag=True, save_dir=os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_S3', 'Figure_S3.pdf'))
    exit()
    # ----------------------------------
    # Time course plot
    # ----------------------------------
   # Simulate time courses
    alpha_time_courses = []
    for d in [10, 90, 600, 4000, 8000]: #[10, 90, 600, 4000, 8000]:
        alpha_time_courses.append(Mixed_Model.timecourse(list(linspace(0, 60, 30)), 'TotalpSTAT',
                                                         {'Ia': d * 6.022E23 * 1E-5 * 1E-12, 'Ib': 0},
                                                         return_type='dataframe', dataframe_labels=['Alpha', d]))
    beta_time_courses = []
    for d in [10, 90, 600, 2000, 11000]:
        beta_time_courses.append(Mixed_Model.timecourse(list(linspace(0, 60, 30)), 'TotalpSTAT',
                                                        {'Ib': d * 6.022E23 * 1E-5 * 1E-12, 'Ia': 0},
                                                        return_type='dataframe', dataframe_labels=['Beta', d]))
    # Scale simulations
    for i in range(30):
        for j in range(5):
            alpha_time_courses[j].loc['Alpha'].iloc[:, i] = alpha_time_courses[j].loc['Alpha'].iloc[:, i].apply(scale_data)
        for j in range(5):
            beta_time_courses[j].loc['Beta'].iloc[:, i] = beta_time_courses[j].loc['Beta'].iloc[:, i].apply(scale_data)
    # Turn into IfnData objects
    alpha_IfnData_objects = []
    beta_IfnData_objects = []
    for j in range(5):
        alpha_IfnData_objects.append(IfnData('custom', df=alpha_time_courses[j], conditions={'Alpha': {'Ib': 0}}))
    for j in range(5):
        beta_IfnData_objects.append(IfnData('custom', df=beta_time_courses[j], conditions={'Beta': {'Ia': 0}}))
    # Generate plot
    new_fit = TimecoursePlot((1, 2))
    alpha_mask = [4000, 8000]
    beta_mask = []

    # Add fits
    for j, dose in enumerate([10, 90, 600, 4000, 8000]):
        if dose not in alpha_mask:
            new_fit.add_trajectory(alpha_IfnData_objects[j], 'plot', alpha_palette[j], (0, 0),
                                   label='Alpha ' + str(dose))
    for j, dose in enumerate([10, 90, 600, 2000, 11000]):
        if dose not in beta_mask:
            new_fit.add_trajectory(beta_IfnData_objects[j], 'plot', beta_palette[j], (0, 1),
                                   label='Beta ' + str(dose))
    # Add data
    for idx, d in enumerate([10, 90, 600, 4000, 8000]):
        # Optional mask:
        if d not in alpha_mask:
            atc = IfnData('custom', df=Sagar_data.data_set.loc['Alpha', d, :])
            new_fit.add_trajectory(atc, 'scatter', 'o', (0, 0), label='Alpha ' + str(d), color=alpha_palette[idx])
            new_fit.add_trajectory(atc, 'errorbar', alpha_palette[idx], (0, 0), color=alpha_palette[idx])
    for idx, d in enumerate([10, 90, 600, 2000, 11000]):
        if d not in beta_mask:
            btc = IfnData('custom', df=Sagar_data.data_set.loc['Beta', d, :])
            new_fit.add_trajectory(btc, 'scatter', 'o', (0, 1), label='Beta ' + str(d), color=beta_palette[idx])
            new_fit.add_trajectory(btc, 'errorbar', beta_palette[idx], (0, 1), color=beta_palette[idx])

    new_fit.show_figure()



