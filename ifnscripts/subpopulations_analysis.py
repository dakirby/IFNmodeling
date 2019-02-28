from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import numpy as np
import seaborn as sns
import pandas as pd
from ifnclass.ifnfit import StepwiseFit

if __name__ == '__main__':
    raw_data = IfnData("20190119_pSTAT1_IFN_Bcell")
    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')

    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    times = [2.5, 5, 7.5, 10, 20, 60]
    alpha_doses_20190108 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses_20190108 = [0, 0.2, 6, 20, 60, 200, 600, 2000]

    # Parameters arbitrarily taken from best fit to GAB 20190119 data
    Mixed_Model.set_parameters({'R2': 5700, 'R1': 1800,
                                'k_a1': 4.98E-14 * 2, 'k_a2': 1.328e-12, 'k_d3': 2.4e-06, 'k_d4': 0.228,
                                'kSOCSon': 8e-07, 'kpu': 0.0011,
                                'ka1': 3.3e-15, 'ka2': 1.22e-12, 'kd4': 0.86,
                                'kd3': 1.74e-05,
                                'kint_a': 0.000124, 'kint_b': 0.00086,
                                'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05})

    scale_factor = 0.242052437849
    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])

    # Make predictions

    # -------------------------
    # Subpopulations
    # -------------------------
    # SUBPOPULATION 1
    Mixed_Model.set_parameters({'R2': 1000, 'R1': 1000})
    sim_doses_a = list(logspace(log10(alpha_doses_20190108[1]), log10(alpha_doses_20190108[-1])))
    sim_doses_b = list(logspace(log10(beta_doses_20190108[1]), log10(beta_doses_20190108[-1])))
    dradf_1 = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', sim_doses_a,
                                     parameters={'Ib': 0}, return_type='list')['TotalpSTAT']
    drbdf_1 = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', sim_doses_b,
                                     parameters={'Ia': 0}, return_type='list')['TotalpSTAT']
    tc_a = Mixed_Model.timecourse(np.linspace(0, 60), 'TotalpSTAT',
                                  parameters={'Ia': 1000, 'Ib': 0}, return_type='dataframe',
                                  dataframe_labels=['Alpha', 1000])
    tc_b = Mixed_Model.timecourse(np.linspace(0, 60), 'TotalpSTAT',
                                  parameters={'Ib': 1000, 'Ia': 0}, return_type='dataframe',
                                  dataframe_labels=['Beta', 1000])

    # SUBPOPULATION 2
    Mixed_Model.set_parameters({'R2': 8000, 'R1': 8000})
    dradf_2 = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', sim_doses_a,
                                     parameters={'Ib': 0}, return_type='list')['TotalpSTAT']
    drbdf_2 = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', sim_doses_b,
                                     parameters={'Ia': 0}, return_type='list')['TotalpSTAT']

    # Add subpopulations with appropriate population-fractional weights
    subpop_1_fraction = 0.80
    subpop_2_fraction = 0.20
    dradf = np.add(np.multiply(dradf_1, subpop_1_fraction), np.multiply(dradf_2, subpop_2_fraction))
    drbdf = np.add(np.multiply(drbdf_1, subpop_1_fraction), np.multiply(drbdf_2, subpop_2_fraction))


    # -------------------------
    # -------------------------

    # Put into dataframe object
    dradf = [['Alpha', sim_doses_a[row], *[(el, nan) for el in dradf[row]]] for row in range(0, len(dradf))]
    drbdf = [['Beta', sim_doses_b[row], *[(el, nan) for el in drbdf[row]]] for row in range(0, len(drbdf))]
    all_data = dradf + drbdf

    column_labels = ['Dose_Species', 'Dose (pM)'] + [str(el) for el in times]
    drdf = pd.DataFrame.from_records(all_data, columns=column_labels)
    drdf.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)

    # Scale dataframe
    for i in range(len(times)):
        drdf.loc['Alpha'].iloc[:, i] = drdf.loc['Alpha'].iloc[:, i].apply(scale_data)
        drdf.loc['Beta'].iloc[:, i] = drdf.loc['Beta'].iloc[:, i].apply(scale_data)

    # Put into IfnData objects
    dr60 = IfnData('custom', df=drdf, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})

    new_fit = DoseresponsePlot((2, 2))
    alpha_mask = [7.5]
    beta_mask = [7.5]
    # Add fits
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(dr60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='Alpha ' + str(t))
        if t not in beta_mask:
            new_fit.add_trajectory(dr60, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label='Beta ' + str(t))

    # --------------------------------------
    # Add original population for comparison
    # --------------------------------------
    dradf_1 = [['Alpha', sim_doses_a[row], *[(el, nan) for el in dradf_1[row]]] for row in range(0, len(dradf_1))]
    drbdf_1 = [['Beta', sim_doses_b[row], *[(el, nan) for el in drbdf_1[row]]] for row in range(0, len(drbdf_1))]
    all_data_1 = dradf_1 + drbdf_1
    drdf_1 = pd.DataFrame.from_records(all_data_1, columns=column_labels)
    drdf_1.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    for i in range(len(times)):
        drdf_1.loc['Alpha'].iloc[:, i] = drdf_1.loc['Alpha'].iloc[:, i].apply(scale_data)
        drdf_1.loc['Beta'].iloc[:, i] = drdf_1.loc['Beta'].iloc[:, i].apply(scale_data)
    dr60_1 = IfnData('custom', df=drdf_1, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(dr60_1, t, 'plot', alpha_palette[idx], (1, 0), 'Alpha', label='Alpha ' + str(t))
        if t not in beta_mask:
            new_fit.add_trajectory(dr60_1, t, 'plot', beta_palette[idx], (1, 1), 'Beta', label='Beta ' + str(t))

    """
    # Add data
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(raw_data, t, 'plot', '--', (0, 0), 'Alpha', label='Alpha ' + str(t),
                                   color=alpha_palette[idx])
            new_fit.add_trajectory(raw_data, t, 'scatter', 'ro', (0, 0), 'Alpha', label='', color=alpha_palette[idx])
        if t not in beta_mask:
            new_fit.add_trajectory(raw_data, t, 'plot', '--', (0, 1), 'Beta', label='Beta ' + str(t),
                                   color=beta_palette[idx])
            new_fit.add_trajectory(raw_data, t, 'scatter', 'go', (0, 1), 'Beta', label='', color=beta_palette[idx])
    """
    new_fit.axes[0][0].set_title(r"Heterogenous population")
    new_fit.axes[1][0].set_title(r"Homogenous population")
    new_fit.show_figure(save_flag=False)

    # ----------------------
    # Time Course Comparison
    # Simulate time courses
    # SUBPOPULATION 1
    Mixed_Model.set_parameters({'R2': 1000, 'R1': 1000})
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
    alpha_mask = [0]
    beta_mask = [0]

    # Add fits
    for j, dose in enumerate(alpha_doses_20190108):
        if dose not in alpha_mask:
            new_fit.add_trajectory(alpha_IfnData_objects[j], 'plot', alpha_palette[j], (0, 0),
                                   label='Alpha ' + str(dose))
    for j, dose in enumerate(beta_doses_20190108):
        if dose not in beta_mask:
            new_fit.add_trajectory(beta_IfnData_objects[j], 'plot', beta_palette[j], (0, 1),
                                   label='Beta ' + str(dose))

