from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import seaborn as sns


if __name__ == '__main__':
    raw_data = IfnData("20190108_pSTAT1_IFN")
    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')

    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    times = [2.5, 5, 7.5, 10, 20, 60]
    alpha_doses_20190108 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses_20190108 = [0, 0.2, 6, 20, 60, 200, 600, 2000]

    Mixed_Model.set_parameters({'R2': 2300 * 2.5, 'R1': 1800 * 1.8,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 4, 'k_d3': 2.4e-06, 'k_d4': 0.06,
         'kSOCSon': 0.9e-8,'kpu': 0.0028,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3, 'kd4': 1.0, 'kd3': 0.001,
         'kint_a': 0.0007, 'kint_b': 0.0014,
         'krec_a1': 9e-03, 'krec_a2': 0.05, 'krec_b1': 0.001, 'krec_b2': 0.01})
    scale_factor = 0.25
    scale_data = lambda q: (scale_factor*q[0], scale_factor*q[1])

    # Make predictions
    dradf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia',
                                     list(logspace(log10(alpha_doses_20190108[1]), log10(alpha_doses_20190108[-1]))),
                                     parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib',
                                     list(logspace(log10(beta_doses_20190108[1]), log10(beta_doses_20190108[-1]))),
                                     parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')
    for i in range(len(times)):
        dradf.loc['Alpha'].iloc[:, i] = dradf.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf.loc['Beta'].iloc[:, i] = drbdf.loc['Beta'].iloc[:, i].apply(scale_data)

    dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    new_fit = DoseresponsePlot((1, 2))
    alpha_mask = []
    beta_mask = []
    # Add fits
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='Alpha '+str(t))
        if t not in beta_mask:
            new_fit.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label='Beta '+str(t))
    # Add data
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(raw_data, t, 'plot', '--', (0, 0), 'Alpha', label='Alpha '+str(t), color=alpha_palette[idx])
            new_fit.add_trajectory(raw_data, t, 'scatter', 'ro', (0, 0), 'Alpha', label='', color=alpha_palette[idx])
        if t not in beta_mask:
            new_fit.add_trajectory(raw_data, t, 'plot', '--', (0, 1), 'Beta', label='Beta '+str(t), color=beta_palette[idx])
            new_fit.add_trajectory(raw_data, t, 'scatter', 'go', (0, 1), 'Beta', label='', color=beta_palette[idx])

    new_fit.show_figure(save_flag=False)

    # ----------------------------------
    # Time course plot
    # ----------------------------------
    alpha_palette = sns.color_palette("Reds", 8)
    beta_palette = sns.color_palette("Greens", 8)

   # Simulate time courses
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
            alpha_time_courses[j].loc['Alpha'].iloc[:, i] = alpha_time_courses[j].loc['Alpha'].iloc[:, i].apply(scale_data)
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
    alpha_mask = []
    beta_mask = []

    # Add fits
    print(len(alpha_doses_20190108))
    print(len(alpha_IfnData_objects))
    for j, dose in enumerate(alpha_doses_20190108):
        if dose not in alpha_mask:
            new_fit.add_trajectory(alpha_IfnData_objects[j], 'plot', alpha_palette[j], (0, 0),
                                   label='Alpha ' + str(dose))
    for j, dose in enumerate(beta_doses_20190108):
        if dose not in beta_mask:
            new_fit.add_trajectory(beta_IfnData_objects[j], 'plot', beta_palette[j], (0, 1),
                                   label='Beta ' + str(dose))
    # Add data
    for idx, d in enumerate(alpha_doses_20190108):
        # Optional mask:
        if d not in alpha_mask:
            atc = IfnData('custom', df=raw_data.data_set.loc['Alpha', d, :])
            new_fit.add_trajectory(atc, 'scatter', 'o', (0, 0), label='Alpha ' + str(d), color=alpha_palette[idx])
            new_fit.add_trajectory(atc, 'errorbar', alpha_palette[idx], (0, 0), color=alpha_palette[idx])
    for idx, d in enumerate(beta_doses_20190108):
        if d not in beta_mask:
            btc = IfnData('custom', df=raw_data.data_set.loc['Beta', d, :])
            new_fit.add_trajectory(btc, 'scatter', 'o', (0, 1), label='Beta ' + str(d), color=beta_palette[idx])
            new_fit.add_trajectory(btc, 'errorbar', beta_palette[idx], (0, 1), color=beta_palette[idx])

    new_fit.show_figure()



