from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import seaborn as sns
from ifnclass.ifnfit import StepwiseFit

if __name__ == '__main__':
    raw_data = IfnData("20190121_pSTAT1_IFN_Bcell")
    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')

    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    times = [2.5, 5, 7.5, 10, 20, 60]
    alpha_doses_20190108 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses_20190108 = [0, 0.2, 6, 20, 60, 200, 600, 2000]

    Mixed_Model.set_parameters({'R2': 5700, 'R1': 1800,
                                'k_a1': 2.49e-15, 'k_a2': 1.328e-12, 'k_d3': 2.4e-06, 'k_d4': 0.228,
                                'kSOCSon': 5e-08, 'kpu': 0.0024,
                                'ka1': 1.65e-15, 'ka2': 1.22e-12, 'kd4': 0.86,
                                'kd3': 1.74e-05,
                                'kint_a': 0.00124, 'kint_b': 0.00086,
                                'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05})
    # -----------------------------
    # Stepwise fit
    # -----------------------------
    """ Early time fitting:
                            {'kpu': (0.0005, 0.0020),
                             'ka1': (3.3e-14 * 0.1, 3.3e-14 * 2),
                             'ka2': (5e-13 * 0.1, 5e-13 * 2),
                             'k_a1': (4.98E-14 * 0.1, 4.98E-14 * 2),
                             'k_a2': (8.30e-13 * 1, 8.30e-13 * 4)}, n=6

    print(best_parameters) # OrderedDict([('kpu', 0.0014), ('ka2', 2.39e-13), ('ka1', 3.3e-15), ('k_a1', 9.96e-14), ('k_a2', 3.32e-12)])                             
    """

    """ Late time fitting:
                            {'kSOCSon': (5e-8, 8e-7),
                             'kint_a': (0.0001, 0.002), 'kint_b': (0.002, 0.0001),
                             'krec_a1': (1e-03, 1e-02), 'krec_a2': (0.1, 0.01),
                             'krec_b1': (0.005, 0.0005), 'krec_b2': (0.05, 0.005)}, n=6
    OrderedDict([('kint_a', 0.0001), ('kSOCSon', 1.9999999999999999e-07), ('kint_b', 0.00047999999999999996), ('krec_a1', 0.001), ('krec_a2', 0.01), ('krec_b1', 0.0050000000000000001), ('krec_b2', 0.050000000000000003)])
    sf = 0.230475751632
    """

    """ Refinement:
                            {'R1': (1800, 5700), 'R2': (1800, 5700),
                             'kd4': (0.2, 1), 'k_d4': (0.06, 0.9),
                             'kint_a': (0.00005, 0.0002),
                             'krec_a1': (1e-04, 2e-03), 'krec_a2': (0.001, 0.02),
                             'krec_b1': (0.001, 0.1), 'krec_b2': (0.005, 0.1)}, n=6
OrderedDict([('k_d4', 0.22800000000000001), ('R1', 1800.0), ('kd4', 0.84000000000000008), ('kint_a', 0.00020000000000000001), ('R2', 5700.0), ('krec_a1', 0.0001), ('krec_a2', 0.02), ('krec_b1', 0.001), ('krec_b2', 0.0050000000000000001), ('kd3', 0.00016800000000000002), ('k_d3', 0.00045600000000000003)])
    0.260432986902                             
    """
    #stepfit = StepwiseFit(Mixed_Model, raw_data,
    #                      {'kpu': (0.001, 0.008),
    #                       'ka1': (3.3e-14 * 0.05, 3.3e-14 * 2),
    #                       'ka2': (5e-13 * 0.1, 5e-13 * 4),
    #                       'k_a1': (4.98E-14 * 0.05, 4.98E-14 * 2),
    #                       'k_a2': (8.30e-13 * 1, 8.30e-13 * 4)}, n=6)
    #best_parameters, scale_factor = stepfit.fit()
    #print(best_parameters)
    #print(scale_factor)
    #Mixed_Model = stepfit.model

    # -----------------------------
    # End Stepwise Fit
    # -----------------------------

    scale_factor = 0.478136435145
    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])

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
    alpha_mask = [7.5]
    beta_mask = [7.5]
    # Add fits
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='Alpha ' + str(t))
        if t not in beta_mask:
            new_fit.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label='Beta ' + str(t))
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
    alpha_mask = []
    beta_mask = []

    # Add fits
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



