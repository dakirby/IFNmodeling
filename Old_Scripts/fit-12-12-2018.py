from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import seaborn as sns


if __name__ == '__main__':
    Sagar_data = IfnData("MacParland_Extended")
    Sagar_data.data_set.drop(labels=[4000, 8000], level=1, inplace=True)
    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
    '''
    fit_parameters = OrderedDict(
        [('kd4', 1.0), ('krec_a1', 3.0000000000000001e-05), ('krec_a2', 0.050000000000000003), ('krec_b2', 0.01),
         ('krec_b1', 0.001), ('k_d4', 0.00059999999999999995), ('kSOCSon', 1e-08), ('kd3', 0.001),
         ('k_d3', 2.3999999999999999e-06)])
    Mixed_Model.set_parameters(fit_parameters)
    Mixed_Model.save_model('fitting_2_5.p')
    '''
    #Mixed_Model.load_model('fitting_2_5.p')

    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    # ---------------------------
    # Now try to improve the fit:
    # ---------------------------
    """
    Results of the 5 minute stepwise fit
    OrderedDict([('R2', 500.0), ('kpu', 0.0001), ('R1', 821.42857142857144), ('k_d4', 0.10000000000000001), ('kd4', 0.21642857142857141), ('kd3', 0.00021642857142857143), ('k_d3', 0.00040000000000000002)])
    10.6992020307
    # Previous attempt                 
    Mixed_Model.set_parameters({'kpa': 2e-6, 'kpu': 0.0008,
                                'R2': 4464, 'R1': 4785,
                                'k_d4': 0.001, 'kd4': 0.092,
                                'k_a1': 4.98e-14 * 0.02, 'k_a2': 8.3e-13 * 0.02,
                                'ka1': 3.32e-14 * 0.1, 'ka2': 4.98e-13 * 0.1,
                                'kSOCSon': 2e-6, 'kSOCS': 0.001,
                                'kint_b': 0.0004, 'krec_b2': 0.001, 'krec_b1': 0.001,
                                'kint_a': 0.00444, 'krec_a1': 3e-05, 'krec_a2': 5e-4})                 
    """
    """ 
    # Best of 11-12-2018-11pm
    Mixed_Model.set_parameters({'kpu': 0.00025, 'kpa': 5e-7,
                                'R2': 1742, 'R1': 1785,
                                'k_d4': 0.06, 'kd4': 0.803, 'kd3': 0.008,
                                'k_a2': 8.3e-13 * 0.1,
                                'ka2': 4.98173364330787e-13 * 0.05, 'ka1': 3.321155762205247e-14 * 0.1,
                                'ka4': 0.2,
                                'kSOCS': 0.005, 'kSOCSon': 1e-3, 'SOCSdeg': 0,
                                'kint_b': 0.0009, 'kint_a': 0.001})
    """
    Mixed_Model.set_parameters({'kpu': 0.0004, 'kpa': 1e-6,
                                'R2': 1742, 'R1': 1785,
                                'k_d4': 0.06, 'kd4': 0.3,
                                'k_a2': 8.3e-13 * 0.25, 'k_a1': 4.98e-14 * 0.01,
                                'ka2': 4.98173364330787e-13 * 2, 'ka1': 3.321155762205247e-14 * 1,
                                'ka4': 0.001,
                                'kSOCS': 0.01, 'kSOCSon': 2e-3, 'SOCSdeg': 0.2,
                                'kint_b': 0.0, 'kint_a': 0.04,
                                'krec_a1': 3e-05, 'krec_a2': 0.05,
                                'kdeg_a': 8E-5})

    scale_factor = 18
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
    scale_data = lambda q: (scale_factor*q[0], scale_factor*q[1])
    times = [5, 15, 30, 60]
    dradf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(1, log10(600))),
                                           parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(1, log10(11000))),
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
    for idx, t in enumerate([el for el in times]):
        if t not in alpha_mask:
            new_fit.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='Alpha '+str(t))
        if t not in beta_mask:
            new_fit.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label='Beta '+str(t))
    # Add data
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(Sagar_data, t, 'errorbar', alpha_palette[idx], (0, 0), 'Alpha', dn=1)
            new_fit.add_trajectory(Sagar_data, t, 'scatter', 'ro', (0, 0), 'Alpha', dn=1, color=alpha_palette[idx], label='Alpha ' + str(t))
        if t not in beta_mask:
            new_fit.add_trajectory(Sagar_data, t, 'errorbar', beta_palette[idx], (0, 1), 'Beta', dn=1)
            new_fit.add_trajectory(Sagar_data, t, 'scatter', 'go', (0, 1), 'Beta', dn=1, color=beta_palette[idx], label='Beta ' + str(t))


    new_fit.show_figure(save_flag=False)
    print(Mixed_Model.parameters)

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



