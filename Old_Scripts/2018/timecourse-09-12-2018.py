from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from ifnclass.ifnfit import StepwiseFit
from numpy import linspace, logspace, log10, nan
import seaborn as sns
from smooth_B6_IFN import *  # Imports smoothed data to fit to

if __name__ == '__main__':
    newdata = IfnData("20181113_B6_IFNs_Dose_Response_Bcells")
    Mixed_Model = IfnModel('')
    Mixed_Model.load_model('fitting_2_5.p')

    Mixed_Model.set_parameters(
        {'kpu': 0.0028, 'R2': 2300 * 2.5, 'R1': 1800 * 1.8, 'k_d4': 0.06, 'kint_b': 0.0003, 'krec_b1': 0.001,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 4, 'kSOCSon': 0.9e-8,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3,
         'kint_a': 0.0014, 'krec_a1': 9e-03})
    scale_factor = 0.036
    alpha_palette = sns.color_palette("Reds", 9)
    beta_palette = sns.color_palette("Greens", 9)

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
    
    exp_doses_a = [5, 50, 250, 500, 5000, 25000, 50000]
    exp_doses_b = [0.1, 1, 5, 10, 100, 200, 1000]
    """
    # Additional fitting

    # Scale function
    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])
    # Simulate time courses
    alpha_time_courses = []
    for d in [5, 50, 250, 500, 5000, 25000, 50000]:
        alpha_time_courses.append(Mixed_Model.timecourse(list(linspace(0, 60, 25)), 'TotalpSTAT',
                                                         {'Ia': d * 6.022E23 * 1E-5 * 1E-12, 'Ib': 0},
                                                         return_type='dataframe', dataframe_labels=['Alpha', d]))
    beta_time_courses = []
    for d in [0.1, 1, 5, 10, 100, 200, 1000]:
        beta_time_courses.append(Mixed_Model.timecourse(list(linspace(0, 60, 25)), 'TotalpSTAT',
                                                        {'Ib': d * 6.022E23 * 1E-5 * 1E-12, 'Ia': 0},
                                                        return_type='dataframe', dataframe_labels=['Beta', d]))
    # Scale simulations
    for i in range(25):
        for j in range(7):
            alpha_time_courses[j].loc['Alpha'].iloc[:, i] = alpha_time_courses[j].loc['Alpha'].iloc[:, i].apply(scale_data)
            beta_time_courses[j].loc['Beta'].iloc[:, i] = beta_time_courses[j].loc['Beta'].iloc[:, i].apply(scale_data)
    # Turn into IfnData objects
    alpha_IfnData_objects = []
    beta_IfnData_objects = []
    for j in range(7):
        alpha_IfnData_objects.append(IfnData('custom', df=alpha_time_courses[j], conditions={'Alpha': {'Ib': 0}}))
        beta_IfnData_objects.append(IfnData('custom', df=beta_time_courses[j], conditions={'Beta': {'Ia': 0}}))
    # Generate plot
    new_fit = TimecoursePlot((1, 2))
    alpha_mask = [5, 50, 250]
    beta_mask = [0.1, 5, 200]
    # Add fits
    for j, dose in enumerate([5, 50, 250, 500, 5000, 25000, 50000]):
        if dose not in alpha_mask:
            new_fit.add_trajectory(alpha_IfnData_objects[j], 'plot', alpha_palette[j+2], (0, 0), label='Alpha '+str(dose))
    for j, dose in enumerate([0.1, 1, 5, 10, 100, 200, 1000]):
        if dose not in beta_mask:
            new_fit.add_trajectory(beta_IfnData_objects[j], 'plot', beta_palette[j+2], (0, 1), label='Beta '+str(dose))
    # Add data
    for idx, d in enumerate([5, 50, 250, 500, 5000, 25000, 50000]):
        # Optional mask:
        if d not in alpha_mask:
            atc = IfnData('custom', df=newdata.data_set.loc['Alpha', d, :])
            new_fit.add_trajectory(atc, 'scatter', alpha_palette[idx+2], (0, 0), label='Alpha '+str(d))
            new_fit.add_trajectory(atc, 'plot', '--', (0, 0), color=alpha_palette[idx+2])
    for j, dose in enumerate([0.1, 1, 5, 10, 100, 200, 1000]):
        if dose not in beta_mask:
            btc = IfnData('custom', df=newdata.data_set.loc['Beta', dose, :])
            new_fit.add_trajectory(btc, 'scatter', beta_palette[j+2], (0, 1), label='Beta '+str(dose))
            new_fit.add_trajectory(btc, 'plot', '--', (0, 1), color=beta_palette[j+2])

    new_fit.show_figure()
    print(Mixed_Model.parameters)
