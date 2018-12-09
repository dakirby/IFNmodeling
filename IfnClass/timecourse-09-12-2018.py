from ifndata import IfnData
from ifnmodel import IfnModel
from ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from ifnfit import StepwiseFit
from numpy import linspace, logspace, log10, nan
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from smooth_B6_IFN import *  # Imports smoothed data to fit to
import pickle
from collections import OrderedDict

if __name__ == '__main__':
    newdata = IfnData("20181113_B6_IFNs_Dose_Response_Bcells")
    Mixed_Model = IfnModel('')
    '''
    fit_parameters = OrderedDict(
        [('kd4', 1.0), ('krec_a1', 3.0000000000000001e-05), ('krec_a2', 0.050000000000000003), ('krec_b2', 0.01),
         ('krec_b1', 0.001), ('k_d4', 0.00059999999999999995), ('kSOCSon', 1e-08), ('kd3', 0.001),
         ('k_d3', 2.3999999999999999e-06)])
    Mixed_Model.set_parameters(fit_parameters)
    Mixed_Model.save_model('fitting_2_5.p')
    '''
    Mixed_Model.load_model('fitting_2_5.p')

    alpha_palette = sns.color_palette("Reds", 7)
    beta_palette = sns.color_palette("Greens", 7)

    dra60df = Mixed_Model.doseresponse([2.5, 10, 20, 60], 'TotalpSTAT', 'Ia', list(logspace(-3, 5)),
                                       parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drb60df = Mixed_Model.doseresponse([2.5, 10, 20, 60], 'TotalpSTAT', 'Ib', list(logspace(-3, 4)),
                                       parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')

    scale_factor = 0.03094064
    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])
    for i in range(4):
        dra60df.loc['Alpha'].iloc[:, i] = dra60df.loc['Alpha'].iloc[:, i].apply(scale_data)
        drb60df.loc['Beta'].iloc[:, i] = drb60df.loc['Beta'].iloc[:, i].apply(scale_data)

    tca5000IfnData = IfnData('custom', df=dra60df, conditions={'Alpha': {'Ib': 0}})
    tca100IfnData = IfnData('custom', df=drb60df, conditions={'Beta': {'Ia': 0}})

    """
    initial_fit = DoseresponsePlot((1, 2))
    for idx, t in enumerate(['2.5', '10', '20', '60']):
        initial_fit.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha')
        initial_fit.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1), 'Beta')
    for idx, t in enumerate([2.5, 10, 20, 60]):
        initial_fit.add_trajectory(newdata, t, 'scatter', alpha_palette[idx], (0, 0), 'Alpha', dn=1)
        initial_fit.add_trajectory(newdata, t, 'scatter', beta_palette[idx], (0, 1), 'Beta', dn=1)

    initial_fit.show_figure(save_flag=False)
    """

    # ---------------------------
    # Now try to improve the fit:
    # ---------------------------

    Mixed_Model.set_parameters({'kpu': 0.002, 'R2': 2300 * 2, 'R1': 1800 * 2, 'k_d4': 0.06, 'kint_b': 0.00002,
                                'k_a1': 4.98E-14 * 2, 'k_a2': 8.30e-13 * 2, 'kSOCSon': 1e-8,
                                'ka1': 3.321155762205247e-14 * 0.5, 'ka2': 4.98173364330787e-13 * 0.5})
    scale_factor = 0.02894064

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
    # Add fits
    for j, dose in enumerate([5, 50, 250, 500, 5000, 25000, 50000]):
        new_fit.add_trajectory(alpha_IfnData_objects[j], 'plot', alpha_palette[j], (0, 0), label='Alpha '+str(dose))
    for j, dose in enumerate([0.1, 1, 5, 10, 100, 200, 1000]):
        new_fit.add_trajectory(beta_IfnData_objects[j], 'plot', beta_palette[j], (0, 1), label='Beta '+str(dose))
    # Add data
    for idx, d in enumerate([5, 50, 250, 500, 5000, 25000, 50000]):
        atc = IfnData('custom', df=newdata.data_set.loc['Alpha', d, :])
        new_fit.add_trajectory(atc, 'scatter', alpha_palette[idx], (0, 0), label='Alpha '+str(d))
        new_fit.add_trajectory(atc, 'plot', '--', (0, 0), color=alpha_palette[idx])
    for idx, d in enumerate([0.1, 1, 5, 10, 100, 200, 1000]):
        btc = IfnData('custom', df=newdata.data_set.loc['Beta', d, :])
        new_fit.add_trajectory(btc, 'scatter', beta_palette[idx], (0, 1), label='Beta '+str(d))
        new_fit.add_trajectory(btc, 'plot', '--', (0, 1), color=beta_palette[idx])

    new_fit.show_figure()
    print(Mixed_Model.parameters)
