from ifndata import IfnData
from ifnmodel import IfnModel
from ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from ifnfit import StepwiseFit
from numpy import linspace, logspace, log10, nan
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from smooth_B6_IFN import * # Imports smoothed data to fit to
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

    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    dra60df = Mixed_Model.doseresponse([2.5, 10, 20, 60], 'TotalpSTAT', 'Ia', list(logspace(-3, 5)),
                                           parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drb60df = Mixed_Model.doseresponse([2.5, 10, 20, 60], 'TotalpSTAT', 'Ib', list(logspace(-3, 4)),
                                           parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')

    scale_factor = 0.03094064
    scale_data = lambda q: (scale_factor*q[0], scale_factor*q[1])
    for i in range(4):
        dra60df.loc['Alpha'].iloc[:, i] = dra60df.loc['Alpha'].iloc[:, i].apply(scale_data)
        drb60df.loc['Beta'].iloc[:, i] = drb60df.loc['Beta'].iloc[:, i].apply(scale_data)

    dra60 = IfnData('custom', df=dra60df, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drb60df, conditions={'Beta': {'Ia': 0}})

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

    Mixed_Model.set_parameters({'kpu': 0.0028, 'R2': 2300 * 2.5, 'R1': 1800 * 1.8, 'k_d4': 0.06, 'kint_b': 0.0003, 'krec_b1': 0.001,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 4, 'kSOCSon': 0.9e-8,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3,
         'kint_a': 0.0014, 'krec_a1': 9e-03})
    scale_factor = 0.036 #0.02894064
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
    times = [2.5, 5, 7.5, 10, 20, 60]
    dradf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-1, 5)),
                                           parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 4)),
                                           parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')

    for i in range(len(times)):
        dradf.loc['Alpha'].iloc[:, i] = dradf.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf.loc['Beta'].iloc[:, i] = drbdf.loc['Beta'].iloc[:, i].apply(scale_data)

    dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    new_fit = DoseresponsePlot((1, 2))
    alpha_mask = [2.5, 7.5]
    beta_mask = [2.5, 7.5]
    # Add fits
    for idx, t in enumerate([str(el) for el in times]):
        if t not in [str(el) for el in alpha_mask]:
            new_fit.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='Alpha '+t)
        if t not in [str(el) for el in beta_mask]:
            new_fit.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label='Beta '+t)
    # Add data
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(newdata, t, 'scatter', alpha_palette[idx], (0, 0), 'Alpha', dn=1)
            new_fit.add_trajectory(newdata, t, 'plot', '--', (0, 0), 'Alpha', dn=1, color=alpha_palette[idx], label='Alpha '+str(t))
        if t not in beta_mask:
            new_fit.add_trajectory(newdata, t, 'scatter', beta_palette[idx], (0, 1), 'Beta', dn=1)
            new_fit.add_trajectory(newdata, t, 'plot', '--', (0, 1), 'Beta', dn=1, color=beta_palette[idx], label='Beta '+str(t))
    # Add Michaelis-Menten curves
    #new_fit.add_trajectory(a25smoothIfnData, 2.5, 'plot', 'r:', (0, 0), 'Alpha', dn=1)
    ##new_fit.add_trajectory(a10smoothIfnData, 10, 'plot', 'r:', (0, 0), 'Alpha', dn=1)
    ##new_fit.add_trajectory(a20smoothIfnData, 20, 'plot', 'r:', (0, 0), 'Alpha', dn=1)
    #new_fit.add_trajectory(a60smoothIfnData, 60, 'plot', 'r:', (0, 0), 'Alpha', dn=1)

    #new_fit.add_trajectory(b25smoothIfnData, 2.5, 'plot', 'g:', (0, 1), 'Beta', dn=1)
    ##new_fit.add_trajectory(b10smoothIfnData, 10, 'plot', 'g:', (0, 1), 'Beta', dn=1)
    ##new_fit.add_trajectory(b20smoothIfnData, 20, 'plot', 'g:', (0, 1), 'Beta', dn=1)
    #new_fit.add_trajectory(b60smoothIfnData, 60, 'plot', 'g:', (0, 1), 'Beta', dn=1)


    new_fit.show_figure(save_flag=False)
    print(Mixed_Model.parameters)



