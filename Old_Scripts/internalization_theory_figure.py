from ifnclass.ifnmodel import IfnModel
from ifnclass.ifndata import IfnData
from ifnclass.ifnplot import DoseresponsePlot, TimecoursePlot

from numpy import linspace, logspace
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')

    # This is the fitting_2_5 script best fit parameters, best at fitting Ratnadeep's B cell data
    Mixed_Model.set_parameters(
        {'R2': 2300 * 2.5,
         'R1': 1800 * 1.8, 'k_d4': 0.06, 'kint_b': 0.0003,
         'kpu': 0.0028,
         'krec_b1': 0.001, 'krec_b2': 0.01,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 4, 'kSOCSon': 0.9e-8,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3, 'kd4': 1.0, 'kd3': 0.001,
         'kint_a': 0.0014, 'krec_a1': 9e-03, 'krec_a2': 0.05})
    scale_factor = 0.036
    scale_data = lambda q: (scale_factor*q[0], scale_factor*q[1])

    dose_list = list(logspace(-2, 8, num=35))

    times = [2.5, 5, 15, 20, 30, 60]
    dradf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-1, 5)),
                                           parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 4)),
                                           parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')

    for i in range(len(times)):
        dradf.loc['Alpha'].iloc[:, i] = dradf.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf.loc['Beta'].iloc[:, i] = drbdf.loc['Beta'].iloc[:, i].apply(scale_data)

    dra = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    dose_response_plot = DoseresponsePlot((3, 2))
    alpha_mask = []
    beta_mask = []
    # Add fits
    for idx, t in enumerate([str(el) for el in times]):
        if t not in [str(el) for el in alpha_mask]:
            dose_response_plot.add_trajectory(dra, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='Alpha ' + t, linewidth=2.0)
        if t not in [str(el) for el in beta_mask]:
            dose_response_plot.add_trajectory(drb, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label='Beta ' + t, linewidth=2.0)

    # ----------------------------------------------
    # Now symmetrically double internalization rates
    # ----------------------------------------------
    Mixed_Model.set_parameters({'kint_a': 0.0014 * 2, 'kint_b': 0.0003 * 2})
    dradf1 = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-1, 5)),
                                           parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf1 = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 4)),
                                           parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')

    for i in range(len(times)):
        dradf1.loc['Alpha'].iloc[:, i] = dradf1.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf1.loc['Beta'].iloc[:, i] = drbdf1.loc['Beta'].iloc[:, i].apply(scale_data)

    dra1 = IfnData('custom', df=dradf1, conditions={'Alpha': {'Ib': 0}})
    drb1 = IfnData('custom', df=drbdf1, conditions={'Beta': {'Ia': 0}})

    # Add fits
    for idx, t in enumerate([str(el) for el in times]):
        if t not in [str(el) for el in alpha_mask]:
            dose_response_plot.add_trajectory(dra1, t, 'plot', alpha_palette[idx], (1, 0), 'Alpha', label='Alpha ' + t, linewidth=2.0)
        if t not in [str(el) for el in beta_mask]:
            dose_response_plot.add_trajectory(drb1, t, 'plot', beta_palette[idx], (1, 1), 'Beta', label='Beta ' + t, linewidth=2.0)

    # ----------------------------------------------
    # Now asymmetrically double recycling rates
    # ----------------------------------------------
    Mixed_Model.set_parameters({'krec_a2': 0.05 * 0.01, 'krec_b2': 0.01 * 0.01})
    dradf2 = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-1, 5)),
                                      parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf2 = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 4)),
                                      parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')

    for i in range(len(times)):
        dradf2.loc['Alpha'].iloc[:, i] = dradf2.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf2.loc['Beta'].iloc[:, i] = drbdf2.loc['Beta'].iloc[:, i].apply(scale_data)

    dra2 = IfnData('custom', df=dradf2, conditions={'Alpha': {'Ib': 0}})
    drb2 = IfnData('custom', df=drbdf2, conditions={'Beta': {'Ia': 0}})

    # Add fits
    for idx, t in enumerate([str(el) for el in times]):
        if t not in [str(el) for el in alpha_mask]:
            dose_response_plot.add_trajectory(dra2, t, 'plot', alpha_palette[idx], (2, 0), 'Alpha', label='Alpha ' + t,
                                              linewidth=2.0)
        if t not in [str(el) for el in beta_mask]:
            dose_response_plot.add_trajectory(drb2, t, 'plot', beta_palette[idx], (2, 1), 'Beta', label='Beta ' + t,
                                              linewidth=2.0)

    dose_response_plot.axes[0][0].set_title(r"IFN$\alpha$")
    dose_response_plot.axes[0][1].set_title(r"IFN$\beta$")

    dose_response_plot.axes[1][0].set_title(r"Double Internalization Rate")
    dose_response_plot.axes[1][1].set_title(r"Double Internalization Rate")

    dose_response_plot.axes[2][0].set_title(r"0.01 x R1 Recycling")
    dose_response_plot.axes[2][1].set_title(r"0.01 x R1 Recycling")

    dose_response_plot.show_figure(save_flag=False, save_dir='results')
