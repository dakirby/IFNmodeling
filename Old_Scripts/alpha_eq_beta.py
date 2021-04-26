from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnfit import DualMixedPopulation
from numpy import linspace, logspace
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from ifnclass.ifnplot import DoseresponsePlot


if __name__ == '__main__':
    # --------------------
    # Set up Model
    # --------------------
    # Parameters found by stepwise fitting GAB mean data
    # Note: can remove multiplicative factors on all K1, K2, K4 and still get
    # very good fit to data (worst is 5 min beta)
    initial_parameters = {'k_a1': 4.98E-14 * 2, 'k_a2': 8.30e-13 * 2,
                          'k_d4': 0.006 * 3.8,
                          'kpu': 0.00095,
                          'ka2': 4.98e-13 * 2.45, 'kd4': 0.3 * 2.867,
                          'kint_a': 0.000124, 'kint_b': 0.00086,
                          'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005,
                          'krec_b2': 0.05}
    dual_parameters = {'kint_a': 0.00052, 'kSOCSon': 6e-07, 'kint_b': 0.32,
                       'krec_a1': 0.001, 'krec_a2': 0.1,
                       'krec_b1': 0.00, 'krec_b2': 0.0}
    scale_factor = 1.227

    Mixed_Model = DualMixedPopulation('Mixed_IFN_ppCompatible', 0.8, 0.2)
    Mixed_Model.model_1.set_parameters(initial_parameters)
    Mixed_Model.model_1.set_parameters(dual_parameters)
    Mixed_Model.model_1.set_parameters({'R1': 12000.0, 'R2': 1511.1})
    Mixed_Model.model_2.set_parameters(initial_parameters)
    Mixed_Model.model_2.set_parameters(dual_parameters)
    Mixed_Model.model_2.set_parameters({'R1': 6755.56, 'R2': 1511.1})

    for m in [Mixed_Model.model_1, Mixed_Model.model_2]:
        m.set_parameters({'R1': m.parameters['R1'] / 100,
                          'R2': m.parameters['R2'] / 100,
                          'kd4': m.parameters['kd4'] * 15,
                          'k_d4': m.parameters['k_d4'] * 15,
                          'k_a1': 4.98E-14 * 0.5,
                          'k_a2': 8.30e-13 * 0.5,
                          'ka1': m.parameters['ka1'] * 2,
                          'ka2': 4.98e-13 * 2})


    # Make predictions
    times = [30.0]
    alpha_doses_20190108 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses_20190108 = [0, 0.2, 6, 20, 60, 200, 600, 2000]

    dradf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia',
                                            list(logspace(1, 5.2)),
                                            parameters={'Ib': 0},
                                            sf=scale_factor)
    drbdf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ib',
                                            list(logspace(0, 5.2)),
                                            parameters={'Ia': 0},
                                            sf=scale_factor)

    dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    # -------------------------------
    # Plot model dose response curves
    # -------------------------------
    alpha_palette = sns.color_palette("rocket_r", 6)
    beta_palette = sns.color_palette("rocket_r", 6)

    drPlot = DoseresponsePlot((1, 1))
    # drPlot.axes[0].set_title(r'IFN$\alpha$')

    t_mask = []
    # Add fits
    for idx, t in enumerate(times):
        if t not in t_mask:
            drPlot.add_trajectory(dra60, t, 'plot', 'r',
                                   (0, 0), 'Alpha', label= r'IFN$\alpha$ ' +
                                                           str(t)+' min',
                                   linewidth=2)
        if t not in t_mask:
            drPlot.add_trajectory(drb60, t, 'plot', 'g', (0, 1),
                                   'Beta', label=r'IFN$\beta$ ' +
                                                 str(t) + ' min',
                                   linewidth=2)

    # Dose response aesthetics
    # for ax in drPlot.axes:
    #     ax.set_xlim((1E-2, 1E4))

    drPlot.show_figure()
