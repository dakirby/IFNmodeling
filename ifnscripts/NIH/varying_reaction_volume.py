from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnfit import DualMixedPopulation
from numpy import linspace, logspace, transpose
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from ifnclass.ifnplot import TimecoursePlot, DoseresponsePlot
import matplotlib.gridspec as gridspec


if __name__ == '__main__':
    volEC1 = 1/(25E9)
    volEC2 = 1E-5
    # --------------------
    # Set up Model
    # --------------------
    # Parameters found by stepwise fitting GAB mean data
    # Note: can remove multiplicative factors on all K1, K2, K4 and still get very good fit to data (worst is 5 min beta)
    initial_parameters = {'k_a1': 4.98E-14 * 2, 'k_a2': 8.30e-13 * 2, 'k_d4': 0.006 * 3.8,
                          'kpu': 0.00095,
                          'ka2': 4.98e-13 * 2.45, 'kd4': 0.3 * 2.867,
                          'kint_a': 0.000124, 'kint_b': 0.00086,
                          'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05}
    dual_parameters = {'kint_a': 0.00052, 'kSOCSon': 6e-07, 'kint_b': 0.00052, 'krec_a1': 0.001, 'krec_a2': 0.1,
                       'krec_b1': 0.005, 'krec_b2': 0.05}
    scale_factor = 1.227

    Mixed_Model = DualMixedPopulation('Mixed_IFN_ppCompatible', 0.8, 0.2)
    Mixed_Model.model_1.set_parameters(initial_parameters)
    Mixed_Model.model_1.set_parameters(dual_parameters)
    Mixed_Model.model_1.set_parameters({'R1': 12000.0, 'R2': 1511.1})
    Mixed_Model.model_2.set_parameters(initial_parameters)
    Mixed_Model.model_2.set_parameters(dual_parameters)
    Mixed_Model.model_2.set_parameters({'R1': 6755.56, 'R2': 1511.1})

    # ---------------------------------
    # Make theory dose response curves
    # ---------------------------------
    # Make predictions
    times = np.arange(0, 120, 0.5)# [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
    alpha_doses_20190108 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses_20190108 = [0, 0.2, 6, 20, 60, 200, 600, 2000]

    # 1 uL
    ka1 = Mixed_Model.model_1.parameters['ka1']
    ka2 = Mixed_Model.model_1.parameters['ka2']
    k_a1 = Mixed_Model.model_1.parameters['k_a1']
    k_a2 = Mixed_Model.model_1.parameters['k_a2']
    Mixed_Model.set_global_parameters({'volEC': volEC1, 'ka1': ka1*1E-5/volEC1, 'ka2': ka2*1E-5/volEC1,
                                       'k_a1': k_a1*1E-5/volEC1, 'k_a2': k_a2*1E-5/volEC1})
    dradf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia', list(logspace(1, 5.2)),
                                            parameters={'Ib': 0}, sf=scale_factor)
    drbdf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ib', list(logspace(-1, 4)),
                                            parameters={'Ia': 0}, sf=scale_factor)

    dra15uL = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb15uL = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    # 1 mL
    Mixed_Model.set_global_parameters({'volEC': volEC2, 'ka1': ka1*1E-5/volEC2, 'ka2': ka2*1E-5/volEC2,
                                       'k_a1': k_a1*1E-5/volEC2, 'k_a2': k_a2*1E-5/volEC2})
    dradf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia', list(logspace(1, 5.2)),
                                            parameters={'Ib': 0}, sf=scale_factor)
    drbdf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ib', list(logspace(-1, 4)),
                                            parameters={'Ia': 0}, sf=scale_factor)

    dra5mL = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb5mL = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    # -------------------------------
    # Plot model dose response curves
    # -------------------------------
    alpha_palette = sns.color_palette("deep", 6)
    beta_palette = sns.color_palette("deep", 6)

    new_fit = DoseresponsePlot((2, 2))

    alpha_mask = [2.5, 5.0, 10.0, 20.0, 60.0]
    beta_mask = [2.5, 5.0, 10.0, 20.0, 60.0]
    # Add fits
    color_counter = -1
    for idx, t in enumerate(times):
        if t in alpha_mask:
            color_counter += 1
            new_fit.add_trajectory(dra15uL, t, 'plot', alpha_palette[color_counter], (0, 0), 'Alpha', label=str(t) + ' min',
                                   linewidth=2)
            new_fit.add_trajectory(dra5mL, t, 'plot', alpha_palette[color_counter], (1, 0), 'Alpha', label=str(t) + ' min',
                                   linewidth=2)

        if t in beta_mask:
            new_fit.add_trajectory(drb15uL, t, 'plot', beta_palette[color_counter], (0, 1), 'Beta', label=str(t) + ' min',
                                   linewidth=2)
            new_fit.add_trajectory(drb5mL, t, 'plot', beta_palette[color_counter], (1, 1), 'Beta', label=str(t) + ' min',
                                   linewidth=2)

    dr_fig, dr_axes = new_fit.show_figure()
    dr_axes[0][0].set_title(r'IFN$\alpha$ at {} L'.format(volEC1))
    dr_axes[0][1].set_title(r'IFN$\beta$ at {} L'.format(volEC1))
    dr_axes[1][0].set_title(r'IFN$\alpha$ at {} L'.format(volEC2))
    dr_axes[1][1].set_title(r'IFN$\beta$ at {} L'.format(volEC2))
    dr_fig.tight_layout()
    dr_fig.savefig('varying_reaction_volume_dr.pdf')

    # -------------------------------
    # Plot model dose response curves
    # -------------------------------
    tc_figure = TimecoursePlot((1, 2))

    alpha_testdose = list(logspace(1, 5.2))[24]
    beta_testdose = list(logspace(-1, 4))[17]
    print("Using {} for IFNalpha".format(alpha_testdose))
    print("Using {} for IFNbeta".format(beta_testdose))

    tc_figure.add_trajectory(dra15uL, 'plot', '-', subplot_idx=(0, 0), label=r'IFN$\alpha$ at {} L'.format(volEC1),
                             doseslice=alpha_testdose, dose_species='Alpha', color=alpha_palette[1])
    tc_figure.add_trajectory(dra5mL, 'plot', '-', subplot_idx=(0, 0), label=r'IFN$\alpha$ at {} L'.format(volEC2),
                             doseslice=alpha_testdose, dose_species='Alpha', color=alpha_palette[4])

    tc_figure.add_trajectory(drb15uL, 'plot', '-', subplot_idx=(0, 1), label=r'IFN$\beta$ at {} L'.format(volEC1),
                             doseslice=beta_testdose, dose_species='Beta', color=beta_palette[0])
    tc_figure.add_trajectory(drb5mL, 'plot', '-', subplot_idx=(0, 1), label=r'IFN$\beta$ at {} L'.format(volEC2),
                             doseslice=beta_testdose, dose_species='Beta', color=beta_palette[2])
    tc_figure.show_figure(save_flag=True, save_dir='varying_reaction_volume_tc.pdf')

    # ------------------------------------
    # Now let's quickly look at IFN levels
    # ------------------------------------
    # 1 uL
    Mixed_Model.set_global_parameters({'volEC': volEC1, 'ka1': ka1*1E-5/volEC1, 'ka2': ka2*1E-5/volEC1,
                                       'k_a1': k_a1*1E-5/volEC1, 'k_a2': k_a2*1E-5/volEC1})
    dradf = Mixed_Model.mixed_dose_response(times, 'Free_Ia', 'Ia', list(logspace(1, 5.2)),
                                            parameters={'Ib': 0}, sf=scale_factor)
    drbdf = Mixed_Model.mixed_dose_response(times, 'Free_Ib', 'Ib', list(logspace(-1, 4)),
                                            parameters={'Ia': 0}, sf=scale_factor)

    dra15uL = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb15uL = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    # 1 mL
    Mixed_Model.set_global_parameters({'volEC': volEC2, 'ka1': ka1*1E-5/volEC2, 'ka2': ka2*1E-5/volEC2,
                                       'k_a1': k_a1*1E-5/volEC2, 'k_a2': k_a2*1E-5/volEC2})
    dradf = Mixed_Model.mixed_dose_response(times, 'Free_Ia', 'Ia', list(logspace(1, 5.2)),
                                            parameters={'Ib': 0}, sf=scale_factor)
    drbdf = Mixed_Model.mixed_dose_response(times, 'Free_Ib', 'Ib', list(logspace(-1, 4)),
                                            parameters={'Ia': 0}, sf=scale_factor)

    dra5mL = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb5mL = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    tc_figure = TimecoursePlot((1, 2))

    tc_figure.add_trajectory(dra15uL, 'plot', '-', subplot_idx=(0, 0), label=r'IFN$\alpha$ at {} L'.format(volEC1),
                             doseslice=alpha_testdose, dose_species='Alpha', color=alpha_palette[1])
    tc_figure.add_trajectory(dra5mL, 'plot', '-', subplot_idx=(0, 0), label=r'IFN$\alpha$ at {} L'.format(volEC2),
                             doseslice=alpha_testdose, dose_species='Alpha', color=alpha_palette[4])

    tc_figure.add_trajectory(drb15uL, 'plot', '-', subplot_idx=(0, 1), label=r'IFN$\beta$ at {} L'.format(volEC1),
                             doseslice=beta_testdose, dose_species='Beta', color=beta_palette[0])
    tc_figure.add_trajectory(drb5mL, 'plot', '-', subplot_idx=(0, 1), label=r'IFN$\beta$ at {} L'.format(volEC2),
                             doseslice=beta_testdose, dose_species='Beta', color=beta_palette[2])
    for ax in tc_figure.axes.flatten():
        ax.set_yscale('log')
    ifn_fig, ifn_axes = tc_figure.show_figure()
    ifn_axes[0].set_ylabel('IFN-alpha2')
    ifn_axes[1].set_ylabel('IFN-beta')
    ifn_fig.savefig('varying_reaction_volume_tc_IFN.pdf')
