from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnfit import DualMixedPopulation
from numpy import linspace, logspace, transpose
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from ifnclass.ifnplot import Trajectory, DoseresponsePlot
from collections import OrderedDict

if __name__ == '__main__':
    # Model
    # Parameters found by stepwise fitting GAB mean data
    initial_parameters = {'k_a1': 4.98E-14 * 2, 'k_a2': 1.328e-12, 'k_d3': 2.4e-06, 'k_d4': 0.228,
                          'kSOCSon': 8e-07, 'kpu': 0.0011,
                          'ka1': 3.3e-15, 'ka2': 1.22e-12, 'kd4': 0.86,
                          'kd3': 1.74e-05,
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

    times = [10.0, 60.0]

    # Make predictions
    dradf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia', list(logspace(1, 5.2)),
                                            parameters={'Ib': 0}, sf=scale_factor)
    drbdf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ib', list(logspace(-1, 4)),
                                            parameters={'Ia': 0}, sf=scale_factor)

    dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    # Plot primary predictions
    alpha_palette = sns.color_palette("deep", 6)
    beta_palette = sns.color_palette("deep", 6)
    new_fit = DoseresponsePlot((1, 2))
    for idx, t in enumerate(times):
        new_fit.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label=str(t)+' min',
                               linewidth=2)
        new_fit.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label=str(t) +' min',
                               linewidth=2)

    # Vary parameters 20%
    for subpop_idx, subpop in enumerate([Mixed_Model.model_1, Mixed_Model.model_2]):
        test_keys = OrderedDict([('R1', 12000.0), ('R2', 1511.1), ('S', 10000), ('ka1', 3.3e-15), ('kd1', 1), ('ka2', 1.22e-12), ('kd2', 0.015), ('ka4', 0.0003623188), ('kd4', 0.86), ('ka3', 0.0003623188), ('kd3', 1.74e-05), ('k_a1', 9.96e-14), ('k_d1', 0.03), ('k_a2', 1.328e-12), ('k_d2', 0.002), ('k_a3', 0.000362), ('k_d3', 2.4e-06), ('k_a4', 0.000362), ('k_d4', 0.228), ('kpa', 1e-06), ('kpu', 0.0011), ('kIntBasal_r1', 0.0001), ('kIntBasal_r2', 2e-05), ('krec_r1', 0.0001), ('krec_r2', 0.0001), ('kint_a', 0.00052), ('kint_b', 0.00052), ('kdeg_a', 0.0008), ('kdeg_b', 0.0008), ('krec_a1', 0.001), ('krec_a2', 0.1), ('krec_b1', 0.005), ('krec_b2', 0.05), ('kSOCS', 0.004), ('SOCSdeg', 0.0025), ('kSOCSon', 6e-07), ('kSOCSoff', 0.00055)]).keys()
        len_test_keys = len(test_keys)
        for p_idx, p in enumerate(test_keys):
            print("{}%".format(subpop_idx * 50 + p_idx/len_test_keys * 100))
            # Increase by 20%
            subpop.set_parameters({p: subpop.parameters[p] * 1.2})
            response = IfnData('custom', df=Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia', list(logspace(1, 5.2)),
                                            parameters={'Ib': 0}, sf=scale_factor),
                               conditions={'Alpha': {'Ib': 0}})
            for idx, t in enumerate(times):
                new_fit.add_trajectory(response, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='', linewidth=1, alpha=0.2)
            response = IfnData('custom', df=Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ib', list(logspace(-1, 4)),
                                            parameters={'Ia': 0}, sf=scale_factor),
                               conditions={'Beta': {'Ia': 0}})
            for idx, t in enumerate(times):
                new_fit.add_trajectory(response, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label='', linewidth=1, alpha=0.2)
            # Decrease by 20%
            subpop.set_parameters({p: subpop.parameters[p] * 0.666})
            response = IfnData('custom',
                               df=Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia', list(logspace(1, 5.2)),
                                                                  parameters={'Ib': 0}, sf=scale_factor),
                               conditions={'Alpha': {'Ib': 0}})
            for idx, t in enumerate(times):
                new_fit.add_trajectory(response, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='', linewidth=1,
                                   alpha=0.2)
            response = IfnData('custom',
                               df=Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ib', list(logspace(-1, 4)),
                                                                  parameters={'Ia': 0}, sf=scale_factor),
                               conditions={'Beta': {'Ia': 0}})
            for idx, t in enumerate(times):
                new_fit.add_trajectory(response, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label='', linewidth=1,
                                   alpha=0.2)

    # Aesthetics
    new_fit.axes[0].set_title(r'IFN$\alpha$')
    new_fit.axes[1].set_title(r'IFN$\beta$')
    for ax in new_fit.axes:
        ax.set_ylim((0, 6000))
    new_fit.show_figure(save_flag=True, save_dir=os.path.join(os.getcwd(), 'results', 'kimmel_sensitivity.pdf'))
