from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import seaborn as sns
from ifnclass.ifnfit import StepwiseFit
import pandas as pd
from collections import OrderedDict
import matplotlib
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)


if __name__ == '__main__':
    # ---------------------------
    # Plot simple model:
    # ---------------------------
    Simple_Model = IfnModel('Mixed_IFN_ppCompatible')
    # Set best fit parameters
    # Best fit parameters for single pop. with GAB data
    Simple_Model.set_parameters({'R2': 4920, 'R1': 1200,
                                'k_a1': 2.0e-13, 'k_a2': 1.328e-12, 'k_d3': 1.13e-4, 'k_d4': 0.9,
                                'kSOCSon': 5e-08, 'kpu': 0.0022, 'kpa': 2.36e-06,
                                'ka1': 3.3e-15, 'ka2': 1.85e-12, 'kd4': 2.0,
                                'kd3': 6.52e-05,
                                'kint_a':  0.0015, 'kint_b': 0.002,
                                'krec_a1': 0.01, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05})
    scale_factor = 1.46182313424
    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])

    times = [2.5, 5.0, 7.5, 10., 20., 60.]
    dradf = Simple_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-1, 5)),
                                      parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf = Simple_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, log10(2000))),
                                      parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')
    for i in range(len(times)):
        dradf.loc['Alpha'].iloc[:, i] = dradf.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf.loc['Beta'].iloc[:, i] = drbdf.loc['Beta'].iloc[:, i].apply(scale_data)
    dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    total_data = IfnData('custom', df=pd.concat([dradf, dradf]), conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})

    # ---------------------------
    # Now repeat for detailed model:
    # ---------------------------
    Detailed_Model = IfnModel('Mixed_IFN_detailed')
    Detailed_Model.set_parameters({'R2': 4920, 'R1': 1200,
                                'k_a1': 2.0e-13, 'k_a2': 1.328e-12, 'k_d3': 1.13e-4, 'k_d4': 0.9,
                                'kSOCSon': 5e-08, 'kpu': 0.0022, 'kpa': 2.36e-06,
                                'ka1': 3.3e-15, 'ka2': 1.85e-12, 'kd4': 2.0,
                                'kd3': 6.52e-05,
                                'kint_a':  0.0015, 'kint_b': 0.002,
                                'krec_a1': 0.01, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05})
    # Now match the simple model predictions using only unconstrained parameters:
    best_match = OrderedDict([('kdeloc', 1.0), ('kpa', 10.0), ('kSOCSmRNA', 0.100), ('mRNAtrans', 0.100),
                 ('mRNAdeg', 5.00e-06), ('kloc', 1.25e-05), ('kpu', 0.00342)])
    scale_factor = 5.2419060515
    Detailed_Model.set_parameters(best_match)

    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])

    # Make detailed model predictions
    times = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
    dradf = Detailed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-1, 5)),
                                     parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf = Detailed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, log10(2000))),
                                     parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')
    for i in range(len(times)):
        dradf.loc['Alpha'].iloc[:, i] = dradf.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf.loc['Beta'].iloc[:, i] = drbdf.loc['Beta'].iloc[:, i].apply(scale_data)
    dra60_d = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60_d = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    # ----------------------------------------
    # Finally, plot both models in comparison
    # ----------------------------------------
    new_fit = DoseresponsePlot((1, 2))
    alpha_palette = sns.color_palette("deep", 6)
    beta_palette = sns.color_palette("deep", 6)
    alpha_mask = [2.5, 7.5]
    beta_mask = [2.5, 7.5]
    # Add fits
    for idx, t in enumerate([str(el) for el in times]):
        if t not in [str(el) for el in alpha_mask]:
            new_fit.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='{} min'.format(t))
            new_fit.add_trajectory(dra60_d, t, 'plot', '--', (0, 0), 'Alpha', color=alpha_palette[idx])
        if t not in [str(el) for el in beta_mask]:
            new_fit.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1), 'Beta')
            new_fit.add_trajectory(drb60_d, t, 'plot', '--', (0, 1), 'Beta', color=beta_palette[idx])

    new_fit.add_trajectory(drb60, 60.0, 'plot', beta_palette[idx], (0, 1), 'Beta', label='Simple Model')
    new_fit.add_trajectory(drb60_d, 60.0, 'plot', '--', (0, 1), 'Beta', color=beta_palette[idx], label='Detailed Model')
    matplotlib.rc
    new_fit.show_figure(save_flag=False)




