from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import seaborn as sns
from ifnclass.ifnfit import StepwiseFit
import os

if __name__ == '__main__':
    raw_data = IfnData("20190121_pSTAT1_IFN_Bcell")
    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')

    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    times = [2.5, 5, 7.5, 10, 20, 60]
    alpha_doses_20190108 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses_20190108 = [0, 0.2, 6, 20, 60, 200, 600, 2000]

    Mixed_Model.set_parameters({'R2': 4140, 'R1': 4920,
                                'k_a1': 2.49e-15, 'k_a2': 1.328e-12, 'k_d3': 7.5e-06, 'k_d4': 0.06,
                                'kSOCSon': 5e-08, 'kpu': 0.0024, 'kpa': 2.08e-06,
                                'ka1': 5.3e-15, 'ka2': 1.22e-12, 'kd4': 0.86,
                                'kd3': 5.47e-05,
                                'kint_a':  0.0002, 'kint_b': 0.00086,
                                'krec_a1': 0.0001, 'krec_a2': 0.02, 'krec_b1': 0.001, 'krec_b2': 0.005})

    scale_factor = 0.2050499
    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])

    # Make predictions
    dradf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(0, 9)),
                                     parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-1, 9)),
                                     parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')
    for i in range(len(times)):
        dradf.loc['Alpha'].iloc[:, i] = dradf.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf.loc['Beta'].iloc[:, i] = drbdf.loc['Beta'].iloc[:, i].apply(scale_data)

    dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    new_fit = DoseresponsePlot((1, 2))
    new_fit.fig.set_size_inches(16, 8)
    alpha_mask = [7.5]
    beta_mask = [7.5]
    # Add fits
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='Alpha ' + str(t))
        if t not in beta_mask:
            new_fit.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label='Beta ' + str(t))

    new_fit.save_figure(save_dir=os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_3'))




