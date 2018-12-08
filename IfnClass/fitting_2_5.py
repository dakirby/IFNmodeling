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

if __name__ == '__main__':
    newdata = IfnData("20181113_B6_IFNs_Dose_Response_Bcells")
    Mixed_Model = IfnModel('')
    Mixed_Model.load_model('stepwisefitmodel.p')

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

    initial_fit = DoseresponsePlot((1, 2))
    for idx, t in enumerate(['2.5', '10', '20', '60']):
        initial_fit.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha')
        initial_fit.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1), 'Beta')
    for idx, t in enumerate([2.5, 10, 20, 60]):
        initial_fit.add_trajectory(newdata, t, 'scatter', alpha_palette[idx], (0, 0), 'Alpha', dn=1)
        initial_fit.add_trajectory(newdata, t, 'scatter', beta_palette[idx], (0, 1), 'Beta', dn=1)

    initial_fit.show_figure(save_flag=False)

    # ---------------------------
    # Now try to improve the fit:
    # ---------------------------
    Mixed_Model.set_parameters({'kd4':0.9})
    dradf = Mixed_Model.doseresponse([2.5, 10, 20, 60], 'TotalpSTAT', 'Ia', list(logspace(-3, 5)),
                                           parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf = Mixed_Model.doseresponse([2.5, 10, 20, 60], 'TotalpSTAT', 'Ib', list(logspace(-3, 4)),
                                           parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')

    for i in range(4):
        dradf.loc['Alpha'].iloc[:, i] = dradf.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf.loc['Beta'].iloc[:, i] = drbdf.loc['Beta'].iloc[:, i].apply(scale_data)

    dra60 = IfnData('custom', df=dra60df, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drb60df, conditions={'Beta': {'Ia': 0}})

    new_fit = DoseresponsePlot((1, 2))
    for idx, t in enumerate(['2.5', '10', '20', '60']):
        new_fit.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha')
        new_fit.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1), 'Beta')
    for idx, t in enumerate([2.5, 10, 20, 60]):
        new_fit.add_trajectory(newdata, t, 'scatter', alpha_palette[idx], (0, 0), 'Alpha', dn=1)
        new_fit.add_trajectory(newdata, t, 'scatter', beta_palette[idx], (0, 1), 'Beta', dn=1)

        new_fit.show_figure(save_flag=False)



