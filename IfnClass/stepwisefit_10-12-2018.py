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
    newdata = IfnData("MacParland_Extended")
    df5 = pd.DataFrame(newdata.data_set.loc[:, 5])
    df15 = pd.DataFrame(newdata.data_set.loc[:, 15])
    df30 = pd.DataFrame(newdata.data_set.loc[:, 30])
    df60 = pd.DataFrame(newdata.data_set.loc[:, 60])
    IfnData_5 = IfnData('custom', df=df5, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
    IfnData_15 = IfnData('custom', df=df15, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
    IfnData_30 = IfnData('custom', df=df30, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
    IfnData_60 = IfnData('custom', df=df60, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})

    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    # Stepwise fitting
    # ----------------
    # First fit the 30 minute data
    stepfit25 = StepwiseFit(Mixed_Model, IfnData_30,
                            {'kpu': (0.0001, 0.01), 'kpa': (1e-7, 1e-5), 'kd4': (0.03, 0.9), 'k_d4': (0.001, 0.1),
                             'kSOCS': (0.0001, 0.001), 'ka4': (0.01, 1), 'kSOCSon': (1E-9, 1E-7)}, n=15)
    best_parameters, best_scale_factor = stepfit25.fit()
    print("The fit for 30 minutes was:")
    print(best_parameters)
    print(best_scale_factor)
    scale_data = lambda q: (best_scale_factor*q[0], best_scale_factor*q[1])

    # Generate a figure of this fit
    #   First simulate continuous dose-response curve
    dra25df = stepfit25.model.doseresponse([5, 15, 30, 60], 'TotalpSTAT', 'Ia', list(logspace(1, 4)),
                                  parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drb25df = stepfit25.model.doseresponse([5, 15, 30, 60], 'TotalpSTAT', 'Ib', list(logspace(1, 4.1)),
                                  parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')

    # Scale by best fit scale factor, so comparison to data is good
    for i in range(4):
        dra25df.loc['Alpha'].iloc[:, i] = dra25df.loc['Alpha'].iloc[:, i].apply(scale_data)
        drb25df.loc['Beta'].iloc[:, i] = drb25df.loc['Beta'].iloc[:, i].apply(scale_data)

    # Make the DR curves into IfnData objects, for easy plotting
    dra25 = IfnData('custom', df=dra25df, conditions={'Alpha': {'Ib': 0}})
    drb25 = IfnData('custom', df=drb25df, conditions={'Beta': {'Ia': 0}})

    # Generate figure by adding simulations, then data
    results_plot = DoseresponsePlot((1, 2))
    for idx, t in enumerate([5, 15, 30, 60]):
        results_plot.add_trajectory(dra25, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha')
        results_plot.add_trajectory(drb25, t, 'plot', beta_palette[idx], (0,1), 'Beta')
    for idx, t in enumerate([5, 15, 30, 60]):
        results_plot.add_trajectory(newdata, t, 'errorbar', alpha_palette[idx], (0, 0), 'Alpha', dn=1)
        results_plot.add_trajectory(newdata, t, 'errorbar', beta_palette[idx], (0, 1), 'Beta', dn=1)

    results_plot.show_figure()

    # Now try to fit the 60 minute data starting from the 5 minute results
    # ----------------------------------------------------------------------
    stepfit60 = StepwiseFit(stepfit25.model, IfnData_60,
                            {'kSOCSon': (1E-9, 1E-7),
                             'krec_a2': (0.0001, 0.005), 'krec_b2': (0.0001, 0.01),
                             'krec_a1': (0.00003, 0.003), 'krec_b1': (0.0001, 0.01)}, n=15)
    best_parameters, best_scale_factor60 = stepfit60.fit()
    scale_data60 = lambda q: (best_scale_factor60*q[0], best_scale_factor60*q[1])

    with open('stepwisefitmodel.p','wb') as f:
        pickle.dump(stepfit60.model.__dict__,f,2)
    with open('stepwisefitobject.p','wb') as f:
       pickle.dump(stepfit60.__dict__,f,2)
    print(stepfit60.model.parameters)
    print("The final fit was:")
    print(best_parameters)
    print(best_scale_factor60)
    dra60df = stepfit60.model.doseresponse([5, 15, 30, 60], 'TotalpSTAT', 'Ia', list(logspace(1, 4)),
                                  parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drb60df = stepfit60.model.doseresponse([5, 15, 30, 60], 'TotalpSTAT', 'Ib', list(logspace(1, 4.1)),
                                  parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')

    with open('dra60df.p','wb') as f:
        pickle.dump(dra60df,f,2)
    with open('drb60df.p','wb') as f:
        pickle.dump(drb60df,f,2)

    for i in range(4):
        dra60df.loc['Alpha'].iloc[:, i] = dra60df.loc['Alpha'].iloc[:, i].apply(scale_data60)
        drb60df.loc['Beta'].iloc[:, i] = drb60df.loc['Beta'].iloc[:, i].apply(scale_data60)

    dra60 = IfnData('custom', df=dra60df, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drb60df, conditions={'Beta': {'Ia': 0}})

    results_plot2 = DoseresponsePlot((1, 2))
    # Add fits
    for idx, t in enumerate([5, 15, 30, 60]):
        results_plot2.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha')
        results_plot2.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0,1), 'Beta')
    # Add data
    for idx, t in enumerate([5, 15, 30, 60]):
        results_plot2.add_trajectory(newdata, t, 'errorbar', alpha_palette[idx], (0, 0), 'Alpha', dn=1)
        results_plot2.add_trajectory(newdata, t, 'errorbar', beta_palette[idx], (0, 1), 'Beta', dn=1)

    results_plot2.show_figure(save_flag=True)
