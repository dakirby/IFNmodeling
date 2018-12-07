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

if __name__ == '__main__':
    newdata = IfnData("20181113_B6_IFNs_Dose_Response_Bcells")
    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    # Stepwise fitting
    # ----------------
    # First fit the 2.5 minute data
    stepfit25 = StepwiseFit(Mixed_Model, smooth25IfnData,
                            {'kpa': (1E-8, 1E-3), 'kd4': (0.06, 1), 'k_d4': (0.0006, 0.3),
                             'krec_a2': (0.0005, 0.05),
                             'krec_b2': (0.0001, 0.01)}, n=20)
    best_parameters, best_scale_factor = stepfit25.fit()
    print("The fit for 2.5 minutes was:")
    print(best_parameters)
    print(best_scale_factor)

    # Generate a figure of this fit
    #   First simulate continuous dose-response curve
    dra25df = stepfit25.model.doseresponse([2.5, 10, 20, 60], 'TotalpSTAT', 'Ia', list(logspace(-3, 4)),
                                  parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drb25df = stepfit25.model.doseresponse([2.5, 10, 20, 60], 'TotalpSTAT', 'Ib', list(logspace(-3, 4)),
                                  parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')

    # Scale by best fit scale factor, so comparison to data is good
    for i in range(4):
        dra25df.loc['Alpha'].iloc[:, i] *= best_scale_factor
        drb25df.loc['Beta'].iloc[:, i] *= best_scale_factor

    # Make the DR curves into IfnData objects, for easy plotting
    dra25 = IfnData('custom', df=dra25df, conditions={'Alpha': {'Ib': 0}})
    drb25 = IfnData('custom', df=drb25df, conditions={'Beta': {'Ia': 0}})

    # Generate figure by adding simulations, then data
    results_plot = DoseresponsePlot((1, 2))
    for idx, t in enumerate(['2.5', '10', '20', '60']):
        results_plot.add_trajectory(dra25, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha')
        results_plot.add_trajectory(drb25, t, 'plot', beta_palette[idx], (0,1), 'Beta')
    for idx, t in enumerate([2.5, 10, 20, 60]):
        results_plot.add_trajectory(newdata, t, 'scatter', alpha_palette[idx], (0, 0), 'Alpha', dn=1)
        results_plot.add_trajectory(newdata, t, 'scatter', beta_palette[idx], (0, 1), 'Beta', dn=1)

    results_plot.save_figure()

    # Now try to fit the 60 minute data starting from the 2.5 minute results
    # ----------------------------------------------------------------------
    stepfit60 = StepwiseFit(stepfit25.model, smooth60IfnData,
                            {'kpa': (1E-8, 1E-3), 'kd4': (0.06, 1), 'k_d4': (0.0006, 0.3),
                             'R2': (200, 12000), 'R1': (200, 12000), 'kSOCSon': (1E-8, 5E-4)}, n=20)
    best_parameters, best_scale_factor = stepfit60.fit()
    print("The final fit was:")
    print(best_parameters)
    print(best_scale_factor)
    dra60df = stepfit60.model.doseresponse([2.5, 10, 20, 60], 'TotalpSTAT', 'Ia', list(logspace(-3, 4)),
                                  parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drb60df = stepfit60.model.doseresponse([2.5, 10, 20, 60], 'TotalpSTAT', 'Ib', list(logspace(-3, 4)),
                                  parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')

    for i in range(4):
        dra60df.loc['Alpha'].iloc[:, i] *= best_scale_factor
        drb60df.loc['Beta'].iloc[:, i] *= best_scale_factor

    dra60 = IfnData('custom', df=dra60df, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drb60df, conditions={'Beta': {'Ia': 0}})

    results_plot2 = DoseresponsePlot((1, 2))
    for idx, t in enumerate(['2.5', '10', '20', '60']):
        results_plot2.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha')
        results_plot2.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0,1), 'Beta')
    for idx, t in enumerate([2.5, 10, 20, 60]):
        results_plot2.add_trajectory(newdata, t, 'scatter', alpha_palette[idx], (0, 0), 'Alpha', dn=1)
        results_plot2.add_trajectory(newdata, t, 'scatter', beta_palette[idx], (0, 1), 'Beta', dn=1)

    results_plot2.show_figure(save_flag=True)
