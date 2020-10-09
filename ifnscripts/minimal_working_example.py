from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import DoseresponsePlot
from numpy import linspace, logspace
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    dataset_names = ["20190108", "20190119", "20190121", "20190214"]

    # --------------------
    # Set up Model
    # --------------------
    best_fit_params = {'k_a1': 4.98E-14 * 2, 'k_a2': 8.30e-13 * 2,
                       'k_d4': 0.006 * 3.8,
                       'kpu': 0.0095,
                       'ka2': 4.98e-13 * 2.45, 'kd4': 0.3 * 2.867,
                       'kint_a': 0.00052, 'kint_b': 0.00052,
                       'krec_a1': 0.001, 'krec_a2': 0.1,
                       'krec_b1': 0.005, 'krec_b2': 0.05,
                       'kSOCSon': 6e-07,
                       'R1': 9000.0, 'R2': 1511.1}
    testModel = IfnModel('Mixed_IFN_ppCompatible')
    testModel.set_parameters(best_fit_params)

    # --------------------
    # Perform simulations
    # --------------------
    times = [2.5, 5.0, 10.0, 20.0, 30.0, 60.0]
    doses = [10, 100, 300, 1000, 3000, 10000, 100000]

    df = testModel.doseresponse(times, 'TotalpSTAT', 'Ia',
                                doses,
                                parameters={'Ib': 0},
                                scale_factor=4.1,
                                return_type='dataframe',
                                dataframe_labels='Alpha')
    print(df)
    DR_Simulation = IfnData('custom', df=df, conditions={'Alpha': {'Ib': 0}})
    DR_Simulation.drop_sigmas()

    # -----------------------------------------------------------
    # Load several sets of experimental data and align them
    # -----------------------------------------------------------

    # These datasets are already prepared for use in /ifndatabase
    expdata_1 = IfnData("20190214_pSTAT1_IFN_Bcell")
    expdata_2 = IfnData("20190121_pSTAT1_IFN_Bcell")
    expdata_3 = IfnData("20190119_pSTAT1_IFN_Bcell")
    expdata_4 = IfnData("20190108_pSTAT1_IFN_Bcell")

    # Aligned data, to get scale factors for each data set
    alignment = DataAlignment()
    alignment.add_data([expdata_1, expdata_2, expdata_3, expdata_4])
    alignment.align()
    alignment.get_scaled_data()

    # Provides a new IfnData instance with standard error computed as the
    # std. dev. between experimental replicates:
    mean_data = alignment.summarize_data()

    # ---------------------------------
    # Make dose-response plot
    # ---------------------------------
    colour_palette = sns.color_palette("rocket_r", 6)

    DR_plot = DoseresponsePlot((1, 1))  # shape=(1,1) since we only want 1 panel

    # Add simulations and data to the DoseresponsePlot
    for idx, t in enumerate([2.5, 10.0, 20.0, 60.0]):
        DR_plot.add_trajectory(DR_Simulation,
                               t,  # the time slice being added
                               'plot',  # controls the line type
                               colour_palette[idx],
                               (0, 0),  # the row and column index for the panel; useful when making multi-panel plots
                               'Alpha',  # the dose species, required to index the Pandas dataframe
                               label=str(t)+' min',  # used to generate a legend
                               linewidth=2)
        DR_plot.add_trajectory(mean_data,
                               t,
                               'errorbar',  # the line type will be points with error bars
                               'o',  # the marker shape to use for the mean response
                               (0, 0), # plot on the same panel as the simulation
                               'Alpha',
                               color=colour_palette[idx])

    dr_fig, dr_axes = DR_plot.show_figure()
