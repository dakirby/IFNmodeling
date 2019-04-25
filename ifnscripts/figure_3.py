from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import matplotlib.pyplot as plt
import seaborn as sns
from ifnclass.ifnfit import DualMixedPopulation
import os

if __name__ == '__main__':
    # GAB Data
    # Get all data set EC50 time courses
    newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
    newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
    newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
    newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")
    alignment = DataAlignment()
    alignment.add_data([newdata_4, newdata_3, newdata_2, newdata_1])
    alignment.align()
    alignment.get_scaled_data()
    mean_data = alignment.summarize_data()

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


    # Make predictions
    times = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
    alpha_doses_20190108 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses_20190108 = [0, 0.2, 6, 20, 60, 200, 600, 2000]

    dradf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia', list(logspace(1, 5.2)),
                                            parameters={'Ib': 0}, sf=scale_factor)
    drbdf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ib', list(logspace(-1, 4)),
                                            parameters={'Ia': 0}, sf=scale_factor)

    dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    # Plotting
    alpha_palette = sns.color_palette("deep", 6)
    beta_palette = sns.color_palette("deep", 6)
    new_fit = DoseresponsePlot((1, 2))
    new_fit.fig.set_size_inches(16, 8)
    alpha_mask = [7.5]
    beta_mask = [7.5]
    # Add fits
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label=str(t)+' min',
                                   linewidth=2)
            new_fit.add_trajectory(mean_data, t, 'errorbar', 'o', (0, 0), 'Alpha', color=alpha_palette[idx])
        if t not in beta_mask:
            new_fit.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label=str(t) +' min',
                                   linewidth=2)
            new_fit.add_trajectory(mean_data, t, 'errorbar', 'o', (0, 1), 'Beta', color=beta_palette[idx])
    # Plot aesthetics
    ylim = (0, 5500)
    for each in new_fit.axes:
        each.set_ylim(ylim)
    new_fit.axes[0].set_title(r'IFN$\alpha$')
    new_fit.axes[1].set_title(r'IFN$\beta$')
    new_fit.save_figure(save_dir=os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_3', 'Figure_3.pdf'))




