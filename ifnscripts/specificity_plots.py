from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnfit import DualMixedPopulation
from numpy import linspace, logspace, transpose
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from ifnclass.ifnplot import Trajectory, DoseresponsePlot
import pandas as pd
from sys import platform


if __name__ == '__main__':
    # ----------------------
    # Set up figure layout
    # ----------------------
    spec_fig = DoseresponsePlot((3, 2))
    spec_fig.fig.set_size_inches(12, 8)
    # ---------------------------
    # Set up data for comparison
    # ---------------------------
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)
    data_palette = sns.color_palette("muted", 6)
    marker_shape = ["o", "v", "s", "P", "d", "1", "x", "*"]
    dataset_names = ["20190108", "20190119", "20190121", "20190214"]

    newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
    newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
    newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
    newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")

    # Aligned data, to get scale factors for each data set
    alignment = DataAlignment()
    alignment.add_data([newdata_4, newdata_3, newdata_2, newdata_1])
    alignment.align()
    alignment.get_scaled_data()
    mean_data = alignment.summarize_data()

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
    times = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
    test_doses = list(logspace(-1, 5.2))

    dradf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia', test_doses,
                                            parameters={'Ib': 0}, sf=scale_factor)
    drbdf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ib', test_doses,
                                            parameters={'Ia': 0}, sf=scale_factor)

    dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    # -------------------------------------------
    # Get comparisons of beta to alpha response
    # -------------------------------------------
    dra60_noSigma = dra60.drop_sigmas(in_place=False)
    drb60_noSigma = drb60.drop_sigmas(in_place=False)

    temp1 = drb60_noSigma.data_set.loc['Beta'] / dra60_noSigma.data_set.loc['Alpha']
    temp1 = pd.concat([temp1], keys=['Ratio'], names=['Dose_Species'])
    ratio_response = IfnData('custom', df=temp1)
    ratio_response.add_sigmas()

    temp2 = drb60_noSigma.data_set.loc['Beta'] - dra60_noSigma.data_set.loc['Alpha']
    temp2 = pd.concat([temp2], keys=['Difference'], names=['Dose_Species'])
    difference_response = IfnData('custom', df=temp2)
    difference_response.add_sigmas()

    # ---------------------------------------------
    # Signal integration requires a bit more work
    # ---------------------------------------------
    integral_times = [float(el) for el in list(np.arange(0.0, 61.0, 1.0))]

    dra_fine_time_df = Mixed_Model.mixed_dose_response(integral_times, 'TotalpSTAT', 'Ia', test_doses,
                                            parameters={'Ib': 0}, sf=scale_factor)
    drb_fine_time_df = Mixed_Model.mixed_dose_response(integral_times, 'TotalpSTAT', 'Ib', test_doses,
                                            parameters={'Ia': 0}, sf=scale_factor)
    drafine = IfnData('custom', df=dra_fine_time_df, conditions={'Alpha': {'Ib': 0}})
    drbfine = IfnData('custom', df=drb_fine_time_df, conditions={'Beta': {'Ia': 0}})
    drafine_noSigma = drafine.drop_sigmas(in_place=False)
    drbfine_noSigma = drbfine.drop_sigmas(in_place=False)

    t = str(integral_times[-1])
    temp3 = pd.DataFrame(list(zip(['Alpha_Integral' for _ in range(len(test_doses))],
                                  test_doses,
                                  np.trapz(drafine_noSigma.data_set.loc['Alpha'], axis=1))),
                         columns=['Dose_Species', 'Dose (pM)', t])
    temp3.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)

    temp4 = pd.DataFrame(list(zip(['Beta_Integral' for _ in range(len(test_doses))],
                                  test_doses,
                                  np.trapz(drbfine_noSigma.data_set.loc['Beta'], axis=1))),
                         columns=['Dose_Species', 'Dose (pM)', t])
    temp4.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)

    temp5 = pd.concat([temp3,temp4])

    integral_response = IfnData('custom', df=temp5)

    temp6 = integral_response.data_set.loc['Beta_Integral'] - integral_response.data_set.loc['Alpha_Integral']
    temp6 = pd.concat([temp6], keys=['Integral_Difference'], names=['Dose_Species'])
    integral_difference_response = IfnData('custom', df=temp6)

    integral_difference_response.add_sigmas()
    integral_response.add_sigmas()

    # -------------------------------
    # Plot model dose response curves
    # -------------------------------
    alpha_palette = sns.color_palette("deep", 6)
    beta_palette = sns.color_palette("deep", 6)

    alpha_mask = [7.5]
    beta_mask = [7.5]
    # Add fits
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            spec_fig.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label=str(t) + ' min',
                                   linewidth=2)
            spec_fig.add_trajectory(mean_data, t, 'errorbar', 'o', (0, 0), 'Alpha', color=alpha_palette[idx])
        if t not in beta_mask:
            spec_fig.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label=str(t) + ' min',
                                   linewidth=2)
            spec_fig.add_trajectory(mean_data, t, 'errorbar', 'o', (0, 1), 'Beta', color=beta_palette[idx])
    spec_fig.axes[0][0].set_title(r'IFN$\alpha$ fit')
    spec_fig.axes[0][1].set_title(r'IFN$\beta$ fit')

    # -----------------------
    # Plot ratio vs time
    # -----------------------
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            spec_fig.add_trajectory(ratio_response, t, 'plot', alpha_palette[idx], (1, 0), 'Ratio',
                                    label=str(t) + ' min', linewidth=2)
    spec_fig.axes[1][0].set_title(r'$\frac{\mathrm{IFN}\beta \mathrm{\, response}}{\mathrm{IFN}\alpha \mathrm{\, response}}$')

    # -----------------------
    # Plot difference vs time
    # -----------------------
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            spec_fig.add_trajectory(difference_response, t, 'plot', alpha_palette[idx], (1, 1), 'Difference',
                                    label=str(t) + ' min', linewidth=2)
    spec_fig.axes[1][1].set_title(r'$\mathrm{IFN}\beta \mathrm{\, response} - \mathrm{IFN}\alpha \mathrm{\, response}$')

    # -----------------------
    # Plot integral vs time
    # -----------------------
    t = integral_times[-1]
    spec_fig.add_trajectory(integral_response, t, 'plot', alpha_palette[3], (2, 0), 'Alpha_Integral',
                                    label=r'$\mathrm{IFN}\alpha$ 60 min', linewidth=2)
    spec_fig.add_trajectory(integral_response, t, 'plot', beta_palette[2], (2, 0), 'Beta_Integral',
                                    label=r'$\mathrm{IFN}\beta$ 60 min', linewidth=2)
    spec_fig.axes[2][0].set_title(r'$\int \mathrm{pSTAT}[t] dt$')


    # save figure
    if platform == "linux" or platform == "linux2":
        spec_fig.show_figure(save_flag=True,
                             save_dir=os.path.join(os.getcwd(), 'results', 'Figures', 'Pure_Theory', 'Alpha_vs_Beta_response.pdf'))
    else: # Windows
        spec_fig.show_figure(save_flag=True,
                             save_dir=os.path.join(os.getcwd(), 'results', 'Figures', 'Pure Theory', 'Alpha_vs_Beta_response.pdf'))
