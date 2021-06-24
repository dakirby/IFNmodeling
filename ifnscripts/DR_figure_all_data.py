from ifnclass.ifndata import IfnData, DataAlignment
from numpy import linspace, logspace
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from ifnclass.ifnplot import DoseresponsePlot
import matplotlib.gridspec as gridspec
import load_model as lm


if __name__ == '__main__':
    # ----------------------------------
    # Get all data sets
    # ----------------------------------
    newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
    newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
    newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
    newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")

    # 20190108
    ec50_20190108 = newdata_1.get_ec50s()

    # 20190119
    ec50_20190119 = newdata_2.get_ec50s()

    # 20190121
    ec50_20190121 = newdata_3.get_ec50s()

    # 20190214
    ec50_20190214 = newdata_4.get_ec50s()

    # Aligned data, to get scale factors for each data set
    alignment = DataAlignment()
    alignment.add_data([newdata_4, newdata_3, newdata_2, newdata_1])
    alignment.align()
    alignment.get_scaled_data()
    mean_data = alignment.summarize_data()

    # -------------------------------
    # Plot model dose response curves
    # -------------------------------
    alpha_palette = sns.color_palette("rocket_r")
    beta_palette = sns.color_palette("rocket_r")

    new_fit = DoseresponsePlot((1, 2))
    new_fit.axes[0].set_ylabel('pSTAT1 (MFI)')
    new_fit.axes[1].set_ylabel('pSTAT1 (MFI)')

    alpha_mask = [7.5, 10.0]
    beta_mask = [7.5, 10.0]
    # Add fits
    for idx, t in enumerate([2.5, 5.0, 7.5, 10.0, 20.0, 60.0]):
        new_fit.add_trajectory(mean_data, t, 'errorbar', 'o', (0, 0),
                               'Alpha', color=alpha_palette[idx])
        new_fit.add_trajectory(mean_data, t, 'errorbar', 'o', (0, 1),
                               'Beta', color=beta_palette[idx], label=str(t) + ' min')

    dr_fig, dr_axes = new_fit.show_figure(show_flag=False)

    # Format Figure_2
    # Dose response aesthetics
    for ax in dr_axes:
        ax.set_ylim((0, 5000))
    dr_axes[0].set_title(r'IFN$\alpha$2')
    dr_axes[1].set_title(r'IFN$\beta$')
    for direction in ['top', 'right']:
        dr_axes[0].spines[direction].set_visible(False)
        dr_axes[1].spines[direction].set_visible(False)
    dr_fig.set_size_inches(12, 4)
    dr_fig.savefig(os.path.join(os.getcwd(), 'results', 'Figures',
                     'Supplementary', 'all_data.pdf'))
