from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnplot import DoseresponsePlot, TimecoursePlot
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    # ------------
    # Process data
    # ------------
    small_1 = IfnData("20190108_pSTAT1_IFN_Small_Bcells")
    small_2 = IfnData("20190119_pSTAT1_IFN_Small_Bcells")
    small_3 = IfnData("20190121_pSTAT1_IFN_Small_Bcells")
    small_4 = IfnData("20190214_pSTAT1_IFN_Small_Bcells")

    large_1 = IfnData("20190108_pSTAT1_IFN_Large_Bcells")
    large_2 = IfnData("20190119_pSTAT1_IFN_Large_Bcells")
    large_3 = IfnData("20190121_pSTAT1_IFN_Large_Bcells")
    large_4 = IfnData("20190214_pSTAT1_IFN_Large_Bcells")


    small_alignment = DataAlignment()
    small_alignment.add_data([small_4, small_3, small_2, small_1])
    small_alignment.align()
    small_alignment.get_scaled_data()
    mean_small_data = small_alignment.summarize_data()

    large_alignment = DataAlignment()
    large_alignment.add_data([large_4, large_3, large_2, large_1])
    large_alignment.align()
    large_alignment.get_scaled_data()
    mean_large_data = large_alignment.summarize_data()

    # ------------
    # Plot data
    # ------------
    alpha_palette = sns.color_palette("deep", 6)
    beta_palette = sns.color_palette("deep", 6)

    dr_plot = DoseresponsePlot((2, 2))
    for r_idx, cell_size in enumerate([mean_small_data, mean_large_data]):
        for c_idx, species in enumerate(['Alpha', 'Beta']):
            for t_idx, time in enumerate(cell_size.get_times()[species]):
                if species == 'Alpha':
                    c = alpha_palette[t_idx]
                else:
                    c = beta_palette[t_idx]
                dr_plot.add_trajectory(cell_size, time, 'errorbar', 'o--', (r_idx, c_idx), species,
                                       label=str(time)+" min", color=c)
    dr_plot.axes[0][0].set_title(r'IFN$\alpha$')
    dr_plot.axes[0][0].set_ylabel('Small cell response')
    dr_plot.axes[0][1].set_title(r'IFN$\beta$')
    dr_plot.axes[1][0].set_ylabel('Large cell response')

    for ax in [item for sublist in dr_plot.axes for item in sublist]:
        ax.set_ylim(top=6000)
    dr_plot.show_figure()