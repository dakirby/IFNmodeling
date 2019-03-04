from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnplot import DoseresponsePlot, TimecoursePlot
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    # ------------------------------
    # Align all data
    # ------------------------------
    newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
    newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
    newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
    newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")

    alignment = DataAlignment()
    alignment.add_data([newdata_4, newdata_3, newdata_2, newdata_1])
    alignment.align()
    alignment.get_scaled_data()
    alignment.summarize_data()
    exit()

    # -------------------------------
    # Plot alignment of Dose Response
    # -------------------------------
    species = alignment.scaled_data[0].get_dose_species()
    #   Make subtitles
    recorded_times = alignment.scaled_data[0].get_times()[species[0]]
    titles = alignment.scaled_data[0].get_dose_species()
    titles[0] += "\r\n\r\n{} min".format(recorded_times[0])
    titles[1] += "\r\n\r\n{} min".format(recorded_times[0])
    for t in recorded_times[1:]:
        for s in species:
            titles.append(str(t)+' min')
    num_times = len(alignment.scaled_data[0].get_times()[species[0]])
    clist = ["Reds", "Greens", "Blues", "Purples", "Oranges", "Greys"]
    dr_plot = DoseresponsePlot((num_times, len(species)))
    for i in range(num_times):
        for j in range(len(species)):
            dr_plot.axes[i][j].set_title(titles[i*len(species) + j])

    # Add fits
    for d_idx, dataset in enumerate(alignment.scaled_data):
        colour_palette = sns.color_palette(clist[d_idx], 1)
        for ax_idx, spec in enumerate(dataset.get_dose_species()):
            for idx, t in enumerate(alignment.scaled_data[0].get_times()[spec]):
                spec_data = pd.concat([dataset.data_set.loc[spec]], keys=[spec], names=['Dose_Species'])
                spec_IfnData = IfnData('custom', df=spec_data, conditions=dataset.conditions)
                dr_plot.add_trajectory(spec_IfnData, t, 'plot', colour_palette[0], (idx, ax_idx), spec,
                                       label=dataset.name[0:8])
    dr_fig, dr_axes = dr_plot.show_figure(save_flag=False)
    dr_fig.set_size_inches(10.5, 7.1*num_times)
    dr_fig.savefig('results/GAB_NewData/align_dr.pdf')

    # -------------------------------
    # Plot alignment of Time Course
    # -------------------------------
    species = alignment.scaled_data[0].get_dose_species()
    #   Make subtitles
    recorded_doses = alignment.scaled_data[0].get_doses()
    num_doses = len(recorded_doses[species[0]])
    titles = [species[0],species[1]]
    titles[0] += "\r\n\r\n{} pM".format(recorded_doses[species[0]][0])
    titles[1] += "\r\n\r\n{} pM".format(recorded_doses[species[1]][0])
    for d in range(1, num_doses):
        for s in species:
            titles.append(str(recorded_doses[s][d])+' pM')

    # Add fits
    tc_plot = TimecoursePlot((num_doses, len(species)))
    for i in range(num_doses):
        for j in range(len(species)):
            tc_plot.axes[i][j].set_title(titles[i*len(species) + j])

    for d_idx, dataset in enumerate(alignment.scaled_data):
        colour_palette = sns.color_palette(clist[d_idx], 1)
        for spec_idx, spec in enumerate(species):
            for dose_idx, dose in enumerate(recorded_doses[spec]):
                dose_dataset = pd.concat([dataset.data_set.loc[spec, dose, :]], keys=[spec], names=['Dose_Species'])
                dose_IfnData = IfnData('custom', df=dose_dataset, conditions=dataset.conditions)
                tc_plot.add_trajectory(dose_IfnData, 'plot', colour_palette[0], (dose_idx, spec_idx),
                                       label=dataset.name[0:8])
    tc_fig, tc_axes = tc_plot.show_figure(save_flag=False)
    tc_fig.set_size_inches(10.5, 7.1*num_doses)
    tc_fig.savefig('results/GAB_NewData/align_tc.pdf')

