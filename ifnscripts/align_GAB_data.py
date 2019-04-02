from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnplot import DoseresponsePlot, TimecoursePlot
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
    newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
    newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
    newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")

    alignment = DataAlignment()
    alignment.add_data([newdata_4, newdata_3, newdata_2, newdata_1])
    alignment.align()
    alignment.get_scaled_data()
    mean_data = alignment.summarize_data()

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

    # -------------------------------------
    # Plot mean Dose Response & Time Course
    # -------------------------------------
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)
    times = newdata_4.get_times('Alpha')

    mean_dr_plot = DoseresponsePlot((1, 2))
    alpha_mask = [0.0, 7.5]
    beta_mask = [0.0, 7.5]
    # Add data
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            mean_dr_plot.add_trajectory(mean_data, t, 'errorbar', 'o-', (0, 0), 'Alpha', label=str(t)+' min',
                                        color=alpha_palette[idx])
        if t not in beta_mask:
            mean_dr_plot.add_trajectory(mean_data, t, 'errorbar', 'o-', (0, 1), 'Beta', label=str(t)+' min',
                                        color=beta_palette[idx])
    fig, axes = mean_dr_plot.show_figure(show_flag=False)
    mean_dr_plot.axes[0].legend()
    mean_dr_plot.axes[1].legend()
    mean_dr_plot.axes[0].set_title(r'IFN$\alpha$')
    mean_dr_plot.axes[1].set_title(r'IFN$\beta$')
    mean_dr_plot.fig.show()
    mean_dr_plot.fig.savefig('results/GAB_NewData/mean_dr.pdf')


    mean_tc_plot = TimecoursePlot((1, 2))
    doses_alpha = newdata_4.get_doses('Alpha')
    doses_beta = newdata_4.get_doses('Beta')
    alpha_palette = sns.color_palette("Reds", 8)
    beta_palette = sns.color_palette("Greens", 8)
    # Add data
    for idx, d in enumerate(doses_alpha):
        # Optional mask:
        if d not in alpha_mask:
            atc = IfnData('custom', df=mean_data.data_set.loc['Alpha', d, :])
            mean_tc_plot.add_trajectory(atc, 'errorbar', 'o-', (0, 0), color=alpha_palette[idx], label='Alpha ' + str(d))
    for idx, d in enumerate(doses_beta):
        if d not in beta_mask:
            btc = IfnData('custom', df=mean_data.data_set.loc['Beta', d, :])
            mean_tc_plot.add_trajectory(btc, 'errorbar', 'o-', (0, 1), label='Beta ' + str(d), color=beta_palette[idx])
    mean_tc_plot.axes[0].legend()
    mean_tc_plot.axes[1].legend()
    mean_tc_plot.show_figure(save_flag=True, save_dir='results/GAB_NewData/mean_tc.pdf')

