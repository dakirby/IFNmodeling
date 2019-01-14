from ifnclass.ifndata import IfnData
from ifnclass.ifnplot import DoseresponsePlot, TimecoursePlot
import seaborn as sns


if __name__ == '__main__':
    olddata = IfnData("20181113_B6_IFNs_Dose_Response_Bcells")
    newdata = IfnData("20190108_pSTAT1_IFN")
    times = [2.5, 5, 7.5, 10, 20, 60]
    alpha_doses_20181113 = [5, 50, 250, 500, 5000, 25000, 50000]
    alpha_doses_20190108 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses_20181113 = [0.1, 1, 5, 10, 100, 200, 1000]
    beta_doses_20190108 = [0, 0.2, 6, 20, 60, 200, 600, 2000]
    alpha_palette_new = sns.color_palette("Reds", 6)
    beta_palette_new = sns.color_palette("Greens", 6)
    alpha_palette_old = sns.color_palette("Reds", 6)
    beta_palette_old = sns.color_palette("Greens", 6)

    data_plot_dr = DoseresponsePlot((2, 2))
    alpha_mask = []
    beta_mask = []

    # Add data
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            data_plot_dr.add_trajectory(newdata, t, 'scatter', alpha_palette_new[idx], (0, 0), 'Alpha', dn=1)
            data_plot_dr.add_trajectory(newdata, t, 'plot', '--', (0, 0), 'Alpha', dn=1, color=alpha_palette_new[idx], label='Alpha ' + str(t))
            data_plot_dr.add_trajectory(olddata, t, 'scatter', alpha_palette_old[idx], (1, 0), 'Alpha', dn=1)
            data_plot_dr.add_trajectory(olddata, t, 'plot', '--', (1, 0), 'Alpha', dn=1, color=alpha_palette_old[idx], label='Alpha ' + str(t))

        if t not in beta_mask:
            data_plot_dr.add_trajectory(newdata, t, 'scatter', beta_palette_new[idx], (0, 1), 'Beta', dn=1)
            data_plot_dr.add_trajectory(newdata, t, 'plot', '--', (0, 1), 'Beta', dn=1, color=beta_palette_new[idx], label='Beta ' + str(t))
            data_plot_dr.add_trajectory(olddata, t, 'scatter', beta_palette_old[idx], (1, 1), 'Beta', dn=1)
            data_plot_dr.add_trajectory(olddata, t, 'plot', '--', (1, 1), 'Beta', dn=1, color=beta_palette_old[idx], label='Beta ' + str(t))

    data_plot_dr.axes[0][0].set_title(r"IFN$\alpha$ new data")
    data_plot_dr.axes[0][1].set_title(r"IFN$\beta$ new data")
    data_plot_dr.axes[1][0].set_title(r"IFN$\alpha$ old data")
    data_plot_dr.axes[1][1].set_title(r"IFN$\beta$ old data")

    data_plot_dr.show_figure(save_flag=False)

    # -------------------------
    # Time courses
    # -------------------------
    alpha_palette_new = sns.color_palette("Reds", 8)
    beta_palette_new = sns.color_palette("Greens", 8)
    alpha_palette_old = sns.color_palette("Reds", 8)
    beta_palette_old = sns.color_palette("Greens", 8)

    data_plot_tc = TimecoursePlot((2, 2))
    alpha_mask = []
    beta_mask = []
    # Add data

    # Alpha
    for idx, d in enumerate(alpha_doses_20190108):
        # Optional mask:
        if d not in alpha_mask:
            atc = IfnData('custom', df=newdata.data_set.loc['Alpha', d, :])
            data_plot_tc.add_trajectory(atc, 'scatter', alpha_palette_new[idx], (0, 0), label='Alpha '+str(d))
            data_plot_tc.add_trajectory(atc, 'plot', '--', (0, 0), color=alpha_palette_new[idx])
    for idx, d in enumerate(alpha_doses_20181113):
        # Optional mask:
        if d not in alpha_mask:
            atc = IfnData('custom', df=olddata.data_set.loc['Alpha', d, :])
            data_plot_tc.add_trajectory(atc, 'scatter', alpha_palette_old[idx], (1, 0), label='Alpha '+str(d))
            data_plot_tc.add_trajectory(atc, 'plot', '--', (1, 0), color=alpha_palette_old[idx])

    # Beta
    for j, dose in enumerate(beta_doses_20190108):
        if dose not in beta_mask:
            btc = IfnData('custom', df=newdata.data_set.loc['Beta', dose, :])
            data_plot_tc.add_trajectory(btc, 'scatter', beta_palette_new[j], (0, 1), label='Beta '+str(dose))
            data_plot_tc.add_trajectory(btc, 'plot', '--', (0, 1), color=beta_palette_new[j])

    for j, dose in enumerate(beta_doses_20181113):
        if dose not in beta_mask:
            btc = IfnData('custom', df=olddata.data_set.loc['Beta', dose, :])
            data_plot_tc.add_trajectory(btc, 'scatter', beta_palette_old[j], (1, 1), label='Beta '+str(dose))
            data_plot_tc.add_trajectory(btc, 'plot', '--', (1, 1), color=beta_palette_old[j])

    # Add titles
    data_plot_tc.axes[0][0].set_title(r"IFN$\alpha$ new data")
    data_plot_tc.axes[0][1].set_title(r"IFN$\beta$ new data")
    data_plot_tc.axes[1][0].set_title(r"IFN$\alpha$ old data")
    data_plot_tc.axes[1][1].set_title(r"IFN$\beta$ old data")

    data_plot_tc.show_figure(save_flag=False)



