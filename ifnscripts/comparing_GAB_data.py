from ifnclass.ifndata import IfnData
from ifnclass.ifnplot import DoseresponsePlot, TimecoursePlot
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    #olddata = IfnData("20181113_B6_IFNs_Dose_Response_Bcells")
    newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
    #newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
    newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
    newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")


    times = [2.5, 5, 7.5, 10, 20, 60]
    alpha_doses = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses = [0, 0.2, 6, 20, 60, 200, 600, 2000]
    #beta_doses_20181113 = [0.1, 1, 5, 10, 100, 200, 1000]
    #alpha_doses_20181113 = [5, 50, 250, 500, 5000, 25000, 50000]

    alpha_palette_new = sns.color_palette("Reds", 6)
    beta_palette_new = sns.color_palette("Greens", 6)
    alpha_palette_old = sns.color_palette("Reds", 6)
    beta_palette_old = sns.color_palette("Greens", 6)

    data_plot_dr = DoseresponsePlot((3, 2))
    alpha_mask = []
    beta_mask = []

    # Add data
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            data_plot_dr.add_trajectory(newdata_4, t, 'scatter', alpha_palette_new[idx], (2, 0), 'Alpha', dn=1)
            data_plot_dr.add_trajectory(newdata_4, t, 'plot', '--', (2, 0), 'Alpha', dn=1, color=alpha_palette_new[idx], label='Alpha ' + str(t))
            data_plot_dr.add_trajectory(newdata_3, t, 'scatter', alpha_palette_new[idx], (1, 0), 'Alpha', dn=1)
            data_plot_dr.add_trajectory(newdata_3, t, 'plot', '--', (1, 0), 'Alpha', dn=1, color=alpha_palette_new[idx], label='Alpha ' + str(t))
            #data_plot_dr.add_trajectory(newdata_2, t, 'scatter', alpha_palette_new[idx], (1, 0), 'Alpha', dn=1)
            #data_plot_dr.add_trajectory(newdata_2, t, 'plot', '--', (1, 0), 'Alpha', dn=1, color=alpha_palette_new[idx], label='Alpha ' + str(t))
            data_plot_dr.add_trajectory(newdata_1, t, 'scatter', alpha_palette_new[idx], (0, 0), 'Alpha', dn=1)
            data_plot_dr.add_trajectory(newdata_1, t, 'plot', '--', (0, 0), 'Alpha', dn=1, color=alpha_palette_new[idx], label='Alpha ' + str(t))
            #data_plot_dr.add_trajectory(olddata, t, 'scatter', alpha_palette_old[idx], (0, 0), 'Alpha', dn=1)
            #data_plot_dr.add_trajectory(olddata, t, 'plot', '--', (0, 0), 'Alpha', dn=1, color=alpha_palette_old[idx], label='Alpha ' + str(t))

        if t not in beta_mask:
            data_plot_dr.add_trajectory(newdata_4, t, 'scatter', beta_palette_new[idx], (2, 1), 'Beta', dn=1)
            data_plot_dr.add_trajectory(newdata_4, t, 'plot', '--', (2, 1), 'Beta', dn=1, color=beta_palette_new[idx], label='Beta ' + str(t))
            data_plot_dr.add_trajectory(newdata_3, t, 'scatter', beta_palette_new[idx], (1, 1), 'Beta', dn=1)
            data_plot_dr.add_trajectory(newdata_3, t, 'plot', '--', (1, 1), 'Beta', dn=1, color=beta_palette_new[idx], label='Beta ' + str(t))
            #data_plot_dr.add_trajectory(newdata_2, t, 'scatter', beta_palette_new[idx], (1, 1), 'Beta', dn=1)
            #data_plot_dr.add_trajectory(newdata_2, t, 'plot', '--', (1, 1), 'Beta', dn=1, color=beta_palette_new[idx], label='Beta ' + str(t))
            data_plot_dr.add_trajectory(newdata_1, t, 'scatter', beta_palette_new[idx], (0, 1), 'Beta', dn=1)
            data_plot_dr.add_trajectory(newdata_1, t, 'plot', '--', (0, 1), 'Beta', dn=1, color=beta_palette_new[idx], label='Beta ' + str(t))
            #data_plot_dr.add_trajectory(olddata, t, 'scatter', beta_palette_old[idx], (0, 1), 'Beta', dn=1)
            #data_plot_dr.add_trajectory(olddata, t, 'plot', '--', (0, 1), 'Beta', dn=1, color=beta_palette_old[idx], label='Beta ' + str(t))

    #data_plot_dr.axes[0][0].set_title(r"IFN$\alpha$ old data")
    #data_plot_dr.axes[0][1].set_title(r"IFN$\beta$ old data")
    data_plot_dr.axes[0][0].set_title(r"IFN$\alpha$ 2019-01-08 data")
    data_plot_dr.axes[0][1].set_title(r"IFN$\beta$ 2019-01-08 data")
    #data_plot_dr.axes[1][0].set_title(r"IFN$\alpha$ 2019-01-19 data")
    #data_plot_dr.axes[1][1].set_title(r"IFN$\beta$ 2019-01-19 data")
    data_plot_dr.axes[1][0].set_title(r"IFN$\alpha$ 2019-01-21 data")
    data_plot_dr.axes[1][1].set_title(r"IFN$\beta$ 2019-01-21 data")
    data_plot_dr.axes[2][0].set_title(r"IFN$\alpha$ 2019-02-14 data")
    data_plot_dr.axes[2][1].set_title(r"IFN$\beta$ 2019-02-14 data")

    dr_fig, dr_axes = data_plot_dr.show_figure(save_flag=False)
    dr_fig.set_size_inches(10.5, 21.4)
    dr_fig.savefig('results/GAB_NewData/compare_data_dr.pdf')

    # -------------------------
    # Time courses
    # -------------------------
    alpha_palette_new = sns.color_palette("Reds", 8)
    beta_palette_new = sns.color_palette("Greens", 8)
    alpha_palette_old = sns.color_palette("Reds", 8)
    beta_palette_old = sns.color_palette("Greens", 8)

    data_plot_tc = TimecoursePlot((3, 2))
    alpha_mask = [0]
    beta_mask = [0]
    # Add data

    # Alpha
    for idx, d in enumerate(alpha_doses):
        # Optional mask:
        if d not in alpha_mask:
            atc = IfnData('custom', df=newdata_4.data_set.loc['Alpha', d, :])
            data_plot_tc.add_trajectory(atc, 'scatter', alpha_palette_new[idx], (2, 0), label='Alpha '+str(d))
            data_plot_tc.add_trajectory(atc, 'plot', '--', (2, 0), color=alpha_palette_new[idx])

    # alpha_doses_20190121
    for idx, d in enumerate(alpha_doses):
        # Optional mask:
        if d not in alpha_mask:
            atc = IfnData('custom', df=newdata_3.data_set.loc['Alpha', d, :])
            data_plot_tc.add_trajectory(atc, 'scatter', alpha_palette_new[idx], (1, 0), label='Alpha '+str(d))
            data_plot_tc.add_trajectory(atc, 'plot', '--', (1, 0), color=alpha_palette_new[idx])
    """
    # alpha_doses_20190119
    for idx, d in enumerate(alpha_doses):
        # Optional mask:
        if d not in alpha_mask:
            atc = IfnData('custom', df=newdata_2.data_set.loc['Alpha', d, :])
            data_plot_tc.add_trajectory(atc, 'scatter', alpha_palette_new[idx], (1, 0), label='Alpha '+str(d))
            data_plot_tc.add_trajectory(atc, 'plot', '--', (1, 0), color=alpha_palette_new[idx])
    """
    # alpha_doses_20190108
    for idx, d in enumerate(alpha_doses):
        # Optional mask:
        if d not in alpha_mask:
            atc = IfnData('custom', df=newdata_1.data_set.loc['Alpha', d, :])
            data_plot_tc.add_trajectory(atc, 'scatter', alpha_palette_new[idx], (0, 0), label='Alpha '+str(d))
            data_plot_tc.add_trajectory(atc, 'plot', '--', (0, 0), color=alpha_palette_new[idx])
    # original
    #for idx, d in enumerate(alpha_doses_20181113):
    #    # Optional mask:
    #    if d not in alpha_mask:
    #        atc = IfnData('custom', df=olddata.data_set.loc['Alpha', d, :])
    #        data_plot_tc.add_trajectory(atc, 'scatter', alpha_palette_old[idx], (0, 0), label='Alpha '+str(d))
    #        data_plot_tc.add_trajectory(atc, 'plot', '--', (0, 0), color=alpha_palette_old[idx])

    # Beta
    for j, dose in enumerate(beta_doses):
        if dose not in beta_mask:
            btc = IfnData('custom', df=newdata_4.data_set.loc['Beta', dose, :])
            data_plot_tc.add_trajectory(btc, 'scatter', beta_palette_new[j], (2, 1), label='Beta '+str(dose))
            data_plot_tc.add_trajectory(btc, 'plot', '--', (2, 1), color=beta_palette_new[j])
    # beta_doses_20190121
    for j, dose in enumerate(beta_doses):
        if dose not in beta_mask:
            btc = IfnData('custom', df=newdata_3.data_set.loc['Beta', dose, :])
            data_plot_tc.add_trajectory(btc, 'scatter', beta_palette_new[j], (1, 1), label='Beta '+str(dose))
            data_plot_tc.add_trajectory(btc, 'plot', '--', (1, 1), color=beta_palette_new[j])
    """
    # beta_doses_20190119
    for j, dose in enumerate(beta_doses):
        if dose not in beta_mask:
            btc = IfnData('custom', df=newdata_2.data_set.loc['Beta', dose, :])
            data_plot_tc.add_trajectory(btc, 'scatter', beta_palette_new[j], (1, 1), label='Beta '+str(dose))
            data_plot_tc.add_trajectory(btc, 'plot', '--', (1, 1), color=beta_palette_new[j])
    """
    # beta_doses_20190108
    for j, dose in enumerate(beta_doses):
        if dose not in beta_mask:
            btc = IfnData('custom', df=newdata_1.data_set.loc['Beta', dose, :])
            data_plot_tc.add_trajectory(btc, 'scatter', beta_palette_new[j], (0, 1), label='Beta '+str(dose))
            data_plot_tc.add_trajectory(btc, 'plot', '--', (0, 1), color=beta_palette_new[j])
    # original
    #for j, dose in enumerate(beta_doses_20181113):
    #    if dose not in beta_mask:
    #        btc = IfnData('custom', df=olddata.data_set.loc['Beta', dose, :])
    #        data_plot_tc.add_trajectory(btc, 'scatter', beta_palette_old[j], (0, 1), label='Beta '+str(dose))
    #        data_plot_tc.add_trajectory(btc, 'plot', '--', (0, 1), color=beta_palette_old[j])

    # Add titles
    data_plot_tc.axes[2][0].set_title(r"IFN$\alpha$ 2019-02-14 data")
    data_plot_tc.axes[2][1].set_title(r"IFN$\beta$ 2019-02-14 data")
    data_plot_tc.axes[1][0].set_title(r"IFN$\alpha$ 2019-01-21 data")
    data_plot_tc.axes[1][1].set_title(r"IFN$\beta$ 2019-01-21 data")
    #data_plot_tc.axes[1][0].set_title(r"IFN$\alpha$ 2019-01-19 data")
    #data_plot_tc.axes[1][1].set_title(r"IFN$\beta$ 2019-01-19 data")
    data_plot_tc.axes[0][0].set_title(r"IFN$\alpha$ 2019-01-08 data")
    data_plot_tc.axes[0][1].set_title(r"IFN$\beta$ 2019-01-08 data")
    #data_plot_tc.axes[0][0].set_title(r"IFN$\alpha$ old data")
    #data_plot_tc.axes[0][1].set_title(r"IFN$\beta$ old data")

    tc_fig, tc_axes = data_plot_tc.show_figure(save_flag=False)
    tc_fig.set_size_inches(10.5, 21.4)
    tc_fig.savefig('results/GAB_NewData/compare_data_tc.pdf')
