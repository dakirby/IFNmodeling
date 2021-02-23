import load_model
from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnplot import TimecoursePlot
from numpy import logspace
import seaborn as sns
import matplotlib
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
from ifnclass.ifnplot import DoseresponsePlot

# ---------------------------------------------------
# First make the figure where we increase K4
# ---------------------------------------------------
def increase_K4_figure():
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    # --------------------
    # Set up Model
    # --------------------
    Mixed_Model = load_model.load_model()
    scale_factor = load_model.scale_factor

    dose_list = list(logspace(-2, 8, num=35))

    dr_curve_a = [el[0][0] for el in Mixed_Model.mixed_dose_response([60], 'TotalpSTAT', 'Ia', dose_list,
                                                                     parameters={'Ib': 0}, sf=scale_factor).values]

    dr_curve_b = [el[0][0] for el in Mixed_Model.mixed_dose_response([60], 'TotalpSTAT', 'Ib', dose_list,
                                                                     parameters={'Ia': 0}, sf=scale_factor).values]

    # Now compute the 15* refractory response
    k4sf1 = 15
    kd4_reference = Mixed_Model.model_1.parameters['kd4']
    kd3_reference = Mixed_Model.model_1.parameters['kd3']
    k_d4_reference = Mixed_Model.model_1.parameters['k_d4']
    k_d3_reference = Mixed_Model.model_1.parameters['k_d3']
    Mixed_Model.set_global_parameters({'kd4': kd4_reference*k4sf1, 'k_d4': k_d4_reference*k4sf1})

    dr_curve_a15 = [el[0][0] for el in Mixed_Model.mixed_dose_response([60], 'TotalpSTAT', 'Ia', dose_list,
                                                                     parameters={'Ib': 0}, sf=scale_factor).values]
    dr_curve_b15 = [el[0][0] for el in Mixed_Model.mixed_dose_response([60], 'TotalpSTAT', 'Ib', dose_list,
                                                                     parameters={'Ia': 0}, sf=scale_factor).values]

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(16, 8)
    axes.set_xlabel("Dose (pM)", fontsize=14)
    axes.set_title("Relative Refractory Response", fontsize=16)
    axes.set_ylabel("pSTAT1 Relative to Primary Response", fontsize=14)
    axes.set(xscale='log', yscale='linear')

    axes.plot(dose_list, np.divide(dr_curve_a15, dr_curve_a), label=r'IFN$\alpha$ $K_{D4}\times$'+'{}'.format(k4sf1), color=alpha_palette[4], linestyle='--', linewidth=2)

    axes.plot(dose_list, np.divide(dr_curve_b15, dr_curve_b), label=r'IFN$\beta$ $K_{D4}\times$'+'{}'.format(k4sf1), color=beta_palette[4], linestyle='--', linewidth=2)

    axes.legend(loc=2, prop={'size': 8})
    fig.set_size_inches(8, 8)
    fig.savefig(os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_4', 'Figure_4_Refractoriness_By_K4.pdf'))

    # Also plot absolute curves
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[6.4, 6.4])

    axes.set_xlabel("Dose (pM)", fontsize=18)

    axes.set_ylabel("pSTAT1", fontsize=18)
    axes.set(xscale='log', yscale='linear')

    axes.plot(dose_list, dr_curve_a, label=r'IFN$\alpha$', color=alpha_palette[5], linewidth=4)
    axes.plot(dose_list, dr_curve_a15, label=r'IFN$\alpha$ $K_{D4}\times$' + '{}'.format(k4sf1), color=alpha_palette[5], linewidth=4, linestyle='dashed')

    axes.plot(dose_list, dr_curve_b, label=r'IFN$\beta$', color=beta_palette[5], linewidth=4)
    axes.plot(dose_list, dr_curve_b15, label=r'IFN$\beta$ $K_{D4}\times$' + '{}'.format(k4sf1), color=beta_palette[5], linewidth=4, linestyle='dashed')

    axes.legend(loc=2, prop={'size': 12})

    for tick in axes.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in axes.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    fig.savefig(os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_4', 'Figure_4_Refractoriness_By_K4_Absolute.pdf'))


def timecourse_figure():
    # Get experimental data
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

    # Plot
    green = sns.color_palette("deep")[2]
    red = sns.color_palette("deep")[3]
    light_green = sns.color_palette("pastel")[2]
    light_red = sns.color_palette("pastel")[3]
    plot = TimecoursePlot((1, 1))
    plot.add_trajectory(mean_data, 'errorbar', 'o--', (0, 0), label=r'10 pM IFN$\alpha$2', color=light_red, dose_species='Alpha', doseslice=10.0, alpha=0.5)
    plot.add_trajectory(mean_data, 'errorbar', 'o--', (0, 0), label=r'6 pM IFN$\beta$', color=light_green, dose_species='Beta', doseslice=6.0, alpha=0.5)
    plot.add_trajectory(mean_data, 'errorbar', 'o-', (0, 0), label=r'3000 pM IFN$\alpha$2', color=red, dose_species='Alpha', doseslice=3000.0)
    plot.add_trajectory(mean_data, 'errorbar', 'o-', (0, 0), label=r'2000 pM IFN$\beta$', color=green, dose_species='Beta', doseslice=2000.0)
    fname = os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_4', 'Timecourse.pdf')
    plot.axes.set_ylabel('pSTAT1 (MFI)')
    plot.show_figure(show_flag=False, save_flag=True, save_dir=fname)


if __name__ == '__main__':
    K4_flag = True
    TC_flag = True
    # ---------------------------------------------
    # ---------------------------------------------
    if K4_flag:
        increase_K4_figure()
    if TC_flag:
        timecourse_figure()
