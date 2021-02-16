from ifnclass.ifndata import IfnData
from ifnclass.ifnfit import DualMixedPopulation
from ifnclass.ifnplot import DoseresponsePlot
from AP_AV_DATA import Thomas2011IFNalpha2AV, Thomas2011IFNalpha2YNSAV,\
 Thomas2011IFNalpha7AV, Thomas2011IFNomegaAV, Thomas2011IFNalpha2YNSAP,\
 Thomas2011IFNalpha2AP, Thomas2011IFNalpha7AP, Thomas2011IFNomegaAP

from figure_5_theory import antiViralActivity, antiProliferativeActivity,\
 PARAM_LOWER_BOUNDS, PARAM_UPPER_BOUNDS
from figure_5_simulations import figure_5_simulations
import os
from numpy import logspace
import copy
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.rcParams.update({'font.size': 16})


def find_nearest(array, value, idx_flag=False):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if idx_flag:
        return idx
    else:
        return array[idx]


def R2(ydata, ymodel, MSE=False):
    SStot = np.sum(np.square(ydata - np.mean(ydata)))
    SSres = np.sum(np.square(ydata - ymodel))
    if MSE:
        print("MSE is {}".format(SSres/len(ydata)))
    r_squared = 1 - (SSres / SStot)
    return r_squared


def MSE(l1, l2):
    return np.sum(np.square(np.subtract(l1, l2))) / len(l1)


TIMES = [60]
USP18_sf = 15


def pSTAT_response(test_doses, tag):
    dir = os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_5')
    if not os.path.exists(dir):
        os.makedirs(dir)
    figure_5_simulations(USP18_sf, TIMES, test_doses, dir, tag=tag)


def figure_5_fitting(simulate_pSTAT, fit, plot):
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    AV_doses = [el[0] for el in Thomas2011IFNalpha2YNSAV]
    # include factor of 1E3 because data is in nM but model expects pM
    AP_doses = [1E3*el[0] for el in Thomas2011IFNalpha2YNSAP]
    dir = os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_5')
    if not os.path.exists(dir):
        os.makedirs(dir)

    if simulate_pSTAT:
        pSTAT_response(AV_doses, '_fitting_AV')
        pSTAT_response(AP_doses, '_fitting_AP')

    pSTAT_a2_AV = np.load(dir + os.sep + 'pSTAT_a2_fitting_AV.npy')
    pSTAT_a2YNS_AV = np.load(dir + os.sep + 'pSTAT_a2YNS_fitting_AV.npy')
    pSTAT_a7_AV = np.load(dir + os.sep + 'pSTAT_a7_fitting_AV.npy')
    pSTAT_w_AV = np.load(dir + os.sep + 'pSTAT_w_fitting_AV.npy')

    pSTAT_a2_refractory_AP = np.load(dir + os.sep + 'pSTAT_a2_refractory_fitting_AP.npy')
    pSTAT_a2YNS_refractory_AP = np.load(dir + os.sep + 'pSTAT_a2YNS_refractory_fitting_AP.npy')
    pSTAT_a7_refractory_AP = np.load(dir + os.sep + 'pSTAT_a7_refractory_fitting_AP.npy')
    pSTAT_w_refractory_AP = np.load(dir + os.sep + 'pSTAT_w_refractory_fitting_AP.npy')

    pSTAT_AV_list = [pSTAT_a2_AV, pSTAT_a2YNS_AV, pSTAT_a7_AV, pSTAT_w_AV]
    AV_data = [np.array(el) for el in [Thomas2011IFNalpha2AV, Thomas2011IFNalpha2YNSAV, Thomas2011IFNalpha7AV, Thomas2011IFNomegaAV]]
    pSTAT_AP_list = [pSTAT_a2YNS_refractory_AP, pSTAT_a2_refractory_AP, pSTAT_a7_refractory_AP, pSTAT_w_refractory_AP]
    AP_data = [np.array(el) for el in [Thomas2011IFNalpha2YNSAP, Thomas2011IFNalpha2AP, Thomas2011IFNalpha7AP, Thomas2011IFNomegaAP]]

    ydata = []
    for i in range(4):
        ydata.append(AV_data[i][:, 1])
        ydata.append(AP_data[i][:, 1])
    ydata = np.concatenate(ydata)

    def function(placeholder, KM_AV, KM, L, c, n):
        params = [KM_AV, KM, L, c, n]
        record = []
        for i in range(4):
            AV_sim = 100 - antiViralActivity(pSTAT_AV_list[i], KM=params[0])
            AP_sim = 100 - antiProliferativeActivity(pSTAT_AP_list[i], *params[1:])
            record.append(AV_sim)
            record.append(AP_sim)
        return np.concatenate(record)

    if fit:
        fit_params, _ = curve_fit(function, None, ydata, bounds=(PARAM_LOWER_BOUNDS, PARAM_UPPER_BOUNDS))
        print(fit_params)
        KM_AV_fit = fit_params[0]
        AP_fit = fit_params[1:]
        np.save(dir + os.sep + 'AV_AP_fit_params.npy', fit_params)

        # get R squared value for fit parameters
        r2 = R2(ydata, function(None, *fit_params), MSE=True)
        print("R squared value of fit is {:.2f}".format(r2))

    if plot:
        colour_palette = sns.color_palette("deep", 4)
        fit_params = np.load(dir + os.sep + 'AV_AP_fit_params.npy')
        KM_AV_fit = fit_params[0]
        AP_fit = fit_params[1:]

        try:
            sim_doses = np.load(dir + os.sep + 'doses.npy')
        except FileNotFoundError:
            sim_doses = list(logspace(-4, 6))

        pSTAT_a2 = np.load(dir + os.sep + 'pSTAT_a2.npy')
        pSTAT_a2YNS = np.load(dir + os.sep + 'pSTAT_a2YNS.npy')
        pSTAT_a7 = np.load(dir + os.sep + 'pSTAT_a7.npy')
        pSTAT_w = np.load(dir + os.sep + 'pSTAT_w.npy')
        pSTAT_a2_refractory = np.load(dir + os.sep + 'pSTAT_a2_refractory.npy')
        pSTAT_a2YNS_refractory = np.load(dir + os.sep + 'pSTAT_a2YNS_refractory.npy')
        pSTAT_a7_refractory = np.load(dir + os.sep + 'pSTAT_a7_refractory.npy')
        pSTAT_w_refractory = np.load(dir + os.sep + 'pSTAT_w_refractory.npy')
        # ------------------------------------------------
        # Get anti-viral and anti-proliferative responses
        # ------------------------------------------------
        IFNa2_AV = antiViralActivity(pSTAT_a2, KM=KM_AV_fit)
        IFNa2YNS_AV = antiViralActivity(pSTAT_a2YNS, KM=KM_AV_fit)
        IFNa7_AV = antiViralActivity(pSTAT_a7, KM=KM_AV_fit)
        IFNw_AV = antiViralActivity(pSTAT_w, KM=KM_AV_fit)

        IFNa2YNS_AP = antiProliferativeActivity(pSTAT_a2YNS_refractory, *AP_fit)
        IFNa2_AP = antiProliferativeActivity(pSTAT_a2_refractory, *AP_fit)
        IFNa7_AP = antiProliferativeActivity(pSTAT_a7_refractory, *AP_fit)
        IFNw_AP = antiProliferativeActivity(pSTAT_w_refractory, *AP_fit)

        # ------------------------
        # Plot fit to Thomas 2011
        # ------------------------
        colour_palette = sns.color_palette("deep", 4)
        labels = [r"IFN$\alpha$2", r"IFN$\alpha$7", r"IFN$\omega$", r"IFN$\alpha$2-YNS"]
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12., 5.))
        axes[0].set_xscale('log')
        axes[1].set_xscale('log')
        # Anti-viral activity
        exp_data = list(map(np.array, [Thomas2011IFNalpha2AV, Thomas2011IFNalpha7AV, Thomas2011IFNomegaAV, Thomas2011IFNalpha2YNSAV]))
        sim_data = list(map(np.array, [IFNa2_AV, IFNa7_AV, IFNw_AV, IFNa2YNS_AV]))
        for idx in range(len(exp_data)):
            axes[0].scatter(exp_data[idx][:, 0], exp_data[idx][:, 1], color=colour_palette[idx], label=labels[idx])
            axes[0].plot(sim_doses, 100-sim_data[idx], color=colour_palette[idx], linewidth=3)
        # Anti-proliferative activity
        exp_data = list(map(np.array, [Thomas2011IFNalpha2AP, Thomas2011IFNalpha7AP, Thomas2011IFNomegaAP, Thomas2011IFNalpha2YNSAP]))
        sim_data = list(map(np.array, [IFNa2_AP, IFNa7_AP, IFNw_AP, IFNa2YNS_AP]))
        for idx in range(len(exp_data)):
            # include factor of 1E3 because data is in nM but axis is in pM
            axes[1].scatter(1E3*exp_data[idx][:, 0], exp_data[idx][:, 1], color=colour_palette[idx], label=labels[idx])
            axes[1].plot(sim_doses, [max(0, el) for el in 100-sim_data[idx]], color=colour_palette[idx], linewidth=3)

        axes[0].set_title('Anti-viral activity assay')
        axes[0].set_ylabel('HCV Replication (%)')
        axes[1].set_title('Anti-proliferative activity assay')
        axes[1].set_ylabel('Relative Cell Density (%)')
        axes[0].legend()  # more space in the AV plot for a legend
        axes[0].set_xlim(left=1E-4, right=3E2)
        axes[1].set_xlim(left=1E-1, right=1E5)
        for ax in axes:
            ax.set_xscale('log')
            ax.set_xlabel('[IFN] (pM)')

        # save figure
        plt.tight_layout()
        fig.savefig(os.path.join(dir, 'Figure_5_fit.pdf'))


if __name__ == '__main__':
    simulate_pSTAT = True
    fit = True
    plot = True
    figure_5_fitting(simulate_pSTAT, fit, plot)
