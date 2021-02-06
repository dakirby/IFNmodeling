from ifnclass.ifndata import IfnData
from ifnclass.ifnfit import DualMixedPopulation
from ifnclass.ifnplot import DoseresponsePlot
from AP_AV_DATA import Thomas2011IFNalpha2AV, Thomas2011IFNalpha2YNSAV,\
 Thomas2011IFNalpha7AV, Thomas2011IFNomegaAV, Thomas2011IFNalpha2YNSAP,\
 Thomas2011IFNalpha2AP, Thomas2011IFNalpha7AP, Thomas2011IFNomegaAP
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


def antiViralActivity(pSTAT, KM):
    return np.nan_to_num(100 * pSTAT / (pSTAT + KM))


def antiProliferativeActivity(pSTAT, H, KM):
    return np.nan_to_num(100 * (pSTAT**H / (pSTAT**H + KM**H) + pSTAT**(2*H) / (pSTAT**(2*H) + (2*KM)**(2*H))) / 2)


def MSE(l1, l2):
    return np.sum(np.square(np.subtract(l1, l2))) / len(l1)


TIMES = [60]
USP18_sf = 15


def pSTAT_response(test_doses, tag):
    dir = os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_5')
    if not os.path.exists(dir):
        os.makedirs(dir)
    # --------------------
    # Set up Model
    # --------------------
    # Parameters same as Figure 2
    initial_parameters = {'k_a1': 4.98E-14 * 2, 'k_a2': 8.30e-13 * 2,
                          'k_d4': 0.006 * 3.8,
                          'kpu': 0.00095,
                          'ka2': 4.98e-13 * 2.45, 'kd4': 0.3 * 2.867,
                          'kint_a': 0.000124, 'kint_b': 0.00086,
                          'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005,
                          'krec_b2': 0.05}
    dual_parameters = {'kint_a': 0.00052, 'kSOCSon': 6e-07, 'kint_b': 0.00052,
                       'krec_a1': 0.001, 'krec_a2': 0.1,
                       'krec_b1': 0.005, 'krec_b2': 0.05}
    scale_factor = 1.227

    Mixed_Model = DualMixedPopulation('Mixed_IFN_ppCompatible', 0.8, 0.2)
    Mixed_Model.model_1.set_parameters(initial_parameters)
    Mixed_Model.model_1.set_parameters(dual_parameters)
    Mixed_Model.model_1.set_parameters({'R1': 12000.0, 'R2': 1511.1})
    Mixed_Model.model_2.set_parameters(initial_parameters)
    Mixed_Model.model_2.set_parameters(dual_parameters)
    Mixed_Model.model_2.set_parameters({'R1': 6755.56, 'R2': 1511.1})

    params = copy.deepcopy(Mixed_Model.get_parameters())
    # ----------------------------------
    # Get initial pSTAT1/2 responses
    # ----------------------------------
    # Use the fit IFNa2 parameters
    pSTAT_a2 = Mixed_Model.mixed_dose_response(TIMES, 'TotalpSTAT', 'Ia',
                                               test_doses,
                                               parameters={'Ib': 0},
                                               sf=scale_factor)
    pSTAT_a2 = np.array([el[0][0] for el in pSTAT_a2.values])

    pSTAT_a2YNS = Mixed_Model.mixed_dose_response(TIMES, 'TotalpSTAT', 'Ib',
                                                  test_doses,
                                                  parameters={'Ia': 0},
                                                  sf=scale_factor)
    pSTAT_a2YNS = np.array([el[0][0] for el in pSTAT_a2YNS.values])

    # IFNa7 has K1 and K2 half that of IFNa2  (taken from Mathematica notebook)
    kd1_a7 = params['kd1'] * 0.5
    kd2_a7 = params['kd2'] * 0.5
    IFNa7_params = {'Ib': 0, 'kd1': kd1_a7, 'kd2': kd2_a7}
    pSTAT_a7 = Mixed_Model.mixed_dose_response(TIMES, 'TotalpSTAT', 'Ia',
                                               test_doses,
                                               parameters=IFNa7_params,
                                               sf=scale_factor)
    pSTAT_a7 = np.array([el[0][0] for el in pSTAT_a7.values])

    # IFNw has K1 = 0.08 * K1 of IFNa2  and K2 = 0.4 * K2 of IFNa2
    kd1_w = params['kd1'] * 0.08
    kd2_w = params['kd2'] * 0.4
    IFNw_params = {'Ib': 0, 'kd1': kd1_w, 'kd2': kd2_w}
    pSTAT_w = Mixed_Model.mixed_dose_response(TIMES, 'TotalpSTAT', 'Ia',
                                              test_doses,
                                              parameters=IFNw_params,
                                              sf=scale_factor)
    pSTAT_w = np.array([el[0][0] for el in pSTAT_w.values])

    print("Finished simulating primary pSTAT response")

    # -----------------------------------
    # Repeat for the refractory responses
    # -----------------------------------
    kd4_a2 = params['kd4'] * USP18_sf  # K3 handled automatically by detailed balance
    IFNa2_params = {'Ib': 0, 'kd4': kd4_a2}
    pSTAT_a2_refractory = Mixed_Model.mixed_dose_response(TIMES, 'TotalpSTAT', 'Ia',
                                                          test_doses,
                                                          parameters=IFNa2_params,
                                                          sf=scale_factor)
    pSTAT_a2_refractory = np.array([el[0][0] for el in pSTAT_a2_refractory.values])

    # Use the IFNa2 kd3 parameter for IFNa2YNS
    # kd4_a2YNS = kd4_a2YNS * USP18_sf
    # IFNa2YNS_params = {'Ib': 0, 'kd4': kd4_a2YNS}
    # pSTAT_a2YNS_refractory = Mixed_Model.mixed_dose_response(TIMES, 'TotalpSTAT',
    #                                                          'Ia', test_doses,
    #                                                          parameters=IFNa2YNS_params,
    #                                                          sf=scale_factor)
    pSTAT_a2YNS_refractory = Mixed_Model.mixed_dose_response(TIMES, 'TotalpSTAT',
                                                             'Ib', test_doses,
                                                             parameters={'Ia':0, 'k_d4': params['k_d4'] * USP18_sf},
                                                             sf=scale_factor)
    pSTAT_a2YNS_refractory = np.array([el[0][0] for el in pSTAT_a2YNS_refractory.values])

    # scale kd1, kd2, and kd4 to get correct Kd for IFNa7
    kd4_a7 = USP18_sf * params['kd4']
    IFNa7_params.update({'kd4': kd4_a7})
    pSTAT_a7_refractory = Mixed_Model.mixed_dose_response(TIMES, 'TotalpSTAT', 'Ia',
                                                          test_doses,
                                                          parameters=IFNa7_params,
                                                          sf=scale_factor)
    pSTAT_a7_refractory = np.array([el[0][0] for el in pSTAT_a7_refractory.values])

    # scale kd1, kd2, and kd4 to get correct Kd for IFNw
    kd4_w = USP18_sf * params['kd4']
    IFNw_params.update({'kd4': kd4_w})
    pSTAT_w_refractory = Mixed_Model.mixed_dose_response(TIMES, 'TotalpSTAT', 'Ia',
                                                         test_doses,
                                                         parameters=IFNw_params,
                                                         sf=scale_factor)
    pSTAT_w_refractory = np.array([el[0][0] for el in pSTAT_w_refractory.values])

    print("Finished simulating refractory pSTAT response")

    # store all results in save_dir to save on computation
    np.save(dir + os.sep + 'pSTAT_a2_fitting{}.npy'.format(tag), pSTAT_a2)
    np.save(dir + os.sep + 'pSTAT_a2YNS_fitting{}.npy'.format(tag), pSTAT_a2YNS)
    np.save(dir + os.sep + 'pSTAT_a7_fitting{}.npy'.format(tag), pSTAT_a7)
    np.save(dir + os.sep + 'pSTAT_w_fitting{}.npy'.format(tag), pSTAT_w)
    np.save(dir + os.sep + 'pSTAT_a2_refractory_fitting{}.npy'.format(tag), pSTAT_a2_refractory)
    np.save(dir + os.sep + 'pSTAT_a2YNS_refractory_fitting{}.npy'.format(tag), pSTAT_a2YNS_refractory)
    np.save(dir + os.sep + 'pSTAT_a7_refractory_fitting{}.npy'.format(tag), pSTAT_a7_refractory)
    np.save(dir + os.sep + 'pSTAT_w_refractory_fitting{}.npy'.format(tag), pSTAT_w_refractory)


if __name__ == '__main__':
    simulate_pSTAT = False
    fit = True
    plot = True
    KM_AV_guess, KM_AP_guess, H_AP_guess = 4.39249, 7000., 0.75
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    AV_doses = [el[0] for el in Thomas2011IFNalpha2YNSAV]
    # include factor of 1E3 because data is in nM but model expects pM
    AP_doses = [1E3*el[0] for el in Thomas2011IFNalpha2YNSAP]
    dir = os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_5')
    if not os.path.exists(dir):
        os.makedirs(dir)

    if simulate_pSTAT:
        pSTAT_response(AV_doses, '_AV')
        pSTAT_response(AP_doses, '_AP')

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

    def function(placeholder, KM_AV, KM_AP, H_AP):
        # cost = 0
        record = []
        for i in range(4):
            AV_sim = 100 - antiViralActivity(pSTAT_AV_list[i], KM=KM_AV)
            AP_sim = 100 - antiProliferativeActivity(pSTAT_AP_list[i], H=H_AP, KM=KM_AP)
            record.append(AV_sim)
            record.append(AP_sim)
            # cost += MSE(AV_sim, AV_data[i]) + MSE(AP_sim, AP_data[i])
        return np.concatenate(record)

    if fit:
        fit_params, _ = curve_fit(function, None, ydata, bounds=([0.1, 0.1, 0.1], [1.E6, 1.E6, 1.5]))
        KM_AV_fit, KM_AP_fit, H_AP_fit = fit_params
        print(fit_params)
        np.save(dir + os.sep + 'AV_AP_fit_KMAV_KMAP_HAP.npy', fit_params)
        # optimal parameters are: 3.62373009  3.1698321   0.66073875

    if plot:
        test_doses = list(logspace(-4, 6))
        colour_palette = sns.color_palette("deep", 4)
        KM_AV_fit, KM_AP_fit, H_AP_fit = np.load(dir + os.sep + 'AV_AP_fit_KMAV_KMAP_HAP.npy')

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

        fake = 1.08
        IFNa2YNS_AP = antiProliferativeActivity(pSTAT_a2YNS_refractory, H=H_AP_fit, KM=KM_AP_fit)
        IFNa2_AP = antiProliferativeActivity(pSTAT_a2_refractory, H=H_AP_fit, KM=KM_AP_fit)
        IFNa7_AP = antiProliferativeActivity(pSTAT_a7_refractory, H=H_AP_fit, KM=KM_AP_fit)
        IFNw_AP = antiProliferativeActivity(pSTAT_w_refractory, H=H_AP_fit, KM=KM_AP_fit)

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
            axes[0].plot(test_doses, 100-sim_data[idx], color=colour_palette[idx], linewidth=3)
        # Anti-proliferative activity
        exp_data = list(map(np.array, [Thomas2011IFNalpha2AP, Thomas2011IFNalpha7AP, Thomas2011IFNomegaAP, Thomas2011IFNalpha2YNSAP]))
        sim_data = list(map(np.array, [IFNa2_AP, IFNa7_AP, IFNw_AP, IFNa2YNS_AP]))
        for idx in range(len(exp_data)):
            # include factor of 1E3 because data is in nM but axis is in pM
            axes[1].scatter(1E3*exp_data[idx][:, 0], exp_data[idx][:, 1], color=colour_palette[idx], label=labels[idx])
            axes[1].plot(np.array(test_doses), [max(0, el) for el in 100-sim_data[idx]], color=colour_palette[idx], linewidth=3)

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
