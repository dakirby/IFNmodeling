from ifnclass.ifndata import IfnData
from ifnclass.ifnfit import DualMixedPopulation
from ifnclass.ifnplot import DoseresponsePlot
from AP_AV_Bar_Chart import plot_barchart
import os
from numpy import logspace
import copy
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from AP_AV_DATA import Thomas2011IFNalpha2AV, Thomas2011IFNalpha2YNSAV,\
 Thomas2011IFNalpha7AV, Thomas2011IFNomegaAV, Thomas2011IFNalpha2YNSAP,\
 Thomas2011IFNalpha2AP, Thomas2011IFNalpha7AP, Thomas2011IFNomegaAP,\
 Schreiber2017AV, Schreiber2017AP

plt.rcParams.update({'font.size': 16})


def find_nearest(array, value, idx_flag=False):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if idx_flag:
        return idx
    else:
        return array[idx]


def antiViralActivity(pSTAT, KM=4.39249):
    return 100 * pSTAT / (pSTAT + KM)


def antiProliferativeActivity(pSTAT, KM1, KM2):
    H1 = 2
    H2 = 4
    return np.nan_to_num(100 * (pSTAT**H1 / (pSTAT**H1 + KM1**H1) + pSTAT**H2 / (pSTAT**H2 + KM2**H2)) / 2)


if __name__ == '__main__':
    simulate_DR = False
    simulate_scaling = False
    plot_scaling = False

    USP18_sf = 15

    times = [60]
    test_doses = list(logspace(-4, 6))  # used if simulate_DR
    test_affinities = list(logspace(-2, 4, num=10))  # used if simulate_scaling
    reference_affinity_idx = find_nearest(test_affinities, 1.0, idx_flag=True)
    assert(test_affinities[reference_affinity_idx] == 1.0)
    typicalDose = 1  # (pM), to use for plotting how activity scales with K1

    # --------------------
    # Set up Model
    # --------------------
    dir = os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_5')
    if not os.path.exists(dir):
        os.makedirs(dir)

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

    if simulate_DR:
        # ----------------------------------
        # Get initial pSTAT1/2 responses
        # ----------------------------------
        # Use the fit IFNa2 parameters
        pSTAT_a2 = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia',
                                                   test_doses,
                                                   parameters={'Ib': 0},
                                                   sf=scale_factor)
        pSTAT_a2 = np.array([el[0][0] for el in pSTAT_a2.values])

        # IFNa2YNS has K1 = 0.03 / 5 * K1, K2 = 1.5 / 5 * K2, K4=1/60 * K4 of IFNa2
        # kd1_a2YNS = params['kd1'] * 0.03 / 5.
        # kd2_a2YNS = params['kd2'] * 1.5 / 5.
        # kd4_a2YNS = params['kd4'] / 60.
        # IFNa2YNS_params = {'Ib': 0, 'kd1': kd1_a2YNS, 'kd2': kd2_a2YNS, 'kd4': kd4_a2YNS}
        # pSTAT_a2YNS = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia',
        #                                               test_doses,
        #                                               parameters=IFNa2YNS_params,
        #                                               sf=scale_factor)
        pSTAT_a2YNS = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ib',
                                                      test_doses,
                                                      parameters={'Ia': 0},
                                                      sf=scale_factor)
        pSTAT_a2YNS = np.array([el[0][0] for el in pSTAT_a2YNS.values])

        # IFNa7 has K1 and K2 half that of IFNa2  (taken from Mathematica notebook)
        kd1_a7 = params['kd1'] * 0.5
        kd2_a7 = params['kd2'] * 0.5
        IFNa7_params = {'Ib': 0, 'kd1': kd1_a7, 'kd2': kd2_a7}
        pSTAT_a7 = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia',
                                                   test_doses,
                                                   parameters=IFNa7_params,
                                                   sf=scale_factor)
        pSTAT_a7 = np.array([el[0][0] for el in pSTAT_a7.values])

        # IFNw has K1 = 0.08 * K1 of IFNa2  and K2 = 0.4 * K2 of IFNa2
        kd1_w = params['kd1'] * 0.08
        kd2_w = params['kd2'] * 0.4
        IFNw_params = {'Ib': 0, 'kd1': kd1_w, 'kd2': kd2_w}
        pSTAT_w = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia',
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
        pSTAT_a2_refractory = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia',
                                                              test_doses,
                                                              parameters=IFNa2_params,
                                                              sf=scale_factor)
        pSTAT_a2_refractory = np.array([el[0][0] for el in pSTAT_a2_refractory.values])

        # Use the IFNa2 kd3 parameter for IFNa2YNS
        # kd4_a2YNS = kd4_a2YNS * USP18_sf
        # IFNa2YNS_params = {'Ib': 0, 'kd4': kd4_a2YNS}
        # pSTAT_a2YNS_refractory = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT',
        #                                                          'Ia', test_doses,
        #                                                          parameters=IFNa2YNS_params,
        #                                                          sf=scale_factor)
        pSTAT_a2YNS_refractory = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT',
                                                                 'Ib', test_doses,
                                                                 parameters={'Ia':0, 'k_d4': params['k_d4'] * USP18_sf},
                                                                 sf=scale_factor)
        pSTAT_a2YNS_refractory = np.array([el[0][0] for el in pSTAT_a2YNS_refractory.values])

        # scale kd1, kd2, and kd4 to get correct Kd for IFNa7
        kd4_a7 = USP18_sf * params['kd4']
        IFNa7_params.update({'kd4': kd4_a7})
        pSTAT_a7_refractory = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia',
                                                              test_doses,
                                                              parameters=IFNa7_params,
                                                              sf=scale_factor)
        pSTAT_a7_refractory = np.array([el[0][0] for el in pSTAT_a7_refractory.values])

        # scale kd1, kd2, and kd4 to get correct Kd for IFNw
        kd4_w = USP18_sf * params['kd4']
        IFNw_params.update({'kd4': kd4_w})
        pSTAT_w_refractory = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia',
                                                             test_doses,
                                                             parameters=IFNw_params,
                                                             sf=scale_factor)
        pSTAT_w_refractory = np.array([el[0][0] for el in pSTAT_w_refractory.values])

        print("Finished simulating refractory pSTAT response")

        # store all results in save_dir to save on computation
        np.save(dir + os.sep + 'pSTAT_a2.npy', pSTAT_a2)
        np.save(dir + os.sep + 'pSTAT_a2YNS.npy', pSTAT_a2YNS)
        np.save(dir + os.sep + 'pSTAT_a7.npy', pSTAT_a7)
        np.save(dir + os.sep + 'pSTAT_w.npy', pSTAT_w)
        np.save(dir + os.sep + 'pSTAT_a2_refractory.npy', pSTAT_a2_refractory)
        np.save(dir + os.sep + 'pSTAT_a2YNS_refractory.npy', pSTAT_a2YNS_refractory)
        np.save(dir + os.sep + 'pSTAT_a7_refractory.npy', pSTAT_a7_refractory)
        np.save(dir + os.sep + 'pSTAT_w_refractory.npy', pSTAT_w_refractory)

    else:
        pSTAT_a2 = np.load(dir + os.sep + 'pSTAT_a2.npy')
        pSTAT_a2YNS = np.load(dir + os.sep + 'pSTAT_a2YNS.npy')
        pSTAT_a7 = np.load(dir + os.sep + 'pSTAT_a7.npy')
        pSTAT_w = np.load(dir + os.sep + 'pSTAT_w.npy')
        pSTAT_a2_refractory = np.load(dir + os.sep + 'pSTAT_a2_refractory.npy')
        pSTAT_a2YNS_refractory = np.load(dir + os.sep + 'pSTAT_a2YNS_refractory.npy')
        pSTAT_a7_refractory = np.load(dir + os.sep + 'pSTAT_a7_refractory.npy')
        pSTAT_w_refractory = np.load(dir + os.sep + 'pSTAT_w_refractory.npy')

    if simulate_scaling:
        typicalDose = find_nearest(test_doses, typicalDose)
        print("Using typical dose of {} pM".format(typicalDose))
        AV_EC50_record = []
        AP_EC50_record = []
        AV_typical_record = []
        AP_typical_record = []
        AV_typical_ref = 0
        AP_typical_ref = 0
        # ----------------------------------------------------------------------
        # Get anti-viral and anti-proliferative EC50 for a variety of K1 and K4
        # ----------------------------------------------------------------------
        for affinity_idx, i in enumerate(test_affinities):
            # Simulate dose-response curve
            test_params = {'Ib': 0, 'kd1': params['kd1'] / i, 'kd4': params['kd4'] / i}
            pSTAT_primary = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT',
                                                            'Ia', test_doses,
                                                            parameters=test_params,
                                                            sf=scale_factor)
            pSTAT_primary = np.array([el[0][0] for el in pSTAT_primary.values])
            AV_dose_response = antiViralActivity(pSTAT_primary)
            AV_df = pd.DataFrame.from_dict({'Dose_Species': ['Alpha']*len(AV_dose_response),
                                            'Dose (pM)': test_doses,
                                            times[0]: AV_dose_response})
            AV_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
            AV_data = IfnData('custom', df=AV_df, conditions={'Alpha': {'Ib': 0}})

            test_params.update({'kd4': USP18_sf * params['kd4'] / i})
            pSTAT_refractory = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT',
                                                            'Ia', test_doses,
                                                            parameters=test_params,
                                                            sf=scale_factor)
            pSTAT_refractory = np.array([el[0][0] for el in pSTAT_refractory.values])
            AP_dose_response = antiProliferativeActivity(pSTAT_refractory)
            AP_df = pd.DataFrame.from_dict({'Dose_Species': ['Alpha']*len(AP_dose_response),
                                            'Dose (pM)': test_doses,
                                            times[0]: AP_dose_response})
            AP_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
            AP_data = IfnData('custom', df=AP_df, conditions={'Alpha': {'Ib': 0}})

            # Get EC50
            AV_EC50 = AV_data.get_ec50s()['Alpha'][0][1] # species: DR curve, (time, EC50)
            AP_EC50 = AP_data.get_ec50s()['Alpha'][0][1]

            # Add to record
            AV_EC50_record.append([i, AV_EC50])
            AP_EC50_record.append([i, AP_EC50])

            # Add response at typicalDose
            AV_typical_record.append([i, AV_data.data_set.loc['Alpha', typicalDose][60]])
            AP_typical_record.append([i, AP_data.data_set.loc['Alpha', typicalDose][60]])
            if affinity_idx == reference_affinity_idx:
                AV_typical_ref = AV_data.data_set.loc['Alpha', typicalDose][60]
                AP_typical_ref = AP_data.data_set.loc['Alpha', typicalDose][60]

        # Make typical response relative to reference response
        for i in range(len(AV_typical_record)):
            AV_typical_record[i][1] = AV_typical_record[i][1] / AV_typical_ref
            AP_typical_record[i][1] = AP_typical_record[i][1] / AP_typical_ref

        # Save to dir
        np.save(dir + os.sep + 'AV_EC50.npy', np.array(AV_EC50_record))
        np.save(dir + os.sep + 'AP_EC50.npy', np.array(AP_EC50_record))
        np.save(dir + os.sep + 'AV_typical.npy', np.array(AV_typical_record))
        np.save(dir + os.sep + 'AP_typcial.npy', np.array(AP_typical_record))

    else:
        AV_EC50_record = np.array(np.load(dir + os.sep + 'AV_EC50.npy'))
        AP_EC50_record = np.array(np.load(dir + os.sep + 'AP_EC50.npy'))
        AV_typical_record = np.array(np.load(dir + os.sep + 'AV_typical.npy'))
        AP_typical_record = np.array(np.load(dir + os.sep + 'AP_typcial.npy'))

    # ------------------------------------------------
    # Get anti-viral and anti-proliferative responses
    # ------------------------------------------------
    KM_AV_fit, KM1_AP_fit, KM2_AP_fit = np.load(dir + os.sep + 'AV_AP_fit_params.npy')

    IFNa2_AV = antiViralActivity(pSTAT_a2, KM=KM_AV_fit)
    IFNa2YNS_AV = antiViralActivity(pSTAT_a2YNS, KM=KM_AV_fit)
    IFNa7_AV = antiViralActivity(pSTAT_a7, KM=KM_AV_fit)
    IFNw_AV = antiViralActivity(pSTAT_w, KM=KM_AV_fit)

    IFNa2YNS_AP = antiProliferativeActivity(pSTAT_a2YNS_refractory, KM1=KM1_AP_fit, KM2=KM2_AP_fit)
    IFNa2_AP = antiProliferativeActivity(pSTAT_a2_refractory, KM1=KM1_AP_fit, KM2=KM2_AP_fit)
    IFNa7_AP = antiProliferativeActivity(pSTAT_a7_refractory, KM1=KM1_AP_fit, KM2=KM2_AP_fit)
    IFNw_AP = antiProliferativeActivity(pSTAT_w_refractory, KM1=KM1_AP_fit, KM2=KM2_AP_fit)

    # ------------------------------------------------------------------------
    # Set up plot
    # ------------------------------------------------------------------------
    if plot_scaling:
        layout_aspect = (8.5, 11.)
        layout_scale = 1.25
    else:
        layout_aspect = (8.5, 2/3 * 11.)
        layout_scale = 1.25
    dim = tuple((el * layout_scale for el in layout_aspect))
    fig = plt.figure(figsize=dim)
    if plot_scaling:
        gs = gridspec.GridSpec(nrows=3, ncols=4)
    else:
        gs = gridspec.GridSpec(nrows=2, ncols=4)
    panelA = fig.add_subplot(gs[0, 0:3])
    A_legend = fig.add_subplot(gs[0, 3])
    panelB = fig.add_subplot(gs[1, 0:2])
    panelC = fig.add_subplot(gs[1, 2:])
    if plot_scaling:
        panelD = fig.add_subplot(gs[2, 0:2])
        panelE = fig.add_subplot(gs[2, 2:])
    # fig.delaxes(all_axes[1][2]) # odd number of panels

    # --------------------------------------
    # Plot bar chart of EC50 vs IFN mutants
    # --------------------------------------
    plot_barchart(panelA)
    plt.setp(panelA.xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    handles, labels = panelA.get_legend_handles_labels()  # get labels and handles
    A_legend.legend(handles, labels)
    A_legend.axis('off')
    panelA.get_legend().remove()  # now turn legend off in Panel A

    # ------------------------
    # Plot fit to Thomas 2011
    # ------------------------
    colour_palette = sns.color_palette("deep", 4)
    labels = [r"IFN$\alpha$2", r"IFN$\alpha$7", r"IFN$\omega$", r"IFN$\alpha$2-YNS"]
    axes = [panelB, panelC]  # all_axes[0][1:]  # top row of plot, second and third panels
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
        axes[1].plot(test_doses, [max(0, el) for el in 100-sim_data[idx]], color=colour_palette[idx], linewidth=3)

    axes[0].set_title('Anti-viral activity assay')
    axes[0].set_ylabel('HCV Replication (%)')
    axes[1].set_title('Anti-proliferative activity assay')
    axes[1].set_ylabel('Relative Cell Density (%)')
    axes[0].legend()  # more space in the AV plot for a legend
    axes[0].set_xlim(left=1E-4, right=3E2)
    axes[1].set_xlim(left=1E-2, right=3E5)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel('[IFN] (pM)')

    # -----------------------------------
    # Plot scaling of AV and AP activity
    # -----------------------------------
    if plot_scaling:
        red = sns.color_palette("tab10")[3]
        blue = sns.color_palette("tab10")[0]

        # EC50 scaling
        ax = panelD  # all_axes[1][0]
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Binding Affinity\n' + r'(Relative to IFN$\alpha$2)')
        ax.set_ylabel(r'$EC_{50}$ (Relative to IFN$\alpha$2)')
        # ax.scatter(Schreiber2017AV[:,0], Schreiber2017AV[:,1], color=red, label='Anti-viral')
        # ax.scatter(Schreiber2017AP[:,0], Schreiber2017AP[:,1], color=blue, label='Anti-proliferative')
        ax.plot(AV_EC50_record[:, 0], (AV_EC50_record[:, 1]/AV_EC50_record[reference_affinity_idx, 1]), color=red, linewidth=3, label='Anti-viral')
        ax.plot(AP_EC50_record[:, 0], (AP_EC50_record[:, 1]/AP_EC50_record[reference_affinity_idx, 1]), color=blue, linewidth=3, label='Anti-proliferative')
        ax.legend()

        # Typical response scaling
        ax = panelE  # all_axes[1][1]
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Binding Affinity\n' + r'(Relative to IFN$\alpha$2)')
        ax.set_ylabel('Biological Activity\n' + r'(Relative to IFN$\alpha$2)')
        ax.plot(AV_typical_record[:, 0], AV_typical_record[:, 1], color=red, linewidth=3)
        ax.plot(AP_typical_record[:, 0], AP_typical_record[:, 1], color=blue, linewidth=3)

    # -----------------------------------
    # save figure
    # -----------------------------------
    plt.tight_layout()
    fig.savefig(os.path.join(dir, 'Figure_5.pdf'))
