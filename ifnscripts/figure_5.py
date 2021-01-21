from ifnclass.ifndata import IfnData
from ifnclass.ifnfit import DualMixedPopulation
from ifnclass.ifnplot import DoseresponsePlot
import os
from numpy import logspace
import copy
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})


def antiViralActivity(pSTAT, KM=4.39249):
    return 100 * pSTAT / (pSTAT + KM)


def antiProliferativeActivity(pSTAT, H=0.75, KM=7000.):
    return 100 * pSTAT**H / (pSTAT**H + KM**H)


Thomas2011IFNalpha2AP = [[0.00001, 100], [0.0000296, 100], [0.0000873, 100],
                         [0.000258, 100], [0.000763, 100], [0.00225, 88],
                         [0.00666, 95], [0.0197, 90], [0.0582, 87],
                         [0.172, 84], [0.508, 74], [1.5, 38], [4.44, 29],
                         [13.1, 20], [38.7, 16], [115., 13], [338., 13],
                         [1000., 13]]

Thomas2011IFNalpha2AV = [[0.0001, 100], [0.001, 98], [0.01, 88], [0.1, 55],
                         [1., 5], [10., 0], [100., 0], [1000., 0]]

Thomas2011IFNalpha7AP = [[0.00001, 100], [0.0000296, 100], [0.0000873, 100],
                         [0.000258, 100], [0.000763, 100], [0.00225, 84],
                         [0.00666, 92], [0.0197, 100], [0.0582, 105],
                         [0.172, 88], [0.508, 77], [1.5, 57], [4.44, 40],
                         [13.1, 26], [38.7, 22], [115., 18], [338., 10],
                         [1000., 10]]

Thomas2011IFNalpha7AV = [[0.0001, 100], [0.001, 98], [0.01, 98], [0.1, 13],
                         [1., 2], [10., 0], [100., 0], [1000., 0]]

Thomas2011IFNomegaAP = [[0.00001, 100], [0.0000296, 100], [0.0000873, 100],
                        [0.000258, 100], [0.000763, 100], [0.00225, 90],
                        [0.00666, 105], [0.0197, 98], [0.0582, 92],
                        [0.172, 76], [0.508, 53], [1.5, 30], [4.44, 20],
                        [13.1, 15], [38.7, 12], [115., 10], [338., 0],
                        [1000., 0]]

Thomas2011IFNomegaAV = [[0.0001, 100], [0.001, 92], [0.01, 82], [0.1, 23],
                        [1., 2], [10., 0], [100., 0], [1000., 0]]

Thomas2011IFNalpha2YNSAP = [[0.00001, 100], [0.0000296, 105], [0.0000873, 88],
                            [0.000258, 82], [0.000763, 70], [0.00225, 38],
                            [0.00666, 25], [0.0197, 14], [0.0582, 8],
                            [0.172, 5], [0.508, 0], [1.5, 0], [4.44, 0],
                            [13.1, 0], [38.7, 0], [115., 0], [338., 0],
                            [1000., 0]]

Thomas2011IFNalpha2YNSAV = [[0.0001, 100], [0.001, 95], [0.01, 75], [0.1, 4],
                            [1., 0], [10., 0], [100., 0], [1000., 0]]

Schreiber2017AV = np.array(
                  [[0.0001, 0.0015], [0.00034, 0.0025], [0.0014, 0.01],
                   [0.0051, 0.02], [0.01, 0.027], [0.019, 0.05],
                   [0.024, 0.042], [0.028, 0.17], [0.073, 0.077],
                   [0.078, 0.1], [0.104, 0.33], [0.12, 0.22], [0.18, 0.3],
                   [0.2, 0.58], [0.34, 0.74], [0.32, 0.63], [0.58, 0.75],
                   [1., 0.45], [1.33, 0.29], [2.98, 2.12], [7.5, 1.06],
                   [14.8, 4], [25., 2.1], [39., 2.1], [74., 2.1]])

Schreiber2017AP = np.array(
                  [[0.0001, 0.00022], [0.00034, 0.00096], [0.0014, 0.0025],
                   [0.0012, 0.0081], [0.0051, 0.015], [0.01, 0.045],
                   [0.019, 0.05], [0.017, 0.13], [0.024, 0.016], [0.028, 0.13],
                   [0.073, 0.277], [0.078, 0.3], [0.104, 0.24], [0.12, 0.22],
                   [0.18, 0.44], [0.2, 0.36], [0.34, 0.39], [0.32, 0.58],
                   [0.58, 0.3], [1., 0.9], [2.98, 3.24], [7.5, 2.38],
                   [14.8, 9.3], [25., 25.8], [39., 41.], [74., 71.3]])


if __name__ == '__main__':
    AandB_flag = False
    C_flag = False
    USP18_sf = 15
    dir = os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_5')
    if not os.path.exists(dir):
        os.makedirs(dir)

    times = [60]
    test_doses = list(logspace(-4, 6))
    test_affinities = list(logspace(-2, 4, num=10)) # THIS CODE ASSUMES THAT THE 4TH ELEMENT IN IS 1.0
    KMfit = 1E6  # used in AP phenomenogical function

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

    if AandB_flag:
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

    if C_flag:
        AV_EC50_record = []
        AP_EC50_record = []
        # ----------------------------------------------------------------------
        # Get anti-viral and anti-proliferative EC50 for a variety of K1 and K4
        # ----------------------------------------------------------------------
        for i in test_affinities:
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

        np.save(dir + os.sep + 'AV_EC50.npy', np.array(AV_EC50_record))
        np.save(dir + os.sep + 'AP_EC50.npy', np.array(AP_EC50_record))

    else:
        AV_EC50_record = np.array(np.load(dir + os.sep + 'AV_EC50.npy'))
        AP_EC50_record = np.array(np.load(dir + os.sep + 'AP_EC50.npy'))

    # ------------------------------------------------
    # Get anti-viral and anti-proliferative responses
    # ------------------------------------------------
    IFNa2_AV = antiViralActivity(pSTAT_a2)
    IFNa2YNS_AV = antiViralActivity(pSTAT_a2YNS)
    IFNa7_AV = antiViralActivity(pSTAT_a7)
    IFNw_AV = antiViralActivity(pSTAT_w)

    IFNa2YNS_AP = antiProliferativeActivity(pSTAT_a2YNS_refractory, KM=KMfit)
    AP_scale = 150. / max(IFNa2YNS_AP)
    IFNa2YNS_AP = AP_scale * IFNa2YNS_AP
    IFNa2_AP = AP_scale * antiProliferativeActivity(pSTAT_a2_refractory, KM=KMfit)
    IFNa7_AP = AP_scale * antiProliferativeActivity(pSTAT_a7_refractory, KM=KMfit)
    IFNw_AP = AP_scale * antiProliferativeActivity(pSTAT_w_refractory, KM=KMfit)

    # ------------------------
    # Plot fit to Thomas 2011
    # ------------------------
    colour_palette = sns.color_palette("deep", 4)
    labels = [r"IFN$\alpha$2", r"IFN$\alpha$7", r"IFN$\omega$", r"IFN$\alpha$2-YNS"]
    fig, all_axes = plt.subplots(nrows=2, ncols=2, figsize=(10., 10.))
    axes = all_axes[0] # top row of plots
    axes[0].set_xscale('log')
    axes[1].set_xscale('log')
    # Anti-viral activity
    exp_data = list(map(np.array, [Thomas2011IFNalpha2AV, Thomas2011IFNalpha7AV, Thomas2011IFNomegaAV, Thomas2011IFNalpha2YNSAV]))
    sim_data = list(map(np.array, [IFNa2_AV, IFNa7_AV, IFNw_AV, IFNa2YNS_AV]))
    for idx in range(len(exp_data)):
        axes[0].scatter(exp_data[idx][:, 0], exp_data[idx][:, 1], color=colour_palette[idx], label=labels[idx])
        axes[0].plot(test_doses, 100-sim_data[idx], color=colour_palette[idx], linewidth=2)
    # Anti-proliferative activity
    exp_data = list(map(np.array, [Thomas2011IFNalpha2AP, Thomas2011IFNalpha7AP, Thomas2011IFNomegaAP, Thomas2011IFNalpha2YNSAP]))
    sim_data = list(map(np.array, [IFNa2_AP, IFNa7_AP, IFNw_AP, IFNa2YNS_AP]))
    for idx in range(len(exp_data)):
        # include factor of 1E3 because data is in nM but axis is in pM
        axes[1].scatter(1E3*exp_data[idx][:, 0], exp_data[idx][:, 1], color=colour_palette[idx], label=labels[idx])
        axes[1].plot(test_doses, [max(0, el) for el in 100-sim_data[idx]], color=colour_palette[idx], linewidth=2)

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
    red = sns.color_palette("tab10")[3]
    blue = sns.color_palette("tab10")[0]
    fig.delaxes(all_axes[1][1])
    ax = all_axes[1][0]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Binding Affinity\n' + r'(Relative to IFN$\alpha$2)')
    ax.set_ylabel('Biological Activity\n' + r'(Relative to IFN$\alpha$2)')
    ax.scatter(Schreiber2017AV[:,0], Schreiber2017AV[:,1], color=red, label='Anti-viral')
    ax.scatter(Schreiber2017AP[:,0], Schreiber2017AP[:,1], color=blue, label='Anti-proliferative')
    ax.plot(1/AV_EC50_record[:, 0], AV_EC50_record[:, 1]/AV_EC50_record[3, 1], color=red, linewidth=2)
    ax.plot(1/AP_EC50_record[:, 0], AP_EC50_record[:, 1]/AP_EC50_record[3, 1], color=blue, linewidth=2)
    ax.legend()

    # save figure
    plt.tight_layout()
    fig.savefig(os.path.join(dir, 'Figure_5.pdf'))
