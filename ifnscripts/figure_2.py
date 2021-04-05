from ifnclass.ifndata import IfnData, DataAlignment
from numpy import linspace, logspace
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from ifnclass.ifnplot import DoseresponsePlot
import matplotlib.gridspec as gridspec
import load_model as lm


def equilibrium_T_EC50(Rt, Delta, K1, K2, K4):
    X = K2 / K1
    B = 1 + (K4 / Rt) * (1 + np.sqrt(X))**2
    D = Rt / K4 + 1 + 10 * np.sqrt(X) + X + 3 * (Rt / K4) * \
        np.sqrt(B**2 - 1 + (Delta / Rt)**2)
    Keff = (K1 / 8) * (D - np.sqrt(D**2 - 64 * X))
    return Keff


def equilibrium_TMax(Rt, Delta, K1, K2, K4):
    B = 1 + (K4 / Rt) * (1 + np.sqrt(K2 / K1))**2
    return (Rt / 2) * (B - np.sqrt(B**2 - 1 + (Delta / Rt)**2))


def pSTAT_EC50(Rt, Delta, K1, K2, K4, A, KP):
    Keff = equilibrium_T_EC50(Rt, Delta, K1, K2, K4)
    Tmax = equilibrium_TMax(Rt, Delta, K1, K2, K4)
    return Keff / (1 + Tmax * A / KP)


def pSTAT_Max(Rt, Delta, K1, K2, K4, A, KP, STAT):
    Tmax = equilibrium_TMax(Rt, Delta, K1, K2, K4)
    return STAT * (Tmax / (Tmax + KP / A))


def avg_theory(model):
    """
    The outputs avg_pSTAT_Max_Beta & avg_pSTAT_Max_Alpha will be in [molec.]
    The outputs avg_pSTAT_EC50_Beta & avg_pSTAT_EC50_Alpha will be in [M]
    """
    avg_pSTAT_Max_Alpha = 0
    avg_pSTAT_EC50_Alpha = 0
    avg_pSTAT_Max_Beta = 0
    avg_pSTAT_EC50_Beta = 0

    for idx, m in enumerate([model.model_1, model.model_2]):
        volPM = m.parameters['volPM']  # [m**2]
        NA = m.parameters['NA']
        volEC = m.parameters['volEC']  # [m**3]
        # [Rt] = [Delta] = molec./ m**2
        Rt = (m.parameters['R1'] + m.parameters['R2']) / volPM
        Delta = (m.parameters['R1'] - m.parameters['R2']) / volPM
        # [K1] = [K2] = Molar, at least below since I divide by NA*volEC
        K1 = m.parameters['kd1'] / m.parameters['ka1'] / NA / volEC
        K2 = m.parameters['kd2'] / m.parameters['ka2'] / NA / volEC
        # [K4] = molec./ m**2, since I divide by volPM
        K4 = m.parameters['kd4'] / m.parameters['ka4'] / volPM
        A = m.parameters['volPM']  # [m**2]
        # [KP] = [], i.e. dimensionless
        KP = m.parameters['kpu'] / m.parameters['kpa']
        STAT = m.parameters['S']
        pSTAT_Max_Alpha = pSTAT_Max(Rt, Delta, K1, K2, K4, A, KP, STAT)
        EC50_Alpha = pSTAT_EC50(Rt, Delta, K1, K2, K4, A, KP)
        weight = [model.w1, model.w2][idx]
        avg_pSTAT_Max_Alpha += weight * pSTAT_Max_Alpha
        avg_pSTAT_EC50_Alpha += weight * EC50_Alpha

        K1 = m.parameters['k_d1'] / m.parameters['k_a1'] / NA / volEC
        K2 = m.parameters['k_d2'] / m.parameters['k_a2'] / NA / volEC
        # [K4] = molec./ m**2, since I divide by volPM
        K4 = m.parameters['k_d4'] / m.parameters['k_a4'] / volPM
        pSTAT_Max_Beta = pSTAT_Max(Rt, Delta, K1, K2, K4, A, KP, STAT)
        EC50_Beta = pSTAT_EC50(Rt, Delta, K1, K2, K4, A, KP)
        avg_pSTAT_Max_Beta += weight * pSTAT_Max_Beta
        avg_pSTAT_EC50_Beta += weight * EC50_Beta

    return avg_pSTAT_Max_Alpha, avg_pSTAT_EC50_Alpha, avg_pSTAT_Max_Beta, avg_pSTAT_EC50_Beta


def prior_theory(m):
    """
    The outputs avg_pSTAT_Max_Beta & avg_pSTAT_Max_Alpha will be in [molec.]
    The outputs avg_pSTAT_EC50_Beta & avg_pSTAT_EC50_Alpha will be in [M]
    """
    volPM = m.prior_parameters['volPM']  # [m**2]
    NA = m.prior_parameters['NA']
    volEC = m.prior_parameters['volEC']  # [m**3]
    # [Rt] = [Delta] = molec./ m**2
    Rt = (m.prior_parameters['R1'] + m.prior_parameters['R2']) / volPM
    Delta = (m.prior_parameters['R1'] - m.prior_parameters['R2']) / volPM
    # [K1] = [K2] = Molar, at least below since I divide by NA*volEC
    K1 = m.prior_parameters['kd1'] / m.prior_parameters['ka1'] / NA / volEC
    K2 = m.prior_parameters['kd2'] / m.prior_parameters['ka2'] / NA / volEC
    # [K4] = molec./ m**2, since I divide by volPM
    K4 = m.prior_parameters['kd4'] / m.prior_parameters['ka4'] / volPM
    A = m.prior_parameters['volPM']  # [m**2]
    # [KP] = [], i.e. dimensionless
    KP = m.prior_parameters['kpu'] / m.prior_parameters['kpa']
    STAT = m.prior_parameters['S']
    pSTAT_Max_Alpha = pSTAT_Max(Rt, Delta, K1, K2, K4, A, KP, STAT)
    EC50_Alpha = pSTAT_EC50(Rt, Delta, K1, K2, K4, A, KP)
    avg_pSTAT_Max_Alpha = pSTAT_Max_Alpha
    avg_pSTAT_EC50_Alpha = EC50_Alpha

    K1 = m.prior_parameters['k_d1'] / m.prior_parameters['k_a1'] / NA / volEC
    K2 = m.prior_parameters['k_d2'] / m.prior_parameters['k_a2'] / NA / volEC
    # [K4] = molec./ m**2, since I divide by volPM
    K4 = m.prior_parameters['k_d4'] / m.prior_parameters['k_a4'] / volPM
    pSTAT_Max_Beta = pSTAT_Max(Rt, Delta, K1, K2, K4, A, KP, STAT)
    EC50_Beta = pSTAT_EC50(Rt, Delta, K1, K2, K4, A, KP)
    avg_pSTAT_Max_Beta = pSTAT_Max_Beta
    avg_pSTAT_EC50_Beta = EC50_Beta

    return avg_pSTAT_Max_Alpha, avg_pSTAT_EC50_Alpha, avg_pSTAT_Max_Beta, avg_pSTAT_EC50_Beta


if __name__ == '__main__':
    print('Figure 2')
    # ----------------------
    # Set up Figure 2 layout
    # ----------------------
    Figure_2 = plt.figure(tight_layout=True, figsize=(11., 6.))
    gs = gridspec.GridSpec(nrows=2, ncols=4, height_ratios=[3, 2])

    Figure_2.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()

    # --------------------
    # Set up EC50 figures
    # --------------------
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)
    data_palette = sns.color_palette("muted", 6)
    marker_shape = ["o", "v", "s", "P", "d", "1", "x", "*"]
    dataset_names = ["20190108", "20190119", "20190121", "20190214"]

    # --------------------
    # Set up Model
    # --------------------
    Mixed_Model, DR_method = lm.load_model()
    scale_factor, DR_KWARGS, PLOT_KWARGS = lm.SCALE_FACTOR, lm.DR_KWARGS, lm.PLOT_KWARGS

    # ---------------------------------
    # Make theory dose response curves
    # ---------------------------------
    # Equilibrium predictions
    if lm.ENSEMBLE:
        avg_pSTAT_Max_Alpha, avg_pSTAT_EC50_Alpha, avg_pSTAT_Max_Beta, avg_pSTAT_EC50_Beta = prior_theory(Mixed_Model)
    else:
        avg_pSTAT_Max_Alpha, avg_pSTAT_EC50_Alpha, avg_pSTAT_Max_Beta, avg_pSTAT_EC50_Beta = avg_theory(Mixed_Model)

    # Make predictions
    times = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
    alpha_doses_20190108 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses_20190108 = [0, 0.2, 6, 20, 60, 200, 600, 2000]

    dra60 = DR_method(times, 'TotalpSTAT', 'Ia',
                                            list(logspace(1, 5.2)),
                                            parameters={'Ib': 0},
                                            sf=scale_factor,
                                            **DR_KWARGS)

    drb60 = DR_method(times, 'TotalpSTAT', 'Ib',
                                            list(logspace(-1, 4)),
                                            parameters={'Ia': 0},
                                            sf=scale_factor,
                                            **DR_KWARGS)

    # ----------------------------------
    # Get all data set EC50 time courses
    # ----------------------------------
    newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
    newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
    newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
    newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")

    # 20190108
    ec50_20190108 = newdata_1.get_ec50s()

    # 20190119
    ec50_20190119 = newdata_2.get_ec50s()

    # 20190121
    ec50_20190121 = newdata_3.get_ec50s()

    # 20190214
    ec50_20190214 = newdata_4.get_ec50s()

    # Aligned data, to get scale factors for each data set
    alignment = DataAlignment()
    alignment.add_data([newdata_4, newdata_3, newdata_2, newdata_1])
    alignment.align()
    alignment.get_scaled_data()
    mean_data = alignment.summarize_data()

    # ----------------------------------
    # Make model predictions for EC50
    # ----------------------------------
    time_list = list(linspace(2.5, 60, num=15))

    dfa = DR_method(time_list, 'TotalpSTAT', 'Ia',
                                          list(logspace(-3, 5)),
                                          parameters={'Ib': 0},
                                          sf=scale_factor,
                                          **{'return_type': 'IfnData'})

    alpha_ec_aggregate = [el[1] for el in dfa.get_ec50s()['Alpha']]
    alpha_peak_aggregate = [el[1] for el in dfa.get_max_responses()['Alpha']]

    dfb = DR_method(time_list, 'TotalpSTAT', 'Ib',
                                          list(logspace(-3, 5)),
                                          parameters={'Ia': 0},
                                          sf=scale_factor,
                                          **{'return_type': 'IfnData'})

    beta_ec_aggregate = [el[1] for el in dfb.get_ec50s()['Beta']]
    beta_peak_aggregate = [el[1] for el in dfb.get_max_responses()['Beta']]

    # -----------------------
    # Plot EC50 vs time
    # -----------------------
    ec50_axes_list = [Figure_2.add_subplot(gs[1, 0]),
                      Figure_2.add_subplot(gs[1, 1]),
                      Figure_2.add_subplot(gs[1, 2]),
                      Figure_2.add_subplot(gs[1, 3])]

    ec50_axes = ec50_axes_list[0:2]
    ec50_axes[0].set_xlabel("Time (min)")
    ec50_axes[1].set_xlabel("Time (min)")
    ec50_axes[0].set_title(r"EC50 vs Time for IFN$\alpha$")
    ec50_axes[1].set_title(r"EC50 vs Time for IFN$\beta$")
    ec50_axes[0].set_ylabel("EC50 (pM)")
    ec50_axes[0].set_yscale('log')
    ec50_axes[1].set_yscale('log')
    # Add models
    ec50_axes[0].plot(time_list, alpha_ec_aggregate, label=r'IFN$\alpha$',
                      color=alpha_palette[5], linewidth=2)
    ec50_axes[1].plot(time_list, beta_ec_aggregate, label=r'IFN$\beta$',
                      color=beta_palette[5], linewidth=2)

    # Average the data
    combined_a_ec50 = np.empty((4, len(ec50_20190108['Alpha'])))
    combined_b_ec50 = np.empty((4, len(ec50_20190108['Beta'])))
    for idx, ec50 in enumerate([ec50_20190108, ec50_20190119,
                                ec50_20190121, ec50_20190214]):
        combined_a_ec50[idx, :] = np.array([el[1] for el in ec50['Alpha']])
        combined_b_ec50[idx, :] = np.array([el[1] for el in ec50['Beta']])

    ec50_axes[0].errorbar([el[0] for el in ec50_20190108['Alpha']],
                          np.mean(combined_a_ec50, axis=0),
                          yerr=np.std(combined_a_ec50, axis=0),
                          color='k', ls='none')
    ec50_axes[1].errorbar([el[0] for el in ec50_20190108['Beta']],
                          np.mean(combined_b_ec50, axis=0),
                          yerr=np.std(combined_b_ec50, axis=0),
                          color='k', ls='none')
    ec50_axes[0].scatter([el[0] for el in ec50_20190108['Alpha']],
                         np.mean(combined_a_ec50, axis=0),
                         color='k')
    ec50_axes[1].scatter([el[0] for el in ec50_20190108['Beta']],
                         np.mean(combined_b_ec50, axis=0),
                         color='k')

    # Add equilibrium expressions
    ec50_axes[0].plot(time_list, [avg_pSTAT_EC50_Alpha / 1E-12
                                  for _ in time_list],
                      '--', color=alpha_palette[5], linewidth=2)
    ec50_axes[1].plot(time_list, [avg_pSTAT_EC50_Beta / 1E-12
                                  for _ in time_list],
                      '--', color=beta_palette[5], linewidth=2)
    # -----------------
    # Data max response
    # -----------------
    # 20190108
    max_20190108 = newdata_1.get_max_responses()

    # 20190119
    max_20190119 = newdata_2.get_max_responses()

    # 20190121
    max_20190121 = newdata_3.get_max_responses()

    # 20190214
    max_20190214 = newdata_4.get_max_responses()

    # -------------------
    # Plot max response
    # -------------------
    # fig, axes = plt.subplots(nrows=1, ncols=2)
    max_response_axes = ec50_axes_list[2:4]
    max_response_axes[0].set_xlabel("Time (min)")
    max_response_axes[1].set_xlabel("Time (min)")
    max_response_axes[0].set_title(r"Max pSTAT vs Time for IFN$\alpha$")
    max_response_axes[1].set_title(r"Max pSTAT vs Time for IFN$\beta$")
    max_response_axes[0].set_ylabel("Max pSTAT (MFI)")

    # Add models
    max_response_axes[0].plot(time_list, alpha_peak_aggregate,
                              color=alpha_palette[5], linewidth=2)
    max_response_axes[1].plot(time_list, beta_peak_aggregate,
                              color=beta_palette[5], linewidth=2)
    # Average the data
    combined_a_maxpSTAT = np.empty((4, len(max_20190108['Alpha'])))
    combined_b_maxpSTAT = np.empty((4, len(max_20190108['Beta'])))
    for idx, maxpSTAT in enumerate([max_20190108, max_20190119,
                                    max_20190121, max_20190214]):
        scale_factor = alignment.scale_factors[3 - idx]
        combined_a_maxpSTAT[idx, :] = np.array([el[1] * scale_factor
                                                for el in maxpSTAT['Alpha']])
        combined_b_maxpSTAT[idx, :] = np.array([el[1] * scale_factor
                                                for el in maxpSTAT['Beta']])

    max_response_axes[0].errorbar([el[0] for el in max_20190108['Alpha']],
                                  np.mean(combined_a_maxpSTAT, axis=0),
                                  yerr=np.std(combined_a_maxpSTAT, axis=0),
                                  color='k', ls='none')
    max_response_axes[1].errorbar([el[0] for el in max_20190108['Beta']],
                                  np.mean(combined_b_maxpSTAT, axis=0),
                                  yerr=np.std(combined_b_maxpSTAT, axis=0),
                                  color='k', ls='none')
    max_response_axes[0].scatter([el[0] for el in max_20190108['Alpha']],
                                 np.mean(combined_a_maxpSTAT, axis=0),
                                 color='k')
    max_response_axes[1].scatter([el[0] for el in max_20190108['Beta']],
                                 np.mean(combined_b_maxpSTAT, axis=0),
                                 color='k')

    # Add equilibrium expressions
    # max_response_axes[0].plot(time_list,
    #                          [avg_pSTAT_Max_Alpha for _ in time_list],
    #                          '--', color=alpha_palette[5], linewidth=2)
    # max_response_axes[1].plot(time_list,
    #                          [avg_pSTAT_Max_Beta for _ in time_list],
    #                          '--', color=alpha_palette[5], linewidth=2)

    # -------------------------------
    # Plot model dose response curves
    # -------------------------------
    alpha_palette = sns.color_palette("rocket_r", 6)
    beta_palette = sns.color_palette("rocket_r", 6)

    new_fit = DoseresponsePlot((1, 2))
    new_fit.axes = [Figure_2.add_subplot(gs[0, 0:2]),
                    Figure_2.add_subplot(gs[0, 2:4])]
    new_fit.axes[0].set_xscale('log')
    new_fit.axes[0].set_xlabel('Dose (pM)')
    new_fit.axes[0].set_ylabel('pSTAT (MFI)')
    new_fit.axes[1].set_xscale('log')
    new_fit.axes[1].set_xlabel('Dose (pM)')
    new_fit.axes[1].set_ylabel('pSTAT (MFI)')
    new_fit.fig = Figure_2

    alpha_mask = [7.5]
    beta_mask = [7.5]
    # Add fits
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(dra60, t, PLOT_KWARGS['line_type'],
                                   alpha_palette[idx], (0, 0),
                                   'Alpha', label=str(t)+' min',
                                   linewidth=2, alpha=PLOT_KWARGS['alpha'])
            new_fit.add_trajectory(mean_data, t, 'errorbar', 'o', (0, 0),
                                   'Alpha', color=alpha_palette[idx])
        if t not in beta_mask:
            new_fit.add_trajectory(drb60, t, PLOT_KWARGS['line_type'],
                                   beta_palette[idx], (0, 1),
                                   'Beta', label=str(t) + ' min',
                                   linewidth=2, alpha=PLOT_KWARGS['alpha'])
            new_fit.add_trajectory(mean_data, t, 'errorbar', 'o', (0, 1),
                                   'Beta', color=beta_palette[idx])

    plt.figure(Figure_2.number)
    dr_fig, dr_axes = new_fit.show_figure(show_flag=False)

    # Format Figure_2
    # Dose response aesthetics
    for ax in Figure_2.axes[4:6]:
        ax.set_ylim((0, 5000))
    Figure_2.axes[4].set_title(r'IFN$\alpha$')
    Figure_2.axes[5].set_title(r'IFN$\beta$')

    # max pSTAT aesthetics
    for ax in Figure_2.axes[2:4]:
        ax.set_ylim([1500, 5500])

    # EC50 aesthetics
    for ax in Figure_2.axes[0:2]:
        ax.set_ylim([1, 5000])

    Figure_2.savefig(os.path.join(os.getcwd(), 'results', 'Figures',
                     'Figure_2', 'Figure_2.pdf'))
