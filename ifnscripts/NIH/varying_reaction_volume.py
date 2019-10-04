from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnfit import DualMixedPopulation
from numpy import linspace, logspace, transpose
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from ifnclass.ifnplot import TimecoursePlot, DoseresponsePlot

def testing_specific_values():
    volEC1 = 1 / (25E9)
    volEC2 = 1E-5
    # --------------------
    # Set up Model
    # --------------------
    # Parameters found by stepwise fitting GAB mean data
    # Note: can remove multiplicative factors on all K1, K2, K4 and still get very good fit to data (worst is 5 min beta)
    initial_parameters = {'k_a1': 4.98E-14 * 2, 'k_a2': 8.30e-13 * 2, 'k_d4': 0.006 * 3.8,
                          'kpu': 0.00095,
                          'ka2': 4.98e-13 * 2.45, 'kd4': 0.3 * 2.867,
                          'kint_a': 0.000124, 'kint_b': 0.00086,
                          'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05}
    dual_parameters = {'kint_a': 0.00052, 'kSOCSon': 6e-07, 'kint_b': 0.00052, 'krec_a1': 0.001, 'krec_a2': 0.1,
                       'krec_b1': 0.005, 'krec_b2': 0.05}
    scale_factor = 1.227

    Mixed_Model = DualMixedPopulation('Mixed_IFN_ppCompatible', 0.8, 0.2)
    Mixed_Model.model_1.set_parameters(initial_parameters)
    Mixed_Model.model_1.set_parameters(dual_parameters)
    Mixed_Model.model_1.set_parameters({'R1': 12000.0, 'R2': 1511.1})
    Mixed_Model.model_2.set_parameters(initial_parameters)
    Mixed_Model.model_2.set_parameters(dual_parameters)
    Mixed_Model.model_2.set_parameters({'R1': 6755.56, 'R2': 1511.1})

    # ---------------------------------
    # Make theory dose response curves
    # ---------------------------------
    # Make predictions
    times = np.arange(0, 120, 0.5)  # [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
    alpha_doses_20190108 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses_20190108 = [0, 0.2, 6, 20, 60, 200, 600, 2000]

    # 1 uL
    ka1 = Mixed_Model.model_1.parameters['ka1']
    ka2 = Mixed_Model.model_1.parameters['ka2']
    k_a1 = Mixed_Model.model_1.parameters['k_a1']
    k_a2 = Mixed_Model.model_1.parameters['k_a2']
    Mixed_Model.set_global_parameters({'volEC': volEC1, 'ka1': ka1 * 1E-5 / volEC1, 'ka2': ka2 * 1E-5 / volEC1,
                                       'k_a1': k_a1 * 1E-5 / volEC1, 'k_a2': k_a2 * 1E-5 / volEC1})
    dradf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia', list(logspace(1, 5.2)),
                                            parameters={'Ib': 0}, sf=scale_factor)
    drbdf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ib', list(logspace(-1, 4)),
                                            parameters={'Ia': 0}, sf=scale_factor)

    dra15uL = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb15uL = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    # 1 mL
    Mixed_Model.set_global_parameters({'volEC': volEC2, 'ka1': ka1 * 1E-5 / volEC2, 'ka2': ka2 * 1E-5 / volEC2,
                                       'k_a1': k_a1 * 1E-5 / volEC2, 'k_a2': k_a2 * 1E-5 / volEC2})
    dradf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia', list(logspace(1, 5.2)),
                                            parameters={'Ib': 0}, sf=scale_factor)
    drbdf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ib', list(logspace(-1, 4)),
                                            parameters={'Ia': 0}, sf=scale_factor)

    dra5mL = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb5mL = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    # -------------------------------
    # Plot model dose response curves
    # -------------------------------
    alpha_palette = sns.color_palette("deep", 6)
    beta_palette = sns.color_palette("deep", 6)

    new_fit = DoseresponsePlot((2, 2))

    alpha_mask = [2.5, 5.0, 10.0, 20.0, 60.0]
    beta_mask = [2.5, 5.0, 10.0, 20.0, 60.0]
    # Add fits
    color_counter = -1
    for idx, t in enumerate(times):
        if t in alpha_mask:
            color_counter += 1
            new_fit.add_trajectory(dra15uL, t, 'plot', alpha_palette[color_counter], (0, 0), 'Alpha',
                                   label=str(t) + ' min',
                                   linewidth=2)
            new_fit.add_trajectory(dra5mL, t, 'plot', alpha_palette[color_counter], (1, 0), 'Alpha',
                                   label=str(t) + ' min',
                                   linewidth=2)

        if t in beta_mask:
            new_fit.add_trajectory(drb15uL, t, 'plot', beta_palette[color_counter], (0, 1), 'Beta',
                                   label=str(t) + ' min',
                                   linewidth=2)
            new_fit.add_trajectory(drb5mL, t, 'plot', beta_palette[color_counter], (1, 1), 'Beta',
                                   label=str(t) + ' min',
                                   linewidth=2)

    dr_fig, dr_axes = new_fit.show_figure()
    dr_axes[0][0].set_title(r'IFN$\alpha$ at {} L'.format(volEC1))
    dr_axes[0][1].set_title(r'IFN$\beta$ at {} L'.format(volEC1))
    dr_axes[1][0].set_title(r'IFN$\alpha$ at {} L'.format(volEC2))
    dr_axes[1][1].set_title(r'IFN$\beta$ at {} L'.format(volEC2))
    dr_fig.tight_layout()
    dr_fig.savefig('varying_reaction_volume_dr.pdf')

    # -------------------------------
    # Plot model dose response curves
    # -------------------------------
    tc_figure = TimecoursePlot((1, 2))

    alpha_testdose = list(logspace(1, 5.2))[24]
    beta_testdose = list(logspace(-1, 4))[17]
    print("Using {} for IFNalpha".format(alpha_testdose))
    print("Using {} for IFNbeta".format(beta_testdose))

    tc_figure.add_trajectory(dra15uL, 'plot', '-', subplot_idx=(0, 0), label=r'IFN$\alpha$ at {} L'.format(volEC1),
                             doseslice=alpha_testdose, dose_species='Alpha', color=alpha_palette[1])
    tc_figure.add_trajectory(dra5mL, 'plot', '-', subplot_idx=(0, 0), label=r'IFN$\alpha$ at {} L'.format(volEC2),
                             doseslice=alpha_testdose, dose_species='Alpha', color=alpha_palette[4])

    tc_figure.add_trajectory(drb15uL, 'plot', '-', subplot_idx=(0, 1), label=r'IFN$\beta$ at {} L'.format(volEC1),
                             doseslice=beta_testdose, dose_species='Beta', color=beta_palette[0])
    tc_figure.add_trajectory(drb5mL, 'plot', '-', subplot_idx=(0, 1), label=r'IFN$\beta$ at {} L'.format(volEC2),
                             doseslice=beta_testdose, dose_species='Beta', color=beta_palette[2])
    tc_figure.show_figure(save_flag=True, save_dir='varying_reaction_volume_tc.pdf')

    # ------------------------------------
    # Now let's quickly look at IFN levels
    # ------------------------------------
    # 1 uL
    Mixed_Model.set_global_parameters({'volEC': volEC1, 'ka1': ka1 * 1E-5 / volEC1, 'ka2': ka2 * 1E-5 / volEC1,
                                       'k_a1': k_a1 * 1E-5 / volEC1, 'k_a2': k_a2 * 1E-5 / volEC1})
    dradf = Mixed_Model.mixed_dose_response(times, 'Free_Ia', 'Ia', list(logspace(1, 5.2)),
                                            parameters={'Ib': 0}, sf=scale_factor)
    drbdf = Mixed_Model.mixed_dose_response(times, 'Free_Ib', 'Ib', list(logspace(-1, 4)),
                                            parameters={'Ia': 0}, sf=scale_factor)

    dra15uL = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb15uL = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    # 1 mL
    Mixed_Model.set_global_parameters({'volEC': volEC2, 'ka1': ka1 * 1E-5 / volEC2, 'ka2': ka2 * 1E-5 / volEC2,
                                       'k_a1': k_a1 * 1E-5 / volEC2, 'k_a2': k_a2 * 1E-5 / volEC2})
    dradf = Mixed_Model.mixed_dose_response(times, 'Free_Ia', 'Ia', list(logspace(1, 5.2)),
                                            parameters={'Ib': 0}, sf=scale_factor)
    drbdf = Mixed_Model.mixed_dose_response(times, 'Free_Ib', 'Ib', list(logspace(-1, 4)),
                                            parameters={'Ia': 0}, sf=scale_factor)

    dra5mL = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb5mL = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    tc_figure = TimecoursePlot((1, 2))

    tc_figure.add_trajectory(dra15uL, 'plot', '-', subplot_idx=(0, 0), label=r'IFN$\alpha$ at {} L'.format(volEC1),
                             doseslice=alpha_testdose, dose_species='Alpha', color=alpha_palette[1])
    tc_figure.add_trajectory(dra5mL, 'plot', '-', subplot_idx=(0, 0), label=r'IFN$\alpha$ at {} L'.format(volEC2),
                             doseslice=alpha_testdose, dose_species='Alpha', color=alpha_palette[4])

    tc_figure.add_trajectory(drb15uL, 'plot', '-', subplot_idx=(0, 1), label=r'IFN$\beta$ at {} L'.format(volEC1),
                             doseslice=beta_testdose, dose_species='Beta', color=beta_palette[0])
    tc_figure.add_trajectory(drb5mL, 'plot', '-', subplot_idx=(0, 1), label=r'IFN$\beta$ at {} L'.format(volEC2),
                             doseslice=beta_testdose, dose_species='Beta', color=beta_palette[2])
    for ax in tc_figure.axes.flatten():
        ax.set_yscale('log')
    ifn_fig, ifn_axes = tc_figure.show_figure()
    ifn_axes[0].set_ylabel('IFN-alpha2')
    ifn_axes[1].set_ylabel('IFN-beta')
    ifn_fig.savefig('varying_reaction_volume_tc_IFN.pdf')


def run_smooth_trajectories(cell_densities, IFNAlpha_panel, IFNBeta_panel, times, IFN_in_concentration=True):
    # Experimental parameters
    volume_panel = [1 / i for i in cell_densities]
    # --------------------
    # Set up Model
    # --------------------
    # Parameters found by stepwise fitting GAB mean data
    # Note: can remove multiplicative factors on all K1, K2, K4 and still get very good fit to data (worst is 5 min beta)
    initial_parameters = {'k_a1': 4.98E-14 * 2, 'k_a2': 8.30e-13 * 2, 'k_d4': 0.006 * 3.8,
                          'kpu': 0.00095,
                          'ka2': 4.98e-13 * 2.45, 'kd4': 0.3 * 2.867,
                          'kint_a': 0.000124, 'kint_b': 0.00086,
                          'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05}
    dual_parameters = {'kint_a': 0.00052, 'kSOCSon': 6e-07, 'kint_b': 0.00052, 'krec_a1': 0.001, 'krec_a2': 0.1,
                       'krec_b1': 0.005, 'krec_b2': 0.05}
    scale_factor = 1.227

    Mixed_Model = DualMixedPopulation('Mixed_IFN_ppCompatible', 0.8, 0.2)
    Mixed_Model.model_1.set_parameters(initial_parameters)
    Mixed_Model.model_1.set_parameters(dual_parameters)
    Mixed_Model.model_1.set_parameters({'R1': 12000.0, 'R2': 1511.1})
    Mixed_Model.model_2.set_parameters(initial_parameters)
    Mixed_Model.model_2.set_parameters(dual_parameters)
    Mixed_Model.model_2.set_parameters({'R1': 6755.56, 'R2': 1511.1})

    ka1 = Mixed_Model.model_1.parameters['ka1']
    ka2 = Mixed_Model.model_1.parameters['ka2']
    k_a1 = Mixed_Model.model_1.parameters['k_a1']
    k_a2 = Mixed_Model.model_1.parameters['k_a2']

    # ---------------------------------
    # Make theory dose response curves
    # ---------------------------------
    # Make predictions
    predictions = {}
    for vidx, volume in enumerate(volume_panel):
        Mixed_Model.set_global_parameters(
            {'volEC': volume, 'ka1': ka1 * 1E-5 / volume, 'ka2': ka2 * 1E-5 / volume,
             'k_a1': k_a1 * 1E-5 / volume, 'k_a2': k_a2 * 1E-5 / volume})
        dradf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia', IFNAlpha_panel,
                                                parameters={'Ib': 0}, sf=scale_factor)
        drbdf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ib', IFNBeta_panel,
                                                parameters={'Ia': 0}, sf=scale_factor)
        drIadf = Mixed_Model.mixed_dose_response(times, 'Free_Ia', 'Ia', IFNAlpha_panel,
                                                 parameters={'Ib': 0}, sf=scale_factor)
        drIbdf = Mixed_Model.mixed_dose_response(times, 'Free_Ib', 'Ib', IFNBeta_panel,
                                                 parameters={'Ia': 0}, sf=scale_factor)

        if IFN_in_concentration:
            for d in drIadf.loc['Alpha'].index:
                for t in drIadf.loc['Alpha'].columns:
                    drIadf.loc['Alpha', t][d] = (
                    drIadf.loc['Alpha', t][d][0] / (6.022E23 * volume * 1E-9), drIadf.loc['Alpha', t][d][1])
            for d in drIbdf.loc['Beta'].index:
                for t in drIbdf.loc['Beta'].columns:
                    drIbdf.loc['Beta', t][d] = (
                    drIbdf.loc['Beta', t][d][0] / (6.022E23 * volume * 1E-9), drIbdf.loc['Beta', t][d][1])

        draIfnData = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
        drbIfnData = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})
        drIaIfnData = IfnData('custom', df=drIadf, conditions={'Alpha': {'Ib': 0}})
        drIbIfnData = IfnData('custom', df=drIbdf, conditions={'Beta': {'Ia': 0}})
        predictions[cell_densities[vidx]] = [draIfnData, drbIfnData, drIaIfnData, drIbIfnData]
    return predictions


def experimental_panel(cell_densities, IFNAlpha_panel, IFNBeta_panel, times, IFN_in_concentration=True):
    # ----------
    # Run model
    # ----------
    predictions = run_smooth_trajectories(cell_densities, IFNAlpha_panel, IFNBeta_panel, times)
    # -------------------------------
    # Plot model dose response curves
    # -------------------------------
    alpha_palette = sns.color_palette("deep", 6)
    beta_palette = sns.color_palette("deep", 6)

    # pSTAT plot
    STAT_plot = TimecoursePlot((len(IFNAlpha_panel), 2))
    for dose_idx, dose in enumerate(IFNAlpha_panel):
        color_counter = -1
        for density_idx, density in enumerate(cell_densities):
            color_counter += 1
            STAT_plot.add_trajectory(predictions[density][0], 'plot', '-', subplot_idx=(dose_idx, 0),
                                     label=r'IFN$\alpha$ at {}$\times 10^6$ cells/mL'.format(cell_densities[density_idx]/1E9),
                                     doseslice=dose, dose_species='Alpha', color=alpha_palette[color_counter],
                                     linewidth=2)
    for dose_idx, dose in enumerate(IFNBeta_panel):
        color_counter = -1
        for density_idx, density in enumerate(cell_densities):
            color_counter += 1
            STAT_plot.add_trajectory(predictions[density][1], 'plot', '-', subplot_idx=(dose_idx, 1),
                                     label=r'IFN$\beta$ at {}$\times 10^6$ cells/mL'.format(cell_densities[density_idx]/1E9),
                                     doseslice=dose, dose_species='Beta', color=beta_palette[color_counter],
                                     linewidth=2)
    fig, axes = STAT_plot.show_figure()
    for i in range(len(IFNAlpha_panel)):
        axes[i][0].set_title(r'IFN$\alpha$ at {} pM'.format(IFNAlpha_panel[i]))
        axes[i][1].set_title(r'IFN$\beta$ at {} pM'.format(IFNBeta_panel[i]))
    fig.set_size_inches(16, 9)
    fig.tight_layout()
    fig.savefig('varying_reaction_volume_STATtc.pdf')

    # Free extracellular IFN plot
    IFN_plot = TimecoursePlot((len(IFNAlpha_panel), 2))
    for ax in IFN_plot.axes.flat:
        ax.set_yscale('log')
    for i in range(len(IFNAlpha_panel)):
        if IFN_in_concentration:
            IFN_plot.axes[i][0].set_ylim(bottom=1E-4, top=max(IFNAlpha_panel)*1E-2)
            IFN_plot.axes[i][1].set_ylim(bottom=1E-4, top=max(IFNBeta_panel)*1E-2)
        else:
            IFN_plot.axes[i][0].set_ylim(bottom=1, top=max(IFNAlpha_panel) * 10 ** -12 * max(volume_panel) * 6.022E23)
            IFN_plot.axes[i][1].set_ylim(bottom=1, top=max(IFNBeta_panel) * 10 ** -12 * max(volume_panel) * 6.022E23)

    # Add fits
    for dose_idx, dose in enumerate(IFNAlpha_panel):
        color_counter = -1
        for density_idx, density in enumerate(cell_densities):
            color_counter += 1
            IFN_plot.add_trajectory(predictions[density][2], 'plot', '-', subplot_idx=(dose_idx, 0),
                                     label=r'{}$\times 10^6$ cells/mL'.format(cell_densities[density_idx]/1E9),
                                     doseslice=dose, dose_species='Alpha', color=alpha_palette[color_counter],
                                     linewidth=2)
    for dose_idx, dose in enumerate(IFNBeta_panel):
        color_counter = -1
        for density_idx, density in enumerate(cell_densities):
            color_counter += 1
            IFN_plot.add_trajectory(predictions[density][3], 'plot', '-', subplot_idx=(dose_idx, 1),
                                     label=r'{}$\times 10^6$ cells/mL'.format(cell_densities[density_idx]/1E9),
                                     doseslice=dose, dose_species='Beta', color=beta_palette[color_counter],
                                     linewidth=2)
    fig, axes = IFN_plot.show_figure()
    for i in range(len(IFNAlpha_panel)):
        axes[i][0].set_title(r'IFN$\alpha$ at {} pM'.format(IFNAlpha_panel[i]))
        axes[i][0].set_ylabel(r'Free IFN$\alpha$ (nM)')
        axes[i][1].set_title(r'IFN$\beta$ at {} pM'.format(IFNBeta_panel[i]))
        axes[i][1].set_ylabel(r'Free IFN$\beta$ (nM)')

    fig.set_size_inches(16, 12)
    fig.tight_layout()
    fig.savefig('varying_reaction_volume_IFNtc.pdf')

def time_to_reach_robot_threshold(cell_densities, IFNAlpha_panel, IFNBeta_panel, times, threshold=1E-3):
    predictions = run_smooth_trajectories(cell_densities, IFNAlpha_panel, IFNBeta_panel, times,
                                          IFN_in_concentration=True)
    alpha_times = {}
    beta_times = {}
    for cd in cell_densities:
        Iadf = predictions[cd][2]
        tau_record = []
        for d in Iadf.data_set.loc['Alpha'].index:
            tc = [i[0] for i in Iadf.data_set.loc['Alpha'].loc[d].values]
            time_pt = np.inf
            for pt_idx, pt in enumerate(tc):
                if pt < threshold:
                    time_pt = times[pt_idx]
                    break
            tau_record.append(time_pt)
        alpha_times[cd / 1E9] = tau_record
    for cd in cell_densities:
        Ibdf = predictions[cd][3]
        tau_record = []
        for d in Ibdf.data_set.loc['Beta'].index:
            tc = [i[0] for i in Ibdf.data_set.loc['Beta'].loc[d].values]
            time_pt = np.inf
            for pt_idx, pt in enumerate(tc):
                if pt < threshold:
                    time_pt = times[pt_idx]
                    break
            tau_record.append(time_pt)
        beta_times[cd / 1E9] = tau_record
    alpha_df = pd.DataFrame.from_dict(alpha_times, orient='index', columns=IFNAlpha_panel)
    beta_df = pd.DataFrame.from_dict(beta_times, orient='index', columns=IFNBeta_panel)
    alpha_df.to_csv('alpha_threshold_times.txt', sep='\t')
    beta_df.to_csv('beta_threshold_times.txt', sep='\t')
    return alpha_df, beta_df


def characteristic_time_figures(cell_densities, IFNAlpha_panel, IFNBeta_panel, times):
    predictions = run_smooth_trajectories(cell_densities, IFNAlpha_panel, IFNBeta_panel, times,
                                          IFN_in_concentration=False)
    # ----------------------------------
    # Make characteristic time heatmaps
    # ----------------------------------
    alpha_half_lives = {}
    beta_half_lives = {}
    for cd in cell_densities:
        Iadf = predictions[cd][2]
        tau_record = []
        for d in Iadf.data_set.loc['Alpha'].index:
            tc = [i[0] for i in Iadf.data_set.loc['Alpha'].loc[d].values]
            if not np.isnan(Iadf.data_set.loc['Alpha'].loc[d].values[0][1]):
                sigmas = [i[1] for i in Iadf.data_set.loc['Alpha'].loc[d].values]
            else:
                sigmas = None

            def f(x, tau):
                return [tc[0] * np.exp(-t / tau) for t in x]
            def get_halflife(ydata, times=times, sigmas=None):
                params, params_covariance = curve_fit(f, times, ydata, sigma=sigmas)
                return params[0]

            tau = get_halflife(tc, times, sigmas=sigmas)
            tau_record.append(tau)
            # Visual checking
            #plt.figure()
            #plt.plot(times, tc)
            #plt.plot(times, [f([i], tau)[0] for i in times], 'r')
            #yvals = [t for t in tc if t <= tc[0]/2]
            #plt.plot([tau for _ in range(len(yvals))], yvals, 'k--')
            #plt.show()

        alpha_half_lives[cd/1E9] = tau_record

        Ibdf = predictions[cd][3]
        tau_record = []
        for d in Ibdf.data_set.loc['Beta'].index:
            tc = [i[0] for i in Ibdf.data_set.loc['Beta'].loc[d].values]
            if not np.isnan(Ibdf.data_set.loc['Beta'].loc[d].values[0][1]):
                sigmas = [i[1] for i in Ibdf.data_set.loc['Beta'].loc[d].values]
            else:
                sigmas = None
            tau = get_halflife(tc, times, sigmas=sigmas)
            tau_record.append(tau)
        beta_half_lives[cd/1E9] = tau_record

    alpha_heatmap_df = pd.DataFrame.from_dict(alpha_half_lives, orient='index', columns=IFNAlpha_panel)
    beta_heatmap_df = pd.DataFrame.from_dict(beta_half_lives, orient='index', columns=IFNBeta_panel)

    # Plot heatmaps
    f, (ax1, ax2, axcb) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 0.08]})
    ax1.get_shared_y_axes().join(ax2)
    g1 = sns.heatmap(alpha_heatmap_df, cmap="viridis", cbar=False, ax=ax1, vmin=100, vmax=5000)
    g1.set_ylabel('')
    g1.set_xlabel('')
    g2 = sns.heatmap(beta_heatmap_df, cmap="viridis", ax=ax2, cbar_ax=axcb, vmin=100, vmax=5000)
    g2.set_ylabel('')
    g2.set_xlabel('')
    g2.set_yticks([])

    # may be needed to rotate the ticklabels correctly:
    species = [r'IFN$\alpha$', r'IFN$\beta$']
    for ax_idx, ax in enumerate([g1, g2]):
        tl = ax.get_xticklabels()
        ax.set_xticklabels(tl, rotation=90)
        tly = ax.get_yticklabels()
        ax.set_yticklabels(tly, rotation=0)
        ax.title.set_text(species[ax_idx])

    plt.show()
    f.tight_layout()
    f.savefig('halflife_heatmap.pdf')

    return


if __name__ == '__main__':
    # Experimental parameters
    #cell_densities = [0.5E9, 5E9, 10E9, 20E9]
    cell_densities = [0.25E9, 2.5E9, 5E9, 10E9] # divide by 2
    #cell_densities = [0.33E9, 3.33E9, 6.67E9, 13.33E9] # divide by 1.5
    cell_densities = [0.75*i for i in cell_densities]

    volume_panel = [1/i for i in cell_densities]
    IFNBeta_EC50 = 4  # pM
    IFNAlpha_EC50 = 1000  # pM
    IFNBeta_panel = [0.5 * IFNBeta_EC50, IFNBeta_EC50, 5 * IFNBeta_EC50, 10 * IFNBeta_EC50]
    IFNAlpha_panel = [0.1 * IFNAlpha_EC50, IFNAlpha_EC50, 5 * IFNAlpha_EC50, 10 * IFNAlpha_EC50]

    # Model predictions
    times = np.arange(0, 60 * 12 + 0.5, 0.5)
    time_to_reach_robot_threshold(cell_densities, IFNAlpha_panel, IFNBeta_panel, times)
    characteristic_time_figures(cell_densities, IFNAlpha_panel, IFNBeta_panel, times)
    times = [0, 20, 40, 60, 80, 100, 120, 160, 200, 300, 400, 500, 720] # 12 hours total, 20 min intervals minimum
    experimental_panel(cell_densities, IFNAlpha_panel, IFNBeta_panel, times)
    #testing_specific_values()