from ifnclass.ifndata import IfnData
from ifnclass.ifnplot import DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import seaborn as sns
import load_model as lm
import copy
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    out_dir = os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_3')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fname = out_dir + os.sep + 'negative_feedback_figure.pdf'

    # # This is the best fit parameters for GAB aligned data
    # Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
    # # Optimal parameters for fitting mean GAB data
    # default_parameters = {'R2': 4920, 'R1': 1200,
    #  'k_a1': 2.0e-13, 'k_a2': 1.328e-12, 'k_d3': 1.13e-4, 'k_d4': 0.9,
    #  'kSOCSon': 5e-08, 'kpu': 0.0022, 'kpa': 2.36e-06,
    #  'ka1': 3.3e-15, 'ka2': 1.85e-12, 'kd4': 2.0,
    #  'kd3': 6.52e-05,
    #  'kint_a': 0.0015, 'kint_b': 0.002,
    #  'krec_a1': 0.01, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05}
    #
    # scale_factor = 1.46182313424
    #
    # Mixed_Model.set_parameters(default_parameters)
    # Mixed_Model.default_parameters.update(default_parameters)

    # --------------------
    # Set up Model
    # --------------------
    Mixed_Model, DR_method = lm.load_model()
    scale_factor, DR_KWARGS, PLOT_KWARGS, ENSEMBLE = lm.SCALE_FACTOR, lm.DR_KWARGS, lm.PLOT_KWARGS, lm.ENSEMBLE
    # initial_parameters = {'k_a1': 4.98E-14 * 1.33, 'k_a2': 8.30e-13 * 2,
    #                       'k_d4': 0.006 * 3.8,
    #                       'kpu': 0.00095,
    #                       'ka2': 4.98e-13 * 1.33, 'kd4': 0.3 * 2.867,
    #                       'kint_a': 0.000124, 'kint_b': 0.00056,
    #                       'krec_a1': 0.0028, 'krec_a2': 0.01,
    #                       'krec_b1': 0.005, 'krec_b2': 0.05}
    # Mixed_Model.set_parameters(initial_parameters)

    if ENSEMBLE:
        raise NotImplementedError("Ensemble sampling not yet implemented")
    else:
        Mixed_Model.model_1.default_parameters = copy.deepcopy(Mixed_Model.model_1.parameters)
        Mixed_Model.model_2.default_parameters = copy.deepcopy(Mixed_Model.model_2.parameters)

    # --------------------
    # Run Simulations
    # --------------------
    times = [60]
    # Control Dose-Response
    Mixed_Model.set_parameters({'kSOCSon': 0, 'kIntBasal_r1': 0, 'kIntBasal_r2': 0, 'kint_a': 0, 'kint_b': 0})
    dradf = DR_method(times, 'TotalpSTAT', 'Ia', list(logspace(-2, 8)),
                      parameters={'Ib': 0}, return_type='DataFrame', dataframe_labels='Alpha',
                      scale_factor=scale_factor)
    drbdf = DR_method(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 8)),
                      parameters={'Ia': 0}, return_type='DataFrame', dataframe_labels='Beta',
                      scale_factor=scale_factor)

    # Show internalization effects
    Mixed_Model.reset_global_parameters()
    Mixed_Model.set_parameters({'kSOCSon': 0})
    dradf_int = DR_method(times, 'TotalpSTAT', 'Ia', list(logspace(-2, 8)),
                          parameters={'Ib': 0}, return_type='DataFrame', dataframe_labels='Alpha',
                          scale_factor=scale_factor)
    drbdf_int = DR_method(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 8)),
                          parameters={'Ia': 0}, return_type='DataFrame', dataframe_labels='Beta',
                          scale_factor=scale_factor)

    # Show SOCS effects
    Mixed_Model.reset_global_parameters()
    Mixed_Model.set_parameters({'kIntBasal_r1': 0, 'kIntBasal_r2': 0, 'kint_a': 0, 'kint_b': 0})
    dradf_SOCS = DR_method(times, 'TotalpSTAT', 'Ia', list(logspace(-2, 8)),
                           parameters={'Ib': 0}, return_type='DataFrame', dataframe_labels='Alpha',
                           scale_factor=scale_factor)
    drbdf_SOCS = DR_method(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 8)),
                           parameters={'Ia': 0}, return_type='DataFrame', dataframe_labels='Beta',
                           scale_factor=scale_factor)

    # Make IfnData objects
    dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})
    dra60_int = IfnData('custom', df=dradf_int, conditions={'Alpha': {'Ib': 0}})
    drb60_int = IfnData('custom', df=drbdf_int, conditions={'Beta': {'Ia': 0}})
    dra60_SOCS = IfnData('custom', df=dradf_SOCS, conditions={'Alpha': {'Ib': 0}})
    drb60_SOCS = IfnData('custom', df=drbdf_SOCS, conditions={'Beta': {'Ia': 0}})

    # --------------------
    # Make Plot
    # --------------------
    dr_plot = DoseresponsePlot((1, 1))
    alpha_mask = []
    beta_mask = []
    for idx, t in enumerate([el for el in times]):
        if t not in alpha_mask:
            dr_plot.add_trajectory(dra60, t, 'plot', alpha_palette[5], (0, 0), 'Alpha',  linewidth=2.0)
            dr_plot.add_trajectory(dra60_int, t, 'plot', '--', (0, 0), 'Alpha', color=alpha_palette[5], linewidth=2.0)
            dr_plot.add_trajectory(dra60_SOCS, t, 'plot', ':', (0, 0), 'Alpha', color=alpha_palette[5], linewidth=2.0)
        if t not in beta_mask:
            dr_plot.add_trajectory(drb60, t, 'plot', beta_palette[5], (0, 0), 'Beta', linewidth=2.0)
            dr_plot.add_trajectory(drb60_int, t, 'plot', '--', (0, 0), 'Beta', color=beta_palette[5], linewidth=2.0)
            dr_plot.add_trajectory(drb60_SOCS, t, 'plot', ':', (0, 0), 'Beta', color=beta_palette[5], linewidth=2.0)
    # Legend:
    plt.scatter([], [], color=alpha_palette[5], label=r'IFN$\alpha$2', figure=dr_plot.fig)
    plt.scatter([], [], color=beta_palette[5], label=r'IFN$\beta$', figure=dr_plot.fig)
    plt.plot([], [], c='grey', label='No Feedback', linewidth=2.0, figure=dr_plot.fig)
    plt.plot([], [], '--', c='grey', label='Effect of Internalization', linewidth=2.0, figure=dr_plot.fig)
    plt.plot([], [], ':', c='grey', label='Effect of SOCS', linewidth=2.0, figure=dr_plot.fig)

    # Plot formatting
    dr_plot.fig.set_size_inches((5, 4))
    dr_plot.axes.set_title('Effects of Negative Feedback')
    dr_plot.axes.set_ylabel('pSTAT')
    dr_plot.axes.spines['top'].set_visible(False)
    dr_plot.axes.spines['right'].set_visible(False)
    dr_plot.save_figure(save_dir=fname, tight=True)
