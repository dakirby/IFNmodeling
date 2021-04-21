from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import seaborn as sns
import load_model as lm
import copy

if __name__ == '__main__':
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

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
    scale_factor, DR_KWARGS, PLOT_KWARGS = lm.SCALE_FACTOR, lm.DR_KWARGS, lm.PLOT_KWARGS
    Mixed_Model.model_1.default_parameters = copy.deepcopy(Mixed_Model.model_1.parameters)
    Mixed_Model.model_2.default_parameters = copy.deepcopy(Mixed_Model.model_2.parameters)

    # Produce plots
    times = [60]
    # No Negative Feedback
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


    # Show SOCS tuned to alpha internalization
    Mixed_Model.set_parameters({'kSOCSon': Mixed_Model.model_1.parameters['kSOCSon'] * 2.6})
    dradf_match_1 = DR_method(times, 'TotalpSTAT', 'Ia', list(logspace(-2, 8)),
                              parameters={'Ib': 0}, return_type='DataFrame', dataframe_labels='Alpha',
                              scale_factor=scale_factor)
    drbdf_match_1 = DR_method(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 8)),
                              parameters={'Ia': 0}, return_type='DataFrame', dataframe_labels='Beta',
                              scale_factor=scale_factor)

    # Show internalization matched to SOCS
    Mixed_Model.reset_global_parameters()
    Mixed_Model.set_parameters({'kint_a': Mixed_Model.model_1.parameters['kint_a'] * 0.4,
                                'kint_b': Mixed_Model.model_1.parameters['kint_b'] * 0.3,
                                'kSOCSon': 0})
    dradf_match_2 = DR_method(times, 'TotalpSTAT', 'Ia', list(logspace(-2, 8)),
                              parameters={'Ib': 0}, return_type='DataFrame', dataframe_labels='Alpha',
                              scale_factor=scale_factor)
    drbdf_match_2 = DR_method(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 8)),
                              parameters={'Ia': 0}, return_type='DataFrame', dataframe_labels='Beta',
                              scale_factor=scale_factor)

    # Make IfnData objects
    dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})
    dra60_int = IfnData('custom', df=dradf_int, conditions={'Alpha': {'Ib': 0}})
    drb60_int = IfnData('custom', df=drbdf_int, conditions={'Beta': {'Ia': 0}})
    dra60_SOCS = IfnData('custom', df=dradf_SOCS, conditions={'Alpha': {'Ib': 0}})
    drb60_SOCS = IfnData('custom', df=drbdf_SOCS, conditions={'Beta': {'Ia': 0}})
    dra60_match1 = IfnData('custom', df=dradf_match_1, conditions={'Alpha': {'Ib': 0}})
    drb60_match1 = IfnData('custom', df=drbdf_match_1, conditions={'Beta': {'Ia': 0}})
    dra60_match2 = IfnData('custom', df=dradf_match_2, conditions={'Alpha': {'Ib': 0}})
    drb60_match2 = IfnData('custom', df=drbdf_match_2, conditions={'Beta': {'Ia': 0}})

    dr_plot = DoseresponsePlot((1, 3))
    alpha_mask = []
    beta_mask = []
    # Add fits
    # Panel A: Effect of SOCS vs effect of internalization
    for idx, t in enumerate([el for el in times]):
        if t not in alpha_mask:
            dr_plot.add_trajectory(dra60, t, 'plot', alpha_palette[5], (0, 0), 'Alpha', label='Alpha - No Feedback', linewidth=2.0)
            dr_plot.add_trajectory(dra60_int, t, 'plot', '--', (0, 0), 'Alpha', color=alpha_palette[3],
                                   label='Internalization only', linewidth=2.0, )
            dr_plot.add_trajectory(dra60_SOCS, t, 'plot', ':', (0, 0), 'Alpha', color=alpha_palette[1], label='SOCS only', linewidth=2.0)
        if t not in beta_mask:
            dr_plot.add_trajectory(drb60, t, 'plot', beta_palette[5], (0, 0), 'Beta', label='Beta - No Feedback', linewidth=2.0)
            dr_plot.add_trajectory(drb60_int, t, 'plot', '--', (0, 0), 'Beta', color=beta_palette[3],
                                   label='Internalization only', linewidth=2.0)
            dr_plot.add_trajectory(drb60_SOCS, t, 'plot', ':', (0, 0), 'Beta', color=beta_palette[1], label='SOCS only', linewidth=2.0)

    # Panel B: Effect of SOCS vs matched to alpha internalization
    for idx, t in enumerate([el for el in times]):
        if t not in alpha_mask:
            dr_plot.add_trajectory(dra60_int, t, 'plot', '--', (0, 1), 'Alpha', color=alpha_palette[3],
                                   label='Internalization only', linewidth=2.0, )
            dr_plot.add_trajectory(dra60_match1, t, 'plot', ':', (0, 1), 'Alpha', color=alpha_palette[1],
                                   label='SOCS matched to internalization', linewidth=2.0)
        if t not in beta_mask:
            dr_plot.add_trajectory(drb60_int, t, 'plot', '--', (0, 1), 'Beta', color=beta_palette[3],
                                   label='Internalization only', linewidth=2.0)
            dr_plot.add_trajectory(drb60_match1, t, 'plot', ':', (0, 1), 'Beta', color=beta_palette[1],
                                   label='SOCS matched to internalization', linewidth=2.0)

    # Panel C: Effect of internalization matched to SOCS
    for idx, t in enumerate([el for el in times]):
        if t not in alpha_mask:
            dr_plot.add_trajectory(dra60_match2, t, 'plot', '--', (0, 2), 'Alpha', color=alpha_palette[3],
                                   label='Internalization matched to SOCS', linewidth=2.0, )
            dr_plot.add_trajectory(dra60_SOCS, t, 'plot', ':', (0, 2), 'Alpha', color=alpha_palette[1], label='SOCS only', linewidth=2.0)
        if t not in beta_mask:
            dr_plot.add_trajectory(drb60_match2, t, 'plot', '--', (0, 2), 'Beta', color=beta_palette[3],
                                   label='Internalization matched to SOCS', linewidth=2.0)
            dr_plot.add_trajectory(drb60_SOCS, t, 'plot', ':', (0, 2), 'Beta', color=beta_palette[1], label='SOCS only', linewidth=2.0)

    #dr_plot.fig.suptitle('Internalization vs SOCS')
    dr_plot.fig.set_size_inches((14, 4))
    dr_plot.axes[0].set_title('Effects of Negative Feedback')
    dr_plot.axes[1].set_title('Effect of SOCS Matched to\n Alpha Internalization')
    dr_plot.axes[2].set_title('Effect of Internalization Matched to SOCS')
    dr_plot.show_figure()
