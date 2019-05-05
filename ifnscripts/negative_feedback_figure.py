from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import seaborn as sns


if __name__ == '__main__':
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    # This is the best fit parameters for GAB aligned data
    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
    # Optimal parameters for fitting mean GAB data
    default_parameters = {'R2': 4920, 'R1': 1200,
     'k_a1': 2.0e-13, 'k_a2': 1.328e-12, 'k_d3': 1.13e-4, 'k_d4': 0.9,
     'kSOCSon': 5e-08, 'kpu': 0.0022, 'kpa': 2.36e-06,
     'ka1': 3.3e-15, 'ka2': 1.85e-12, 'kd4': 2.0,
     'kd3': 6.52e-05,
     'kint_a': 0.0015, 'kint_b': 0.002,
     'krec_a1': 0.01, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05}

    scale_factor = 1.46182313424

    Mixed_Model.set_parameters(default_parameters)
    Mixed_Model.default_parameters.update(default_parameters)

    # Produce plots
    times = [60]
    # No Negative Feedback
    Mixed_Model.set_parameters({'kSOCSon': 0, 'kIntBasal_r1': 0, 'kIntBasal_r2': 0, 'kint_a': 0, 'kint_b': 0})

    dradf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-2, 8)),
                                     parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha',
                                     scale_factor=scale_factor)
    drbdf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 8)),
                                     parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta',
                                     scale_factor=scale_factor)

    # Show internalization effects
    Mixed_Model.reset_parameters()
    Mixed_Model.set_parameters({'kSOCSon': 0})
    dradf_int = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-2, 8)),
                                         parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha',
                                         scale_factor=scale_factor)
    drbdf_int = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 8)),
                                         parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta',
                                         scale_factor=scale_factor)

    # Show SOCS effects
    Mixed_Model.reset_parameters()
    Mixed_Model.set_parameters({'kIntBasal_r1': 0, 'kIntBasal_r2': 0, 'kint_a': 0, 'kint_b': 0})
    #Mixed_Model.set_parameters({'kSOCS': Mixed_Model.parameters['kSOCS'] * 2.5})
    dradf_rec = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-2, 8)),
                                         parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha',
                                         scale_factor = scale_factor)
    drbdf_rec = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 8)),
                                         parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta',
                                         scale_factor=scale_factor)

    dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})
    dra60_int = IfnData('custom', df=dradf_int, conditions={'Alpha': {'Ib': 0}})
    drb60_int = IfnData('custom', df=drbdf_int, conditions={'Beta': {'Ia': 0}})
    dra60_rec = IfnData('custom', df=dradf_rec, conditions={'Alpha': {'Ib': 0}})
    drb60_rec = IfnData('custom', df=drbdf_rec, conditions={'Beta': {'Ia': 0}})

    dr_plot = DoseresponsePlot((1, 1))
    alpha_mask = []
    beta_mask = []
    # Add fits
    for idx, t in enumerate([el for el in times]):
        if t not in alpha_mask:
            dr_plot.add_trajectory(dra60, t, 'plot', alpha_palette[5], (0, 0), 'Alpha', label='Alpha - No Feedback', linewidth=2.0)
            dr_plot.add_trajectory(dra60_int, t, 'plot', '--', (0, 0), 'Alpha', color=alpha_palette[3],
                                   label='Internalization only', linewidth=2.0, )
            dr_plot.add_trajectory(dra60_rec, t, 'plot', ':', (0, 0), 'Alpha', color=alpha_palette[1], label='SOCS only', linewidth=2.0)
        if t not in beta_mask:
            dr_plot.add_trajectory(drb60, t, 'plot', beta_palette[5], (0, 0), 'Beta', label='Beta - No Feedback', linewidth=2.0)
            dr_plot.add_trajectory(drb60_int, t, 'plot', '--', (0, 0), 'Beta', color=beta_palette[3],
                                   label='Internalization only', linewidth=2.0)
            dr_plot.add_trajectory(drb60_rec, t, 'plot', ':', (0, 0), 'Beta', color=beta_palette[1], label='SOCS only', linewidth=2.0)
    dr_plot.fig.suptitle('Internalization vs SOCS')
    dr_plot.show_figure()
