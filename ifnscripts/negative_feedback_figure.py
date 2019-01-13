from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import seaborn as sns


if __name__ == '__main__':
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')

    # This is the fitting_2_5 script best fit parameters, best at fitting Ratnadeep's B cell data
    Mixed_Model.set_parameters(
        {'R2': 2300 * 2.5,
         'R1': 1800 * 1.8, 'k_d4': 0.06, 'kint_b': 0.0003,
         'kpu': 0.0028,
         'krec_b1': 0.001, 'krec_b2': 0.01,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 4, 'kSOCSon': 0.9e-8,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3, 'kd4': 1.0, 'kd3': 0.001,
         'kint_a': 0.0014, 'krec_a1': 9e-03, 'krec_a2': 0.05})
    scale_factor = 0.036
    scale_data = lambda q: (scale_factor*q[0], scale_factor*q[1])

    # Produce plots
    times = [60]
    # No Negative Feedback
    Mixed_Model.set_parameters({'kSOCSon': 0, 'kIntBasal_r1': 0, 'kIntBasal_r2': 0, 'kint_a': 0, 'kint_b': 0})

    dradf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-2, 8)),
                                           parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 8)),
                                           parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')

    # Show internalization effects
    Mixed_Model.set_parameters({'kIntBasal_r1': 0.0001, 'kIntBasal_r2': 0.00002, 'kint_a': 0.0014, 'kint_b': 0.0003})
    dradf_int = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-2, 8)),
                                           parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf_int = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 8)),
                                           parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')

    # Show SOCS effects
    Mixed_Model.set_parameters({'kSOCSon': 0.9e-8, 'kIntBasal_r1': 0, 'kIntBasal_r2': 0, 'kint_a': 0, 'kint_b': 0})
    dradf_rec = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-2, 8)),
                                           parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf_rec = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 8)),
                                           parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')
    for i in range(len(times)):
        dradf.loc['Alpha'].iloc[:, i] = dradf.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf.loc['Beta'].iloc[:, i] = drbdf.loc['Beta'].iloc[:, i].apply(scale_data)
        dradf_int.loc['Alpha'].iloc[:, i] = dradf_int.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf_int.loc['Beta'].iloc[:, i] = drbdf_int.loc['Beta'].iloc[:, i].apply(scale_data)
        dradf_rec.loc['Alpha'].iloc[:, i] = dradf_rec.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf_rec.loc['Beta'].iloc[:, i] = drbdf_rec.loc['Beta'].iloc[:, i].apply(scale_data)

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
            dr_plot.add_trajectory(dra60_int, t, 'plot', alpha_palette[3], (0, 0), 'Alpha', label='Alpha - Internalization', linewidth=2.0)
            dr_plot.add_trajectory(dra60_rec, t, 'plot', alpha_palette[1], (0, 0), 'Alpha', label='Alpha - SOCS', linewidth=2.0)
        if t not in beta_mask:
            dr_plot.add_trajectory(drb60, t, 'plot', beta_palette[5], (0, 0), 'Beta', label='Beta - No Feedback', linewidth=2.0)
            dr_plot.add_trajectory(drb60_int, t, 'plot', beta_palette[3], (0, 0), 'Beta', label='Beta - Internalization', linewidth=2.0)
            dr_plot.add_trajectory(drb60_rec, t, 'plot', beta_palette[1], (0, 0), 'Beta', label='Beta - SOCS', linewidth=2.0)

    dr_plot.show_figure(save_flag=False)
