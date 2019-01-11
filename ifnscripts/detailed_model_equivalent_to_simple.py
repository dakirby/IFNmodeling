from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import seaborn as sns
from smooth_B6_IFN import * # Imports smoothed data to fit to


if __name__ == '__main__':
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)
    newdata = IfnData("20181113_B6_IFNs_Dose_Response_Bcells")

    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')

    Mixed_Model.set_parameters(
        {'R2': 2300 * 2.5,
         'R1': 1800 * 1.8, 'k_d4': 0.06, 'kint_b': 0.0003,
         'kpu': 0.0028,
         'krec_b1': 0.001, 'krec_b2': 0.01,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 4, 'kSOCSon': 0.9e-8,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3, 'kd4': 1.0, 'kd3': 0.001,
         'kint_a': 0.0014, 'krec_a1': 9e-03, 'krec_a2': 0.05})

    # Now try to improve the fit:
    Mixed_Model.set_parameters({'kpu': 0.0028, 'R2': 2300 * 2.5, 'R1': 1800 * 1.8, 'k_d4': 0.06, 'kint_b': 0.0003, 'krec_b1': 0.001,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 4, 'kSOCSon': 0.9e-8,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3,
         'kint_a': 0.0014, 'krec_a1': 9e-03})
    scale_factor = 0.036

    # Additional fitting
    scale_data = lambda q: (scale_factor*q[0], scale_factor*q[1])
    times = [2.5, 5, 7.5, 10, 20, 60]
    dradf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-1, 5)),
                                           parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 4)),
                                           parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')

    for i in range(len(times)):
        dradf.loc['Alpha'].iloc[:, i] = dradf.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf.loc['Beta'].iloc[:, i] = drbdf.loc['Beta'].iloc[:, i].apply(scale_data)

    dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    new_fit = DoseresponsePlot((1, 2))
    alpha_mask = [2.5, 7.5]
    beta_mask = [2.5, 7.5]
    # Add fits
    for idx, t in enumerate([str(el) for el in times]):
        if t not in [str(el) for el in alpha_mask]:
            new_fit.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='Alpha '+t)
        if t not in [str(el) for el in beta_mask]:
            new_fit.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label='Beta '+t)
    # Add data
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(newdata, t, 'scatter', alpha_palette[idx], (0, 0), 'Alpha', dn=1)
            new_fit.add_trajectory(newdata, t, 'plot', '--', (0, 0), 'Alpha', dn=1, color=alpha_palette[idx], label='Alpha '+str(t))
        if t not in beta_mask:
            new_fit.add_trajectory(newdata, t, 'scatter', beta_palette[idx], (0, 1), 'Beta', dn=1)
            new_fit.add_trajectory(newdata, t, 'plot', '--', (0, 1), 'Beta', dn=1, color=beta_palette[idx], label='Beta '+str(t))

    new_fit.show_figure(save_flag=False)

    # ---------------------------
    # Now repeat for detailed model:
    # ---------------------------
    Detailed_Model = IfnModel('Mixed_IFN_detailed')

    Detailed_Model.set_parameters(
        {'R2': 2300 * 2.5,
         'R1': 1800 * 1.8, 'k_d4': 0.06, 'kint_b': 0.0003,
         'kpu': 0.0028,
         'krec_b1': 0.001, 'krec_b2': 0.01,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 4, 'kSOCSon': 0.9e-8,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3, 'kd4': 1.0, 'kd3': 0.001,
         'kint_a': 0.0014, 'krec_a1': 9e-03, 'krec_a2': 0.05})

    # Now try to improve the fit:
    Detailed_Model.set_parameters(
        {'kpu': 0.0028, 'R2': 2300 * 2.5, 'R1': 1800 * 1.8, 'k_d4': 0.06, 'kint_b': 0.0008, 'krec_b1': 0.001,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 6, 'kSOCSon': 0.9e-8,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3,
         'kint_a': 0.0014, 'krec_a1': 9e-03})
    scale_factor = 0.1375

    # Additional fitting
    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])
    times = [2.5, 5, 7.5, 10, 20, 60]
    dradf = Detailed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-1, 5)),
                                     parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf = Detailed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 4)),
                                     parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')

    for i in range(len(times)):
        dradf.loc['Alpha'].iloc[:, i] = dradf.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf.loc['Beta'].iloc[:, i] = drbdf.loc['Beta'].iloc[:, i].apply(scale_data)

    dra60_d = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60_d = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    new_fit = DoseresponsePlot((1, 2))
    alpha_mask = [2.5, 7.5]
    beta_mask = [2.5, 7.5]
    # Add fits
    for idx, t in enumerate([str(el) for el in times]):
        if t not in [str(el) for el in alpha_mask]:
            new_fit.add_trajectory(dra60_d, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='Alpha ' + t)
        if t not in [str(el) for el in beta_mask]:
            new_fit.add_trajectory(drb60_d, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label='Beta ' + t)
    # Add data
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(newdata, t, 'scatter', alpha_palette[idx], (0, 0), 'Alpha', dn=1)
            new_fit.add_trajectory(newdata, t, 'plot', '--', (0, 0), 'Alpha', dn=1, color=alpha_palette[idx], label='Alpha '+str(t))
        if t not in beta_mask:
            new_fit.add_trajectory(newdata, t, 'scatter', beta_palette[idx], (0, 1), 'Beta', dn=1)
            new_fit.add_trajectory(newdata, t, 'plot', '--', (0, 1), 'Beta', dn=1, color=beta_palette[idx], label='Beta '+str(t))
    new_fit.show_figure(save_flag=False)

    # ----------------------------------------
    # Finally, plot both models in comparison
    # ----------------------------------------
    new_fit = DoseresponsePlot((1, 2))
    alpha_mask = [2.5, 7.5]
    beta_mask = [2.5, 7.5, 60]
    # Add fits
    for idx, t in enumerate([str(el) for el in times]):
        if t not in [str(el) for el in alpha_mask]:
            new_fit.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha')
            new_fit.add_trajectory(dra60_d, t, 'plot', '--', (0, 0), 'Alpha', color=alpha_palette[idx])
        if t not in [str(el) for el in beta_mask]:
            new_fit.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1), 'Beta')
            new_fit.add_trajectory(drb60_d, t, 'plot', '--', (0, 1), 'Beta', color=beta_palette[idx])

    new_fit.add_trajectory(dra60, 60, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='Alpha Simple Model')
    new_fit.add_trajectory(dra60_d, 60, 'plot', '--', (0, 0), 'Alpha', color=alpha_palette[idx], label='Alpha Detailed Model')
    new_fit.add_trajectory(drb60, 60, 'plot', beta_palette[idx], (0, 1), 'Beta', label='Beta Simple Model')
    new_fit.add_trajectory(drb60_d, 60, 'plot', '--', (0, 1), 'Beta', color=beta_palette[idx], label='Beta Detailed Model')
    new_fit.show_figure(save_flag=False)




