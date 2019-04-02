from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import seaborn as sns
from ifnclass.ifnfit import StepwiseFit
import pandas as pd


if __name__ == '__main__':
    # Import data
    newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
    newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
    newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
    newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")

    alignment = DataAlignment()
    alignment.add_data([newdata_4, newdata_3, newdata_2, newdata_1])
    alignment.align()
    alignment.get_scaled_data()
    mean_data = alignment.summarize_data()

    # ---------------------------
    # Plot simple model:
    # ---------------------------
    Simple_Model = IfnModel('Mixed_IFN_ppCompatible')
    # Set best fit parameters
    Simple_Model.set_parameters({'R2': 4920, 'R1': 1200,
                                'k_a1': 2.0e-13, 'k_a2': 1.328e-12, 'k_d3': 1.13e-4, 'k_d4': 0.9,
                                'kSOCSon': 5e-08, 'kpu': 0.0022, 'kpa': 2.36e-06,
                                'ka1': 3.3e-15, 'ka2': 1.85e-12, 'kd4': 2.0,
                                'kd3': 6.52e-05,
                                'kint_a':  0.0015, 'kint_b': 0.002,
                                'krec_a1': 0.01, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05})
    scale_factor = 1.46182313424
    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])

    times = [2.5, 5.0, 7.5, 10., 20., 60.]
    dradf = Simple_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-1, 5)),
                                      parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf = Simple_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, log10(2000))),
                                      parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')
    for i in range(len(times)):
        dradf.loc['Alpha'].iloc[:, i] = dradf.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf.loc['Beta'].iloc[:, i] = drbdf.loc['Beta'].iloc[:, i].apply(scale_data)
    dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})
    print(pd.concat([dradf, dradf]))
    total_data = IfnData('custom', df=pd.concat([dradf, dradf]), conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
    # Plot detailed model vs data to validate parameter choices
    new_fit = DoseresponsePlot((1, 2))
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)
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
            new_fit.add_trajectory(mean_data, t, 'scatter', alpha_palette[idx], (0, 0), 'Alpha', dn=1)
            new_fit.add_trajectory(mean_data, t, 'plot', '--', (0, 0), 'Alpha', dn=1, color=alpha_palette[idx], label='Alpha '+str(t))
        if t not in beta_mask:
            new_fit.add_trajectory(mean_data, t, 'scatter', beta_palette[idx], (0, 1), 'Beta', dn=1)
            new_fit.add_trajectory(mean_data, t, 'plot', '--', (0, 1), 'Beta', dn=1, color=beta_palette[idx], label='Beta '+str(t))

    new_fit.show_figure(save_flag=False)

    # ---------------------------
    # Now repeat for detailed model:
    # ---------------------------
    Detailed_Model = IfnModel('Mixed_IFN_detailed')

    Detailed_Model.set_parameters({'R2': 4920, 'R1': 1200,
                                'k_a1': 2.0e-13, 'k_a2': 1.328e-12, 'k_d3': 1.13e-4, 'k_d4': 0.9,
                                'kSOCSon': 5e-08, 'kSTATunbinding': 0.0022, 'kSTATbinding': 2.36e-06,
                                'ka1': 3.3e-15, 'ka2': 1.85e-12, 'kd4': 2.0,
                                'kd3': 6.52e-05,
                                'kint_a':  0.0015, 'kint_b': 0.002,
                                'krec_a1': 0.01, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05})

    # Now match the simple model predictions using only unconstrained parameters:
    stepfit = StepwiseFit(Detailed_Model, mean_data,
                          {'kpa': (0.01, 10), 'kpu': (1e-5, 1e-1),
                           'kloc': (1.25e-5, 1.25e-1), 'kdeloc': (1e-4, 1),
                           'kSOCSmRNA': (1e-5, 1e-1),
                           'mRNAdeg': (5e-6, 5e-02), 'mRNAtrans': (1e-5, 1e-1)}, n=8)
    best_parameters, scale_factor = stepfit.fit()
    print(best_parameters)
    print(scale_factor)
    Detailed_Model = stepfit.model
    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])

    # Make detailed model predictions
    times = [2.5, 5., 7.5, 10., 20., 60.]
    dradf = Detailed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-1, 5)),
                                     parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf = Detailed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, log10(2000))),
                                     parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')
    for i in range(len(times)):
        dradf.loc['Alpha'].iloc[:, i] = dradf.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf.loc['Beta'].iloc[:, i] = drbdf.loc['Beta'].iloc[:, i].apply(scale_data)
    dra60_d = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60_d = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    # Plot detailed vs data to validate parameter choices
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
            new_fit.add_trajectory(mean_data, t, 'scatter', alpha_palette[idx], (0, 0), 'Alpha', dn=1)
            new_fit.add_trajectory(mean_data, t, 'plot', '--', (0, 0), 'Alpha', dn=1, color=alpha_palette[idx], label='Alpha '+str(t))
        if t not in beta_mask:
            new_fit.add_trajectory(mean_data, t, 'scatter', beta_palette[idx], (0, 1), 'Beta', dn=1)
            new_fit.add_trajectory(mean_data, t, 'plot', '--', (0, 1), 'Beta', dn=1, color=beta_palette[idx], label='Beta '+str(t))
    new_fit.show_figure(save_flag=False)

    # ----------------------------------------
    # Finally, plot both models in comparison
    # ----------------------------------------
    new_fit = DoseresponsePlot((1, 2))
    alpha_mask = [2.5, 7.5]
    beta_mask = [2.5, 7.5]
    # Add fits
    for idx, t in enumerate([str(el) for el in times]):
        if t not in [str(el) for el in alpha_mask]:
            new_fit.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha')
            new_fit.add_trajectory(dra60_d, t, 'plot', '--', (0, 0), 'Alpha', color=alpha_palette[idx])
        if t not in [str(el) for el in beta_mask]:
            new_fit.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1), 'Beta')
            new_fit.add_trajectory(drb60_d, t, 'plot', '--', (0, 1), 'Beta', color=beta_palette[idx])

    new_fit.add_trajectory(dra60, 60., 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='Alpha Simple Model')
    new_fit.add_trajectory(dra60_d, 60., 'plot', '--', (0, 0), 'Alpha', color=alpha_palette[idx], label='Alpha Detailed Model')
    new_fit.add_trajectory(drb60, 60., 'plot', beta_palette[idx], (0, 1), 'Beta', label='Beta Simple Model')
    new_fit.add_trajectory(drb60_d, 60., 'plot', '--', (0, 1), 'Beta', color=beta_palette[idx], label='Beta Detailed Model')
    new_fit.show_figure(save_flag=False)




