from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import DoseresponsePlot
import seaborn as sns
from numpy import logspace, log10


if __name__ == '__main__':
    Mixed_Model = IfnModel('Mixed_IFN_Internalized_Signaling')
    # Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')

    alpha_palette = sns.color_palette("Reds", 3)
    beta_palette = sns.color_palette("Greens", 3)

    # Arbitrarily chose best fit parameters for GAB 20190119 data
    Mixed_Model.set_parameters({'R2': 5700, 'R1': 1800,
                                'k_a1': 4.98E-14 * 2, 'k_a2': 1.328e-12, 'k_d3': 2.4e-06, 'k_d4': 0.228,
                                'kSOCSon': 5e-08, 'kpu': 0.0011,
                                'ka1': 3.3e-15, 'ka2': 1.22e-12, 'kd4': 0.86,
                                'kd3': 1.74e-05,
                                'kint_a': 0.000124, 'kint_b': 0.00086,
                                'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05})

    # ---------------------------------------------------------------
    # Hypothesis: differential recycling could make beta similar to alpha
    # ---------------------------------------------------------------
    Mixed_Model.set_parameters({'kSOCSon': 1e-07,
                                'kint_a': 0.00124, 'kint_b': 0.0586,
                                'krec_a1': 0.28, 'krec_a2': 0.1,
                                'krec_b1': 0.00001, 'krec_b2': 0.00001})


    scale_factor = 0.242052437849
    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])

    times = [1, 5, 60]
    experimental_doses = [0.051, 0.51, 1.54, 5.13, 15.4, 51.3]
    simulation_doses = list(logspace(log10(0.01), log10(2000)))
    dradf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', simulation_doses,
                                           parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', simulation_doses,
                                           parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')

    for i in range(len(times)):
        dradf.loc['Alpha'].iloc[:, i] = dradf.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf.loc['Beta'].iloc[:, i] = drbdf.loc['Beta'].iloc[:, i].apply(scale_data)

    dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    new_fit = DoseresponsePlot((3, 1))
    new_fit.axes[0].set_title("{} minutes".format(times[0]))
    new_fit.axes[1].set_title("{} minutes".format(times[1]))
    new_fit.axes[2].set_title("{} minutes".format(times[2]))

    alpha_mask = []
    beta_mask = []
    # Add fits
    for idx, t in enumerate([el for el in times]):
        if t not in alpha_mask:
            new_fit.add_trajectory(dra60, t, 'plot', alpha_palette[2], (idx, 0), 'Alpha', label='Alpha')
        if t not in beta_mask:
            new_fit.add_trajectory(drb60, t, 'plot', beta_palette[2], (idx, 0), 'Beta', label='Beta')

    new_fit.show_figure(save_flag=False)




