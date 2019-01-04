from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import DoseresponsePlot
import seaborn as sns


if __name__ == '__main__':
    Sagar_data = IfnData("MacParland_Extended")
    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')

    alpha_palette = sns.color_palette("Reds", 2)
    beta_palette = sns.color_palette("Greens", 2)

    '''
    Mixed_Model.set_parameters({'kpu': 0.00025, 'kpa': 5e-7,
                                'R2': 1742, 'R1': 1785,
                                'k_d4': 0.06, 'kd4': 0.803, 'kd3': 0.008,
                                'k_a2': 8.3e-13 * 0.1,
                                'ka2': 4.98173364330787e-13 * 0.1, 'ka1': 3.321155762205247e-14,
                                'ka4': 0.2,
                                'kSOCS': 0.005, 'kSOCSon': 1e-3, 'SOCSdeg': 0,
                                'kint_b': 0.0009, 'kint_a': 0.001})
    '''
    Mixed_Model.set_parameters({'ka4': 0.0003623*0.01, 'ka2': 4.981e-13*0.01, 'R2': 2000, 'R1': 500})

    scale_factor = 1

    scale_data = lambda q: (scale_factor*q[0], scale_factor*q[1])
    times = [1, 60]
    dradf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', [0.051, 0.51, 1.54, 5.13, 15.4, 51.3],
                                           parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', [0.051, 0.51, 1.54, 5.13, 15.4, 51.3],
                                           parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')

    for i in range(len(times)):
        dradf.loc['Alpha'].iloc[:, i] = dradf.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf.loc['Beta'].iloc[:, i] = drbdf.loc['Beta'].iloc[:, i].apply(scale_data)

    dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    new_fit = DoseresponsePlot((2, 1))
    alpha_mask = []
    beta_mask = []
    # Add fits
    for idx, t in enumerate([el for el in times]):
        if t not in alpha_mask:
            new_fit.add_trajectory(dra60, t, 'scatter', alpha_palette[idx], (idx, 0), 'Alpha', label='Alpha '+str(t)+' min')
            new_fit.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (idx, 0), 'Alpha',
                                   label='Alpha ' + str(t) + ' min')
        if t not in beta_mask:
            new_fit.add_trajectory(drb60, t, 'scatter', beta_palette[idx], (idx, 0), 'Beta', label='Beta '+str(t)+' min')
            new_fit.add_trajectory(drb60, t, 'plot', beta_palette[idx], (idx, 0), 'Beta',
                                   label='Beta ' + str(t) + ' min')

    new_fit.show_figure(save_flag=False)
    print(Mixed_Model.parameters)




