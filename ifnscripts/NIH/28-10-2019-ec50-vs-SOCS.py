from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import DoseresponsePlot
from ifnclass.ifndata import IfnData
import numpy as np
from numpy import logspace
import matplotlib.pyplot as plt
import seaborn as sns

ImmuneCell_Model = IfnModel('Immune_Cell_model')

IFNg_doses = list(logspace(-1.5, 2, num=15))
times = [15, 30, 60]
fig = DoseresponsePlot((2, 2))
fig.axes[0][0].set_title('pSTAT1')
fig.axes[0][1].set_title('pSTAT3')

print("Dose-response 1")
IFNg_naive_res = ImmuneCell_Model.doseresponse(times, ['pSTAT1', 'pSTAT3'], 'IFN_gamma_IC', IFNg_doses,
                                                       parameters={'IFNAR1_IC':0, 'IFNAR2_IC':0, 'IFN_alpha2_IC':0,
                                                                   'IFN_beta_IC':0, 'SOCS2_IC':0},
                                                       return_type='dataframe', dataframe_labels='IFNgamma')
naive_IFNg_pSTAT1_df = IfnData(name='custom', df=IFNg_naive_res.xs(['pSTAT1']))
naive_IFNg_pSTAT3_df = IfnData(name='custom', df=IFNg_naive_res.xs(['pSTAT3']))

print("Dose-response 2")
IFNg_SOCS_res = ImmuneCell_Model.doseresponse(times, ['pSTAT1', 'pSTAT3'], 'IFN_gamma_IC', IFNg_doses,
                                                       parameters={'IFNAR1_IC':0, 'IFNAR2_IC':0, 'IFN_alpha2_IC':0,
                                                                   'IFN_beta_IC':0, 'SOCS2_IC':200},
                                                       return_type='dataframe', dataframe_labels='IFNgamma')
SOCS_IFNg_pSTAT1_df = IfnData(name='custom', df=IFNg_SOCS_res.xs(['pSTAT1']))
SOCS_IFNg_pSTAT3_df = IfnData(name='custom', df=IFNg_SOCS_res.xs(['pSTAT3']))


fig.add_trajectory(naive_IFNg_pSTAT1_df, 15, 'plot', 'r', (0, 0), 'IFNgamma', label='15 min')
fig.add_trajectory(naive_IFNg_pSTAT1_df, 30, 'plot', 'r', (0, 0), 'IFNgamma', label='30 min')
fig.add_trajectory(naive_IFNg_pSTAT1_df, 60, 'plot', 'r', (0, 0), 'IFNgamma', label='60 min')

fig.add_trajectory(SOCS_IFNg_pSTAT1_df, 15, 'plot', 'r', (1, 0), 'IFNgamma', label='15 min')
fig.add_trajectory(SOCS_IFNg_pSTAT1_df, 30, 'plot', 'r', (1, 0), 'IFNgamma', label='30 min')
fig.add_trajectory(SOCS_IFNg_pSTAT1_df, 60, 'plot', 'r', (1, 0), 'IFNgamma', label='60 min')

fig.add_trajectory(naive_IFNg_pSTAT3_df, 15, 'plot', 'r', (0, 1), 'IFNgamma', label='15 min')
fig.add_trajectory(naive_IFNg_pSTAT3_df, 30, 'plot', 'r', (0, 1), 'IFNgamma', label='30 min')
fig.add_trajectory(naive_IFNg_pSTAT3_df, 60, 'plot', 'r', (0, 1), 'IFNgamma', label='60 min')

fig.add_trajectory(SOCS_IFNg_pSTAT3_df, 15, 'plot', 'r', (1, 1), 'IFNgamma', label='15 min')
fig.add_trajectory(SOCS_IFNg_pSTAT3_df, 30, 'plot', 'r', (1, 1), 'IFNgamma', label='30 min')
fig.add_trajectory(SOCS_IFNg_pSTAT3_df, 60, 'plot', 'r', (1, 1), 'IFNgamma', label='60 min')


fig.show_figure()

def plot_ec50():
    # IFN gamma
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0][0].set_title('pSTAT1')
    ax[0][1].set_title('pSTAT3')
    ax[0][0].set_ylabel('EC50 (pM)')
    ax[1][0].set_ylabel('Max response (# molec.)')
    ax[1][0].set_xlabel('SOCS (# molec.)')
    ax[1][1].set_xlabel('SOCS (# molec.)')

    colour_palette = sns.color_palette("dark", 6)

    for sidx, s in enumerate(np.linspace(0, 2000, num=15)):
        IFNg_naive_res = ImmuneCell_Model.doseresponse(times, ['pSTAT1', 'pSTAT3'], 'IFN_gamma_IC', IFNg_doses,
                                                       parameters={'IFNAR1_IC':0, 'IFNAR2_IC':0, 'IFN_alpha2_IC':0,
                                                                   'IFN_beta_IC':0, 'SOCS2_IC':s},
                                                       return_type='dataframe', dataframe_labels='IFNgamma')

        naive_IFNg_pSTAT1_df = IfnData(name='custom', df=IFNg_naive_res.xs(['pSTAT1']))
        naive_IFNg_pSTAT3_df = IfnData(name='custom', df=IFNg_naive_res.xs(['pSTAT3']))
        naive_IFNgec50_pSTAT1 = naive_IFNg_pSTAT1_df.get_ec50s()['IFNgamma']
        naive_IFNgec50_pSTAT3 = naive_IFNg_pSTAT3_df.get_ec50s()['IFNgamma']

        naive_pSTAT1_response_at_IFNgec50 = {key: val for (key, val) in naive_IFNg_pSTAT1_df.get_max_responses()['IFNgamma']}
        naive_pSTAT3_response_at_IFNgec50 = {key: val for (key, val) in naive_IFNg_pSTAT3_df.get_max_responses()['IFNgamma']}

        for tidx, t in enumerate(times):
            if sidx==0:
                ax[0][0].scatter(s, naive_IFNgec50_pSTAT1[tidx][1], c=colour_palette[tidx], label='pSTAT1 {} min'.format(t))
                ax[0][1].scatter(s, naive_IFNgec50_pSTAT3[tidx][1], c=colour_palette[tidx], label='pSTAT3 {} min'.format(t))
                ax[1][0].scatter(s, naive_pSTAT1_response_at_IFNgec50[str(t)], c=colour_palette[tidx], label='pSTAT1 {} min'.format(t))
                ax[1][1].scatter(s, naive_pSTAT3_response_at_IFNgec50[str(t)], c=colour_palette[tidx], label='pSTAT3 {} min'.format(t))
                plt.legend()
            else:
                ax[0][0].scatter(s, naive_IFNgec50_pSTAT1[tidx][1], c=colour_palette[tidx])
                ax[0][1].scatter(s, naive_IFNgec50_pSTAT3[tidx][1], c=colour_palette[tidx])
                ax[1][0].scatter(s, naive_pSTAT1_response_at_IFNgec50[str(t)], c=colour_palette[tidx])
                ax[1][1].scatter(s, naive_pSTAT3_response_at_IFNgec50[str(t)], c=colour_palette[tidx])

    plt.show()
    plt.savefig('ec50_SOCS.pdf')
