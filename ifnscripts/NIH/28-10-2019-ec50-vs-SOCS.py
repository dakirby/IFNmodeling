from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import DoseresponsePlot
from ifnclass.ifndata import IfnData
import numpy as np
from numpy import logspace
import matplotlib.pyplot as plt
import seaborn as sns

ImmuneCell_Model = IfnModel('Immune_Cell_model')
# -------------------------------------------
# Get naive responses (i.e. one IFN at a time)
# -------------------------------------------
IFNg_doses = list(logspace(-2, 2, num=30))

times = [15, 30, 60]
# IFN gamma
fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0][0].set_title('pSTAT1')
ax[0][1].set_title('pSTAT3')
ax[0][0].set_ylabel('EC50 (pM)')
ax[1][0].set_ylabel('Max response (# molec.)')
ax[1][0].set_xlabel('SOCS (# molec.)')
ax[1][1].set_xlabel('SOCS (# molec.)')

colour_palette = sns.color_palette("dark", 6)

for s in np.linspace(0, 2000, num=15):
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
    print(naive_pSTAT3_response_at_IFNgec50)

    for tidx, t in enumerate(times):
        ax[0][0].scatter(s, naive_IFNgec50_pSTAT1[tidx][1], c=colour_palette[tidx])
        ax[0][1].scatter(s, naive_IFNgec50_pSTAT3[tidx][1], c=colour_palette[tidx])
        ax[1][0].scatter(s, naive_pSTAT1_response_at_IFNgec50[str(t)], c=colour_palette[tidx])
        ax[1][1].scatter(s, naive_pSTAT3_response_at_IFNgec50[str(t)], c=colour_palette[tidx])
plt.show()
