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
IFNa_doses = list(logspace(0, 4, num=30)) # (pM)
IFNb_doses = list(logspace(-1, 3, num=30)) # (pM)

times = [15, 30, 60]
# IFN gamma
IFNg_naive_res = ImmuneCell_Model.doseresponse(times, ['pSTAT1', 'pSTAT3'], 'IFN_gamma_IC', IFNg_doses,
                                               parameters={'IFNAR1_IC':0, 'IFNAR2_IC':0, 'IFN_alpha2_IC':0, 'IFN_beta_IC':0},
                                               return_type='dataframe', dataframe_labels='IFNgamma')
naive_IFNg_pSTAT1_df = IfnData(name='custom', df=IFNg_naive_res.xs(['pSTAT1']))
naive_IFNg_pSTAT3_df = IfnData(name='custom', df=IFNg_naive_res.xs(['pSTAT3']))
naive_IFNgec50_pSTAT1 = naive_IFNg_pSTAT1_df.get_ec50s()
naive_IFNgec50_pSTAT3 = naive_IFNg_pSTAT3_df.get_ec50s()

naive_pSTAT1_response_at_IFNgec50 = {key: val/2 for (key, val) in naive_IFNg_pSTAT1_df.get_max_responses()['IFNgamma']}
naive_pSTAT3_response_at_IFNgec50 = {key: val/2 for (key, val) in naive_IFNg_pSTAT3_df.get_max_responses()['IFNgamma']}

# IFN alpha
IFNa_naive_res = ImmuneCell_Model.doseresponse(times, ['pSTAT1', 'pSTAT3'], 'IFN_gamma_IC', IFNa_doses,
                                               parameters={'IFNAR1_IC':0, 'IFNAR2_IC':0, 'IFN_alpha2_IC':0, 'IFN_beta_IC':0},
                                               return_type='dataframe', dataframe_labels='IFNalpha')
naive_IFNa_pSTAT1_df = IfnData(name='custom', df=IFNa_naive_res.xs(['pSTAT1']))
naive_IFNa_pSTAT3_df = IfnData(name='custom', df=IFNa_naive_res.xs(['pSTAT3']))
naive_IFNaec50_pSTAT1 = naive_IFNa_pSTAT1_df.get_ec50s()
naive_IFNaec50_pSTAT3 = naive_IFNa_pSTAT3_df.get_ec50s()
naive_pSTAT1_response_at_IFNaec50 = {key: val/2 for (key, val) in naive_IFNa_pSTAT1_df.get_max_responses()['IFNalpha']}
naive_pSTAT3_response_at_IFNaec50 = {key: val/2 for (key, val) in naive_IFNa_pSTAT3_df.get_max_responses()['IFNalpha']}
# IFN beta
IFNb_naive_res = ImmuneCell_Model.doseresponse(times, ['pSTAT1', 'pSTAT3'], 'IFN_gamma_IC', IFNb_doses,
                                               parameters={'IFNAR1_IC':0, 'IFNAR2_IC':0, 'IFN_alpha2_IC':0, 'IFN_beta_IC':0},
                                               return_type='dataframe', dataframe_labels='IFNbeta')
naive_IFNb_pSTAT1_df = IfnData(name='custom', df=IFNb_naive_res.xs(['pSTAT1']))
naive_IFNb_pSTAT3_df = IfnData(name='custom', df=IFNb_naive_res.xs(['pSTAT3']))
naive_IFNbec50_pSTAT1 = naive_IFNb_pSTAT1_df.get_ec50s()
naive_IFNbec50_pSTAT3 = naive_IFNb_pSTAT3_df.get_ec50s()
naive_pSTAT1_response_at_IFNbec50 = {key: val/2 for (key, val) in naive_IFNb_pSTAT1_df.get_max_responses()['IFNbeta']}
naive_pSTAT3_response_at_IFNbec50 = {key: val/2 for (key, val) in naive_IFNb_pSTAT3_df.get_max_responses()['IFNbeta']}

# -------------------------------------------
# Titrate in Type 1 IFN
# -------------------------------------------
alpha_PSTAT1_responses = {}
beta_PSTAT1_responses = {}
alpha_PSTAT3_responses = {}
beta_PSTAT3_responses = {}

for d in IFNa_doses:
    res = ImmuneCell_Model.doseresponse(times, ['pSTAT1', 'pSTAT3'], 'IFN_gamma_IC', IFNg_doses,
                                        parameters={'IFN_alpha2_IC': d*1E-12*.022E23*ImmuneCell_Model.parameters['volEC'], 'IFN_beta_IC': 0},
                                        return_type='dataframe', dataframe_labels='IFNgamma')
    alpha_PSTAT1_responses[d] = IfnData(name='custom', df=res.xs(['pSTAT1']))
    alpha_PSTAT3_responses[d] = IfnData(name='custom', df=res.xs(['pSTAT3']))
for d in IFNb_doses:
    res = ImmuneCell_Model.doseresponse(times, ['pSTAT1', 'pSTAT3'], 'IFN_gamma_IC', IFNg_doses,
                                        parameters={'IFN_beta_IC': d*1E-12*.022E23*ImmuneCell_Model.parameters['volEC'], 'IFN_alpha2_IC': 0},
                                        return_type='dataframe', dataframe_labels='IFNgamma')
    beta_PSTAT1_responses[d] = IfnData(name='custom', df=res.xs(['pSTAT1']))
    beta_PSTAT3_responses[d] = IfnData(name='custom', df=res.xs(['pSTAT3']))

# -----------------------------------------------
# Compare combined response to sum of naive ones
# -----------------------------------------------
naive_IFNa_pSTAT1_df.data_set
print(naive_IFNa_pSTAT1_df.data_set.values)
#sum_alpha_gamma_naive_pSTAT1 = np.add(naive_IFNa_pSTAT1_df.data_set.values[:,1:-1], naive_IFNg_pSTAT1_df.data_set.values[:,1:-1])
#sum_beta_gamma_naive_pSTAT1 = np.add(naive_IFNb_pSTAT1_df.data_set.values[:,1:-1], naive_IFNg_pSTAT1_df.data_set.values[:,1:-1])


