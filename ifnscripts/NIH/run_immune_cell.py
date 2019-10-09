from ifnclass.ifnmodel import IfnModel
import numpy as np
from numpy import logspace
import matplotlib.pyplot as plt

ImmuneCell_Model = IfnModel('Immune_Cell_model')
#IL10test = ImmuneCell_Model.doseresponse([60], 'Active_IL10_Receptor', 'IL10_IC', list(logspace(-6, 1, num=10)))
#IL20test1 = ImmuneCell_Model.doseresponse([60], 'Active_IL20_Type1', 'IL20_IC', list(logspace(-6, 1, num=10)))
#IL20test2 = ImmuneCell_Model.doseresponse([60], 'Active_IL20_Type2', 'IL20_IC', list(logspace(-6, 1, num=10)))
#IL12test = ImmuneCell_Model.doseresponse([60], 'Active_IL12_Receptor', 'IL12_IC', list(logspace(-6, 1, num=10)))
#IL23test = ImmuneCell_Model.doseresponse([60], 'Active_IL23_Receptor', 'IL12_IC', list(logspace(-6, 1, num=10)))
#IFNatest = ImmuneCell_Model.doseresponse([60], 'Active_IFNa_Receptor', 'IFN_alpha2_IC', list(logspace(-2, 3, num=10)))
#IFNbtest = ImmuneCell_Model.doseresponse([60], 'Active_IFNb_Receptor', 'IFN_beta_IC', list(logspace(-2, 3, num=10)))
#IFNgtest = ImmuneCell_Model.doseresponse([60], 'Active_IFNg_Receptor', 'IFN_gamma_IC', list(logspace(-6, 1, num=10)))
#IL21test = ImmuneCell_Model.doseresponse([60], 'Active_IL21_Receptor', 'IL21_IC', list(logspace(-6, 1, num=10)))
#IL15test = ImmuneCell_Model.doseresponse([60], 'Active_IL15_Receptor', 'IL15_IC', list(logspace(-6, 1, num=10)))
#IL9test = ImmuneCell_Model.doseresponse([60], 'Active_IL9_Receptor', 'IL9_IC', list(logspace(-6, 1, num=10)))
#IL7test = ImmuneCell_Model.doseresponse([60], 'Active_IL7_Receptor', 'IL7_IC', list(logspace(-6, 1, num=10)))
#IL4test1 = ImmuneCell_Model.doseresponse([60], 'Active_IL4_Receptor', 'IL4_IC', list(logspace(-6, 1, num=10)))
#IL4test2 = ImmuneCell_Model.doseresponse([60], 'Active_IL4_IL13RA1_Receptor', 'IL4_IC', list(logspace(-6, 1, num=10)))
#Il13test1 = ImmuneCell_Model.doseresponse([60], 'Active_Tyk2_IL13_Receptor', 'IL13_IC', list(logspace(-6, 1, num=10)))
#Il13test2 = ImmuneCell_Model.doseresponse([60], 'Active_Jak2_IL13_Receptor', 'IL13_IC', list(logspace(-6, 1, num=10)))
#Il13test3 = ImmuneCell_Model.doseresponse([60], 'Inactive_IL13_Receptor', 'IL13_IC', list(logspace(-6, 1, num=10)))
#IL2test = ImmuneCell_Model.doseresponse([60], 'Active_IL2_Receptor', 'IL2_IC', list(logspace(-6, 1, num=10)))
#BetaCtest = ImmuneCell_Model.doseresponse([60], 'Beta_c_dimer', 'Beta_c_IC', list(logspace(-6, 1, num=10)))
#GMCSFtest1 = ImmuneCell_Model.doseresponse([60], 'GM_CSF_BetaC_dimer_GM_CSF', 'GMCSF_IC', list(logspace(-6, 1, num=10)))
#GMCSFtest2 = ImmuneCell_Model.doseresponse([60], 'GM_CSF_Dodecamer', 'GMCSF_IC', list(logspace(-6, 1, num=10)))
#IL3test = ImmuneCell_Model.doseresponse([60], 'IL3_Dodecamer', 'IL3_IC', list(logspace(-6, 1, num=10)))
#IL5test = ImmuneCell_Model.doseresponse([60], 'IL5_Dodecamer', 'IL5_IC', list(logspace(-6, 1, num=10)))
#IL6test = ImmuneCell_Model.doseresponse([60], 'IL6_Receptor', 'IL6_IC', list(logspace(-6, 1, num=10)))
#GCSFtest = ImmuneCell_Model.doseresponse([60], 'GCSF_Receptor', 'GCSF_IC', list(logspace(-6, 1, num=10)))
#IL11test = ImmuneCell_Model.doseresponse([60], 'IL11_Receptor', 'IL11_IC', list(logspace(-6, 1, num=10)))
#IL27test = ImmuneCell_Model.doseresponse([60], 'IL27_Receptor', 'IL27_IC', list(logspace(-6, 1, num=10)))

doses = list(logspace(-9, -4, num=8))
times = [60]
IFNg_pSTAT1 = ImmuneCell_Model.doseresponse(times, 'pSTAT1', 'IFN_gamma_IC', doses)['pSTAT1']

fig = plt.figure()
ax = plt.gca()
ax.set_xscale('log')
plt.plot(doses, IFNg_pSTAT1)
ax.set_xlabel('Dose (pM)')
ax.set_ylabel('Response (#)')
ax.set_title('Dose Response')
plt.show()
