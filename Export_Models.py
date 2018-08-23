# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from pysb.export import export
import numpy as np
import matplotlib.pyplot as plt
modelfiles = ['IFN_alpha_altSOCS_ppCompatible','IFN_beta_altSOCS_ppCompatible']

alpha_model = __import__(modelfiles[0])
py_output = export(alpha_model.model, 'sbml')
with open('SBML_alpha.sbml','w') as f:
    f.write(py_output)
beta_model = __import__(modelfiles[1])
py_output = export(beta_model.model, 'sbml')
with open('SBML_beta.sbml','w') as f:
    f.write(py_output)

import ODE_system_alpha as m
mA = m.Model()
alpha_parameters=[]
for p in mA.parameters:
    if p[0]=='kd4':
        alpha_parameters.append(0.485)
    elif p[0]=='kSOCSon':
        alpha_parameters.append(7.16-7)        
    elif p[0]=='kpa':
        alpha_parameters.append(6.69E-7)   
    elif p[0]=='I':
        alpha_parameters.append(6022000000)   
    else:
        alpha_parameters.append(p.value)
(_, sim) = mA.simulate(np.linspace(0,3600), param_values=alpha_parameters)
print(sim['TotalpSTAT'])
print(sim['T'])

plt.close('all')
plt.figure
plt.plot(np.linspace(0,3600),sim['TotalpSTAT'])
