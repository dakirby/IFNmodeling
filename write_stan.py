# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 08:49:04 2018

@author: Duncan

Export to STAN
"""
from pysb.export import export
modelfiles = ['ifnmodels.IFN_alpha_altSOCS_Internalization_ppCompatible','ifnmodels.IFN_beta_altSOCS_Internalization_ppCompatible']

alpha_model = __import__(modelfiles[0],fromlist=['ifnmodels'])
py_output = export(alpha_model.model, 'stan')
with open('STAN_alpha.stan','w') as f:
    f.write(py_output)
beta_model = __import__(modelfiles[1],fromlist=['ifnmodels'])
py_output = export(beta_model.model, 'stan')
with open('STAN_beta.stan','w') as f:
    f.write(py_output)

