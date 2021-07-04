from ifnclass.ifndata import IfnData, DataAlignment
from numpy import linspace, logspace
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from ifnclass.ifnplot import DoseresponsePlot
import load_model as lm
from tqdm import tqdm
import copy


if __name__=='__main__':
    # --------------------
    # Set up Model
    # --------------------
    Mixed_Model, DR_method = lm.load_model()
    scale_factor, DR_KWARGS, PLOT_KWARGS = lm.SCALE_FACTOR, lm.DR_KWARGS, lm.PLOT_KWARGS
    Mixed_Model.num_dist_samples = 10

    # Compute response for various receptor abundances
    RT_test = np.logspace(1, 3.7, 10)
    Delta_test = np.linspace(-2000, 2000, 15)
    RT_EC50, RT_pSTATmax, Delta_EC50, Delta_pSTATmax = [], [], [], []

    for rtest in RT_test:
        idx = np.where(Mixed_Model.parameter_names=='R1_mu*')[0][0]
        Mixed_Model.parameters[0][idx] = rtest
        idx = np.where(Mixed_Model.parameter_names=='R2_mu*')[0][0]
        Mixed_Model.parameters[0][idx] = rtest

        dr1 = DR_method([60.], 'TotalpSTAT', 'Ia', list(logspace(-3, 5)),
                        parameters={'Ib': 0}, sf=scale_factor, **DR_KWARGS)
        ec50 = [el[1] for el in dr1.get_ec50s()['Alpha']][0]
        pSTATmax = [el[1] for el in dr1.get_max_responses()['Alpha']][0]
        RT_EC50.append(ec50)
        RT_pSTATmax.append(pSTATmax)

    for Delta in Delta_test:
        idx = np.where(Mixed_Model.parameter_names=='R1_mu*')[0][0]
        Mixed_Model.parameters[0][idx] = (2000 + Delta)/2
        idx = np.where(Mixed_Model.parameter_names=='R2_mu*')[0][0]
        Mixed_Model.parameters[0][idx] = (2000 - Delta)/2

        dr1 = DR_method([60.], 'TotalpSTAT', 'Ia', list(logspace(-3, 5)),
                        parameters={'Ib': 0}, sf=scale_factor, **DR_KWARGS)
        ec50 = [el[1] for el in dr1.get_ec50s()['Alpha']][0]
        pSTATmax = [el[1] for el in dr1.get_max_responses()['Alpha']][0]
        Delta_EC50.append(ec50)
        Delta_pSTATmax.append(pSTATmax)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    axes[0][0].plot(RT_test, RT_pSTATmax)
    axes[0][1].plot(RT_test, RT_EC50)
    axes[1][0].plot(Delta_test, Delta_pSTATmax)
    axes[1][1].plot(Delta_test, Delta_EC50)

    axes[0][1].set_xscale('log')
    axes[0][0].set_xlabel(r'$R_T$')
    axes[0][1].set_xlabel(r'$R_T$')
    axes[0][0].set_ylabel('pSTATmax')
    axes[0][1].set_ylabel(r'$EC_{50}$')
    axes[1][1].set_xscale('log')
    axes[1][0].set_xlabel(r'$\Delta$')
    axes[1][1].set_xlabel(r'$\Delta$')
    axes[1][0].set_ylabel('pSTATmax')
    axes[1][1].set_ylabel(r'$EC_{50}$')
