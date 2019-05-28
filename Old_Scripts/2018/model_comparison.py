import matplotlib.pyplot as plt
import pysbplotlib as pyplt
import IFN_alpha_altSOCS as testA
import IFN_beta_altSOCS as testB
import IFN_simplified_model_alpha as oriA
import IFN_simplified_model_beta as oriB
import numpy as np

def main():
    plt.close('all')
    t=np.linspace(0,3600)
    dA=pyplt.timecourse(testA, t, [['TotalpSTAT',"Total pSTAT"]],
                     ['Time (s)','Molecules/cell'],suppress=True)
    dB=pyplt.timecourse(testB, t, [['TotalpSTAT',"Total pSTAT"]],
                     ['Time (s)','Molecules/cell'],suppress=True)    
    doA=pyplt.timecourse(oriA, t, [['TotalpSTAT',"Total pSTAT"]],
                     ['Time (s)','Molecules/cell'],suppress=True)
    doB=pyplt.timecourse(oriB, t, [['TotalpSTAT',"Total pSTAT"]],
                     ['Time (s)','Molecules/cell'],suppress=True)    
    plt.ion()
    fig, ax = plt.subplots()
    plt.xlabel('time (s)', fontsize=18)
    plt.ylabel('pSTAT Molecules per Cell', fontsize=18)
    fig.suptitle('Old model vs New SOCS Inhibition Model', fontsize=18)
    ax.tick_params(labelsize=14)
    ax.plot(t, dA['TotalpSTAT'], 'r--', label='New SOCS IFNa', linewidth=2.0)
    ax.plot(t, dB['TotalpSTAT'], 'g--', label='New SOCS IFNb', linewidth=2.0)
    ax.plot(t, doA['TotalpSTAT'], 'r', label='Old IFNa', linewidth=2.0)
    ax.plot(t, doB['TotalpSTAT'], 'g', label='Old IFNa', linewidth=2.0)
    plt.legend()
    plt.show()
    
    oADR = pyplt.doseresponse(oriA, ['IFN',np.logspace(-14,-2,num=50)], t,
                                       [['TotalpSTAT',"Total pSTAT"]],suppress=True)[0]
    oBDR = pyplt.doseresponse(oriB, ['IFN',np.logspace(-14,-2,num=50)], t,
                                       [['TotalpSTAT',"Total pSTAT"]],suppress=True)[0]
    ADR = pyplt.doseresponse(testA, ['IFN',np.logspace(-14,-2,num=50)], t,
                                       [['TotalpSTAT',"Total pSTAT"]],suppress=True)[0]
    BDR = pyplt.doseresponse(testB, ['IFN',np.logspace(-14,-2,num=50)], t,
                                       [['TotalpSTAT',"Total pSTAT"]],suppress=True)[0]
    fig2, ax2 = plt.subplots()
    ax2.set(xscale='log',yscale='linear')
    plt.xlabel('Dose IFN', fontsize=18)
    plt.ylabel('pSTAT Molecules per Cell', fontsize=18)
    fig.suptitle('Old model vs New SOCS Inhibition Model', fontsize=18)
    ax2.tick_params(labelsize=14)
    ax2.plot(np.logspace(-14,-2,num=50), ADR, 'r--', label='New SOCS IFNa', linewidth=2.0)
    ax2.plot(np.logspace(-14,-2,num=50), BDR, 'g--', label='New SOCS IFNb', linewidth=2.0)
    ax2.plot(np.logspace(-14,-2,num=50), oADR, 'r', label='Old IFNa', linewidth=2.0)
    ax2.plot(np.logspace(-14,-2,num=50), oBDR, 'g', label='Old IFNa', linewidth=2.0)
    plt.legend()
    plt.show()
    