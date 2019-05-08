from ifnclass.ifnmodel import IfnModel
from numpy import linspace, logspace
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
    USP18_Model = IfnModel('Mixed_IFN_explicitUSP18')

    Mixed_Model.set_parameters(
        {'R2': 2300 * 2.5,
         'R1': 1800 * 1.8, 'k_d4': 0.06, 'kint_b': 0.0003,
         'kpu': 0.0028,
         'krec_b1': 0.001, 'krec_b2': 0.01,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 4, 'kSOCSon': 0.9e-8,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3, 'kd4': 1.0, 'kd3': 0.001,
         'kint_a': 0.0014, 'krec_a1': 9e-03, 'krec_a2': 0.05})

    dose_list = list(logspace(-2, 8, num=35))

    # ---------------------------------------------------
    # First make the figure where we increase K4
    # ---------------------------------------------------
    dr_curve_a = [el[0] for el in Mixed_Model.doseresponse([60], 'TotalpSTAT', 'Ia', dose_list,
                                                   parameters={'Ib': 0}, return_type='list')['TotalpSTAT']]
    dr_curve_b = [el[0] for el in Mixed_Model.doseresponse([60], 'TotalpSTAT', 'Ib', dose_list,
                                                   parameters={'Ia': 0}, return_type='list')['TotalpSTAT']]
    # Now compute the 20* refractory response
    k4sf1 = 2
    Mixed_Model.set_parameters({'kd4': 1.0*k4sf1, 'k_d4': 0.06*k4sf1})

    dr_curve_a20 = [el[0] for el in Mixed_Model.doseresponse([60], 'TotalpSTAT', 'Ia', dose_list,
                                                   parameters={'Ib': 0}, return_type='list')['TotalpSTAT']]
    dr_curve_b20 = [el[0] for el in Mixed_Model.doseresponse([60], 'TotalpSTAT', 'Ib', dose_list,
                                                   parameters={'Ia': 0}, return_type='list')['TotalpSTAT']]
    # Now compute the 60* refractory response
    k4sf2 = 5
    Mixed_Model.set_parameters({'kd4': 1.0*k4sf2, 'k_d4': 0.06*k4sf2})

    dr_curve_a60 = [el[0] for el in Mixed_Model.doseresponse([60], 'TotalpSTAT', 'Ia', dose_list,
                                                   parameters={'Ib': 0}, return_type='list')['TotalpSTAT']]
    dr_curve_b60 = [el[0] for el in Mixed_Model.doseresponse([60], 'TotalpSTAT', 'Ib', dose_list,
                                                   parameters={'Ia': 0}, return_type='list')['TotalpSTAT']]

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(16, 8)
    axes.set_xlabel("Dose (pM)", fontsize=14)
    axes.set_title("Relative Refractory Response", fontsize=16)
    axes.set_ylabel("pSTAT1 Relative to Primary Response", fontsize=14)
    axes.set(xscale='log', yscale='linear')
    axes.plot(dose_list, np.divide(dr_curve_a20, dr_curve_a),
                 label=r'IFN$\alpha$ $K_{D4}\times$'+'{}'.format(k4sf1), color=alpha_palette[4], linewidth=2)
    axes.plot(dose_list, np.divide(dr_curve_a60, dr_curve_a),
                 label=r'IFN$\alpha$ $K_{D4}\times$'+'{}'.format(k4sf2), color=alpha_palette[4], linestyle='dashed', linewidth=2)
    axes.plot(dose_list, np.divide(dr_curve_b20, dr_curve_b),
                 label=r'IFN$\beta$ $K_{D4}\times$'+'{}'.format(k4sf1), color=beta_palette[4], linewidth=2)
    axes.plot(dose_list, np.divide(dr_curve_b60, dr_curve_b),
                 label=r'IFN$\beta$ $K_{D4}\times$'+'{}'.format(k4sf2), color=beta_palette[4], linestyle='dashed', linewidth=2)

    axes.legend(loc=2, prop={'size': 8})
    fig.set_size_inches(8, 8)
    fig.savefig(os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_4', 'Figure_4.pdf'))