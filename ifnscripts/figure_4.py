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
    #USP18_Model = IfnModel('Mixed_IFN_explicitUSP18')

    # Minimal fit parameters
    Mixed_Model.set_parameters({'R2': 7280 * 0.5, 'R1': 4920 * 0.5,
                                'kSOCSon': 1e-7, 'kpu': 0.001, 'kpa': 1e-06,
                                'kSOCS': 0.0046,
                                'kint_a': 0.0001, 'kint_b': 0.01})

    dose_list = list(logspace(-2, 8, num=35))

    # ---------------------------------------------------
    # First make the figure where we increase K4
    # ---------------------------------------------------
    dr_curve_a = [el[0] for el in Mixed_Model.doseresponse([60], 'TotalpSTAT', 'Ia', dose_list,
                                                   parameters={'Ib': 0}, return_type='list')['TotalpSTAT']]
    dr_curve_b = [el[0] for el in Mixed_Model.doseresponse([60], 'TotalpSTAT', 'Ib', dose_list,
                                                   parameters={'Ia': 0}, return_type='list')['TotalpSTAT']]
    # Now compute the 20* refractory response
    k4sf1 = 20
    kd4_reference = Mixed_Model.parameters['kd4']
    kd3_reference = Mixed_Model.parameters['kd3']
    k_d4_reference = Mixed_Model.parameters['k_d4']
    k_d3_reference = Mixed_Model.parameters['k_d3']
    Mixed_Model.set_parameters({'kd4': kd4_reference*k4sf1, 'k_d4': k_d4_reference*k4sf1,
                                'kd3': kd3_reference * k4sf1, 'k_d3': k_d3_reference * k4sf1})

    dr_curve_a20 = [el[0] for el in Mixed_Model.doseresponse([60], 'TotalpSTAT', 'Ia', dose_list,
                                                   parameters={'Ib': 0}, return_type='list')['TotalpSTAT']]
    dr_curve_b20 = [el[0] for el in Mixed_Model.doseresponse([60], 'TotalpSTAT', 'Ib', dose_list,
                                                   parameters={'Ia': 0}, return_type='list')['TotalpSTAT']]
    # Now compute the 60* refractory response
    k4sf2 = 60
    Mixed_Model.set_parameters({'kd4': kd4_reference*k4sf2, 'k_d4': k_d4_reference*k4sf2,
                                'kd3': kd3_reference * k4sf2, 'k_d3': k_d3_reference * k4sf2})

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
    fig.savefig(os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_4', 'Figure_4_new.pdf'))