from ifnclass.ifnmodel import IfnModel
from numpy import linspace, logspace
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)


if __name__ == '__main__':
    # ---------------------------------------------------
    # Explicitly model USP18
    # ---------------------------------------------------
    dose_list = list(logspace(-2, 8, num=35))
    # Using parameters from figure_2
    initial_parameters = {'k_a1': 4.98E-14 * 2, 'k_a2': 1.328e-12, 'k_d3': 2.4e-06, 'k_d4': 0.228,
                          'kSOCSon': 8e-07, 'kpu': 0.0011,
                          'ka1': 3.3e-15, 'ka2': 1.22e-12, 'kd4': 0.86,
                          'kd3': 1.74e-05,
                          'kint_a': 0.000124, 'kint_b': 0.00086,
                          'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05}
    USP18_Model = IfnModel('Mixed_IFN_explicitUSP18')
    USP18_Model.set_parameters(initial_parameters)
    # Increase effect of USP18
    USP18_Model.set_parameters({'kd4_USP18': USP18_Model.parameters['kd4_USP18'] * 4,
                                'kd3_USP18': USP18_Model.parameters['kd3_USP18'] * 4,
                                'k_d4_USP18': USP18_Model.parameters['k_d4_USP18'] * 4,
                                'k_d3_USP18': USP18_Model.parameters['k_d3_USP18'] * 4,
                                })
    # Baseline
    USP18_Model.set_parameters({'R2_0': 2300 * 2.5, 'R2USP18_0': 0})
    dr_curve_a = [el[0] for el in USP18_Model.doseresponse([60], 'TotalpSTAT', 'Ia', dose_list,
                                                   parameters={'Ib': 0}, return_type='list')['TotalpSTAT']]
    dr_curve_b = [el[0] for el in USP18_Model.doseresponse([60], 'TotalpSTAT', 'Ib', dose_list,
                                                   parameters={'Ia': 0}, return_type='list')['TotalpSTAT']]
    # Refractory
    USP18_Model.set_parameters({'R2_0': 2300 * 2.5 * 0.4, 'R2USP18_0': 2300 * 2.5 * 0.6})

    dr_curve_a_refrac = [el[0] for el in USP18_Model.doseresponse([60], 'TotalpSTAT', 'Ia', dose_list,
                                                   parameters={'Ib': 0}, return_type='list')['TotalpSTAT']]
    dr_curve_b_refrac = [el[0] for el in USP18_Model.doseresponse([60], 'TotalpSTAT', 'Ib', dose_list,
                                                   parameters={'Ia': 0}, return_type='list')['TotalpSTAT']]
    # Plot
    alpha_palette = sns.color_palette("deep", 6)
    beta_palette = sns.color_palette("deep", 6)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(16, 8)
    axes[0].set_xlabel("Dose (pM)")
    axes[1].set_xlabel("Dose (pM)")
    axes[0].set_title("Relative Refractory Response")
    axes[1].set_title(r"Absolute Responses")
    axes[0].set_ylabel("pSTAT1")
    axes[0].set(xscale='log', yscale='linear')
    axes[1].set(xscale='log', yscale='linear')
    axes[0].plot(dose_list, np.divide(dr_curve_a_refrac, dr_curve_a),
                 label=r'IFN$\alpha$', color=alpha_palette[1], linewidth=2)
    axes[0].plot(dose_list, np.divide(dr_curve_b_refrac, dr_curve_b),
                 label=r'IFN$\beta$', color=beta_palette[2], linewidth=2)
    axes[1].plot(dose_list, dr_curve_a, label=r'IFN$\alpha$', color=alpha_palette[1], linewidth=2)
    axes[1].plot(dose_list, dr_curve_a_refrac, '--', label=r'IFN$\alpha$ refractory', color=alpha_palette[1], linewidth=2)
    axes[1].plot(dose_list, dr_curve_b, label=r'IFN$\beta$', color=beta_palette[2], linewidth=2)
    axes[1].plot(dose_list, dr_curve_b_refrac, '--', label=r'IFN$\beta$ refractory', color=beta_palette[2], linewidth=2)

    axes[0].legend(loc=2, prop={'size': 5})
    axes[1].legend(loc=2, prop={'size': 5})
    fig.show()
    print("Writing Explicit-USP18-Refractory figure to results/Figures/Figure_S4/refractoriness_explicitUSP18.pdf")
    fig.savefig('results/Figures/Figure_S4/refractoriness_explicitUSP18.pdf')