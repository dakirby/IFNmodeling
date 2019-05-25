from ifnclass.ifnmodel import IfnModel
from numpy import linspace, logspace
import seaborn as sns
import matplotlib
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
    #USP18_Model = IfnModel('Mixed_IFN_explicitUSP18')

    # fit_all_GAB_data parameters plus K1 * 0.2 and K2 * 0.2
    Mixed_Model.set_parameters({'R2': 4920, 'R1': 1200,
                                'k_a1': 2.0e-13 * 5, 'k_a2': 1.328e-12 * 5, 'k_d3': 1.13e-4, 'k_d4': 0.9,
                                'kSOCSon': 5e-08, 'kpu': 0.0022, 'kpa': 2.36e-06,
                                'ka1': 3.3e-15 * 5, 'ka2': 1.85e-12 * 5, 'kd4': 2.0,
                                'kd3': 6.52e-05,
                                'kint_a': 0.0015, 'kint_b': 0.002,
                                'krec_a1': 0.01, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05})

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
    k4sf2 = 5
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

    # ---------------------------------------------------
    # Now make the figure where we explicitly model USP18
    # ---------------------------------------------------
    fraction_USP18_occupied = 0.6
    dose_list = list(logspace(-2, 5, num=46))
    USP18_Model = IfnModel('Mixed_IFN_explicitUSP18')
    USP18_Model.set_parameters(
        {'R2_0': 2300 * 2.5 * (1 - fraction_USP18_occupied), 'R2USP18_0': 2300 * 2.5 * fraction_USP18_occupied,
         'R1': 1800 * 1.8, 'k_d4': 0.06, 'k_d4_USP18': 0.06 * 15, 'kint_b': 0.0003,
         'kpu': 0.0028,
         'krec_b1': 0.001, 'krec_b2': 0.01,
         'kSOCSon': 0.9e-8,
         'kd4': 1.0, 'kd3': 0.001,
         'kint_a': 0.0014, 'krec_a1': 9e-03, 'krec_a2': 0.05})
    USP18_Model.set_parameters({'ka1': USP18_Model.parameters['ka1'] * 5,
                                'ka2': USP18_Model.parameters['ka2'] * 5,
                                'k_a1': USP18_Model.parameters['k_a1'] * 5,
                                'k_a2': USP18_Model.parameters['k_a2'] * 5})

    # Refractory
    dr_curve_a_refrac = [el[0] for el in USP18_Model.doseresponse([60], 'TotalpSTAT', 'Ia', dose_list,
                                                                  parameters={'Ib': 0}, return_type='list')['TotalpSTAT']]
    dr_curve_b_refrac = [el[0] for el in USP18_Model.doseresponse([60], 'TotalpSTAT', 'Ib', dose_list,
                                                                  parameters={'Ia': 0}, return_type='list')['TotalpSTAT']]

    # Baseline
    USP18_Model.set_parameters({'R2_0': 2300 * 2.5, 'R2USP18_0': 0})
    dr_curve_a = [el[0] for el in USP18_Model.doseresponse([60], 'TotalpSTAT', 'Ia', dose_list,
                                                           parameters={'Ib': 0}, return_type='list')['TotalpSTAT']]
    dr_curve_b = [el[0] for el in USP18_Model.doseresponse([60], 'TotalpSTAT', 'Ib', dose_list,
                                                           parameters={'Ia': 0}, return_type='list')['TotalpSTAT']]

    # Plot
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
                 label=r'IFN$\alpha$', color=alpha_palette[4])
    axes[0].plot(dose_list, np.divide(dr_curve_b_refrac, dr_curve_b),
                 label=r'IFN$\beta$', color=beta_palette[4])
    axes[1].plot(dose_list, dr_curve_a, label=r'IFN$\alpha$', color=alpha_palette[0])
    axes[1].plot(dose_list, dr_curve_a_refrac, label=r'IFN$\alpha$ refractory', color=alpha_palette[4])
    axes[1].plot(dose_list, dr_curve_b, label=r'IFN$\beta$', color=beta_palette[0])
    axes[1].plot(dose_list, dr_curve_b_refrac, label=r'IFN$\beta$ refractory', color=beta_palette[4])

    axes[0].legend(loc=2, prop={'size': 5})
    axes[1].legend(loc=2, prop={'size': 5})
    fig.show()
    fig.savefig(os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_4', 'Figure_4_explicitUSP18.pdf'))
