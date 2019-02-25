from ifnclass.ifnmodel import IfnModel
from numpy import linspace, logspace
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

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
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(16, 8)
    axes[0].set_xlabel("Dose (pM)")
    axes[1].set_xlabel("Dose (pM)")
    axes[0].set_title("Relative Refractory Response")
    axes[1].set_title(r"Absolute Responses")
    axes[0].set_ylabel("pSTAT1")
    axes[0].set(xscale='log', yscale='linear')
    axes[1].set(xscale='log', yscale='linear')
    axes[0].plot(dose_list, np.divide(dr_curve_a20, dr_curve_a),
                 label=r'IFN$\alpha$ $K_{D4}\times$'+'{}'.format(k4sf1), color=alpha_palette[4])
    axes[0].plot(dose_list, np.divide(dr_curve_a60, dr_curve_a),
                 label=r'IFN$\alpha$ $K_{D4}\times$'+'{}'.format(k4sf2), color=alpha_palette[4], linestyle='dashed')
    axes[0].plot(dose_list, np.divide(dr_curve_b20, dr_curve_b),
                 label=r'IFN$\beta$ $K_{D4}\times$'+'{}'.format(k4sf1), color=beta_palette[4])
    axes[0].plot(dose_list, np.divide(dr_curve_b60, dr_curve_b),
                 label=r'IFN$\beta$ $K_{D4}\times$'+'{}'.format(k4sf2), color=beta_palette[4], linestyle='dashed')
    axes[1].plot(dose_list, dr_curve_a, label=r'IFN$\alpha$', color=alpha_palette[0])
    axes[1].plot(dose_list, dr_curve_a20, label=r'IFN$\alpha$ $K_{D4}\times$'+'{}'.format(k4sf1), color=alpha_palette[2])
    axes[1].plot(dose_list, dr_curve_a60, label=r'IFN$\alpha$ $K_{D4}\times$'+'{}'.format(k4sf2), color=alpha_palette[4])
    axes[1].plot(dose_list, dr_curve_b, label=r'IFN$\beta$', color=beta_palette[0])
    axes[1].plot(dose_list, dr_curve_b20, label=r'IFN$\beta$ $K_{D4}\times$'+'{}'.format(k4sf1), color=beta_palette[2])
    axes[1].plot(dose_list, dr_curve_b60, label=r'IFN$\beta$ $K_{D4}\times$'+'{}'.format(k4sf2), color=beta_palette[4])

    axes[0].legend(loc=2, prop={'size': 5})
    axes[1].legend(loc=2, prop={'size': 5})
    fig.show()
    print("Writing Increased-K4-Refractory figure to results/refractoriness.pdf")
    fig.savefig('results/refractoriness.pdf')

    # ---------------------------------------------------
    # Now make the figure where we explicitly model USP18
    # ---------------------------------------------------
    Mixed_Model.reset_parameters()
    Mixed_Model.set_parameters(
        {'R2': 2300 * 2.5,
         'R1': 1800 * 1.8, 'k_d4': 0.06, 'kint_b': 0.0003,
         'kpu': 0.0028,
         'krec_b1': 0.001, 'krec_b2': 0.01,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 4, 'kSOCSon': 0.9e-8,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3, 'kd4': 1.0, 'kd3': 0.001,
         'kint_a': 0.0014, 'krec_a1': 9e-03, 'krec_a2': 0.05})
    USP18_Model.set_parameters(
         {'R2_0': 2300 * 2.5 * 0.4, 'R2USP18_0': 2300 * 2.5 * 0.6,
         'R1': 1800 * 1.8, 'k_d4': 0.06, 'k_d4_USP18': 0.06*15, 'kint_b': 0.0003,
         'kpu': 0.0028,
         'krec_b1': 0.001, 'krec_b2': 0.01,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 4, 'kSOCSon': 0.9e-8,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3, 'kd4': 1.0, 'kd3': 0.001,
         'kint_a': 0.0014, 'krec_a1': 9e-03, 'krec_a2': 0.05})

    # Baseline
    dr_curve_a = [el[0] for el in Mixed_Model.doseresponse([60], 'TotalpSTAT', 'Ia', dose_list,
                                                   parameters={'Ib': 0}, return_type='list')['TotalpSTAT']]
    dr_curve_b = [el[0] for el in Mixed_Model.doseresponse([60], 'TotalpSTAT', 'Ib', dose_list,
                                                   parameters={'Ia': 0}, return_type='list')['TotalpSTAT']]
    # Refractory
    dr_curve_a_refrac = [el[0] for el in USP18_Model.doseresponse([60], 'TotalpSTAT', 'Ia', dose_list,
                                                   parameters={'Ib': 0}, return_type='list')['TotalpSTAT']]
    dr_curve_b_refrac = [el[0] for el in USP18_Model.doseresponse([60], 'TotalpSTAT', 'Ib', dose_list,
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
    print("Writing Explicit-USP18-Refractory figure to results/refractoriness_explicitUSP18.pdf")
    fig.savefig('results/refractoriness_explicitUSP18.pdf')


