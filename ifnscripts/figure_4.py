from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnfit import DualMixedPopulation
from ifnclass.ifndata import IfnData
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
from scipy.optimize import curve_fit


def MM(xdata, top, k):
    ydata = [top * x / (k + x) for x in xdata]
    return ydata


def fit_MM(doses, responses, guesses):
    top = guesses[0]
    K = guesses[1]
    results, covariance = curve_fit(MM, doses, responses, p0=[top, K])
    top = results[0]
    K = results[1]
    if K > 4E3:
        top = max(responses) * 0.5
        for i, r in enumerate(responses):
            if r > top:
                K = 10 ** ((np.log10(doses[i-1]) + np.log10(doses[i])) / 2.0)
                break
    return top, K


def get_ec50(model: IfnModel, times: list or int, dose_species: str, response_species: str, custom_parameters={},
             rflag=False):
    if type(times) == int or type(times) == float:
        dr_curve = [el[0] for el in model.doseresponse([times], response_species, dose_species, list(logspace(-3, 5)),
                                      parameters=custom_parameters, return_type='list')[response_species]]

        top, K = fit_MM(list(logspace(-3, 5)), dr_curve, [max(dr_curve), 1000])
        if rflag:
            return top, K
        else:
            return K

    elif type(times) == list:
        dr_curve = model.doseresponse(times, response_species, dose_species, list(logspace(-3, 5)),
                                      parameters=custom_parameters, return_type='list')[response_species]

        top_list = []
        K_list = []
        for t in range(len(times)):
            tslice = [el[t] for el in dr_curve]
            top, K = fit_MM(list(logspace(-3, 5)), tslice, [max(tslice), 1000])
            top_list.append(top)
            K_list.append(K)
        if rflag:
            return top_list, K_list
        else:
            return K_list

    else:
        raise TypeError("Could not identify type for variable times")


if __name__ == '__main__':
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')

    # fit_all_GAB_data parameters plus K_1 * 0.2 and K_2 * 0.2
    Mixed_Model.set_parameters({'R2': 4920, 'R1': 1200,
                                'k_a1': 2.0e-13 * 5, 'k_a2': 1.328e-12 * 5, 'k_d3': 1.13e-4, 'k_d4': 0.9,
                                'kSOCSon': 5e-08, 'kpu': 0.0022, 'kpa': 2.36e-06,
                                'ka1': 3.3E-14*5, 'ka2': 5E-13*5, 'kd4': 2.0, 'kd3': 6.52e-05,
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
    fig.savefig(os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_4', 'Figure_4_Refractoriness_By_K4.pdf'))

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

    # -----------------------------------------------------------------------
    # Finally, make a figure about how EC50 changes as a function of K3 & K4
    # -----------------------------------------------------------------------
    # Make model predictions
    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
    # Optimal parameters for fitting mean GAB data
    opt_params = {'R2': 4920, 'R1': 1200,
                    'k_a1': 2.0e-13, 'k_a2': 1.328e-12, 'k_d3': 1.13e-4, 'k_d4': 0.9,
                    'kSOCSon': 5e-08, 'kpu': 0.0022, 'kpa': 2.36e-06,
                    'ka1': 3.3e-15, 'ka2': 1.85e-12, 'kd4': 2.0,
                    'kd3': 6.52e-05,
                    'kint_a': 0.0015, 'kint_b': 0.002,
                    'krec_a1': 0.01, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05}
    Mixed_Model.set_parameters(opt_params)
    Mixed_Model.default_parameters.update(opt_params)
    scale_factor = 1.46182313424
    # Fold changes to test over
    fold_changes = list(np.logspace(-2, 2))

    def curve_builder(fold_change):
        time_list = list(linspace(2.5, 60.0, num=15))
        Mixed_Model.set_parameters({'ka3': Mixed_Model.parameters['ka3'] / fold_change,
                                    'ka4': Mixed_Model.parameters['ka4'] / fold_change,
                                    'k_a3': Mixed_Model.parameters['k_a3'] / fold_change,
                                    'k_a4': Mixed_Model.parameters['k_a4'] / fold_change})

        alpha_peak_aggregate, alpha_ec_aggregate = get_ec50(Mixed_Model, time_list, 'Ia', 'TotalpSTAT',
                                                            custom_parameters={'Ib': 0}, rflag=True)
        beta_peak_aggregate, beta_ec_aggregate = get_ec50(Mixed_Model, time_list, 'Ib', 'TotalpSTAT',
                                                          custom_parameters={'Ia': 0}, rflag=True)
        # alpha_peak_aggregate = np.multiply(alpha_peak_aggregate, scale_factor)
        # beta_peak_aggregate = np.multiply(beta_peak_aggregate, scale_factor)
        Mixed_Model.reset_parameters()
        return alpha_ec_aggregate[-1], beta_ec_aggregate[-1]

    alpha_curve = []
    beta_curve = []
    for f in fold_changes:
        x1, x2 = curve_builder(f)
        alpha_curve.append(x1)
        beta_curve.append(x2)

    # Set up plot
    ec50_plot, ec50_axes = plt.subplots(1, 1, figsize=(7.5, 6))
    ec50_axes.set_xlabel("Fold change in K3 & K4")
    ec50_axes.set_ylabel(r"$EC_{50}$ (pM)")
    plt.suptitle(r"EC50 at 60 minutes")
    ec50_axes.set_yscale('log')
    ec50_axes.set_xscale('log')
    ec50_axes.plot(fold_changes, alpha_curve, color=alpha_palette[5], linewidth=2.0, label=r'IFN$\alpha$')
    ec50_axes.plot(fold_changes, beta_curve, color=beta_palette[5], linewidth=2.0, label=r'IFN$\beta$')
    plt.legend()
    ec50_plot.savefig(os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_4', 'Figure_4_EC50_vs_K3_and_K4.pdf'))