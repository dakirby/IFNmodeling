from ifnclass.ifnmodel import IfnModel
from numpy import linspace, logspace
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np


def MM(xdata, top, n, k):
    ydata = [top * x ** n / (k ** n + x ** n) for x in xdata]
    return ydata


def fit_MM(doses, responses, guesses):
    top = guesses[0]
    n = guesses[1]
    K = guesses[2]
    results, covariance = curve_fit(MM, doses, responses, p0=[top, n, K])
    top = results[0]
    n = results[1]
    K = results[2]
    if n > 5:
        n = 5
    return top, n, K


def get_ec50(model: IfnModel, times: list or int, dose_species: str, response_species: str, custom_parameters={},
             rflag=False):
    if type(times) == int or type(times) == float:
        dr_curve = [el[0] for el in model.doseresponse([times], response_species, dose_species, list(logspace(-3, 5)),
                                      parameters=custom_parameters, return_type='list')[response_species]]

        top, n, K = fit_MM(list(logspace(-3, 5)), dr_curve, [max(dr_curve), 1, 1000])
        if rflag:
            return top, n, K
        else:
            return K

    elif type(times) == list:
        dr_curve = model.doseresponse(times, response_species, dose_species, list(logspace(-3, 5)),
                                      parameters=custom_parameters, return_type='list')[response_species]

        top_list = []
        n_list = []
        K_list = []
        for t in range(len(times)):
            tslice = [el[t] for el in dr_curve]
            top, n, K = fit_MM(list(logspace(-3, 5)), tslice, [max(tslice), 1, 1000])
            top_list.append(top)
            n_list.append(n)
            K_list.append(K)
        if rflag:
            return top_list, n_list, K_list
        else:
            return K_list

    else:
        raise TypeError("Could not identify type for variable times")


if __name__ == '__main__':
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
    USP18_Model = IfnModel('Mixed_IFN_explicitUSP18')
    print(USP18_Model.parameters)
    exit()

    Mixed_Model.set_parameters(
        {'kpu': 0.0028, 'R2': 2300 * 2.5, 'R1': 1800 * 1.8, 'k_d4': 0.06, 'kint_b': 0.0003,
         'krec_b1': 0.001, 'krec_b2': 0.01,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 4, 'kSOCSon': 0.9e-8,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3, 'kd4': 1.0, 'kd3': 0.001,
         'kint_a': 0.0014, 'krec_a1': 9e-03, 'krec_a2': 0.05})

    scale_factor = 0.036  # 0.02894064
    dose_list = list(logspace(-2, 6, num=20))
    dr_curve_a = [el[0] for el in Mixed_Model.doseresponse([60], 'TotalpSTAT', 'Ia', dose_list,
                                                   parameters={'Ib': 0}, return_type='list')['TotalpSTAT']]
    dr_curve_b = [el[0] for el in Mixed_Model.doseresponse([60], 'TotalpSTAT', 'Ib', dose_list,
                                                   parameters={'Ia': 0}, return_type='list')['TotalpSTAT']]
    # Now compute the 5* refractory response
    k4sf1 = 3.5
    Mixed_Model.set_parameters({'kd4': 1.0*k4sf1, 'k_d4': 0.06*k4sf1})
    dr_curve_a20 = [el[0] for el in Mixed_Model.doseresponse([60], 'TotalpSTAT', 'Ia', dose_list,
                                                   parameters={'Ib': 0}, return_type='list')['TotalpSTAT']]
    dr_curve_b20 = [el[0] for el in Mixed_Model.doseresponse([60], 'TotalpSTAT', 'Ib', dose_list,
                                                   parameters={'Ia': 0}, return_type='list')['TotalpSTAT']]
    # Now compute the 10* refractory response
    k4sf2 = 10
    Mixed_Model.set_parameters({'kd4': 1.0*k4sf2, 'k_d4': 0.06*k4sf2})
    dr_curve_a60 = [el[0] for el in Mixed_Model.doseresponse([60], 'TotalpSTAT', 'Ia', dose_list,
                                                   parameters={'Ib': 0}, return_type='list')['TotalpSTAT']]
    dr_curve_b60 = [el[0] for el in Mixed_Model.doseresponse([60], 'TotalpSTAT', 'Ib', dose_list,
                                                   parameters={'Ia': 0}, return_type='list')['TotalpSTAT']]

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].set_xlabel("Dose (pM)")
    axes[1].set_xlabel("Dose (pM)")
    axes[0].set_title("Relative Refractory Response")
    axes[1].set_title(r"Absolute Responses")
    axes[0].set_ylabel("pSTAT1")
    axes[0].set(xscale='log', yscale='linear')
    axes[1].set(xscale='log', yscale='linear')
    axes[0].plot(dose_list, np.divide(dr_curve_a20, dr_curve_a),
                 label=r'IFN$\alpha$ $K_{D4}\times$'+'{}'.format(k4sf1), color=alpha_palette[2])
    axes[0].plot(dose_list, np.divide(dr_curve_a60, dr_curve_a),
                 label=r'IFN$\alpha$ $K_{D4}\times$'+'{}'.format(k4sf2), color=alpha_palette[4])
    axes[0].plot(dose_list, np.divide(dr_curve_b20, dr_curve_b),
                 label=r'IFN$\beta$ $K_{D4}\times$'+'{}'.format(k4sf1), color=beta_palette[2])
    axes[0].plot(dose_list, np.divide(dr_curve_b60, dr_curve_b),
                 label=r'IFN$\beta$ $K_{D4}\times$'+'{}'.format(k4sf2), color=beta_palette[4])
    axes[1].plot(dose_list, dr_curve_a, label=r'IFN$\alpha$', color=alpha_palette[0])
    axes[1].plot(dose_list, dr_curve_a20, label=r'IFN$\alpha$ $K_{D4}\times$'+'{}'.format(k4sf1), color=alpha_palette[2])
    axes[1].plot(dose_list, dr_curve_a60, label=r'IFN$\alpha$ $K_{D4}\times$'+'{}'.format(k4sf2), color=alpha_palette[4])
    axes[1].plot(dose_list, dr_curve_b, label=r'IFN$\beta$', color=beta_palette[0])
    axes[1].plot(dose_list, dr_curve_b20, label=r'IFN$\beta$ $K_{D4}\times$'+'{}'.format(k4sf1), color=beta_palette[2])
    axes[1].plot(dose_list, dr_curve_b60, label=r'IFN$\beta$ $K_{D4}\times$'+'{}'.format(k4sf2), color=beta_palette[4])

    axes[0].legend(loc=2, prop={'size': 5})
    axes[1].legend(loc=2, prop={'size': 5})
    fig.show()
    fig.savefig('results/refractoriness.pdf')


