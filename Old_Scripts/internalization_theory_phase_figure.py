from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import seaborn as sns
from scipy.optimize import curve_fit
import pandas as pd


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
    if n > 3:
        n = 3
    return top, n, K


def get_ec50(model: IfnModel, times: list or int, dose_species: str, response_species: str, custom_parameters={},
             rflag=False):
    if type(times) == int or type(times) == float:
        dr_curve = [el[0] for el in model.doseresponse([times], response_species, dose_species, list(logspace(-3, 5)),
                                      parameters=custom_parameters, return_type='list')[response_species]]

        top, n, K = fit_MM(list(logspace(-3, 5)), dr_curve, [max(dr_curve), 1, 500])
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

    Mixed_Model.set_parameters(
        {'R2': 2300 * 2.5,
         'R1': 1800 * 1.8, 'k_d4': 0.06, 'kint_b': 0.0003,
         'kpu': 0.0028,
         'krec_b1': 0.001, 'krec_b2': 0.01,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 4, 'kSOCSon': 0.9e-8,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3, 'kd4': 1.0, 'kd3': 0.001,
         'kint_a': 0.0014, 'krec_a1': 9e-03, 'krec_a2': 0.05})

    # Produce plots
    time = 60
    x_axis_values = list( 0.0014 * logspace(-2,2)) # kint_a values
    y_axis_values = list(0.0008 * logspace(-2,2)) # kdeg_a values
    ec50_phase_a = []
    # Iterate through parameter values
    progress = 0
    for r in y_axis_values[::-1]:
        print("{:.1f}% done".format(progress/len(y_axis_values)*100))
        progress += 1
        row = []
        for c in x_axis_values:
            Mixed_Model.set_parameters({'kint_a': c, 'kdeg_a': r})
            alpha_ec50 = get_ec50(Mixed_Model, time, 'Ia', 'TotalpSTAT', custom_parameters={'Ib': 0})
            row.append(alpha_ec50)
        ec50_phase_a.append(row)
    sns_ec50_a = sns.heatmap(pd.DataFrame(ec50_phase_a,index=y_axis_values[::-1],columns=x_axis_values))
    sns_ec50_a.savefig('results/ec50_heatmap_a.pdf')