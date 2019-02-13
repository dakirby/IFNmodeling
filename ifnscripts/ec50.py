from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from numpy import linspace, logspace, transpose
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


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

    # Get all data set EC50 time courses
    # 20190119
    data_20190119 = IfnData("20190119_pSTAT1_IFN_Bcell")
    ec50_20190119 = data_20190119.get_ec50s()
    ec50_20190119['Alpha'][-1] = (ec50_20190119['Alpha'][-1][0], 800.)

    # 20190121
    data_20190121 = IfnData("20190121_pSTAT1_IFN_Bcell")
    ec50_20190121 = data_20190121.get_ec50s()
    #ec50_20190121['Alpha'][-1] = (ec50_20190121['Alpha'][-1][0], 800.)

    # 20190108
    data_20190108 = IfnData("20190108_pSTAT1_IFN")
    ec50_20190108 = data_20190108.get_ec50s()
    ec50_20190108['Beta'][-1] = (ec50_20190108['Beta'][-1][0], 20.)


    # Make model predictions
    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
    fit_parameters = {'kd4': 1.0, 'krec_a1': 3.0000000000000001e-05, 'krec_a2': 0.050000000000000003, 'krec_b2': 0.01,
         'krec_b1': 0.001, 'k_d4': 0.00059999999999999995, 'kSOCSon': 1e-08, 'kd3': 0.001,
         'k_d3': 2.3999999999999999e-06}
    Mixed_Model.set_parameters(fit_parameters)

    Mixed_Model.set_parameters(
        {'kpu': 0.0028, 'R2': 2300 * 2.5, 'R1': 1800 * 1.8, 'k_d4': 0.06, 'kint_b': 0.0003, 'krec_b1': 0.001,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 4, 'kSOCSon': 0.9e-8,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3,
         'kint_a': 0.0014, 'krec_a1': 9e-03})

    scale_factor = 0.036  # 0.02894064
    time_list = list(linspace(2.5, 60, num=15))
    alpha_peak, alpha_n, alpha_ec50 = get_ec50(Mixed_Model, time_list, 'Ia', 'TotalpSTAT', custom_parameters={'Ib': 0}, rflag=True)
    beta_peak, beta_n, beta_ec50 = get_ec50(Mixed_Model, time_list, 'Ib', 'TotalpSTAT', custom_parameters={'Ia': 0}, rflag=True)

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].set_xlabel("Time (s)")
    axes[1].set_xlabel("Time (s)")
    axes[0].set_title(r"EC50 vs Time for IFN$\alpha$")
    axes[1].set_title(r"EC50 vs Time for IFN$\beta$")
    axes[0].set_ylabel("EC50")
    # Add model
    axes[0].plot(time_list, alpha_ec50, label=r'IFN$\alpha$ EC50', color=alpha_palette[4])
    axes[1].plot(time_list, beta_ec50, label=r'IFN$\beta$ EC50', color=beta_palette[4])
    # Add data
    axes[0].scatter([el[0] for el in ec50_20190119['Alpha']], [el[1] for el in ec50_20190119['Alpha']], label='data', color=alpha_palette[3])
    axes[1].scatter([el[0] for el in ec50_20190119['Beta']], [el[1] for el in ec50_20190119['Beta']], label='data', color=alpha_palette[3])
    fig.show()
    fig.savefig('results\ec50_vs_time.pdf')