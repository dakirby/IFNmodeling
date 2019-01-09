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
    # Get data EC50's
    """
    newdata = IfnData("20181113_B6_IFNs_Dose_Response_Bcells")
    data_times = newdata.data_set.loc['Beta'].columns.tolist()
    data_doses = newdata.data_set.loc['Beta'].index.tolist()
    data_responses = transpose([[el[0] for el in row] for row in newdata.data_set.loc['Alpha'].values])
    data_fit_pairs = [[],[], [2.5, 5.0, 7.5, 10.0, 20.0, 60.0], [5206.0315092746596, 5360.5625768863702, 5000.0, 1952.5080877034763, 3151.680990434103, 1000.0]]
    for time in range(len(data_times)):
        top, n, K = fit_MM(data_doses, data_responses[time], [5, 0.8, 100])
        data_fit_pairs[0].append(data_times[time])
        data_fit_pairs[1].append(K)

        plt.figure()
        plt.semilogx(list(logspace(1,4)), MM(list(logspace(1,5)), top, n, K))
        plt.scatter(data_doses, data_responses[time])
        plt.plot([K,K],[0,0.5*top],'r')
        plt.plot([0, K], [0.5 * top, 0.5 * top], 'r')
        plt.show()
        print(top, n, K)
    """

    data_fit_pairs = [[2.5, 5.0, 7.5, 10.0, 20.0, 60.0], [101.96844057741836, 102.68147858829717, 77, 35.016268816362334, 41, 10.710845593890753], [2.5, 5.0, 7.5, 10.0, 20.0, 60.0], [5206.03150927466, 5360.56257688637, 5000.0, 1952.5080877034763, 3151.680990434103, 1000.0]]

    # Get model EC50's
    Mixed_Model = IfnModel('')
    Mixed_Model.load_model('fitting_2_5.p')

    Mixed_Model.set_parameters(
        {'kpu': 0.0028, 'R2': 2300 * 2.5, 'R1': 1800 * 1.8, 'k_d4': 0.06, 'kint_b': 0.0003, 'krec_b1': 0.001,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 4, 'kSOCSon': 0.9e-8,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3,
         'kint_a': 0.0014, 'krec_a1': 9e-03})

    scale_factor = 0.036  # 0.02894064
    time_list = list(linspace(2.5, 60, num=15))
    alpha_peak, alpha_n, alpha_ec50 = get_ec50(Mixed_Model, time_list, 'Ia', 'TotalpSTAT', custom_parameters={'Ib': 0}, rflag=True)
    beta_peak, beta_n, beta_ec50 = get_ec50(Mixed_Model, time_list, 'Ib', 'TotalpSTAT', custom_parameters={'Ia': 0}, rflag=True)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].set_xlabel("Time (s)")
    axes[1].set_xlabel("Time (s)")
    axes[0].set_title(r"EC50 vs Time for IFN$\alpha$")
    axes[1].set_title(r"EC50 vs Time for IFN$\beta$")
    axes[0].set_ylabel("EC50")
    #axes[0].set(xscale='linear', yscale='log')
    #axes[1].set(xscale='linear', yscale='log')
    axes[0].plot(time_list, alpha_ec50, label=r'IFN$\alpha$ EC50', color=alpha_palette[4])
    axes[1].plot(time_list, beta_ec50, label=r'IFN$\beta$ EC50', color=beta_palette[4])
    axes[0].scatter(data_fit_pairs[2], data_fit_pairs[3], label='data', color=alpha_palette[3])
    axes[1].scatter(data_fit_pairs[0], data_fit_pairs[1], label='data', color=alpha_palette[3])
    fig.show()
    fig.savefig('results\ec50_vs_time.pdf')
    exit()

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].set_xlabel("Time (s)")
    axes[1].set_xlabel("Time (s)")
    axes[0].set_title(r"Peak Response vs Time for IFN$\alpha$")
    axes[1].set_title(r"Peak Response vs Time for IFN$\beta$")
    axes[0].set_ylabel("Peak Response")
    axes[0].plot(time_list, alpha_peak, label=r'IFN$\alpha$ Peak Response', color=alpha_palette[4])
    axes[1].plot(time_list, beta_peak, label=r'IFN$\beta$ Peak Response', color=beta_palette[4])
    fig.show()
    fig.savefig('results\peak_response_vs_time.pdf')

