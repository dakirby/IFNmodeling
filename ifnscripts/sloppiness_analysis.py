from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numdifftools as nd
import os


def chi2(model: IfnModel, new_parameters: dict, times: list):
    reset_dict = dict(zip(new_parameters, [model.parameters[key] for key in new_parameters.keys()]))
    dra1 = model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-1, 5)),
                                     parameters={'Ib': 0}, return_type='list')['TotalpSTAT']
    drb1 = model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 4)),
                                     parameters={'Ia': 0}, return_type='list')['TotalpSTAT']

    model.set_parameters(new_parameters)

    dra2 = model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-1, 5)),
                                     parameters={'Ib': 0}, return_type='list')['TotalpSTAT']
    drb2 = model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 4)),
                                     parameters={'Ia': 0}, return_type='list')['TotalpSTAT']

    model.set_parameters(reset_dict)

    return (np.sum(np.square(np.divide(np.subtract(dra1, dra2), max(max(dra1), max(dra2))))) +
            np.sum(np.square(np.divide(np.subtract(drb1, drb2), max(max(drb1), max(drb2))))))/60.


def model_hessian(model: IfnModel, test_params: list):
    t = [2.5, 5, 7.5, 10, 20, 60]
    epsilon = 1E-6
    Hessian = np.zeros((len(test_params), len(test_params)))
    for p1, i in enumerate(test_params):
        for p2, j in enumerate(test_params):
            if p1 == p2:
                testval = {p1: model.parameters[p1] * 2 * epsilon}
            else:
                testval = {p1: model.parameters[p1] * epsilon, p2: model.parameters[p2] * epsilon}
            diff = chi2(model, testval, t)
            Hessian[i][j] = diff
    return Hessian


def function_builder(model: IfnModel, test_params: list, times: list):
    def _function(x):
        theta = dict(zip(test_params, x))
        scalar = chi2(model, theta, times)
        return scalar
    return _function

if __name__ == '__main__':
    Detailed_Model = IfnModel('Mixed_IFN_detailed')
    # Put detailed model into parameter state best fit for Ratnadeep's data (20181113_B6_IFNs_Dose_Response_Bcells)
    baseline_parameters = {'R2': 2300 * 2.5,
         'R1': 1800 * 1.8, 'k_d4': 0.06, 'kint_b': 0.0008,
         'kpu': 0.0028,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 6, 'kSOCSon': 0.9e-8, 'k_d4': 0.06,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3, 'kd4': 1.0, 'kd3': 0.001,
         'kint_a': 0.0014, 'krec_a1': 9e-03, 'krec_a2': 0.05}
    Detailed_Model.set_parameters(baseline_parameters)
    # -----------------
    # Compute Hessian
    # -----------------
    #   First create parameter vector to compute Hessian at
    parameters_to_test = list(baseline_parameters.keys())
    all_parameters = ['ka1', 'kd1', 'ka2', 'kd2', 'ka3', 'kd3', 'ka4', 'kd4', 'k_a1', 'k_d1', 'k_a2', 'k_d2', 'k_a3', 'k_d3', 'k_a4', 'k_d4',
     'kint_a', 'kint_b', 'krec_a1', 'krec_a2', 'krec_b1', 'krec_b2', 'kIntBasal_r1', 'kIntBasal_r2', 'krec_r1', 'krec_r2', 'kdeg_a', 'kdeg_b',
     'kSTATbinding', 'kSTATunbinding', 'kpa', 'kpu', 'kloc', 'kdeloc', 'mRNAdeg', 'mRNAtrans', 'kSOCS', 'SOCSdeg', 'kSOCSon', 'kSOCSoff']
    parameters_to_test.extend([el for el in all_parameters if el not in parameters_to_test])

    best_parameters = [Detailed_Model.parameters[key] for key in all_parameters]

    #   Now define the X**2 function which we want to compute Hessian of. It is defined using higher order function
    #   since this allows the Hessian to defined dynamically.
    new_function = function_builder(Detailed_Model, parameters_to_test, list(np.linspace(0, 60, num=61)))

    # Use numdifftools to compute the Hessian
    H = nd.Hessian(new_function)(best_parameters)
    H = np.nan_to_num(H)
    pickle.dump(H, open(os.path.join(os.getcwd(), 'results', 'Sloppiness', 'Hessian.pkl'), "wb"))
    print(H)
    # Compute eigenspectrum and normalize by largest eigenvalue
    #   - small eigenvalues correspond to sloppy parameters while large eigenvalues give stiff parameters
    #   - strictly speaking the eigenvalues correspond to the sloppiness along principal axes (eigenvectors) of H
    #   - the width of the error ellipse along the corresponding axis is 1/sqrt(eval)
    evals = np.linalg.eigvals(H)
    evals = np.nan_to_num(evals)
    normalized_evals = np.divide(evals, max(evals))
    log_normalized_evals = log10(normalized_evals)

    # Plot spectrum on log scale
    spec_plot = sns.distplot(log_normalized_evals, kde=False, rug=True)
    spec_plot.set(xlabel=r'log($\lambda / \lambda_{max}$)', ylabel='bin count')
    minX, maxX = plt.xlim()
    minY, maxY = plt.ylim()
    yval = (maxY + minY) / 2. * 0.6
    spec_plot.annotate('Sloppiness', xy=(minX, yval), xytext=((maxX + minX) / 2. * 0.8, yval + 0.1),
                       # draws an arrow from one set of coordinates to the other
                       arrowprops=dict(facecolor='black', width=3),  # sets style of arrow and colour
                       annotation_clip=False)  # This enables the arrow to be outside of the plot
    spec_plot.annotate('', xy=((maxX + minX) / 2. * 0.8 + 1.1, yval), xytext=(maxX, yval),
                       # draws an arrow from one set of coordinates to the other
                       arrowprops=dict(facecolor='black', width=3),  # sets style of arrow and colour
                       annotation_clip=False)  # This enables the arrow to be outside of the plot

    plt.savefig(os.path.join(os.getcwd(), 'results', 'Sloppiness', 'spectrum.pdf'))
    plt.show()



