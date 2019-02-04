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
    alpha_norm = max(max([item for sublist in dra1 for item in sublist]), max([item for sublist in dra2 for item in sublist]))
    beta_norm = max(max([item for sublist in drb1 for item in sublist]), max([item for sublist in drb2 for item in sublist]))

    value = (np.sum(np.square(np.divide(np.subtract(dra1, dra2), alpha_norm))) +
            np.sum(np.square(np.divide(np.subtract(drb1, drb2), beta_norm))))/times[-1]
    value = np.nan_to_num(value)
    return value


def function_builder(model: IfnModel, test_params: list, times: list):
    def _function(x):
        theta = dict(zip(test_params, np.exp(x)))
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
    print(len(parameters_to_test))
    exit()
    # Add jitter to best fit parameters to avoid numerical instability of finding Hessian at functional 0 (?)
    best_parameters = np.log([Detailed_Model.parameters[key] * np.random.uniform(0.97, 1.03) for key in parameters_to_test])

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



