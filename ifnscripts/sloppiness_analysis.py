from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import numpy as np
import seaborn as sns


def chi2(model: IfnModel, new_parameters: dict, times: list):
    dra1 = model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-1, 5)),
                                     parameters={'Ib': 0}, return_type='list')['TotalpSTAT']
    drb1 = model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 4)),
                                     parameters={'Ia': 0}, return_type='list')['TotalpSTAT']
    model.set_parameters(new_parameters)
    dra2 = model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-1, 5)),
                                     parameters={'Ib': 0}, return_type='list')['TotalpSTAT']
    drb2 = model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 4)),
                                     parameters={'Ia': 0}, return_type='list')['TotalpSTAT']

    return np.sum(np.square(np.subtract(dra1, dra2)))+np.sum(np.square(np.subtract(drb1, drb2)))


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


if __name__ == '__main__':
    Detailed_Model = IfnModel('Mixed_IFN_detailed')
    # Put detailed model into parameter state best fit for Ratnadeep's data (20181113_B6_IFNs_Dose_Response_Bcells)
    Detailed_Model.set_parameters(
        {'R2': 2300 * 2.5,
         'R1': 1800 * 1.8, 'k_d4': 0.06, 'kint_b': 0.0008,
         'kpu': 0.0028,
         'krec_b1': 0.001, 'krec_b2': 0.01,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 6, 'kSOCSon': 0.9e-8, 'k_d4': 0.06,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3, 'kd4': 1.0, 'kd3': 0.001,
         'kint_a': 0.0014, 'krec_a1': 9e-03, 'krec_a2': 0.05})



