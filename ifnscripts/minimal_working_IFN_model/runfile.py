from model_class_file import IfnModel
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    best_fit_params = {'k_a1': 4.98E-14 * 2, 'k_a2': 8.30e-13 * 2,
                       'k_d4': 0.006 * 3.8,
                       'kpu': 0.0095,
                       'ka2': 4.98e-13 * 2.45, 'kd4': 0.3 * 2.867,
                       'kint_a': 0.00052, 'kint_b': 0.00052,
                       'krec_a1': 0.001, 'krec_a2': 0.1,
                       'krec_b1': 0.005, 'krec_b2': 0.05,
                       'kSOCSon': 6e-07,
                       'R1': 9000.0, 'R2': 1511.1}
    testModel = IfnModel('IFN_model')
    testModel.set_parameters(best_fit_params)
    tc = testModel.timecourse([0, 5, 15, 30], 'TotalpSTAT', return_type='dict',
                              dataframe_labels=['Alpha', 1])
    dr = testModel.doseresponse([0, 5, 15, 30], ['Ta', 'TotalpSTAT'], 'Ia',
                                [1, 10, 100],
                                return_type='dataframe',
                                dataframe_labels='Alpha',
                                scale_factor=1.5)
    print(tc)
    print(dr)

    fig, ax = plt.subplots(nrows=1,ncols=1)
    plt.plot([1,10,100], dr.values[3:,1])
    ax.set_xscale('log')
    plt.show()
