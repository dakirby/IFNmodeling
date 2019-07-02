from ifnclass.ifnmodel import IfnModel
from numpy import linspace, logspace
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)


if __name__ == '__main__':
    # ---------------------------------------------------
    # Now make the figure where we explicitly model USP18
    # ---------------------------------------------------
    def explicit_UPS18_figure():
        alpha_palette = sns.color_palette("Reds", 6)
        beta_palette = sns.color_palette("Greens", 6)
        # --------------------
        # Set up Model
        # --------------------
        fraction_USP18_occupied = 0.2
        dose_list = list(logspace(-2, 5, num=46))

        # Parameters found by stepwise fitting GAB mean data
        initial_parameters = {'k_a1': 4.98E-14 * 2, 'k_a2': 8.30e-13 * 2, 'k_d4': 0.006 * 3.8,
                              'kpu': 0.00095,
                              'ka2': 4.98e-13 * 2.45, 'kd4': 0.3 * 2.867,
                              'kint_a': 0.000124, 'kint_b': 0.00086,
                              'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05}
        dual_parameters = {'kint_a': 0.00052, 'kSOCSon': 6e-07, 'kint_b': 0.00052, 'krec_a1': 0.001, 'krec_a2': 0.1,
                           'krec_b1': 0.005, 'krec_b2': 0.05}
        scale_factor = 1.227

        USP18_Model = DualMixedPopulation('Mixed_IFN_explicitUSP18', 0.8, 0.2)
        USP18_Model.model_1.set_parameters(initial_parameters)
        USP18_Model.model_1.set_parameters(dual_parameters)
        USP18_Model.model_1.set_parameters({'R1': 12000.0, 'R2_0': 1511.1 * (1 - fraction_USP18_occupied),
                                            'R2USP18_0': 1511.1 * fraction_USP18_occupied})
        USP18_Model.model_2.set_parameters(initial_parameters)
        USP18_Model.model_2.set_parameters(dual_parameters)
        USP18_Model.model_2.set_parameters({'R1': 6755.56, 'R2_0': 1511.1 * (1 - fraction_USP18_occupied),
                                            'R2USP18_0': 1511.1 * fraction_USP18_occupied})

        # Prepare model for reset
        USP18_Model.model_1.default_parameters = USP18_Model.model_1.parameters
        USP18_Model.model_2.default_parameters = USP18_Model.model_2.parameters

        """
        fraction_USP18_occupied = 0.6
        dose_list = list(logspace(-2, 5, num=46))
        USP18_parameters = {'R2_0': 2300 * 2.5 * (1 - fraction_USP18_occupied),
                            'R2USP18_0': 2300 * 2.5 * fraction_USP18_occupied,
                            'R1': 1800 * 1.8, 'k_d4': 0.06, 'k_d4_USP18': 0.06 * 15, 'kint_b': 0.0003,
                            'kpu': 0.0028,
                            'krec_b1': 0.001, 'krec_b2': 0.01,
                            'kSOCSon': 0.9e-8,
                            'kd4': 1.0, 'kd3': 0.001,
                            'kint_a': 0.0014, 'krec_a1': 9e-03, 'krec_a2': 0.05}
        scale_factor = 1.227

        USP18_Model = DualMixedPopulation('Mixed_IFN_explicitUSP18', 1.0, 0.0)
        USP18_Model.model_1.set_parameters(USP18_parameters)
        USP18_Model.model_2.set_parameters(USP18_parameters)
        """

        # Refractory
        dr_curve_a_refrac = [el[0][0] for el in USP18_Model.mixed_dose_response([60], 'TotalpSTAT', 'Ia', dose_list,
                                                                                parameters={'Ib': 0},
                                                                                sf=scale_factor).values]
        dr_curve_b_refrac = [el[0][0] for el in USP18_Model.mixed_dose_response([60], 'TotalpSTAT', 'Ib', dose_list,
                                                                                parameters={'Ia': 0},
                                                                                sf=scale_factor).values]
        # Baseline
        USP18_Model.set_global_parameters({'R2_0': 2300 * 2.5, 'R2USP18_0': 0})

        dr_curve_a = [el[0][0] for el in USP18_Model.mixed_dose_response([60], 'TotalpSTAT', 'Ia', dose_list,
                                                                         parameters={'Ib': 0}, sf=scale_factor).values]
        dr_curve_b = [el[0][0] for el in USP18_Model.mixed_dose_response([60], 'TotalpSTAT', 'Ib', dose_list,
                                                                         parameters={'Ia': 0}, sf=scale_factor).values]

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
        fig.savefig(os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_4', 'Figure_S4_explicitUSP18.pdf'))

    explicit_UPS18_figure()