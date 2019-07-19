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
from ifnclass.ifnplot import DoseresponsePlot


if __name__ == '__main__':
    # ---------------------------------------------------
    # First make the figure where we increase K4
    # ---------------------------------------------------
    def increase_K4_figure():
        alpha_palette = sns.color_palette("Reds", 6)
        beta_palette = sns.color_palette("Greens", 6)

        # --------------------
        # Set up Model
        # --------------------
        # Parameters found by stepwise fitting GAB mean data
        initial_parameters = {'k_a1': 4.98E-14 * 2, 'k_a2': 8.30e-13 * 2, 'k_d4': 0.006 * 3.8,
                              'kpu': 0.00095,
                              'ka2': 4.98e-13 * 2.45, 'kd4': 0.3 * 2.867,
                              'kint_a': 0.000124, 'kint_b': 0.00086,
                              'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05}
        dual_parameters = {'kint_a': 0.00052, 'kSOCSon': 6e-07, 'kint_b': 0.00052, 'krec_a1': 0.001, 'krec_a2': 0.1,
                           'krec_b1': 0.005, 'krec_b2': 0.05}
        scale_factor = 1.227

        Mixed_Model = DualMixedPopulation('Mixed_IFN_ppCompatible', 0.8, 0.2)
        Mixed_Model.model_1.set_parameters(initial_parameters)
        Mixed_Model.model_1.set_parameters(dual_parameters)
        Mixed_Model.model_1.set_parameters({'R1': 12000.0, 'R2': 1511.1})
        Mixed_Model.model_2.set_parameters(initial_parameters)
        Mixed_Model.model_2.set_parameters(dual_parameters)
        Mixed_Model.model_2.set_parameters({'R1': 6755.56, 'R2': 1511.1})

        dose_list = list(logspace(-2, 8, num=35))

        dr_curve_a = [el[0][0] for el in Mixed_Model.mixed_dose_response([60], 'TotalpSTAT', 'Ia', dose_list,
                                                                         parameters={'Ib': 0}, sf=scale_factor).values]

        dr_curve_b = [el[0][0] for el in Mixed_Model.mixed_dose_response([60], 'TotalpSTAT', 'Ib', dose_list,
                                                                         parameters={'Ia': 0}, sf=scale_factor).values]

        # Now compute the 20* refractory response
        k4sf1 = 15
        kd4_reference = Mixed_Model.model_1.parameters['kd4']
        kd3_reference = Mixed_Model.model_1.parameters['kd3']
        k_d4_reference = Mixed_Model.model_1.parameters['k_d4']
        k_d3_reference = Mixed_Model.model_1.parameters['k_d3']
        Mixed_Model.set_global_parameters({'kd4': kd4_reference*k4sf1, 'k_d4': k_d4_reference*k4sf1,
                                           'kd3': kd3_reference * k4sf1, 'k_d3': k_d3_reference * k4sf1})

        dr_curve_a20 = [el[0][0] for el in Mixed_Model.mixed_dose_response([60], 'TotalpSTAT', 'Ia', dose_list,
                                                                         parameters={'Ib': 0}, sf=scale_factor).values]
        dr_curve_b20 = [el[0][0] for el in Mixed_Model.mixed_dose_response([60], 'TotalpSTAT', 'Ib', dose_list,
                                                                         parameters={'Ia': 0}, sf=scale_factor).values]
        # Now compute the 60* refractory response
        k4sf2 = 60
        Mixed_Model.set_global_parameters({'kd4': kd4_reference*k4sf2, 'k_d4': k_d4_reference*k4sf2,
                                           'kd3': kd3_reference * k4sf2, 'k_d3': k_d3_reference * k4sf2})

        dr_curve_a60 = [el[0][0] for el in Mixed_Model.mixed_dose_response([60], 'TotalpSTAT', 'Ia', dose_list,
                                                                         parameters={'Ib': 0}, sf=scale_factor).values]
        dr_curve_b60 = [el[0][0] for el in Mixed_Model.mixed_dose_response([60], 'TotalpSTAT', 'Ib', dose_list,
                                                                         parameters={'Ia': 0}, sf=scale_factor).values]

        # Plot
        fig, axes = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(16, 8)
        axes.set_xlabel("Dose (pM)", fontsize=14)
        axes.set_title("Relative Refractory Response", fontsize=16)
        axes.set_ylabel("pSTAT1 Relative to Primary Response", fontsize=14)
        axes.set(xscale='log', yscale='linear')
        #axes.plot(dose_list, np.divide(dr_curve_a20, dr_curve_a), label=r'IFN$\alpha$ $K_{D4}\times$'+'{}'.format(k4sf1), color=alpha_palette[4], linestyle='-', linewidth=2)
        axes.plot(dose_list, np.divide(dr_curve_a60, dr_curve_a), label=r'IFN$\alpha$ $K_{D4}\times$'+'{}'.format(k4sf2), color=alpha_palette[4], linestyle='--', linewidth=2)
        #axes.plot(dose_list, np.divide(dr_curve_b20, dr_curve_b), label=r'IFN$\beta$ $K_{D4}\times$'+'{}'.format(k4sf1), color=beta_palette[4], linestyle='-', linewidth=2)
        axes.plot(dose_list, np.divide(dr_curve_b60, dr_curve_b), label=r'IFN$\beta$ $K_{D4}\times$'+'{}'.format(k4sf2), color=beta_palette[4], linestyle='--', linewidth=2)

        axes.legend(loc=2, prop={'size': 8})
        fig.set_size_inches(8, 8)
        fig.savefig(os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_4', 'Figure_4_Refractoriness_By_K4.pdf'))

        # Also plot absolute curves
        fig, axes = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(16, 8)
        axes.set_xlabel("Dose (pM)", fontsize=14)
        axes.set_title("Absolute Refractory Response", fontsize=16)
        axes.set_ylabel("pSTAT1", fontsize=14)
        axes.set(xscale='log', yscale='linear')

        axes.plot(dose_list, dr_curve_a, label=r'IFN$\alpha$', color=alpha_palette[2], linewidth=2)
        #axes.plot(dose_list, dr_curve_a20, label=r'IFN$\alpha$ $K_{D4}\times$' + '{}'.format(k4sf1), color=alpha_palette[4], linewidth=2)
        axes.plot(dose_list, dr_curve_a60, label=r'IFN$\alpha$ $K_{D4}\times$' + '{}'.format(k4sf2), color=alpha_palette[5], linestyle='dashed', linewidth=2)

        axes.plot(dose_list, dr_curve_b, label=r'IFN$\beta$', color=beta_palette[2], linewidth=2)
        #axes.plot(dose_list, dr_curve_b20, label=r'IFN$\beta$ $K_{D4}\times$' + '{}'.format(k4sf1), color=beta_palette[4], linewidth=2)
        axes.plot(dose_list, dr_curve_b60, label=r'IFN$\beta$ $K_{D4}\times$' + '{}'.format(k4sf2), color=beta_palette[5], linestyle='dashed', linewidth=2)

        axes.legend(loc=2, prop={'size': 8})
        fig.set_size_inches(8, 8)
        fig.savefig(os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_4', 'Figure_4_Refractoriness_By_K4_Absolute.pdf'))

    # -----------------------------------------------------------------------
    # Finally, make a figure about how EC50 changes as a function of K3 & K4
    # -----------------------------------------------------------------------
    def ec50_vs_K3_figure():
        alpha_palette = sns.color_palette("Reds", 6)
        beta_palette = sns.color_palette("Greens", 6)

        # --------------------
        # Set up Model
        # --------------------
        # Parameters found by stepwise fitting GAB mean data
        initial_parameters = {'k_a1': 4.98E-14 * 2, 'k_a2': 8.30e-13 * 2, 'k_d4': 0.006 * 3.8,
                              'kpu': 0.00095,
                              'ka2': 4.98e-13 * 2.45, 'kd4': 0.3 * 2.867,
                              'kint_a': 0.000124, 'kint_b': 0.00086,
                              'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05}
        dual_parameters = {'kint_a': 0.00052, 'kSOCSon': 6e-07, 'kint_b': 0.00052, 'krec_a1': 0.001, 'krec_a2': 0.1,
                           'krec_b1': 0.005, 'krec_b2': 0.05}
        scale_factor = 1.227

        Mixed_Model = DualMixedPopulation('Mixed_IFN_ppCompatible', 0.8, 0.2)
        Mixed_Model.model_1.set_parameters(initial_parameters)
        Mixed_Model.model_1.set_parameters(dual_parameters)
        Mixed_Model.model_1.set_parameters({'R1': 12000.0, 'R2': 1511.1})
        Mixed_Model.model_2.set_parameters(initial_parameters)
        Mixed_Model.model_2.set_parameters(dual_parameters)
        Mixed_Model.model_2.set_parameters({'R1': 6755.56, 'R2': 1511.1})

        # Prepare model for reset
        Mixed_Model.model_1.default_parameters = Mixed_Model.model_1.parameters
        Mixed_Model.model_2.default_parameters = Mixed_Model.model_2.parameters


        # Fold changes to test over
        fold_changes = list(np.logspace(-1, 4))

        def curve_builder(fold_change):
            time_list = list(linspace(2.5, 60.0, num=15))
            alpha_dose_list = list(logspace(-2, 5, num=35))
            beta_dose_list = list(logspace(-2, 4, num=35))

            fold_change_param_dict = {#'kd3': Mixed_Model.model_1.parameters['kd3'] * fold_change,
                                      'kd4': Mixed_Model.model_1.parameters['kd4'] * fold_change,
                                      #'k_d3': Mixed_Model.model_1.parameters['k_d3'] * fold_change,
                                      'k_d4': Mixed_Model.model_1.parameters['k_d4'] * fold_change}

            temp1 = Mixed_Model.mixed_dose_response(time_list, 'TotalpSTAT', 'Ia', alpha_dose_list,
                                                             parameters=dict({'Ib': 0}, **fold_change_param_dict),
                                                             sf=scale_factor)
            alpha_response = IfnData(name='custom', df=temp1)
            temp2 = Mixed_Model.mixed_dose_response(time_list, 'TotalpSTAT', 'Ib', beta_dose_list,
                                                             parameters=dict({'Ia': 0}, **fold_change_param_dict),
                                                             sf=scale_factor)
            beta_response = IfnData(name='custom', df=temp2)
            Mixed_Model.reset_global_parameters()
            alpha_ec50 = alpha_response.get_ec50s()['Cytokine'][-1][1]
            beta_ec50 = beta_response.get_ec50s()['Cytokine'][-1][1]


            # For debugging purposes only
            # Put the next line above "def curve_builder():"
            #fig = DoseresponsePlot((1,1))
            #fig.add_trajectory(alpha_response, 60.0, 'plot', '-', (0,0), 'Cytokine', label=fold_change, linewidth=np.random.random())
            #fig.add_trajectory(beta_response, 60.0, 'plot', '-', (0,1), 'Cytokine', label=fold_change, linewidth=np.random.random())
            # Put the next line after "beta_curve.append(x2)"
            #fig.show_figure()

            return alpha_ec50, beta_ec50

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


    # ----------------------------------------------------
    # Plot
    # ----------------------------------------------------
    increase_K4_figure()
    #ec50_vs_K3_figure()