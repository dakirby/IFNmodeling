from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import numpy as np
import seaborn as sns
import pandas as pd
from ifnclass.ifnfit import StepwiseFit
from scipy.optimize import minimize
from collections import OrderedDict
import os
import json


class DualMixedPopulation:
    def __init__(self, name, pop1_weight, pop2_weight):
        self.name = name
        self.model_1 = IfnModel(name)
        self.model_2 = IfnModel(name)
        self.w1 = pop1_weight
        self.w2 = pop2_weight

    def mixed_dose_response(self, times, observable, dose_species, doses, parameters={}, sf=1):
        response_1 = self.model_1.doseresponse(times, observable, dose_species, doses, parameters=parameters)[observable]
        response_2 = self.model_1.doseresponse(times, observable, dose_species, doses, parameters=parameters)[observable]

        weighted_sum_response = np.add(np.multiply(response_1, self.w1), np.multiply(response_2, self.w2))
        if sf != 1:
            weighted_sum_response = [[el*sf for el in row] for row in weighted_sum_response]
        if parameters == {'Ib': 0}:
            labelled_data = [['Alpha', doses[row], *[(el, nan) for el in weighted_sum_response[row]]] for row in range(0, len(weighted_sum_response))]
        elif parameters == {'Ia': 0}:
            labelled_data = [['Beta', doses[row], *[(el, nan) for el in weighted_sum_response[row]]] for row in range(0, len(weighted_sum_response))]
        else:
            labelled_data = [['Cytokine', doses[row], *[(el, nan) for el in weighted_sum_response[row]]] for row in range(0, len(weighted_sum_response))]

        column_labels = ['Dose_Species', 'Dose (pM)'] + [str(el) for el in times]

        drdf = pd.DataFrame.from_records(labelled_data, columns=column_labels)
        drdf.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
        return drdf

    def score_mixed_models(self, shared_parameters, mixed_parameters, data):
        # ------------------------------
        # Initialize variables
        # ------------------------------
        times = data.get_times(species='Alpha')
        alpha_doses = data.get_doses(species='Alpha')
        beta_doses = data.get_doses(species='Beta')

        model_1_old_parameters = self.model_1.parameters
        model_2_old_parameters = self.model_2.parameters

        # Set parameters for each population
        self.model_1.set_parameters(shared_parameters)
        self.model_1.set_parameters(mixed_parameters[0])

        self.model_2.set_parameters(shared_parameters)
        self.model_1.set_parameters(mixed_parameters[1])

        # -------------------------
        # Make predictions
        # -------------------------
        alpha_response = self.mixed_dose_response(times, 'TotalpSTAT', 'Ia', alpha_doses, parameters={'Ib': 0})
        beta_response = self.mixed_dose_response(times, 'TotalpSTAT', 'Ib', beta_doses, parameters={'Ia': 0})
        total_response = pd.concat([alpha_response, beta_response])

        # -------------------------
        # Score predictions vs data
        # -------------------------
        def score_target(scf, data, sim):
            diff_table = np.zeros((len(data), len(data[0])))
            for r in range(len(data)):
                for c in range(len(data[r])):
                    if not np.isnan(data[r][c][1]):
                        diff_table[r][c] = (sim[r][c][0] * scf - data[r][c][0]) / data[r][c][1]
                    else:
                        diff_table[r][c] = (sim[r][c][0] * scf - data[r][c][0])
            return np.sum(np.square(diff_table))

        opt = minimize(score_target, [0.1], args=(mean_data.data_set.values, total_response.values))
        sf = opt['x'][0]
        score = opt['fun']

        self.model_1.set_parameters(model_1_old_parameters)
        self.model_2.set_parameters(model_2_old_parameters)
        return score, sf

    def stepwise_fit(self, data, parameters_to_test, ntest_per_param, mixed_p):
        number_of_parameters = len(parameters_to_test.keys())
        final_fit = OrderedDict({})

        # Local scope function
        def separate_parameters(p_to_test, mixed_p_list):
            shared_variables = {}
            mixed_variables = [{}, {}]
            for key, value in p_to_test.items():
                if key[-3:] == '__1':
                    if key[0:-3] in mixed_p_list:
                        mixed_variables[0].update({key[0:-3]: value})
                elif key[-3:] == '__2':
                    if key[0:-3] in mixed_p_list:
                        mixed_variables[1].update({key[0:-3]: value})
                else:
                    shared_variables.update({key: value})
            return shared_variables, mixed_variables,

        # Fit each parameter, ordered from most important to least
        initial_score, _ = self.score_mixed_models({}, [{}, {}], data)
        for i in range(number_of_parameters):
            print("{}% of the way done".format(i * 100 / number_of_parameters))
            reference_score = 0
            best_scale_factor = 1
            best_parameter = []
            # Test all remaining parameters, using previously fit values
            for p, (min_test_val, max_test_val) in parameters_to_fit.items():
                residuals = []
                scale_factor_list = []
                # Try all test values for current parameter
                for j in np.linspace(min_test_val, max_test_val, ntest_per_param):
                    test_parameters = {p: j, **final_fit}  # Includes previously fit parameters
                    base_parameters, subpopulation_parameters = separate_parameters(test_parameters, mixed_p)
                    score, scale_factor = self.score_mixed_models(base_parameters, subpopulation_parameters, data)
                    residuals.append(score)
                    scale_factor_list.append(scale_factor)
                # Choose best value for current parameter
                best_parameter_value = np.linspace(min_test_val, max_test_val,
                                                   ntest_per_param)[residuals.index(min(residuals))]
                # Decide if this is the best parameter so far in this round of 'i' loop
                if min(residuals) < reference_score or reference_score == 0:
                    reference_score = min(residuals)
                    best_scale_factor = scale_factor_list[residuals.index(min(residuals))]
                    best_parameter = [p, best_parameter_value]
            # Record the next best parameter and remove it from parameters left to test
            final_fit.update({best_parameter[0]: best_parameter[1]})
            final_scale_factor = best_scale_factor
            del parameters_to_fit[best_parameter[0]]
        print("Score improved from {} to {} after {} iterations".format(initial_score,
                                                                        reference_score, number_of_parameters))
        final_shared_parameters, final_mixed_parameters = separate_parameters(final_fit, mixed_p)
        return final_shared_parameters, final_mixed_parameters, final_scale_factor


if __name__ == '__main__':
    # ------------------------------
    # Align all data
    # ------------------------------
    newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
    newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
    newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
    newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")

    alignment = DataAlignment()
    alignment.add_data([newdata_4, newdata_3, newdata_2, newdata_1])
    alignment.align()
    alignment.get_scaled_data()
    mean_data = alignment.summarize_data()

    # -------------------------
    # Set up model scoring f'n
    # -------------------------
    # Parameters arbitrarily taken from best fit to GAB 20190119 data
    base_parameters = {'k_a1': 4.98E-14 * 2, 'k_a2': 1.328e-12, 'k_d3': 2.4e-06, 'k_d4': 0.228,
                       'kSOCSon': 8e-07, 'kpu': 0.0011,
                       'ka1': 3.3e-15, 'ka2': 1.22e-12, 'kd4': 0.86,
                       'kd3': 1.74e-05,
                       'kint_a': 0.000124, 'kint_b': 0.00086,
                       'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05}

    mixed_parameters = [{'R2': 570, 'R1': 180}, {'R2': 5700, 'R1': 8000} ]

    het_model = DualMixedPopulation('Mixed_IFN_ppCompatible', 0.8, 0.2)
    het_model.model_1.set_parameters(base_parameters)
    het_model.model_1.set_parameters(mixed_parameters[0])
    het_model.model_2.set_parameters(base_parameters)
    het_model.model_2.set_parameters(mixed_parameters[1])

    # ----------------------
    # Perform stepwise fit
    # ----------------------
    parameters_to_fit = {'R1__1': (200, 12000), 'R1__2': (200, 12000),
                         'R2__1': (200, 12000), 'R2__2': (200, 12000),
                         'kSOCSon': (1E-07, 1E-06),
                         'kint_a': (0.0001, 0.002), 'kint_b': (0.002, 0.0001),
                         'krec_a1': (1e-03, 1e-02), 'krec_a2': (0.1, 0.01),
                         'krec_b1': (0.005, 0.0005), 'krec_b2': (0.05, 0.005)}
    mixed_parameters = ['R1', 'R2']
    best_shared_params, best_mixed_params, best_sf = het_model.stepwise_fit(mean_data, parameters_to_fit, 10,
                                                                            mixed_parameters)

    # -------------------------
    # Simulate the best fit
    # -------------------------
    times = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
    alpha_doses = [10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses = [0.2, 6, 20, 60, 200, 600, 2000]
    best_fit_model = het_model
    best_fit_model.model_1.set_parameters(best_shared_params)
    best_fit_model.model_2.set_parameters(best_shared_params)
    best_fit_model.model_1.set_parameters(best_mixed_params[0])
    best_fit_model.model_2.set_parameters(best_mixed_params[1])

    alpha_response = best_fit_model.mixed_dose_response(times, 'TotalpSTAT', 'Ia',
                                                        np.logspace(1, 5), parameters={'Ib': 0}, sf=best_sf)
    beta_response = best_fit_model.mixed_dose_response(times, 'TotalpSTAT', 'Ib',
                                                       np.logspace(np.log10(0.2), np.log10(2000)),
                                                       parameters={'Ia': 0}, sf=best_sf)
    total_response_df = pd.concat([alpha_response, beta_response])
    total_response = IfnData('custom', df=total_response_df)

    # -------------------------
    # Plot the results
    # -------------------------
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    new_fit = DoseresponsePlot((1, 2))
    alpha_mask = [0.0, 7.5]
    beta_mask = [0.0, 7.5]
    # Add fits
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(total_response, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='Alpha ' + str(t))
        if t not in beta_mask:
            new_fit.add_trajectory(total_response, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label='Beta ' + str(t))


    # Add data
    times = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(mean_data, t, 'errorbar', 'o--', (0, 0), 'Alpha', label='', color=alpha_palette[idx])
        if t not in beta_mask:
            new_fit.add_trajectory(mean_data, t, 'errorbar', 'o--', (0, 1), 'Beta', label='', color=beta_palette[idx])

    new_fit.axes[0].set_title(r"IFN$\alpha$")
    new_fit.axes[1].set_title(r"IFN$\beta$")
    new_fit.show_figure(save_flag=False)
    print(str(best_shared_params))
    print(str(best_mixed_params[0]))
    print(str(best_mixed_params[1]))
    with open(os.path.join('results','GAB_NewData','stepwise_params.txt'),'w') as f:
        f.write(json.dumps(best_shared_params))
        f.write(json.dumps(best_mixed_params[0]))
        f.write(json.dumps(best_mixed_params[1]))

    """
    # ----------------------
    # Time Course Comparison
    # Simulate time courses
    # SUBPOPULATION 1
    Mixed_Model.set_parameters({'R2': 1000, 'R1': 1000})
    alpha_time_courses = []
    for d in alpha_doses:
        alpha_time_courses.append(Mixed_Model.timecourse(list(linspace(0, 60, 30)), 'TotalpSTAT',
                                                         {'Ia': d * 6.022E23 * 1E-5 * 1E-12, 'Ib': 0},
                                                         return_type='dataframe', dataframe_labels=['Alpha', d]))
    beta_time_courses = []
    for d in beta_doses:
        beta_time_courses.append(Mixed_Model.timecourse(list(linspace(0, 60, 30)), 'TotalpSTAT',
                                                        {'Ib': d * 6.022E23 * 1E-5 * 1E-12, 'Ia': 0},
                                                        return_type='dataframe', dataframe_labels=['Beta', d]))

    for i in range(30):
        for j in range(len(alpha_doses)):
            alpha_time_courses[j].loc['Alpha'].iloc[:, i] = alpha_time_courses[j].loc['Alpha'].iloc[:, i].apply(
                scale_data)
        for j in range(len(beta_doses)):
            beta_time_courses[j].loc['Beta'].iloc[:, i] = beta_time_courses[j].loc['Beta'].iloc[:, i].apply(scale_data)
    # Turn into IfnData objects
    alpha_IfnData_objects = []
    beta_IfnData_objects = []
    for j in range(len(alpha_doses)):
        alpha_IfnData_objects.append(IfnData('custom', df=alpha_time_courses[j], conditions={'Alpha': {'Ib': 0}}))
    for j in range(len(beta_doses)):
        beta_IfnData_objects.append(IfnData('custom', df=beta_time_courses[j], conditions={'Beta': {'Ia': 0}}))
    # Generate plot
    new_fit = TimecoursePlot((1, 2))
    alpha_mask = [0]
    beta_mask = [0]

    # Add fits
    for j, dose in enumerate(alpha_doses):
        if dose not in alpha_mask:
            new_fit.add_trajectory(alpha_IfnData_objects[j], 'plot', alpha_palette[j], (0, 0),
                                   label='Alpha ' + str(dose))
    for j, dose in enumerate(beta_doses):
        if dose not in beta_mask:
            new_fit.add_trajectory(beta_IfnData_objects[j], 'plot', beta_palette[j], (0, 1),
                                   label='Beta ' + str(dose))

"""
