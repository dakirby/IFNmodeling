try:
    from ifnclass.ifndata import IfnData
    from ifnclass.ifnmodel import IfnModel
    from ifnclass.ifnplot import TimecoursePlot
except (ImportError, ModuleNotFoundError):
    from ifndata import IfnData
    from ifnmodel import IfnModel
    from ifnplot import TimecoursePlot
from numpy import linspace, shape, square, isnan, zeros, log, nan
import numpy.random as rnd
import numpy as np
import pandas as pd
import seaborn as sns
from collections import OrderedDict
from scipy.optimize import minimize
from multiprocessing import Process, Queue, JoinableQueue, cpu_count
import pickle
import copy
import os
import matplotlib.pyplot as plt


class StepwiseFit:
    """
     Documentation - A StepwiseFit instance is characterized by the IfnModel and IfnData associated with it.

     Parameters
     ----------
     model (IfnModel): the model to fit
     data (IfnData): the data to fit
     parameters (list): the names of the model parameters to fit
     n (int): the number of test values to use for each parameter

     Attributes
     ----------
     model (IfnModel): the model to fit
     data (IfnData): the data to fit
     parameters (list): the names of the model parameters to fit
     best_fit_parameters (dict): the result of stepwise fitting the parameters
     num_test_vals (int): the number of test values to use for each parameter

     Methods
     -------
     fit(): perform a stepwise fit
     score_parameter(parameter_dict): score the model with the custom parameter dict given
     """

    # Initializer / Instance Attributes
    def __init__(self, model, data, parameters, n=10):
        self.model = model
        self.data = data
        self.parameters_to_fit = parameters
        self.best_fit_parameters = {}
        self.num_test_vals = n

    # Instance methods
    def score_parameter(self, parameter_dict):
        def score_target(scf, data, sim, data_stddev):
            diff_table = zeros(shape(sim))
            for r in range(len(data)):
                for c in range(len(data[r])):
                    if not isnan(data[r][c]):
                        if isnan(data_stddev[r][c]):
                            diff_table[r][c] = sim[r][c] * scf - data[r][c]
                        else:
                            diff_table[r][c] = (sim[r][c] * scf - data[r][c])/data_stddev[r][c]
            return np.sum(square(diff_table))

        score = 0
        parameter_copy = copy.deepcopy(self.model.parameters)
        self.model.set_parameters(parameter_dict)
        for dose_species in self.data.get_dose_species():
            simulation_times = self.data.get_times()[dose_species]
            simulation_doses = self.data.get_doses()[dose_species]
            datatable = self.data.get_responses()[dose_species]
            datatable = [[el[0] for el in r] for r in datatable]
            data_stddev = [[el[1] for el in r] for r in self.data.get_responses()[dose_species]]
            if dose_species == 'Alpha':
                spec = 'Ia'
            else:
                spec = 'Ib'
            simulation = self.model.doseresponse(simulation_times, 'TotalpSTAT', spec, simulation_doses,
                                                 parameters=self.data.conditions[dose_species],
                                                 return_type='list', dataframe_labels='Alpha')['TotalpSTAT']
            opt = minimize(score_target, [0.1], args=(datatable, simulation, data_stddev))
            sf = opt['x']
            score += opt['fun']
        self.model.set_parameters(parameter_copy)
        return score, sf

    def fit(self):
        print("Beginning stepwise fit")
        final_fit = OrderedDict({})
        final_scale_factor = 1
        number_of_parameters = len(self.parameters_to_fit)
        total_tests = (number_of_parameters+1)*number_of_parameters*self.num_test_vals/2
        print('total test: {}'.format(total_tests))
        initial_score = self.score_parameter({})[0]
        # Fit each parameter, ordered from most important to least
        for i in range(number_of_parameters):
            print("{}% of the way done".format(i*100/number_of_parameters))
            reference_score = 0
            best_scale_factor = 1
            best_parameter = []
            # Test all remaining parameters, using previously fit values
            for p, (min_test_val, max_test_val) in self.parameters_to_fit.items():
                residuals = []
                scale_factor_list = []
                # Try all test values for current parameter
                for j in linspace(min_test_val, max_test_val, self.num_test_vals):
                    test_parameters = {**{p: j}, **final_fit}  # Includes previously fit parameters
                    score, scale_factor = self.score_parameter(test_parameters)
                    residuals.append(score)
                    scale_factor_list.append(scale_factor)
                # Choose best value for current parameter
                best_parameter_value = linspace(min_test_val, max_test_val, self.num_test_vals)[
                    residuals.index(min(residuals))]
                # Decide if this is the best parameter so far in this round of 'i' loop
                if min(residuals) < reference_score or reference_score == 0:
                    reference_score = min(residuals)
                    best_scale_factor = scale_factor_list[residuals.index(min(residuals))]
                    best_parameter = [p, best_parameter_value]
            # Record the next best parameter and remove it from parameters left to test
            final_fit.update({best_parameter[0]: best_parameter[1]})
            final_scale_factor = best_scale_factor
            del self.parameters_to_fit[best_parameter[0]]
        print("Score improved from {} to {} after {} iterations".format(initial_score, reference_score, number_of_parameters))
        self.model.set_parameters(final_fit)
        self.best_fit_parameters = final_fit
        self.model.default_parameters = copy.deepcopy(self.model.parameters)
        with open('stepwisefit.p', 'wb') as f:
            pickle.dump(self.__dict__, f, 2)
        return final_fit, final_scale_factor[0]


class DualMixedPopulation:
    """
        Documentation - A DualMixedPopulation instance contains two IfnModels which describe two subpopulations.

        Attributes
        ----------
        name = name of model (for import)
        model_1 = IfnModel(name)
        model_2 = IfnModel(name)
        w1 = float in [0,1] reflecting fraction of total population described by model_1
        w2 = float in [0,1] reflecting fraction of total population described by model_2

        Methods
        -------
        mixed_dose_response(): perform a dose response as per any IfnModel, but weighted by each subpopulation
        stepwise_fit(): perform a stepwise fit of the mixed population model to given data
        """
    def __init__(self, name, pop1_weight, pop2_weight):
        self.name = name
        self.model_1 = IfnModel(name)
        self.model_2 = IfnModel(name)
        self.w1 = pop1_weight
        self.w2 = pop2_weight

    def set_global_parameters(self, param_dict):
        self.model_1.set_parameters(param_dict)
        self.model_2.set_parameters(param_dict)

    def reset_global_parameters(self):
        """
        This method is not safe for maintaining detailed balance (ie. no D.B. check)
        :return: None
        """
        self.model_1.reset_parameters()
        self.model_2.reset_parameters()

    def update_parameters(self, param_dict):
        """
        This method will act like set_global_parameters for any parameters that do not end in '_1' or '_2'.
        Parameters with names ending in '_1' or '_2' will be updated only in Model 1 or Model 2 respectively.
        :param param_dict: dictionary of parameter names and the values to use
        :return: 0
        """
        shared_parameters = {key: value for key, value in param_dict.items() if key[-2] != '_'}
        model1_parameters = {key[:-2]: value for key, value in param_dict.items() if key[-2:] == '_1'}
        model2_parameters = {key[:-2]: value for key, value in param_dict.items() if key[-2:] == '_2'}
        self.model_1.set_parameters(shared_parameters)
        self.model_1.set_parameters(model1_parameters)
        self.model_2.set_parameters(model2_parameters)
        return 0

    def get_parameters(self):
        """
        This method will retrieve all parameters from each of model_1 and model_2, and return a parameter dictionary
        of the form {pname: pvalue} where the pname will have '_1' if its value is unique to model_1 and '_2' if it is
        unique to model_2.
        :return: dict
        """
        all_parameters = {}
        for key, value in self.model_1.parameters.items():
            if self.model_1.parameters[key] != self.model_2.parameters[key]:
                all_parameters[key+'_1'] = self.model_1.parameters[key]
                all_parameters[key+'_2'] = self.model_2.parameters[key]
            else:
                all_parameters[key] = self.model_1.parameters[key]
        return all_parameters

    def mixed_dose_response(self, times, observable, dose_species, doses, parameters={}, sf=1):
        response_1 = self.model_1.doseresponse(times, observable, dose_species, doses, parameters=parameters)[observable]
        response_2 = self.model_2.doseresponse(times, observable, dose_species, doses, parameters=parameters)[observable]

        weighted_sum_response = np.add(np.multiply(response_1, self.w1), np.multiply(response_2, self.w2))
        if sf != 1:
            weighted_sum_response = [[el*sf for el in row] for row in weighted_sum_response]
        if dose_species == 'Ia':
            labelled_data = [['Alpha', doses[row], *[(el, nan) for el in weighted_sum_response[row]]] for row in range(0, len(weighted_sum_response))]
        elif dose_species == 'Ib':
            labelled_data = [['Beta', doses[row], *[(el, nan) for el in weighted_sum_response[row]]] for row in range(0, len(weighted_sum_response))]
        else:
            labelled_data = [['Cytokine', doses[row], *[(el, nan) for el in weighted_sum_response[row]]] for row in range(0, len(weighted_sum_response))]

        column_labels = ['Dose_Species', 'Dose (pM)'] + [str(el) for el in times]

        drdf = pd.DataFrame.from_records(labelled_data, columns=column_labels)
        drdf.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
        return drdf

    def __score_mixed_models__(self, shared_parameters, mixed_parameters, data):
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
        def __score_target__(scf, data, sim):
            diff_table = np.zeros((len(data), len(data[0])))
            for r in range(len(data)):
                for c in range(len(data[r])):
                    if not np.isnan(data[r][c][1]):
                        diff_table[r][c] = (sim[r][c][0] * scf - data[r][c][0]) / data[r][c][1]
                    else:
                        diff_table[r][c] = (sim[r][c][0] * scf - data[r][c][0])
            return np.sum(np.square(diff_table))

        opt = minimize(__score_target__, [0.1], args=(data.data_set.values, total_response.values))
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
        initial_score, _ = self.__score_mixed_models__({}, [{}, {}], data)
        for i in range(number_of_parameters):
            print("{}% of the way done".format(i * 100 / number_of_parameters))
            reference_score = 0
            best_scale_factor = 1
            best_parameter = []
            # Test all remaining parameters, using previously fit values
            for p, (min_test_val, max_test_val) in parameters_to_test.items():
                residuals = []
                scale_factor_list = []
                # Try all test values for current parameter
                for j in np.linspace(min_test_val, max_test_val, ntest_per_param):
                    test_parameters = {p: j, **final_fit}  # Includes previously fit parameters
                    base_parameters, subpopulation_parameters = separate_parameters(test_parameters, mixed_p)
                    score, scale_factor = self.__score_mixed_models__(base_parameters, subpopulation_parameters, data)
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
            del parameters_to_test[best_parameter[0]]
        print("Score improved from {} to {} after {} iterations".format(initial_score,
                                                                        reference_score, number_of_parameters))
        final_shared_parameters, final_mixed_parameters = separate_parameters(final_fit, mixed_p)
        return final_shared_parameters, final_mixed_parameters, final_scale_factor


class Prior:
    """
    Documentation - A Prior instance fully characterizes a prior distribution, and can generate random values from the
                    prior distribution or caclulate the probability density for a given value being drawn from the prior

    Parameters/Attributes
    ----------
    type_of_distribution (str): can be 'lognormal', 'normal', or 'uniform'
    mean (float): the mean of the prior distrubiton; default is 1 (does not need to be specified for uniform priors)
                For lognormal priors this is what is subtracted from the logarithm of the value in
                the exponent of the Gaussian (ie. this is the log_mean).
    sigma (float): the standard deviation of the prior distribution; default is 1 (does not need to be specified for
                    uniform priors). For lognormal priors this is what the logarithm of the value is divided by in
                    the exponent of the Gaussian.
    lower_bound, upper_bound (float): the upper and lower bounds for truncating the prior distrubution
                                      these are only used for uniform type priors, in which case defaults are 0 and 1
    Methods
    -------
    draw_value(): returns a float drawn from the distrubtion
    get_log_probability(test_value): returns the negative log probability for the test_value coming from the prior
                                    If prior is lognormal, the test value is assumed to NOT be in log form
                                    If prior is uniform and test value is outside bounds, returns 1E15
    """

    # Initializer / Instance Attributes
    def __init__(self, type_of_distribution, mean=1., sigma=1., lower_bound=0., upper_bound=1.):
        self.type_of_distribution = type_of_distribution
        self.mean = mean
        self.sigma = sigma
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    # Instance methods
    def draw_value(self):
        if self.type_of_distribution == 'lognormal':
            return rnd.lognormal(self.mean, self.sigma)
        elif self.type_of_distribution == 'normal':
            return rnd.normal(self.mean, self.sigma)
        else:
            return rnd.uniform(self.lower_bound, self.upper_bound)

    def get_log_probability(self, test_value):
        if self.type_of_distribution == 'lognormal':
            return ((log(test_value) - self.mean) / self.sigma) ** 2
        elif self.type_of_distribution == 'normal':
            return ((test_value - self.mean) / self.sigma) ** 2
        else:  # uniform
            if test_value < self.lower_bound or test_value > self.upper_bound:
                return 1E15
            else:
                return 0


def __unpackMCMC__(ID, jobs, result, countQ, build_model, build_data, model_parameters, temperature):
    """
    This function is used by the MCMC class but must be defined externally. 
    Should not be used by any other program.
    """
    model_name, fit_parameters, priors, jump_distributions = build_model
    data_set, data_name, build_conditions = build_data
    model = IfnModel(model_name)
    model.parameters = model_parameters
    if data_name is None:
        data = IfnData(name='custom', df=data_set, conditions=build_conditions)
    else:
        data = IfnData(data_name)
    processMCMC = MCMC(model, data, fit_parameters, priors, jump_distributions)
    processMCMC.temperature = temperature
    processMCMC.__run_chain__(ID, jobs, result, countQ)


class MCMC:
    """
     Documentation - An MCMC instance is characterized by the IfnModel and IfnData associated with it.
                     This class offers a way to fit parameters for an IfnModel using MCMC.

     Parameters
     ----------
     model (IfnModel): the model to fit
     data (IfnData): the data to fit
     parameters (list): the names of the model parameters to fit
     priors (dict): keys corresponding to parameters, values are Prior objects
     jump_distributions (dict): keys corresponding to parameters, values are the sigmas for jump distributions
     * Optional parameters *
     beta (float or int): scales the mean square error component of the cost function (in the case that MSE 
                          is not of the same order as prior cost) (default = 1)
     temperature (float or int): scales the acceptance rate, permitting worse fits (default = 1)
     burn_in (float): the fraction of the beginning of MCMC sampling to discard (default = 0.2)
     down_sample (int): the frequency with which to keep successful samples (ie. down_sample = 3 means keep
                        every third sample) (default = 5)
     num_samples (int): the number of samples to keep after burn in and down sampling (default = 100)
     num_chains (int): the number of MCMC chains to run. Must choose greater than 1 to use G.R. statistics
                       (default = 1)
     filename (str): the path to the directory where results of the fit can be stored 
                     (default = './mcmc_results/initial_model.p')

     Attributes
     ----------
     model (IfnModel): the model to fit
     data (IfnData): the data to fit
     parameters (list): the names of the model parameters to fit
     parameters_to_fit (list): the names of the model parameters to fit
     priors (dict): keys corresponding to parameters, values are Prior objects
     jump_distributions (dict): keys corresponding to parameters, values are the sigmas for jump distributions

     Methods
     -------
     score_current_model(): returns the combined mean-squared-error (divided by self.beta) and prior penalty
                            - ie. returns the log posterior probability assuming Gaussian noise and given priors,
                                    scaled by beta
     """

    # Initializer / Instance Attributes
    def __init__(self, model: IfnModel, data: IfnData, parameters: list, priors: dict,
                 jump_distributions: dict):
        self.model = model
        self.data = data
        self.parameters_to_fit = parameters
        self.priors = priors
        self.jump_distributions = jump_distributions
        # Default simulation parameters. Can be set by user
        self.beta = 1
        self.temperature = 1
        self.burn_in = 0.2
        self.down_sample = 5
        self.num_samples = 100
        self.num_chains = 1
        # The following attributes will store the results of the MCMC fit
        self.filename = os.path.join('mcmc_results','initial_model.p')
        self.best_fit_parameters = {}
        self.best_fit_scale_factor = 1
        self.best_fit_score = 1E15
        self.average_acceptance = 1
        self.parameter_history = []
        self.scale_factor_history = []
        self.score_history = []
        self.thinned_parameter_samples = []
        self.thinned_parameter_scale_factors = []
        self.thinned_parameter_scores = []
        self.chain_lengths = []

    # Private methods
    # ---------------
    def __MSE_of_parametric_model__(self, parameter_dict={}, sf_flag=0):
        def score_target(sf, data, sim):
            diff_table = zeros(shape(sim))
            for r in range(len(data)):
                for c in range(len(data[r])):
                    if not isnan(data[r][c]):
                        diff_table[r][c] = sim[r][c] * sf - data[r][c]
            return np.sum(square(diff_table))

        # ------------------------------
        # Initialize variables
        # ------------------------------
        times = self.data.get_times(species='Alpha')
        alpha_doses = self.data.get_doses(species='Alpha')
        beta_doses = self.data.get_doses(species='Beta')

        old_parameters = {key: self.model.parameters[key] for key in parameter_dict.keys()} # To ensure changes to model aren't permanent

        # Set parameters for each population
        self.model.set_parameters(parameter_dict)

        # -------------------------
        # Make predictions
        # -------------------------
        alpha_response = self.model.doseresponse(times, 'TotalpSTAT', 'Ia', alpha_doses, parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
        beta_response = self.model.doseresponse(times, 'TotalpSTAT', 'Ib', beta_doses, parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')
        total_response = pd.concat([alpha_response, beta_response])

        # -------------------------
        # Score predictions vs data
        # -------------------------
        def __score_target__(scf, data, sim):
            diff_table = np.zeros((len(data), len(data[0])))
            for r in range(len(data)):
                for c in range(len(data[r])):
                    if not np.isnan(data[r][c][1]):
                        diff_table[r][c] = (sim[r][c][0] * scf - data[r][c][0]) / data[r][c][1]
                    else:
                        diff_table[r][c] = (sim[r][c][0] * scf - data[r][c][0])
            return np.sum(np.square(np.nan_to_num(diff_table)))

        opt = minimize(__score_target__, [0.1], args=(self.data.data_set.values, total_response.values))
        sf = opt['x'][0]
        score = opt['fun']

        self.model.set_parameters(old_parameters) # Reset parameters

        if sf_flag == 1:
            return score, opt
        else:
            return score

    def __prior_penalty__(self, parameter_dict={}):
        if parameter_dict == {}: 
            parameter_dict = self.model.parameters
        penalty = 0
        for parameter, value in parameter_dict.items():
            if parameter in self.parameters_to_fit:
                penalty += self.priors[parameter].get_log_probability(value)
        return penalty

    def __parameter_jump__(self):
        new_parameters = {}
        for parameter in self.parameters_to_fit:
            new_parameters.update({parameter: self.priors[parameter].draw_value()})
        return new_parameters

    def __check_input__(self):
        if self.burn_in > 1:
            raise ValueError('Burn rate should be in the range [0,1)')
        if self.down_sample > self.num_samples:
            raise ValueError('Cannot thin more than there are samples')
        # lenPost = int(np.floor([chains * (n + 1) * (1 - burn_rate) / down_sample])[0])
        # print("It's estimated this simulation will produce {} posterior samples.".format(lenPost))
        return True

    def __autochoose_beta__(self):
        MSE = 0
        for i in range(50):
            temp = self.__MSE_of_parametric_model__()
            if not temp == np.inf:
                MSE = temp
                break
        priorCost = self.__prior_penalty__()
        order_of_magnitude = 10 ** np.floor(np.log10(MSE/priorCost))
        self.beta = order_of_magnitude

    def __generate_parameters_from_priors__(self, n):
        pList = []
        for i in range(n):
            pList.append(self.__parameter_jump__())
        return pList

    def __score_and_sf_for_current_model__(self):
        MSE, sf = self.__MSE_of_parametric_model__(sf_flag=1)
        return MSE / self.beta + self.__prior_penalty__(), sf

    def __run_chain__(self, ID, jobs, result, countQ):
        print("Chain {} started".format(ID))
        while True:
            initial_parameters = jobs.get()
            if initial_parameters is None:
                break
            initial_parameters = initial_parameters[0]
            self.model.set_parameters(initial_parameters)
            current_score, _ = self.__score_and_sf_for_current_model__()
            current_parameters = initial_parameters

            progress_bar = (self.num_samples / (1 - self.burn_in) * self.down_sample) / 10

            acceptance = 0
            attempts = 0
            while acceptance < self.num_samples / (1 - self.burn_in) * self.down_sample:
                # Identify failed chains
                if attempts > 2000 and acceptance == 0:
                    print("Chain {} failed to start".format(ID))
                    break
                attempts += 1
                # Monitor acceptance rate
                if acceptance > progress_bar:
                    print("{:.1f}% done".format(acceptance/self.num_samples * 100))
                    print("Chain {} acceptance rate = {:.1f}%".format(ID, acceptance/attempts*100))
                    # Record progress to text file
                    with open(os.path.join('mcmc_results','progress.txt'),'a') as f:
                        f.write("Chain {} is {:.1f}% done, currently averaging {:.1f}% acceptance.\n".format(ID, acceptance/self.num_samples*100, acceptance/attempts*100))
                    # Save state at checkpoint
                    with open(os.path.join('mcmc_results','chain_results','{}chain.p'.format(str(ID))), 'wb') as f:
                        pickle.dump(self.__dict__, f, 2)
                    progress_bar += (self.num_samples / (1 - self.burn_in) * self.down_sample) / 10

                new_parameters = self.__parameter_jump__()
                self.model.set_parameters(new_parameters)

                new_score, new_scale_factor = self.__score_and_sf_for_current_model__()

                """ Note:
                Asymmetry factor for lognormal jumping distributions with x* proposed and x the current value: 
                    C = PDF(LogNormal(log(x*),sigma), x)/PDF(LogNormal(log(x),sigma), x*)
                    C = x*/x
                """
                asymmetry_factor = 1
                for key, new_value in new_parameters.items():
                    try:
                        if self.priors[key].type_of_distribution == 'lognormal':
                            asymmetry_factor *= new_value/current_parameters[key]
                    except KeyError:
                        pass
                alpha = asymmetry_factor * np.exp(-(new_score-current_score)/self.temperature)
                if new_score < current_score or np.random.rand() < alpha:
                    # Search for MAP:
                    if new_score < self.best_fit_score:
                        self.best_fit_parameters = new_parameters
                        self.best_fit_scale_factor = new_scale_factor
                    # Add to chain
                    new_parameters = {key: new_parameters[key] for key in self.parameters_to_fit}
                    self.parameter_history.append(new_parameters)
                    self.scale_factor_history.append(new_scale_factor)
                    self.score_history.append(new_score)
                    current_score = new_score
                    current_parameters = new_parameters
                    acceptance += 1
                else:
                    self.model.set_parameters({key: current_parameters[key] for key in self.parameters_to_fit})

            # Save final state
            with open(os.path.join('mcmc_results','chain_results','{}chain.p'.format(str(ID))), 'wb') as f:
                pickle.dump(self.__dict__, f, 2)

            result.put([self.parameter_history, self.scale_factor_history, self.score_history])
            countQ.put([ID, acceptance / attempts * 100])
        print("Chain {} exiting".format(ID))

        with open(os.path.join('mcmc_results','progress.txt'), 'a') as f:
            f.write("Chain {} is exiting".format(ID))

    # Public methods
    # --------------
    def save(self, alt_filename=''):
        if alt_filename=='':
            with open(self.filename, 'wb') as f:
                pickle.dump(self.__dict__, f, 2)
        else:
            with open(alt_filename, 'wb') as f:
                pickle.dump(self.__dict__, f, 2)

    def load(self):
        with open(self.filename, 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)

    def score_current_model(self):
        return self.__MSE_of_parametric_model__(self.model.parameters) / self.beta + self.__prior_penalty__(
            self.model.parameters)

    def gelman_rubin_convergence(self):
        stats_list = []
        indices = [int(np.sum(self.chain_lengths[0:i])) for i in range(len(self.chain_lengths)+1)]
        if indices == [0]:
            indices = [0, len(self.parameter_history)]
        for variable in self.parameters_to_fit:
            chain_mean = [np.mean([j[variable] for j in self.parameter_history[indices[i]:indices[i+1]]]) for i in range(self.num_chains)]
            overall_mean = np.mean(chain_mean)
            B = np.sum([(indices[i+1] - indices[i]) * (chain_mean - overall_mean) ** 2 for i in range(self.num_chains)]) / (self.num_chains - 1)
            W = np.mean([np.var([j[variable] for j in self.parameter_history[indices[i]:indices[i+1]]]) for i in range(self.num_chains)])
            N = np.mean([indices[i+1]-indices[i] for i in range(self.num_chains)])
            Var = (1 - 1 / N) * W + (1 + 1 / self.num_chains) * B / N
            Rhat = np.sqrt(Var / W)
            stats_list.append([variable, Rhat])
        df = pd.DataFrame.from_records(stats_list, columns=['variable', 'GR Statistic'])
        title = os.path.join(os.getcwd(), "mcmc_results", "gelman-rubin_statistics.csv")
        df.to_csv(title)
        return stats_list

    def describe_parameter_statistics(self, title=''):
        statistics_record = []
        plist = [(key, [self.parameter_history[i][key] for i in range(len(self.parameter_history))]) for key in self.parameters_to_fit]
        for item in plist:
            statistics_record.append([item[0], np.percentile(item[1], 2.5), np.percentile(item[1], 25),
                                      np.mean(item[1]), np.percentile(item[1], 75), np.percentile(item[1], 95)])
        labels = ['parameter', '2.5%', '25%', '50%', '75%', '95%']
        df = pd.DataFrame.from_records(statistics_record, columns=labels)
        if title == '':
            title = os.path.join(os.getcwd(), "mcmc_results", "parameter_statistics.csv")
            df.to_csv(title)
        else:
            df.to_csv('mcmc_records/'+title+'.csv')


    def plot_parameter_distributions(self, save=False, title=''):
        if self.num_chains == 1:
            k = len(self.parameters_to_fit)  # total number subplots
            n = 2  # number of chart columns
            m = k // n + k % n  # number of chart rows
            fig, axes = plt.subplots(m, n, figsize=(n * 5, m * 3))
            if k % 2 == 1:  # avoids extra empty subplot
                axes[-1][n - 1].set_axis_off()
            # format data for plotting
            plist = [(key, [self.parameter_history[i][key] for i in range(len(self.parameter_history))]) for key in self.parameters_to_fit]
            # add plots
            for i, (name, col) in enumerate(plist):
                r, c = i // n, i % n
                ax = axes[r, c]  # get axis object
                # determine whether or not to plot on log axis
                if abs(int(np.log10(np.max(col))) - int(np.log10(np.min(col)))) >= 4:
                    ax.set(xscale='log', yscale='linear')
                # Plot histogram with kde for chain
                sns.distplot(col, ax=ax, hist=True, kde=True,
                             color='darkblue',
                             hist_kws={'edgecolor': 'black'},
                             kde_kws={'linewidth': 4})
                ax.set_title(name)
            fig.tight_layout()
            if save == True:
                if title == '':
                    plt.savefig(os.path.join('mcmc_results','parameter_distributions.pdf'))
                else:
                    plt.savefig(os.path.join('mcmc_results', title + '.pdf'))
            return (fig, axes)

    def fit(self, num_accepted_steps: int, num_chains: int, burn_rate: float, down_sample_frequency: int, beta: float,
            cpu=None, initialise=True):
        # Check input parameters
        self.__check_input__()
        print("Performing MCMC Analysis")
        # Create results directory
        if not os.path.isdir(os.path.join(os.getcwd(), "mcmc_results")):
            os.mkdir(os.path.join(os.getcwd(), "mcmc_results"))
        if not os.path.isdir(os.path.join(os.getcwd(), "mcmc_results","chain_results")):
            os.mkdir(os.path.join(os.getcwd(), "mcmc_results","chain_results"))
        # Define simulation parameters for instance
        self.num_samples = num_accepted_steps
        self.num_chains = num_chains
        self.burn_in = burn_rate
        self.down_sample = down_sample_frequency
        # Selecting optimal beta (scale factor for MSE)
        if beta == -1:
            self.__autochoose_beta__()
        # Overdisperse chains
        if initialise == True:
            initial_parameters = self.__generate_parameters_from_priors__(self.num_chains)
        else:
            initial_parameters = [{key: self.model.parameters[key] for key in self.parameters_to_fit} for _ in range(self.num_chains)]
        # Sample using MCMC
        print("Sampling from posterior distribution")
        # Set up simulation processes
        if self.num_chains >= cpu_count():
            number_of_processes = cpu_count() - 1
        else:
            number_of_processes = self.num_chains
        if cpu is not None:
            number_of_processes = cpu  # Manual override of core number selection
        print("Using {} processes".format(number_of_processes))
        with open(os.path.join('mcmc_results','progress.txt'), 'w') as f:  # clear previous progress report
            f.write('')
        jobs = Queue()  # put jobs on queue
        result = JoinableQueue()
        countQ = JoinableQueue()
        # Start up chains
        # Prepare instance for multiprocessing by pickling
        build_model = [self.model.name, self.parameters_to_fit, self.priors, self.jump_distributions]
        build_data = [self.data.data_set, self.data.name, self.data.conditions]
        # Run chains
        if number_of_processes == 1:
            jobs.put([initial_parameters[0]])
            jobs.put(None)
            self.__run_chain__(0, jobs, result, countQ)
        else:
            # Put jobs in queue
            for m in range(self.num_chains):
                jobs.put([initial_parameters[m]])
            # Add signals for each process that there are no more jobs
            for w in range(number_of_processes):
                jobs.put(None)
            [Process(target=__unpackMCMC__, args=(i, jobs, result, countQ, build_model, build_data,
                                                  self.model.parameters, self.temperature)).start()
             for i in range(number_of_processes)]
        # Pull in the results from each thread
        pool_results = []
        chain_attempts = []
        for m in range(self.num_chains):
            print("Getting results")
            r = result.get()
            pool_results.append(r)
            result.task_done()
            a = countQ.get()
            chain_attempts.append(a)
        # close all extra threads
        jobs.close()
        result.join()
        result.close()
        countQ.close()

        # Perform data analysis
        # Record average acceptance across all chains
        self.average_acceptance = np.mean([el[1] for el in chain_attempts])
        print("Average acceptance rate was {:.1f}%".format(self.average_acceptance))
        # Consolidate results into attributes
        if self.num_chains != 1:
            for chain in pool_results:
                self.chain_lengths.append(len(chain))
                self.parameter_history += chain[0]
                self.scale_factor_history += chain[1]
                self.score_history += chain[2]
        # Perform burn-in and down sampling
        for chain in pool_results:
            try:
                sample_pattern = range(int(burn_rate*len(chain)), len(chain), down_sample_frequency)
                self.thinned_parameter_samples += [chain[0][i] for i in sample_pattern]
                self.thinned_parameter_scale_factors += [chain[1][i] for i in sample_pattern]
                self.thinned_parameter_scores += [chain[2][i] for i in sample_pattern]
            except IndexError:
                pass
        # Write summary file
        with open(os.path.join("mcmc_results","simulation_summary.txt"), 'w') as f:
            f.write('Temperature used was {}\n'.format(self.beta))
            f.write('Number of chains = {}\n'.format(self.num_chains))
            f.write("Average acceptance rate was {:.1f}%\n".format(self.average_acceptance))
            f.write("Initial conditions were\n")
            for i in initial_parameters:
                f.write(str(i))
                f.write("\n")
            f.write("Individual chain acceptance rates were:\n")
            for i in chain_attempts:
                f.write("Chain {}: {:.1f}%".format(i[0], i[1]))

        # Save object
        self.save(alt_filename=os.path.join("mcmc_results","mcmc_fit.p"))
        pickle.dump(self.parameter_history, open(os.path.join('mcmc_results','mcmc_parameter_history.p'),'wb'), 2)
        pickle.dump(self.scale_factor_history, open(os.path.join('mcmc_results','mcmc_scale_factor_history.p'),'wb'), 2)
        pickle.dump(self.score_history, open(os.path.join('mcmc_results','mcmc_score_history.p'),'wb'), 2)

if __name__ == '__main__':
    testData = IfnData("MacParland_Extended")
    testModel = IfnModel('Mixed_IFN_ppCompatible')

    """
    stepfit = StepwiseFit(testModel, testData, {'kpa': (1E-7,1E-5), 'kSOCSon': (1E-7,1E-5), 'R1': (200,12000),
                                                'R2': (200,12000), 'kd4': (0.006,1)})
    stepfit.fit()

    sw_tc = stepfit.model.timecourse(list(linspace(0, 30)), 'TotalpSTAT', return_type='dataframe',
                              dataframe_labels=['Alpha', 1E-9])

    testplot = TimecoursePlot((1, 1))
    testplot.add_trajectory(sw_tc, 'plot', 'r', (0, 0))
    testplot.show_figure()
    """
    jump_dists = {'kpa': 1, 'kSOCSon': 1, 'kd4': 0.5, 'k_d4': 0.5, 'R1': 0.5, 'R2': 0.5}
    mixed_IFN_priors = {'kpa': Prior('lognormal', mean=1E-6, sigma=4),
                        'kSOCSon': Prior('lognormal', mean=1E-6, sigma=4),
                        'R1': Prior('uniform', lower_bound=200, upper_bound=12000),
                        'R2': Prior('uniform', lower_bound=200, upper_bound=12000),
                        'kd4': Prior('lognormal', mean=0.3, sigma=1.8),
                        'k_d4': Prior('lognormal', mean=0.006, sigma=1.8)}
    mcmcFit = MCMC(testModel, testData, ['kpa', 'kSOCSon', 'R1', 'R2', 'kd4', 'k_d4'], mixed_IFN_priors, jump_dists)
    mcmcFit.fit(10, 1, 0, 1, 1E8)

    # test_parameters = {'kpa': 1.13e-06, 'kSOCSon': 1.4e-06, 'R1': 1377, 'R2': 574, 'kd4': 0.257, 'k_d4': 0.0085}
