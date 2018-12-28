from ifndata import IfnData
from ifnmodel import IfnModel
from ifnplot import TimecoursePlot
from numpy import linspace, shape, square, isnan, zeros, log
import numpy.random as rnd
import numpy as np
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
        def score_target(scf, data, sim):
            diff_table = zeros(shape(sim))
            for r in range(len(data)):
                for c in range(len(data[r])):
                    if not isnan(data[r][c]):
                        diff_table[r][c] = sim[r][c] * scf - data[r][c]
            return np.sum(square(diff_table))

        score = 0
        parameter_copy = copy.deepcopy(self.model.parameters)
        self.model.set_parameters(parameter_dict)
        for dose_species in self.data.get_dose_species():
            simulation_times = self.data.get_times()[dose_species]
            simulation_doses = self.data.get_doses()[dose_species]
            datatable = self.data.get_responses()[dose_species]
            datatable = [[el[0] for el in r] for r in datatable]
            if dose_species == 'Alpha':
                spec = 'Ia'
            else:
                spec = 'Ib'
            simulation = self.model.doseresponse(simulation_times, 'TotalpSTAT', spec, simulation_doses,
                                                 parameters=self.data.conditions[dose_species],
                                                 return_type='list', dataframe_labels='Alpha')['TotalpSTAT']
            opt = minimize(score_target, [0.1], args=(datatable, simulation))
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
        initial_score = 0
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
                    if initial_score == 0:
                        initial_score = min(residuals)
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


class Prior:
    """
    Documentation - A Prior instance fully characterizes a prior distribution, and can generate random values from the
                    prior distribution or caclulate the probability density for a given value being drawn from the prior

    Parameters/Attributes
    ----------
    type_of_distribution (str): can be 'lognormal', 'normal', or 'uniform'
    mean (float): the mean of the prior distrubiton; default is 1 (does not need to be specified for uniform priors)
                For lognormal priors this is what is subtracted from the logarithm of the value in
                the exponent of the Gaussian.
    sigma (float): the standard deviation of the prior distribution; default is 1 (does not need to be specified for
                    uniform priors. For lognormal priors this is what the logarithm of the value is divided by in
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


def __unpackMCMC__(ID, jobs, result, countQ):
    processMCMC = MCMC()
    processMCMC.load()
    processMCMC.__run_chain__(ID, jobs, result, countQ)


class MCMC:
    """
     Documentation - An MCMC instance is characterized by the IfnModel and IfnData associated with it.

     Parameters
     ----------
     model (IfnModel): the model to fit
     data (IfnData): the data to fit
     parameters (list): the names of the model parameters to fit
     priors (dict): keys corresponding to parameters, values are Prior objects
     jump_distributions (dict): keys corresponding to parameters, values are the sigmas for jump distributions
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
        self.filename = 'results/initial_model.p'
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

    # Private methods
    # ---------------
    def __MSE_of_parametric_model__(self, parameter_dict, sf_flag=0):
        def score_target(sf, data, sim):
            diff_table = zeros(shape(sim))
            for r in range(len(data)):
                for c in range(len(data[r])):
                    if not isnan(data[r][c]):
                        diff_table[r][c] = sim[r][c] * sf - data[r][c]
            return np.sum(square(diff_table))

        self.model.set_parameters(parameter_dict)
        total_data_table = []
        total_sim_table = []
        for dose_species in self.data.get_dose_species():
            # Get simulation parameters
            simulation_times = self.data.get_times()[dose_species]
            simulation_doses = self.data.get_doses()[dose_species]
            # Get data for comparison to
            datatable = self.data.get_responses()[dose_species]
            datatable = [[el[0] for el in r] for r in datatable]
            if not total_data_table:
                total_data_table = [el for el in datatable]
            else:
                total_data_table += datatable
            # Perform simulation
            if dose_species == 'Alpha':
                spec = 'Ia'
            else:
                spec = 'Ib'
            simulation = self.model.doseresponse(simulation_times, 'TotalpSTAT', spec, simulation_doses,
                                                 parameters=self.data.conditions[dose_species],
                                                 return_type='list', dataframe_labels=None)['TotalpSTAT']
            if not total_sim_table:
                total_sim_table = [el for el in simulation.tolist()]
            else:
                total_sim_table += simulation.tolist()
        # Score results by mean squared error
        opt = minimize(score_target, [40], args=(total_data_table, total_sim_table))
        score = opt['fun']
        sf = opt['x']
        if sf_flag == 1:
            return score, opt
        else:
            return score

    def __prior_penalty__(self, parameter_dict):
        penalty = 0
        for parameter, value in parameter_dict.items():
            if parameter in self.parameters_to_fit:
                penalty += self.priors[parameter].get_log_probability(value)
        return penalty

    def __parameter_jump__(self, temperature=1):
        new_parameters = {}
        for parameter in self.parameters_to_fit:
            current_value = log(self.model.parameters[parameter])
            new_value = rnd.lognormal(current_value, temperature * self.jump_distributions[parameter])
            new_parameters.update({parameter: new_value})
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
        self.beta = 1

    def __generate_parameters_from_priors__(self, n):
        pList = []
        for i in range(n):
            pList.append(self.__parameter_jump__(self.temperature))
        return pList

    def __score_and_sf_for_current_model__(self):
        MSE, sf = self.__MSE_of_parametric_model__(self.model.parameters, sf_flag=1)
        return MSE / self.beta + self.__prior_penalty__(self.model.parameters), sf

    def __run_chain__(self, ID, jobs, result, countQ):
        while True:
            initial_parameters = jobs.get()
            if initial_parameters is None:
                break
            initial_parameters = initial_parameters[0]
            self.model.set_parameters(initial_parameters)
            current_score, _ = self.__score_and_sf_for_current_model__()
            current_parameters = copy.deepcopy(self.model.parameters)

            progress_bar = self.num_samples / 10

            acceptance = 0
            attempts = 0
            while acceptance < self.num_samples:
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
                    with open('mcmc_results/progress.txt','a') as f:
                        f.write("Chain {} is {:.1f}% done, currently averaging {:.1f}% acceptance.\n".format(ID, acceptance/self.num_samples*100, acceptance/attempts*100))
                    # Save state at checkpoint
                    with open('mcmc_results/chain_results/{}chain.p'.format(str(ID)), 'wb') as f:
                        pickle.dump(self.__dict__, f, 2)
                    progress_bar += self.num_samples/10

                new_parameters = self.__parameter_jump__()
                self.model.set_parameters(new_parameters)

                new_score, new_scale_factor = self.__score_and_sf_for_current_model__()

                """
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

                alpha = asymmetry_factor * np.exp(-(new_score-current_score))
                if new_score < current_score or np.random.rand() < alpha:
                    # Search for MAP:
                    if new_score < self.best_fit_score:
                        self.best_fit_parameters = new_parameters
                        self.best_fit_scale_factor = new_scale_factor
                    # Add to chain
                    self.parameter_history.append(new_parameters)
                    self.scale_factor_history.append(new_scale_factor)
                    self.score_history.append(new_score)
                    current_score = new_score
                    current_parameters.update(new_parameters)
                    acceptance += 1
                else:
                    self.model.set_parameters(current_parameters)

            # Save final state
            with open('mcmc_results/chain_results/{}chain.p'.format(str(ID)), 'wb') as f:
                pickle.dump(self.__dict__, f, 2)

            result.put([self.parameter_history, self.scale_factor_history, self.score_history])
            countQ.put([ID, acceptance / attempts * 100])
        print("Chain {} exiting".format(ID))

        with open('mcmc_results/progress.txt', 'a') as f:
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

    def plot_parameter_distributions(self):
        if self.num_chains == 1:
            k = len(self.parameters_to_fit)  # total number subplots
            n = 2  # number of chart columns
            m = (k - 1) // n + 1  # number of chart rows
            fig, axes = plt.subplots(m, n, figsize=(n * 5, m * 3))
            if k % 2 == 1:  # avoids extra empty subplot
                axes[-1][n - 1].set_axis_off()
            for i, (name, col) in enumerate(df.iteritems()):
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
            fig.tight_layout()
            if save == True:
                if title == '':
                    plt.savefig(results_dir + 'parameter_distributions.pdf')
                else:
                    plt.savefig(results_dir + title + '.pdf')
            return (fig, axes)

    def fit(self, num_accepted_steps: int, num_chains: int, burn_rate: float, down_sample_frequency: int, beta: float,
            cpu=None, initialise=True):
        # Check input parameters
        self.__check_input__
        print("Performing MCMC Analysis")
        # Create results directory
        if not os.path.isdir("mcmc_results"):
            os.mkdir("mcmc_results")
        if not os.path.isdir("mcmc_results\chain_results"):
            os.mkdir("mcmc_results\chain_results")
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
            initial_parameters = [initialise for _ in range(self.num_chains)]
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
        with open('mcmc_results/progress.txt', 'w') as f:  # clear previous progress report
            f.write('')
        jobs = Queue()  # put jobs on queue
        result = JoinableQueue()
        countQ = JoinableQueue()
        # Start up chains
        # Prepare instance for multiprocessing by pickling
        self.save()
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
            [Process(target=__unpackMCMC__, args=(i, jobs, result, countQ)).start() for i in range(number_of_processes)]
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
        for chain in pool_results:
            self.parameter_history += chain[0]
            self.scale_factor_history += chain[1]
            self.score_history += chain[2]
        # Perform burn-in and down sampling
        for chain in pool_results:
            sample_pattern = range(int(burn_rate*len(chain)), len(chain), down_sample_frequency)
            self.thinned_parameter_samples += [chain[0][i] for i in sample_pattern]
            self.thinned_parameter_scale_factors += [chain[1][i] for i in sample_pattern]
            self.thinned_parameter_scores += [chain[2][i] for i in sample_pattern]
        # Write summary file
        with open("mcmc_results/simulation_summary.txt", 'w') as f:
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
        self.save(alt_filename="mcmc_results/mcmc_fit.p")


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
