# Pydream imports
from pydream.core import run_dream
import numpy as np

# PySB imports
from ifnclass.ifndata import IfnData, DataAlignment

from scipy.stats import norm
import os
import shutil

from pydream.convergence import Gelman_Rubin

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class IFN_posterior_object():
    """
    Define posterior function which will evaluate the probability of a given
    model for the likelihood and model prior
    The function returned from this decorator takes as input a parameter vector
    (parameter values are in the same order as sampled_parameter_names).
    The function returns a log probability value for the parameter vector given
    the experimental data.
    """
    def __init__(self, sampled_parameter_names, model, alignment):
        self.sampled_parameter_names = sampled_parameter_names
        self.model = model
        self.tspan = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
        self.alpha_doses = [10, 100, 300, 1000, 3000, 10000, 100000]
        self.beta_doses = [0.2, 6, 20, 60, 200, 600, 2000]
        self.sf = 1.0
        self.aligned_data = alignment

        # Create scipy normal probability distributions for data likelihoods
        mean_data = alignment.summarize_data()
        exp_data = np.array([[el[0] for el in dose] for dose in
                             mean_data.data_set.values]).flatten()
        exp_data_std = np.array([[el[1] for el in dose] for dose in
                                mean_data.data_set.values]).flatten()
        self.like_ctot = norm(loc=exp_data, scale=exp_data_std)

    # -------------------------------------------------------------------------
    def IFN_posterior(self, parameter_vector):
        # Change model parameter values to current location in parameter space
        # (values are in log(value) format)
        shared_param_dict = {pname: 10 ** pvalue for pname, pvalue in
                             zip(self.sampled_parameter_names,
                                 parameter_vector)}

        self.model.set_parameters(shared_param_dict)

        # Simulate experimentally measured TotalpSTAT values.
        # Alpha
        dradf = self.model.doseresponse(self.tspan, 'TotalpSTAT', 'Ia',
                                        self.alpha_doses,
                                        parameters={'Ib': 0},
                                        scale_factor=self.sf,
                                        return_type='dataframe',
                                        dataframe_labels='Alpha')
        drbdf = self.model.doseresponse(self.tspan, 'TotalpSTAT', 'Ib',
                                        self.beta_doses,
                                        parameters={'Ia': 0},
                                        scale_factor=self.sf,
                                        return_type='dataframe',
                                        dataframe_labels='Beta')

        # Concatenate and flatten
        total_simulation_data = IfnData(name='custom',
                                        df=pd.concat([dradf, drbdf]))
        sim_data = np.array([[el[0] for el in dose] for dose in
                            total_simulation_data.data_set.values]).flatten()

        # Calculate log probability from simulated experimental values

        logp_ctotal = np.sum(self.like_ctot.logpdf(sim_data))

        # If model simulation failed due to integrator errors, return a log
        # probability of -inf.
        if np.isnan(logp_ctotal):
            logp_ctotal = -np.inf

        return logp_ctotal


def DREAM_fit(model, priors_list, posterior, start_params,
              sampled_param_names, niterations, nchains, sim_name,
              save_dir, custom_params={}):
    """
    The DREAM fitting algorithm as implemented in run_dream(), plus decorations
    for saving run parameters, checking convergence, and post fitting analysis.
    """
    converged = False
    total_iterations = niterations

    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    sampled_params, log_ps = run_dream(priors_list, posterior,
                                       start=start_params,
                                       niterations=niterations,
                                       nchains=nchains,
                                       multitry=False,
                                       gamma_levels=4, adapt_gamma=True,
                                       history_thin=1, model_name=sim_name,
                                       verbose=True)

    # Save sampling output (sampled param values and their corresponding logps)
    for chain in range(len(sampled_params)):
        np.save(os.path.join(save_dir, sim_name+str(chain) + '_' +
                             str(total_iterations)), sampled_params[chain])
        np.save(os.path.join(save_dir, sim_name+str(chain) + '_' +
                             str(total_iterations)), log_ps[chain])

    # Check convergence and continue sampling if not converged

    GR = Gelman_Rubin(sampled_params)
    print('At iteration: ', total_iterations, ' GR = ', GR)
    np.savetxt(os.path.join(save_dir, sim_name + str(total_iterations) +
                            '.txt'), GR)

    old_samples = sampled_params
    if np.any(GR > 1.2):
        starts = [sampled_params[chain][-1, :] for chain in range(nchains)]
        while not converged:
            total_iterations += niterations

            sampled_params, log_ps = run_dream(priors_list, posterior,
                                               start=starts,
                                               niterations=niterations,
                                               nchains=nchains, multitry=False,
                                               gamma_levels=4,
                                               adapt_gamma=True,
                                               history_thin=1,
                                               model_name=sim_name,
                                               verbose=True, restart=True)

            for chain in range(len(sampled_params)):
                np.save(os.path.join(save_dir, sim_name + '_' + str(chain) +
                                     '_' + str(total_iterations)),
                        sampled_params[chain])
                np.save(os.path.join(save_dir, sim_name + '_' + str(chain) +
                                     '_' + str(total_iterations)),
                        log_ps[chain])

            old_samples = [np.concatenate((old_samples[chain],
                           sampled_params[chain])) for chain in range(nchains)]
            GR = Gelman_Rubin(old_samples)
            print('At iteration: ', total_iterations, ' GR = ', GR)
            np.savetxt(os.path.join(save_dir, sim_name + '_' +
                                    str(total_iterations)+'.txt'), GR)

            if np.all(GR < 1.2):
                converged = True

    log_ps = np.array(log_ps)
    sampled_params = np.array(sampled_params)
    print(log_ps.shape)
    print(sampled_params.shape)
    try:
        # Maximum posterior model:
        max_in_each_chain = [np.argmax(chain) for chain in log_ps]
        global_max_chain_idx = np.argmax([log_ps[chain][max_idx] for
                                          chain, max_idx in
                                          enumerate(max_in_each_chain)])
        ml_params = sampled_params[global_max_chain_idx,
                                   max_in_each_chain[global_max_chain_idx]]
        ml_model = {pname: 10 ** pvalue for pname, pvalue in
                    zip(sampled_param_names, ml_params)}
        print(ml_model,
              file=open(os.path.join(save_dir, sim_name +
                                     '_ML_params.txt'), 'w'))

    except IndexError:
        print("IndexError finding maximum posterior parameters")
        pass

    try:
        # Plot output
        total_iterations = len(old_samples[0])
        burnin = int(total_iterations / 2)
        samples = np.concatenate(list((old_samples[i][burnin:, :] for
                                       i in range(len(old_samples)))))
        np.save(os.path.join(save_dir, sim_name+'_samples'), samples)
        ndims = len(old_samples[0][0])
        colors = sns.color_palette(n_colors=ndims)
        for dim in range(ndims):
            sns.distplot(samples[:, dim], color=colors[dim])
            plt.savefig(os.path.join(save_dir, sim_name + '_dimension_' +
                                     str(dim) + '_' +
                                     sampled_param_names[dim] +
                                     '.pdf'))

        # Convert to dataframe
        df = pd.DataFrame(samples, columns=sampled_param_names)
        g = sns.pairplot(df)
        for i, j in zip(*np.triu_indices_from(g.axes, 1)):
            g.axes[i, j].set_visible(False)
        g.savefig(os.path.join(save_dir, 'corner_plot.pdf'))

        # Basic statistics
        mean_parameters = np.mean(samples, axis=0)
        median_parameters = np.median(samples, axis=0)
        np.save(os.path.join(save_dir, 'mean_parameters'), mean_parameters)
        np.save(os.path.join(save_dir, 'median_parameters'), median_parameters)
        df.describe().to_csv(os.path.join(save_dir,
                             'descriptive_statistics.csv'))

    except (ImportError, OSError):
        pass

    # Clean up stray files
    shutil.move(os.path.join(os.getcwd(), '*_DREAM_chain_*.*'),
                save_dir)


def posterior_prediction(model, parameter_vector, parameter_names, sf,
                         test_times=[2.5, 5.0, 7.5, 10.0, 20.0, 60.0],
                         alpha_doses=np.logspace(-1, 5, 15),
                         beta_doses=np.logspace(-1, 5, 15)):
    """
    Produce predictions for IFNa and IFNb using model with parameters given
    as input to the function.
    """
    # Make predictions
    model.set_parameters(parameter_vector)
    dradf = model.doseresponse(test_times, 'TotalpSTAT', 'Ia',
                               alpha_doses, parameters={'Ib': 0},
                               scale_factor=sf,
                               return_type='dataframe',
                               dataframe_labels='Alpha')
    drbdf = model.doseresponse(test_times, 'TotalpSTAT', 'Ib',
                               beta_doses, parameters={'Ia': 0},
                               scale_factor=sf,
                               return_type='dataframe',
                               dataframe_labels='Beta')

    posterior = IfnData('custom', df=pd.concat((dradf, drbdf)),
                        conditions={'Alpha': {'Ib': 0},
                                    'Beta': {'Ia': 0}})
    posterior.drop_sigmas()
    return posterior


def posterior_IFN_summary_statistics(posterior_predictions):
    """
    Encapsulates the code to compute summary statistics from a list of
    posterior predictions. Used to increase readability in the main runfile.
    """
    mean_alpha_predictions = np.mean([posterior_predictions[i].data_set.
                                     loc['Alpha'].values for i in
                                     range(len(posterior_predictions))],
                                     axis=0)
    mean_beta_predictions = np.mean([posterior_predictions[i].data_set.
                                    loc['Beta'].values for i in
                                    range(len(posterior_predictions))],
                                    axis=0)

    std_alpha_predictions = np.std([posterior_predictions[i].data_set.
                                   loc['Alpha'].values.astype(np.float64) for
                                   i in range(len(posterior_predictions))],
                                   axis=0)
    std_beta_predictions = np.std([posterior_predictions[i].data_set.
                                  loc['Beta'].values.astype(np.float64) for
                                  i in range(len(posterior_predictions))],
                                  axis=0)

    return mean_alpha_predictions, std_alpha_predictions,\
        mean_beta_predictions, std_beta_predictions


def _get_data_coordinates(data: IfnData):
    """Generates a list of all (species, dose, time) triplets with non-NaN
    values in the IfnData data_set attribute. Returns this list of 3-elemnt
    tuples.
    """
    coord = []
    for s in data.get_dose_species():
        for d in data.get_doses(species=s):
            for t in data.get_times(species=s):
                c = (s, d, t)
                if not pd.isnull(data.data_set.loc[c[0:2]][c[2]]):
                    if type(data.data_set.loc[c[0:2]][c[2]]) == tuple:
                        if not pd.isnull(data.data_set.loc[c[0:2]][c[2]][0]):
                            coord.append(c)
                    else:
                        coord.append(c)
    return coord


def _split_data(datalist, withhold):
    """Splits a list of IfnData instances into test and train subsets, placing
    <withold> percentage of the data in the test subset. The testing subset is
    then aligned using a DataAlignment instance, and the training subset is
    scaled according to the *testing* subset scale factors. The test and train
    aligned IfnData objects output by the DataAlignment.summarize_data() method
    are returned.
    """
    assert 0 <= withhold <= 100
    # Build mask which selects <withhold> points for test subset
    data_coord = _get_data_coordinates(datalist[0])
    test_size = int((100-withhold) * len(data_coord) / 100.0)
    test_idcs = np.random.choice(len(data_coord), test_size, False)
    test_coord = [data_coord[i] for i in test_idcs]
    train_coord = [c for c in data_coord if c not in test_coord]

    # Sepearte data into test and train subsets
    test_datalist = [d.copy() for d in datalist]
    train_datalist = [d.copy() for d in datalist]
    for obj in test_datalist:
        for c in test_coord:
            obj.data_set.loc[c[0:2]][c[2]] = np.NaN

    for obj in train_datalist:
        for c in train_coord:
            obj.data_set.loc[c[0:2]][c[2]] = np.NaN

    train_alignment = DataAlignment()
    train_alignment.add_data(train_datalist)
    train_alignment.align()
    train_alignment.get_scaled_data()
    train = train_alignment.summarize_data()

    if withhold == 0:
        test = None
    else:
        test_alignment = DataAlignment()
        test_alignment.add_data(test_datalist)
        test_alignment.scale_factors = train_alignment.scale_factors
        test_alignment.get_scaled_data()
        test = test_alignment.summarize_data()

    return train, test


def bootstrap(model, datalist, priors_list, start_params,
              sampled_param_names, niterations, nchains, sim_name,
              save_dir, withhold: int, epochs: int):
    """
    Given a list of IfnData objects, splits the entire set of data into
    train and test samples, aligns each subset, fits the train subset,
    and checks the std. dev. adjusted mean square error on the test subset.

    The percentage of data (as an int) in the test set is input as <whithhold>.
    Repeats the process <epochs> number of times, splitting randomly each time.

    Assumes that each IfnData object in datalist has the same dose species,
    doses, and times.
    """
    dir_list = []
    for epoch in range(epochs):
        # split data
        train, test = _split_data(datalist, withhold)

        print(test.data_set)
        exit()
        # build posterior
        posterior_obj = IFN_posterior_object(sampled_param_names, model, train)

        # fit training data
        epoch_save_dir = os.makedirs(os.path.join(os.getcwd(), save_dir,
                                                  'Batch_{}'.format(epoch)),
                                     exist_ok=True)
        dir_list.append(epoch_save_dir)

        DREAM_fit(model=model, priors_list=priors_list,
                  posterior=posterior_obj.IFN_posterior,
                  start_params=start_params,
                  sampled_param_names=sampled_param_names,
                  niterations=niterations,
                  nchains=nchains, sim_name=sim_name, save_dir=epoch_save_dir)

    # analyse results
    all_data, _ = _split_data(datalist, 0)
    all_data.drop_sigmas()

    mean_y = np.mean(all_data.data_set.values)
    SStot = np.sum(
                   np.square(
                             np.subtract(all_data.data_set.values, mean_y)))

    SSres_list = []
    for dir in dir_list:
        with open(os.path.join(dir, sim_name + '_ML_params.txt'), 'r') as f:
            map = eval(f.read())
        pred = posterior_prediction(model, map, sampled_param_names, 1.0,
                                    test_times=[2.5, 5.0, 7.5, 10.0, 20., 60.],
                                    alpha_doses=[10, 100, 300, 1000, 3000,
                                                 10000, 100000],
                                    beta_doses=[0.2, 6, 20, 60, 200, 600,
                                                2000])
        MSE = np.sum(
                np.square(
                    np.subtract(
                        pred.data_set.values, all_data.data_set.values)))
        SSres_list.append(MSE)

    mean_R2 = np.mean([1 - SSres / SStot for SSres in SSres_list])

    with open(os.path.join(save_dir, 'bootstrap_analysis.txt'), 'w') as f:
        f.write("mean R2 = {}\nmean MSE = {}".format(mean_R2,
                                                     np.mean(SSres_list)))
