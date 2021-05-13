from pysb.export import export
from pysb import bng
from collections import OrderedDict
import copy
from numpy import multiply, zeros, nan, asarray, load, power, logspace
import numpy as np
import pandas as pd
import pickle
import time
import os
from tqdm import tqdm
from random import randint
from ifnclass.ifndata import IfnData, DataAlignment


class IfnModel:
    """
    Documentation - An IfnModel object is the primary object for modelling
    experimental IFN dose-response or timecourse data. This is the expected
    model object used for plotting and fitting within the IFNmodeling module.

    Parameters
    ----------
    name : string
        The name of the ifnmodels model file written using PySB, including
        file extension.
    Attributes
    -------
    name : string
        The filename used to find source files for this IfnData instance
    model : PySB standalone python model
        An instance of the python Model class, specific for this model
    parameters : ordered dict
        A dictionary with all the Model parameters. The dictionary is
        ordered to match the order in the PySB model.
    default_parameters : ordered dict
        A dictionary containing all the original imported Model parameters
    Methods
    -------
    build_model -> Model
        loads and returns a PySB model instance
    check_if_parameters_in_model -> Bool
        takes a dictionary of parameters and checks if they are in the model
    set_parameters -> int
        takes a dictionary of parameters and updates the model parameters
        maintains detailed balance since this is an IFN model
        returns 0 if successful and returns 1 otherwise
    reset_parameters
        sets self.parameters to the initially imported parameter values
    timecourse -> dict = keys corresponding to observable names, values are lists of trajectory values
                        * This method does not check the input 'parameters' for maintained detailed balance *
        Inputs:
            times (list) : time points to output, in minutes
            observables (string or list) : name of model observable(s) to return
                                           if list is given, function returns a list of trajectories
                                           if string is given, function returns one trajectory
                                           in either case, a trajectory is a list of observable values
                                                corresponding to each time point in times
            parameters (dict) : parameters for simulation; default is the current
                                parameters stored in self.parameters
            return_type = 'list' or 'dataframe': indicates whether you want the trajectory returned as a
                                                 list or Pandas DataFrame
                                                 Must also input dataframe_labels
            dataframe_labels (list): REQUIRED FOR  return_type='dataframe'; contains values for the DataFrame labels
                                     [dose species (string), dose in pM (float)]
    doseresponse() -> dict = keys corresponding to observable names, values are arrays where each column is a time slice
                            and each row corresponds to a dose value
                            * This method does not check the input 'parameters' for maintained detailed balance *
        Inputs:
            times (list) : time points to output, in minutes
            observables (string or list) : name of model observable(s) to return
                                           if list is given, function returns a list of trajectories
                                           if string is given, function returns one trajectory
                                           in either case, a trajectory is a list of observable values
                                                corresponding to each time point in times
            dose_species (string) : name of model observable modified during dose response
            doses (list) : list of values to substitute for dose_species in model in simulating dose response
            parameters (dict) : parameters for simulation; default is the current
                                parameters stored in self.parameters
            return_type = 'list' or 'dataframe': indicates whether you want the trajectory returned as a
                                                 list or Pandas DataFrame
                                                 Input variable dataframe_labels is OPTIONAL
            dataframe_labels (string): a label for the dose_species if Model lable isn't desired; default is None
    """

    # Initializer / Instance Attributes
    def __init__(self, name):
        self.name = name
        self.model = self.build_model(self.name)
        self.parameters = self.build_parameters(self.model)
        self.default_parameters = copy.deepcopy(self.parameters)

    # Instance methods
    def build_model(self, name):
        if name == '':
            return None
        else:
            model_code = __import__('ifnmodels.' + name, fromlist=['ifnmodels'])
            py_output = export(model_code.model, 'python')
            ODE_filename = "ODE_system_{}_{}.py".format(time.strftime("%Y%m%d-%H%M%S"), randint(100000, 999999))
            with open(ODE_filename, 'w') as f:
                f.write(py_output)
            ODE_system = __import__(ODE_filename[:-3])
            model_obj = ODE_system.Model()
            os.remove(ODE_filename)
            return model_obj

    def save_model(self, name):
        file_text = export(self.model, 'bng_net')
        print(file_text)
        with open(name, 'w') as f:
            f.write(file_text)

    def load_model(self, name):
        try:
            with open('ifnmodels/'+name, 'rb') as f:
                tmp_dict = pickle.load(f)
            self.__dict__.update(tmp_dict)
        except FileNotFoundError:
            try:
                with open(name, 'rb') as f:
                    tmp_dict = pickle.load(f)
                self.__dict__.update(tmp_dict)
            except:
                with open(name, 'r') as f:
                    netfile = f.read()
                    bng.load_equations(self.model, netfile)


    def build_parameters(self, pysb_model):
        if pysb_model is not None:
            parameter_dict = OrderedDict({})
            for p in pysb_model.parameters:
                parameter_dict.update({p[0]: p[1]})
            return parameter_dict
        else:
            return {}

    def check_if_parameters_in_model(self, test_parameters):
        list1 = [element for element in test_parameters.keys() if element in self.parameters.keys()]
        return list1 == list(test_parameters.keys())

    def check_for_detailed_balance_parameters(self, new_parameters):
        # Check if any parameters which must maintain detailed balance are even in new_parameters
        db_parameters_present = False
        for key in ['ka1','kd1','ka2','kd2','ka3','kd3','ka4','kd4','k_a1','k_d1','k_a2','k_d2','k_a3','k_d3','k_a4','k_d4']:
            db_parameters_present = (key in new_parameters.keys()) or db_parameters_present
        return db_parameters_present

    def __check_detailed_balance__(self):
        alpha_check = True
        beta_check = True
        if 'kd4' in self.parameters.keys():
            q1 = self.parameters['ka1'] / self.parameters['kd1']
            q2 = self.parameters['ka2'] / self.parameters['kd2']
            q3 = self.parameters['ka3'] / self.parameters['kd3']
            q4 = self.parameters['ka4'] / self.parameters['kd4']
            alpha_check =  (abs(q1 * q3 / (q2 * q4) - 1) < 1E-4)
        if 'k_d4' in self.parameters.keys():
            q1 = self.parameters['k_a1'] / self.parameters['k_d1']
            q2 = self.parameters['k_a2'] / self.parameters['k_d2']
            q3 = self.parameters['k_a3'] / self.parameters['k_d3']
            q4 = self.parameters['k_a4'] / self.parameters['k_d4']
            beta_check =  (abs(q1 * q3 / (q2 * q4) - 1) < 1E-4)
        if ('kd4' in self.parameters.keys()) or ('k_d4' in self.parameters.keys()):
            return (alpha_check and beta_check)
        else:
            print("Could not find detailed balance parameters")
            return False

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, new_parameters: dict, db_check=True):
        if self.check_if_parameters_in_model(new_parameters):
            if self.check_for_detailed_balance_parameters(new_parameters) and db_check:
                # Use all input values given, default to current model values otherwise
                detailed_balance_dict = {'ka1': 1, 'kd1': 1,
                                         'ka2': 1, 'kd2': 1,
                                         'ka3': 1, 'kd3': 1,
                                         'ka4': 1, 'kd4': 1,
                                         'k_a1': 1, 'k_d1': 1,
                                         'k_a2': 1, 'k_d2': 1,
                                         'k_a3': 1, 'k_d3': 1,
                                         'k_a4': 1, 'k_d4': 1}
                # These will be used to check which parameter is free for maintaining detailed balance
                detailed_balance_keys_alpha = ['ka1','kd1','ka2','kd2','ka3','kd3','ka4','kd4']
                detailed_balance_keys_beta = ['k_a1','k_d1','k_a2','k_d2','k_a3','k_d3','k_a4','k_d4']
                for key in detailed_balance_dict.keys():
                    if key in new_parameters.keys():
                        detailed_balance_dict.update({key: new_parameters[key]})
                        # Parameter is not free, so remove from appropriate list
                        try:
                            detailed_balance_keys_alpha.remove(key)
                        except ValueError:
                            # If it is not in alpha it must be in beta
                            detailed_balance_keys_beta.remove(key)
                    else:
                        detailed_balance_dict.update({key: self.parameters[key]})
                # Maintain detailed balance
                if detailed_balance_keys_alpha == [] or detailed_balance_keys_beta == []:
                    raise ValueError("Not enough free parameters to maintain detailed balance")
                else:
                    # Find the IFNa parameter left free to maintain detailed balance and add it to new_parameters
                    if 'kd3' in detailed_balance_keys_alpha:
                        q1 = detailed_balance_dict['ka1'] / detailed_balance_dict['kd1']
                        q2 = detailed_balance_dict['ka2'] / detailed_balance_dict['kd2']
                        q4 = detailed_balance_dict['ka4'] / detailed_balance_dict['kd4']
                        q3 = q2 * q4 / q1
                        kd3 = detailed_balance_dict['ka3'] / q3
                        new_parameters.update({'kd3': kd3})
                    elif 'kd4' in detailed_balance_keys_alpha:
                        q1 = detailed_balance_dict['ka1'] / detailed_balance_dict['kd1']
                        q2 = detailed_balance_dict['ka2'] / detailed_balance_dict['kd2']
                        q3 = detailed_balance_dict['ka3'] / detailed_balance_dict['kd3']
                        q4 = q1 * q3 / q2
                        kd4 = detailed_balance_dict['ka4'] / q4
                        new_parameters.update({'kd4': kd4})
                    elif 'ka3' in detailed_balance_keys_alpha:
                        q1 = detailed_balance_dict['ka1'] / detailed_balance_dict['kd1']
                        q2 = detailed_balance_dict['ka2'] / detailed_balance_dict['kd2']
                        q4 = detailed_balance_dict['ka4'] / detailed_balance_dict['kd4']
                        q3 = q2 * q4 / q1
                        ka3 = detailed_balance_dict['kd3'] * q3
                        new_parameters.update({'ka3': ka3})
                    elif 'ka4' in detailed_balance_keys_alpha:
                        q1 = detailed_balance_dict['ka1'] / detailed_balance_dict['kd1']
                        q2 = detailed_balance_dict['ka2'] / detailed_balance_dict['kd2']
                        q3 = detailed_balance_dict['ka3'] / detailed_balance_dict['kd3']
                        q4 = q1 * q3 / q2
                        ka4 = detailed_balance_dict['kd4'] * q4
                        new_parameters.update({'ka4': ka4})
                    elif 'kd1' in detailed_balance_keys_alpha:
                        q3 = detailed_balance_dict['ka3'] / detailed_balance_dict['kd3']
                        q2 = detailed_balance_dict['ka2'] / detailed_balance_dict['kd2']
                        q4 = detailed_balance_dict['ka4'] / detailed_balance_dict['kd4']
                        q1 = q2 * q4 / q3
                        kd1 = detailed_balance_dict['ka1'] / q1
                        new_parameters.update({'kd1': kd1})
                    elif 'kd2' in detailed_balance_keys_alpha:
                        q1 = detailed_balance_dict['ka1'] / detailed_balance_dict['kd1']
                        q4 = detailed_balance_dict['ka4'] / detailed_balance_dict['kd4']
                        q3 = detailed_balance_dict['ka3'] / detailed_balance_dict['kd3']
                        q2 = q1 * q3 / q4
                        kd2 = detailed_balance_dict['ka2'] / q2
                        new_parameters.update({'kd2': kd2})
                    elif 'ka1' in detailed_balance_keys_alpha:
                        q3 = detailed_balance_dict['ka3'] / detailed_balance_dict['kd3']
                        q2 = detailed_balance_dict['ka2'] / detailed_balance_dict['kd2']
                        q4 = detailed_balance_dict['ka4'] / detailed_balance_dict['kd4']
                        q1 = q2 * q4 / q3
                        ka1 = detailed_balance_dict['kd1'] * q1
                        new_parameters.update({'ka1': ka1})
                    elif 'ka2' in detailed_balance_keys_alpha:
                        q1 = detailed_balance_dict['ka1'] / detailed_balance_dict['kd1']
                        q4 = detailed_balance_dict['ka4'] / detailed_balance_dict['kd4']
                        q3 = detailed_balance_dict['ka3'] / detailed_balance_dict['kd3']
                        q2 = q1 * q3 / q4
                        ka2 = detailed_balance_dict['kd2'] * q2
                        new_parameters.update({'ka2': ka2})

                    # Find the IFNb parameter left free to maintain detailed balance and add it to new_parameters
                    if 'k_d3' in detailed_balance_keys_beta:
                        q1 = detailed_balance_dict['k_a1'] / detailed_balance_dict['k_d1']
                        q2 = detailed_balance_dict['k_a2'] / detailed_balance_dict['k_d2']
                        q4 = detailed_balance_dict['k_a4'] / detailed_balance_dict['k_d4']
                        q3 = q2 * q4 / q1
                        k_d3 = detailed_balance_dict['k_a3'] / q3
                        new_parameters.update({'k_d3': k_d3})
                    elif 'k_d4' in detailed_balance_keys_beta:
                        q1 = detailed_balance_dict['k_a1'] / detailed_balance_dict['k_d1']
                        q2 = detailed_balance_dict['k_a2'] / detailed_balance_dict['k_d2']
                        q3 = detailed_balance_dict['k_a3'] / detailed_balance_dict['k_d3']
                        q4 = q1 * q3 / q2
                        k_d4 = detailed_balance_dict['k_a4'] / q4
                        new_parameters.update({'k_d4': k_d4})
                    elif 'k_a3' in detailed_balance_keys_beta:
                        q1 = detailed_balance_dict['k_a1'] / detailed_balance_dict['k_d1']
                        q2 = detailed_balance_dict['k_a2'] / detailed_balance_dict['k_d2']
                        q4 = detailed_balance_dict['k_a4'] / detailed_balance_dict['k_d4']
                        q3 = q2 * q4 / q1
                        k_a3 = detailed_balance_dict['k_d3'] * q3
                        new_parameters.update({'k_a3': k_a3})
                    elif 'k_a4' in detailed_balance_keys_beta:
                        q1 = detailed_balance_dict['k_a1'] / detailed_balance_dict['k_d1']
                        q2 = detailed_balance_dict['k_a2'] / detailed_balance_dict['k_d2']
                        q3 = detailed_balance_dict['k_a3'] / detailed_balance_dict['k_d3']
                        q4 = q1 * q3 / q2
                        k_a4 = detailed_balance_dict['k_d4'] * q4
                        new_parameters.update({'k_a4': k_a4})
                    elif 'k_d1' in detailed_balance_keys_beta:
                        q3 = detailed_balance_dict['k_a3'] / detailed_balance_dict['k_d3']
                        q2 = detailed_balance_dict['k_a2'] / detailed_balance_dict['k_d2']
                        q4 = detailed_balance_dict['k_a4'] / detailed_balance_dict['k_d4']
                        q1 = q2 * q4 / q3
                        k_d1 = detailed_balance_dict['k_a1'] / q1
                        new_parameters.update({'k_d1': k_d1})
                    elif 'k_d2' in detailed_balance_keys_beta:
                        q1 = detailed_balance_dict['k_a1'] / detailed_balance_dict['k_d1']
                        q4 = detailed_balance_dict['k_a4'] / detailed_balance_dict['k_d4']
                        q3 = detailed_balance_dict['k_a3'] / detailed_balance_dict['k_d3']
                        q2 = q1 * q3 / q4
                        k_d2 = detailed_balance_dict['k_a2'] / q2
                        new_parameters.update({'k_d2': k_d2})
                    elif 'k_a1' in detailed_balance_keys_beta:
                        q3 = detailed_balance_dict['k_a3'] / detailed_balance_dict['k_d3']
                        q2 = detailed_balance_dict['k_a2'] / detailed_balance_dict['k_d2']
                        q4 = detailed_balance_dict['k_a4'] / detailed_balance_dict['k_d4']
                        q1 = q2 * q4 / q3
                        k_a1 = detailed_balance_dict['k_d1'] * q1
                        new_parameters.update({'k_a1': k_a1})
                    elif 'k_a2' in detailed_balance_keys_beta:
                        q1 = detailed_balance_dict['k_a1'] / detailed_balance_dict['k_d1']
                        q4 = detailed_balance_dict['k_a4'] / detailed_balance_dict['k_d4']
                        q3 = detailed_balance_dict['k_a3'] / detailed_balance_dict['k_d3']
                        q2 = q1 * q3 / q4
                        k_a2 = detailed_balance_dict['k_d2'] * q2
                        new_parameters.update({'k_a2': k_a2})

            self.parameters.update(new_parameters)
            return 0
        else:
            print("Some of the parameters were not found in the model. Did not update parameters.")
            list1 = [element for element in new_parameters.keys() if element not in self.parameters.keys()]
            print("Perhaps the following are not in the model?")
            print(list1)
            return 1

    def reset_parameters(self):
        self.parameters = copy.deepcopy(self.default_parameters)

    def timecourse(self, times, observable, parameters=None, return_type='list', dataframe_labels=[], scale_factor=1):
        # Keep current parameter state
        initial_parameters = copy.deepcopy(self.parameters)
        # Substitute in new simulation-specific values
        if parameters is not None:
            self.set_parameters(parameters, db_check=False)
        # Simulate
        if type(times[0]) == str:
            for i in range(len(times)):
                times[i] = float(times[i])
        noTflag = False
        if times[0] != 0:
            noTflag = True
            times = [0, *times]
        (_, sim) = self.model.simulate(multiply(times, 60).tolist(), param_values=list(self.parameters.values()))
        if noTflag == True:
            sim = sim[1:]
        # Reset model parameter vector so that timecourse() is not permanently altering model parameters
        self.parameters = initial_parameters
        # Return trajectory(ies)
        if return_type == 'list':
            if type(observable) == list:
                if scale_factor == 1:
                    return {obs: sim[obs] for obs in observable}
                else:
                    return {obs: multiply(scale_factor, sim[obs]).tolist() for obs in observable}
            else:
                if scale_factor == 1:
                    return {observable: sim[observable]}
                else:
                    return {observable: multiply(scale_factor, sim[observable]).tolist()}
        elif return_type == 'dataframe':
            # Check if mandatory labels also provided; return as list if not provided
            if dataframe_labels == []:
                print("Insufficient labels provided for dataframe return type")
                if type(observable) == list:
                    if scale_factor == 1:
                        return {obs: sim[obs] for obs in observable}
                    else:
                        return {obs: multiply(scale_factor, sim[obs]).tolist() for obs in observable}
                else:
                    if scale_factor == 1:
                        return {observable: sim[observable]}
                    else:
                        return {observable: multiply(scale_factor, sim[observable]).tolist()}
            # Build dataframe
            if type(observable) == list:
                rows = [[obs, dataframe_labels[0], dataframe_labels[1]] + [(val, nan) for val in sim[obs]] for obs in observable]
                dataframe = pd.DataFrame.from_records(rows, columns=['Observable', 'Dose_Species', 'Dose (pM)']+times)
                dataframe.set_index(['Observable', 'Dose_Species', 'Dose (pM)'], inplace=True)
                if scale_factor == 1:
                    return dataframe
                else:
                    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])
                    for obs in observable:
                        dataframe.loc[obs].loc[dataframe_labels[0]].iloc[:, i] = dataframe.loc[obs].loc[dataframe_labels[0]].iloc[:, i].apply(scale_data)
                    return dataframe
            else:
                row = [(dataframe_labels[0], dataframe_labels[1], *[(val, nan) for val in sim[observable]])]
                dataframe = pd.DataFrame.from_records(row, columns=['Dose_Species', 'Dose (pM)']+times)
                dataframe.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
                if scale_factor == 1:
                    return dataframe
                else:
                    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])
                    dataframe.loc[dataframe_labels[0]].iloc[:, i] = dataframe.loc[dataframe_labels[0]].iloc[:, i].apply(scale_data)
                    return dataframe

    def doseresponse(self, times, observable, dose_species, doses, parameters={}, return_type='list',
                     dataframe_labels=None, scale_factor=1):
        # create dose_response_table dictionary
        if type(observable) == list:
            dose_response_table = {obs: zeros((len(doses), len(times))) for obs in observable}
        else:
            dose_response_table = {observable: zeros((len(doses), len(times)))}
        # prepare custom parameters dictionary
        dose_parameters = copy.deepcopy(self.parameters)
        dose_parameters.update(parameters)
        # iterate through all doses
        picoMolar = 1E-12
        Avogadro = 6.022E23
        volEC = self.parameters['volEC']
        for idx, d in enumerate(doses):
            dose_parameters.update({dose_species: d*picoMolar*Avogadro*volEC})
            trajectories = self.timecourse(times, observable, parameters=dose_parameters)
            # add results to dose_response_table dictionary
            for observable_species in trajectories.keys():
                dose_response_table[observable_species][idx] = trajectories[observable_species]
        # return dose response curves
        if return_type == 'list':
            for observable_species in dose_response_table.keys():
                dose_response_table[observable_species] = dose_response_table[observable_species].tolist()
            if scale_factor == 1:
                return dose_response_table
            else:
                if type(observable) == list:
                    return [multiply(scale_factor, asarray(dose_response_table[obs])).tolist() for obs in observable]
                else:
                    return multiply(scale_factor, asarray(dose_response_table[observable])).tolist()
        elif return_type == 'dataframe':
            if type(observable) != list:
                if dataframe_labels is None:
                    dataframe_labels == dose_species
                data_dict = {'Dose_Species': [dataframe_labels for d in range(len(doses))],
                             'Dose (pM)': [d for d in doses]}
                for t in range(len(times)):
                    data_dict.update({str(times[t]): [(dose_response_table[observable][d][t], nan) for d in range(len(doses))]})
                df = pd.DataFrame.from_dict(data_dict)
                df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
                if scale_factor == 1:
                    return df
                else:
                    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])
                    if dose_species == 'Ia':
                        dose_species = 'Alpha'
                    elif dose_species == 'Ib':
                        dose_species = 'Beta'

                    for i in range(len(times)):
                        df.loc[dose_species].iloc[:, i] = df.loc[dose_species].iloc[:, i].apply(scale_data)
                    return df
            else:
                if dataframe_labels is None:
                    dataframe_labels == dose_species
                total_df = pd.DataFrame()
                for obs in observable:
                    data_dict = {'Observable_Species': [obs for _ in range(len(doses))],
                                 'Dose_Species': [dataframe_labels for _ in range(len(doses))],
                                 'Dose (pM)': [d for d in doses]}
                    for t in range(len(times)):
                        data_dict.update({str(times[t]): [(dose_response_table[obs][d][t], nan) for d in range(len(doses))]})
                    df = pd.DataFrame.from_dict(data_dict)
                    total_df = total_df.append(df)
                total_df.set_index(['Observable_Species', 'Dose_Species', 'Dose (pM)'], inplace=True)
                if scale_factor == 1:
                    return total_df
                else:
                    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])
                    if dose_species == 'Ia':
                        dose_species = 'Alpha'
                    elif dose_species == 'Ib':
                        dose_species = 'Beta'
                    for obs in observable:
                        for i in range(len(times)):
                            total_df.loc[obs].loc[dose_species].iloc[:, i] = total_df.loc[obs].loc[dose_species].iloc[:, i].apply(scale_data)
                    return total_df


class EnsembleModel():
    """
    Documentation - An EnsembleModel object is designed to enable ensemble
    sampling of IfnModel predictions from a collection of parameter vectors
    previously generated by a MCMC method.

    Init Parameters
    ---------------
    name : string
        The name of the IfnModel model file written using PySB, including
        file extension.
    param_file : string
        The name of the file containing MCMC sampled parameter vectors, as
        saved by PyDREAM method in pydream module.
    parameter_names : list
        The string names of each of the parameters in the parameter vector

    Methods
    -------
    posterior_prediction -> IfnData
        Make predictions from the ensemble of models

    """
    def __init__(self, name, param_fname, parameter_names, prior_fname, **kwargs):
        self.name = name
        self.model = IfnModel(self.name)
        with open(prior_fname, 'rb') as f:
            self.prior_parameters = pickle.load(f)

        if param_fname[-3:] == 'npy':
            log10_parameters = load(param_fname)
            self.parameters = power(10, log10_parameters)
        elif param_fname[-3:] == 'txt':
            with open(param_fname, 'r') as f:
                pdicts = eval(f.read())
                self.parameters = np.array([list(d.values()) for d in pdicts])
        self.parameter_names = parameter_names
        # Check if any parameters were fit as distributions
        self.param_dist_flag = False
        for name in self.parameter_names:
            if name[-1] == '*':
                self.param_dist_flag = True
        self.num_dist_samples = kwargs.get('num_dist_samples', 10)

    def __posterior_prediction__(self, parameter_dict, test_times,
                                 observable, dose_species, doses,
                                 scale_factor, conditions):
        """
        Produce predictions for IFNa and IFNb using model with parameters given
        as input to the function.
        """
        # Update parameters
        if self.param_dist_flag:
            # find all distribution parameters
            dist_param_names = []
            for key in parameter_dict.keys():
                if key.endswith('_mu*'):
                    dist_param_names.append(key[:-4])
            # sample according to mu, std
            dist_param_dict = {}
            for pname in dist_param_names:
                mu = parameter_dict[pname + '_mu*']
                std = parameter_dict[pname + '_std*']
                sample = 10 ** np.random.normal(loc=np.log10(mu), scale=std)
                dist_param_dict.update({pname: sample})
                # remove distribution parameters
                parameter_dict.pop(pname + '_mu*')
                parameter_dict.pop(pname + '_std*')
            # add sample to parameter_dict
            parameter_dict.update(dist_param_dict)

        parameter_dict.update(conditions)

        # Make predictions
        if dose_species == 'Ia':
            dataframe_label = 'Alpha'
        if dose_species == 'Ib':
            dataframe_label = 'Beta'
        df = self.model.doseresponse(test_times, observable, dose_species,
                                     doses, parameters=parameter_dict,
                                     scale_factor=scale_factor,
                                     return_type='dataframe',
                                     dataframe_labels=dataframe_label)
        posterior = IfnData('custom', df=df, conditions=conditions)
        posterior.drop_sigmas()
        return posterior

    def __posterior_IFN_summary_statistics__(self, posterior_predictions, dose_species):
        """
        Encapsulates the code to compute summary statistics from a list of
        posterior predictions. Used to increase readability in the main func.

        Parameters
        ----------
        posterior_predictions : list
            a list of IfnData objects representing ensemble of predictions

        Return : float, float
            The mean prediction, standard deviation of prediction,
        """
        mean_alpha_predictions = np.mean([posterior_predictions[i].data_set.
                                         loc[dose_species].values for i in
                                         range(len(posterior_predictions))],
                                         axis=0)
        std_alpha_predictions = np.std([posterior_predictions[i].data_set.
                                       loc[dose_species].values.astype(np.float64) for
                                       i in range(len(posterior_predictions))],
                                       axis=0)

        return mean_alpha_predictions, std_alpha_predictions

    def posterior_prediction(self, test_times, observable, dose_species, doses,
                             parameters, sf=1, **kwargs):
        """
        Class method for producing ensemble model predictions
        """
        num_checks = kwargs.get('num_checks', 50)
        if dose_species == 'Ia':
            dataframe_label = 'Alpha'
        if dose_species == 'Ib':
            dataframe_label = 'Beta'

        # Prepare parameter vectors
        parameters_to_check = []
        params_list_len = len(self.parameters)
        burn_in_len = int(params_list_len / 2)
        if params_list_len - burn_in_len < num_checks:
            print("Skipping burn in due to insufficient sample size")
            indices_to_check = list(range(params_list_len))
        else:
            indices_to_check = list(np.random.randint(burn_in_len, high=params_list_len, size=num_checks))
        for i in indices_to_check:
            parameters_to_check.append(self.parameters[i])

        # Compute posterior sample trajectories
        posterior_trajectories = []
        for p in parameters_to_check:
            param_dict = {key: value for key, value in zip(self.parameter_names, p)}

            if self.param_dist_flag:
                traj_subsamples = []
                for _ in tqdm(range(self.num_dist_samples)):
                    pp = self.__posterior_prediction__(param_dict, test_times, observable, dose_species, doses, sf, parameters)
                    traj_subsamples.append(pp)
                mean_pred, _ = self.__posterior_IFN_summary_statistics__(traj_subsamples, dataframe_label)
                # Convert to IfnData object
                mean_pred_ifndata = copy.deepcopy(pp)
                for didx, d in enumerate(mean_pred_ifndata.get_doses()[dataframe_label]):
                    for tidx, t in enumerate(mean_pred_ifndata.get_times()[dataframe_label]):
                        mean_pred_ifndata.data_set.loc[dataframe_label][str(t)].loc[d] = mean_pred[didx][tidx]
                posterior_trajectories.append(copy.deepcopy(mean_pred_ifndata)) # getting paranoid about memory leaks, so deep copy
            else:
                pp = self.__posterior_prediction__(param_dict, test_times, observable, dose_species, doses, sf, parameters)
                posterior_trajectories.append(pp)

        # Make aggregate predicitions
        mean_pred, std_pred = self.__posterior_IFN_summary_statistics__(posterior_trajectories, dataframe_label)

        mean_model = copy.deepcopy(posterior_trajectories[0])
        for didx, d in enumerate(mean_model.get_doses()[dataframe_label]):
            for tidx, t in enumerate(mean_model.get_times()[dataframe_label]):
                mean_model.data_set.loc[dataframe_label][str(t)].loc[d] = (mean_pred[didx][tidx], std_pred[didx][tidx])

        return mean_model

    def get_parameters(self):
        return self.model.get_parameters()

    def set_parameters(self, new_parameters, db_check=True):
        self.model.set_parameters(new_parameters, db_check)


if __name__ == '__main__':
    testModel = IfnModel('Mixed_IFN_ppCompatible')
    tc = testModel.timecourse([0, 5, 15, 30], 'TotalpSTAT', return_type='list', dataframe_labels=['Alpha', 1])
    dr = testModel.doseresponse([0, 5, 15, 30], ['Ta', 'TotalpSTAT'], 'Ia', [1, 10, 100],
                                return_type='dataframe', dataframe_labels='Alpha')
    print(tc)
    print(dr)
