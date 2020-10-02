from pysb.export import export
from collections import OrderedDict
import copy
from numpy import multiply, zeros, asarray
import pandas as pd
import time
import os
from random import randint


class IfnModel:
    """
    Documentation - An IfnModel object is the primary object for modelling
    experimental IFN dose-response or timecourse data.

    Attributes
    -------
    name (str): The filename used to find source files for this class instance
    model (object): An instance of a PySB standalone python model
    parameters (ordered dict): A dictionary with all the Model parameters.
                               The dictionary is ordered to match the order
                               listed in the PySB model source code.

    ****************************************************************************
    Methods
    ****************************************************************************

    __init__()
    Inputs:
        name (str): the name of the PySB model file to use. This file should be
                    located in the same directory as model_class_file.py
    Outputs:
        None

    ----------------------------------------------------------------------------
    set_parameters()
        Inputs:
            new_parameters (dict): dictionary of parameters to update the model
                                   with. Keys must match the parameter names as
                                   they appear in the PySB model file
            db_check (Bool): flag for checking if detailed balance is maintained
                             in the new model parameters (default is True)
        Outputs:
            returns 0 if successful and returns 1 otherwise

    ----------------------------------------------------------------------------
    timecourse()
        Inputs:
            times (list) : time points to output, in minutes
            observable (string or list) : name of model observable(s) to return
            parameters (dict) : parameters for simulation; default is the
                                current parameters stored in self.parameters
            return_type = 'dict' or 'dataframe':
                         indicates whether you want the trajectory returned as
                         a dict or Pandas DataFrame
                         If 'dataframe', must also input `dataframe_labels`
            dataframe_labels (list): REQUIRED FOR  return_type='dataframe';
                                     contains values for the DataFrame labels:
                                    [dose species (string), dose in pM (float)]
            scale_factor (float): a global multiplicative scale factor to apply
                                  to all values of observables at all time
                                  points. This is especially useful when
                                  comparing absolute numbers of molecules from
                                  simulation to experimental measurements of
                                  fluorescence intensity.
                                  (default scale_factor is 1)

        Outputs:
        If return_type is 'dict':
            the returned object is a dictionary with entries of the form
           {observable name: list of observable values at each time in `times`}
        If return_type is 'dataframe':
           the returned object is a Pandas dataframe Multiindex object with
           indices 'Dose_Species' and 'Dose (pM)', and column labels are the
           values from `times`

           An example of the format output is give below:

           *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
                                             0            5           15           30
Observable_Species Dose_Species Dose (pM)
Ta                 Alpha        1          0.0     0.177913     0.009665     0.003353
                                10         0.0     1.773732     0.096694     0.033541
                                100        0.0    17.218972     0.971300     0.336169
TotalpSTAT         Alpha        1          0.0  2524.623425  3775.389077  2713.805462
                                10         0.0  2524.618428  3774.935237  2713.686892
                                100        0.0  2524.494148  3770.499037  2712.518404
            *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

    ----------------------------------------------------------------------------
    doseresponse(self, times, observable, dose_species, doses,
                     parameters={}, return_type='dict', dataframe_labels=None,
                     scale_factor=1)
            Inputs:
                times (list) : time points to output, in minutes
                observable (string or list) : name of model observable(s) to
                                              simulate
                dose_species (string) : name of the PySB model Observable to be
                                        modified during dose response
                doses (list) : list of values to substitute for dose_species in
                               model in simulating a dose-response curve.
                               Values should be given in picomolar, i.e.
                               `
                               doses=[12, 24, 36]
                               dose_species='Ia'
                               `
                               would simulate time courses for species 'Ia' at
                               12 pM, then at 24 pM, then at 36 pM.
                parameters (dict) : parameters for simulation; default is the
                                    current parameters stored in self.parameters
                return_type = 'dict' or 'dataframe':
                             indicates whether you want the trajectory returned as
                             a dict or Pandas DataFrame
                             If 'dataframe', must also input `dataframe_labels`
                dataframe_labels (list): REQUIRED FOR  return_type='dataframe';
                                         contains values for the DataFrame labels:
                                        [dose species (string), dose in pM (float)]
                scale_factor (float): a global multiplicative scale factor to apply
                                      to all values of observables at all time
                                      points. This is especially useful when
                                      comparing absolute numbers of molecules from
                                      simulation to experimental measurements of
                                      fluorescence intensity.
                                      (default scale_factor is 1)
    """

    # Initializer / Instance Attributes
    def __init__(self, name):
        self.name = name
        self.model = self.__build_model__(self.name)
        self.parameters = self.__build_parameters__(self.model)
        self.default_parameters = copy.deepcopy(self.parameters)

    # Instance methods
    def __build_model__(self, name):
        if name == '':
            return None
        else:
            model_code = __import__(name)
            py_output = export(model_code.model, 'python')
            ODE_filename = "ODE_system_{}_{}.py".format(
                            time.strftime("%Y%m%d-%H%M%S"),
                            randint(100000, 999999))
            with open(ODE_filename, 'w') as f:
                f.write(py_output)
            ODE_system = __import__(ODE_filename[:-3])
            model_obj = ODE_system.Model()
            os.remove(ODE_filename)
            return model_obj

    def __build_parameters__(self, pysb_model):
        if pysb_model is not None:
            parameter_dict = OrderedDict({})
            for p in pysb_model.parameters:
                parameter_dict.update({p[0]: p[1]})
            return parameter_dict
        else:
            return {}

    def __check_if_parameters_in_model__(self, test_parameters):
        list1 = [element for element in test_parameters.keys() if element in
                 self.parameters.keys()]
        return list1 == list(test_parameters.keys())

    def __check_for_detailed_balance_parameters__(self, new_parameters):
        db_parameters_present = False
        for key in ['ka1', 'kd1', 'ka2', 'kd2', 'ka3', 'kd3', 'ka4', 'kd4',
                    'k_a1', 'k_d1', 'k_a2', 'k_d2', 'k_a3', 'k_d3', 'k_a4',
                    'k_d4']:
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
            alpha_check = (abs(q1 * q3 / (q2 * q4) - 1) < 1E-4)
        if 'k_d4' in self.parameters.keys():
            q1 = self.parameters['k_a1'] / self.parameters['k_d1']
            q2 = self.parameters['k_a2'] / self.parameters['k_d2']
            q3 = self.parameters['k_a3'] / self.parameters['k_d3']
            q4 = self.parameters['k_a4'] / self.parameters['k_d4']
            beta_check = (abs(q1 * q3 / (q2 * q4) - 1) < 1E-4)
        if ('kd4' in self.parameters.keys()) or ('k_d4' in self.parameters.keys()):
            return (alpha_check and beta_check)
        else:
            print("Could not find detailed balance parameters")
            return False

    def set_parameters(self, new_parameters: dict, db_check=True):
        if self.__check_if_parameters_in_model__(new_parameters):
            if self.__check_for_detailed_balance_parameters__(new_parameters) and db_check:
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

    def timecourse(self, times, observable, parameters=None,
                   return_type='dict', dataframe_labels=[], scale_factor=1):
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
        (_, sim) = self.model.simulate(multiply(times, 60).tolist(),
                                       param_values=list(self.parameters.values()))
        if noTflag:
            sim = sim[1:]

        # Reset model parameter vector so that timecourse() is not permanently
        # altering model parameters:
        self.parameters = initial_parameters

        if return_type == 'dataframe' and dataframe_labels == []:
            print("Insufficient labels provided for dataframe return type")
            return_type = 'dict'

        # Return trajectory(ies)
        if return_type == 'dict':
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
            # Build dataframe
            if type(observable) == list:
                rows = [[obs, dataframe_labels[0], dataframe_labels[1]] + sim[obs] for obs in observable]
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
                row = [(dataframe_labels[0], dataframe_labels[1], *sim[observable])]
                dataframe = pd.DataFrame.from_records(row, columns=['Dose_Species', 'Dose (pM)']+times)
                dataframe.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
                if scale_factor == 1:
                    return dataframe
                else:
                    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])
                    dataframe.loc[dataframe_labels[0]].iloc[:, i] = dataframe.loc[dataframe_labels[0]].iloc[:, i].apply(scale_data)
                    return dataframe

    def doseresponse(self, times, observable, dose_species, doses,
                     parameters={}, return_type='dict', dataframe_labels=None,
                     scale_factor=1):
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
        if return_type == 'dict':
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
                    data_dict.update({str(times[t]): [dose_response_table[observable][d][t] for d in range(len(doses))]})
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
                        data_dict.update({str(times[t]): [dose_response_table[obs][d][t] for d in range(len(doses))]})
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
