from pysb.export import export
from collections import OrderedDict
import copy
from numpy import multiply, zeros, nan
import pandas as pd
import pickle


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
            with open("ODE_system.py", 'w') as f:
                f.write(py_output)
            import ODE_system
            model_obj = ODE_system.Model()
            return model_obj

    def save_model(self, name):
        with open(name, 'wb') as f:
            pickle.dump(self.__dict__, f,2)

    def load_model(self, name):
        try:
            with open('ifnmodels/'+name, 'rb') as f:
                tmp_dict = pickle.load(f)
            self.__dict__.update(tmp_dict)
        except FileNotFoundError:
            with open(name, 'rb') as f:
                tmp_dict = pickle.load(f)
            self.__dict__.update(tmp_dict)

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
        # First we check if any parameters which must maintain detailed balance are even in new_parameters
        db_parameters_present = 'kd4' in new_parameters.keys() or 'kd3' in new_parameters.keys() or 'k_d4' in new_parameters.keys() or 'k_d3' in new_parameters.keys()
        # Second we check if new_parameters is trying to over-constrain the model, violating detailed balance
        no_db_parameters_conflict = (not ('kd4' in new_parameters.keys() and 'kd3' in new_parameters.keys())) or (
            not ('k_d4' in new_parameters.keys() and 'k_d3' in new_parameters.keys()))
        return no_db_parameters_conflict and db_parameters_present

    def __check_detailed_balance__(self):
        if 'kd4' in self.parameters.keys():
            q1 = self.parameters['ka1'] / self.parameters['kd1']
            q2 = self.parameters['ka2'] / self.parameters['kd2']
            q3 = self.parameters['ka3'] / self.parameters['kd3']
            q4 = self.parameters['ka4'] / self.parameters['kd4']
            return q1 * q3 == q2 * q4
        elif 'k_d4' in self.parameters.keys():
            q1 = self.parameters['k_a1'] / self.parameters['k_d1']
            q2 = self.parameters['k_a2'] / self.parameters['k_d2']
            q3 = self.parameters['k_a3'] / self.parameters['k_d3']
            q4 = self.parameters['k_a4'] / self.parameters['k_d4']
            return q1 * q3 == q2 * q4
        else:
            print("Could not find detailed balance parameters")
            return False

    def set_parameters(self, new_parameters: dict):
        if self.check_if_parameters_in_model(new_parameters):
            if self.check_for_detailed_balance_parameters(new_parameters):
                if 'kd4' in new_parameters.keys():
                    q1 = self.parameters['ka1'] / self.parameters['kd1']
                    q2 = self.parameters['ka2'] / self.parameters['kd2']
                    q4 = self.parameters['ka4'] / new_parameters['kd4']
                    q3 = q2 * q4 / q1
                    kd3 = self.parameters['ka3'] / q3
                    new_parameters.update({'kd3': kd3})
                if 'kd3' in new_parameters.keys():
                    q1 = self.parameters['ka1'] / self.parameters['kd1']
                    q2 = self.parameters['ka2'] / self.parameters['kd2']
                    q3 = self.parameters['ka3'] / new_parameters['kd3']
                    q4 = q1 * q3 / q2
                    kd4 = self.parameters['ka4'] / q4
                    new_parameters.update({'kd4': kd4})
                if 'k_d4' in new_parameters.keys():
                    q1 = self.parameters['k_a1'] / self.parameters['k_d1']
                    q2 = self.parameters['k_a2'] / self.parameters['k_d2']
                    q4 = self.parameters['k_a4'] / new_parameters['k_d4']
                    q3 = q2 * q4 / q1
                    k_d3 = self.parameters['k_a3'] / q3
                    new_parameters.update({'k_d3': k_d3})
                if 'k_d3' in new_parameters.keys():
                    q1 = self.parameters['k_a1'] / self.parameters['k_d1']
                    q2 = self.parameters['k_a2'] / self.parameters['k_d2']
                    q3 = self.parameters['k_a3'] / new_parameters['k_d3']
                    q4 = q1 * q3 / q2
                    k_d4 = self.parameters['k_a4'] / q4
                    new_parameters.update({'k_d4': k_d4})
            self.parameters.update(new_parameters)
            return 0
        else:
            print("Some of the parameters were not found in the model. Did not update parameters.")
            return 1

    def reset_parameters(self):
        self.parameters = copy.deepcopy(self.default_parameters)

    def timecourse(self, times, observable, parameters=None, return_type='list', dataframe_labels=[]):
        # Keep current parameter state
        initial_parameters = copy.deepcopy(self.parameters)
        # Substitute in new simulation-specific values
        if parameters is not None:
            self.set_parameters(parameters)
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
                return {obs: sim[obs] for obs in observable}
            else:
                return {observable: sim[observable]}
        elif return_type == 'dataframe':
            # Check if mandatory labels also provided; return as list if not provided
            if dataframe_labels == []:
                print("Insufficient labels provided for dataframe return type")
                if type(observable) == list:
                    return {obs: sim[obs] for obs in observable}
                else:
                    return {observable: sim[observable]}
            # Build dataframe
            if type(observable) == list:
                rows = [[obs, dataframe_labels[0], dataframe_labels[1]] + [(val, nan) for val in sim[obs]] for obs in observable]
                dataframe = pd.DataFrame.from_records(rows, columns=['Observable', 'Dose_Species', 'Dose (pM)']+times)
                dataframe.set_index(['Observable', 'Dose_Species', 'Dose (pM)'], inplace=True)
                return dataframe
            else:
                row = [(dataframe_labels[0], dataframe_labels[1], *[(val, nan) for val in sim[observable]])]
                dataframe = pd.DataFrame.from_records(row, columns=['Dose_Species', 'Dose (pM)']+times)
                dataframe.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
                return dataframe

    def doseresponse(self, times, observable, dose_species, doses, parameters={}, return_type='list', dataframe_labels=None):
        # create dose_response_table dictionary
        if type(observable) == list:
            dose_response_table = {obs: zeros((len(doses), len(times))) for obs in observable}
        else:
            dose_response_table = {observable: zeros((len(doses), len(times)))}
        # prepare custom parameters dictionary
        dose_parameters = copy.deepcopy(self.parameters)
        dose_parameters.update(parameters)
        # iterate through all doses
        for idx, d in enumerate(doses):
            dose_parameters.update({dose_species: d*1E-12*6.022E23*1E-5})
            trajectories = self.timecourse(times, observable, parameters=dose_parameters)
            # add results to dose_response_table dictionary
            for observable_species in trajectories.keys():
                dose_response_table[observable_species][idx] = trajectories[observable_species]
        # return dose response curves
        if return_type == 'list':
            return dose_response_table
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
                return total_df

if __name__ == '__main__':
    testModel = IfnModel('Mixed_IFN_ppCompatible')
    tc = testModel.timecourse([0, 5, 15, 30], 'TotalpSTAT', return_type='list', dataframe_labels=['Alpha', 1])
    dr = testModel.doseresponse([0, 5, 15, 30], ['Ta', 'TotalpSTAT'], 'Ia', [1, 10, 100],
                                return_type='dataframe', dataframe_labels='Alpha')
    print(tc)
    print(dr)
