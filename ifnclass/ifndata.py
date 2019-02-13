"""
Created on Sun Nov 25 10:05:14 2018

@author: Duncan

IfnData is the standardized python object for IFN data sets, used for fitting 
and plotting data.
"""
import os
import pickle
import ast
import ifndatabase.process_csv
from scipy.optimize import curve_fit
import numpy as np


class IfnData:
    """
    Documentation - An IfnData object is a standardized object for holding 
    experimental IFN dose-response or timecourse data. This is the expected 
    data object used for plotting and fitting within the IFNmodeling module.

    The standard column labels are as follows:


    Parameters
    ----------
    name : string
        The name of the pandas DataFrame pickled object containing the data
    Attributes
    ----------
    name : string
        The filename used to find source files for this IfnData instance
    data_set : DataFrame
        The experimental data
    conditions : dict
        A dictionary with keys corresponding to controlled experimental 
        parameters and values at which the experiments were performed
    Methods
    -------
    get_dose_range -> tuple = (min_dose, max_dose)
        min_dose = the minimum dose used in the entire experiment
        max_dose = the maximum dose used in the entire experiment
    
    """

    # Initializer / Instance Attributes
    def __init__(self, name, df=None, conditions=None):
        if name == 'custom':
            self.name = None
            self.data_set = df
            self.conditions = conditions
        else:
            self.name = name
            self.data_set = self.load_data()
            self.conditions = self.load_conditions()

    # Instance methods
    def load_data(self):
        cwd = os.getcwd()
        parent_wd = cwd.split("IFNmodeling")[0] + "IFNmodeling"
        # attempt loading DataFrame object
        try:
            return pickle.load(open(os.path.join(parent_wd, "ifndatabase","{}.p".format(self.name)), 'rb'))
        except FileNotFoundError:
            # Attempt initializing module and then importing DataFrame object
            try:
                print("Trying to build data sets")
                ifndatabase.process_csv.build_database(os.path.join(parent_wd, "ifndatabase"))
                return pickle.load(open(os.path.join(parent_wd, "ifndatabase", "{}.p".format(self.name)), 'rb'))
            except FileNotFoundError:
                # Attempt loading a local DataFrame object
                try:
                    return pickle.load(open("{}.p".format(self.name), 'rb'))
                except FileNotFoundError:
                    raise FileNotFoundError("Could not find the data file specified")

    def load_conditions(self):
        cwd = os.getcwd()
        parent_wd = cwd.split("IFNmodeling")[0] + "IFNmodeling"
        # attempt loading DataFrame object
        try:
            with open(os.path.join(parent_wd, "ifndatabase", "{}.txt".format(self.name)), 'r') as inf:
                return ast.literal_eval(inf.read())
        except FileNotFoundError:
            # Attempt loading a local conditions file if none found in data dir
            try:
                with open("{}.txt".format(self.name), 'r') as inf:
                    return ast.literal_eval(inf.read())
            # Return default None if no experimental conditions provided
            except FileNotFoundError:
                return None

    def get_dose_species(self) -> list:
        return list(self.data_set.index.levels[0])

    def get_times(self) -> dict:
        keys = self.get_dose_species()
        if type(self.data_set.loc[keys[0]].columns.get_values().tolist()) == str:
            return dict(zip(keys, [[int(el) for el in self.data_set.loc[key].columns.get_values().tolist()] for key in keys]))
        else:
            return dict(zip(keys, [[el for el in self.data_set.loc[key].columns.get_values().tolist()] for key in keys]))

    def get_doses(self) -> dict:
        keys = self.get_dose_species()

        dose_spec_names = [dose_species for dose_species, dose_species_data in
                           self.data_set.groupby(level='Dose_Species')]
        dose_list = [list(self.data_set.loc[spec].index) for spec in dose_spec_names]
        return dict(zip(keys, dose_list))

    def get_responses(self) -> dict:
        datatable = {}
        times = self.get_times()
        for key, t in times.items():
            if str(t[0]) in self.data_set.loc[key].index:
                t = [str(n) for n in t]
                datatable.update({key: self.data_set.loc[key][t].values})
            else:
                datatable.update({key: self.data_set.loc[key][t].values})
        return datatable

    def __MM__(self, xdata, top, n, k):
        ydata = [top * x ** n / (k ** n + x ** n) for x in xdata]
        return ydata

    def get_ec50s(self, hill_coeff_guess = 0.1):
        def augment_data(x_data, y_data):
            new_xdata = [x_data[0]*0.1, x_data[0]*0.3, x_data[0]*0.8, *x_data, x_data[-1]*2, x_data[-1]*5, x_data[-1]*8]
            new_ydata = [y_data[0], y_data[0], y_data[0], *y_data, y_data[-1], y_data[-1], y_data[-1]]
            return new_xdata, new_ydata
        ec50_dict = {}
        for key in self.get_dose_species():
            response_array = np.transpose([[el[0] for el in row] for row in self.get_responses()[key]])
            ec50_array = []
            for t in enumerate(self.get_times()[key]):
                doses, responses = augment_data(self.get_doses()[key][1:],  response_array[t[0]][1:])
                results, covariance = curve_fit(self.__MM__, doses, responses,
                                                p0=[max(responses), hill_coeff_guess, doses[int(len(doses)/2)]])
                ec50_array.append((t[1], results[2]))
            ec50_dict[key] = ec50_array
        return ec50_dict

    def get_max_responses(self):
        Tmax_dict = {}
        for key in self.get_dose_species():
            response_array = np.transpose([[el[0] for el in row] for row in self.get_responses()[key]])
            Tmax_array = []
            for t in enumerate(self.get_times()[key]):
                Tmax_array.append(max(response_array[t[0]]))
            Tmax_dict[key] = Tmax_array
        return Tmax_dict
