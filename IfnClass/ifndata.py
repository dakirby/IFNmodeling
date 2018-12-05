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
    def __init__(self, name):
        self.name = name
        self.data_set = self.load_data()
        self.conditions = self.load_conditions()

    # Instance methods
    def load_data(self):
        cwd = os.getcwd()
        parent_wd = cwd.split("IFNmodeling")[0] + "IFNmodeling\\"
        # attempt loading DataFrame object
        try:
            return pickle.load(open(parent_wd + "ifndatabase\{}.p".format(self.name), 'rb'))
        except FileNotFoundError:
            # Attempt initializing module and then importing DataFrame object
            try:
                print("Trying to build data sets")
                ifndatabase.process_csv.build_database(parent_wd + "ifndatabase\\")
                return pickle.load(open(parent_wd + "ifndatabase\{}.p".format(self.name), 'rb'))
            except FileNotFoundError:
                # Attempt loading a local DataFrame object
                try:
                    return pickle.load(open("{}.p".format(self.name), 'rb'))
                except FileNotFoundError:
                    raise FileNotFoundError("Could not find the data file specified")

    def load_conditions(self):
        cwd = os.getcwd()
        parent_wd = cwd.split("IFNmodeling")[0] + "IFNmodeling\\"
        # attempt loading DataFrame object
        try:
            with open(parent_wd + "ifndatabase\{}.txt".format(self.name), 'r') as inf:
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
        return dict(zip(keys, [[int(el) for el in self.data_set.loc[key].columns.get_values().tolist()] for key in keys]))

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
            t = [str(n) for n in t]
            datatable.update({key: self.data_set.loc[key][t].values})
        return datatable
