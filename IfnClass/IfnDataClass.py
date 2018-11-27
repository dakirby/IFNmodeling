# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 10:05:14 2018

@author: Duncan

IfnData is the standardized python object for IFN data sets, used for fitting 
and plotting data.
"""
import pandas as pd
import numpy as np
import os
import pickle
import ast

cwd = os.getcwd()
parent_wd = cwd.split("IFNmodeling")[0]+"IFNmodeling\\"
#sys.path.append(parent_wd)
import IfnData

class IfnData:
    """
    Documentation - An IfnData object is a standardized object for holding 
    experimental IFN dose-response or timecourse data. This is the expected 
    data object used for plotting and fitting within the IFNmodeling module.

    Parameters
    ----------
    name : string
        The name of the pandas DataFrame pickled object containing the data
    Attributes
    -------
    name : string
        The filename used to find source files for this IfnData instance
    data_set : DataFrame
        The experimental data
    conditions : dict
        A dictionary with keys corresponding to controlled experimental 
        parameters and values at which the experiments were performed
    Methods
    -------
    get_dose_range : tuple = (min_dose, max_dose)
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
        parent_wd = cwd.split("IFNmodeling")[0]+"IFNmodeling\\"
        # attempt loading DataFrame object
        try:
            return pickle.load(open(parent_wd+"IfnData\{}.p".format(self.name), 'rb'))
        except FileNotFoundError:
            # Attempt initializing module and then importing DataFrame object
            try:
                from IfnData.process_csv import build_DataFrames
                print("Trying to build data sets")
                build_DataFrames(parent_wd+"IfnData\\")
                return pickle.load(open(parent_wd+"IfnData\{}.p".format(self.name), 'rb'))
            except FileNotFoundError:
                # Attempt loading a local DataFrame object
                try:
                    return pickle.load(open("{}.p".format(self.name), 'rb'))
                except FileNotFoundError:
                    raise FileNotFoundError("Could not find the data file specified")
    def load_conditions(self):
        cwd = os.getcwd()
        parent_wd = cwd.split("IFNmodeling")[0]+"IFNmodeling\\"
        # attempt loading DataFrame object
        try:
            with open(parent_wd+"IfnData\{}.txt".format(self.name),'r') as inf:
                return ast.literal_eval(inf.read())
        except FileNotFoundError:
            # Attempt loading a local conditions file if none found in data dir
            try:
                with open("{}.txt".format(self.name),'r') as inf:
                    return ast.literal_eval(inf.read())
            # Return default None if no experimental conditions provided
            except FileNotFoundError:
                return None
        
    def get_dose_range(self):
        dose_spec_names = [dose_species for dose_species, dose_species_data in self.data_set.groupby(level='Dose_Species')]
        dose_list = [list(self.data_set.loc[spec].index) for spec in dose_spec_names]
        return (np.min(dose_list),np.max(dose_list))


def load_Experimental_Data():
    return IfnData("Experimental_Data")

testData = IfnData("Experimental_Data")    
                                                            