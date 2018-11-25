# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 10:05:14 2018

@author: Duncan

IfnData is the standardized python object for IFN data sets, used for fitting 
and plotting data.
"""
import pandas as pd
import os

class IfnData:
    """
    Documentation - An IfnData object is a standardized object for holding 
    experimental IFN dose-response or timecourse data. This is the expected 
    data object used for plotting and fitting within the IFNmodeling module.

    Parameters
    ----------
    name : string
        The name of the csv file containing the data. Data must be in the form:
            'Dose (pM)', dose species name (string), dose1, dose2, ...
    Attributes
    -------
    get_dose_range : tuple = (min_dose, max_dose)
        min_dose = the minimum dose used in the experiment
        max_dose = the maximum dose used in the experiment
    
    """
    # Class Attribute
    #species = 'mammal'
                    
    # Initializer / Instance Attributes
    def __init__(self, name):
        self.name = name
        self.data_set = self.load_data()

    # Instance methods
    
    def load_data(self):
        cwd = os.getcwd()
        parent_wd = cwd.split("IFNmodeling")[0]+"IFNmodeling\\"
        try:
            return pd.read_csv(parent_wd+"IFNdata\{}.csv".format(self.name))
        except FileNotFoundError:
            try:
                return pd.read_csv("{}.csv".format(self.name))
            except FileNotFoundError:
                raise FileNotFoundError("Could not find the data file specified")

    def get_dose_range(self):
        print(self.data_set.loc[self.data_set.columns)
        

                                        
testData = IfnData("Experimental_Data")    
testData.get_dose_range()

                                                            