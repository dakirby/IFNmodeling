# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 12:30:11 2018

@author: Duncan

Build pandas DataFrame objects from csv files
"""
import pandas as pd
dataset_1 = pd.read_csv("Experimental_Data.csv")
print(dataset_1.loc[:,'Dose (pM)']==10)
print(dataset_1.loc[(dataset_1.loc[:,'Dose (pM)']==10) & (dataset_1.loc[:,'Interferon']=="Alpha"),['0','5','15','30','60']])

