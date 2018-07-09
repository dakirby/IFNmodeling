# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 08:35:40 2018

@author: Duncan

Experimental data set
"""
import pandas as pd
data = pd.read_csv("Experimental_Data.csv")
print(data.to_string())