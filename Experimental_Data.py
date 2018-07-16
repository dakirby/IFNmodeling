# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 08:35:40 2018

@author: Duncan

Experimental data set
"""
import pandas as pd
try:
    data = pd.read_csv("IFNmodeling\Experimental_Data.csv")
except FileNotFoundError:
    try:
        data = pd.read_csv("Experimental_Data.csv")
    except:
        raise
#print(data)
#print(data.loc[(data.loc[:,'Dose (pM)']==10) & (data.loc[:,'Interferon']=="Alpha"),:])
#print(data.loc[(data.loc[:,'Dose (pM)']==10) & (data.loc[:,'Interferon']=="Alpha"),'0':'60'])
