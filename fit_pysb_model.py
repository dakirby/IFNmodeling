# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 20:00:22 2018

@author: Duncan

Script for formatting experimental data and fitting the PySB model
"""
modelfileA = "IFN_simplified_model_alpha_ppCompatible"
modelfileB = "IFN_simplified_model_beta_ppCompatible"
import pysb_parallel as pp
import Experimental_Data as ED
import os # For renaming output file to prevent overwriting results

def main():
    NA = 6.022E23
    volEC = 1E-5    
    # Read in data
    # Format the data so that each measurement is stored in a list
    #   and all the other important quantities are stored in a 
    #   a list which is ordered corresponding to each data point
    #   ie. ydata = [observation_1, observation_2,...]
    #       xdata = [[['column name', column value], ['column name', column value],...] ,
    #                [[],[],[]], 
    #                ...for each data point in ydata]

    # Assume the data is time course data. Dose-response data reading can be added later.
    table = ED.data
    col_labels = [l for l in table.columns.values]
    for label in range(len(col_labels)):
        if col_labels[label] == 'Dose (pM)':
            col_labels[label] = 'IFN'
        elif col_labels[label] == 'Interferon':
            col_labels[label] = 'type'
        else:
            try: 
                int(col_labels[label])
                col_labels[label] = 'time'
            except ValueError:
                print("Unsure what this label is")
                print(col_labels[label])
                return 1
    xdata_Alpha=[]
    ydata_Alpha=[]
    xdata_Beta=[]
    ydata_Beta=[]
    for r in range(len(table)):
        tc_data = table.iloc[r]
        IFNconcentration=0
        IFNtype=''
        for col in range(len(col_labels)):
            if col_labels[col]=='IFN':
                IFNconcentration = float(tc_data[col])
            elif col_labels[col]=='type':
                IFNtype = tc_data[col]
            elif col_labels[col]=='time':
                if IFNtype=='Alpha':
                    ydata_Alpha.append(tc_data[col])
                    xdata_Alpha.append([['I',NA*volEC*1E-12*IFNconcentration], ['time', int(table.columns.values[col])*60]])
                if IFNtype=='Beta':
                    ydata_Beta.append(tc_data[col])
                    xdata_Beta.append([['I',NA*volEC*1E-12*IFNconcentration], ['time', int(table.columns.values[col])*60]])
                if IFNtype=='Alpha_std':
                    for exp in xdata_Alpha:
                        if exp == [['I',NA*volEC*1E-12*IFNconcentration], ['time', int(table.columns.values[col])*60]]:
                            exp.append(['sigma',tc_data[col]])
                            break
                if IFNtype=='Beta_std':
                    for exp in xdata_Beta:
                        if exp == [['I',NA*volEC*1E-12*IFNconcentration], ['time', int(table.columns.values[col])*60]]:
                            exp.append(['sigma',tc_data[col]])
                            break
    # Fill in any un-provided uncertainties with ones
    Alpha_uncertainty = []
    Beta_uncertainty = []
    for x in range(len(xdata_Alpha)):
        provided=False
        for c in reversed(range(len(xdata_Alpha[x]))):
            if xdata_Alpha[x][c][0]=='sigma':
                provided=True
                Alpha_uncertainty.append(xdata_Alpha[x][c][1])                
                xdata_Alpha[x] = xdata_Alpha[x][0:c]+xdata_Alpha[x][c+1:len(xdata_Alpha[x])]
                break
        if provided == False:
            Alpha_uncertainty.append(1)
    for x in range(len(xdata_Beta)):
        provided=False
        for c in reversed(range(len(xdata_Beta[x]))):
            if xdata_Beta[x][c][0]=='sigma':
                provided=True
                Beta_uncertainty.append(xdata_Beta[x][c][1])                
                xdata_Beta[x] = xdata_Beta[x][0:c]+xdata_Beta[x][c+1:len(xdata_Beta[x])]
                break
        if provided == False:
            Beta_uncertainty.append(1)
    # Now fit the models
    pp.fit_model(modelfileA, xdata_Alpha, ['TotalpSTAT',ydata_Alpha], ['kpa','kSOCSon'],
                     p0=[1E-6,1E-6], sigma=Alpha_uncertainty)
    os.rename('modelfit.txt', 'modelfit_alpha.txt')
    pp.fit_model(modelfileB, xdata_Beta, ['TotalpSTAT',ydata_Beta], ['kpa','kSOCSon'],
                     p0=[1E-6,1E-6], sigma=Beta_uncertainty)
    os.rename('modelfit.txt', 'modelfit_beta.txt')
    
if __name__ == '__main__':
    main()

