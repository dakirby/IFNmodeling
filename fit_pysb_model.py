# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 20:00:22 2018

@author: Duncan

Script for formatting experimental Interferon flow cytometry data stored in 
Experimental_Data so that it can be fairly compared to a PySB model, and then
fitting the PySB model to this data by fine tuning the parameters specified.

This script makes the following assumptions:
    data is stored in Experimental_Data
    data is time course data as measured by flow cytometry
    measurements are MFI for IFN-stimulated cells stained with anti-pSTAT antibody
    measurements made at t=0 MUST preceed all other measurements in the same time course
"""
modelfileA = "IFN_simplified_model_alpha_ppCompatible"
modelfileB = "IFN_simplified_model_beta_ppCompatible"
import pysb_parallel as pp
import Experimental_Data as ED
import os # For renaming output file to prevent overwriting results

# =============================================================================
# Parameter tests for IFN Alpha
# =============================================================================
Alpha_tests = ['kpa','kSOCSon','R1','R2']
p0_Alpha=[[1E-6,1E-9,1E-3,'log'],
          [1E-6,1E-9,1E-3,'log'],
          [2000,100,9000,'linear'],
          [2000,100,9000,'linear']]

# =============================================================================
# Parameter tests for IFN Beta
# =============================================================================
Beta_tests = ['kpa','kSOCSon','R1','R2']
p0_Beta=[[1E-6,1E-9,1E-3,'log'],
          [1E-6,1E-9,1E-3,'log'],
          [2000,100,9000,'linear'],
          [2000,100,9000,'linear']]



def main():
    if os.path.isfile('modelfit_alpha.txt') or os.path.isfile('modelfit_beta.txt'):
        print("Cannot overwrite previous model fit. Remove modelfit_xxx.txt and try re-running.")
        return 1
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
    # Reformat:
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
        zero = 0
        for col in range(len(col_labels)):
            if col_labels[col]=='IFN':
                IFNconcentration = float(tc_data[col])
            elif col_labels[col]=='type':
                IFNtype = tc_data[col]
            elif col_labels[col]=='time':
                if IFNtype=='Alpha':
                    if int(table.columns.values[col]) == 0:
                        zero = tc_data[col]
                    ydata_Alpha.append(tc_data[col]-zero)
                    xdata_Alpha.append([['I',NA*volEC*1E-12*IFNconcentration], ['time', int(table.columns.values[col])*60]])
                if IFNtype=='Beta':
                    if int(table.columns.values[col]) == 0:
                        zero = tc_data[col]                    
                    ydata_Beta.append(tc_data[col]-zero)
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
    pp.fit_model(modelfileA, xdata_Alpha, ['TotalpSTAT',ydata_Alpha], Alpha_tests,
                     p0=p0_Alpha, sigma=Alpha_uncertainty, n=15000)
    os.rename('modelfit.txt', 'modelfit_alpha.txt')
    pp.fit_model(modelfileB, xdata_Beta, ['TotalpSTAT',ydata_Beta], Beta_tests,
                     p0=p0_Beta, sigma=Beta_uncertainty, n=15000)
    os.rename('modelfit.txt', 'modelfit_beta.txt')
    
if __name__ == '__main__':
    main()

