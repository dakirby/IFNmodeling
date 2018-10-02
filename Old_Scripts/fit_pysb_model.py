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
from operator import itemgetter
import numpy as np

# =============================================================================
# Parameter tests for IFN Alpha
# =============================================================================
Alpha_tests = ['kd4','gamma']#'kpa','kSOCSon','kSOCS','R1','R2',
p0_Alpha=[[0.3,0.03,3,'log'],
          [20,5,30,'linear']]
# =============================================================================
# [1E-6,1E-8,1E-4,'log'],
#           [1E-6,1E-8,1E-4,'log'],
#           [4E-3,4E-4,4E-2,'linear'],
#           [2000,500,9000,'linear'],
#           [2000,500,9000,'linear'],
# =============================================================================
          
# =============================================================================
# Parameter tests for IFN Beta
# =============================================================================
Beta_tests = ['k_d4','gamma']#'kpa','kSOCSon','kSOCS','R1','R2',
p0_Beta=[[0.006,0.0006,0.06,'log'],
          [20,5,30,'linear']]
# =============================================================================
# [1E-6,1E-8,1E-4,'log'],
#           [1E-6,1E-8,1E-4,'log'],
#           [4E-3,4E-4,4E-2,'linear'],
#           [2000,500,9000,'linear'],
#           [2000,500,9000,'linear'],
# =============================================================================
          
# =============================================================================
# Global gamma (default is 1)
# =============================================================================
gamma = 1
# =============================================================================
# Script to automate combining alpha and beta models
# =============================================================================
def fit_alpha_and_beta(x=0):
    modelbase = []
    # Read in all models and their scores
    linecount=0
    with open('modelfit_alpha.txt', 'r') as df:
        df.readline()
        df.readline()
        header = df.readline()
        labels = header.split()
        if x==1: #parse for K4
            k4_index = labels.index('kd4')
            labels[k4_index]='K4 factor'
        while True:
            line = df.readline().split()
            if not line: break
            linecount+=1
            score = float(line[-1])
            key = [[labels[i], line[i]] for i in range(len(line)-1)]
            # Re-express kd4 as a ratio over initial guess; assumes intervals were symmetric for kd4 and k_d4            
            key[k4_index][1]=round(float(key[k4_index][1])/p0_Alpha[Alpha_tests.index('kd4')][0],2)
            modelbase.append([key,score])
    print("Summing {} scores".format(linecount))
    # Now add the scores for the fit to IFN beta data to each model   
    progress=0
    threshold=0.05
    mismatches=0
    with open('modelfit_beta.txt', 'r') as df:
        df.readline()
        df.readline()
        labels = df.readline().split()
        if x==1: #parse for K4
            k4_index = labels.index('k_d4')
            labels[k4_index]='K4 factor'
        while True:
            line = df.readline().split()
            if not line: break
            progress += 1
            if progress/linecount > threshold:
                threshold += 0.05
                print("{0:.1f}% done".format(progress/linecount*100.))
            score = float(line[-1])
            key = [[labels[i], line[i]] for i in range(len(line)-1)]
            # Re-express kd4 as a ratio over initial guess; assumes intervals were symmetric for kd4 and k_d4            
            key[k4_index][1]=round(float(key[k4_index][1])/p0_Beta[Beta_tests.index('k_d4')][0],2)
            found = False
            for model in modelbase:
                if key == model[0]:
                    found=True
                    model[1]+=score
                    break
            if found == False:
                print(key)
                mismatches+=1
                
    # Rank the models based on their new scores
    print("Re-ranking models")
    modelbase = sorted(modelbase, key = itemgetter(1))
    with open('modelfit_alpha_and_beta.txt', 'w') as outfile:
        outfile.write("{} models\n".format(len(modelbase)))
        outfile.write("{} models were not matched between beta and alpha\n".format(mismatches))
        outfile.write("---------------------------------------------------------\n")
        outfile.write(header)
        for model in modelbase:
            l = ""
            for pval in model[0]:
                l+=str(pval[1])+"    "
            l+=str(model[1])+'\n'
            outfile.write(l)
        

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
    # Set global gamma if provided:
    if gamma != 1:
        print("Global gamma set: {}".format(gamma))
        ydata_Alpha = np.divide(ydata_Alpha,gamma)
        ydata_Beta = np.divide(ydata_Beta,gamma)
        Alpha_uncertainty = np.divide(Alpha_uncertainty,gamma)
        Beta_uncertainty = np.divide(Beta_uncertainty,gamma)
    # Now fit the models
    pp.fit_model(modelfileA, xdata_Alpha, ['TotalpSTAT',ydata_Alpha], Alpha_tests,
                     p0=p0_Alpha, sigma=Alpha_uncertainty, n=5, method="brute")
    os.rename('modelfit.txt', 'modelfit_alpha.txt')
    pp.fit_model(modelfileB, xdata_Beta, ['TotalpSTAT',ydata_Beta], Beta_tests,
                     p0=p0_Beta, sigma=Beta_uncertainty, n=5, method="brute")
    os.rename('modelfit.txt', 'modelfit_beta.txt')
    # Combine the alpha and beta fits to get the best overall model
    print("Finding the best model for alpha and beta Interferon")
    if 'kd4' in Alpha_tests:
        fit_alpha_and_beta(1)
    else:
        fit_alpha_and_beta()
    
if __name__ == '__main__':
    main()

