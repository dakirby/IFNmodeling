"""
--------------
Data available
--------------
As DataFrames:
      Alpha
    5 minutes       a5smooth
    15 minutes      a15smooth
    30 minutes      a30smooth
    60 minutes      a60smooth

      Beta

    5 minutes       b5smooth
    15 minutes      b15smooth
    30 minutes      b30smooth
    60 minutes      b60smooth

      Combined

    5 minutes       smooth5
    15 minutes      smooth15
    30 minutes      smooth30
    60 minutes      smooth60

    Combined        smooth_total

As IfnData objects:
       Alpha

    5 minutes       a5smoothIfnData
    15 minutes      a15smoothIfnData
    30 minutes      a30smoothIfnData
    60 minutes      a60smoothIfnData

       Beta

    5 minutes       b5smoothIfnData
    15 minutes      b15smoothIfnData
    30 minutes      b30smoothIfnData
    60 minutes      b60smoothIfnData

      Combined

    5 minutes       smooth5IfnData
    15 minutes      smooth15IfnData
    30 minutes      smooth30IfnData
    60 minutes      smooth60IfnData

    Complete        smoothIfnData
"""

from ifnclass.ifndata import IfnData
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import seaborn as sns
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from math import isnan

def MM_data(xdata, top, n, k):
    ydata = [(top * x ** n / (k ** n + x ** n), nan) for x in xdata]
    return ydata


def MM(xdata, top, n, k):
    ydata = [top * x ** n / (k ** n + x ** n) for x in xdata]
    return ydata


def fit_MM(doses, responses, guesses):
    top = guesses[0]
    n = guesses[1]
    K = guesses[2]
    results, covariance = curve_fit(MM, doses, responses, p0=[top, n, K])
    top = results[0]
    n = results[1]
    K = results[2]
    return top, n, K


newdata = IfnData("MacParland_Extended")
data = np.transpose([[el[0] for el in r] for r in newdata.data_set.values])
alpha_data = [r[0:5].tolist() for r in data]
beta_data = [r[5:].tolist() for r in data]

# testModel = IfnModel('Mixed_IFN_ppCompatible')

# dra = testModel.doseresponse([0, 5, 15, 30, 60], 'TotalpSTAT', 'Ia', list(logspace(-3, 4)),
#                             parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')

exp_doses_a = [10, 90, 600, 4000, 8000]
exp_doses_b = [10, 90, 600, 3000, 11000]
alpha5 = [[], []]
alpha15 = [[], []]
alpha30 = [[], []]
alpha60 = [[], []]
beta5 = [[], []]
beta15 = [[], []]
beta30 = [[], []]
beta60 = [[], []]

for i in range(len(exp_doses_a)):
    if not isnan(alpha_data[1][i]):
        alpha5[0].append(exp_doses_a[i])
        alpha5[1].append(alpha_data[1][i])
    if not isnan(alpha_data[2][i]):
        alpha15[0].append(exp_doses_a[i])
        alpha15[1].append(alpha_data[2][i])
    if not isnan(alpha_data[3][i]):
        alpha30[0].append(exp_doses_a[i])
        alpha30[1].append(alpha_data[3][i])
    if not isnan(alpha_data[4][i]):
        alpha60[0].append(exp_doses_a[i])
        alpha60[1].append(alpha_data[4][i])
    if not isnan(beta_data[1][i]):
        beta5[0].append(exp_doses_b[i])
        beta5[1].append(beta_data[1][i])
    if not isnan(beta_data[2][i]):
        beta15[0].append(exp_doses_b[i])
        beta15[1].append(beta_data[2][i])
    if not isnan(beta_data[3][i]):
        beta30[0].append(exp_doses_b[i])
        beta30[1].append(beta_data[3][i])
    if not isnan(beta_data[4][i]):
        beta60[0].append(exp_doses_b[i])
        beta60[1].append(beta_data[4][i])


# Fit MM curves
alpha5_top, alpha5_n, alpha5_K = fit_MM(alpha5[0], alpha5[1], [12260, 0.8, 477])
alpha15_top, alpha15_n, alpha15_K = fit_MM(alpha15[0], alpha15[1], [12260, 0.8, 477])
alpha30_top, alpha30_n, alpha30_K = fit_MM(alpha30[0], alpha30[1], [12260, 0.8, 477])
alpha60_top, alpha60_n, alpha60_K = fit_MM(alpha60[0][:-1], alpha60[1][:-1], [85.5118645656, 1.47556196309, 1421.430])
beta5_top, beta5_n, beta5_K = fit_MM(beta5[0], beta5[1], [12260, 0.8, 477])
beta15_top, beta15_n, beta15_K = fit_MM(beta15[0], beta15[1], [12260, 0.8, 477])
beta30_top, beta30_n, beta30_K = fit_MM(beta30[0], beta30[1], [12260, 0.8, 477])
beta60_top, beta60_n, beta60_K = fit_MM(beta60[0], beta60[1], [168.199554682, 0.745160269961, 8.1995661411])

alpha15_n = 2.2

# Generate points from MM
alpha_doses = list(logspace(log10(5), log10(8000), 15))+list(logspace(2.5, 3, 10))
alpha_doses.sort()
beta_doses = list(logspace(0, log10(11000), 15))+list(logspace(2, 3.8, 10))
beta_doses.sort()

alpha5_smooth_data = [alpha_doses, MM_data(alpha_doses, alpha5_top, alpha5_n, alpha5_K)]
alpha15_smooth_data = [alpha_doses, MM_data(alpha_doses, alpha15_top, alpha15_n, alpha15_K)]
alpha30_smooth_data = [alpha_doses, MM_data(alpha_doses, alpha30_top, alpha30_n, alpha30_K)]
alpha60_smooth_data = [alpha_doses, MM_data(alpha_doses, alpha60_top, alpha60_n, alpha60_K)]
beta5_smooth_data = [beta_doses, MM_data(beta_doses, beta5_top, beta5_n, beta5_K)]
beta15_smooth_data = [beta_doses, MM_data(beta_doses, beta15_top, beta15_n, beta15_K)]
beta30_smooth_data = [beta_doses, MM_data(beta_doses, beta30_top, beta30_n, beta30_K)]
beta60_smooth_data = [beta_doses, MM_data(beta_doses, beta60_top, beta60_n, beta60_K)]

# Format into DataFrames
a5smooth = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha5_smooth_data[0]), alpha5_smooth_data[0], alpha5_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 5])
a5smooth = a5smooth.set_index(['Dose_Species', 'Dose (pM)'])
a15smooth = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha15_smooth_data[0]), alpha15_smooth_data[0], alpha15_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 15])
a15smooth = a15smooth.set_index(['Dose_Species', 'Dose (pM)'])
a30smooth = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha30_smooth_data[0]), alpha30_smooth_data[0], alpha30_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 30])
a30smooth = a30smooth.set_index(['Dose_Species', 'Dose (pM)'])
a60smooth = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha60_smooth_data[0]), alpha60_smooth_data[0], alpha60_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 60])
a60smooth = a60smooth.set_index(['Dose_Species', 'Dose (pM)'])
b5smooth = pd.DataFrame.from_records(
    list(zip(['Beta'] * len(beta5_smooth_data[0]), beta5_smooth_data[0], beta5_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 5])
b5smooth = b5smooth.set_index(['Dose_Species', 'Dose (pM)'])
b15smooth = pd.DataFrame.from_records(
    list(zip(['Beta'] * len(beta15_smooth_data[0]), beta15_smooth_data[0], beta15_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 15])
b15smooth = b15smooth.set_index(['Dose_Species', 'Dose (pM)'])
b30smooth = pd.DataFrame.from_records(
    list(zip(['Beta'] * len(beta30_smooth_data[0]), beta30_smooth_data[0], beta30_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 30])
b30smooth = b30smooth.set_index(['Dose_Species', 'Dose (pM)'])
b60smooth = pd.DataFrame.from_records(
    list(zip(['Beta'] * len(beta60_smooth_data[0]), beta60_smooth_data[0], beta60_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 60])
b60smooth = b60smooth.set_index(['Dose_Species', 'Dose (pM)'])

# Make Data Frames of combined data, for fitting
smooth5 = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha5_smooth_data[0]), alpha5_smooth_data[0], alpha5_smooth_data[1])) +
    list(zip(['Beta'] * len(beta5_smooth_data[0]), beta5_smooth_data[0], beta5_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 5])
smooth5 = smooth5.set_index(['Dose_Species', 'Dose (pM)'])
smooth15 = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha15_smooth_data[0]), alpha15_smooth_data[0], alpha15_smooth_data[1])) +
    list(zip(['Beta'] * len(beta15_smooth_data[0]), beta15_smooth_data[0], beta15_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 15])
smooth15 = smooth15.set_index(['Dose_Species', 'Dose (pM)'])
smooth30 = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha30_smooth_data[0]), alpha30_smooth_data[0], alpha30_smooth_data[1])) +
    list(zip(['Beta'] * len(beta30_smooth_data[0]), beta30_smooth_data[0], beta30_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 30])
smooth30 = smooth30.set_index(['Dose_Species', 'Dose (pM)'])
smooth60 = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha60_smooth_data[0]), alpha60_smooth_data[0], alpha60_smooth_data[1])) +
    list(zip(['Beta'] * len(beta60_smooth_data[0]), beta60_smooth_data[0], beta60_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 60])
smooth60 = smooth60.set_index(['Dose_Species', 'Dose (pM)'])

smooth_total = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha60_smooth_data[0]), alpha60_smooth_data[0],
             alpha5_smooth_data[1], alpha15_smooth_data[1], alpha30_smooth_data[1], alpha60_smooth_data[1])) +
    list(zip(['Beta'] * len(beta60_smooth_data[0]), beta60_smooth_data[0],
             beta5_smooth_data[1], beta15_smooth_data[1], beta30_smooth_data[1], beta60_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 5, 15, 30, 60])
smooth_total = smooth_total.set_index(['Dose_Species', 'Dose (pM)'])

# Build IfnData objects
# ---------------------
a5smoothIfnData = IfnData('custom', df=a5smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
a15smoothIfnData = IfnData('custom', df=a15smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
a30smoothIfnData = IfnData('custom', df=a30smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
a60smoothIfnData = IfnData('custom', df=a60smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
b5smoothIfnData = IfnData('custom', df=b5smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
b15smoothIfnData = IfnData('custom', df=b15smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
b30smoothIfnData = IfnData('custom', df=b30smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
b60smoothIfnData = IfnData('custom', df=b60smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
smooth5IfnData = IfnData('custom', df=smooth5, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
smooth15IfnData = IfnData('custom', df=smooth15, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
smooth30IfnData = IfnData('custom', df=smooth30, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
smooth60IfnData = IfnData('custom', df=smooth60, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
smoothIfnData = IfnData('custom', df=smooth_total, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})


# Check plots
def check_plots():
    smooth_plot = DoseresponsePlot((1, 2))
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    smooth_plot.add_trajectory(a5smoothIfnData, 5, 'plot', alpha_palette[0], (0, 0), 'Alpha')
    smooth_plot.add_trajectory(a15smoothIfnData, 15, 'plot', alpha_palette[1], (0, 0), 'Alpha')
    smooth_plot.add_trajectory(a30smoothIfnData, 30, 'plot', alpha_palette[2], (0, 0), 'Alpha')
    smooth_plot.add_trajectory(a60smoothIfnData, 60, 'plot', alpha_palette[3], (0, 0), 'Alpha')

    smooth_plot.add_trajectory(b5smoothIfnData, 5, 'plot', beta_palette[0], (0, 1), 'Beta')
    smooth_plot.add_trajectory(b15smoothIfnData, 15, 'plot', beta_palette[1], (0, 1), 'Beta')
    smooth_plot.add_trajectory(b30smoothIfnData, 30, 'plot', beta_palette[2], (0, 1), 'Beta')
    smooth_plot.add_trajectory(b60smoothIfnData, 60, 'plot', beta_palette[3], (0, 1), 'Beta')

    for idx, t in enumerate([5, 15, 30, 60]):
        smooth_plot.add_trajectory(newdata, t, 'scatter', alpha_palette[idx], (0, 0), 'Alpha', dn=1)
        smooth_plot.add_trajectory(newdata, t, 'scatter', beta_palette[idx], (0, 1), 'Beta', dn=1)

    fig, axes = smooth_plot.show_figure()


if __name__ == '__main__':
    check_plots()
