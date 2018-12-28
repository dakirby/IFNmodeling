"""
Data available
--------------
As DataFrames:
      Alpha
    2.5 minutes     a25smooth
    5 minutes       a5smooth
    7.5 minutes     a75smooth
    10 minutes      a10smooth
    20 minutes      a20smooth
    60 minutes      a60smooth

      Beta

    2.5 minutes     b25smooth
    5 minutes       b5smooth
    7.5 minutes     b75smooth
    10 minutes      b10smooth
    20 minutes      b20smooth
    60 minutes      b60smooth

      Combined

    2.5 minutes     smooth25
    5 minutes       smooth5
    7.5 minutes     smooth75
    10 minutes      smooth10
    20 minutes      smooth20
    60 minutes      smooth60

As IfnData objects:
       Alpha

    2.5 minutes     a25smoothIfnData
    5 minutes       a5smoothIfnData
    7.5 minutes     a75smoothIfnData
    10 minutes      a10smoothIfnData
    20 minutes      a20smoothIfnData
    60 minutes      a60smoothIfnData

       Beta

    2.5 minutes     b25smoothIfnData
    5 minutes       b5smoothIfnData
    7.5 minutes     b75smoothIfnData
    10 minutes      b10smoothIfnData
    20 minutes      b20smoothIfnData
    60 minutes      b60smoothIfnData

      Combined

    2.5 minutes     smooth25IfnData
    5 minutes       smooth5IfnData
    7.5 minutes     smooth75IfnData
    10 minutes      smooth10IfnData
    20 minutes      smooth20IfnData
    60 minutes      smooth60IfnData

"""

from ifndata import IfnData
from ifnmodel import IfnModel
from ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from ifnfit import StepwiseFit
from numpy import linspace, logspace, log10, nan
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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


newdata = IfnData("20181113_B6_IFNs_Dose_Response_Bcells")
data = np.transpose([[el[0] for el in r] for r in newdata.data_set.values])
alpha_data = [r[1:8].tolist() for r in data]
beta_data = [r[9:].tolist() for r in data]

# testModel = IfnModel('Mixed_IFN_ppCompatible')

# dra = testModel.doseresponse([0, 5, 15, 30, 60], 'TotalpSTAT', 'Ia', list(logspace(-3, 4)),
#                             parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')

exp_doses_a = [5, 50, 250, 500, 5000, 25000, 50000]
exp_doses_b = [0.1, 1, 5, 10, 100, 200, 1000]

alpha2_5 = [exp_doses_a[2:], alpha_data[0][2:]]
alpha5 = [exp_doses_a, alpha_data[1]]
alpha7_5 = [exp_doses_a, alpha_data[2]]
alpha10 = [exp_doses_a, alpha_data[3]]
alpha20 = [exp_doses_a, alpha_data[4]]
alpha60 = [exp_doses_a[3:], alpha_data[5][3:]]
beta2_5 = [exp_doses_b[1:], beta_data[0][1:]]
beta5 = [exp_doses_b, beta_data[1]]
beta7_5 = [exp_doses_b, beta_data[2]]
beta10 = [exp_doses_b, beta_data[3]]
beta20 = [exp_doses_b, beta_data[4]]
beta60 = [exp_doses_b[1:], beta_data[5][1:]]

# Fit MM curves
alpha25_top, alpha25_n, alpha25_K = fit_MM(alpha2_5[0], alpha2_5[1], [65, 0.1, 1421])
alpha5_top, alpha5_n, alpha5_K = fit_MM(alpha5[0], alpha5[1], [65, 0.1, 1421])
alpha75_top, alpha75_n, alpha75_K = fit_MM(alpha7_5[0], alpha7_5[1], [65, 0.1, 1421])
alpha10_top, alpha10_n, alpha10_K = fit_MM(alpha10[0], alpha10[1], [65, 0.1, 1421])
alpha20_top, alpha20_n, alpha20_K = fit_MM(alpha20[0], alpha20[1], [65, 0.1, 1421])
alpha60_top, alpha60_n, alpha60_K = fit_MM(alpha60[0], alpha60[1], [85.5118645656, 1.47556196309, 1421.43092465])
beta25_top, beta25_n, beta25_K = fit_MM(beta2_5[0], beta2_5[1], [150, 0.1, 8])
beta5_top, beta5_n, beta5_K = fit_MM(beta5[0], beta5[1], [150, 0.1, 8])
beta75_top, beta75_n, beta75_K = fit_MM(beta7_5[0], beta7_5[1], [150, 0.1, 8])
beta10_top, beta10_n, beta10_K = fit_MM(beta10[0], beta10[1], [150, 0.1, 8])
beta20_top, beta20_n, beta20_K = fit_MM(beta20[0], beta20[1], [150, 0.1, 8])
beta60_top, beta60_n, beta60_K = fit_MM(beta60[0], beta60[1], [168.199554682, 0.745160269961, 8.1995661411])

alpha25_n = 3
alpha5_n = 3
beta25_n = 3
#print(alpha25_top, alpha25_n, alpha25_K)
#print(alpha60_top, alpha60_n, alpha60_K)
#print(beta25_top, beta25_n, beta25_K)
#print(beta60_top, beta60_n, beta60_K)

# Generate points from MM
alpha_doses = list(logspace(log10(5), log10(50000), 15))+list(logspace(2.5, 2.95, 10))
alpha_doses.sort()
beta_doses = list(logspace(0, 3, 15))+list(logspace(1.35, 2.65, 10))
beta_doses.sort()

alpha2_5_smooth_data = [alpha_doses, MM_data(alpha_doses, alpha25_top, alpha25_n, alpha25_K)]
alpha5_smooth_data = [alpha_doses, MM_data(alpha_doses, alpha5_top, alpha5_n, alpha5_K)]
alpha7_5_smooth_data = [alpha_doses, MM_data(alpha_doses, alpha75_top, alpha75_n, alpha75_K)]
alpha10_smooth_data = [alpha_doses, MM_data(alpha_doses, alpha10_top, alpha10_n, alpha10_K)]
alpha20_smooth_data = [alpha_doses, MM_data(alpha_doses, alpha20_top, alpha20_n, alpha20_K)]
alpha60_smooth_data = [alpha_doses, MM_data(alpha_doses, alpha60_top, alpha60_n, alpha60_K)]
beta2_5_smooth_data = [beta_doses, MM_data(beta_doses, beta25_top, beta25_n, beta25_K)]
beta5_smooth_data = [beta_doses, MM_data(beta_doses, beta5_top, beta5_n, beta5_K)]
beta7_5_smooth_data = [beta_doses, MM_data(beta_doses, beta75_top, beta75_n, beta75_K)]
beta10_smooth_data = [beta_doses, MM_data(beta_doses, beta10_top, beta10_n, beta10_K)]
beta20_smooth_data = [beta_doses, MM_data(beta_doses, beta20_top, beta20_n, beta20_K)]
beta60_smooth_data = [beta_doses, MM_data(beta_doses, beta60_top, beta60_n, beta60_K)]

# Format into DataFrames
a25smooth = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha2_5_smooth_data[0]), alpha2_5_smooth_data[0], alpha2_5_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 2.5])
a25smooth = a25smooth.set_index(['Dose_Species', 'Dose (pM)'])
a5smooth = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha5_smooth_data[0]), alpha5_smooth_data[0], alpha5_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 5])
a5smooth = a5smooth.set_index(['Dose_Species', 'Dose (pM)'])
a75smooth = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha7_5_smooth_data[0]), alpha7_5_smooth_data[0], alpha7_5_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 7.5])
a75smooth = a75smooth.set_index(['Dose_Species', 'Dose (pM)'])
a10smooth = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha10_smooth_data[0]), alpha10_smooth_data[0], alpha10_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 10])
a10smooth = a10smooth.set_index(['Dose_Species', 'Dose (pM)'])
a20smooth = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha20_smooth_data[0]), alpha20_smooth_data[0], alpha20_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 20])
a20smooth = a20smooth.set_index(['Dose_Species', 'Dose (pM)'])
a60smooth = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha60_smooth_data[0]), alpha60_smooth_data[0], alpha60_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 60])
a60smooth = a60smooth.set_index(['Dose_Species', 'Dose (pM)'])
b25smooth = pd.DataFrame.from_records(
    list(zip(['Beta'] * len(beta2_5_smooth_data[0]), beta2_5_smooth_data[0], beta2_5_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 2.5])
b25smooth = b25smooth.set_index(['Dose_Species', 'Dose (pM)'])
b5smooth = pd.DataFrame.from_records(
    list(zip(['Beta'] * len(beta5_smooth_data[0]), beta5_smooth_data[0], beta5_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 5])
b5smooth = b5smooth.set_index(['Dose_Species', 'Dose (pM)'])
b75smooth = pd.DataFrame.from_records(
    list(zip(['Beta'] * len(beta7_5_smooth_data[0]), beta7_5_smooth_data[0], beta7_5_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 7.5])
b75smooth = b75smooth.set_index(['Dose_Species', 'Dose (pM)'])
b10smooth = pd.DataFrame.from_records(
    list(zip(['Beta'] * len(beta10_smooth_data[0]), beta10_smooth_data[0], beta10_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 10])
b10smooth = b10smooth.set_index(['Dose_Species', 'Dose (pM)'])
b20smooth = pd.DataFrame.from_records(
    list(zip(['Beta'] * len(beta20_smooth_data[0]), beta20_smooth_data[0], beta20_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 20])
b20smooth = b20smooth.set_index(['Dose_Species', 'Dose (pM)'])
b60smooth = pd.DataFrame.from_records(
    list(zip(['Beta'] * len(beta60_smooth_data[0]), beta60_smooth_data[0], beta60_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 60])
b60smooth = b60smooth.set_index(['Dose_Species', 'Dose (pM)'])

# Make Data Frames of combined data, for fitting
smooth25 = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha2_5_smooth_data[0]), alpha2_5_smooth_data[0], alpha2_5_smooth_data[1])) +
    list(zip(['Beta'] * len(beta2_5_smooth_data[0]), beta2_5_smooth_data[0], beta2_5_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 2.5])
smooth25 = smooth25.set_index(['Dose_Species', 'Dose (pM)'])
smooth5 = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha5_smooth_data[0]), alpha5_smooth_data[0], alpha5_smooth_data[1])) +
    list(zip(['Beta'] * len(beta5_smooth_data[0]), beta5_smooth_data[0], beta5_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 5])
smooth5 = smooth5.set_index(['Dose_Species', 'Dose (pM)'])
smooth75 = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha7_5_smooth_data[0]), alpha7_5_smooth_data[0], alpha7_5_smooth_data[1])) +
    list(zip(['Beta'] * len(beta7_5_smooth_data[0]), beta7_5_smooth_data[0], beta7_5_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 7.5])
smooth75 = smooth75.set_index(['Dose_Species', 'Dose (pM)'])
smooth10 = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha10_smooth_data[0]), alpha10_smooth_data[0], alpha10_smooth_data[1])) +
    list(zip(['Beta'] * len(beta10_smooth_data[0]), beta10_smooth_data[0], beta10_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 10])
smooth10 = smooth10.set_index(['Dose_Species', 'Dose (pM)'])
smooth20 = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha20_smooth_data[0]), alpha20_smooth_data[0], alpha20_smooth_data[1])) +
    list(zip(['Beta'] * len(beta20_smooth_data[0]), beta20_smooth_data[0], beta20_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 20])
smooth20 = smooth20.set_index(['Dose_Species', 'Dose (pM)'])
smooth60 = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha60_smooth_data[0]), alpha60_smooth_data[0], alpha60_smooth_data[1])) +
    list(zip(['Beta'] * len(beta60_smooth_data[0]), beta60_smooth_data[0], beta60_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 60])
smooth60 = smooth60.set_index(['Dose_Species', 'Dose (pM)'])

# Build IfnData objects
# ---------------------
a25smoothIfnData = IfnData('custom', df=a25smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
a5smoothIfnData = IfnData('custom', df=a5smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
a75smoothIfnData = IfnData('custom', df=a75smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
a10smoothIfnData = IfnData('custom', df=a10smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
a20smoothIfnData = IfnData('custom', df=a20smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
a60smoothIfnData = IfnData('custom', df=a60smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
b25smoothIfnData = IfnData('custom', df=b25smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
b5smoothIfnData = IfnData('custom', df=b5smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
b75smoothIfnData = IfnData('custom', df=b75smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
b10smoothIfnData = IfnData('custom', df=b10smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
b20smoothIfnData = IfnData('custom', df=b20smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
b60smoothIfnData = IfnData('custom', df=b60smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
smooth25IfnData = IfnData('custom', df=smooth25, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
smooth5IfnData = IfnData('custom', df=smooth5, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
smooth75IfnData = IfnData('custom', df=smooth75, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
smooth10IfnData = IfnData('custom', df=smooth10, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
smooth20IfnData = IfnData('custom', df=smooth20, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
smooth60IfnData = IfnData('custom', df=smooth60, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})


# Check plots
def check_plots():
    smooth_plot = DoseresponsePlot((2, 2))
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    smooth_plot.add_trajectory(a25smoothIfnData, 2.5, 'plot', alpha_palette[4], (0, 0), 'Alpha')
    smooth_plot.add_trajectory(a60smoothIfnData, 60, 'plot', alpha_palette[5], (0, 0), 'Alpha')

    smooth_plot.add_trajectory(a25smoothIfnData, 2.5, 'plot', alpha_palette[0], (1, 0), 'Alpha')
    smooth_plot.add_trajectory(a5smoothIfnData, 5, 'plot', alpha_palette[1], (1, 0), 'Alpha')
    smooth_plot.add_trajectory(a75smoothIfnData, 7.5, 'plot', alpha_palette[2], (1, 0), 'Alpha')
    smooth_plot.add_trajectory(a10smoothIfnData, 10, 'plot', alpha_palette[3], (1, 0), 'Alpha')
    smooth_plot.add_trajectory(a20smoothIfnData, 20, 'plot', alpha_palette[4], (1, 0), 'Alpha')
    smooth_plot.add_trajectory(a60smoothIfnData, 60, 'plot', alpha_palette[5], (1, 0), 'Alpha')

    smooth_plot.add_trajectory(b25smoothIfnData, 2.5, 'plot', beta_palette[4], (0, 1), 'Beta')
    smooth_plot.add_trajectory(b60smoothIfnData, 60, 'plot', beta_palette[5], (0, 1), 'Beta')

    smooth_plot.add_trajectory(b25smoothIfnData, 2.5, 'plot', beta_palette[0], (1, 1), 'Beta')
    smooth_plot.add_trajectory(b5smoothIfnData, 5, 'plot', beta_palette[1], (1, 1), 'Beta')
    smooth_plot.add_trajectory(b75smoothIfnData, 7.5, 'plot', beta_palette[2], (1, 1), 'Beta')
    smooth_plot.add_trajectory(b10smoothIfnData, 10, 'plot', beta_palette[3], (1, 1), 'Beta')
    smooth_plot.add_trajectory(b20smoothIfnData, 20, 'plot', beta_palette[4], (1, 1), 'Beta')
    smooth_plot.add_trajectory(b60smoothIfnData, 60, 'plot', beta_palette[5], (1, 1), 'Beta')

    for idx, t in enumerate([2.5, 60]):
        smooth_plot.add_trajectory(newdata, t, 'scatter', alpha_palette[idx * 2], (0, 0), 'Alpha', dn=1)
        smooth_plot.add_trajectory(newdata, t, 'scatter', beta_palette[idx * 2], (0, 1), 'Beta', dn=1)

    fig, axes = smooth_plot.show_figure()


if __name__ == '__main__':
    check_plots()
