from ifndata import IfnData
from ifnmodel import IfnModel
from ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from ifnfit import StepwiseFit
from numpy import linspace, logspace, log10, nan
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd


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
print(newdata)

# testModel = IfnModel('Mixed_IFN_ppCompatible')

# dra = testModel.doseresponse([0, 5, 15, 30, 60], 'TotalpSTAT', 'Ia', list(logspace(-3, 4)),
#                             parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')

alpha2_5 = [[250, 500, 5000, 25000, 50000], [4.69, 3.19, 15.8, 53.4, 34.2]]
alpha10 = [[250, 500, 5000, 25000, 50000], [4.69, 3.19, 15.8, 53.4, 34.2]]
alpha60 = [[500, 5000, 25000, 50000], [15.09, 73.9, 84.7, 84.7]]
beta2_5 = [[1, 5, 10, 100, 200, 1000], [23.29, 9.2, 15, 59.6, 152.9, 127.9]]
beta60 = [[1, 5, 10, 100, 200, 1000], [27, 73, 85.9, 162.9, 129.9, 171.9]]
alpha25_top, alpha25_n, alpha25_K = fit_MM(alpha2_5[0], alpha2_5[1], [65, 0.1, 1421])
alpha60_top, alpha60_n, alpha60_K = fit_MM(alpha60[0], alpha60[1], [85.5118645656, 1.47556196309, 1421.43092465])
beta25_top, beta25_n, beta25_K = fit_MM(beta2_5[0], beta2_5[1], [150, 0.1, 8])
beta60_top, beta60_n, beta60_K = fit_MM(beta60[0], beta60[1], [168.199554682, 0.745160269961, 8.1995661411])

alpha25_n = 3
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
alpha60_smooth_data = [alpha_doses, MM_data(alpha_doses, alpha60_top, alpha60_n, alpha60_K)]
beta2_5_smooth_data = [beta_doses, MM_data(beta_doses, beta25_top, beta25_n, beta25_K)]
beta60_smooth_data = [beta_doses, MM_data(beta_doses, beta60_top, beta60_n, beta60_K)]
# Format into DataFrames
a25smooth = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha2_5_smooth_data[0]), alpha2_5_smooth_data[0], alpha2_5_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 2.5])
a25smooth = a25smooth.set_index(['Dose_Species', 'Dose (pM)'])
a60smooth = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha60_smooth_data[0]), alpha60_smooth_data[0], alpha60_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 60])
a60smooth = a60smooth.set_index(['Dose_Species', 'Dose (pM)'])
b25smooth = pd.DataFrame.from_records(
    list(zip(['Beta'] * len(beta2_5_smooth_data[0]), beta2_5_smooth_data[0], beta2_5_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 2.5])
b25smooth = b25smooth.set_index(['Dose_Species', 'Dose (pM)'])
b60smooth = pd.DataFrame.from_records(
    list(zip(['Beta'] * len(beta60_smooth_data[0]), beta60_smooth_data[0], beta60_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 60])
b60smooth = b60smooth.set_index(['Dose_Species', 'Dose (pM)'])

smooth25 = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha2_5_smooth_data[0]), alpha2_5_smooth_data[0], alpha2_5_smooth_data[1])) +
    list(zip(['Beta'] * len(beta2_5_smooth_data[0]), beta2_5_smooth_data[0], beta2_5_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 2.5])
smooth25 = smooth25.set_index(['Dose_Species', 'Dose (pM)'])

smooth60 = pd.DataFrame.from_records(
    list(zip(['Alpha'] * len(alpha60_smooth_data[0]), alpha60_smooth_data[0], alpha60_smooth_data[1])) +
    list(zip(['Beta'] * len(beta60_smooth_data[0]), beta60_smooth_data[0], beta60_smooth_data[1])),
    columns=['Dose_Species', 'Dose (pM)', 2.5])
smooth60 = smooth60.set_index(['Dose_Species', 'Dose (pM)'])

# Build IfnData objects
# ---------------------
a25smoothIfnData = IfnData('custom', df=a25smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
a60smoothIfnData = IfnData('custom', df=a60smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
b25smoothIfnData = IfnData('custom', df=b25smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
b60smoothIfnData = IfnData('custom', df=b60smooth, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
smooth25IfnData = IfnData('custom', df=smooth25, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
smooth60IfnData = IfnData('custom', df=smooth60, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})


# Check plots
smooth_plot = DoseresponsePlot((2, 2))
alpha_palette = sns.color_palette("Reds", 6)
beta_palette = sns.color_palette("Greens", 6)

smooth_plot.add_trajectory(a25smoothIfnData, 2.5, 'plot', alpha_palette[4], (0, 0), 'Alpha')
smooth_plot.add_trajectory(a25smoothIfnData, 2.5, 'plot', alpha_palette[4], (1, 0), 'Alpha')

smooth_plot.add_trajectory(a60smoothIfnData, 60, 'plot', alpha_palette[5], (0, 0), 'Alpha')
smooth_plot.add_trajectory(a60smoothIfnData, 60, 'plot', alpha_palette[5], (1, 0), 'Alpha')

smooth_plot.add_trajectory(b25smoothIfnData, 2.5, 'plot', beta_palette[4], (0, 1), 'Beta')
smooth_plot.add_trajectory(b25smoothIfnData, 2.5, 'plot', beta_palette[4], (1, 1), 'Beta')

smooth_plot.add_trajectory(b60smoothIfnData, 60, 'plot', beta_palette[5], (0, 1), 'Beta')
smooth_plot.add_trajectory(b60smoothIfnData, 60, 'plot', beta_palette[5], (1, 1), 'Beta')

for idx, t in enumerate([2.5, 60]):
    smooth_plot.add_trajectory(newdata, t, 'scatter', alpha_palette[idx * 2], (0, 0), 'Alpha', dn=1)
    smooth_plot.add_trajectory(newdata, t, 'scatter', beta_palette[idx * 2], (0, 1), 'Beta', dn=1)

fig, axes = smooth_plot.show_figure()