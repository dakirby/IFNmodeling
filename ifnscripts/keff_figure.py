from ifnclass.ifndata import IfnData
from ifnclass.ifnfit import DualMixedPopulation
from ifnclass.ifnplot import DoseresponsePlot
import os
from numpy import logspace
import copy
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from AP_AV_DATA import Thomas2011IFNalpha2AV, Thomas2011IFNalpha2YNSAV,\
 Thomas2011IFNalpha7AV, Thomas2011IFNomegaAV, Thomas2011IFNalpha2YNSAP,\
 Thomas2011IFNalpha2AP, Thomas2011IFNalpha7AP, Thomas2011IFNomegaAP,\
 Schreiber2017AV, Schreiber2017AP,\
 Thomas2011IFNalpha2AV_s, Thomas2011IFNalpha2YNSAV_s,\
 Thomas2011IFNalpha7AV_s, Thomas2011IFNomegaAV_s, Thomas2011IFNalpha2YNSAP_s,\
 Thomas2011IFNalpha2AP_s, Thomas2011IFNalpha7AP_s, Thomas2011IFNomegaAP_s

from AP_AV_theory import antiViralActivity, antiProliferativeActivity
from AP_AV_simulations import AP_AV_simulations
from AP_AV_fitting import AP_AV_fitting
from AP_AV_setup_barchart import AP_AV_setup_barchart as setup_barchart

plt.rcParams.update({'font.size': 16})


def find_nearest(array, value, idx_flag=False):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if idx_flag:
        return idx
    else:
        return array[idx]


def IC50(dose, response, target=50):
    idx = find_nearest(response, target, idx_flag=True)
    if np.abs((response[idx]-target)/target) < 0.02:
        # if error is less than 2%, consider as acceptably close
        return dose[idx]
    else:
        # if error is greater than 2%, perform log-linear interpolation to nearest point on other side of 50%
        # first define the index direction of increasing response
        if response[idx] > response[idx-1]:
            pos_grad = 1
        else:
            pos_grad = -1
        # figure out what direction to step the index to move towards target
        if response[idx] > target:
            idx_grad = -pos_grad
        else:
            idx_grad = pos_grad

        # step towards target
        interpolation_idx = idx

        def condition(val1, val2):
            return (val1 < target and val2 > target) or (val1 > target and val2 < target)

        while not condition(response[idx], response[interpolation_idx]):
            interpolation_idx += idx_grad

        # build log-linear interpolation
        x1 = min([dose[idx], dose[interpolation_idx]])
        x2 = max([dose[idx], dose[interpolation_idx]])
        y1 = response[np.where(dose == x1)]
        y2 = response[np.where(dose == x2)]
        m = (y2 - y1) / (np.log10(x2) - np.log10(x1))
        b = y1 - np.log10(x1) * m
        ic50 = 10 ** ((target - b) / m)
        return ic50[0]  # ic50 is a one-element ndarray


def plot_barchart(axis=None, df=None, custom_order=None):
    if df is None:
        # Import data
        df = pd.read_csv("AP_AV_Bar_Chart.csv")
    # Prepare data
    df = pd.melt(df, id_vars=['Name'])
    # Plot data
    custom_palette = sns.color_palette("Paired")[1:6]
    ax = sns.barplot(x="Name", y='value', hue='variable', data=df,
                     palette=custom_palette, ax=axis, order=custom_order)
    ax.set_yscale('log')
    ax.set_xlabel(None)
    ax.set_ylabel(r"$IC_{50}$ relative to WT")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    ax = plt.gca()
    return ax


def load_data(dir):
    data = {}
    for fname in os.listdir(dir):
        if fname.endswith(".npy"):
            data[fname[:-4]] = np.load(dir + os.sep + fname)
    return data


if __name__ == '__main__':
    USP18_sf = 15

    times = [60]
    test_doses = list(logspace(-4, 6))  # used if simulate_DR

    dir = os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_6')

    # --------------------
    # Set up Model
    # --------------------
    DATA = load_data(dir)

    # ------------------------------------------------------------
    # Get anti-viral and anti-proliferative responses, plus IC50s
    # ------------------------------------------------------------
    fit_params = np.load(dir + os.sep + 'AV_AP_fit_params.npy')
    KM_AV_fit = fit_params[0]
    AP_params = fit_params[1:]

    for key in copy.deepcopy(list(DATA.keys())):
        if key.startswith('pSTAT') and not (key.endswith('_refractory') or key[:-3].endswith('_fitting')):
            DATA[key + '_AV'] = antiViralActivity(DATA[key], KM=KM_AV_fit)
            DATA[key + '_AV_IC50'] = IC50(DATA['doses'], 100-DATA[key + '_AV'])
        if key.startswith('pSTAT') and key.endswith('_refractory'):
            DATA[key[:-11] + '_AP'] = antiProliferativeActivity(DATA[key], *AP_params)
            DATA[key[:-11] + '_AP_IC50'] = IC50(DATA['doses'], 100-DATA[key[:-11] + '_AP'])

    # --------------------------------------
    # Load up data from Figure 6C barchart
    # --------------------------------------
    setup_barchart()
    baseline_df = pd.read_csv("AP_AV_Bar_Chart.csv")  # this csv has the relative binding affinities
    barchart_variants = ['L30A', 'R149A', 'A145G', 'L26A', 'YNSL153A', 'YNSM148A', 'a2YNS']
    av_update, ap_update = [], []
    for key in barchart_variants:
        av_update.append(DATA['pSTAT_' + key + '_AV_IC50'] / DATA['pSTAT_a2_AV_IC50'])
        ap_update.append(DATA['pSTAT_' + key + '_AP_IC50'] / DATA['pSTAT_a2_AP_IC50'])
    av_update = pd.DataFrame({'AV Model': av_update})
    ap_update = pd.DataFrame({'AP Model': ap_update})

    names_update = baseline_df['Name'].values
    for idx, n in enumerate(names_update):  # make the plot label nicer to read
        if n in ['YNSM148A', 'YNSL153A']:
            names_update[idx] = n[:3] + ', ' + n[3:]
            barchart_variants[idx] = barchart_variants[idx][:3] + ', ' + barchart_variants[idx][3:]
        if n == 'a2YNS':
            names_update[idx] = 'YNS'
            barchart_variants[idx] = 'YNS'

    baseline_df.update(av_update)
    baseline_df.update(ap_update)
    baseline_df.update(names_update)

    # -----------------------------------
    # Make figure S5
    # -----------------------------------
    fig, ax = plt.subplots(nrows=1, ncols=1)
    cp = sns.color_palette("Paired")[1:6]
    ax.scatter(baseline_df['K1 x K2'].values, baseline_df['AV Model'].values, color=cp[2], label=r'Anti-viral $\mathrm{IC}_{50}$')
    ax.scatter(baseline_df['K1 x K2'].values, baseline_df['AP Model'].values, color=cp[4], label=r'Anti-proliferative $\mathrm{IC}_{50}$')
    plt.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$K_1 \times K_2$ ($\mu \mathrm{M}^2$)')
    ax.set_ylabel(r'$\mathrm{IC}_{50}$ ($\mu \mathrm{M}$)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    dir = os.path.join(os.getcwd(), 'results', 'Figures', 'Supplementary')
    fig.savefig(os.path.join(dir, 'IC50_vs_K1K2.pdf'))
