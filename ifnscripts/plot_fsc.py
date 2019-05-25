from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnplot import DoseresponsePlot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import seaborn as sns
import os
import warnings
import fcsparser

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def grab_data(fname, well=None):
    path_RawFiles = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
                                 "ifndatabase", "{}_raw_data".format(fname), "Compensated Export\\")

    Experiment = 'pSTAT1 kinetics [{}]'.format(fname)

    FileNames = []
    for ii in range(96):
        FileNames.append([x for x in os.listdir(path_RawFiles) if x.find('0' + str(ii + 1) + '_Lymphocytes') > -1][0])

    IndexFiles = [x[20:20 + x[20:].find('_')] for x in FileNames]
    Rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    Columns = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    Concentrations_IFNa = (np.array([1e-7, 1e-8, 3e-9, 1e-9, 3e-10, 1e-10, 1e-11, 0]))
    Concentrations_IFNb = (np.array([2e-9, 6e-10, 2e-10, 6e-11, 2e-11, 6e-12, 2e-13, 0]))

    Timepoints = ([2.5, 5, 7.5, 10, 20, 60])
    Cytokines = ['IFN-alpha', 'IFN-beta']
    Concentrations = pd.DataFrame(np.array([Concentrations_IFNa, Concentrations_IFNb]).T, columns=Cytokines)
    Readout = ['pSTAT1 in B cells', 'pSTAT1 in CD8+ T cells', 'pSTAT1 in CD4+ T cells']
    df1 = pd.DataFrame(np.zeros((96,)))
    df1.index = FileNames
    df1.columns = ['Concentration (Mol)']
    df2 = []  # =pd.DataFrame(np.zeros((96,)),index=FileNames,columns=['Cytokine'])
    df3 = pd.DataFrame(np.zeros((96,)), index=FileNames, columns=['Time (min)'])

    for ff in FileNames:
        ii = ff[20:20 + ff[20:].find('_')]
        if int(ii[1:]) % 2 == 1:
            df1.loc[ff]['Concentration (Mol)'] = Concentrations_IFNa[Rows.index(ii[0])]
            df2.append('IFN-alpha')
            df3.loc[ff]['Time (min)'] = Timepoints[int(np.floor((int(ii[1:]) - 1) / 2))]
        else:
            df1.loc[ff]['Concentration (Mol)'] = Concentrations_IFNb[Rows.index(ii[0])]
            df2.append('IFN-beta')
            df3.loc[ff]['Time (min)'] = Timepoints[int(np.floor((int(ii[1:]) - 1) / 2))]

    df2 = pd.DataFrame(df2, index=FileNames, columns=['Cytokine'])
    df0 = pd.DataFrame([ff[20:20 + ff[20:].find('_')] for ff in FileNames], columns=['Well'], index=FileNames)
    Conditions = pd.concat([df0, df1, df2, df3], axis=1)

    AllData = {}
    markers = []

    for ff in FileNames:
        meta, df = fcsparser.parse(path_RawFiles + ff, reformat_meta=True)
        if len(AllData) == 0:
            Fluorophores = [x for x in meta['_channel_names_'] if (x.find('FJComp') > -1) or (x.find('FSC') > -1)]
            channels = list(meta['_channels_']['$PnN'].values)
            for mm in Fluorophores:
                if mm.find('SC') > -1:
                    markers.append(mm)
                else:
                    identifier = '$P' + str(channels.index(mm) + 1) + 'S'
                    markers.append(meta[identifier] + ' (' + mm[mm.find('-') + 1:-2] + ')')
        AllData[len(AllData)] = df[Fluorophores]
        AllData[len(AllData) - 1].columns = markers

    # This is the normal approach, where we take the mean fluorescence in a well
    if well == None:
        pSTAT1 = pd.DataFrame(np.zeros((96, 2)),
                              columns=['pSTAT1 in B cells', 'FSC'], index=FileNames)
        for ff in FileNames:
            index = (AllData[FileNames.index(ff)]['CD19 (PE ( 561 ))'] > 600) & \
                    (AllData[FileNames.index(ff)]['IA_IE (BV510)'] > 2000)
            pSTAT1.loc[ff]['pSTAT1 in B cells'] = np.arcsinh(np.mean(np.sinh(AllData[FileNames.index(ff)][index]['pSTAT1 (FITC)'])))
            pSTAT1.loc[ff]['FSC'] = np.arcsinh(np.mean(np.sinh(AllData[FileNames.index(ff)][index]['FSC-A'])))

        pSTAT1 = pd.concat([Conditions, pSTAT1], axis=1)
        return pSTAT1
    # This is the optional, alternate approach where we look at the distribution within one well
    else:
        for ff in FileNames:
            if ff.find(well+'_') > -1:
                #print("Looking at {}".format(ff))
                index = (AllData[FileNames.index(ff)]['CD19 (PE ( 561 ))'] > 600) & \
                        (AllData[FileNames.index(ff)]['IA_IE (BV510)'] > 2000)
                numcells = index.sum()
                pSTAT1 = pd.DataFrame(np.zeros((numcells, 2)),
                                      columns=[ 'FSC', 'pSTAT1 in B cells'])
                pSTAT1['pSTAT1 in B cells'] = AllData[FileNames.index(ff)][index]['pSTAT1 (FITC)']
                pSTAT1['FSC'] = AllData[FileNames.index(ff)][index]['FSC-A']
        return pSTAT1.dropna()

    # Create dose-response curves for different subpopulations
def grab_subpop_dose_response(q, fname):
    alpha_response_sub_pop = []
    alpha_response_super_pop = []
    alpha_doses = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    for well in ['A11','B11','C11','D11','E11','F11','G11','H11']:
        data = grab_data(fname, well)
        threshold = data['FSC'].quantile(q=q)
        index = data['FSC'] < threshold
        alpha_response_sub_pop.append(data[index]['pSTAT1 in B cells'].mean())
        alpha_response_super_pop.append(data[~index]['pSTAT1 in B cells'].mean())

    beta_response_sub_pop = []
    beta_response_super_pop = []
    beta_doses = [0, 0.2, 6, 20, 60, 200, 600, 2000]
    for well in ['A12', 'B12', 'C12', 'D12', 'E12', 'F12', 'G12', 'H12']:
        data = grab_data('20190214', well)
        threshold = data['FSC'].quantile(q=q)
        index = data['FSC'] < threshold
        beta_response_sub_pop.append(data[index]['pSTAT1 in B cells'].mean())
        beta_response_super_pop.append(data[~index]['pSTAT1 in B cells'].mean())
    return alpha_doses, alpha_response_sub_pop[::-1], alpha_response_super_pop[::-1], \
           beta_doses, beta_response_sub_pop[::-1], beta_response_super_pop[::-1]

if __name__ == '__main__':
    # Well code:
    #  |    2.5 min | 5 min  |  7.5 min | 10 min |  20 min  |    60 min   |
    #   ____1____2____3____4____5____6____7____8____9____10____11____12___
    # A| alpha  beta alpha beta
    #  | 1E-7   2E-9 1E-7  2E-9
    # B| alpha  beta
    #  | 1E-8   6E-10                -----------> increasing
    # C| alpha  beta                |               time
    #  | 3E-9   2E-10               |
    # D| alpha  beta                V
    #  | 1E-9   6E-11            decreasing
    # E| alpha  beta             concentration
    #  | 3E-10  2E-11
    # F| alpha  beta
    #  | 1E-10  6E-12
    # G| alpha  beta
    #  | 1E-11  2E-13
    # H| alpha  beta
    #  | 0 pM   0 pM

    """
    # I want time = 60 min so that means columns 11 (alpha) and 12 (beta)

    # Check the joint distribution of pSTAT1 vs FSC
    data = grab_data('20190214', 'D11')
    outliers = (data['FSC'] > 80000) | (data['pSTAT1 in B cells'] > 30000) | (data['pSTAT1 in B cells'] < -5000)
    sns.jointplot('FSC', 'pSTAT1 in B cells', data=data.loc[~outliers], kind='reg',
                  joint_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.05}})\
        .plot_joint(sns.kdeplot, zorder=0, n_levels=30, color='k')
    plt.suptitle('Dependence on Cell Size at 60 minutes\n' +r'for 1 nM IFN$\alpha$', fontsize=18)
    plt.xlabel('Forward Scatter', fontsize=16)
    plt.ylabel('pSTAT1', fontsize=16)
    plt.show()
    exit()
    """
    """
    # Check distribution of FSC
    # data = grab_data('20190214', 'A11')
    #data.hist('FSC', bins=250)
    #threshold = data['FSC'].quantile(q=0.8)
    #plt.plot([threshold, threshold], [0, 6000], 'r')
    #plt.show()
    a, b, c, d, e, f = grab_subpop_dose_response(0.8, '20190214')
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].set(xscale='log', yscale='linear')
    axes[1].set(xscale='log', yscale='linear')

    axes[0].plot(a, b, 'r--')
    axes[0].plot(a, c, 'k--')
    axes[1].plot(d, e, 'g--')
    axes[1].plot(d, f, 'k--')
    plt.show()
    exit()
    """


    # Get dose-response data
    dataset_names = ['20190121', '20190214', '20190119', '20190108']
    times = [2.5, 2.5, 5.0, 5.0, 7.5, 7.5, 10.0, 10.0, 20.0, 20.0, 60.0, 60.0]
    doses = {'Alpha': np.divide([1E-7, 1E-8, 3E-9, 1E-9, 3E-10, 1E-10, 1E-11][::-1], 1E-12),
             'Beta': np.divide([2E-9, 6E-10, 2E-10, 6E-11, 2E-11, 6E-12, 2E-13][::-1], 1E-12)}
    large_cell_percentile = 0.2
    average_large_fraction_dict = {}
    small_IfnData_list = []
    large_IfnData_list = []
    for dataset in dataset_names:
        column_labels = ['Dose_Species', 'Dose (pM)', 'time', 'pSTAT']
        small_df = []
        large_df = []
        # Gates:
        upper_cutoff = 50000

        average_large_fraction = 0
        for dose_idx, concentration in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G']):
            for time_idx, time in enumerate(times):
                # Annotate
                if time_idx % 2 == 0:
                    species = 'Beta'
                else:
                    species = 'Alpha'
                well = concentration+str(time_idx+1)
                # Clean data
                data = grab_data(dataset, well)
                outliers = (data['FSC'] > upper_cutoff) | (data['pSTAT1 in B cells'] < 0)
                data = data.loc[~outliers]
                small_large_threshold = data.quantile(q=1 - large_cell_percentile).loc['FSC']
                # Partition data
                small_cells = data[(data['FSC'] <= small_large_threshold)]
                large_cells = data[(data['FSC'] > small_large_threshold)]
                average_large_fraction += len(large_cells)/(len(small_cells) + len(large_cells))
                pSTAT_small = small_cells.mean().loc['pSTAT1 in B cells']
                pSTAT_large = large_cells.mean().loc['pSTAT1 in B cells']
                small_df.append([species, doses[species][dose_idx], time, (pSTAT_small, np.nan)])
                large_df.append([species, doses[species][dose_idx], time, (pSTAT_large, np.nan)])
        average_large_fraction_dict[dataset] = average_large_fraction = average_large_fraction/(12*7)

        # Make dataframes
        small_df = pd.DataFrame.from_records(small_df, columns=column_labels)
        small_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
        small_df = pd.pivot_table(small_df, values='pSTAT', index=['Dose_Species', 'Dose (pM)'], columns=['time'],
                                  aggfunc=np.sum)
        small_df.columns.name = None

        large_df = pd.DataFrame.from_records(large_df, columns=column_labels)
        large_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
        large_df = pd.pivot_table(large_df, values='pSTAT', index=['Dose_Species', 'Dose (pM)'], columns=['time'],
                                  aggfunc=np.sum)
        large_df.columns.name = None

        # Make IfnData object
        small_cell_IfnData = IfnData('custom', df=small_df, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
        large_cell_IfnData = IfnData('custom', df=large_df, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
        small_IfnData_list.append(small_cell_IfnData)
        large_IfnData_list.append(large_cell_IfnData)

    # Check that thresholding worked (it works)
    #print(average_large_fraction_dict)

    # EC50 lists
    #ec50_small_cells = {dataset_names[i]: small_cell_IfnData[i].get_ec50s() for i in range(len(dataset_names))}
    #ec50_large_cells = {dataset_names[i]: large_cell_IfnData[i].get_ec50s() for i in range(len(dataset_names))}

    # Align data
    small_alignment = DataAlignment()
    small_alignment.add_data(small_cell_IfnData)
    small_alignment.align()
    small_alignment.get_scaled_data()
    mean_small_data = small_alignment.summarize_data()

    large_alignment = DataAlignment()
    large_alignment.add_data(large_cell_IfnData)
    large_alignment.align()
    large_alignment.get_scaled_data()
    mean_large_data = large_alignment.summarize_data()

    # ----------------------
    # Set up Figure layout
    # ----------------------
    Figure_3 = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 1])
    Figure_3.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()

    # Set up dose response figures
    new_fit = DoseresponsePlot((1, 2))
    new_fit.fig = Figure_3
    plt.figure(Figure_3.number)
    new_fit.axes = [Figure_3.add_subplot(gs[0, 0]), Figure_3.add_subplot(gs[0, 1])]
    new_fit.axes[0].set_xscale('log')
    new_fit.axes[0].set_xlabel('Dose (pM)')
    new_fit.axes[0].set_ylabel('pSTAT (MFI)')
    new_fit.axes[1].set_xscale('log')
    new_fit.axes[1].set_xlabel('Dose (pM)')
    new_fit.axes[1].set_ylabel('pSTAT (MFI)')

    # Plot Dose respsonse data
    times = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
    alpha_palette = sns.color_palette("deep", 6)
    beta_palette = sns.color_palette("deep", 6)
    alpha_mask = [7.5]
    beta_mask = [5.0, 7.5, 10.0]
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(mean_large_data, t, 'errorbar', 'o--', (0, 0), 'Alpha', color=alpha_palette[idx],
                                   label='{} min'.format(t))
            new_fit.add_trajectory(mean_small_data, t, 'errorbar', 'o-', (0, 0), 'Alpha', color=alpha_palette[idx])
        if t not in beta_mask:
            new_fit.add_trajectory(mean_large_data, t, 'errorbar', 'o--', (0, 1), 'Beta', color=beta_palette[idx],
                                   label='{} min'.format(t))
            new_fit.add_trajectory(mean_small_data, t, 'errorbar', 'o-', (0, 1), 'Beta', color=beta_palette[idx])

    # Set up EC50 figures
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)
    data_palette = sns.color_palette("muted", 6)
    marker_shape = ["o", "v", "s", "P", "d", "1", "x", "*"]
    # Get EC50s
    small_ec50, small_errorbars = small_alignment.get_ec50s()
    large_ec50, large_errorbars = large_alignment.get_ec50s()
    # Plot EC50 vs time
    ec50_axes = [Figure_3.add_subplot(gs[1, 0]), Figure_3.add_subplot(gs[1, 1])]
    ec50_axes[0].set_xlabel("Time (min)")
    ec50_axes[1].set_xlabel("Time (min)")
    ec50_axes[0].set_title(r"EC50 vs Time for IFN$\alpha$")
    ec50_axes[1].set_title(r"EC50 vs Time for IFN$\beta$")
    ec50_axes[0].set_ylabel("EC50 (pM)")
    ec50_axes[0].set_yscale('log')
    ec50_axes[1].set_yscale('log')
    # Add data
    line_style_idx=0
    line_styles = ['-', '--']
    for ec50, errorbars, fmt in [[small_ec50, small_errorbars], [large_ec50, large_errorbars]]:
        ec50_axes[0].errorbar([el[0] for el in ec50['Alpha']], [el[1] for el in ec50['Alpha']],
                              yerr=errorbars['Alpha'], fmt=line_styles[line_style_idx], color=data_palette[3])
        ec50_axes[1].errorbar([el[0] for el in ec50['Beta']], [el[1] for el in ec50['Beta']],
                              yerr=errorbars['Beta'], fmt=line_styles[line_style_idx], color=data_palette[3])
        line_style_idx += 1

    plt.show()