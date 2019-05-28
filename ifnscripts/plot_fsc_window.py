from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnplot import DoseresponsePlot
from ifnclass.ifnmodel import IfnModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
font = {'family': 'normal',
        'weight': 'normal',
        'size': 14}
matplotlib.rc('font', **font)
import seaborn as sns
import os
import warnings
import fcsparser
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def grab_data(fname, well=None):
    path_RawFiles = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
                                 "ifndatabase", "{}_raw_data".format(fname), "Compensated Export")

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
        meta, df = fcsparser.parse(os.path.join(path_RawFiles, ff), reformat_meta=True)
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
            pSTAT1.loc[ff]['pSTAT1 in B cells'] = np.arcsinh(
                np.mean(np.sinh(AllData[FileNames.index(ff)][index]['pSTAT1 (FITC)'])))
            pSTAT1.loc[ff]['FSC'] = np.arcsinh(np.mean(np.sinh(AllData[FileNames.index(ff)][index]['FSC-A'])))

        pSTAT1 = pd.concat([Conditions, pSTAT1], axis=1)
        return pSTAT1
    # This is the optional, alternate approach where we look at the distribution within one well
    else:
        for ff in FileNames:
            if ff.find(well + '_') > -1:
                # print("Looking at {}".format(ff))
                index = (AllData[FileNames.index(ff)]['CD19 (PE ( 561 ))'] > 600) & \
                        (AllData[FileNames.index(ff)]['IA_IE (BV510)'] > 2000)
                numcells = index.sum()
                pSTAT1 = pd.DataFrame(np.zeros((numcells, 2)),
                                      columns=['FSC', 'pSTAT1 in B cells'])
                pSTAT1['pSTAT1 in B cells'] = AllData[FileNames.index(ff)][index]['pSTAT1 (FITC)']
                pSTAT1['FSC'] = AllData[FileNames.index(ff)][index]['FSC-A']
        return pSTAT1.dropna()

    # Create dose-response curves for different subpopulations


def grab_subpop_dose_response(q, fname):
    alpha_response_sub_pop = []
    alpha_response_super_pop = []
    alpha_doses = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    for well in ['A11', 'B11', 'C11', 'D11', 'E11', 'F11', 'G11', 'H11']:
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

    # Get dose-response data
    dataset_names = ['20190214', '20190121', '20190119', '20190108']
    time = 60.0
    well_ID = 'G11'
    dose = 10
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    small_IfnData_list = []

    for dataset in dataset_names:
        print('Working on dataset {}'.format(dataset))
        small_df = []
        large_df = []
        # Gates:
        upper_cutoff = 50000
        # Clean data
        data = grab_data(dataset, well_ID)
        outliers = (data['FSC'] > upper_cutoff) | (data['pSTAT1 in B cells'] < 0)
        data = data.loc[~outliers]
        for window in range(1, len(quantiles)):
            window_bottom = data.quantile(q=quantiles[window-1]).loc['FSC']
            window_top = data.quantile(q=quantiles[window]).loc['FSC']
            mean_fsc_in_window = (window_bottom + window_top) / 2.0
            # Partition data
            small_cells = data[(data['FSC'] > window_bottom)&(data['FSC'] <= window_top)]
            pSTAT_small = small_cells.mean().loc['pSTAT1 in B cells']
            # Append data
            small_df.append((mean_fsc_in_window, pSTAT_small))
        small_IfnData_list.append(small_df)

    # Align datasets
    def __score_sf__(scf, data, reftable):
        diff_table = np.subtract(reftable, np.multiply(data, scf[0]))
        return np.sum(np.square(diff_table))
    reference_table = [el[1] for el in small_IfnData_list[0]]
    print(reference_table)
    sf_list = []
    for row in small_IfnData_list[1:]:
        datatable = [el[1] for el in row]
        print(datatable)
        opt = minimize(__score_sf__, [2.0], args=(datatable, reference_table))
        print(opt['x'])
        sf_list.append(opt['x'])
    aligned_data = [small_IfnData_list[0]]
    print(sf_list)
    for idx in range(len(sf_list)):
        x_values = [el[0] for el in small_IfnData_list[idx]]
        y_values = np.multiply(sf_list[idx], [el[1] for el in small_IfnData_list[idx]])
        aligned_data.append(list(zip(x_values, y_values)))

    # Mean response
    mean_window_values = []
    mean_response = []
    y_variances = []
    x_variances = []
    for cidx in range(len(aligned_data[1])):
        windows = []
        responses = []
        for ridx in range(len(aligned_data)):
            windows.append(aligned_data[ridx][cidx][0])
            responses.append(aligned_data[ridx][cidx][1])
        print(responses)
        mean_response.append(np.mean(responses))
        mean_window_values.append(np.mean(windows))
        y_variances.append(np.std(responses))
        x_variances.append(np.std(windows))

    # Model
    # -------------------------------
    # Initialize model
    # -------------------------------
    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
    initial_parameters = {'k_a1': 4.98E-14 * 2, 'k_a2': 1.328e-12, 'k_d3': 2.4e-06, 'k_d4': 0.228,
                          'kSOCSon': 8e-07, 'kpu': 0.0011,
                          'ka1': 3.3e-15, 'ka2': 1.22e-12, 'kd4': 0.86,
                          'kd3': 1.74e-05,
                          'kint_a': 0.000124, 'kint_b': 0.00086,
                          'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05}
    dual_parameters = {'kint_a': 0.00052, 'kSOCSon': 6e-07, 'kint_b': 0.00052, 'krec_a1': 0.001, 'krec_a2': 0.1,
                       'krec_b1': 0.005, 'krec_b2': 0.05}
    Mixed_Model.set_parameters(initial_parameters)
    Mixed_Model.set_parameters(dual_parameters)

    scale_factor = 1.227
    times = [60.0]
    doses_alpha = np.logspace(0, 7)

    # -------------------------------
    # Scanning effect of cell size
    # 60 minute IFN dosed at 10 pM
    # -------------------------------
    alpha_cell_size_curve = []
    beta_cell_size_curve = []
    volPM_typical = 2 * 30E-6 ** 2 + 4 * 30E-6 * 8E-6
    volCP_typical = 8E-6 * 30E-6 ** 2
    radii = list(np.logspace(np.log10(30E-7), np.log10(30E-5)))
    for radius in radii:
        volPM = 2 * radius ** 2 + 4 * radius * 8E-6
        volCP = 8E-6 * radius ** 2
        # Alpha
        response = Mixed_Model.timecourse(list(np.linspace(0, 60)), 'TotalpSTAT',
                                          parameters={'Ia': 10E-12 * 6.022E23 * 1E-5, 'Ib': 0,
                                                      'R1': (volPM / volPM_typical) * 1200,
                                                      'R2': (volPM / volPM_typical) * 4920,
                                                      'S': (volCP / volCP_typical) * 1E4},
                                          return_type='list', scale_factor=scale_factor)['TotalpSTAT'][-1]
        normalized_response = response / ((volCP / volCP_typical) * 1E4)
        alpha_cell_size_curve.append(normalized_response)

    # -------------------------------
    # Plot cell size panel
    # -------------------------------
    alpha_palette = sns.color_palette("Reds", 8)
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    dr_fig = DoseresponsePlot((1, 2))
    ax = dr_fig.axes[0]
    xlabels = np.divide(radii, 1E-6)
    ax.plot(xlabels, alpha_cell_size_curve, color=alpha_palette[-1], label='Alpha', linewidth=2)
    ax.legend()
    ax.set_xlabel(r'Cell radius ($\mu$m)', fontsize=14)
    ax.set_ylabel('pSTAT/STAT', fontsize=14)
    ax.set_xlim((1, 300))
    ax.set_title('Fraction pSTAT vs Cell Radius\n 10 pM IFN at 60 minutes', fontsize=16)

    ax = dr_fig.axes[1]
    ax.errorbar(mean_window_values, mean_response, xerr=x_variances, yerr=y_variances, fmt='o--',
                color=alpha_palette[-1], alpha=0.75)
    ax.set_xlabel('Mean FSC')
    ax.set_ylabel('Mean MFI')
    ax.set_title('pSTAT1 MFI vs FSC')
    ax.set_xscale('linear')

    dr_fig.show_figure()
"""
        # Convert to multiindex
        small_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
        small_df = pd.pivot_table(small_df, values='pSTAT', index=['Dose_Species', 'Dose (pM)'], columns=['time'],
                                  aggfunc=np.sum)
        small_df.columns.name = None

        large_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
        large_df = pd.pivot_table(large_df, values='pSTAT', index=['Dose_Species', 'Dose (pM)'], columns=['time'],
                                  aggfunc=np.sum)
        large_df.columns.name = None

        # Drop zero dose data
        #small_df = small_df.drop(index=0.0, level=1)
        #large_df = large_df.drop(index=0.0, level=1)

        # Make IfnData object
        small_cell_IfnData = IfnData('custom', df=small_df, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
        small_cell_IfnData.name = dataset
        large_cell_IfnData = IfnData('custom', df=large_df, conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
        large_cell_IfnData.name = dataset
        small_IfnData_list.append(small_cell_IfnData)
        large_IfnData_list.append(large_cell_IfnData)
    # Check that thresholding worked (it works)
    #print(average_large_fraction_dict)

    # Align data
    small_alignment = DataAlignment()
    small_alignment.add_data(small_IfnData_list)
    small_alignment.align()
    small_alignment.get_scaled_data()
    mean_small_data = small_alignment.summarize_data()

    large_alignment = DataAlignment()
    large_alignment.add_data(large_IfnData_list)
    large_alignment.align()
    large_alignment.get_scaled_data()
    mean_large_data = large_alignment.summarize_data()

    # Save results
    small_alignment.save('small_alignment', save_dir=os.path.join(os.getcwd(), 'small_alignment'))
    large_alignment.save('large_alignment', save_dir=os.path.join(os.getcwd(), 'large_alignment'))
"""
