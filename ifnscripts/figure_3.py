from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import DoseresponsePlot, TimecoursePlot
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interpn
import os
import warnings
import fcsparser

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


def scatter_density_plot(x , y, ax=None, sort=True, bins=20, **kwargs ):
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins=bins)
    z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])), data, np.vstack([x, y]).T,
                method="splinef2d", bounds_error=False)

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, **kwargs)
    return ax


if __name__ == '__main__':
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

    scale_factor = 0.613 #1.227
    times = [60.0]
    doses_alpha = np.divide([0, 1E-7, 1E-8, 3E-9, 1E-9, 3E-10, 1E-10, 1E-11], 1E-12)
    doses_beta = np.divide([0, 2E-9, 6E-10, 2E-10, 6E-11, 2E-11, 6E-12, 2E-13], 1E-12)
    """
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
        # Beta
        response = Mixed_Model.timecourse(list(np.linspace(0, 60)), 'TotalpSTAT',
                                          parameters={'Ib': 10 * 1E-12 * 6.022E23 * 1E-5, 'Ia': 0,
                                                      'R1': (volPM / volPM_typical) * 1200,
                                                      'R2': (volPM / volPM_typical) * 4920,
                                                      'S': (volCP / volCP_typical) * 1E4},
                                          return_type='list', scale_factor=scale_factor)['TotalpSTAT'][-1]
        normalized_response = response / ((volCP / volCP_typical) * 1E4)
        beta_cell_size_curve.append(normalized_response)

    # -------------------------------
    # Plot cell size panel
    # -------------------------------
    alpha_palette = sns.color_palette("Reds", 8)
    beta_palette = sns.color_palette("Greens", 8)
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    dr_fig = DoseresponsePlot((1, 2))
    ax = dr_fig.axes[0]
    xlabels = np.divide(radii, 1E-6)
    ax.plot(xlabels, alpha_cell_size_curve, color=alpha_palette[-1], label='Alpha', linewidth=2)
    ax.plot(xlabels, beta_cell_size_curve, color=beta_palette[-1], label='Beta', linewidth=2)
    ax.legend()
    ax.set_xlabel(r'Cell radius ($\mu$m)', fontsize=14)
    ax.set_ylabel('pSTAT/STAT', fontsize=14)
    ax.set_xlim((1, 300))
    ax.set_title('Fraction pSTAT vs Cell Radius\n 10 pM IFN at 60 minutes', fontsize=16)
    """
    # -----------------------------
    # Investigate cell variability
    # in data
    # -----------------------------
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
    # Get dose-response data
    dataset_names = ['20190214', '20190121', '20190119', '20190108']
    times = [2.5, 2.5, 5.0, 5.0, 7.5, 7.5, 10.0, 10.0, 20.0, 20.0, 60.0, 60.0]
    well_IDs = ['H', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    doses = {'Alpha': np.divide([0, 1E-7, 1E-8, 3E-9, 1E-9, 3E-10, 1E-10, 1E-11], 1E-12),
            'Beta': np.divide([0, 2E-9, 6E-10, 2E-10, 6E-11, 2E-11, 6E-12, 2E-13], 1E-12)}
    column_labels = ['Dose_Species', 'Dose (pM)', 'time', 'pSTAT']
    large_cell_percentile = 0.2
    average_large_fraction_dict = {}
    small_IfnData_list = []
    large_IfnData_list = []
    for dataset in dataset_names:
        print("Working on dataset {}".format(dataset))
        small_df = []
        large_df = []
        small_zeros = []
        large_zeros = []
        # Gates:
        upper_cutoff = 50000

        average_large_fraction = 0
        for dose_idx, concentration in enumerate(well_IDs):
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
                # Recognize zero dose data and set aside
                if doses[species][dose_idx] == 0.0:
                    small_zeros.append([species, doses[species][dose_idx], time, pSTAT_small])
                    large_zeros.append([species, doses[species][dose_idx], time, pSTAT_large])
                else:
                    small_zero_value = 0
                    for item in small_zeros:
                        if item[0] == species and item[1] == 0.0 and item[2] == time:
                            small_zero_value = item[3]
                            break
                    large_zero_value = 0
                    for item in large_zeros:
                        if item[0] == species and item[1] == 0.0 and item[2] == time:
                            large_zero_value = item[3]
                            break
                    small_df.append([species, doses[species][dose_idx], time, (pSTAT_small-small_zero_value, np.nan)])
                    large_df.append([species, doses[species][dose_idx], time, (pSTAT_large-large_zero_value, np.nan)])
        average_large_fraction_dict[dataset] = average_large_fraction = average_large_fraction/(len(times)*len(well_IDs))
        # Make dataframes
        small_df = pd.DataFrame.from_records(small_df, columns=column_labels)
        large_df = pd.DataFrame.from_records(large_df, columns=column_labels)

        # Convert to multiindex
        small_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
        small_df = pd.pivot_table(small_df, values='pSTAT', index=['Dose_Species', 'Dose (pM)'], columns=['time'],
                                 aggfunc=np.sum)
        small_df.columns.name = None

        large_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
        large_df = pd.pivot_table(large_df, values='pSTAT', index=['Dose_Species', 'Dose (pM)'], columns=['time'],
                                 aggfunc=np.sum)
        large_df.columns.name = None

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

    # Load saved DataAlignment
    small_alignment = DataAlignment()
    small_alignment.load_from_save_file('small_alignment', os.path.join(os.getcwd(), 'small_alignment'))
    large_alignment = DataAlignment()
    large_alignment.load_from_save_file('large_alignment', os.path.join(os.getcwd(), 'large_alignment'))
    small_alignment.align()
    small_alignment.get_scaled_data()
    mean_small_data = small_alignment.summarize_data()
    large_alignment.align()
    large_alignment.get_scaled_data()
    mean_large_data = large_alignment.summarize_data()

    # ----------------------
    # Set up Figure layout
    # ----------------------
    # Set up dose response figures
    new_fit = DoseresponsePlot((1, 2))
    new_fit.axes[0].set_ylabel('pSTAT (MFI)')
    new_fit.axes[1].set_ylabel('pSTAT (MFI)')

    # Plot Dose respsonse data
    times = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
    color_palette = sns.color_palette("Paired", 10)
    alpha_mask = [2.5, 5.0, 7.5, 20.0]
    beta_mask = [2.5, 5.0, 7.5, 20.0]
    for idx, t in enumerate([10.0, 60.0]):
        if t is 10.0:
            new_fit.add_trajectory(mean_large_data, t, 'errorbar', 'o', (0, 0), 'Alpha', color=color_palette[4],
                                   linewidth=2.0)
            new_fit.add_trajectory(mean_small_data, t, 'errorbar', 'o', (0, 0), 'Alpha', color=color_palette[5],
                                   linewidth=2.0)
            new_fit.add_trajectory(mean_large_data, t, 'errorbar', 'o', (0, 1), 'Beta', color=color_palette[2],
                                   linewidth=2.0)
            new_fit.add_trajectory(mean_small_data, t, 'errorbar', 'o', (0, 1), 'Beta', color=color_palette[3],
                                   linewidth=2.0)
        else:
            new_fit.add_trajectory(mean_large_data, t, 'errorbar', 'o', (0, 0), 'Alpha', color=color_palette[6],
                                   linewidth=2.0)
            new_fit.add_trajectory(mean_small_data, t, 'errorbar', 'o', (0, 0), 'Alpha', color=color_palette[7],
                                   linewidth=2.0)
            new_fit.add_trajectory(mean_large_data, t, 'errorbar', 'o', (0, 1), 'Beta', color=color_palette[0],
                                   linewidth=2.0)
            new_fit.add_trajectory(mean_small_data, t, 'errorbar', 'o', (0, 1), 'Beta', color=color_palette[1],
                                   linewidth=2.0)

    # ------------------------------------------------------------------------------
    # Population heterogeneity
    # Large cells are 20% of population and have {'R1': 6755.56, 'R2': 1511.1}
    # Small cells are 80% of the population and have {'R1': 12000.0, 'R2': 1511.1}
    # ------------------------------------------------------------------------------
    times = [2.5, 5.0, 7.5, 10., 20.0, 60.0]
    alpha_doses = list(np.logspace(np.log10(doses_alpha[1]), np.log10(doses_alpha[-1])))
    beta_doses = list(np.logspace(np.log10(doses_beta[1]), np.log10(doses_beta[-1])))
    # Small cells
    radius = 1E-6
    volPM_small = 2 * radius ** 2 + 4 * radius * 8E-6
    volCP_small = 8E-6 * radius ** 2
    R1 = 6755
    R2 = 1511
    STAT = 10000
    small_cells_alpha = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia',
                                                 alpha_doses,
                                                 parameters={'Ib': 0,
                                                             'R1': R1,
                                                             'R2': R2,
                                                             'S': STAT},
                                                 return_type='dataframe', dataframe_labels='Alpha',
                                                 scale_factor=scale_factor)
    small_cells_beta = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib',
                                                beta_doses,
                                                parameters={'Ia': 0,
                                                            'R1': R1,
                                                            'R2': R2,
                                                            'S': STAT},
                                                return_type='dataframe', dataframe_labels='Beta',
                                                scale_factor=scale_factor)
    small_cells_alpha_IFNdata = IfnData('custom', df=small_cells_alpha, conditions={'Alpha': {'Ib': 0}})
    small_cells_beta_IFNdata = IfnData('custom', df=small_cells_beta, conditions={'Beta': {'Ia': 0}})

    # Large (normal) cells
    radius = 1.6**0.5 * radius
    volPM_large = 2 * radius ** 2 + 4 * radius * 8E-6
    volCP_large = 8E-6 * radius ** 2
    R1 = R1 * volPM_large / volPM_small
    R2 = R2 * volPM_large / volPM_small
    STAT = STAT * volCP_large / volCP_small
    large_cells_alpha = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia',
                                                 alpha_doses,
                                                 parameters={'Ib': 0,
                                                             'R1': R1,
                                                             'R2': R2,
                                                             'S': STAT},
                                                 return_type='dataframe', dataframe_labels='Alpha',
                                                 scale_factor=scale_factor)
    large_cells_beta = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib',
                                                beta_doses,
                                                parameters={'Ia': 0,
                                                            'R1': R1,
                                                            'R2': R2,
                                                            'S': STAT},
                                                return_type='dataframe', dataframe_labels='Beta',
                                                scale_factor=scale_factor)
    large_cells_alpha_IFNdata = IfnData('custom', df=large_cells_alpha, conditions={'Alpha': {'Ib': 0}})
    large_cells_beta_IFNdata = IfnData('custom', df=large_cells_beta, conditions={'Beta': {'Ia': 0}})

    # Plot

    dr_plot = new_fit
    dr_axes = dr_plot.axes
    # Add model predictions fits
    # Alpha
    dr_plot.add_trajectory(large_cells_alpha_IFNdata, 60.0, 'plot', '--', (0, 0), 'Alpha', color=color_palette[6],
                           label=r'Large Cells, 60 min', linewidth=2)
    dr_plot.add_trajectory(small_cells_alpha_IFNdata, 60.0, 'plot', color_palette[7], (0, 0), 'Alpha',
                           label=r'Small Cells, 60 min', linewidth=2)
    dr_plot.add_trajectory(large_cells_alpha_IFNdata, 10.0, 'plot', '--', (0, 0), 'Alpha', color=color_palette[4],
                           label=r'Large Cells, 10 min', linewidth=2)
    dr_plot.add_trajectory(small_cells_alpha_IFNdata, 10.0, 'plot', color_palette[5], (0, 0), 'Alpha',
                           label=r'Small Cells, 10 min', linewidth=2)
    # Beta
    dr_plot.add_trajectory(large_cells_beta_IFNdata, 60.0, 'plot', '--', (0, 1), 'Beta', color=color_palette[0],
                           label=r'Large Cells, 60 min', linewidth=2)
    dr_plot.add_trajectory(small_cells_beta_IFNdata, 60.0, 'plot', color_palette[1], (0, 1), 'Beta',
                           label=r'Small Cells, 60 min', linewidth=2)
    dr_plot.add_trajectory(large_cells_beta_IFNdata, 10.0, 'plot', '--', (0, 1), 'Beta', color=color_palette[2],
                           label=r'Large Cells, 10 min', linewidth=2)
    dr_plot.add_trajectory(small_cells_beta_IFNdata, 10.0, 'plot', color_palette[3], (0, 1), 'Beta',
                           label=r'Small Cells, 10 min', linewidth=2)

    dr_axes[0].set_title(r'IFN$\alpha$')
    dr_axes[1].set_title(r'IFN$\beta$')

    dr_fig, dr_axes = dr_plot.show_figure(save_flag=False)
    dr_fig.set_size_inches(12, 5)
    dr_fig.savefig(os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_3', 'Figure_3.pdf'))
