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

    scale_factor = 1.227
    times = [60.0]
    doses_alpha = np.logspace(0, 7)
    doses_beta = np.logspace(0, 6)

    # -------------------------------
    # Scanning effect of cell size
    # 60 minute IFN dosed at 20 pM
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
                                          parameters={'Ia': 1E-9 * 6.022E23 * 1E-5, 'Ib': 0,
                                                      'R1': (volPM / volPM_typical) * 1200,
                                                      'R2': (volPM / volPM_typical) * 4920,
                                                      'S': (volCP / volCP_typical) * 1E4},
                                          return_type='list', scale_factor=scale_factor)['TotalpSTAT'][-1]
        normalized_response = response / ((volCP / volCP_typical) * 1E4)
        alpha_cell_size_curve.append(normalized_response)
        # Beta
        response = Mixed_Model.timecourse(list(np.linspace(0, 60)), 'TotalpSTAT',
                                          parameters={'Ib': 20 * 1E-12 * 6.022E23 * 1E-5, 'Ia': 0,
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
    ax.set_title('Fraction pSTAT vs Cell Radius\n 1 nM IFN at 60 minutes', fontsize=16)

    # -----------------------------
    # Investigate cell variability
    # in data
    # -----------------------------
    # Well code:
    #   ____1____2____3____4____5____6____7____8____9____10____11____12___
    # A| alpha  beta alpha beta
    #  | 1E-7   2E-9 1E-7  1E-9
    # B| alpha
    #  | 1E-8                        -----------> increasing
    # C| alpha                      |               time
    #  | 3E-9                       |
    # D| alpha                      V
    #  | 1E-9                    decreasing
    # E| alpha                   concentration
    #  | 3E-10
    # F| alpha
    #  | 1E-10
    # G| alpha
    #  | 1E-11
    # H| alpha
    #  | 0 pM
    dr_fig.axes[1].set_xscale('linear')
    data = grab_data('20190214', 'D11')
    outliers = (data['FSC'] > 80000) | (data['pSTAT1 in B cells'] > 30000) | (data['pSTAT1 in B cells'] < -5000)
    #sns.regplot('FSC', 'pSTAT1 in B cells', data=data.loc[~outliers], ax=dr_fig.axes[1],
    #              line_kws={'color': 'red'}, scatter_kws={'alpha': 0.05, 'edgecolor':''})
    dr_fig.axes[1].set_title('Dependence on Cell Size at 60 minutes\n' + r'for 1 nM IFN$\alpha$', fontsize=16)
    dr_fig.axes[1].set_xlabel('Forward Scatter', fontsize=14)
    dr_fig.axes[1].set_ylabel('pSTAT1', fontsize=14)
    scatter_density_plot(x=data.loc[~outliers]['FSC'], y=data.loc[~outliers]['pSTAT1 in B cells'],
                         ax=dr_fig.axes[1], bins=50)
    plt.show()

    # ------------------------------
    dr_fig.fig.set_size_inches(16, 8)
    dr_fig.fig.savefig(os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_3', 'Figure_3.pdf'))
"""
    # ------------------------------------------------------------------------------
    # Population heterogeneity
    # Large cells are 20% of population and have {'R1': 6755.56, 'R2': 1511.1}
    # Small cells are 80% of the population and have {'R1': 12000.0, 'R2': 1511.1}
    # ------------------------------------------------------------------------------
    # Small cells
    radius = 1E-6
    volPM = 2 * radius ** 2 + 4 * radius * 8E-6
    volCP = 8E-6 * radius ** 2
    small_cells_alpha = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia',
                                                 list(np.logspace(np.log10(doses_alpha[0]), np.log10(doses_alpha[-1]))),
                                                 parameters={'Ib': 0,
                                                             'R1': 12000,
                                                             'R2': 1511},
                                                 return_type='dataframe', dataframe_labels='Alpha',
                                                 scale_factor=scale_factor * 0.8)
    small_cells_beta = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib',
                                                list(np.logspace(np.log10(doses_beta[0]), np.log10(doses_beta[-1]))),
                                                parameters={'Ia': 0,
                                                            'R1': 12000,
                                                            'R2': 1511},
                                                return_type='dataframe', dataframe_labels='Beta',
                                                scale_factor=scale_factor * 0.8)
    small_cells_alpha_IFNdata = IfnData('custom', df=small_cells_alpha, conditions={'Alpha': {'Ib': 0}})
    small_cells_beta_IFNdata = IfnData('custom', df=small_cells_beta, conditions={'Beta': {'Ia': 0}})

    # Large (normal) cells
    radius = 30E-6
    volPM = 2 * radius ** 2 + 4 * radius * 8E-6
    volCP = 8E-6 * radius ** 2
    large_cells_alpha = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia',
                                                 list(np.logspace(np.log10(doses_alpha[0]), np.log10(doses_alpha[-1]))),
                                                 parameters={'Ib': 0,
                                                             'R1': 6755,
                                                             'R2': 1511},
                                                 return_type='dataframe', dataframe_labels='Alpha',
                                                 scale_factor=scale_factor * 0.2)
    large_cells_beta = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib',
                                                list(np.logspace(np.log10(doses_beta[0]), np.log10(doses_beta[-1]))),
                                                parameters={'Ia': 0,
                                                            'R1': 6755,
                                                            'R2': 1511},
                                                return_type='dataframe', dataframe_labels='Beta',
                                                scale_factor=scale_factor * 0.2)
    large_cells_alpha_IFNdata = IfnData('custom', df=large_cells_alpha, conditions={'Alpha': {'Ib': 0}})
    large_cells_beta_IFNdata = IfnData('custom', df=large_cells_beta, conditions={'Beta': {'Ia': 0}})

    # Plot
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    dr_plot = DoseresponsePlot((1, 2))
    # Add fits
    dr_plot.add_trajectory(large_cells_alpha_IFNdata, 60.0, 'plot', alpha_palette[5], (0, 1), 'Alpha',
                           label=r'IFN$\alpha$ Large Cells (80% of population)', linewidth=2)
    dr_plot.add_trajectory(small_cells_alpha_IFNdata, 60.0, 'plot', alpha_palette[2], (0, 1), 'Alpha',
                           label=r'IFN$\alpha$ Small Cells (20% of population)', linewidth=2)

    dr_plot.add_trajectory(large_cells_beta_IFNdata, 60.0, 'plot', beta_palette[5], (0, 1), 'Beta',
                           label=r'IFN$\beta$ Large Cells (80% of population)', linewidth=2)
    dr_plot.add_trajectory(small_cells_beta_IFNdata, 60.0, 'plot', beta_palette[2], (0, 1), 'Beta',
                           label=r'IFN$\beta$ Small Cells (20% of population)', linewidth=2)

    dr_fig, dr_axes = dr_plot.show_figure(save_flag=False)
    dr_axes[0].set_title('Breakdown of heterogeneous population\nat 60 minutes')
    """