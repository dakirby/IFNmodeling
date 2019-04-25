from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import DoseresponsePlot, TimecoursePlot
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ifnclass.ifnfit import StepwiseFit
import os


if __name__ == '__main__':
    # -------------------------------
    # Initialize model
    # -------------------------------
    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
    Mixed_Model.set_parameters({'R2': 4920, 'R1': 1200,
                                'k_a1': 2.0e-13, 'k_a2': 1.328e-12, 'k_d3': 1.13e-4, 'k_d4': 0.9,
                                'kSOCSon': 5e-08, 'kpu': 0.0022, 'kpa': 2.36e-06,
                                'ka1': 3.3e-15, 'ka2': 1.85e-12, 'kd4': 2.0,
                                'kd3': 6.52e-05,
                                'kint_a': 0.0015, 'kint_b': 0.002,
                                'krec_a1': 0.01, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05})
    scale_factor = 1.46182313424
    times = [60.0]
    doses_alpha = np.logspace(0, 7)
    doses_beta = np.logspace(0, 6)
    # --------------------------------------------------------------
    # Two dose response curves
    # Large cells are 20% of population
    # Small cells have 20% R1 and R2 but are 80% of the population
    # --------------------------------------------------------------
    # Small cells
    radius = 1E-6
    volPM = 2 * radius ** 2 + 4 * radius * 8E-6
    volCP = 8E-6 * radius ** 2
    small_cells_alpha = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia',
                                                 list(np.logspace(np.log10(doses_alpha[0]), np.log10(doses_alpha[-1]))),
                                                 parameters={'Ib': 0,
                                                             'R1': 1200 * 0.20,
                                                             'R2': 4920 * 0.20},
                                                 return_type='dataframe', dataframe_labels='Alpha',
                                                 scale_factor=scale_factor * 0.8)
    small_cells_beta = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib',
                                                list(np.logspace(np.log10(doses_beta[0]), np.log10(doses_beta[-1]))),
                                                parameters={'Ia': 0,
                                                            'R1': 1200 * 0.20,
                                                            'R2': 4920 * 0.20},
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
                                                             'R1': 1200,
                                                             'R2': 4920},
                                                 return_type='dataframe', dataframe_labels='Alpha',
                                                 scale_factor=scale_factor * 0.2)
    large_cells_beta = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib',
                                                list(np.logspace(np.log10(doses_beta[0]), np.log10(doses_beta[-1]))),
                                                parameters={'Ia': 0,
                                                            'R1': 1200,
                                                            'R2': 4920},
                                                return_type='dataframe', dataframe_labels='Beta',
                                                scale_factor=scale_factor * 0.2)
    large_cells_alpha_IFNdata = IfnData('custom', df=large_cells_alpha, conditions={'Alpha': {'Ib': 0}})
    large_cells_beta_IFNdata = IfnData('custom', df=large_cells_beta, conditions={'Beta': {'Ia': 0}})

    # Plot
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    dr_plot = DoseresponsePlot((1, 3))
    # Add fits
    dr_plot.add_trajectory(large_cells_alpha_IFNdata, 60.0, 'plot', alpha_palette[5], (0, 0), 'Alpha',
                           label='Large Cells at 60 minutes', linewidth=2)
    dr_plot.add_trajectory(small_cells_alpha_IFNdata, 60.0, 'plot', alpha_palette[2], (0, 0), 'Alpha',
                           label='Small Cells at 60 minutes', linewidth=2)

    dr_plot.add_trajectory(large_cells_beta_IFNdata, 60.0, 'plot', beta_palette[5], (0, 1), 'Beta',
                           label='Large Cells at 60 minutes', linewidth=2)
    dr_plot.add_trajectory(small_cells_beta_IFNdata, 60.0, 'plot', beta_palette[2], (0, 1), 'Beta',
                           label='Small Cells at 60 minutes', linewidth=2)

    dr_fig, dr_axes = dr_plot.show_figure(save_flag=False)
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
                                          parameters={'Ia': 20 * 1E-12 * 6.022E23 * 1E-5, 'Ib': 0,
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

    # Plot
    alpha_palette = sns.color_palette("Reds", 8)
    beta_palette = sns.color_palette("Greens", 8)

    ax = dr_axes[2]
    ax.set_xscale('log')
    xlabels = np.divide(radii, 1E-6)
    ax.plot(xlabels, alpha_cell_size_curve, color=alpha_palette[-1], label='Alpha')
    ax.plot(xlabels, beta_cell_size_curve, color=beta_palette[-1], label='Beta')
    ax.legend()
    ax.set_xlabel(r'Cell radius ($\mu$m)')
    ax.set_ylabel('pSTAT/STAT')
    ax.set_xlim((1, 300))
    ax.set_title('Fraction pSTAT vs Cell Radius\n 20 pM IFN at 60 minutes')

    dr_fig.set_size_inches(24, 8)
    dr_fig.savefig(os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_4', 'Figure_4.pdf'))
