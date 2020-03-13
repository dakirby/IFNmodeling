from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnfit import DualMixedPopulation
from numpy import linspace, logspace, transpose
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from ifnclass.ifnplot import Trajectory, DoseresponsePlot


# ------------------
# Import ImmGen
# ------------------
#ImmGen_df = pd.read_excel('ImmGen_signaling_with_protein_response.xlsx', sheet_name='Sheet1', axis=1)
ImmGen_raw_df = pd.read_csv('ImmGen_raw.csv')
ImmGen_raw_df.set_index('GeneSymbol', inplace=True)

def get_relative_parameters(cell_type, baseline_celltype, baseline_parameters, parameter_name_to_database_label_dict):
    new_parameters = {}
    baseline_filter_col = [col for col in ImmGen_raw_df if baseline_celltype in col]
    new_celltype_filter_col = [col for col in ImmGen_raw_df if cell_type in col]
    for key, item in parameter_name_to_database_label_dict.items():
        numerator = np.mean(ImmGen_raw_df[new_celltype_filter_col].loc[item].values)
        denominator = np.mean(ImmGen_raw_df[baseline_filter_col].loc[item].values)
        new_value = (numerator / denominator) * baseline_parameters[key]
        new_parameters[key] = new_value
    return new_parameters


# --------------------
# Set up plotting
# --------------------
alpha_palette = sns.color_palette("Reds", 6)
beta_palette = sns.color_palette("Greens", 6)
data_palette = sns.color_palette("muted", 6)
marker_shape = ["o", "v", "s", "P", "d", "1", "x", "*"]
dataset_names = ["20190108", "20190119", "20190121", "20190214"]

# --------------------
# Get Data
# --------------------
newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")

alignment = DataAlignment()
alignment.add_data([newdata_4, newdata_3, newdata_2, newdata_1])
alignment.align()
alignment.get_scaled_data()
mean_data = alignment.summarize_data()

# --------------------
# Set up Model
# --------------------
# Parameters found by stepwise fitting GAB mean data
# Note: can remove multiplicative factors on all K1, K2, K4 and still get very good fit to data (worst is 5 min beta)
initial_parameters = {'k_a1': 4.98E-14 * 2, 'k_a2': 8.30e-13 * 2, 'k_d4': 0.006 * 3.8,
                      'kpu': 0.00095,
                      'ka2': 4.98e-13 * 2.45, 'kd4': 0.3 * 2.867,
                      'kint_a': 0.00052, 'kint_b': 0.00052,
                      'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05,
                      'kSOCSon': 6e-07, 'krec_a1': 0.001,  'krec_a2': 0.1,}
scale_factor = 1.227

IFN_Model = IfnModel('Mixed_IFN_ppCompatible')
IFN_Model.set_parameters(initial_parameters)
IFN_Model.set_parameters({'R1': 12000.0, 'R2': 1511.1})

# ---------------------------------
# Make theory dose response curves
# ---------------------------------
# Make predictions
times = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
alpha_doses_20190108 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
beta_doses_20190108 = [0, 0.2, 6, 20, 60, 200, 600, 2000]

dradf = IFN_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(1, 5.2)),
                               parameters={'Ib': 0}, scale_factor=scale_factor,
                               return_type='dataframe', dataframe_labels='Alpha')
drbdf = IFN_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-1, 4)),
                               parameters={'Ia': 0}, scale_factor=scale_factor,
                               return_type='dataframe', dataframe_labels='Beta')

dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

# ------------------------
# Plot
# ------------------------
alpha_mask = [2.5, 5.0, 7.5, 10.0]
beta_mask = [2.5, 5.0, 7.5, 10.0]
plot_original_fit = False
if plot_original_fit == True:
    DRplot = DoseresponsePlot((1,2))
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            DRplot.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label=str(t) + ' min',
                                   linewidth=2)
            DRplot.add_trajectory(mean_data, t, 'errorbar', 'o', (0, 0), 'Alpha', color=alpha_palette[idx])
        if t not in beta_mask:
            DRplot.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label=str(t) + ' min',
                                   linewidth=2)
            DRplot.add_trajectory(mean_data, t, 'errorbar', 'o', (0, 1), 'Beta', color=beta_palette[idx])
    DRplot.show_figure()

# -----------------------------
# Now generate new predictions
# -----------------------------
original_parameters = IFN_Model.parameters

ImmGen_label_maps = {'R1': 'Ifngr1',
                     'R2': 'Ifngr2',
                     'S': 'Stat1',
                     'Initial_SOCS': 'Socs1'}

# Macrophages
Macrophage_parameters = get_relative_parameters('MF.RP.Sp', 'B.Fo.Sp', original_parameters, ImmGen_label_maps)

Macrophage_model = IfnModel('Mixed_IFN_ppCompatible')
Macrophage_model.set_parameters(initial_parameters)
Macrophage_model.set_parameters(Macrophage_parameters)

MacIfnDataA = IfnData('custom',
                      df=Macrophage_model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(1, 5.2)),
                                                       parameters={'Ib': 0}, scale_factor=scale_factor,
                                                       return_type='dataframe', dataframe_labels='Alpha'),
                      conditions={'Alpha': {'Ib': 0}})
MacIfnDataB = IfnData('custom',
                      df=Macrophage_model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-1, 4)),
                                                       parameters={'Ia': 0}, scale_factor=scale_factor,
                                                       return_type='dataframe', dataframe_labels='Beta'),
                      conditions={'Beta': {'Ia': 0}})

# T cells
Tcell_parameters = get_relative_parameters('T.4Nve.Sp', 'B.Fo.Sp', original_parameters, ImmGen_label_maps)

Tcell_model = IfnModel('Mixed_IFN_ppCompatible')
Tcell_model.set_parameters(initial_parameters)
Tcell_model.set_parameters(Tcell_parameters)

TcellIfnDataA = IfnData('custom',
                        df=Tcell_model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(1, 5.2)),
                                                    parameters={'Ib': 0}, scale_factor=scale_factor,
                                                    return_type='dataframe', dataframe_labels='Alpha'),
                        conditions={'Alpha': {'Ib': 0}})
TcellIfnDataB = IfnData('custom',
                        df=Tcell_model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-1, 4)),
                                                    parameters={'Ia': 0}, scale_factor=scale_factor,
                                                    return_type='dataframe', dataframe_labels='Beta'),
                        conditions={'Beta': {'Ia': 0}})

# DC cells
DC_parameters = get_relative_parameters('DC.8-4-11b+.Sp', 'B.Fo.Sp', original_parameters, ImmGen_label_maps)

DC_model = IfnModel('Mixed_IFN_ppCompatible')
DC_model.set_parameters(initial_parameters)
DC_model.set_parameters(DC_parameters)

DCIfnDataA = IfnData('custom',
                        df=DC_model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(1, 5.2)),
                                                    parameters={'Ib': 0}, scale_factor=scale_factor,
                                                    return_type='dataframe', dataframe_labels='Alpha'),
                        conditions={'Alpha': {'Ib': 0}})
DCIfnDataB = IfnData('custom',
                        df=DC_model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-1, 4)),
                                                    parameters={'Ia': 0}, scale_factor=scale_factor,
                                                    return_type='dataframe', dataframe_labels='Beta'),
                        conditions={'Beta': {'Ia': 0}})

# Tgd cells
Tgd_parameters = get_relative_parameters('Tgd.Sp', 'B.Fo.Sp', original_parameters, ImmGen_label_maps)

Tgd_model = IfnModel('Mixed_IFN_ppCompatible')
Tgd_model.set_parameters(initial_parameters)
Tgd_model.set_parameters(Tgd_parameters)

TgdIfnDataA = IfnData('custom',
                        df=Tgd_model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(1, 5.2)),
                                                    parameters={'Ib': 0}, scale_factor=scale_factor,
                                                    return_type='dataframe', dataframe_labels='Alpha'),
                        conditions={'Alpha': {'Ib': 0}})
TgdIfnDataB = IfnData('custom',
                        df=Tgd_model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-1, 4)),
                                                    parameters={'Ia': 0}, scale_factor=scale_factor,
                                                    return_type='dataframe', dataframe_labels='Beta'),
                        conditions={'Beta': {'Ia': 0}})


# --------------------
# Plot
# --------------------
tmask = [2.5, 5.0, 7.5, 10.0]
stem_palette = sns.color_palette("Greys", 5)
myeloid_palette = sns.color_palette("Greens", 5)
lymphoid_palette = sns.color_palette("cubehelix", 8)[::-1]

hematopoetic_colormap = {"SC": stem_palette[2], "MLP": stem_palette[4],
                         "Mo": sns.color_palette("husl", 8)[1], "GN": myeloid_palette[1],
                         "MF": myeloid_palette[2], "DC": myeloid_palette[4], "pDC": 'g--',
                         "proB": lymphoid_palette[0], "preB": lymphoid_palette[1], "preT": lymphoid_palette[2],
                         "B": lymphoid_palette[6],
                         "T": lymphoid_palette[3], "Tgd": "r--",
                         "NK": lymphoid_palette[4], "NKT": lymphoid_palette[5],
                         "BEC": "tomato", "FRC": "crimson", "Fi": "lightcoral"}

cellTypePlot = DoseresponsePlot((2,2))

for idx, t in enumerate([t for t in times if t not in tmask]):
    cellTypePlot.axes[idx, 0].set_title(r'IFN$\alpha$ {} min'.format(t))
    cellTypePlot.axes[idx, 1].set_title(r'IFN$\beta$ {} min'.format(t))

    cellTypePlot.add_trajectory(mean_data, t, 'errorbar', 'o', (idx, 0), 'Alpha', color=hematopoetic_colormap['B'])
    cellTypePlot.add_trajectory(dra60, t, 'plot', hematopoetic_colormap['B'], (idx, 0), 'Alpha', label='B', linewidth=2)
    cellTypePlot.add_trajectory(MacIfnDataA, t, 'plot', hematopoetic_colormap['MF'], (idx, 0), 'Alpha', label='MF.RP.Sp', linewidth=2)
    cellTypePlot.add_trajectory(TcellIfnDataA, t, 'plot', hematopoetic_colormap['T'], (idx, 0), 'Alpha', label='T.4Nve.Sp', linewidth=2)
    cellTypePlot.add_trajectory(DCIfnDataA, t, 'plot', hematopoetic_colormap['DC'], (idx, 0), 'Alpha', label='DC.8-4-11b+.Sp', linewidth=2)
    cellTypePlot.add_trajectory(TgdIfnDataA, t, 'plot', hematopoetic_colormap['Tgd'], (idx, 0), 'Alpha', label='Tgd.Sp', linewidth=2)

    cellTypePlot.add_trajectory(mean_data, t, 'errorbar', 'o', (idx, 1), 'Beta', color=hematopoetic_colormap['B'])
    cellTypePlot.add_trajectory(drb60, t, 'plot', hematopoetic_colormap['B'], (idx, 1), 'Beta', label='B', linewidth=2)
    cellTypePlot.add_trajectory(MacIfnDataB, t, 'plot', hematopoetic_colormap['MF'], (idx, 1), 'Beta', label='MF.RP.Sp', linewidth=2)
    cellTypePlot.add_trajectory(TcellIfnDataB, t, 'plot', hematopoetic_colormap['T'], (idx, 1), 'Beta', label='T.4Nve.Sp', linewidth=2)
    cellTypePlot.add_trajectory(DCIfnDataB, t, 'plot', hematopoetic_colormap['DC'], (idx, 1), 'Beta', label='DC.8-4-11b+.Sp', linewidth=2)
    cellTypePlot.add_trajectory(TgdIfnDataB, t, 'plot', hematopoetic_colormap['Tgd'], (idx, 1), 'Beta', label='Tgd.Sp', linewidth=2)

cellTypePlot.show_figure()
