import numpy as np
import pandas as pd

def figure_5_setup_barchart():
    Thomas_data = pd.read_csv('2011_Thomas_FigureS5.csv')
    names = ['L30A', 'R149A', 'A145G', 'L26A', 'YNSL153A', 'YNSM148A', 'YNS']  # IFNs of interest
    alt_names = ['IFNa2(L30A)', 'IFNa2(R149A)', 'IFNa2(A145G)', 'IFNa2(L26A)', 'IFNa2(YNS,L153A)', 'IFNa2(YNS,M148A)', 'IFNa2(YNS)']  # names in FiguresS5.csv
    columns = ['Name', 'K1 x K2', 'AV Experiment', 'AV Model', 'AP Experiment', 'AP Model']
    df = Thomas_data[Thomas_data['IFN'].isin(alt_names)]
    K1K2 = df['R1xR2 (relative to WT)'].to_numpy()
    AP_rel_IC50 = df['AP EC50 (nM)'].to_numpy() / Thomas_data[Thomas_data['IFN']=='IFNa2 WT']['AP EC50 (nM)'].values[0]
    AV_rel_IC50 = df['AV EC50 (pM)'].to_numpy() / Thomas_data[Thomas_data['IFN']=='IFNa2 WT']['AV EC50 (pM)'].values[0]
    AP_AV_Bar_Chart = pd.DataFrame(dict(zip(columns, [names, K1K2, AV_rel_IC50, np.zeros_like(K1K2), AP_rel_IC50, np.zeros_like(K1K2)])))
    AP_AV_Bar_Chart.to_csv('AP_AV_Bar_Chart.csv', index=False)
