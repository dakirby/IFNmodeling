"""# -*- coding: utf-8 -*-
#
# Created on Sun Nov 25 12:30:11 2018
#
# @author: Duncan
#
# Build pandas DataFrame objects from csv files
"""

import pandas as pd
import numpy as np
import pickle


def build_database(data_wd: str) -> None:
    # Experimental_Data
    # ------------------- 
    dataset_1 = pd.read_csv(data_wd + "Experimental_Data.csv")
    aData = dataset_1.loc[(dataset_1.loc[:, 'Interferon'] == "Alpha"), ['0', '5', '15', '30', '60']].values
    bData = dataset_1.loc[(dataset_1.loc[:, 'Interferon'] == "Alpha_std"), ['0', '5', '15', '30', '60']].values
    aZipped = [[tuple(el) for el in r] for r in np.dstack((aData, bData))]
    aZipped = [['Alpha', 10] + aZipped[0],
               ['Alpha', 90] + aZipped[1],
               ['Alpha', 600] + aZipped[2]]
    aData = dataset_1.loc[(dataset_1.loc[:, 'Interferon'] == "Beta"), ['0', '5', '15', '30', '60']].values
    bData = dataset_1.loc[(dataset_1.loc[:, 'Interferon'] == "Beta_std"), ['0', '5', '15', '30', '60']].values
    bZipped = [[tuple(el) for el in r] for r in np.dstack((aData, bData))]
    bZipped = [['Beta', 10] + bZipped[0],
               ['Beta', 90] + bZipped[1],
               ['Beta', 600] + bZipped[2]]

    df = pd.DataFrame.from_records(aZipped + bZipped,
                                   columns=['Dose_Species', 'Dose (pM)'] + list(dataset_1.columns[2:]))
    df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)

    pickle.dump(df, open(data_wd + 'Experimental_Data.p', 'wb'))

    # MacParland_Extended
    # -------------------
    dataset_2 = pd.read_csv(data_wd + "MacParland_Extended.csv")
    aData = dataset_2.loc[(dataset_2.loc[:, 'Interferon'] == "Alpha"), ['0', '5', '15', '30', '60']].values
    bData = dataset_2.loc[(dataset_2.loc[:, 'Interferon'] == "Alpha_std"), ['0', '5', '15', '30', '60']].values
    aZipped = [[tuple(el) for el in r] for r in np.dstack((aData, bData))]
    aZipped = [['Alpha', 10] + aZipped[0],
               ['Alpha', 90] + aZipped[1],
               ['Alpha', 600] + aZipped[2],
               ['Alpha', 4000] + aZipped[3],
               ['Alpha', 8000] + aZipped[4]]
    aData = dataset_2.loc[(dataset_2.loc[:, 'Interferon'] == "Beta"), ['0', '5', '15', '30', '60']].values
    bData = dataset_2.loc[(dataset_2.loc[:, 'Interferon'] == "Beta_std"), ['0', '5', '15', '30', '60']].values
    bZipped = [[tuple(el) for el in r] for r in np.dstack((aData, bData))]
    bZipped = [['Beta', 10] + bZipped[0],
               ['Beta', 90] + bZipped[1],
               ['Beta', 600] + bZipped[2],
               ['Beta', 2000] + bZipped[3],
               ['Beta', 11000] + bZipped[4]]
    df = pd.DataFrame.from_records(aZipped + bZipped,
                                   columns=['Dose_Species', 'Dose (pM)'] + list(dataset_2.columns[2:]))
    df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    pickle.dump(df, open(data_wd + 'MacParland_Extended.p', 'wb'))

    # 20181031_pSTAT1_Table
    # -------------------
    dataset_3 = pd.read_csv(data_wd + "20181031_pSTAT1_Table.csv")
    NKdata = dataset_3.loc[:]["NK cells"].values.reshape((8, 8))
    Bdata = dataset_3.loc[:]["B cells"].values.reshape((8, 8))
    Tdata = dataset_3.loc[:]["T cells"].values.reshape((8, 8))
    alpha_doses_20181031 = [0, 0.06, 0.32, 1.6, 8, 40, 200, 1000][::-1]
    beta_doses_20181031 = [0, 0.06, 0.32, 1.6, 8, 40, 200, 1000]

    # NK Cells
    NKZipped = [[alpha_doses_20181031[j], beta_doses_20181031[i], (NKdata[j][i],)] for i in range(8) for j in range(8)]
    NKDataFrame = pd.DataFrame.from_records(NKZipped, columns=['Alpha Dose (pM)', 'Beta Dose (pM)', 20])
    NKDataFrame.set_index(['Alpha Dose (pM)', 'Beta Dose (pM)'], inplace=True)
    pickle.dump(NKDataFrame, open(data_wd + '20181031_pSTAT1_Table_NKcells.p', 'wb'))

    # T Cells
    TZipped = [[alpha_doses_20181031[j], beta_doses_20181031[i], (Tdata[j][i],)] for i in range(8) for j in range(8)]
    TDataFrame = pd.DataFrame.from_records(TZipped, columns=['Alpha Dose (pM)', 'Beta Dose (pM)', 20])
    TDataFrame.set_index(['Alpha Dose (pM)', 'Beta Dose (pM)'], inplace=True)
    pickle.dump(TDataFrame, open(data_wd + '20181031_pSTAT1_Table_Tcells.p', 'wb'))

    # B Cells
    BZipped = [[alpha_doses_20181031[j], beta_doses_20181031[i], (Bdata[j][i],)] for i in range(8) for j in range(8)]
    BDataFrame = pd.DataFrame.from_records(BZipped, columns=['Alpha Dose (pM)', 'Beta Dose (pM)', 20])
    BDataFrame.set_index(['Alpha Dose (pM)', 'Beta Dose (pM)'], inplace=True)
    pickle.dump(BDataFrame, open(data_wd + '20181031_pSTAT1_Table_Bcells.p', 'wb'))

    # 20181113_B6_IFNs_Dose_Response
    # -------------------
    times_20181113_B6 = [2.5, 5, 7.5, 10, 20, 60]
    alpha_doses_20181113_B6 = [0, 5, 50, 250, 500, 5000, 25000, 50000]
    beta_doses_20181113_B6 = [0, 0.1, 1, 5, 10, 100, 200, 1000]
    dataset_4 = pd.read_csv(data_wd + "20181113_B6_IFNs_Dose_Response.csv")
    Bdata = dataset_4.loc[:]["Lymphocytes/B cells | Geometric Mean (Comp-APC-A)"].values.reshape((8, 12)).tolist()
    Tdata = dataset_4.loc[:]["Lymphocytes/T cells | Geometric Mean (Comp-APC-A)"].values.reshape((8, 12)).tolist()
    NIdata = dataset_4.loc[:]["Lymphocytes/NonT, nonB | Geometric Mean (Comp-APC-A)"].values.reshape((8, 12)).tolist()

    # Normalize by Jaki and add non-existent uncertainties
    # Bcells
    Jaki_a = Bdata[0][0]
    Jaki_b = Bdata[0][1]
    for r in range(len(Bdata)):
        for c in range(len(Bdata[r])):
            if c % 2 == 0:
                Bdata[r][c] = (Bdata[r][c] - Jaki_a, np.nan)
            else:
                Bdata[r][c] = (Bdata[r][c] - Jaki_b, np.nan)
    # Tcells
    Jaki_a = Tdata[0][0]
    Jaki_b = Tdata[0][1]
    for r in range(len(Tdata)):
        for c in range(len(Tdata[r])):
            if c % 2 == 0:
                Tdata[r][c] = (Tdata[r][c] - Jaki_a, np.nan)
            else:
                Tdata[r][c] = (Tdata[r][c] - Jaki_b, np.nan)
    # Non-immune cells
    Jaki_a = NIdata[0][0]
    Jaki_b = NIdata[0][1]
    for r in range(len(NIdata)):
        for c in range(len(NIdata[r])):
            if c % 2 == 0:
                NIdata[r][c] = (NIdata[r][c] - Jaki_a, np.nan)
            else:
                NIdata[r][c] = (NIdata[r][c] - Jaki_b, np.nan)

    # Non-lymphocyte cells
    NIalpha = [['Alpha', alpha_doses_20181113_B6[row]] + [NIdata[row][2 * i] for i in range(6)] for row in range(8)]
    NIbeta = [['Beta', beta_doses_20181113_B6[row]] + [NIdata[row][2 * i + 1] for i in range(6)] for row in range(8)]
    df = pd.DataFrame.from_records(NIalpha + NIbeta, columns=['Dose_Species', 'Dose (pM)'] + times_20181113_B6)
    df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    pickle.dump(df, open(data_wd + '20181113_B6_IFNs_Dose_Response_NonLymphocytes.p', 'wb'))

    # B cells
    Balpha = [['Alpha', alpha_doses_20181113_B6[row]] + [Bdata[row][2 * i] for i in range(6)] for row in range(8)]
    Bbeta = [['Beta', beta_doses_20181113_B6[row]] + [Bdata[row][2 * i + 1] for i in range(6)] for row in range(8)]
    df = pd.DataFrame.from_records(Balpha + Bbeta, columns=['Dose_Species', 'Dose (pM)'] + times_20181113_B6)
    df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    pickle.dump(df, open(data_wd + '20181113_B6_IFNs_Dose_Response_Bcells.p', 'wb'))

    # T cells
    Talpha = [['Alpha', alpha_doses_20181113_B6[row]] + [Tdata[row][2 * i] for i in range(6)] for row in range(8)]
    Tbeta = [['Beta', beta_doses_20181113_B6[row]] + [Tdata[row][2 * i + 1] for i in range(6)] for row in range(8)]
    df = pd.DataFrame.from_records(Talpha + Tbeta, columns=['Dose_Species', 'Dose (pM)'] + times_20181113_B6)
    df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    pickle.dump(df, open(data_wd + '20181113_B6_IFNs_Dose_Response_Tcells.p', 'wb'))

    print("Initialized DataFrame objects")

if __name__ == '__main__':
    build_database('C:\\Users\\Duncan\\Documents\\University\\Grad Studies Year 2\\IFNmodeling\\ifndatabase\\')