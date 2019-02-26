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
import os


def build_database(data_wd: str) -> None:
    # Experimental_Data
    # ------------------- 
    dataset_1 = pd.read_csv(os.path.join(data_wd, "Experimental_Data.csv"))
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

    pickle.dump(df, open(os.path.join(data_wd, 'Experimental_Data.p'), 'wb'))

    # MacParland_Extended
    # -------------------
    dataset_2 = pd.read_csv(os.path.join(data_wd, "MacParland_Extended.csv"))
    aData = dataset_2.loc[(dataset_2.loc[:, 'Interferon'] == "Alpha"), [str(el) for el in [0, 5, 15, 30, 60]]].values
    bData = dataset_2.loc[(dataset_2.loc[:, 'Interferon'] == "Alpha_std"), [str(el) for el in [0, 5, 15, 30, 60]]].values
    aZipped = [[tuple(el) for el in r] for r in np.dstack((aData, bData))]
    aZipped = [['Alpha', 10] + aZipped[0],
               ['Alpha', 90] + aZipped[1],
               ['Alpha', 600] + aZipped[2],
               ['Alpha', 4000] + aZipped[3],
               ['Alpha', 8000] + aZipped[4]]
    aData = dataset_2.loc[(dataset_2.loc[:, 'Interferon'] == "Beta"), [str(el) for el in [0, 5, 15, 30, 60]]].values
    bData = dataset_2.loc[(dataset_2.loc[:, 'Interferon'] == "Beta_std"), [str(el) for el in [0, 5, 15, 30, 60]]].values
    bZipped = [[tuple(el) for el in r] for r in np.dstack((aData, bData))]
    bZipped = [['Beta', 10] + bZipped[0],
               ['Beta', 90] + bZipped[1],
               ['Beta', 600] + bZipped[2],
               ['Beta', 2000] + bZipped[3],
               ['Beta', 11000] + bZipped[4]]
    df = pd.DataFrame.from_records(aZipped + bZipped,
                                   columns=['Dose_Species', 'Dose (pM)', 0, 5, 15, 30, 60])
    df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    pickle.dump(df, open(os.path.join(data_wd, 'MacParland_Extended.p'), 'wb'))

    # 20181031_pSTAT1_Table
    # -------------------
    dataset_3 = pd.read_csv(os.path.join(data_wd, "20181031_pSTAT1_Table.csv"))
    NKdata = dataset_3.loc[:]["NK cells"].values.reshape((8, 8))
    Bdata = dataset_3.loc[:]["B cells"].values.reshape((8, 8))
    Tdata = dataset_3.loc[:]["T cells"].values.reshape((8, 8))
    alpha_doses_20181031 = [0, 0.06, 0.32, 1.6, 8, 40, 200, 1000][::-1]
    beta_doses_20181031 = [0, 0.06, 0.32, 1.6, 8, 40, 200, 1000]

    # NK Cells
    NKZipped = [[alpha_doses_20181031[j], beta_doses_20181031[i], (NKdata[j][i],)] for i in range(8) for j in range(8)]
    NKDataFrame = pd.DataFrame.from_records(NKZipped, columns=['Alpha Dose (pM)', 'Beta Dose (pM)', 20])
    NKDataFrame.set_index(['Alpha Dose (pM)', 'Beta Dose (pM)'], inplace=True)
    pickle.dump(NKDataFrame, open(os.path.join(data_wd, '20181031_pSTAT1_Table_NKcells.p'), 'wb'))

    # T Cells
    TZipped = [[alpha_doses_20181031[j], beta_doses_20181031[i], (Tdata[j][i],)] for i in range(8) for j in range(8)]
    TDataFrame = pd.DataFrame.from_records(TZipped, columns=['Alpha Dose (pM)', 'Beta Dose (pM)', 20])
    TDataFrame.set_index(['Alpha Dose (pM)', 'Beta Dose (pM)'], inplace=True)
    pickle.dump(TDataFrame, open(os.path.join(data_wd, '20181031_pSTAT1_Table_Tcells.p'), 'wb'))

    # B Cells
    BZipped = [[alpha_doses_20181031[j], beta_doses_20181031[i], (Bdata[j][i],)] for i in range(8) for j in range(8)]
    BDataFrame = pd.DataFrame.from_records(BZipped, columns=['Alpha Dose (pM)', 'Beta Dose (pM)', 20])
    BDataFrame.set_index(['Alpha Dose (pM)', 'Beta Dose (pM)'], inplace=True)
    pickle.dump(BDataFrame, open(os.path.join(data_wd, '20181031_pSTAT1_Table_Bcells.p'), 'wb'))

    # 20181113_B6_IFNs_Dose_Response
    # -------------------
    times_20181113_B6 = [2.5, 5, 7.5, 10, 20, 60]
    alpha_doses_20181113_B6 = [0, 5, 50, 250, 500, 5000, 25000, 50000]
    beta_doses_20181113_B6 = [0, 0.1, 1, 5, 10, 100, 200, 1000]
    dataset_4 = pd.read_csv(os.path.join(data_wd, "20181113_B6_IFNs_Dose_Response.csv"))
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
    pickle.dump(df, open(os.path.join(data_wd, '20181113_B6_IFNs_Dose_Response_NonLymphocytes.p'), 'wb'))

    # B cells
    Balpha = [['Alpha', alpha_doses_20181113_B6[row]] + [Bdata[row][2 * i] for i in range(6)] for row in range(8)]
    Bbeta = [['Beta', beta_doses_20181113_B6[row]] + [Bdata[row][2 * i + 1] for i in range(6)] for row in range(8)]
    df = pd.DataFrame.from_records(Balpha + Bbeta, columns=['Dose_Species', 'Dose (pM)'] + times_20181113_B6)
    df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    pickle.dump(df, open(os.path.join(data_wd, '20181113_B6_IFNs_Dose_Response_Bcells.p'), 'wb'))

    # T cells
    Talpha = [['Alpha', alpha_doses_20181113_B6[row]] + [Tdata[row][2 * i] for i in range(6)] for row in range(8)]
    Tbeta = [['Beta', beta_doses_20181113_B6[row]] + [Tdata[row][2 * i + 1] for i in range(6)] for row in range(8)]
    df = pd.DataFrame.from_records(Talpha + Tbeta, columns=['Dose_Species', 'Dose (pM)'] + times_20181113_B6)
    df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    pickle.dump(df, open(os.path.join(data_wd, '20181113_B6_IFNs_Dose_Response_Tcells.p'), 'wb'))

    # 20190108_pSTAT1_IFN
    # -------------------
    times_20190108 = [2.5, 5, 7.5, 10, 20, 60][::-1]
    alpha_doses_20190108 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses_20190108 = [0, 0.2, 6, 20, 60, 200, 600, 2000]
    dataset_4 = pd.read_csv(os.path.join(data_wd, "20190108_pSTAT1_IFN.csv"))
    Bdata = dataset_4.loc[:]["Lymphocytes/B cells | Geometric Mean (Comp-FITC-A)"].values.reshape((8, 12)).tolist()

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

    # B cells
    Balpha = [['Alpha', alpha_doses_20190108[row]] + [Bdata[row][2 * i] for i in range(6)] for row in range(8)]
    Bbeta = [['Beta', beta_doses_20190108[row]] + [Bdata[row][2 * i + 1] for i in range(6)] for row in range(8)]
    # Re-order the columns so that times go from 2.5 to 60
    Bbeta = [[row[0], row[1], row[7], row[6], row[5], row[4], row[3], row[2]] for row in Bbeta]
    Balpha = [[row[0], row[1], row[7], row[6], row[5], row[4], row[3], row[2]] for row in Balpha]
    df = pd.DataFrame.from_records(Balpha + Bbeta, columns=['Dose_Species', 'Dose (pM)'] + times_20190108[::-1])
    df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    pickle.dump(df, open(os.path.join(data_wd, '20190108_pSTAT1_IFN.p'), 'wb'))

    # 20190121_pSTAT1_IFN
    # -------------------
    df = pd.read_csv('20190121_pSTAT1_IFN.csv').drop('Unnamed: 0', axis=1)
    # Just FYI, these are the values of df levels:
    # times_20190121 = [2.5, 5, 7.5, 10, 20, 60]
    alpha_doses_20190121 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses_20190121 = [0, 0.2, 6, 20, 60, 200, 600, 2000]
    df = df.sort_values(['Cytokine', 'Well'])
    df.replace('IFN-alpha', 'Alpha', inplace=True)
    df.replace('IFN-beta', 'Beta', inplace=True)
    df.loc[:, 'Concentration (Mol)'] *= 1e12
    newcolumns = [df.columns.values[0], 'Dose (pM)', 'Dose_Species', *df.columns.values[3:]]
    df.columns = newcolumns
    # B cells
    Bcell_df = df.drop(['Well', 'pSTAT1 in CD8+ T cells', 'pSTAT1 in CD4+ T cells'], axis=1)

        # Convert values to (value, error)
    temp = Bcell_df.values
    temp = [[float("{0:.1f}".format(row[0])), row[1], row[2], (float("{0:.2f}".format(row[3])), np.nan)] for row in temp]
    Bcell_df = pd.DataFrame.from_records(temp, columns=['Dose (pM)', 'Dose_Species', 'Time (min)', 'pSTAT1 in B cells'])

        # Put in standard form
    Bcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    Bcell_df = pd.pivot_table(Bcell_df, values='pSTAT1 in B cells', index=['Dose_Species', 'Dose (pM)'], columns=['Time (min)'], aggfunc=np.sum)
    Bcell_df.columns.name = None
        # Save
    pickle.dump(Bcell_df, open(os.path.join(data_wd, '20190121_pSTAT1_IFN_Bcell.p'), 'wb'))

    # CD8+ T cells
    CD8Tcell_df = df.drop(['Well', 'pSTAT1 in B cells', 'pSTAT1 in CD4+ T cells'], axis=1)
        # Convert values to (value, error)
    temp = CD8Tcell_df.values
    temp = [[int("{0:.0f}".format(row[0])), row[1], row[2], (float("{0:.2f}".format(row[3])), np.nan)] for row in temp]
    CD8Tcell_df = pd.DataFrame.from_records(temp, columns=['Dose (pM)', 'Dose_Species', 'Time (min)', 'pSTAT1 in CD8+ T cells'])
    # Put in standard form
    CD8Tcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    CD8Tcell_df = pd.pivot_table(CD8Tcell_df, values='pSTAT1 in CD8+ T cells', index=['Dose_Species', 'Dose (pM)'],
                              columns=['Time (min)'], aggfunc=np.sum)
    CD8Tcell_df.columns.name = None
    # Save
    pickle.dump(CD8Tcell_df, open(os.path.join(data_wd, '20190121_pSTAT1_IFN_CD8Tcell.p'), 'wb'))

    # CD4+ T cells
    CD4Tcell_df = df.drop(['Well', 'pSTAT1 in B cells', 'pSTAT1 in CD8+ T cells'], axis=1)
        # Convert values to (value, error)
    temp = CD4Tcell_df.values
    temp = [[int("{0:.0f}".format(row[0])), row[1], row[2], (float("{0:.2f}".format(row[3])), np.nan)] for row in temp]
    CD4Tcell_df = pd.DataFrame.from_records(temp, columns=['Dose (pM)', 'Dose_Species', 'Time (min)', 'pSTAT1 in CD4+ T cells'])
    # Put in standard form
    CD4Tcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    CD4Tcell_df = pd.pivot_table(CD4Tcell_df, values='pSTAT1 in CD4+ T cells', index=['Dose_Species', 'Dose (pM)'],
                                 columns=['Time (min)'], aggfunc=np.sum)
    CD4Tcell_df.columns.name = None
    # Save
    pickle.dump(CD4Tcell_df, open(os.path.join(data_wd, '20190121_pSTAT1_IFN_CD4Tcell.p'), 'wb'))

    # 20190119_pSTAT1_IFN
    # -------------------
    df = pd.read_csv('20190119_pSTAT1_IFN.csv').drop('Unnamed: 0', axis=1)
    # Just FYI, these are the values of df levels:
    # times_20190119 = [2.5, 5, 7.5, 10, 20, 60]
    alpha_doses_20190119 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses_20190119 = [0, 0.2, 6, 20, 60, 200, 600, 2000]
    df = df.sort_values(['Cytokine', 'Well'])
    df.replace('IFN-alpha', 'Alpha', inplace=True)
    df.replace('IFN-beta', 'Beta', inplace=True)
    df.loc[:, 'Concentration (Mol)'] *= 1e12
    newcolumns = [df.columns.values[0], 'Dose (pM)', 'Dose_Species', *df.columns.values[3:]]
    df.columns = newcolumns
    # B cells
    Bcell_df = df.drop(['Well', 'pSTAT1 in CD8+ T cells', 'pSTAT1 in CD4+ T cells'], axis=1)

    # Convert values to (value-background, error)
    temp = Bcell_df.values
    temp = [[float("{0:.1f}".format(row[0])), row[1], row[2], (float("{0:.2f}".format(row[3])), np.nan)] for row in
            temp]
    Bcell_df = pd.DataFrame.from_records(temp, columns=['Dose (pM)', 'Dose_Species', 'Time (min)', 'pSTAT1 in B cells'])

    # Put in standard form
    Bcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    Bcell_df = pd.pivot_table(Bcell_df, values='pSTAT1 in B cells', index=['Dose_Species', 'Dose (pM)'],
                              columns=['Time (min)'], aggfunc=np.sum)
    Bcell_df.columns.name = None

    # Save
    pickle.dump(Bcell_df, open(os.path.join(data_wd, '20190119_pSTAT1_IFN_Bcell.p'), 'wb'))

    # CD8+ T cells
    CD8Tcell_df = df.drop(['Well', 'pSTAT1 in B cells', 'pSTAT1 in CD4+ T cells'], axis=1)
    # Convert values to (value, error)
    temp = CD8Tcell_df.values
    temp = [[int("{0:.0f}".format(row[0])), row[1], row[2], (float("{0:.2f}".format(row[3])), np.nan)] for row in temp]
    CD8Tcell_df = pd.DataFrame.from_records(temp, columns=['Dose (pM)', 'Dose_Species', 'Time (min)',
                                                           'pSTAT1 in CD8+ T cells'])
    # Put in standard form
    CD8Tcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    CD8Tcell_df = pd.pivot_table(CD8Tcell_df, values='pSTAT1 in CD8+ T cells', index=['Dose_Species', 'Dose (pM)'],
                                 columns=['Time (min)'], aggfunc=np.sum)
    CD8Tcell_df.columns.name = None
    # Save
    pickle.dump(CD8Tcell_df, open(os.path.join(data_wd, '20190119_pSTAT1_IFN_CD8Tcell.p'), 'wb'))

    # CD4+ T cells
    CD4Tcell_df = df.drop(['Well', 'pSTAT1 in B cells', 'pSTAT1 in CD8+ T cells'], axis=1)
    # Convert values to (value, error)
    temp = CD4Tcell_df.values
    temp = [[int("{0:.0f}".format(row[0])), row[1], row[2], (float("{0:.2f}".format(row[3])), np.nan)] for row in temp]
    CD4Tcell_df = pd.DataFrame.from_records(temp, columns=['Dose (pM)', 'Dose_Species', 'Time (min)',
                                                           'pSTAT1 in CD4+ T cells'])
    # Put in standard form
    CD4Tcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    CD4Tcell_df = pd.pivot_table(CD4Tcell_df, values='pSTAT1 in CD4+ T cells', index=['Dose_Species', 'Dose (pM)'],
                                 columns=['Time (min)'], aggfunc=np.sum)
    CD4Tcell_df.columns.name = None
    # Save
    pickle.dump(CD4Tcell_df, open(os.path.join(data_wd, '20190119_pSTAT1_IFN_CD4Tcell.p'), 'wb'))

    # 20190214_pSTAT1_IFN
    # -------------------
    df = pd.read_csv('20190214_pSTAT1_IFN.csv').drop('Unnamed: 0', axis=1)
    # Just FYI, these are the values of df levels:
    # times_20190214 = [2.5, 5, 7.5, 10, 20, 60]
    alpha_doses_20190214 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses_20190214 = [0, 0.2, 6, 20, 60, 200, 600, 2000]
    df = df.sort_values(['Cytokine', 'Well'])
    df.replace('IFN-alpha', 'Alpha', inplace=True)
    df.replace('IFN-beta', 'Beta', inplace=True)
    df.loc[:, 'Concentration (Mol)'] *= 1e12
    newcolumns = [df.columns.values[0], 'Dose (pM)', 'Dose_Species', *df.columns.values[3:]]
    df.columns = newcolumns
    # B cells
    Bcell_df = df.drop(['Well', 'pSTAT1 in CD8+ T cells', 'pSTAT1 in CD4+ T cells'], axis=1)

    # Convert values to (value-background, error)
    temp = Bcell_df.values
    temp = [[float("{0:.1f}".format(row[0])), row[1], row[2], (float("{0:.2f}".format(row[3])), np.nan)] for row in
            temp]
    Bcell_df = pd.DataFrame.from_records(temp, columns=['Dose (pM)', 'Dose_Species', 'Time (min)', 'pSTAT1 in B cells'])

    # Put in standard form
    Bcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    Bcell_df = pd.pivot_table(Bcell_df, values='pSTAT1 in B cells', index=['Dose_Species', 'Dose (pM)'],
                              columns=['Time (min)'], aggfunc=np.sum)
    Bcell_df.columns.name = None

    # Save
    pickle.dump(Bcell_df, open(os.path.join(data_wd, '20190214_pSTAT1_IFN_Bcell.p'), 'wb'))

    # CD8+ T cells
    CD8Tcell_df = df.drop(['Well', 'pSTAT1 in B cells', 'pSTAT1 in CD4+ T cells'], axis=1)
    # Convert values to (value, error)
    temp = CD8Tcell_df.values
    temp = [[int("{0:.0f}".format(row[0])), row[1], row[2], (float("{0:.2f}".format(row[3])), np.nan)] for row in temp]
    CD8Tcell_df = pd.DataFrame.from_records(temp, columns=['Dose (pM)', 'Dose_Species', 'Time (min)',
                                                           'pSTAT1 in CD8+ T cells'])
    # Put in standard form
    CD8Tcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    CD8Tcell_df = pd.pivot_table(CD8Tcell_df, values='pSTAT1 in CD8+ T cells', index=['Dose_Species', 'Dose (pM)'],
                                 columns=['Time (min)'], aggfunc=np.sum)
    CD8Tcell_df.columns.name = None
    # Save
    pickle.dump(CD8Tcell_df, open(os.path.join(data_wd, '20190214_pSTAT1_IFN_CD8Tcell.p'), 'wb'))

    # CD4+ T cells
    CD4Tcell_df = df.drop(['Well', 'pSTAT1 in B cells', 'pSTAT1 in CD8+ T cells'], axis=1)
    # Convert values to (value, error)
    temp = CD4Tcell_df.values
    temp = [[int("{0:.0f}".format(row[0])), row[1], row[2], (float("{0:.2f}".format(row[3])), np.nan)] for row in temp]
    CD4Tcell_df = pd.DataFrame.from_records(temp, columns=['Dose (pM)', 'Dose_Species', 'Time (min)',
                                                           'pSTAT1 in CD4+ T cells'])
    # Put in standard form
    CD4Tcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    CD4Tcell_df = pd.pivot_table(CD4Tcell_df, values='pSTAT1 in CD4+ T cells', index=['Dose_Species', 'Dose (pM)'],
                                 columns=['Time (min)'], aggfunc=np.sum)
    CD4Tcell_df.columns.name = None
    # Save
    pickle.dump(CD4Tcell_df, open(os.path.join(data_wd, '20190214_pSTAT1_IFN_CD4Tcell.p'), 'wb'))


    print("Initialized DataFrame objects")

if __name__ == '__main__':
    build_database(os.getcwd())