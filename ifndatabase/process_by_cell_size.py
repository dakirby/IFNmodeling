"""# -*- coding: utf-8 -*-
#
# Created on Thursday May 02 14:30:11 2019
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
    # ---------------------
    # 20190108
    # ---------------------
    df = pd.read_csv('20190108_raw_data/processing/pSTAT1.csv').drop('Unnamed: 0', axis=1)
    # Just FYI, these are the values of df levels:
    # times_20190214 = [2.5, 5, 7.5, 10, 20, 60]
    # alpha_doses_20190214 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    # beta_doses_20190214 = [0, 0.2, 6, 20, 60, 200, 600, 2000]
    df = df.sort_values(['Cytokine', 'Well'])
    df.replace('IFN-alpha', 'Alpha', inplace=True)
    df.replace('IFN-beta', 'Beta', inplace=True)
    df.loc[:, 'Concentration (Mol)'] *= 1e12
    newcolumns = [df.columns.values[0], 'Dose (pM)', 'Dose_Species', *df.columns.values[3:]]
    df.columns = newcolumns
    # Small B cells
    # -------------
    Bcell_df = df.drop(
        ['Well', 'pSTAT1 in CD8+ T cells', 'pSTAT1 in CD4+ T cells', 'pSTAT1 in B cells', 'pSTAT1 in Large B cells'],
        axis=1)

    # Convert values to (value-background, error)
    temp = Bcell_df.values
    temp = [[float("{0:.1f}".format(row[0])), row[1], row[2], float("{0:.2f}".format(row[3]))] for row in
            temp]
    Bcell_df = pd.DataFrame.from_records(temp,
                                         columns=['Dose (pM)', 'Dose_Species', 'Time (min)', 'pSTAT1 in Small B cells'])

    # Put in standard form
    Bcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    Bcell_df = pd.pivot_table(Bcell_df, values='pSTAT1 in Small B cells', index=['Dose_Species', 'Dose (pM)'],
                              columns=['Time (min)'], aggfunc=np.sum)
    Bcell_df.columns.name = None

    # Zero the values by the 0 dose row
    record = []
    for species in ['Alpha', 'Beta']:
        background = pd.Series(Bcell_df.loc[species].loc[0.0])
        for dose in Bcell_df.loc[species].index:
            values = Bcell_df.loc[species].loc[dose] - background
            values = values.values
            record.append([species, dose] + [(el, np.nan) for el in values])

    col_labels = ['Dose_Species', 'Dose (pM)'] + [float("{0:.1f}".format(t)) for t in
                                                  Bcell_df.loc['Alpha'].columns.values]
    Bcell_df = pd.DataFrame.from_records(record, columns=col_labels)
    # Put in standard form
    Bcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    Bcell_df.drop(0.0, level=1, axis=0, inplace=True)

    # Save
    pickle.dump(Bcell_df, open(os.path.join(data_wd, '20190108_pSTAT1_IFN_Small_Bcells.p'), 'wb'))

    # Large B cells
    # -------------
    Bcell_df = df.drop(
        ['Well', 'pSTAT1 in CD8+ T cells', 'pSTAT1 in CD4+ T cells', 'pSTAT1 in B cells', 'pSTAT1 in Small B cells'],
        axis=1)

    # Convert values to (value-background, error)
    temp = Bcell_df.values
    temp = [[float("{0:.1f}".format(row[0])), row[1], row[2], float("{0:.2f}".format(row[3]))] for row in
            temp]
    Bcell_df = pd.DataFrame.from_records(temp,
                                         columns=['Dose (pM)', 'Dose_Species', 'Time (min)', 'pSTAT1 in Large B cells'])

    # Put in standard form
    Bcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    Bcell_df = pd.pivot_table(Bcell_df, values='pSTAT1 in Large B cells', index=['Dose_Species', 'Dose (pM)'],
                              columns=['Time (min)'], aggfunc=np.sum)
    Bcell_df.columns.name = None

    # Zero the values by the 0 dose row
    record = []
    for species in ['Alpha', 'Beta']:
        background = pd.Series(Bcell_df.loc[species].loc[0.0])
        for dose in Bcell_df.loc[species].index:
            values = Bcell_df.loc[species].loc[dose] - background
            values = values.values
            record.append([species, dose] + [(el, np.nan) for el in values])

    col_labels = ['Dose_Species', 'Dose (pM)'] + [float("{0:.1f}".format(t)) for t in
                                                  Bcell_df.loc['Alpha'].columns.values]
    Bcell_df = pd.DataFrame.from_records(record, columns=col_labels)
    # Put in standard form
    Bcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    Bcell_df.drop(0.0, level=1, axis=0, inplace=True)

    # Save
    pickle.dump(Bcell_df, open(os.path.join(data_wd, '20190108_pSTAT1_IFN_Large_Bcells.p'), 'wb'))


    # ---------------------
    # 20190119
    # ---------------------
    df = pd.read_csv('20190119_raw_data/processing/pSTAT1.csv').drop('Unnamed: 0', axis=1)
    # Just FYI, these are the values of df levels:
    # times_20190214 = [2.5, 5, 7.5, 10, 20, 60]
    # alpha_doses_20190214 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    # beta_doses_20190214 = [0, 0.2, 6, 20, 60, 200, 600, 2000]
    df = df.sort_values(['Cytokine', 'Well'])
    df.replace('IFN-alpha', 'Alpha', inplace=True)
    df.replace('IFN-beta', 'Beta', inplace=True)
    df.loc[:, 'Concentration (Mol)'] *= 1e12
    newcolumns = [df.columns.values[0], 'Dose (pM)', 'Dose_Species', *df.columns.values[3:]]
    df.columns = newcolumns
    # Small B cells
    # -------------
    Bcell_df = df.drop(
        ['Well', 'pSTAT1 in CD8+ T cells', 'pSTAT1 in CD4+ T cells', 'pSTAT1 in B cells', 'pSTAT1 in Large B cells'],
        axis=1)

    # Convert values to (value-background, error)
    temp = Bcell_df.values
    temp = [[float("{0:.1f}".format(row[0])), row[1], row[2], float("{0:.2f}".format(row[3]))] for row in
            temp]
    Bcell_df = pd.DataFrame.from_records(temp,
                                         columns=['Dose (pM)', 'Dose_Species', 'Time (min)', 'pSTAT1 in Small B cells'])

    # Put in standard form
    Bcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    Bcell_df = pd.pivot_table(Bcell_df, values='pSTAT1 in Small B cells', index=['Dose_Species', 'Dose (pM)'],
                              columns=['Time (min)'], aggfunc=np.sum)
    Bcell_df.columns.name = None

    # Zero the values by the 0 dose row
    record = []
    for species in ['Alpha', 'Beta']:
        background = pd.Series(Bcell_df.loc[species].loc[0.0])
        for dose in Bcell_df.loc[species].index:
            values = Bcell_df.loc[species].loc[dose] - background
            values = values.values
            record.append([species, dose] + [(el, np.nan) for el in values])

    col_labels = ['Dose_Species', 'Dose (pM)'] + [float("{0:.1f}".format(t)) for t in
                                                  Bcell_df.loc['Alpha'].columns.values]
    Bcell_df = pd.DataFrame.from_records(record, columns=col_labels)
    # Put in standard form
    Bcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    Bcell_df.drop(0.0, level=1, axis=0, inplace=True)

    # Save
    pickle.dump(Bcell_df, open(os.path.join(data_wd, '20190119_pSTAT1_IFN_Small_Bcells.p'), 'wb'))

    # Large B cells
    # -------------
    Bcell_df = df.drop(
        ['Well', 'pSTAT1 in CD8+ T cells', 'pSTAT1 in CD4+ T cells', 'pSTAT1 in B cells', 'pSTAT1 in Small B cells'],
        axis=1)

    # Convert values to (value-background, error)
    temp = Bcell_df.values
    temp = [[float("{0:.1f}".format(row[0])), row[1], row[2], float("{0:.2f}".format(row[3]))] for row in
            temp]
    Bcell_df = pd.DataFrame.from_records(temp,
                                         columns=['Dose (pM)', 'Dose_Species', 'Time (min)', 'pSTAT1 in Large B cells'])

    # Put in standard form
    Bcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    Bcell_df = pd.pivot_table(Bcell_df, values='pSTAT1 in Large B cells', index=['Dose_Species', 'Dose (pM)'],
                              columns=['Time (min)'], aggfunc=np.sum)
    Bcell_df.columns.name = None

    # Zero the values by the 0 dose row
    record = []
    for species in ['Alpha', 'Beta']:
        background = pd.Series(Bcell_df.loc[species].loc[0.0])
        for dose in Bcell_df.loc[species].index:
            values = Bcell_df.loc[species].loc[dose] - background
            values = values.values
            record.append([species, dose] + [(el, np.nan) for el in values])

    col_labels = ['Dose_Species', 'Dose (pM)'] + [float("{0:.1f}".format(t)) for t in
                                                  Bcell_df.loc['Alpha'].columns.values]
    Bcell_df = pd.DataFrame.from_records(record, columns=col_labels)
    # Put in standard form
    Bcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    Bcell_df.drop(0.0, level=1, axis=0, inplace=True)

    # Save
    pickle.dump(Bcell_df, open(os.path.join(data_wd, '20190119_pSTAT1_IFN_Large_Bcells.p'), 'wb'))


    # ---------------------
    # 20190121
    # ---------------------
    df = pd.read_csv('20190121_raw_data/processing/pSTAT1.csv').drop('Unnamed: 0', axis=1)
    # Just FYI, these are the values of df levels:
    # times_20190214 = [2.5, 5, 7.5, 10, 20, 60]
    # alpha_doses_20190214 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    # beta_doses_20190214 = [0, 0.2, 6, 20, 60, 200, 600, 2000]
    df = df.sort_values(['Cytokine', 'Well'])
    df.replace('IFN-alpha', 'Alpha', inplace=True)
    df.replace('IFN-beta', 'Beta', inplace=True)
    df.loc[:, 'Concentration (Mol)'] *= 1e12
    newcolumns = [df.columns.values[0], 'Dose (pM)', 'Dose_Species', *df.columns.values[3:]]
    df.columns = newcolumns
    # Small B cells
    # -------------
    Bcell_df = df.drop(['Well', 'pSTAT1 in CD8+ T cells', 'pSTAT1 in CD4+ T cells', 'pSTAT1 in B cells', 'pSTAT1 in Large B cells'], axis=1)

    # Convert values to (value-background, error)
    temp = Bcell_df.values
    temp = [[float("{0:.1f}".format(row[0])), row[1], row[2], float("{0:.2f}".format(row[3]))] for row in
            temp]
    Bcell_df = pd.DataFrame.from_records(temp, columns=['Dose (pM)', 'Dose_Species', 'Time (min)', 'pSTAT1 in Small B cells'])

    # Put in standard form
    Bcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    Bcell_df = pd.pivot_table(Bcell_df, values='pSTAT1 in Small B cells', index=['Dose_Species', 'Dose (pM)'],
                              columns=['Time (min)'], aggfunc=np.sum)
    Bcell_df.columns.name = None

    # Zero the values by the 0 dose row
    record = []
    for species in ['Alpha', 'Beta']:
        background = pd.Series(Bcell_df.loc[species].loc[0.0])
        for dose in Bcell_df.loc[species].index:
            values = Bcell_df.loc[species].loc[dose]-background
            values = values.values
            record.append([species, dose]+[(el, np.nan) for el in values])

    col_labels = ['Dose_Species', 'Dose (pM)'] + [float("{0:.1f}".format(t)) for t in Bcell_df.loc['Alpha'].columns.values]
    Bcell_df = pd.DataFrame.from_records(record, columns=col_labels)
    # Put in standard form
    Bcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    Bcell_df.drop(0.0, level=1, axis=0, inplace=True)

    # Save
    pickle.dump(Bcell_df, open(os.path.join(data_wd, '20190121_pSTAT1_IFN_Small_Bcells.p'), 'wb'))

    # Large B cells
    # -------------
    Bcell_df = df.drop(
        ['Well', 'pSTAT1 in CD8+ T cells', 'pSTAT1 in CD4+ T cells', 'pSTAT1 in B cells', 'pSTAT1 in Small B cells'],
        axis=1)

    # Convert values to (value-background, error)
    temp = Bcell_df.values
    temp = [[float("{0:.1f}".format(row[0])), row[1], row[2], float("{0:.2f}".format(row[3]))] for row in
            temp]
    Bcell_df = pd.DataFrame.from_records(temp,
                                         columns=['Dose (pM)', 'Dose_Species', 'Time (min)', 'pSTAT1 in Large B cells'])

    # Put in standard form
    Bcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    Bcell_df = pd.pivot_table(Bcell_df, values='pSTAT1 in Large B cells', index=['Dose_Species', 'Dose (pM)'],
                              columns=['Time (min)'], aggfunc=np.sum)
    Bcell_df.columns.name = None

    # Zero the values by the 0 dose row
    record = []
    for species in ['Alpha', 'Beta']:
        background = pd.Series(Bcell_df.loc[species].loc[0.0])
        for dose in Bcell_df.loc[species].index:
            values = Bcell_df.loc[species].loc[dose] - background
            values = values.values
            record.append([species, dose] + [(el, np.nan) for el in values])

    col_labels = ['Dose_Species', 'Dose (pM)'] + [float("{0:.1f}".format(t)) for t in
                                                  Bcell_df.loc['Alpha'].columns.values]
    Bcell_df = pd.DataFrame.from_records(record, columns=col_labels)
    # Put in standard form
    Bcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    Bcell_df.drop(0.0, level=1, axis=0, inplace=True)

    # Save
    pickle.dump(Bcell_df, open(os.path.join(data_wd, '20190121_pSTAT1_IFN_Large_Bcells.p'), 'wb'))

# ---------------------
    # 20190214
    # ---------------------
    df = pd.read_csv('20190214_raw_data/processing/pSTAT1.csv').drop('Unnamed: 0', axis=1)
    # Just FYI, these are the values of df levels:
    # times_20190214 = [2.5, 5, 7.5, 10, 20, 60]
    # alpha_doses_20190214 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    # beta_doses_20190214 = [0, 0.2, 6, 20, 60, 200, 600, 2000]
    df = df.sort_values(['Cytokine', 'Well'])
    df.replace('IFN-alpha', 'Alpha', inplace=True)
    df.replace('IFN-beta', 'Beta', inplace=True)
    df.loc[:, 'Concentration (Mol)'] *= 1e12
    newcolumns = [df.columns.values[0], 'Dose (pM)', 'Dose_Species', *df.columns.values[3:]]
    df.columns = newcolumns
    # Small B cells
    # -------------
    Bcell_df = df.drop(['Well', 'pSTAT1 in CD8+ T cells', 'pSTAT1 in CD4+ T cells', 'pSTAT1 in B cells', 'pSTAT1 in Large B cells'], axis=1)

    # Convert values to (value-background, error)
    temp = Bcell_df.values
    temp = [[float("{0:.1f}".format(row[0])), row[1], row[2], float("{0:.2f}".format(row[3]))] for row in
            temp]
    Bcell_df = pd.DataFrame.from_records(temp, columns=['Dose (pM)', 'Dose_Species', 'Time (min)', 'pSTAT1 in Small B cells'])

    # Put in standard form
    Bcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    Bcell_df = pd.pivot_table(Bcell_df, values='pSTAT1 in Small B cells', index=['Dose_Species', 'Dose (pM)'],
                              columns=['Time (min)'], aggfunc=np.sum)
    Bcell_df.columns.name = None

    # Zero the values by the 0 dose row
    record = []
    for species in ['Alpha', 'Beta']:
        background = pd.Series(Bcell_df.loc[species].loc[0.0])
        for dose in Bcell_df.loc[species].index:
            values = Bcell_df.loc[species].loc[dose]-background
            values = values.values
            record.append([species, dose]+[(el, np.nan) for el in values])

    col_labels = ['Dose_Species', 'Dose (pM)'] + [float("{0:.1f}".format(t)) for t in Bcell_df.loc['Alpha'].columns.values]
    Bcell_df = pd.DataFrame.from_records(record, columns=col_labels)
    # Put in standard form
    Bcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    Bcell_df.drop(0.0, level=1, axis=0, inplace=True)

    # Save
    pickle.dump(Bcell_df, open(os.path.join(data_wd, '20190214_pSTAT1_IFN_Small_Bcells.p'), 'wb'))

    # Large B cells
    # -------------
    Bcell_df = df.drop(
        ['Well', 'pSTAT1 in CD8+ T cells', 'pSTAT1 in CD4+ T cells', 'pSTAT1 in B cells', 'pSTAT1 in Small B cells'],
        axis=1)

    # Convert values to (value-background, error)
    temp = Bcell_df.values
    temp = [[float("{0:.1f}".format(row[0])), row[1], row[2], float("{0:.2f}".format(row[3]))] for row in
            temp]
    Bcell_df = pd.DataFrame.from_records(temp,
                                         columns=['Dose (pM)', 'Dose_Species', 'Time (min)', 'pSTAT1 in Large B cells'])

    # Put in standard form
    Bcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    Bcell_df = pd.pivot_table(Bcell_df, values='pSTAT1 in Large B cells', index=['Dose_Species', 'Dose (pM)'],
                              columns=['Time (min)'], aggfunc=np.sum)
    Bcell_df.columns.name = None

    # Zero the values by the 0 dose row
    record = []
    for species in ['Alpha', 'Beta']:
        background = pd.Series(Bcell_df.loc[species].loc[0.0])
        for dose in Bcell_df.loc[species].index:
            values = Bcell_df.loc[species].loc[dose] - background
            values = values.values
            record.append([species, dose] + [(el, np.nan) for el in values])

    col_labels = ['Dose_Species', 'Dose (pM)'] + [float("{0:.1f}".format(t)) for t in
                                                  Bcell_df.loc['Alpha'].columns.values]
    Bcell_df = pd.DataFrame.from_records(record, columns=col_labels)
    # Put in standard form
    Bcell_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
    Bcell_df.drop(0.0, level=1, axis=0, inplace=True)

    # Save
    pickle.dump(Bcell_df, open(os.path.join(data_wd, '20190214_pSTAT1_IFN_Large_Bcells.p'), 'wb'))

if __name__ == '__main__':
    build_database(os.getcwd())