import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from functools import reduce
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

ImmGen_df = pd.read_excel('ImmGen_signaling_with_protein_response.xlsx', sheet_name='Sheet1', axis=1)
#create df of results and features from full dataframe
response_variable_names = ['pSTAT1', 'pSTAT3']
response_variables = ImmGen_df[response_variable_names]

predictor_variable_names = [c for c in ImmGen_df.columns.values if c not in response_variable_names + ['Cell_type']]
predictor_variables = ImmGen_df[predictor_variable_names]

def plot_normalized_pSTAT():
    ImmGen_df['pSTAT1norm'] = ImmGen_df['pSTAT1']/ImmGen_df['STAT1']
    ImmGen_df['pSTAT3norm'] = ImmGen_df['pSTAT3']/ImmGen_df['STAT3']

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 5))
    subplot_titles = ['pSTAT1', 'pSTAT3']
    sns.barplot(x="Cell_type", y="pSTAT1norm", data=ImmGen_df, ax=ax[0])
    sns.barplot(x="Cell_type", y="pSTAT3norm", data=ImmGen_df, ax=ax[1])
    for i in [0, 1]:
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=90)
        ax[i].set_title(subplot_titles[i])
    plt.tight_layout()
    plt.show()

def fit_normalized_pSTAT():
    ImmGen_df['pSTAT1norm'] = ImmGen_df['pSTAT1']/ImmGen_df['STAT1']
    ImmGen_df['pSTAT3norm'] = ImmGen_df['pSTAT3']/ImmGen_df['STAT3']

    def prediction(dose, p1, p2, p3, p4, p5, p6, df=ImmGen_df, SOCS_name='SOCS2', fit=True):
        parameters = [p1, p2, p3, p4, p5, p6]
        def __infer_protein__(dataset, cell_type, protein_names):
            if 'allSOCS' in protein_names:
                all_SOCS_names = ['SOCS1', 'SOCS2', 'SOCS3', 'SOCS4', 'SOCS5', 'SOCS6', 'SOCS7', 'CIS', 'SHP1', 'SHP2',
                                  'PIAS1', 'PIAS2', 'PIAS3', 'PIAS4']
                SOCS_transcripts = np.sum(
                    dataset.loc[dataset['Cell_type'] == cell_type][all_SOCS_names].values.flatten())
                transcripts = dataset.loc[dataset['Cell_type'] == cell_type][
                    [p for p in protein_names if p != 'allSOCS']].values.flatten()
                d = dict(zip([p for p in protein_names if p != 'allSOCS'], transcripts))
                d['allSOCS'] = SOCS_transcripts
                return d
            else:
                transcripts = dataset.loc[dataset['Cell_type'] == cell_type][protein_names].values.flatten()
                # mean_transcripts = dataset.mean()[protein_names].values
                # proteins = [mean_transcripts[i] * 2**(np.log10(transcripts[i]/mean_transcripts[i])) for i in range(len(protein_names))]
                return dict(zip(protein_names, transcripts))

        # Create the receptor models
        # global variables applicable to all receptors
        NA = 6.022E23
        cell_density = 1E6  # cells per mL
        volEC = 1E-3 / cell_density  # L
        rad_cell = 5E-6  # m
        pi = np.pi
        volPM = 4 * pi * rad_cell ** 2  # m**2
        volCP = (4 / 3) * pi * rad_cell ** 3  # m**3

        cell_types = ['Granulocytes', 'Ly6Chi_Mo', 'Ly6Clo_Mo', 'preB_FrC', 'preB_FrD', 'MPP34F', 'ST34F', 'CMP', 'GMP',
                      'MEP', 'CDP', 'MDP', 'LT-HSC', 'ST-HSC', 'Mac_Sp', 'CD11b+DC', 'CD11b-DC', 'NK', 'Mac_BM']
        base_parameters = {'pSTAT1': {'R_total': 2000,  # infer from Immgen
                                      'Delta': 0,  # infer from Immgen
                                      'Jak1': 1000,  # infer from Immgen
                                      'Jak2': 1000,  # infer from Immgen
                                      'STAT_total': 2000,  # infer from Immgen
                                      SOCS_name: 200,  # infer from Immgen
                                      'K_ligand': NA * volEC / (4 * pi * 0.5E10),  # from literature
                                      'K_Jak1': 1E6 / (NA * volCP),  # fit for each receptor
                                      'K_Jak2': 1E6 / (NA * volCP),  # fit for each receptor
                                      'K_R1R2': 4 * pi * 0.5E-12 / volPM,  # from literature
                                      'K_STAT': 1000 / volPM,  # fit for each receptor/STAT pair
                                      'K_SOCS': 1000},  # fit for each receptor

                           'pSTAT3': {'R_total': 2000,
                                      'Delta': 0,
                                      'Jak1': 1000,
                                      'Jak2': 1000,
                                      'STAT_total': 2000,
                                      SOCS_name: 200,
                                      'K_ligand': NA * volEC / (4 * pi * 0.5E10),
                                      'K_Jak1': 1E6 / (NA * volCP),
                                      'K_Jak2': 1E6 / (NA * volCP),
                                      'K_R1R2': 4 * pi * 0.5E-12 / volPM,
                                      'K_STAT': 1000 / volPM,
                                      'K_SOCS': 1000}
                           }
        output_names = ['pSTAT1', 'pSTAT3']
        protein_names = ['IFNGR1', 'IFNGR2', 'JAK1', 'JAK2', 'STAT1', 'STAT3', SOCS_name]
        dose = dose * 1E-12 * NA * volEC
        for STAT in output_names:
            base_parameters[STAT]['K_Jak1'] = parameters[0]
            base_parameters[STAT]['K_Jak2'] = parameters[1]
            base_parameters[STAT]['K_SOCS'] = parameters[4]
        # receptor:STAT parameters
        base_parameters['pSTAT1']['K_STAT'] = parameters[2]
        base_parameters['pSTAT3']['K_STAT'] = parameters[3]

        fit_pred = []
        for c in cell_types:
            # transcripts = df.loc[df['Cell_type'] == c][protein_names].values.flatten()
            # mean_transcripts = df.mean()[protein_names].values
            # proteins = [mean_transcripts[i] * 2**(np.log10(transcripts[i]/mean_transcripts[i])) for i in range(len(protein_names))]

            # df.loc[df['Cell_type'] == c][protein_names].values.flatten()
            IFNg_ImmGen_parameters = __infer_protein__(df, c, protein_names)

            STAT_response = []
            for S in output_names:
                base_parameters[S]['R_total'] = IFNg_ImmGen_parameters['IFNGR1'] + IFNg_ImmGen_parameters['IFNGR2']
                base_parameters[S]['Delta'] = IFNg_ImmGen_parameters['IFNGR1'] - IFNg_ImmGen_parameters['IFNGR2']
                base_parameters[S]['Jak1'] = IFNg_ImmGen_parameters['JAK1']
                base_parameters[S]['Jak2'] = IFNg_ImmGen_parameters['JAK2']
                base_parameters[S]['STAT_total'] = IFNg_ImmGen_parameters[
                    S[1:]]  # remove leading 'p' from 'pSTAT1' or 'pSTAT3'
                base_parameters[S][SOCS_name] = IFNg_ImmGen_parameters[SOCS_name]

                cytokine_R = base_parameters[S]['K_R1R2'] / 2 * \
                             (1 + base_parameters[S]['R_total'] / base_parameters[S]['K_R1R2'] - \
                              np.sqrt(1 + (2 * base_parameters[S]['R_total'] * base_parameters[S]['K_R1R2'] \
                                           + base_parameters[S]['Delta'] ** 2) / base_parameters[S]['K_R1R2'] ** 2))
                PR1active = base_parameters[S]['Jak1'] * base_parameters[S]['K_Jak1'] / (
                            1 + base_parameters[S]['Jak1'] * base_parameters[S]['K_Jak1'])
                PR2active = base_parameters[S]['Jak2'] * base_parameters[S]['K_Jak2'] / (
                            1 + base_parameters[S]['Jak2'] * base_parameters[S]['K_Jak2'])
                Rstar = cytokine_R * PR1active * PR2active / (1 + base_parameters[S]['K_ligand'] / dose)
                response = base_parameters[S]['STAT_total'] / (1 + base_parameters[S]['K_STAT'] * (
                        1 + base_parameters[S][SOCS_name] / base_parameters[S]['K_SOCS']) * volPM / Rstar)
                response = response / (parameters[5] * base_parameters[S]['STAT_total'])
                STAT_response.append(response)
            fit_pred.append(STAT_response)
        if fit==True:
            return np.array(fit_pred).flatten()
        else:
            return fit_pred

    ## ['K_Jak1', 'K_Jak2', 'K_STAT_STAT1', 'K_STAT_STAT3', 'K_SOCS', 'scale_factor']
    p_best = [3.17147014e-03,   3.17147014e+00,   3.18309889e+13,   1.00760330e+12,   1.67829414e-02, 10]
    pfit, pcov = curve_fit(prediction, 100, ImmGen_df[['pSTAT1norm', 'pSTAT3norm']].values.flatten(), p0=p_best,
                               bounds=(np.multiply(p_best, 0.000001), np.multiply(p_best, 10000000)))
    print(pfit)
    cell_types = ['Granulocytes','Ly6Chi_Mo','Ly6Clo_Mo','preB_FrC','preB_FrD','MPP34F','ST34F','CMP','GMP','MEP','CDP','MDP','LT-HSC','ST-HSC','Mac_Sp','CD11b+DC','CD11b-DC','NK','Mac_BM']
    best_pred = prediction(100, *pfit, fit=False)
    # r-squared
    residuals = np.subtract(best_pred, ImmGen_df[['pSTAT1norm', 'pSTAT1norm']].values)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((ImmGen_df[['pSTAT1norm', 'pSTAT3norm']].values - np.mean(ImmGen_df[['pSTAT1norm', 'pSTAT3norm']].values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)


    best_pred = [['Model', cell_types[r], best_pred[r][0], best_pred[r][1]] for r in range(len(best_pred))]
    fit_df = pd.DataFrame(best_pred, columns=['Class', 'Cell_type', 'pSTAT1norm', 'pSTAT3norm'])

    ImmGen_df.insert(0, 'Class', ['CyTOF' for _ in range(ImmGen_df.shape[0])])

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 5))
    subplot_titles = ['pSTAT1', 'pSTAT3']
    sns.barplot(x="Cell_type", y="pSTAT1norm",
                data=pd.concat([fit_df, ImmGen_df]),
                ax=ax[0], hue='Class')
    sns.barplot(x="Cell_type", y="pSTAT3norm",
                data=pd.concat([fit_df, ImmGen_df]),
                ax=ax[1], hue='Class')
    for i in [0, 1]:
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=90)
        ax[i].set_title(subplot_titles[i])

    plt.suptitle('R squared = {:.2f}'.format(r_squared))
    plt.tight_layout()
    plt.show()

def plot_pSTAT_space():
    X = ImmGen_df[['pSTAT1','pSTAT3']].values
    #y_pred = KMeans(n_clusters=3, random_state=4).fit_predict(X)
    stem_palette = sns.color_palette("Greys", 5)[::-1]
    myeloid_palette = sns.color_palette("Greens", 5)
    lymphoid_palette = sns.color_palette("cubehelix", 8)
    hematopoetic_colormap = {"MPP34F": stem_palette[2], "ST34F": stem_palette[2], "CMP": stem_palette[3], "GMP": stem_palette[3], "MEP": stem_palette[3],
                             "CDP": stem_palette[3], "MDP": stem_palette[3], "LT-HSC": stem_palette[1], "ST-HSC": stem_palette[1],
                             "Ly6Chi_Mo": myeloid_palette[2], "Ly6Clo_Mo": myeloid_palette[2],
                             "Granulocytes": myeloid_palette[3], "Mac_BM": myeloid_palette[3], "Mac_Sp": myeloid_palette[3],
                             "CD11b+DC": myeloid_palette[3], "CD11b-DC": myeloid_palette[3],
                             "preB_FrC": lymphoid_palette[1], "preB_FrD": lymphoid_palette[1],
                             "NK": lymphoid_palette[4]}
    y_pred = [hematopoetic_colormap[x] for x in ImmGen_df.Cell_type.values]
    plt.scatter(X[:,0], X[:,1], c=y_pred)
    plt.xlabel('pSTAT1')
    plt.ylabel('pSTAT3')
    p1=plt.gca()
    for line in range(0, X.shape[0]):
        p1.text(X[line,0] + 0.2, X[line,1], ImmGen_df.Cell_type.values[line],
                horizontalalignment='left', size='medium', color='black', weight='semibold')
    plt.show()

def plot_expression_space(gene_list=[]):
    df = ImmGen_df.apply(lambda x: np.log10(x) if np.issubdtype(x.dtype, np.number) else x)
    if gene_list==[]:
        hist = df.hist(column=[i for i in ImmGen_df.columns.values if i not in ['pSTAT1', 'pSTAT3']])
    else:
        hist = df.hist(column=gene_list)
    for ax in hist.flatten():
        ax.set_xlabel("log expression")
        ax.set_ylabel("frequency")

    plt.suptitle('R squard = {:.2f}'.format(r_squared))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #plot_expression_space(['STAT3', 'STAT1', 'JAK1', 'JAK2', 'SOCS2', 'Usp18', 'PIAS2', 'SOCS6'])
    plot_pSTAT_space()

    fit_normalized_pSTAT()
