import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from scipy.optimize import curve_fit
import copy

# PyDREAM imports
from pydream.core import run_dream
from pysb.integrate import Solver
from pydream.parameters import SampledParam
from scipy.stats import norm, uniform
import os
from datetime import datetime
import inspect
from pydream.convergence import Gelman_Rubin


# Import data set and split into predictor variables (receptors, JAKs, SOCS, etc.) and response variables (STATs)
IFNg_dose = 100 # pM
ImmGen_df = pd.read_excel('MasterTable_ImmGen_pSTAT.xlsx', sheet_name='G-CSF', axis=1)
ImmGen_df['pSTAT1norm'] = ImmGen_df['pSTAT1']/ImmGen_df['STAT1']
ImmGen_df['pSTAT3norm'] = ImmGen_df['pSTAT3']/ImmGen_df['STAT3']

#create df of results and features from full dataframe
response_variable_names = ['pSTAT1', 'pSTAT3']
response_variables = ImmGen_df[response_variable_names]

predictor_variable_names = [c for c in ImmGen_df.columns.values if c not in response_variable_names + ['Cell_type']]
predictor_variables = ImmGen_df[predictor_variable_names]

# Compute pairwise correlations between all variables
def pairwise_correlation():
    f = plt.figure(figsize=(19, 15))
    df = ImmGen_df[response_variable_names + predictor_variable_names]
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
    plt.tight_layout()
    plt.show()

def SOCS_histograms():
    labels = ['SOCS1', 'SOCS2', 'SOCS3', 'SOCS4', 'SOCS5', 'SOCS6', 'SOCS7', 'PIAS1', 'PIAS2', 'PIAS3', 'PIAS4', 'CIS', 'SHP1', 'SHP2']
    colours = sns.color_palette(n_colors=len(labels))
    # Plot each individually
    fig, axes = plt.subplots(nrows=int(np.ceil(len(labels)/3.)), ncols=3)
    for lidx, l in enumerate(labels):
        sns.distplot(ImmGen_df[l], color=colours[lidx], label=l, ax=axes[int(lidx/3)][lidx%3], axlabel=l)
    plt.tight_layout()
    plt.show()

    # Plot on same axis
    for lidx, l in enumerate(['SOCS1', 'SOCS2', 'SOCS3', 'PIAS1', 'SHP1']):
        sns.distplot(ImmGen_df[l], color=colours[lidx], label=l)
    ax = plt.gca()
    fig = plt.gcf()
    ax.set_xlim(left=30, right=5000)
    ax.set_xscale('log')
    ax.set_xlabel('Gene Expression (# transcripts)')
    ax.set_ylabel('Frequency')
    fig.legend()
    plt.tight_layout()
    plt.show()



# Create the receptor models
#global variables applicable to all receptors
NA = 6.022E23
cell_density = 1E6 # cells per mL
volEC = 1E-3/cell_density # L
rad_cell = 5E-6 # m
pi = np.pi
volPM = 4*pi*rad_cell**2 # m**2
volCP = (4/3)*pi*rad_cell**3 # m**3

class Cytokine_Receptor:
    """Cytokine_Receptor class
    This class allows the user to quickly create copies of the same equilibrium
    receptor model which have different parameters, and then use this model to
    make predictions and fit data.
Attributes:
    STAT_names (list): each element is a string identifying the name of an equilibrium_model_output from activating this receptor
    parameters (dict): each entry is a dict with key from self.STAT_names, and entry is another dict which has
                        for keys the names (str) and values (float or int) for the equilibrium model of that
                        corresponding STAT response; parameters should be given in units of molecules where
                        possible, not molarity.
        keys: R_total, Delta, Jak1, Jak2, STAT_total, K_ligand, K_Jak1, K_Jak2, K_R1R2, K_STAT, K_rc
    cytokine_name (str): the name of the cytokine which binds specifically to this receptor
Methods:
    def equilibrium_model_output():
        :param cytokine_dose: (float) the stimulation of cytokine in pM
        :return STAT_response: (dict) the predicted pSTAT response for each key, STAT, in self.STAT_names
    def equilibrium_model_for_SOCS_competing_with_STAT():
        :param cytokine_dose: (float) the stimulation of cytokine in pM
        :return STAT_response: (dict) the predicted pSTAT response for each key, STAT, in self.STAT_names
"""
    def __init__(self, STAT_names, parameters, cytokine_name):
        self.STAT_names = STAT_names
        self.parameters = parameters
        self.cytokine_name = cytokine_name
        if not all(elem in self.STAT_names for elem in self.parameters.keys()):
            print("Not all parameters were defined for all STAT outputs")
            raise KeyError

    def equilibrium_model_output(self, cytokine_dose):
        dose = cytokine_dose*1E-12*NA*volEC
        STAT_response = {}
        for STAT in self.STAT_names:
            cytokine_R = self.parameters[STAT]['K_R1R2']/2 * \
                         (1 + self.parameters[STAT]['R_total']/self.parameters[STAT]['K_R1R2'] -\
                            np.sqrt(1 + (2*self.parameters[STAT]['R_total']*self.parameters[STAT]['K_R1R2']\
                                         + self.parameters[STAT]['Delta']**2)/self.parameters[STAT]['K_R1R2']**2))
            Rstar = cytokine_R*self.parameters[STAT]['Jak1']*self.parameters[STAT]['K_Jak1']*self.parameters[STAT]['Jak2']*\
                    self.parameters[STAT]['K_Jak2']/(1+self.parameters[STAT]['K_ligand']/dose)
            response = self.parameters[STAT]['STAT_total']/(1+self.parameters[STAT]['K_STAT']*volPM/Rstar)
            STAT_response[STAT] = response
        return STAT_response

    def equilibrium_model_for_SOCS_competing_with_STAT(self, cytokine_dose, SOCS_name):
        dose = cytokine_dose*1E-12*NA*volEC
        STAT_response = {}
        for STAT in self.STAT_names:
            cytokine_R = self.parameters[STAT]['K_R1R2']/2 * \
                         (1 + self.parameters[STAT]['R_total']/self.parameters[STAT]['K_R1R2'] -\
                            np.sqrt(1 + (2*self.parameters[STAT]['R_total']*self.parameters[STAT]['K_R1R2']\
                                         + self.parameters[STAT]['Delta']**2)/self.parameters[STAT]['K_R1R2']**2))
            PR1active = self.parameters[STAT]['Jak1']*self.parameters[STAT]['K_Jak1'] / (1 + self.parameters[STAT]['Jak1']*self.parameters[STAT]['K_Jak1'])
            PR2active = self.parameters[STAT]['Jak2']*self.parameters[STAT]['K_Jak2'] / (1 + self.parameters[STAT]['Jak2']*self.parameters[STAT]['K_Jak2'])
            Rstar = cytokine_R*PR1active*PR2active/(1+self.parameters[STAT]['K_ligand']/dose)
            print(Rstar)
            exit()
            response = self.parameters[STAT]['STAT_total']/(1+self.parameters[STAT]['K_STAT']*(1+self.parameters[STAT][SOCS_name]/self.parameters[STAT]['K_SOCS'])*volPM/Rstar)
            STAT_response[STAT] = response
        return STAT_response

    def equilibrium_model_with_SOCS_ec50(self, SOCS_name):
        ec50_dict = {}
        for STAT in self.STAT_names:
            term1 = self.parameters[STAT]['Jak1'] * self.parameters[STAT]['K_Jak1'] * self.parameters[STAT]['Jak2'] * self.parameters[STAT]['K_Jak2']
            term2 = self.parameters[STAT]['K_R1R2'] + self.parameters[STAT]['R_total'] - self.parameters[STAT]['K_R1R2'] * np.sqrt((self.parameters[STAT]['K_R1R2']**2 + 2*self.parameters[STAT]['K_R1R2']*self.parameters[STAT]['R_total']+self.parameters[STAT]['Delta']**2)/self.parameters[STAT]['K_R1R2']**2)
            term3 = (1+self.parameters[STAT]['K_Jak1']*self.parameters[STAT]['Jak1']) * (1+self.parameters[STAT]['K_Jak2']*self.parameters[STAT]['Jak2'])
            term4 = self.parameters[STAT]['K_STAT']*(self.parameters[STAT]['K_STAT'] + self.parameters[STAT][SOCS_name])
            numerator = 2*volPM*self.parameters[STAT]['K_ligand']*term3*term4
            denominator = 2*volPM*term3*term4 + self.parameters[STAT]['K_SOCS']*term1*term2
            ec50 = numerator/denominator
            ec50_dict[STAT] = ec50/(NA*volEC) # (M)
        return ec50_dict

    def equilibrium_model_with_SOCS_pSTATmax(self, SOCS_name):
        pSTAT_max_dict = {}
        for STAT in self.STAT_names:
            term1 = self.parameters[STAT]['Jak1'] * self.parameters[STAT]['K_Jak1'] * self.parameters[STAT]['Jak2'] * self.parameters[STAT]['K_Jak2']
            term2 = self.parameters[STAT]['K_R1R2'] + self.parameters[STAT]['R_total'] - self.parameters[STAT]['K_R1R2'] * np.sqrt((self.parameters[STAT]['K_R1R2']**2 + 2*self.parameters[STAT]['K_R1R2']*self.parameters[STAT]['R_total']+self.parameters[STAT]['Delta']**2)/self.parameters[STAT]['K_R1R2']**2)
            numerator = self.parameters[STAT]['K_SOCS']*self.parameters[STAT]['STAT_total']*term1*term2
            term3 = (1+self.parameters[STAT]['K_Jak1']*self.parameters[STAT]['Jak1']) * (1+self.parameters[STAT]['K_Jak2']*self.parameters[STAT]['Jak2'])
            denominator = 2*volPM*term3*self.parameters[STAT]['K_STAT']*(self.parameters[STAT]['K_SOCS']+self.parameters[STAT][SOCS_name]) + self.parameters[STAT]['K_SOCS']*term1*term2
            pSTAT_max_dict[STAT] = numerator/denominator
        return pSTAT_max_dict


default_SOCS_name = 'SOCS2'
IFNg_parameters = {'pSTAT1': {'R_total': 2000,                      # infer from Immgen
                              'Delta': 0,                           # infer from Immgen
                              'Jak1': 1000,                         # infer from Immgen
                              'Jak2': 1000,                         # infer from Immgen
                              'STAT_total': 2000,                   # infer from Immgen
                              default_SOCS_name: 200,               # infer from Immgen
                              'USP18': 0,                           # infer from Immgen
                              'K_ligand': NA*volEC/(4*pi*0.5E10),   # from literature
                              'K_Jak1': 1E6/(NA*volCP),             # fit for each receptor
                              'K_Jak2': 1E6/(NA*volCP),             # fit for each receptor
                              'K_R1R2': 4*pi*0.5E-12/volPM,         # from literature
                              'K_STAT': 1000/volPM,                 # fit for each receptor/STAT pair
                              'K_SOCS': 1000,                       # fit for each receptor
                              'K_USP18': 150},                      # fit for each receptor

                   'pSTAT3': {'R_total': 2000,
                              'Delta': 0,
                              'Jak1': 1000,
                              'Jak2': 1000,
                              'STAT_total': 2000,
                              default_SOCS_name: 200,
                              'USP18': 0,
                              'K_ligand': NA*volEC/(4*pi*0.5E10),
                              'K_Jak1': 1E6/(NA*volCP),
                              'K_Jak2': 1E6/(NA*volCP),
                              'K_R1R2': 4*pi*0.5E-12/volPM,
                              'K_STAT': 1000/volPM,
                              'K_SOCS': 1000,
                              'K_USP18': 150}
                    }


def plot_dose_response(IFNg_parameters, SOCS_name, model=1):
    IFNg_receptor = Cytokine_Receptor(['pSTAT1', 'pSTAT3'], IFNg_parameters, 'IFNgamma')
    if model==1:
        pSTAT1_dose_response = [IFNg_receptor.equilibrium_model_output(d, SOCS_name)['pSTAT1'] for d in np.logspace(-2, 3)]
    elif model==2:
        pSTAT1_dose_response = [IFNg_receptor.equilibrium_model_for_SOCS_competing_with_STAT(d, SOCS_name)['pSTAT1'] for d in np.logspace(-2, 3)]
    plt.figure()
    ax = plt.gca()
    ax.set_xscale('log')
    plt.xlabel(r'IFN$\gamma$ (pM)')
    plt.ylabel('pSTAT1 (# molec.)')
    plt.plot(np.logspace(-2, 3), pSTAT1_dose_response)
    plt.show()


def infer_protein(dataset, cell_type, protein_names):
    """
    Assumes that the steady-state protein level is simply one-to-one with the transcript level measured.
    :param dataset: pandas dataframe with ImmGen transcript levels for the cell_type of interest
    :param cell_type: row label to select from dataset
    :param protein_names: column names to extract transcript values from dataset
    :return: proteins (dict): keys are protein_names and values are the numbers of the proteins to use in
            Cytokine_Receptor.parameters
    """
    if 'allSOCS' in protein_names:
        all_SOCS_names = ['SOCS1', 'SOCS2', 'SOCS3', 'SOCS4', 'SOCS5', 'SOCS6', 'SOCS7', 'CIS', 'SHP1', 'SHP2', 'PIAS1', 'PIAS2', 'PIAS3', 'PIAS4']
        SOCS_transcripts = np.sum(dataset.loc[dataset['Cell_type'] == cell_type][all_SOCS_names].values.flatten())
        transcripts = dataset.loc[dataset['Cell_type'] == cell_type][[p for p in protein_names if p!='allSOCS']].values.flatten()
        d = dict(zip([p for p in protein_names if p!='allSOCS'], transcripts))
        d['allSOCS'] = SOCS_transcripts
        return d
    else:
        transcripts = dataset.loc[dataset['Cell_type'] == cell_type][protein_names].values.flatten()
        #mean_transcripts = dataset.mean()[protein_names].values
        #proteins = [mean_transcripts[i] * 2**(np.log10(transcripts[i]/mean_transcripts[i])) for i in range(len(protein_names))]
        return dict(zip(protein_names, transcripts))


def equilibrium_pSTAT1_and_pSTAT3(dose, K_Jak1=1E6 / (NA * volCP), K_Jak2=1E6 / (NA * volCP),
                                  K_STAT_STAT1=1000/volPM, K_STAT_STAT3=1000/volPM, K_USP18=150):
    cell_types = ImmGen_df['Cell_type'].values

    # Set input parameters for model
    default_parameters = IFNg_parameters.copy()
    # receptor parameters
    default_parameters['pSTAT1']['K_Jak1'] = K_Jak1
    default_parameters['pSTAT3']['K_Jak1'] = K_Jak1
    default_parameters['pSTAT1']['K_Jak2'] = K_Jak2
    default_parameters['pSTAT3']['K_Jak2'] = K_Jak2
    default_parameters['pSTAT1']['K_USP18'] = K_USP18
    default_parameters['pSTAT3']['K_USP18'] = K_USP18

    # receptor:STAT parameters
    default_parameters['pSTAT1']['K_STAT'] = K_STAT_STAT1
    default_parameters['pSTAT3']['K_STAT'] = K_STAT_STAT3

    response = {'Cell_type': [], 'pSTAT1': [], 'pSTAT3': []}
    for c in cell_types:
        IFNg_ImmGen_parameters = infer_protein(ImmGen_df, c, ['IFNGR1', 'IFNGR2', 'JAK1', 'JAK2', 'STAT1', 'STAT3'])
        for S in ['pSTAT1', 'pSTAT3']:
            default_parameters[S]['R_total'] = IFNg_ImmGen_parameters['CSF3r']/(1+IFNg_ImmGen_parameters['USP18']/default_parameters[S]['K_USP18'])
            default_parameters[S]['Delta'] = 0
            default_parameters[S]['Jak1'] = IFNg_ImmGen_parameters['JAK1']
            default_parameters[S]['Jak2'] = IFNg_ImmGen_parameters['JAK2']
            default_parameters[S]['STAT_total'] = IFNg_ImmGen_parameters[S[1:]]
        # Make predictions
        IFNg_receptor = Cytokine_Receptor(['pSTAT1', 'pSTAT3'], IFNg_parameters, 'GCSF')
        q = IFNg_receptor.equilibrium_model_output(dose)
        response['Cell_type'].append(c)
        response['pSTAT1'].append(q['pSTAT1'])
        response['pSTAT3'].append(q['pSTAT3'])
    return pd.DataFrame.from_dict(response)


def SOCS_competes_STAT_pSTAT1_and_pSTAT3(dose, SOCS_name, K_Jak1=1E6 / (NA * volCP), K_Jak2=1E6 / (NA * volCP),
                                         K_STAT_STAT1=1000/volPM, K_STAT_STAT3=1000/volPM, K_SOCS=1000, K_USP18=150, df=ImmGen_df):
    cell_types = df['Cell_type'].values

    # Set input parameters for model
    default_parameters = copy.deepcopy(IFNg_parameters)
    # receptor parameters
    default_parameters['pSTAT1']['K_Jak1'] = K_Jak1
    default_parameters['pSTAT3']['K_Jak1'] = K_Jak1
    default_parameters['pSTAT1']['K_Jak2'] = K_Jak2
    default_parameters['pSTAT3']['K_Jak2'] = K_Jak2
    default_parameters['pSTAT1']['K_SOCS'] = K_SOCS
    default_parameters['pSTAT3']['K_SOCS'] = K_SOCS
    default_parameters['pSTAT1']['K_USP18'] = K_USP18
    default_parameters['pSTAT3']['K_USP18'] = K_USP18

    # receptor:STAT parameters
    default_parameters['pSTAT1']['K_STAT'] = K_STAT_STAT1
    default_parameters['pSTAT3']['K_STAT'] = K_STAT_STAT3

    response = {'Cell_type': [], 'pSTAT1': [], 'pSTAT3': []}
    for c in cell_types:
        IFNg_ImmGen_parameters = infer_protein(df, c, ['CSF3r', 'JAK1', 'JAK2', 'STAT1', 'STAT3', SOCS_name, 'USP18'])
        for S in ['pSTAT1', 'pSTAT3']:
            default_parameters[S]['R_total'] = IFNg_ImmGen_parameters['CSF3r']/(1+IFNg_ImmGen_parameters['USP18']/default_parameters[S]['K_USP18'])
            default_parameters[S]['Delta'] = 0
            default_parameters[S]['Jak1'] = IFNg_ImmGen_parameters['JAK1']
            default_parameters[S]['Jak2'] = IFNg_ImmGen_parameters['JAK2']
            default_parameters[S]['STAT_total'] = IFNg_ImmGen_parameters[S[1:]]
            default_parameters[S][SOCS_name] = IFNg_ImmGen_parameters[SOCS_name]
        # Make predictions
        IFNg_receptor = Cytokine_Receptor(['pSTAT1', 'pSTAT3'], default_parameters, 'IFNgamma')
        q = IFNg_receptor.equilibrium_model_for_SOCS_competing_with_STAT(dose, SOCS_name)
        response['Cell_type'].append(c)
        response['pSTAT1'].append(q['pSTAT1'])
        response['pSTAT3'].append(q['pSTAT3'])
    return pd.DataFrame.from_dict(response)


def equilibrium_model(dose, p1, p2, p3, p4, scale_factor):
    y_pred = equilibrium_pSTAT1_and_pSTAT3(dose, K_Jak1=p1, K_Jak2=p2, K_STAT_STAT1=p3, K_STAT_STAT3=p4)[['pSTAT1', 'pSTAT3']]
    return np.divide(y_pred.values.flatten(), scale_factor)


def fit_IFNg_equilibrium():
    default_parameters = [1E7 / (NA * volCP), 1E7 / (NA * volCP), 10000 / volPM, 10000 / volPM, 70]

    y_true = ImmGen_df[['pSTAT1', 'pSTAT3']].values.flatten()
    pfit, pcov = curve_fit(equilibrium_model, IFNg_dose, y_true, p0=default_parameters,
                           bounds=(np.multiply(default_parameters, 0.1), np.multiply(default_parameters, 10)))
    return pfit, pcov


def make_SOCS_competes_STAT_model(SOCS_name, df=ImmGen_df):
    def SOCS_model(dose, p1, p2, p3, p4, p5, scale_factor, p6):
        y_pred = SOCS_competes_STAT_pSTAT1_and_pSTAT3(dose, SOCS_name, K_Jak1=p1, K_Jak2=p2, K_STAT_STAT1=p3,
                                                      K_STAT_STAT3=p4, K_SOCS=p5, K_USP18=p6, df=df)[['pSTAT1', 'pSTAT3']]
        return np.divide(y_pred.values.flatten(), scale_factor)
    return SOCS_model


def fit_IFNg_with_SOCS_competes_STAT(SOCS_name, df=ImmGen_df):
    if SOCS_name=='allSOCS':
        default_parameters = [1E7 / (NA * volCP), 1E8 / (NA * volCP), 1000 / volPM, 1000 / volPM, 1.6e-02, 1]
    else:
        min_response_row = df.loc[df[response_variable_names[0]].idxmin()]
        min_response_SOCS_expression = infer_protein(df, min_response_row.loc['Cell_type'], [SOCS_name])[SOCS_name]
        #
        # [4.64594123e-02,   6.72673774e+00,   9.50605181e+00,   2.92182513e+00,   2.08449593e-01,   1.63205262e+02]
        #                       K_Jak1, K_Jak2, K_STAT_STAT1, K_STAT_STAT3, K_SOCS, scale_factor
        default_parameters = [1E7 / (NA * volCP), 1E8 / (NA * volCP), 1000 / volPM, 1000 / volPM, 0.0006*min_response_SOCS_expression, 1]
    y_true = df[['pSTAT1', 'pSTAT3']].values.flatten()
    pfit, pcov = curve_fit(make_SOCS_competes_STAT_model(SOCS_name, df), IFNg_dose, y_true, p0=default_parameters,
                           bounds=(np.multiply(default_parameters, 0.1), np.multiply(default_parameters, 10)))
    return pfit, pcov


def k_fold_cross_validate_SOCS_competes_STAT(k=10, df=ImmGen_df, neg_feedback_name='SOCS2'):
    """
    Validate the estimated R^2 value for the SOCS model fit using k-fold cross validation
    :param k: (int) number of folds to split data into
    :param df: (DataFrame) combined predictor and response variables
    :return r2, r2_variance: (floats) the estimated R^2 value and variance in the estimate
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    r2_samples = []
    for result in kf.split(df):
        # Split data
        train = df.iloc[result[0]]
        test = df.iloc[result[1]]

        # Fit
        pfit, pcov = fit_IFNg_with_SOCS_competes_STAT(neg_feedback_name, df=train)

        # Predict
        SOCS_model = make_SOCS_competes_STAT_model(neg_feedback_name, df=test)
        fit_pred = np.reshape(SOCS_model(IFNg_dose, *pfit), (test.shape[0], len(response_variable_names)))

        # Compute R**2 value
        residuals = np.subtract(np.log(fit_pred), np.log(test[['pSTAT1', 'pSTAT3']].values))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((np.log(test[['pSTAT1', 'pSTAT3']].values)-np.log(np.mean(test[['pSTAT1', 'pSTAT3']].values)))**2)
        r_squared = 1 - (ss_res / ss_tot)
        r2_samples.append(r_squared)

    return np.mean(r2_samples), np.var(r2_samples)


def fit_without_SOCS(df=ImmGen_df):
    pfit, pcov = fit_IFNg_equilibrium()
    print(pfit)

    # Predict
    fit_pred = np.reshape(equilibrium_model(IFNg_dose, *pfit), (15, len(response_variable_names)))
    fit_pred_labelled = [[df['Cell_type'].values[i], fit_pred[i][0], fit_pred[i][1]] for i in range(df.shape[0])]

    # Compute R**2 value
    residuals = np.subtract(np.log(fit_pred), np.log(df[['pSTAT1', 'pSTAT3']].values))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((np.log(df[['pSTAT1', 'pSTAT3']].values) - np.log(np.mean(df[['pSTAT1', 'pSTAT3']].values))) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print("r-squared value (log scale): ", r_squared)

    # Plot
    fit_prediction = pd.DataFrame.from_records(fit_pred_labelled, columns=['Cell_type', 'pSTAT1', 'pSTAT3'])
    fit_prediction.insert(0, 'Class', ['Model' for _ in range(df.shape[0])])
    measured_response = df[['Cell_type', 'pSTAT1', 'pSTAT3']]
    measured_response.insert(0, 'Class', ['CyTOF' for _ in range(df.shape[0])])

    df = pd.concat([fit_prediction, measured_response])

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 5))
    subplot_titles = ['pSTAT1', 'pSTAT3']
    sns.barplot(x="Cell_type", y="pSTAT1", data=df, hue='Class', ax=ax[0])
    sns.barplot(x="Cell_type", y="pSTAT3", data=df, hue='Class', ax=ax[1])
    for i in [0, 1]:
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=90)
        ax[i].set_title(subplot_titles[i])

    plt.suptitle(r"$R^{2}$ (log scale) = " + "{:.2f}".format(r_squared))
    plt.tight_layout()
    plt.show()


def fit_with_SOCS_competes_STAT(df=ImmGen_df, k_fold=1, neg_feedback_name='SOCS2'):
    pfit, pcov = fit_IFNg_with_SOCS_competes_STAT(neg_feedback_name, df)
    print(pfit)

    # Predict
    SOCS_model = make_SOCS_competes_STAT_model(neg_feedback_name)
    fit_pred = np.reshape(SOCS_model(IFNg_dose, *pfit), (df.shape[0], len(response_variable_names)))
    fit_pred_labelled = [[df['Cell_type'].values[i], fit_pred[i][0], fit_pred[i][1]] for i in range(df.shape[0])]

    # Compute R**2 value
    if k_fold==1:
        residuals = np.subtract(np.log(fit_pred), np.log(df[['pSTAT1', 'pSTAT3']].values))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((np.log(df[['pSTAT1', 'pSTAT3']].values) - np.log(np.mean(df[['pSTAT1', 'pSTAT3']].values))) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print("r-squared value (log scale): ", r_squared)
    else:
        r_squared, r2_var = k_fold_cross_validate_SOCS_competes_STAT(k=k_fold)
        print("k-fold cross validation of R2 value (in log scale) is estimated to be {:.2f} +/- {:.2f}".format(r_squared, np.sqrt(r2_var)))

    # Plot
    fit_prediction = pd.DataFrame.from_records(fit_pred_labelled, columns=['Cell_type', 'pSTAT1', 'pSTAT3'])
    fit_prediction.insert(0, 'Class', ['Model' for _ in range(df.shape[0])])
    measured_response = df[['Cell_type', 'pSTAT1', 'pSTAT3']]
    measured_response.insert(0, 'Class', ['CyTOF' for _ in range(df.shape[0])])

    df = pd.concat([fit_prediction, measured_response])

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 5))
    subplot_titles = ['pSTAT1', 'pSTAT3']
    sns.barplot(x="Cell_type", y="pSTAT1", data=df, hue='Class', ax=ax[0])
    sns.barplot(x="Cell_type", y="pSTAT3", data=df, hue='Class', ax=ax[1])
    for i in [0,1]:
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=90)
        ax[i].set_title(subplot_titles[i])

    if k_fold==1:
        plt.suptitle(r"$R^{2}$ (log scale) = "+ "{:.2f}".format(r_squared))
    else:
        plt.suptitle(r"$R^{2}$ (log scale) = " + "{:.2f}".format(r_squared) + " ({}-fold cross-validation)".format(k_fold))
    plt.tight_layout()
    plt.show()


def compare_model_errors(df=ImmGen_df, neg_feedback_name='SOCS2'):
    # ------------
    # With SOCS
    # ------------
    pfit, pcov = fit_IFNg_with_SOCS_competes_STAT(neg_feedback_name, df)

    # Predict
    SOCS_model = make_SOCS_competes_STAT_model(neg_feedback_name)
    SOCS_fit_pred = np.reshape(SOCS_model(IFNg_dose, *pfit), (df.shape[0], len(response_variable_names)))

    # Residual
    SOCS_residuals = np.divide(np.subtract(SOCS_fit_pred, df[['pSTAT1', 'pSTAT3']].values), df[['pSTAT1', 'pSTAT3']].values)
    #SOCS_res_pred_labelled = [[df['Cell_type'].values[i], SOCS_residuals[i][0], SOCS_residuals[i][1]] for i in range(df.shape[0])]

    # ------------
    # Without SOCS
    # ------------
    pfit, pcov = fit_IFNg_equilibrium()

    # Predict
    fit_pred = np.reshape(equilibrium_model(IFNg_dose, *pfit), (df.shape[0], len(response_variable_names)))

    # Residual
    residuals = np.divide(np.subtract(fit_pred, df[['pSTAT1', 'pSTAT3']].values), df[['pSTAT1', 'pSTAT3']].values)
    #res_pred_labelled = [[df['Cell_type'].values[i], residuals[i][0], residuals[i][1]] for i in range(df.shape[0])]

    # ------------
    # Plot
    # ------------
    fig, ax = plt.subplots(nrows=1, ncols=2)
    for i in [0, 1]:
        ax[i].set_title('pSTAT{}'.format(i*2 + 1))
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_xlabel('% error in no-SOCS model')
        ax[i].set_ylabel('% error in SOCS model')
        ax[i].scatter(np.abs(residuals[:, i]), np.abs(SOCS_residuals[:, i]))
        ax[i].plot(np.logspace(np.log10(min(np.abs(residuals[:, i]))), np.log10(max(np.abs(residuals[:, i])))),
                   np.logspace(np.log10(min(np.abs(residuals[:, i]))), np.log10(max(np.abs(residuals[:, i])))), 'k--')
    plt.tight_layout()
    plt.show()


def fit_with_DREAM(sim_name, parameter_dict, likelihood):
    original_params = [parameter_dict[k] for k in parameter_dict.keys()]

    priors_list = []
    for p in original_params:
        priors_list.append(SampledParam(norm, loc=np.log(p), scale=1.0))
    # Set simulation parameters
    niterations = 10000
    converged = False
    total_iterations = niterations
    nchains = 5

    # Make save directory
    today = datetime.now()
    save_dir = "PyDREAM_" + today.strftime('%d-%m-%Y') + "_" + str(niterations)
    os.makedirs(os.path.join(os.getcwd(), save_dir), exist_ok=True)

    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    sampled_params, log_ps = run_dream(priors_list, likelihood, start=np.log(original_params),
                                       niterations=niterations, nchains=nchains, multitry=False,
                                       gamma_levels=4, adapt_gamma=True, history_thin=1, model_name=sim_name,
                                       verbose=True)

    # Save sampling output (sampled parameter values and their corresponding logps).
    for chain in range(len(sampled_params)):
        np.save(os.path.join(save_dir, sim_name + str(chain) + '_' + str(total_iterations)), sampled_params[chain])
        np.save(os.path.join(save_dir, sim_name + str(chain) + '_' + str(total_iterations)), log_ps[chain])

    # Check convergence and continue sampling if not converged

    GR = Gelman_Rubin(sampled_params)
    print('At iteration: ', total_iterations, ' GR = ', GR)
    np.savetxt(os.path.join(save_dir, sim_name + str(total_iterations) + '.txt'), GR)

    old_samples = sampled_params
    if np.any(GR > 1.2):
        starts = [sampled_params[chain][-1, :] for chain in range(nchains)]
        while not converged:
            total_iterations += niterations

            sampled_params, log_ps = run_dream(priors_list, likelihood, start=starts, niterations=niterations,
                                               nchains=nchains, multitry=False, gamma_levels=4, adapt_gamma=True,
                                               history_thin=1, model_name=sim_name, verbose=True, restart=True)

            for chain in range(len(sampled_params)):
                np.save(os.path.join(save_dir, sim_name + '_' + str(chain) + '_' + str(total_iterations)),
                        sampled_params[chain])
                np.save(os.path.join(save_dir, sim_name + '_' + str(chain) + '_' + str(total_iterations)),
                        log_ps[chain])

            old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
            GR = Gelman_Rubin(old_samples)
            print('At iteration: ', total_iterations, ' GR = ', GR)
            np.savetxt(os.path.join(save_dir, sim_name + '_' + str(total_iterations) + '.txt'), GR)

            if np.all(GR < 1.2):
                converged = True
    try:
        # Plot output
        total_iterations = len(old_samples[0])
        burnin = int(total_iterations / 2)
        samples = np.concatenate(list((old_samples[i][burnin:, :] for i in range(len(old_samples)))))
        np.save(os.path.join(save_dir, sim_name+'_samples'), samples)
        ndims = len(old_samples[0][0])
        colors = sns.color_palette(n_colors=ndims)
        for dim in range(ndims):
            fig = plt.figure()
            sns.distplot(samples[:, dim], color=colors[dim])
            fig.savefig(os.path.join(save_dir, sim_name + '_dimension_' + str(dim) + '_' + list(parameter_dict.keys())[dim]+ '.pdf'))

        # Convert to dataframe
        df = pd.DataFrame(samples, columns=parameter_dict.keys())
        g = sns.pairplot(df)
        for i, j in zip(*np.triu_indices_from(g.axes, 1)):
            g.axes[i,j].set_visible(False)
        g.savefig(os.path.join(save_dir, 'corner_plot.pdf'))

        # Basic statistics
        mean_parameters = np.mean(samples, axis=0)
        median_parameters = np.median(samples, axis=0)
        np.save(os.path.join(save_dir, 'mean_parameters'), mean_parameters)
        np.save(os.path.join(save_dir, 'median_parameters'), median_parameters)
        df.describe().to_csv(os.path.join(save_dir, 'descriptive_statistics.csv'))

    except ImportError:
        pass
    return 0


def fit_IFNg_SOCS_competes_STAT_with_DREAM(SOCS_name, df=ImmGen_df):
    def __likelihood__(parameters, df=ImmGen_df):
        import numpy as np
        from scipy.stats import norm

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

        parameters = np.exp(parameters) # parameters are passed in log form

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
                      'MEP', 'CDP', 'MDP', 'LT-HSC', 'ST-HSC', 'Mac_BM']
        base_parameters = {'pSTAT1': {'R_total': 2000, # infer from Immgen
                              'Delta': 0,                           # infer from Immgen
                              'CSF3r': 2000,                        # infer from Immgen
                              'Jak1': 1000,                         # infer from Immgen
                              'Jak2': 1000,                         # infer from Immgen
                              'STAT_total': 2000,                   # infer from Immgen
                              SOCS_name: 200,                       # infer from Immgen
                              'USP18': 0,                           # infer from ImmGen
                              'K_ligand': NA*volEC/(4*pi*0.5E10),   # from literature
                              'K_Jak1': 1E6/(NA*volCP),             # fit for each receptor
                              'K_Jak2': 1E6/(NA*volCP),             # fit for each receptor
                              'K_R1R2': 4*pi*0.5E-12/volPM,         # from literature
                              'K_STAT': 1000/volPM,           # fit for each receptor/STAT pair
                              'K_SOCS': 1000,                 # fit for each receptor
                              'K_USP18': 150},        # fit for each receptor

                   'pSTAT3': {'R_total': 2000,
                              'Delta': 0,
                              'CSF3r': 2000,
                              'Jak1': 1000,
                              'Jak2': 1000,
                              'STAT_total': 2000,
                               SOCS_name: 200,
                              'USP18': 0,
                              'K_ligand': NA*volEC/(4*pi*0.5E10),
                              'K_Jak1': 1E6/(NA*volCP),
                              'K_Jak2': 1E6/(NA*volCP),
                              'K_R1R2': 4*pi*0.5E-12/volPM,
                              'K_STAT': 1000/volPM,
                              'K_SOCS': 1000,
                              'K_USP18': 150}
                    }
        output_names = ['pSTAT1', 'pSTAT3']
        protein_names = ['CSF3r', 'JAK1', 'JAK2', 'STAT1', 'STAT3', SOCS_name, 'USP18']
        dose = 100 * 1E-12 * NA * volEC
        for STAT in output_names:
            base_parameters[STAT]['K_Jak1'] = parameters[0]
            base_parameters[STAT]['K_Jak2'] = parameters[1]
            base_parameters[STAT]['K_SOCS'] = parameters[4]
            base_parameters[STAT]['K_USP18'] = parameters[6]

        # receptor:STAT parameters
        base_parameters['pSTAT1']['K_STAT'] = parameters[2]
        base_parameters['pSTAT3']['K_STAT'] = parameters[3]

        fit_pred = []
        for c in cell_types:
            #transcripts = df.loc[df['Cell_type'] == c][protein_names].values.flatten()
            #mean_transcripts = df.mean()[protein_names].values
            #proteins = [mean_transcripts[i] * 2**(np.log10(transcripts[i]/mean_transcripts[i])) for i in range(len(protein_names))]

            #df.loc[df['Cell_type'] == c][protein_names].values.flatten()
            IFNg_ImmGen_parameters = __infer_protein__(df, c, protein_names)

            STAT_response = []
            for S in output_names:
                base_parameters[S]['R_total'] = IFNg_ImmGen_parameters['CSF3r']/(1+IFNg_ImmGen_parameters['USP18']/base_parameters[S]['K_USP18'])
                base_parameters[S]['Delta'] = 0
                base_parameters[S]['Jak1'] = IFNg_ImmGen_parameters['JAK1']
                base_parameters[S]['Jak2'] = IFNg_ImmGen_parameters['JAK2']
                base_parameters[S]['STAT_total'] = IFNg_ImmGen_parameters[S[1:]] # remove leading 'p' from 'pSTAT1' or 'pSTAT3'
                base_parameters[S][SOCS_name] = IFNg_ImmGen_parameters[SOCS_name]

                cytokine_R = base_parameters[S]['K_R1R2'] / 2 * \
                             (1 + base_parameters[S]['R_total'] / base_parameters[S]['K_R1R2'] - \
                              np.sqrt(1 + (2 * base_parameters[S]['R_total'] * base_parameters[S]['K_R1R2'] \
                                           + base_parameters[S]['Delta'] ** 2) / base_parameters[S]['K_R1R2'] ** 2))
                PR1active = base_parameters[S]['Jak1'] * base_parameters[S]['K_Jak1'] / (1 + base_parameters[S]['Jak1'] * base_parameters[S]['K_Jak1'])
                PR2active = base_parameters[S]['Jak2'] * base_parameters[S]['K_Jak2'] / (1 + base_parameters[S]['Jak2'] * base_parameters[S]['K_Jak2'])
                Rstar = cytokine_R * PR1active * PR2active / (1 + base_parameters[S]['K_ligand'] / dose)
                response = base_parameters[S]['STAT_total'] / (1 + base_parameters[S]['K_STAT'] * (
                            1 + base_parameters[S][SOCS_name] / base_parameters[S]['K_SOCS']) * volPM / Rstar)
                response = response / parameters[5]
                STAT_response.append(response)
            fit_pred.append(STAT_response)

        # sse
        # GCSF at 100 pM
        data = [[14.6, 37.26],
                [2.43, 16.9],
                [1.42, 5.54],
                [0.18, 0.32],
                [0.64, 0.41],
                [2.32, 51.8],
                [0.18, 34.2],
                [2.49, 18.53],
                [1.74, 22.65],
                [1.05, 10.8],
                [1.09, 3.14],
                [1.51, 27.3],
                [1.99, 11.68],
                [1.09, 29.66],
                [4.9, 7.4]]

        like_ctot = norm(loc=np.log10(data), scale=np.ones(np.shape(data)))
        logp_ctotal = np.sum(like_ctot.logpdf(np.log10(fit_pred)))

        # If model simulation failed due to integrator errors, return a log probability of -inf.
        if np.isnan(logp_ctotal):
            logp_ctotal = -np.inf
        return logp_ctotal


    #min_response_row = df.loc[df[response_variable_names[0]].idxmin()]
    #min_response_SOCS_expression = infer_protein(df, min_response_row.loc['Cell_type'], [SOCS_name])[SOCS_name]
    # old: [1E7 / (NA * volCP), 1E6 / (NA * volCP), 900 / volPM, 1000 / volPM, min_response_SOCS_expression, 1]
    #       K_Jak1, K_Jak2, K_STAT_STAT1, K_STAT_STAT3, K_SOCS, scale_factor
    prior = [3.17147014e-03,   3.17147014e+00,   3.18309889e+11,   1.00760330e+12,   1.67829414e-02, 1e-01, 150]
    prior = dict(zip(['K_Jak1', 'K_Jak2', 'K_STAT_STAT1', 'K_STAT_STAT3', 'K_SOCS', 'scale_factor', 'K_USP18'], prior))

    fit_with_DREAM(SOCS_name, prior, __likelihood__)


def sample_DREAM_IFNg_SOCS_competes_STAT(samples_filename, neg_feedback_name, df=ImmGen_df, step_size=500, find_map=True):
    samples = np.load(samples_filename)
    # Convert back to true value rather than log value
    converted_samples = np.exp(samples)

    # Predict
    SOCS_model = make_SOCS_competes_STAT_model(neg_feedback_name)
    fit_pred = []
    sample_count = 0
    for p in range(0, len(converted_samples), step_size):
        sample_count += 1
        pred = np.reshape(SOCS_model(IFNg_dose, *converted_samples[p]), (15, len(response_variable_names)))
        pred_labelled = [['Model', df['Cell_type'].values[i], pred[i][0], pred[i][1]] for i in range(df.shape[0])]
        fit_pred += pred_labelled
    print("{} samples used".format(sample_count))

    fit_prediction = pd.DataFrame.from_records(fit_pred, columns=['Class', 'Cell_type', 'pSTAT1', 'pSTAT3'])

    cell_type_pred = fit_prediction.drop('Class', axis=1).groupby('Cell_type')

    # Compute posterior predictive p-value
    # First compute the posterior of the test statistic. Use the residual from the log[median prediction].
    # The motivation for this is that for log-normal distributed predictions, the median is the point
    # for which there is a 50 % chance of a given posterior prediction being greater.
    # i.e. E[log[median prediction] - log[data]] = 0 when median prediction = data
    log_median_pred = np.log(cell_type_pred.median().values)
    test_statistic_distribution = []
    total_res_samples = 0
    for p in range(0, len(converted_samples), step_size):
        total_res_samples += 1
        log_pred = np.log(np.reshape(SOCS_model(IFNg_dose, *converted_samples[p]), (15, len(response_variable_names))))
        residual = np.sum(np.subtract(log_median_pred, log_pred))
        test_statistic_distribution.append(residual)
    # Compute what fraction of predictions have a residual greater than the median with the observed data
    data_residual = np.sum(np.subtract(log_median_pred, np.log(df[['pSTAT1', 'pSTAT3']].values)))
    r_count = 0
    for r in test_statistic_distribution:
        if r > data_residual:
            r_count += 1
    ppp_value = r_count/total_res_samples

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 5))
    #ax[0].set_yscale('log')
    #ax[1].set_yscale('log')
    subplot_titles = ['pSTAT1', 'pSTAT3']
    measured_response = df[['Cell_type', 'pSTAT1', 'pSTAT3']]
    measured_response.insert(0, 'Class', ['CyTOF' for _ in range(df.shape[0])])

    plot_df = pd.concat([fit_prediction, measured_response])

    sns.catplot(x="Cell_type", y="pSTAT1", data=plot_df[plot_df['Class']=='Model'], hue='Class', ax=ax[0], kind="bar")
    sns.catplot(x="Cell_type", y="pSTAT3", data=plot_df[plot_df['Class']=='Model'], hue='Class', ax=ax[1], kind="bar")

    sns.catplot(x="Cell_type", y="pSTAT1", data=plot_df[plot_df['Class'] == 'CyTOF'], hue='Class', ax=ax[0],
                kind="strip", size=8, palette=sns.color_palette("hls", 16))
    sns.catplot(x="Cell_type", y="pSTAT3", data=plot_df[plot_df['Class']=='CyTOF'], hue='Class', ax=ax[1],
                kind="strip", size=8, palette=sns.color_palette("hls", 16))
    for i in [0, 1]:
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=90)
        ax[i].set_title(subplot_titles[i])
    fig.suptitle("Posterior p-value = {:.2f}\n (p close to 0.5 is good)".format(ppp_value), fontsize=12)
    #plt.tight_layout()
    plt.show()

    # Find best single estimator
    if find_map:
        best_parameter = []
        best_residual = 1E16
        data = np.log(df[['pSTAT1', 'pSTAT3']].values.clip(min=1E-6)) # necessary because there are a few negative meas.
        for p in range(0, len(converted_samples), 1):
            log_pred = np.log(np.reshape(SOCS_model(IFNg_dose, *converted_samples[p]), (15, len(response_variable_names))))
            square_residual = np.sum(np.square(np.subtract(log_pred, data)))
            if square_residual < best_residual:
                best_parameter = converted_samples[p]
                best_residual = square_residual
        print(best_parameter)
        print(best_residual)


def compare_model_to_ImmGen(pset, SOCS_name='SOCS2'):
    # Predict
    SOCS_model = make_SOCS_competes_STAT_model(SOCS_name)
    fit_pred = np.reshape(SOCS_model(IFNg_dose, *pset), (ImmGen_df.shape[0], len(response_variable_names)))
    fit_pred_labelled = [[ImmGen_df['Cell_type'].values[i], fit_pred[i][0], fit_pred[i][1]] for i in
                         range(ImmGen_df.shape[0])]

    # R-squared
    residuals = np.subtract(np.log(fit_pred), np.log(ImmGen_df[['pSTAT1', 'pSTAT3']].values.clip(min=1E-6)))
    ss_res = np.sum(np.square(residuals))
    ss_tot = np.sum(np.square((np.log(ImmGen_df[['pSTAT1', 'pSTAT3']].values.clip(min=1E-6)) - np.log(np.mean(ImmGen_df[['pSTAT1', 'pSTAT3']].values.clip(min=1E-6))))))
    r_squared = 1 - (ss_res / ss_tot)

    # non-log R-squared
    residuals = np.subtract(fit_pred, ImmGen_df[['pSTAT1', 'pSTAT3']].values.clip(min=1E-6))
    ss_res = np.sum(np.square(residuals))
    ss_tot = np.sum(np.square((ImmGen_df[['pSTAT1', 'pSTAT3']].values.clip(min=1E-6) - np.mean(ImmGen_df[['pSTAT1', 'pSTAT3']].values.clip(min=1E-6)))))
    print("r-squared (linear) = ", 1 - (ss_res / ss_tot))

    # Plot
    fit_prediction = pd.DataFrame.from_records(fit_pred_labelled, columns=['Cell_type', 'pSTAT1', 'pSTAT3'])
    fit_prediction.insert(0, 'Class', ['Model' for _ in range(ImmGen_df.shape[0])])
    measured_response = ImmGen_df[['Cell_type', 'pSTAT1', 'pSTAT3']]
    measured_response.insert(0, 'Class', ['CyTOF' for _ in range(ImmGen_df.shape[0])])

    df = pd.concat([fit_prediction, measured_response])

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 5))
    fig.suptitle('R-squared (log scale) = {:.2f}'.format(r_squared))
    subplot_titles = ['pSTAT1', 'pSTAT3']
    sns.barplot(x="Cell_type", y="pSTAT1", data=df, hue='Class', ax=ax[0])
    sns.barplot(x="Cell_type", y="pSTAT3", data=df, hue='Class', ax=ax[1])
    for i in [0, 1]:
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=90)
        ax[i].set_title(subplot_titles[i])
    plt.tight_layout()
    plt.show()

    # Measured vs Predicted
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 5))
    fig.suptitle('R-squared (log scale) = {:.2f}'.format(r_squared))
    subplot_titles = ['pSTAT1', 'pSTAT3']
    ax[0].scatter(measured_response.pSTAT1.values, fit_prediction.pSTAT1.values)
    ax[0].plot([0, 20], [0, 20], 'k--')
    ax[1].scatter(measured_response.pSTAT3.values, fit_prediction.pSTAT3.values)
    ax[1].plot([0, 40], [0, 40], 'k--')
    for i in [0, 1]:
        ax[i].set_title(subplot_titles[i])
        ax[i].set_xlabel('Measured')
        ax[i].set_ylabel('Predicted')
    plt.show()

if __name__ == "__main__":
    # Check variance in predictors
    #df = ImmGen_df[response_variable_names + predictor_variable_names]
    #print(pd.Series(np.diag(df.cov().values), index=df.columns))

    #pairwise_correlation()
    #SOCS_histograms()

    #fit_without_SOCS()
    #fit_with_SOCS_competes_STAT(neg_feedback_name='allSOCS')

    #compare_model_errors()

    #print(ec50_for_all_cell_types('SOCS2'))
    #make_ec50_predictions_plot()

    #fit_IFNg_SOCS_competes_STAT_with_DREAM('SOCS7')

    save_dir = "PyDREAM_04-12-2019_10000_GCSF"
    sim_name = "SOCS7"
    #sample_DREAM_IFNg_SOCS_competes_STAT(os.path.join(save_dir, sim_name+'_samples' + '.npy'), sim_name, step_size=250, find_map=True)

    ## ['K_Jak1', 'K_Jak2', 'K_STAT_STAT1', 'K_STAT_STAT3', 'K_SOCS', 'scale_factor', 'K_USP18']
    #min_response_row = ImmGen_df.loc[ImmGen_df[response_variable_names[0]].idxmin()]
    #min_response_SOCS_expression = infer_protein(ImmGen_df, min_response_row.loc['Cell_type'], [sim_name])[sim_name]
    # [1E7 / (NA * volCP), 1E6 / (NA * volCP), 900 / volPM, 1000 / volPM, 0.006*min_response_SOCS_expression, 0.5]
    #p_prior = [3.171470138e-02, 3.17147013799e-03, 2.864788975654e+12, 3.183098861837e+12, 5.77815e-01, 5e-01]
    p_best = [3.17147014e-03,   3.17147014e+00,   3.18309889e+11,   1.00760330e+12,   1.67829414e-02, 1e-01]

    # SOCSS2
    p_best_by_MCMC = [9.19293300e+02, 3.33434463e+01, 2.20985969e+04, 2.71081941e+02, 4.54009422e-01, 4.88918115e+01, 1.49339725e-03]

    # SOCS7
    #p_best_by_MCMC = [98.96069551, 850.44550422, 97.8516537, 448.41519007, 52.27743194, 437.50172928, 353.38670015]
    compare_model_to_ImmGen(p_best_by_MCMC, SOCS_name='SOCS2')

