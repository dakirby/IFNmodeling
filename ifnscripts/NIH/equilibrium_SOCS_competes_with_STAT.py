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
ImmGen_df = pd.read_excel('ImmGen_signaling_with_protein_response.xlsx', sheet_name='Sheet1', axis=1)
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
    STAT_names (list): each element is a string identifying the name of a STAT activated by this receptor
    parameters (dict): each entry is a dict with key from self.STAT_names, and entry is another dict which has
                        for keys the names (str) and values (float or int) for the equilibrium model of that
                        corresponding STAT response; parameters should be given in units of molecules where
                        possible, not molarity.
        keys: R_total, Delta, Jak1, Jak2, STAT_total, K_ligand, K_Jak1, K_Jak2, K_R1R2, K_STAT, K_rc
    cytokine_name (str): the name of the cytokine which binds specifically to this receptor

Methods:
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
            response = self.parameters[STAT]['STAT_total']/(1+self.parameters[STAT]['K_STAT']*(1+self.parameters[STAT][SOCS_name]/self.parameters[STAT]['K_SOCS'])*volPM/Rstar)
            STAT_response[STAT] = response
        return STAT_response

    def add_Jak1(self, cytokine_dose, SOCS_name):
        dose = cytokine_dose*1E-12*NA*volEC
        STAT_response = {}
        for STAT in self.STAT_names:
            cytokine_R = self.parameters[STAT]['K_R1R2']/2 * \
                         (1 + self.parameters[STAT]['R_total']/self.parameters[STAT]['K_R1R2'] -\
                            np.sqrt(1 + (2*self.parameters[STAT]['R_total']*self.parameters[STAT]['K_R1R2']\
                                         + self.parameters[STAT]['Delta']**2)/self.parameters[STAT]['K_R1R2']**2))
            PR1active = self.parameters[STAT]['Jak1']*self.parameters[STAT]['K_Jak1'] / (1 + self.parameters[STAT]['Jak1']*self.parameters[STAT]['K_Jak1'])
            Rstar = cytokine_R*PR1active/(1+self.parameters[STAT]['K_ligand']/dose)
            response = self.parameters[STAT]['STAT_total']/(1+self.parameters[STAT]['K_STAT']*volPM/Rstar)
            STAT_response[STAT] = response
        return STAT_response

    def add_Jak2(self, cytokine_dose):
        dose = cytokine_dose*1E-12*NA*volEC
        STAT_response = {}
        for STAT in self.STAT_names:
            cytokine_R = self.parameters[STAT]['K_R1R2']/2 * \
                         (1 + self.parameters[STAT]['R_total']/self.parameters[STAT]['K_R1R2'] -\
                            np.sqrt(1 + (2*self.parameters[STAT]['R_total']*self.parameters[STAT]['K_R1R2']\
                                         + self.parameters[STAT]['Delta']**2)/self.parameters[STAT]['K_R1R2']**2))
            PR2active = self.parameters[STAT]['Jak2']*self.parameters[STAT]['K_Jak2'] / (1 + self.parameters[STAT]['Jak2']*self.parameters[STAT]['K_Jak2'])
            Rstar = cytokine_R*PR2active/(1+self.parameters[STAT]['K_ligand']/dose)
            response = self.parameters[STAT]['STAT_total']/(1+self.parameters[STAT]['K_STAT']*volPM/Rstar)
            STAT_response[STAT] = response
        return STAT_response

    def add_SOCS(self, cytokine_dose, SOCS_name):
        dose = cytokine_dose*1E-12*NA*volEC
        STAT_response = {}
        for STAT in self.STAT_names:
            cytokine_R = self.parameters[STAT]['K_R1R2']/2 * \
                         (1 + self.parameters[STAT]['R_total']/self.parameters[STAT]['K_R1R2'] -\
                            np.sqrt(1 + (2*self.parameters[STAT]['R_total']*self.parameters[STAT]['K_R1R2']\
                                         + self.parameters[STAT]['Delta']**2)/self.parameters[STAT]['K_R1R2']**2))
            Rstar = cytokine_R/(1+self.parameters[STAT]['K_ligand']/dose)
            response = self.parameters[STAT]['STAT_total']/(1+self.parameters[STAT]['K_STAT']*(1+self.parameters[STAT][SOCS_name]/self.parameters[STAT]['K_SOCS'])*volPM/Rstar)
            STAT_response[STAT] = response
        return STAT_response

    def add_both_Jaks(self, cytokine_dose):
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
            response = self.parameters[STAT]['STAT_total']/(1+self.parameters[STAT]['K_STAT']*volPM/Rstar)
            STAT_response[STAT] = response
        return STAT_response

<<<<<<< HEAD
    def receptor_only(self, cytokine_dose):
        dose = cytokine_dose*1E-12*NA*volEC
        STAT_response = {}
        for STAT in self.STAT_names:
            cytokine_R = self.parameters[STAT]['K_R1R2']/2 * \
                         (1 + self.parameters[STAT]['R_total']/self.parameters[STAT]['K_R1R2'] -\
                            np.sqrt(1 + (2*self.parameters[STAT]['R_total']*self.parameters[STAT]['K_R1R2']\
                                         + self.parameters[STAT]['Delta']**2)/self.parameters[STAT]['K_R1R2']**2))
            Rstar = cytokine_R/(1+self.parameters[STAT]['K_ligand']/dose)
            response = self.parameters[STAT]['STAT_total']/(1+self.parameters[STAT]['K_STAT']*volPM/Rstar)
            STAT_response[STAT] = response
        return STAT_response

=======
>>>>>>> 079c0630f326ba4879966fa18364d9eb994ffee5
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
IFNg_parameters = {'pSTAT1': {'R_total': 2000, # infer from Immgen
                              'Delta': 0,                           # infer from Immgen
                              'Jak1': 1000,                         # infer from Immgen
                              'Jak2': 1000,                         # infer from Immgen
                              'STAT_total': 2000,                   # infer from Immgen
                              default_SOCS_name: 200,               # infer from Immgen
                              'K_ligand': NA*volEC/(4*pi*0.5E10),   # from literature
                              'K_Jak1': 1E6/(NA*volCP),             # fit for each receptor
                              'K_Jak2': 1E6/(NA*volCP),             # fit for each receptor
                              'K_R1R2': 4*pi*0.5E-12/volPM,         # from literature
                              'K_STAT': 1000/volPM,           # fit for each receptor/STAT pair
                              'K_SOCS': 1000},                      # fit for each receptor

                   'pSTAT3': {'R_total': 2000,
                              'Delta': 0,
                              'Jak1': 1000,
                              'Jak2': 1000,
                              'STAT_total': 2000,
                              default_SOCS_name: 200,
                              'K_ligand': NA*volEC/(4*pi*0.5E10),
                              'K_Jak1': 1E6/(NA*volCP),
                              'K_Jak2': 1E6/(NA*volCP),
                              'K_R1R2': 4*pi*0.5E-12/volPM,
                              'K_STAT': 1000/volPM,
                              'K_SOCS': 1000}
                    }


def plot_dose_response(IFNg_parameters, SOCS_name, model=1):
    IFNg_receptor = Cytokine_Receptor(['pSTAT1', 'pSTAT3'], IFNg_parameters, 'IFNgamma')
    if model==1:
        pSTAT1_dose_response = [IFNg_receptor.add_both_Jaks(d)['pSTAT1'] for d in np.logspace(-2, 3)]
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
                                  K_STAT_STAT1=1000/volPM, K_STAT_STAT3=1000/volPM):
    cell_types = ImmGen_df['Cell_type'].values

    # Set input parameters for model
    default_parameters = IFNg_parameters.copy()
    # receptor parameters
    default_parameters['pSTAT1']['K_Jak1'] = K_Jak1
    default_parameters['pSTAT3']['K_Jak1'] = K_Jak1
    default_parameters['pSTAT1']['K_Jak2'] = K_Jak2
    default_parameters['pSTAT3']['K_Jak2'] = K_Jak2
    # receptor:STAT parameters
    default_parameters['pSTAT1']['K_STAT'] = K_STAT_STAT1
    default_parameters['pSTAT3']['K_STAT'] = K_STAT_STAT3

    response = {'Cell_type': [], 'pSTAT1': [], 'pSTAT3': []}
    for c in cell_types:
        IFNg_ImmGen_parameters = infer_protein(ImmGen_df, c, ['IFNGR1', 'IFNGR2', 'JAK1', 'JAK2', 'STAT1', 'STAT3'])
        for S in ['pSTAT1', 'pSTAT3']:
            default_parameters[S]['R_total'] = IFNg_ImmGen_parameters['IFNGR1'] + IFNg_ImmGen_parameters['IFNGR2']
            default_parameters[S]['Delta'] = IFNg_ImmGen_parameters['IFNGR1'] - IFNg_ImmGen_parameters['IFNGR2']
            default_parameters[S]['Jak1'] = IFNg_ImmGen_parameters['JAK1']
            default_parameters[S]['Jak2'] = IFNg_ImmGen_parameters['JAK2']
            default_parameters[S]['STAT_total'] = IFNg_ImmGen_parameters[S[1:]]
        # Make predictions
        IFNg_receptor = Cytokine_Receptor(['pSTAT1', 'pSTAT3'], IFNg_parameters, 'IFNgamma')
        q = IFNg_receptor.add_both_Jaks(dose)
        response['Cell_type'].append(c)
        response['pSTAT1'].append(q['pSTAT1'])
        response['pSTAT3'].append(q['pSTAT3'])
    return pd.DataFrame.from_dict(response)


def SOCS_competes_STAT_pSTAT1_and_pSTAT3(dose, SOCS_name, K_Jak1=1E6 / (NA * volCP), K_Jak2=1E6 / (NA * volCP),
                                         K_STAT_STAT1=1000/volPM, K_STAT_STAT3=1000/volPM, K_SOCS=1000, df=ImmGen_df):
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

    # receptor:STAT parameters
    default_parameters['pSTAT1']['K_STAT'] = K_STAT_STAT1
    default_parameters['pSTAT3']['K_STAT'] = K_STAT_STAT3

    response = {'Cell_type': [], 'pSTAT1': [], 'pSTAT3': []}
    for c in cell_types:
        IFNg_ImmGen_parameters = infer_protein(df, c, ['IFNGR1', 'IFNGR2', 'JAK1', 'JAK2', 'STAT1', 'STAT3', SOCS_name])
        for S in ['pSTAT1', 'pSTAT3']:
            default_parameters[S]['R_total'] = IFNg_ImmGen_parameters['IFNGR1'] + IFNg_ImmGen_parameters['IFNGR2']
            default_parameters[S]['Delta'] = IFNg_ImmGen_parameters['IFNGR1'] - IFNg_ImmGen_parameters['IFNGR2']
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
    def SOCS_model(dose, p1, p2, p3, p4, p5, scale_factor):
        y_pred = SOCS_competes_STAT_pSTAT1_and_pSTAT3(dose, SOCS_name, K_Jak1=p1, K_Jak2=p2, K_STAT_STAT1=p3,
                                                      K_STAT_STAT3=p4, K_SOCS=p5, df=df)[['pSTAT1', 'pSTAT3']]
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
    fit_pred = np.reshape(equilibrium_model(IFNg_dose, *pfit), (19, len(response_variable_names)))
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


<<<<<<< HEAD
def compare_model_errors(df=ImmGen_df, neg_feedback_name='SOCS2', add_factor='full'):
    # --------------
    # Receptor only
    # --------------
    # Fit
    def receptor_model_pSTAT1_and_pSTAT3(dose,
                                            K_STAT_STAT1=1000 / volPM, K_STAT_STAT3=1000 / volPM,
                                            df=ImmGen_df):
        cell_types = df['Cell_type'].values
=======
def compare_model_errors(df=ImmGen_df, neg_feedback_name='SOCS2'):
    # ------------
    # With SOCS
    # ------------
    pfit, pcov = fit_IFNg_with_SOCS_competes_STAT(neg_feedback_name, df)
>>>>>>> 079c0630f326ba4879966fa18364d9eb994ffee5

        # Set input parameters for model
        default_parameters = copy.deepcopy(IFNg_parameters)
        # receptor:STAT parameters
        default_parameters['pSTAT1']['K_STAT'] = K_STAT_STAT1
        default_parameters['pSTAT3']['K_STAT'] = K_STAT_STAT3

        response = {'Cell_type': [], 'pSTAT1': [], 'pSTAT3': []}
        for c in cell_types:
            IFNg_ImmGen_parameters = infer_protein(df, c, ['IFNGR1', 'IFNGR2', 'STAT1', 'STAT3'])
            for S in ['pSTAT1', 'pSTAT3']:
                default_parameters[S]['R_total'] = IFNg_ImmGen_parameters['IFNGR1'] + IFNg_ImmGen_parameters[
                    'IFNGR2']
                default_parameters[S]['Delta'] = IFNg_ImmGen_parameters['IFNGR1'] - IFNg_ImmGen_parameters['IFNGR2']
                default_parameters[S]['STAT_total'] = IFNg_ImmGen_parameters[S[1:]]
            # Make predictions
            IFNg_receptor = Cytokine_Receptor(['pSTAT1', 'pSTAT3'], default_parameters, 'IFNgamma')
            q = IFNg_receptor.receptor_only(dose)
            response['Cell_type'].append(c)
            response['pSTAT1'].append(q['pSTAT1'])
            response['pSTAT3'].append(q['pSTAT3'])
        return pd.DataFrame.from_dict(response)

    def model(dose, p3, p4, scale_factor):
        y_pred = receptor_model_pSTAT1_and_pSTAT3(dose, K_STAT_STAT1=p3,
                                                     K_STAT_STAT3=p4, df=df)[['pSTAT1', 'pSTAT3']]
        return np.divide(y_pred.values.flatten(), scale_factor)

    default_parameters = [10000 / volPM, 10000 / volPM, 1]
    y_true = ImmGen_df[['pSTAT1', 'pSTAT3']].values.flatten()
    pfit, pcov = curve_fit(model, IFNg_dose, y_true, p0=default_parameters,
                           bounds=(np.multiply(default_parameters, 0.1), np.multiply(default_parameters, 10)))
    # Predict
    baseline_fit_pred = np.reshape(model(IFNg_dose, *pfit), (df.shape[0], len(response_variable_names)))
    # Residual
<<<<<<< HEAD
    baseline_residuals = np.divide(np.subtract(baseline_fit_pred, df[['pSTAT1', 'pSTAT3']].values),
                          df[['pSTAT1', 'pSTAT3']].values)
=======
    SOCS_residuals = np.divide(np.subtract(SOCS_fit_pred, df[['pSTAT1', 'pSTAT3']].values), df[['pSTAT1', 'pSTAT3']].values)
    #SOCS_res_pred_labelled = [[df['Cell_type'].values[i], SOCS_residuals[i][0], SOCS_residuals[i][1]] for i in range(df.shape[0])]
>>>>>>> 079c0630f326ba4879966fa18364d9eb994ffee5

    # ------------
    # Add SOCS
    # ------------
    if add_factor=='SOCS':
        # Fit
        def add_SOCS_pSTAT1_and_pSTAT3(dose, SOCS_name,
                                        K_STAT_STAT1=1000 / volPM, K_STAT_STAT3=1000 / volPM, K_SOCS=1000,
                                        df=ImmGen_df):
            cell_types = df['Cell_type'].values

            # Set input parameters for model
            default_parameters = copy.deepcopy(IFNg_parameters)
            # receptor parameters
            default_parameters['pSTAT1']['K_SOCS'] = K_SOCS
            default_parameters['pSTAT3']['K_SOCS'] = K_SOCS

            # receptor:STAT parameters
            default_parameters['pSTAT1']['K_STAT'] = K_STAT_STAT1
            default_parameters['pSTAT3']['K_STAT'] = K_STAT_STAT3

            response = {'Cell_type': [], 'pSTAT1': [], 'pSTAT3': []}
            for c in cell_types:
                IFNg_ImmGen_parameters = infer_protein(df, c, ['IFNGR1', 'IFNGR2', 'STAT1', 'STAT3', SOCS_name])
                for S in ['pSTAT1', 'pSTAT3']:
                    default_parameters[S]['R_total'] = IFNg_ImmGen_parameters['IFNGR1'] + IFNg_ImmGen_parameters[
                        'IFNGR2']
                    default_parameters[S]['Delta'] = IFNg_ImmGen_parameters['IFNGR1'] - IFNg_ImmGen_parameters['IFNGR2']
                    default_parameters[S]['STAT_total'] = IFNg_ImmGen_parameters[S[1:]]
                    default_parameters[S][SOCS_name] = IFNg_ImmGen_parameters[SOCS_name]
                # Make predictions
                IFNg_receptor = Cytokine_Receptor(['pSTAT1', 'pSTAT3'], default_parameters, 'IFNgamma')
                q = IFNg_receptor.add_SOCS(dose, SOCS_name)
                response['Cell_type'].append(c)
                response['pSTAT1'].append(q['pSTAT1'])
                response['pSTAT3'].append(q['pSTAT3'])
            return pd.DataFrame.from_dict(response)

        def model(dose, p3, p4, p5, scale_factor):
            y_pred = add_SOCS_pSTAT1_and_pSTAT3(dose, neg_feedback_name, K_STAT_STAT1=p3,
                                                 K_STAT_STAT3=p4, K_SOCS=p5, df=df)[['pSTAT1', 'pSTAT3']]
            return np.divide(y_pred.values.flatten(), scale_factor)

        default_parameters = [10000 / volPM, 10000 / volPM, 70, 1]
        y_true = ImmGen_df[['pSTAT1', 'pSTAT3']].values.flatten()
        pfit, pcov = curve_fit(model, IFNg_dose, y_true, p0=default_parameters,
                               bounds=(np.multiply(default_parameters, 0.1), np.multiply(default_parameters, 10)))
        # Predict
        print(pfit)
        feature_fit_pred = np.reshape(model(IFNg_dose, *pfit),
                                      (df.shape[0], len(response_variable_names)))
        # Residual
        residuals = np.divide(np.subtract(feature_fit_pred, df[['pSTAT1', 'pSTAT3']].values),
                              df[['pSTAT1', 'pSTAT3']].values)

<<<<<<< HEAD
    # ------------
    # Add Jak1
    # ------------
    elif add_factor=='Jak1':
        # Fit
        def add_Jak1_pSTAT1_and_pSTAT3(dose, SOCS_name, K_Jak1=1E6 / (NA * volCP),
                                                 K_STAT_STAT1=1000 / volPM, K_STAT_STAT3=1000 / volPM, K_SOCS=1000,
                                                 df=ImmGen_df):
            cell_types = df['Cell_type'].values

            # Set input parameters for model
            default_parameters = copy.deepcopy(IFNg_parameters)
            # receptor parameters
            default_parameters['pSTAT1']['K_Jak1'] = K_Jak1
            default_parameters['pSTAT3']['K_Jak1'] = K_Jak1
            default_parameters['pSTAT1']['K_SOCS'] = K_SOCS
            default_parameters['pSTAT3']['K_SOCS'] = K_SOCS

            # receptor:STAT parameters
            default_parameters['pSTAT1']['K_STAT'] = K_STAT_STAT1
            default_parameters['pSTAT3']['K_STAT'] = K_STAT_STAT3

            response = {'Cell_type': [], 'pSTAT1': [], 'pSTAT3': []}
            for c in cell_types:
                IFNg_ImmGen_parameters = infer_protein(df, c, ['IFNGR1', 'IFNGR2', 'JAK1', 'JAK2', 'STAT1', 'STAT3',
                                                               SOCS_name])
                for S in ['pSTAT1', 'pSTAT3']:
                    default_parameters[S]['R_total'] = IFNg_ImmGen_parameters['IFNGR1'] + IFNg_ImmGen_parameters[
                        'IFNGR2']
                    default_parameters[S]['Delta'] = IFNg_ImmGen_parameters['IFNGR1'] - IFNg_ImmGen_parameters['IFNGR2']
                    default_parameters[S]['Jak1'] = IFNg_ImmGen_parameters['JAK1']
                    default_parameters[S]['STAT_total'] = IFNg_ImmGen_parameters[S[1:]]
                    default_parameters[S][SOCS_name] = IFNg_ImmGen_parameters[SOCS_name]
                # Make predictions
                IFNg_receptor = Cytokine_Receptor(['pSTAT1', 'pSTAT3'], default_parameters, 'IFNgamma')
                q = IFNg_receptor.add_Jak1(dose, SOCS_name)
                response['Cell_type'].append(c)
                response['pSTAT1'].append(q['pSTAT1'])
                response['pSTAT3'].append(q['pSTAT3'])
            return pd.DataFrame.from_dict(response)

        def model(dose, p2, p3, p4, p5, scale_factor):
            y_pred = add_Jak1_pSTAT1_and_pSTAT3(dose, neg_feedback_name, K_Jak1=p2, K_STAT_STAT1=p3,
                                                          K_STAT_STAT3=p4, K_SOCS=p5, df=df)[['pSTAT1', 'pSTAT3']]
            return np.divide(y_pred.values.flatten(), scale_factor)

        default_parameters = [1E7 / (NA * volCP), 10000 / volPM, 10000 / volPM, 70, 1]
        y_true = ImmGen_df[['pSTAT1', 'pSTAT3']].values.flatten()
        pfit, pcov = curve_fit(model, IFNg_dose, y_true, p0=default_parameters,
                               bounds=(np.multiply(default_parameters, 0.1), np.multiply(default_parameters, 10)))
        # Predict
        print(pfit)
        feature_fit_pred = np.reshape(model(IFNg_dose, *pfit),
                                      (df.shape[0], len(response_variable_names)))
        # Residual
        residuals = np.divide(np.subtract(feature_fit_pred, df[['pSTAT1', 'pSTAT3']].values),
                                          df[['pSTAT1', 'pSTAT3']].values)
    # ------------
    # Add Jak2
    # ------------
    elif add_factor == 'Jak2':
        # Fit
        def add_Jak2_pSTAT1_and_pSTAT3(dose, SOCS_name, K_Jak2=1E7 / (NA * volCP),
                                        K_STAT_STAT1=1000 / volPM, K_STAT_STAT3=1000 / volPM, K_SOCS=1000,
                                        df=ImmGen_df):
            cell_types = df['Cell_type'].values

            # Set input parameters for model
            default_parameters = copy.deepcopy(IFNg_parameters)
            # receptor parameters
            default_parameters['pSTAT1']['K_Jak2'] = K_Jak2
            default_parameters['pSTAT3']['K_Jak2'] = K_Jak2
            default_parameters['pSTAT1']['K_SOCS'] = K_SOCS
            default_parameters['pSTAT3']['K_SOCS'] = K_SOCS

            # receptor:STAT parameters
            default_parameters['pSTAT1']['K_STAT'] = K_STAT_STAT1
            default_parameters['pSTAT3']['K_STAT'] = K_STAT_STAT3

            response = {'Cell_type': [], 'pSTAT1': [], 'pSTAT3': []}
            for c in cell_types:
                IFNg_ImmGen_parameters = infer_protein(df, c, ['IFNGR1', 'IFNGR2', 'JAK1', 'JAK2', 'STAT1', 'STAT3',
                                                               SOCS_name])
                for S in ['pSTAT1', 'pSTAT3']:
                    default_parameters[S]['R_total'] = IFNg_ImmGen_parameters['IFNGR1'] + IFNg_ImmGen_parameters[
                        'IFNGR2']
                    default_parameters[S]['Delta'] = IFNg_ImmGen_parameters['IFNGR1'] - IFNg_ImmGen_parameters['IFNGR2']
                    default_parameters[S]['Jak1'] = IFNg_ImmGen_parameters['JAK1']
                    default_parameters[S]['Jak2'] = IFNg_ImmGen_parameters['JAK2']
                    default_parameters[S]['STAT_total'] = IFNg_ImmGen_parameters[S[1:]]
                    default_parameters[S][SOCS_name] = IFNg_ImmGen_parameters[SOCS_name]
                # Make predictions
                IFNg_receptor = Cytokine_Receptor(['pSTAT1', 'pSTAT3'], default_parameters, 'IFNgamma')
                q = IFNg_receptor.add_Jak2(dose)
                response['Cell_type'].append(c)
                response['pSTAT1'].append(q['pSTAT1'])
                response['pSTAT3'].append(q['pSTAT3'])
            return pd.DataFrame.from_dict(response)

        def model(dose, p2, p3, p4, p5, scale_factor):
            y_pred = add_Jak2_pSTAT1_and_pSTAT3(dose, neg_feedback_name, K_Jak2=p2, K_STAT_STAT1=p3,
                                                 K_STAT_STAT3=p4, K_SOCS=p5, df=df)[['pSTAT1', 'pSTAT3']]
            return np.divide(y_pred.values.flatten(), scale_factor)

        default_parameters = [1E7 / (NA * volCP), 10000 / volPM, 10000 / volPM, 70, 1]
        y_true = ImmGen_df[['pSTAT1', 'pSTAT3']].values.flatten()
        pfit, pcov = curve_fit(model, IFNg_dose, y_true, p0=default_parameters,
                               bounds=(np.multiply(default_parameters, 0.1), np.multiply(default_parameters, 10)))
        # Predict
        feature_fit_pred = np.reshape(model(IFNg_dose, *pfit), (df.shape[0], len(response_variable_names)))
        # Residual
        residuals = np.divide(np.subtract(feature_fit_pred, df[['pSTAT1', 'pSTAT3']].values),
                              df[['pSTAT1', 'pSTAT3']].values)
    # ---------------
    # Add both Jaks
    # ---------------
    elif add_factor == 'both_Jaks':
        # Fit
        def both_Jaks_pSTAT1_and_pSTAT3(dose, K_Jak1=1E7 / (NA * volCP), K_Jak2=1E7 / (NA * volCP),
                                        K_STAT_STAT1=1000 / volPM, K_STAT_STAT3=1000 / volPM,
                                        df=ImmGen_df):
            cell_types = df['Cell_type'].values

            # Set input parameters for model
            default_parameters = copy.deepcopy(IFNg_parameters)
            # receptor parameters
            default_parameters['pSTAT1']['K_Jak1'] = K_Jak1
            default_parameters['pSTAT3']['K_Jak1'] = K_Jak1
            default_parameters['pSTAT1']['K_Jak2'] = K_Jak2
            default_parameters['pSTAT3']['K_Jak2'] = K_Jak2

            # receptor:STAT parameters
            default_parameters['pSTAT1']['K_STAT'] = K_STAT_STAT1
            default_parameters['pSTAT3']['K_STAT'] = K_STAT_STAT3

            response = {'Cell_type': [], 'pSTAT1': [], 'pSTAT3': []}
            for c in cell_types:
                IFNg_ImmGen_parameters = infer_protein(df, c, ['IFNGR1', 'IFNGR2', 'JAK1', 'JAK2', 'STAT1', 'STAT3'])
                for S in ['pSTAT1', 'pSTAT3']:
                    default_parameters[S]['R_total'] = IFNg_ImmGen_parameters['IFNGR1'] + IFNg_ImmGen_parameters[
                        'IFNGR2']
                    default_parameters[S]['Delta'] = IFNg_ImmGen_parameters['IFNGR1'] - IFNg_ImmGen_parameters['IFNGR2']
                    default_parameters[S]['STAT_total'] = IFNg_ImmGen_parameters[S[1:]]
                    default_parameters[S]['Jak1'] = IFNg_ImmGen_parameters['JAK1']
                    default_parameters[S]['Jak2'] = IFNg_ImmGen_parameters['JAK2']
                # Make predictions
                IFNg_receptor = Cytokine_Receptor(['pSTAT1', 'pSTAT3'], default_parameters, 'IFNgamma')
                q = IFNg_receptor.add_both_Jaks(dose)
                response['Cell_type'].append(c)
                response['pSTAT1'].append(q['pSTAT1'])
                response['pSTAT3'].append(q['pSTAT3'])
            return pd.DataFrame.from_dict(response)

        def model(dose, p1, p2, p3, p4, scale_factor):
            y_pred = both_Jaks_pSTAT1_and_pSTAT3(dose, K_Jak1=p1, K_Jak2=p2, K_STAT_STAT1=p3,
                                                 K_STAT_STAT3=p4, df=df)[['pSTAT1', 'pSTAT3']]
            return np.divide(y_pred.values.flatten(), scale_factor)

        default_parameters = [1E7 / (NA * volCP), 1E7 / (NA * volCP), 10000 / volPM, 10000 / volPM, 1]
        y_true = ImmGen_df[['pSTAT1', 'pSTAT3']].values.flatten()
        pfit, pcov = curve_fit(model, IFNg_dose, y_true, p0=default_parameters,
                               bounds=(np.multiply(default_parameters, 0.1), np.multiply(default_parameters, 10)))
        # Predict
        feature_fit_pred = np.reshape(model(IFNg_dose, *pfit), (df.shape[0], len(response_variable_names)))
        # Residual
        residuals = np.divide(np.subtract(feature_fit_pred, df[['pSTAT1', 'pSTAT3']].values),
                              df[['pSTAT1', 'pSTAT3']].values)

    # -----------------------
    # Full model
    # -----------------------
    elif add_factor == 'full':
        # Fit
        def make_SOCS_model(SOCS_name, df):
            def SOCS_model(dose, p1, p2, p3, p4, p5, scale_factor):
                y_pred = SOCS_competes_STAT_pSTAT1_and_pSTAT3(dose, SOCS_name, K_Jak1=p1, K_Jak2=p2, K_STAT_STAT1=p3,
                                                              K_STAT_STAT3=p4, K_SOCS=p5, df=df)[['pSTAT1', 'pSTAT3']]
                return np.divide(y_pred.values.flatten(), scale_factor)
            return SOCS_model

        default_parameters = [1E7 / (NA * volCP), 1E8 / (NA * volCP), 1000 / volPM, 1000 / volPM, 70, 1]
        y_true = df[['pSTAT1', 'pSTAT3']].values.flatten()
        pfit, pcov = curve_fit(make_SOCS_model(neg_feedback_name, df), IFNg_dose, y_true, p0=default_parameters,
                               bounds=(np.multiply(default_parameters, 0.1), np.multiply(default_parameters, 10)))
        print(pfit)
        # Predict
        SOCS_model = make_SOCS_model(neg_feedback_name, df)
        feature_fit_pred = np.reshape(SOCS_model(IFNg_dose, *pfit), (df.shape[0], len(response_variable_names)))
        # Residual
        residuals = np.divide(np.subtract(feature_fit_pred, df[['pSTAT1', 'pSTAT3']].values),
                              df[['pSTAT1', 'pSTAT3']].values)
=======
    # Residual
    residuals = np.divide(np.subtract(fit_pred, df[['pSTAT1', 'pSTAT3']].values), df[['pSTAT1', 'pSTAT3']].values)
    #res_pred_labelled = [[df['Cell_type'].values[i], residuals[i][0], residuals[i][1]] for i in range(df.shape[0])]
>>>>>>> 079c0630f326ba4879966fa18364d9eb994ffee5

    # ------------
    # Plot
    # ------------
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 4.5))
    for i in [0, 1]:
        ax[i].set_title('pSTAT{}'.format(i*2 + 1))
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_xlabel('% error in no-SOCS model')
        ax[i].set_ylabel('% error in SOCS model')
<<<<<<< HEAD
        ax[i].scatter(np.abs(residuals[:, i]), np.abs(baseline_residuals[:, i]))
=======
        ax[i].scatter(np.abs(residuals[:, i]), np.abs(SOCS_residuals[:, i]))
>>>>>>> 079c0630f326ba4879966fa18364d9eb994ffee5
        ax[i].plot(np.logspace(np.log10(min(np.abs(residuals[:, i]))), np.log10(max(np.abs(residuals[:, i])))),
                   np.logspace(np.log10(min(np.abs(residuals[:, i]))), np.log10(max(np.abs(residuals[:, i])))), 'k--')
        ax[i].set_xlim((10**-3, 2*10**2))
        ax[i].set_ylim((10 ** -3, 2 * 10 ** 2))
    plt.tight_layout()
    plt.show()


def ec50_for_all_cell_types(SOCS_name, df=ImmGen_df, pfit=[]):
    # Get parameter fits for K_Jak1, K_Jak2, K_STAT_STAT1, K_STAT_STAT3, K_SOCS,' scale_factor
    if pfit == []:
        pfit, pcov = fit_IFNg_with_SOCS_competes_STAT(SOCS_name, df)

    cell_types = df['Cell_type'].values

    # Set input parameters for model
    default_parameters = IFNg_parameters.copy()
    # receptor parameters
    default_parameters['pSTAT1']['K_Jak1'] = pfit[0]
    default_parameters['pSTAT3']['K_Jak1'] = pfit[0]
    default_parameters['pSTAT1']['K_Jak2'] = pfit[1]
    default_parameters['pSTAT3']['K_Jak2'] = pfit[1]
    default_parameters['pSTAT1']['K_SOCS'] = pfit[4]
    default_parameters['pSTAT3']['K_SOCS'] = pfit[4]
    # receptor:STAT parameters
    default_parameters['pSTAT1']['K_STAT'] = pfit[2]
    default_parameters['pSTAT3']['K_STAT'] = pfit[3]

    record = {}
    for c in cell_types:
        IFNg_ImmGen_parameters = infer_protein(df, c, ['IFNGR1', 'IFNGR2', 'JAK1', 'JAK2', 'STAT1', 'STAT3', SOCS_name])
        for S in ['pSTAT1', 'pSTAT3']:
            default_parameters[S]['R_total'] = IFNg_ImmGen_parameters['IFNGR1'] + IFNg_ImmGen_parameters['IFNGR2']
            default_parameters[S]['Delta'] = IFNg_ImmGen_parameters['IFNGR1'] - IFNg_ImmGen_parameters['IFNGR2']
            default_parameters[S]['Jak1'] = IFNg_ImmGen_parameters['JAK1']
            default_parameters[S]['Jak2'] = IFNg_ImmGen_parameters['JAK2']
            default_parameters[S]['STAT_total'] = IFNg_ImmGen_parameters[S[1:]]
            default_parameters[S][SOCS_name] = IFNg_ImmGen_parameters[SOCS_name]
        # Make predictions
        IFNg_receptor = Cytokine_Receptor(['pSTAT1', 'pSTAT3'], default_parameters, 'IFNgamma')
        ec50 = IFNg_receptor.equilibrium_model_with_SOCS_ec50(SOCS_name)
        record[c] = {'pSTAT1': ec50['pSTAT1']/1E-12, 'pSTAT3': ec50['pSTAT3']/1E-12} # in pM
    return record


def max_pSTAT_for_all_cell_types(SOCS_name, df=ImmGen_df, pfit=[]):
    # Get parameter fits for K_Jak1, K_Jak2, K_STAT_STAT1, K_STAT_STAT3, K_SOCS, scale_factor
    if pfit == []:
        pfit, pcov = fit_IFNg_with_SOCS_competes_STAT(SOCS_name, df)

    cell_types = df['Cell_type'].values

    # Set input parameters for model
    default_parameters = IFNg_parameters.copy()
    # receptor parameters
    default_parameters['pSTAT1']['K_Jak1'] = pfit[0]
    default_parameters['pSTAT3']['K_Jak1'] = pfit[0]
    default_parameters['pSTAT1']['K_Jak2'] = pfit[1]
    default_parameters['pSTAT3']['K_Jak2'] = pfit[1]
    default_parameters['pSTAT1']['K_SOCS'] = pfit[4]
    default_parameters['pSTAT3']['K_SOCS'] = pfit[4]
    # receptor:STAT parameters
    default_parameters['pSTAT1']['K_STAT'] = pfit[2]
    default_parameters['pSTAT3']['K_STAT'] = pfit[3]

    record = {}
    for c in cell_types:
        IFNg_ImmGen_parameters = infer_protein(df, c, ['IFNGR1', 'IFNGR2', 'JAK1', 'JAK2', 'STAT1', 'STAT3', SOCS_name])
        for S in ['pSTAT1', 'pSTAT3']:
            default_parameters[S]['R_total'] = IFNg_ImmGen_parameters['IFNGR1'] + IFNg_ImmGen_parameters['IFNGR2']
            default_parameters[S]['Delta'] = IFNg_ImmGen_parameters['IFNGR1'] - IFNg_ImmGen_parameters['IFNGR2']
            default_parameters[S]['Jak1'] = IFNg_ImmGen_parameters['JAK1']
            default_parameters[S]['Jak2'] = IFNg_ImmGen_parameters['JAK2']
            default_parameters[S]['STAT_total'] = IFNg_ImmGen_parameters[S[1:]]
            default_parameters[S][SOCS_name] = IFNg_ImmGen_parameters[SOCS_name]
        # Make predictions
        IFNg_receptor = Cytokine_Receptor(['pSTAT1', 'pSTAT3'], default_parameters, 'IFNgamma')
        maxpSTAT = IFNg_receptor.equilibrium_model_with_SOCS_pSTATmax(SOCS_name)
        record[c] = {'pSTAT1': maxpSTAT['pSTAT1']/pfit[5], 'pSTAT3': maxpSTAT['pSTAT3']/pfit[5]} # in number of molecules, scaled to match CyTOF
    return record


def make_ec50_predictions_plot(parameters=[]):
    if parameters == []:
        ec50_predictions = ec50_for_all_cell_types('SOCS2')
        maxpSTAT_predictions = max_pSTAT_for_all_cell_types('SOCS2')
    else:
        ec50_predictions = ec50_for_all_cell_types('SOCS2', pfit=parameters)
        maxpSTAT_predictions = max_pSTAT_for_all_cell_types('SOCS2', pfit=parameters)

    SOCS_CyTOF = {c: infer_protein(ImmGen_df, c, ['SOCS2'])['SOCS2'] for c in ImmGen_df['Cell_type'].values}
    pSTAT1_CyTOF = {c: infer_protein(ImmGen_df, c, ['pSTAT1'])['pSTAT1'] for c in ImmGen_df['Cell_type'].values}
    pSTAT3_CyTOF = {c: infer_protein(ImmGen_df, c, ['pSTAT3'])['pSTAT3'] for c in ImmGen_df['Cell_type'].values}

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0][0].set_ylabel('EC50 (pM)')
    ax[1][0].set_ylabel('Max pSTAT (# molecules)')
    ax[1][1].set_xlabel('SOCS expression (# transcripts)')
    ax[1][0].set_xlabel('SOCS expression (# transcripts)')
    ax[0][0].title.set_text('pSTAT1')
    ax[0][1].title.set_text('pSTAT3')

    ax[0][0].scatter([SOCS_CyTOF[c] for c in SOCS_CyTOF.keys()], [ec50_predictions[c]['pSTAT1'] for c in SOCS_CyTOF.keys()], label='prediction')
    ax[0][1].scatter([SOCS_CyTOF[c] for c in SOCS_CyTOF.keys()], [ec50_predictions[c]['pSTAT3'] for c in SOCS_CyTOF.keys()], label='prediction')

    ax[1][0].scatter([SOCS_CyTOF[c] for c in SOCS_CyTOF.keys()], [maxpSTAT_predictions[c]['pSTAT1'] for c in SOCS_CyTOF.keys()], label='prediction')
    ax[1][1].scatter([SOCS_CyTOF[c] for c in SOCS_CyTOF.keys()], [maxpSTAT_predictions[c]['pSTAT3'] for c in SOCS_CyTOF.keys()], label='prediction')
    ax[1][0].scatter([SOCS_CyTOF[c] for c in SOCS_CyTOF.keys()], [pSTAT1_CyTOF[c] for c in SOCS_CyTOF.keys()], label='CyTOF')
    ax[1][1].scatter([SOCS_CyTOF[c] for c in SOCS_CyTOF.keys()], [pSTAT3_CyTOF[c] for c in SOCS_CyTOF.keys()], label='CyTOF')
    plt.legend()
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

        cell_types = ['Granulocytes','Ly6Chi_Mo','Ly6Clo_Mo','preB_FrC','preB_FrD','MPP34F','ST34F','CMP','GMP','MEP','CDP','MDP','LT-HSC','ST-HSC','Mac_Sp','CD11b+DC','CD11b-DC','NK','Mac_BM']
        base_parameters = {'pSTAT1': {'R_total': 2000, # infer from Immgen
                              'Delta': 0,                           # infer from Immgen
                              'Jak1': 1000,                         # infer from Immgen
                              'Jak2': 1000,                         # infer from Immgen
                              'STAT_total': 2000,                   # infer from Immgen
                              SOCS_name: 200,                       # infer from Immgen
                              'K_ligand': NA*volEC/(4*pi*0.5E10),   # from literature
                              'K_Jak1': 1E6/(NA*volCP),             # fit for each receptor
                              'K_Jak2': 1E6/(NA*volCP),             # fit for each receptor
                              'K_R1R2': 4*pi*0.5E-12/volPM,         # from literature
                              'K_STAT': 1000/volPM,           # fit for each receptor/STAT pair
                              'K_SOCS': 1000},                      # fit for each receptor

                   'pSTAT3': {'R_total': 2000,
                              'Delta': 0,
                              'Jak1': 1000,
                              'Jak2': 1000,
                              'STAT_total': 2000,
                               SOCS_name: 200,
                              'K_ligand': NA*volEC/(4*pi*0.5E10),
                              'K_Jak1': 1E6/(NA*volCP),
                              'K_Jak2': 1E6/(NA*volCP),
                              'K_R1R2': 4*pi*0.5E-12/volPM,
                              'K_STAT': 1000/volPM,
                              'K_SOCS': 1000}
                    }
        output_names = ['pSTAT1', 'pSTAT3']
        protein_names = ['IFNGR1', 'IFNGR2', 'JAK1', 'JAK2', 'STAT1', 'STAT3', SOCS_name]
        dose = 100 * 1E-12 * NA * volEC
        for STAT in output_names:
            base_parameters[STAT]['K_Jak1'] = parameters[0]
            base_parameters[STAT]['K_Jak2'] = parameters[1]
            base_parameters[STAT]['K_SOCS'] = parameters[4]
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
                base_parameters[S]['R_total'] = IFNg_ImmGen_parameters['IFNGR1'] + IFNg_ImmGen_parameters['IFNGR2']
                base_parameters[S]['Delta'] = IFNg_ImmGen_parameters['IFNGR1'] - IFNg_ImmGen_parameters['IFNGR2']
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
        data = [[40.6,	18.5],
                [38.23,	20],
                [9.28,	3.75],
                [4.98,	2.66],
                [4.83,	2.48],
                [17.63,	6.18],
                [12.37,	5.06],
                [15.21,	5.72],
                [16.29,	7.81],
                [10.69,	5.14],
                [5.99,	3.99],
                [18.12,	5.41],
                [6.08,	2.38],
                [15.28,	3.5],
                [20.87,	0.23],
                [0.48,	0.21],
                [1.49,	0.52],
                [0.24,	0.46],
                [22.4,	7.54]]
        like_ctot = norm(loc=data, scale=np.ones(np.shape(data)))
        logp_ctotal = np.sum(like_ctot.logpdf(fit_pred))

        # If model simulation failed due to integrator errors, return a log probability of -inf.
        if np.isnan(logp_ctotal):
            logp_ctotal = -np.inf
        return logp_ctotal


    #min_response_row = df.loc[df[response_variable_names[0]].idxmin()]
    #min_response_SOCS_expression = infer_protein(df, min_response_row.loc['Cell_type'], [SOCS_name])[SOCS_name]
    # old: [1E7 / (NA * volCP), 1E6 / (NA * volCP), 900 / volPM, 1000 / volPM, min_response_SOCS_expression, 1]
    #       K_Jak1, K_Jak2, K_STAT_STAT1, K_STAT_STAT3, K_SOCS, scale_factor
    prior = [3.17147014e-03,   3.17147014e+00,   3.18309889e+11,   1.00760330e+12,   1.67829414e-02, 1e-01]
    prior = dict(zip(['K_Jak1', 'K_Jak2', 'K_STAT_STAT1', 'K_STAT_STAT3', 'K_SOCS', 'scale_factor'], prior))

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
        pred = np.reshape(SOCS_model(IFNg_dose, *converted_samples[p]), (19, len(response_variable_names)))
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
        log_pred = np.log(np.reshape(SOCS_model(IFNg_dose, *converted_samples[p]), (19, len(response_variable_names))))
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
        for p in range(0, len(converted_samples), 1):
            log_pred = np.log(np.reshape(SOCS_model(IFNg_dose, *converted_samples[p]), (19, len(response_variable_names))))
            square_residual = np.sum(np.square(np.subtract(log_pred, np.log(df[['pSTAT1', 'pSTAT3']].values))))
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
    residuals = np.subtract(np.log(fit_pred), np.log(ImmGen_df[['pSTAT1', 'pSTAT3']].values))
    ss_res = np.sum(np.square(residuals))
    ss_tot = np.sum(np.square((np.log(ImmGen_df[['pSTAT1', 'pSTAT3']].values) - np.log(np.mean(ImmGen_df[['pSTAT1', 'pSTAT3']].values)))))
    r_squared = 1 - (ss_res / ss_tot)
    print("r2 (linear):", 1 - ( np.sum(np.square(np.subtract(fit_pred, ImmGen_df[['pSTAT1', 'pSTAT3']].values))) / np.sum(np.square((ImmGen_df[['pSTAT1', 'pSTAT3']].values - np.mean(ImmGen_df[['pSTAT1', 'pSTAT3']].values))))))

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

def compare_model_to_ImmGen_normalized(pset, SOCS_name='SOCS2'):
    # Predict
    SOCS_model = make_SOCS_competes_STAT_model(SOCS_name)
    fit_pred = np.reshape(SOCS_model(IFNg_dose, *pset), (ImmGen_df.shape[0], len(response_variable_names)))
    fit_pred_labelled = [[ImmGen_df['Cell_type'].values[i], fit_pred[i][0], fit_pred[i][1]] for i in
                         range(ImmGen_df.shape[0])]

    # R-squared
    residuals = np.subtract(np.log(fit_pred), np.log(ImmGen_df[['pSTAT1', 'pSTAT3']].values))
    ss_res = np.sum(np.square(residuals))
    ss_tot = np.sum(np.square((np.log(ImmGen_df[['pSTAT1', 'pSTAT3']].values) - np.log(np.mean(ImmGen_df[['pSTAT1', 'pSTAT3']].values)))))
    r_squared = 1 - (ss_res / ss_tot)

    # Plot
    fit_prediction = pd.DataFrame.from_records(fit_pred_labelled, columns=['Cell_type', 'pSTAT1', 'pSTAT3'])
    fit_prediction.insert(0, 'Class', ['Model' for _ in range(ImmGen_df.shape[0])])
    measured_response = ImmGen_df[['Cell_type', 'pSTAT1', 'pSTAT3']]
    measured_response.insert(0, 'Class', ['CyTOF' for _ in range(ImmGen_df.shape[0])])

    measured_response['pSTAT1norm'] = measured_response['pSTAT1']/ImmGen_df['STAT1']
    measured_response['pSTAT3norm'] = measured_response['pSTAT3'] / ImmGen_df['STAT3']
    fit_prediction['pSTAT1norm'] = fit_prediction['pSTAT1']/ImmGen_df['STAT1']
    fit_prediction['pSTAT3norm'] = fit_prediction['pSTAT3'] / ImmGen_df['STAT3']
    df = pd.concat([fit_prediction, measured_response])

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 5))
    fig.suptitle('R-squared (log scale) = {:.2f}'.format(r_squared))
    subplot_titles = ['pSTAT1', 'pSTAT3']

    sns.barplot(x="Cell_type", y="pSTAT1norm", data=df, hue='Class', ax=ax[0])
    sns.barplot(x="Cell_type", y="pSTAT3norm", data=df, hue='Class', ax=ax[1])
    for i in [0, 1]:
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=90)
        ax[i].set_title(subplot_titles[i])
    plt.tight_layout()
    plt.show()
<<<<<<< HEAD

=======
>>>>>>> 079c0630f326ba4879966fa18364d9eb994ffee5

if __name__ == "__main__":
    # Check variance in predictors
    #df = ImmGen_df[response_variable_names + predictor_variable_names]
    #print(pd.Series(np.diag(df.cov().values), index=df.columns))

    #pairwise_correlation()
    #SOCS_histograms()

    #fit_without_SOCS()
    #fit_with_SOCS_competes_STAT(neg_feedback_name='allSOCS')

<<<<<<< HEAD
    #compare_model_errors(add_factor='both_Jaks')
=======
    #compare_model_errors()
>>>>>>> 079c0630f326ba4879966fa18364d9eb994ffee5

    #print(ec50_for_all_cell_types('SOCS2'))
    #make_ec50_predictions_plot()

    #fit_IFNg_SOCS_competes_STAT_with_DREAM('SOCS2')

    save_dir = "PyDREAM_08-11-2019_10000"
    sim_name = "SOCS2"
    #sample_DREAM_IFNg_SOCS_competes_STAT(os.path.join(save_dir, sim_name+'_samples' + '.npy'), sim_name, step_size=250, find_map=False)

    ## ['K_Jak1', 'K_Jak2', 'K_STAT_STAT1', 'K_STAT_STAT3', 'K_SOCS', 'scale_factor']
    #min_response_row = ImmGen_df.loc[ImmGen_df[response_variable_names[0]].idxmin()]
    #min_response_SOCS_expression = infer_protein(ImmGen_df, min_response_row.loc['Cell_type'], [sim_name])[sim_name]
    # [1E7 / (NA * volCP), 1E6 / (NA * volCP), 900 / volPM, 1000 / volPM, 0.006*min_response_SOCS_expression, 0.5]
    #p_prior = [3.171470138e-02, 3.17147013799e-03, 2.864788975654e+12, 3.183098861837e+12, 5.77815e-01, 5e-01]
    p_best = [3.17147014e-03,   3.17147014e+00,   3.18309889e+11,   1.00760330e+12,   1.67829414e-02, 1e-01]
<<<<<<< HEAD
    p_log_fit = [3.17147014e-01,   3.17147014e+00,   3.18309886e+11,   3.18309886e+11, 1.60000000e-01, 2.00000000e0] # does pretty poorly
    p_best_by_MCMC = [2.38069129e-03, 6.28888514e-01, 3.93783943e+02, 1.27796666e+03, 1.43456465e-09, 5.76281582e+00]
    compare_model_to_ImmGen(p_best)
    #compare_model_to_ImmGen_normalized(p_best)

=======
    p_best_by_MCMC = [2.38069129e-03, 6.28888514e-01, 3.93783943e+02, 1.27796666e+03, 1.43456465e-09, 5.76281582e+00]
    #compare_model_to_ImmGen(p_best_by_MCMC)
    compare_model_to_ImmGen_normalized(p_best)
>>>>>>> 079c0630f326ba4879966fa18364d9eb994ffee5

