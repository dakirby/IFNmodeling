import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from scipy.optimize import curve_fit

# Import data set and split into predictor variables (receptors, JAKs, SOCS, etc.) and response variables (STATs)
IFNg_dose = 100 # pM
ImmGen_df = pd.read_excel('ImmGen_signaling_with_protein_response.xlsx', sheet_name='Sheet1', axis=1)

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
        keys: R_total, Delta, Jak1, Jak2, STAT_total, K_ligand, K_Jak1, K_Jak2, K_R1R2, K_M, K_um
    cytokine_name (str): the name of the cytokine which binds specifically to this receptor 

Methods: 
    def equilibrium_model_output():
        :param cytokine_dose: (float) the stimulation of cytokine in pM
        :return STAT_response: (dict) the predicted pSTAT response for each key, STAT, in self.STAT_names
    def equilibrium_model_with_SOCS_output():
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
                         (1 + self.parameters[STAT]['R_total']/self.parameters[STAT]['K_R1R2'] +\
                            np.sqrt(1 + (2*self.parameters[STAT]['R_total']*self.parameters[STAT]['K_R1R2']\
                                         + self.parameters[STAT]['Delta']**2)/self.parameters[STAT]['K_R1R2']**2))
            Rstar = cytokine_R*self.parameters[STAT]['Jak1']*self.parameters[STAT]['K_Jak1']*self.parameters[STAT]['Jak2']*\
                    self.parameters[STAT]['K_Jak2']/(1+self.parameters[STAT]['K_ligand']/dose)
            response = self.parameters[STAT]['STAT_total']/(1+self.parameters[STAT]['K_um']*self.parameters[STAT]['K_M']*volPM/Rstar)
            STAT_response[STAT] = response
        return STAT_response

    def equilibrium_model_with_SOCS_output(self, cytokine_dose, SOCS_name):
        dose = cytokine_dose*1E-12*NA*volEC
        STAT_response = {}
        for STAT in self.STAT_names:
            cytokine_R = self.parameters[STAT]['K_R1R2']/2 * \
                         (1 + self.parameters[STAT]['R_total']/self.parameters[STAT]['K_R1R2'] +\
                            np.sqrt(1 + (2*self.parameters[STAT]['R_total']*self.parameters[STAT]['K_R1R2']\
                                         + self.parameters[STAT]['Delta']**2)/self.parameters[STAT]['K_R1R2']**2))
            PR1active = self.parameters[STAT]['Jak1']*self.parameters[STAT]['K_Jak1']*\
                    (1-self.parameters[STAT][SOCS_name]*self.parameters[STAT]['K_S'])*\
                        np.heaviside(1/self.parameters[STAT]['K_S']-self.parameters[STAT][SOCS_name], 1)
            PR2active = self.parameters[STAT]['Jak2']*self.parameters[STAT]['K_Jak2']
            Rstar = cytokine_R*PR1active*PR2active/(1+self.parameters[STAT]['K_ligand']/dose)
            response = self.parameters[STAT]['STAT_total']/(1+self.parameters[STAT]['K_um']*self.parameters[STAT]['K_M']*volPM/Rstar)
            STAT_response[STAT] = response
        return STAT_response

    def equilibrium_model_ec50(self, SOCS_name):
        ec50_dict = {}
        for STAT in self.STAT_names:
            term1 = volPM * self.parameters[STAT]['K_M'] * self.parameters[STAT]['K_um']
            num = 2 * term1 * self.parameters[STAT]['K_ligand'] * (1/self.parameters[STAT]['K_S']-self.parameters[STAT][SOCS_name] > 0)
            ec50 = num / (4 * term1 - (2 * term1 + self.parameters[STAT]['Jak1'] * self.parameters[STAT]['K_Jak1'] * self.parameters[STAT]['K_Jak2'] * self.parameters[STAT]['Jak2'] * \
                                       (-1 + self.parameters[STAT]['K_S'] * self.parameters[STAT][SOCS_name]) *\
                                       (self.parameters[STAT]['K_R1R2'] + self.parameters[STAT]['R_total'] + self.parameters[STAT]['K_R1R2']*\
                                        np.sqrt((self.parameters[STAT]['K_R1R2']**2 + 2*self.parameters[STAT]['K_R1R2']*self.parameters[STAT]['R_total'] + self.parameters[STAT]['Delta']**2)/\
                                                self.parameters[STAT]['K_R1R2']**2))))
            ec50_dict[STAT] = ec50/(NA*volEC) # (M)
        return ec50_dict

    def equilibrium_model_pSTATmax(self, SOCS_name):
        pSTAT_max_dict = {}
        for STAT in self.STAT_names:
            heaviside_term = (1/self.parameters[STAT]['K_S']-self.parameters[STAT][SOCS_name] > 0)
            sqrt_term = np.sqrt(1 + (2*self.parameters[STAT]['K_R1R2']*self.parameters[STAT]['R_total'] + self.parameters[STAT]['Delta']**2)/self.parameters[STAT]['K_R1R2']**2)
            term1 = 1 + self.parameters[STAT]['R_total']/self.parameters[STAT]['K_R1R2'] + sqrt_term
            term2 = self.parameters[STAT]['Jak1']*self.parameters[STAT]['Jak2']*self.parameters[STAT]['K_Jak1']*self.parameters[STAT]['K_Jak2']
            Rt = 2*volPM*self.parameters[STAT]['K_M']*self.parameters[STAT]['K_um']/(term2*self.parameters[STAT]['K_R1R2']*(1-self.parameters[STAT]['K_S']*self.parameters[STAT][SOCS_name])*term1*heaviside_term)
            maxpSTAT = self.parameters[STAT]['STAT_total']/(1+Rt)
            pSTAT_max_dict[STAT] = maxpSTAT
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
                              'K_M': 1000/volPM,                    # fit for each receptor/STAT pair
                              'K_um': 1,                            # fit for each receptor/STAT pair
                              'K_S': 1/430.},                       # fit for each receptor

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
                              'K_M': 1000/volPM,
                              'K_um': 1,
                              'K_S': 1/430.}
                    }


def plot_dose_response(IFNg_parameters, SOCS_name, model=1):
    IFNg_receptor = Cytokine_Receptor(['pSTAT1', 'pSTAT3'], IFNg_parameters, 'IFNgamma')
    if model==1:
        pSTAT1_dose_response = [IFNg_receptor.equilibrium_model_output(d, SOCS_name)['pSTAT1'] for d in np.logspace(-2, 3)]
    elif model==2:
        pSTAT1_dose_response = [IFNg_receptor.equilibrium_model_with_SOCS_output(d, SOCS_name)['pSTAT1'] for d in np.logspace(-2, 3)]
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
        all_SOCS_names = ['SHP1', 'SHP2', 'PIAS1', 'PIAS2', 'PIAS3', 'PIAS4']
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
                                  K_M_STAT1=1000/volPM, K_M_STAT3=1000/volPM, K_um_STAT1=1, K_um_STAT3=1):
    cell_types = ImmGen_df['Cell_type'].values

    # Set input parameters for model
    default_parameters = IFNg_parameters.copy()
    # receptor parameters
    default_parameters['pSTAT1']['K_Jak1'] = K_Jak1
    default_parameters['pSTAT3']['K_Jak1'] = K_Jak1
    default_parameters['pSTAT1']['K_Jak2'] = K_Jak2
    default_parameters['pSTAT3']['K_Jak2'] = K_Jak2
    # receptor:STAT parameters
    default_parameters['pSTAT1']['K_M'] = K_M_STAT1
    default_parameters['pSTAT3']['K_M'] = K_M_STAT3
    default_parameters['pSTAT1']['K_um'] = K_um_STAT1
    default_parameters['pSTAT3']['K_um'] = K_um_STAT3

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
        q = IFNg_receptor.equilibrium_model_output(dose)
        response['Cell_type'].append(c)
        response['pSTAT1'].append(q['pSTAT1'])
        response['pSTAT3'].append(q['pSTAT3'])
    return pd.DataFrame.from_dict(response)


def SOCS_pSTAT1_and_pSTAT3(dose, SOCS_name, K_Jak1=1E6/(NA*volCP), K_Jak2=1E6/(NA*volCP),
                              K_M_STAT1=1000/volPM, K_M_STAT3=1000/volPM, K_um_STAT1=1, K_um_STAT3=1, K_S=1/300,
                           df=ImmGen_df):
    cell_types = df['Cell_type'].values

    # Set input parameters for model
    default_parameters = IFNg_parameters.copy()
    # receptor parameters
    default_parameters['pSTAT1']['K_Jak1'] = K_Jak1
    default_parameters['pSTAT3']['K_Jak1'] = K_Jak1
    default_parameters['pSTAT1']['K_Jak2'] = K_Jak2
    default_parameters['pSTAT3']['K_Jak2'] = K_Jak2
    default_parameters['pSTAT1']['K_S'] = K_S
    default_parameters['pSTAT3']['K_S'] = K_S
    # receptor:STAT parameters
    default_parameters['pSTAT1']['K_M'] = K_M_STAT1
    default_parameters['pSTAT3']['K_M'] = K_M_STAT3
    default_parameters['pSTAT1']['K_um'] = K_um_STAT1
    default_parameters['pSTAT3']['K_um'] = K_um_STAT3

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
        IFNg_receptor = Cytokine_Receptor(['pSTAT1', 'pSTAT3'], IFNg_parameters, 'IFNgamma')
        q = IFNg_receptor.equilibrium_model_with_SOCS_output(dose, SOCS_name)
        response['Cell_type'].append(c)
        response['pSTAT1'].append(q['pSTAT1'])
        response['pSTAT3'].append(q['pSTAT3'])
    return pd.DataFrame.from_dict(response)


def equilibrium_model(dose, p1, p2, p3, p4, p5, p6, scale_factor):
    y_pred = equilibrium_pSTAT1_and_pSTAT3(dose, K_Jak1=p1, K_Jak2=p2, K_M_STAT1=p3, K_M_STAT3=p4,
                                           K_um_STAT1=p5, K_um_STAT3=p6)[['pSTAT1', 'pSTAT3']]
    return np.divide(y_pred.values.flatten(), scale_factor)


def fit_IFNg_equilibrium():
    default_parameters = [1E6 / (NA * volCP), 1E6 / (NA * volCP), 1000 / volPM, 1000 / volPM, 1, 1, 50]
    y_true = ImmGen_df[['pSTAT1', 'pSTAT3']].values.flatten()
    pfit, pcov = curve_fit(equilibrium_model, IFNg_dose, y_true, p0=default_parameters,
                           bounds=(np.multiply(default_parameters, 0.1), np.multiply(default_parameters, 10)))
    return pfit, pcov


def make_SOCS_model(SOCS_name, df=ImmGen_df):
    def SOCS_model(dose, p1, p2, p3, p4, p5, p6, p7, scale_factor):
        y_pred = SOCS_pSTAT1_and_pSTAT3(dose, SOCS_name, K_Jak1=p1, K_Jak2=p2, K_M_STAT1=p3, K_M_STAT3=p4,
                                               K_um_STAT1=p5, K_um_STAT3=p6, K_S=p7, df=df)[['pSTAT1', 'pSTAT3']]
        return np.divide(y_pred.values.flatten(), scale_factor)
    return SOCS_model

def fit_IFNg_with_SOCS(SOCS_name, df=ImmGen_df):
    min_response_row = df.loc[df[response_variable_names[0]].idxmin()]
    min_response_SOCS_expression = infer_protein(df, min_response_row.loc['Cell_type'], [SOCS_name])[SOCS_name]
    default_parameters = [1E6 / (NA * volCP), 1E6 / (NA * volCP), 1000 / volPM, 1000 / volPM, 1, 1, 1/min_response_SOCS_expression, 15]
    y_true = df[['pSTAT1', 'pSTAT3']].values.flatten()
    pfit, pcov = curve_fit(make_SOCS_model(SOCS_name, df), IFNg_dose, y_true, p0=default_parameters,
                           bounds=(np.multiply(default_parameters, 0.1), np.multiply(default_parameters, 10)))
    return pfit, pcov


def k_fold_cross_validate_SOCS_model(k=10, df=ImmGen_df, neg_feedback_name='SOCS2'):
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
        pfit, pcov = fit_IFNg_with_SOCS(neg_feedback_name, df=train)

        # Predict
        SOCS_model = make_SOCS_model(neg_feedback_name, df=test)
        fit_pred = np.reshape(SOCS_model(IFNg_dose, *pfit), (test.shape[0], len(response_variable_names)))

        # Compute R**2 value
        residuals = np.subtract(fit_pred, test[['pSTAT1', 'pSTAT3']].values)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((test[['pSTAT1', 'pSTAT3']].values-np.mean(test[['pSTAT1', 'pSTAT3']].values))**2)
        r_squared = 1 - (ss_res / ss_tot)
        r2_samples.append(r_squared)

    return np.mean(r2_samples), np.var(r2_samples)

# Fit
def fit_without_SOCS(df=ImmGen_df):
    pfit, pcov = fit_IFNg_equilibrium()
    print(pfit)

    # Predict
    fit_pred = np.reshape(equilibrium_model(IFNg_dose, *pfit), (19, len(response_variable_names)))
    fit_pred_labelled = [[df['Cell_type'].values[i], fit_pred[i][0], fit_pred[i][1]] for i in range(df.shape[0])]

    # Compute R**2 value
    residuals = np.subtract(fit_pred, df[['pSTAT1', 'pSTAT3']].values)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((df[['pSTAT1', 'pSTAT3']].values - np.mean(df[['pSTAT1', 'pSTAT3']].values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print("r-squared value: ", r_squared)

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

    plt.suptitle(r"$R^{2}$ = " + "{:.2f}".format(r_squared))
    plt.tight_layout()
    plt.show()


def fit_with_SOCS(df=ImmGen_df, k_fold=1, neg_feedback_name='SOCS2'):
    pfit, pcov = fit_IFNg_with_SOCS(neg_feedback_name, df)
    print(pfit)

    # Predict
    SOCS_model = make_SOCS_model(neg_feedback_name)
    fit_pred = np.reshape(SOCS_model(IFNg_dose, *pfit), (df.shape[0], len(response_variable_names)))
    fit_pred_labelled = [[df['Cell_type'].values[i], fit_pred[i][0], fit_pred[i][1]] for i in range(df.shape[0])]

    # Compute R**2 value
    if k_fold==1:
        residuals = np.subtract(fit_pred, df[['pSTAT1', 'pSTAT3']].values)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((df[['pSTAT1', 'pSTAT3']].values-np.mean(df[['pSTAT1', 'pSTAT3']].values))**2)
        r_squared = 1 - (ss_res / ss_tot)
        print("r-squared value: ", r_squared)
    else:
        r_squared, r2_var = k_fold_cross_validate_SOCS_model(k=k_fold)
        print("k-fold cross validation of R2 value is estimated to be {:.2f} +/- {:.2f}".format(r_squared, np.sqrt(r2_var)))

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
        plt.suptitle(r"$R^{2}$ = "+ "{:.2f}".format(r_squared))
    else:
        plt.suptitle(r"$R^{2}$ = " + "{:.2f}".format(r_squared) + " ({}-fold cross-validation)".format(k_fold))
    plt.tight_layout()
    plt.show()


def compare_model_errors(df=ImmGen_df):
    # ------------
    # With SOCS
    # ------------
    neg_feedback_name = 'SOCS2'
    pfit, pcov = fit_IFNg_with_SOCS(neg_feedback_name, df)

    # Predict
    SOCS_model = make_SOCS_model(neg_feedback_name)
    SOCS_fit_pred = np.reshape(SOCS_model(IFNg_dose, *pfit), (df.shape[0], len(response_variable_names)))

    # Residual
    SOCS_residuals = np.subtract(SOCS_fit_pred, df[['pSTAT1', 'pSTAT3']].values)
    #SOCS_res_pred_labelled = [[df['Cell_type'].values[i], SOCS_residuals[i][0], SOCS_residuals[i][1]] for i in range(df.shape[0])]

    # ------------
    # Without SOCS
    # ------------
    pfit, pcov = fit_IFNg_equilibrium()

    # Predict
    fit_pred = np.reshape(equilibrium_model(IFNg_dose, *pfit), (df.shape[0], len(response_variable_names)))

    # Residual
    residuals = np.subtract(fit_pred, df[['pSTAT1', 'pSTAT3']].values)
    #res_pred_labelled = [[df['Cell_type'].values[i], residuals[i][0], residuals[i][1]] for i in range(df.shape[0])]

    # ------------
    # Plot
    # ------------
    fig, ax = plt.subplots(nrows=1, ncols=2)
    for i in [0, 1]:
        ax[i].set_title('pSTAT{}'.format(i*2 + 1))
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_xlabel(r'abs($\delta$) in no-SOCS model')
        ax[i].set_ylabel(r'abs($\delta$) in SOCS model')
        ax[i].scatter(np.abs(residuals[:, i]), np.abs(SOCS_residuals[:, i]))
        ax[i].plot(np.logspace(np.log10(min(np.abs(residuals[:, i]))), np.log10(max(np.abs(residuals[:, i])))),
                   np.logspace(np.log10(min(np.abs(residuals[:, i]))), np.log10(max(np.abs(residuals[:, i])))), 'k--')
    plt.tight_layout()
    plt.show()


def ec50_for_all_cell_types(SOCS_name, df=ImmGen_df):
    # Get parameter fits for K_Jak1, K_Jak2, K_M_STAT1, K_M_STAT3, K_um_STAT1, K_um_STAT3, K_S, scale_factor
    pfit, pcov = fit_IFNg_with_SOCS(SOCS_name, df)

    cell_types = df['Cell_type'].values

    # Set input parameters for model
    default_parameters = IFNg_parameters.copy()
    # receptor parameters
    default_parameters['pSTAT1']['K_Jak1'] = pfit[0]
    default_parameters['pSTAT3']['K_Jak1'] = pfit[0]
    default_parameters['pSTAT1']['K_Jak2'] = pfit[1]
    default_parameters['pSTAT3']['K_Jak2'] = pfit[1]
    default_parameters['pSTAT1']['K_S'] = pfit[6]
    default_parameters['pSTAT3']['K_S'] = pfit[6]
    # receptor:STAT parameters
    default_parameters['pSTAT1']['K_M'] = pfit[2]
    default_parameters['pSTAT3']['K_M'] = pfit[3]
    default_parameters['pSTAT1']['K_um'] = pfit[4]
    default_parameters['pSTAT3']['K_um'] = pfit[5]

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
        IFNg_receptor = Cytokine_Receptor(['pSTAT1', 'pSTAT3'], IFNg_parameters, 'IFNgamma')
        ec50 = IFNg_receptor.equilibrium_model_ec50(SOCS_name)
        record[c] = {'pSTAT1': ec50['pSTAT1']/1E-12, 'pSTAT3': ec50['pSTAT3']/1E-12} # in pM
    return record


def max_pSTAT_for_all_cell_types(SOCS_name, df=ImmGen_df):
    # Get parameter fits for K_Jak1, K_Jak2, K_M_STAT1, K_M_STAT3, K_um_STAT1, K_um_STAT3, K_S, scale_factor
    pfit, pcov = fit_IFNg_with_SOCS(SOCS_name, df)

    cell_types = df['Cell_type'].values

    # Set input parameters for model
    default_parameters = IFNg_parameters.copy()
    # receptor parameters
    default_parameters['pSTAT1']['K_Jak1'] = pfit[0]
    default_parameters['pSTAT3']['K_Jak1'] = pfit[0]
    default_parameters['pSTAT1']['K_Jak2'] = pfit[1]
    default_parameters['pSTAT3']['K_Jak2'] = pfit[1]
    default_parameters['pSTAT1']['K_S'] = pfit[6]
    default_parameters['pSTAT3']['K_S'] = pfit[6]
    # receptor:STAT parameters
    default_parameters['pSTAT1']['K_M'] = pfit[2]
    default_parameters['pSTAT3']['K_M'] = pfit[3]
    default_parameters['pSTAT1']['K_um'] = pfit[4]
    default_parameters['pSTAT3']['K_um'] = pfit[5]

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
        IFNg_receptor = Cytokine_Receptor(['pSTAT1', 'pSTAT3'], IFNg_parameters, 'IFNgamma')
        maxpSTAT = IFNg_receptor.equilibrium_model_pSTATmax(SOCS_name)
        record[c] = {'pSTAT1': maxpSTAT['pSTAT1']/pfit[7], 'pSTAT3': maxpSTAT['pSTAT3']/pfit[7]} # in number of molecules, scaled to match CyTOF
    return record

def make_ec50_predictions_plot():
    ec50_predictions = ec50_for_all_cell_types('SOCS2')
    maxpSTAT_predictions = max_pSTAT_for_all_cell_types('SOCS2')
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

if __name__ == "__main__":
    # Check variance in predictors
    #df = ImmGen_df[response_variable_names + predictor_variable_names]
    #print(pd.Series(np.diag(df.cov().values), index=df.columns))

    #pairwise_correlation()

    fit_with_SOCS()

    #compare_model_errors()

    #print(ec50_for_all_cell_types('SOCS2'))
    #make_ec50_predictions_plot()

