import os
import pickle
import ast
import ifndatabase.process_csv
from scipy.optimize import curve_fit
import numpy as np
from scipy.optimize import minimize
from copy import deepcopy
import seaborn as sns
import pandas as pd
import pickle

"""
Created on Sun Nov 25 10:05:14 2018

@author: Duncan

IfnData is the standardized python object for IFN data sets, used for fitting 
and plotting data.
"""


class IfnData:
    """
    Documentation - An IfnData object is a standardized object for holding 
    experimental IFN dose-response or timecourse data. This is the expected 
    data object used for plotting and fitting within the IFNmodeling module.

    The standard column labels are as follows:
    'Dose_Species', 'Dose (pM)', 'Time (min)'

    Each data point is of the form (value, std dev)

    Parameters
    ----------
    name : string
        The name of the pandas DataFrame pickled object containing the data
    Attributes
    ----------
    name : string
        The filename used to find source files for this IfnData instance
    data_set : DataFrame
        The experimental data
    conditions : dict
        A dictionary with keys corresponding to controlled experimental 
        parameters and values at which the experiments were performed
    Methods
    -------
    get_dose_range -> tuple = (min_dose, max_dose)
        min_dose = the minimum dose used in the entire experiment
        max_dose = the maximum dose used in the entire experiment
    
    """

    # Initializer / Instance Attributes
    def __init__(self, name, df=None, conditions=None):
        if name == 'custom':
            self.name = None
            self.data_set = df
            self.conditions = conditions
        else:
            self.name = name
            self.data_set = self.load_data()
            self.conditions = self.load_conditions()

    # Instance methods
    def load_data(self):
        cwd = os.getcwd()
        parent_wd = cwd.split("IFNmodeling")[0] + "IFNmodeling"
        # attempt loading DataFrame object
        try:
            return pickle.load(open(os.path.join(parent_wd, "ifndatabase","{}.p".format(self.name)), 'rb'))
        except FileNotFoundError:
            # Attempt initializing module and then importing DataFrame object
            try:
                print("Trying to build data sets")
                ifndatabase.process_csv.build_database(os.path.join(parent_wd, "ifndatabase"))
                return pickle.load(open(os.path.join(parent_wd, "ifndatabase", "{}.p".format(self.name)), 'rb'))
            except FileNotFoundError:
                # Attempt loading a local DataFrame object
                try:
                    return pickle.load(open("{}.p".format(self.name), 'rb'))
                except FileNotFoundError:
                    raise FileNotFoundError("Could not find the data file specified")

    def load_conditions(self):
        cwd = os.getcwd()
        parent_wd = cwd.split("IFNmodeling")[0] + "IFNmodeling"
        # attempt loading DataFrame object
        try:
            with open(os.path.join(parent_wd, "ifndatabase", "{}.txt".format(self.name)), 'r') as inf:
                return ast.literal_eval(inf.read())
        except FileNotFoundError:
            # Attempt loading a local conditions file if none found in data dir
            try:
                with open("{}.txt".format(self.name), 'r') as inf:
                    return ast.literal_eval(inf.read())
            # Return default None if no experimental conditions provided
            except FileNotFoundError:
                return None

    def get_dose_species(self) -> list:
        return list(self.data_set.index.levels[0])

    def get_times(self, species='') -> dict:
        keys = self.get_dose_species()
        if type(self.data_set.loc[keys[0]].columns.get_values().tolist()) == str:
            t = dict(zip(keys, [[int(el) for el in self.data_set.loc[key].columns.get_values().tolist()] for key in keys]))
        else:
            t = dict(zip(keys, [[el for el in self.data_set.loc[key].columns.get_values().tolist()] for key in keys]))
        if species=='':
            return t
        else:
            return t[species]

    def get_doses(self, species='') -> dict:
        keys = self.get_dose_species()

        dose_spec_names = [dose_species for dose_species, dose_species_data in
                           self.data_set.groupby(level='Dose_Species')]
        dose_list = [list(self.data_set.loc[spec].index) for spec in dose_spec_names]
        if species=='':
            return dict(zip(keys, dose_list))
        else:
            return dict(zip(keys, dose_list))[species]

    def get_responses(self) -> dict:
        datatable = {}
        times = self.get_times()
        for key, t in times.items():
            if str(t[0]) in self.data_set.loc[key].index:
                t = [str(n) for n in t]
                datatable.update({key: self.data_set.loc[key][t].values})
            else:
                datatable.update({key: self.data_set.loc[key][t].values})
        return datatable

    def copy(self):
        new_object = IfnData('custom', df=self.data_set.copy(), conditions=deepcopy(self.conditions))
        new_object.name = deepcopy(self.name)
        return new_object

    def __MM__(self, xdata, top, n, k):
        ydata = [top * x ** n / (k ** n + x ** n) for x in xdata]
        return ydata

    def get_ec50s(self, hill_coeff_guess = 1, errorbars=False):
        def augment_data(x_data, y_data, errorbars=False):
            min_response = min(y_data)
            min_response_idx = y_data.tolist().index(min_response)
            new_xdata = [x_data[min_response_idx]*0.1, x_data[min_response_idx]*0.3, x_data[min_response_idx]*0.8,
                         *x_data[min_response_idx:], x_data[-1]*2, x_data[-1]*5, x_data[-1]*8]
            new_ydata = [min_response, min_response, min_response, *y_data[min_response_idx:],
                         y_data[-1], y_data[-1], y_data[-1]]
            if isinstance(errorbars, type(False)):
                if errorbars == False:
                    return new_xdata, new_ydata
                else:
                    raise ValueError
            else:
                new_errs = [errorbars[min_response_idx], errorbars[min_response_idx], errorbars[min_response_idx],
                            *errorbars[min_response_idx:], errorbars[-1], errorbars[-1], errorbars[-1]]
                return new_xdata, new_ydata, new_errs
        ec50_dict = {}
        for key in self.get_dose_species():
            response_array = np.transpose([[el[0] for el in row] for row in self.get_responses()[key]])
            ec50_array = []
            for t in enumerate(self.get_times()[key]):
                # drop 0 pM dose since it can't be included on a log axis
                if self.get_doses()[key][0] == 0 or self.get_doses()[key][0] == 0.:
                    doses_to_fit = self.get_doses()[key][1:]
                    responses_to_fit = response_array[t[0]][1:]
                else:
                    doses_to_fit = self.get_doses()[key]
                    responses_to_fit = response_array[t[0]]

                # Try curve fitting
                try:
                    # Just get EC50
                    if errorbars is False:
                        doses, responses = augment_data(doses_to_fit, responses_to_fit, errorbars=False)
                        results, covariance = curve_fit(self.__MM__, doses, responses,
                                                        p0=[max(responses), hill_coeff_guess, doses[int(len(doses)/2)]])
                    # Get EC50 and error in estimate
                    else:
                        errs_array = np.transpose([[el[1] for el in row] for row in self.get_responses()[key]])
                        doses, responses, errs = augment_data(self.get_doses()[key], response_array[t[0]],
                                                              errorbars=errs_array[t[0]])
                        results, covariance = curve_fit(self.__MM__, doses, responses,
                                                        p0=[max(responses), hill_coeff_guess, doses[int(len(doses)/2)]],
                                                        sigma=errs)
                    # Catch unrealistically large results
                    if results[2] > 4E3:
                        top = max(responses) * 0.5
                        for i, r in enumerate(responses):
                            if r > top:
                                realistic_ec50 = 10 ** ((np.log10(doses[i - 1]) + np.log10(doses[i])) / 2.0)
                                if errorbars is True:
                                    ec50_array.append((t[1], realistic_ec50,
                                                       2 * 10 ** ((np.log10(doses[i - 1]) - np.log10(doses[i])) / 2.0)))
                                else:
                                    ec50_array.append((t[1], realistic_ec50))
                                break
                    # Otherwise just append results to ec50_array
                    else:
                        if errorbars is True:
                            ec50_array.append((t[1], results[2], np.sqrt(np.diag(covariance)[2])))
                        else:
                            ec50_array.append((t[1], results[2]))
                # If curve fitting fails, use some rough estimates:
                except RuntimeError:
                    print('Resorted to default guesses for IFN {} at {} min'.format(key, t[1]))
                    top = max(responses) * 0.5
                    for i, r in enumerate(responses):
                        if r > top:
                            realistic_ec50 = 10 ** ((np.log10(doses[i - 1]) + np.log10(doses[i])) / 2.0)
                            if errorbars is True:
                                ec50_array.append((t[1], realistic_ec50,
                                                   2 * 10 ** ((np.log10(doses[i - 1]) - np.log10(doses[i])) / 2.0)))
                            else:
                                ec50_array.append((t[1], realistic_ec50))
                            break
            # Update final dictionary with the ec50 curve
            ec50_dict.update({key: ec50_array})
        return ec50_dict

    def get_max_responses(self):
        Tmax_dict = {}
        for key in self.get_dose_species():
            response_array = np.transpose([[el[0] for el in row] for row in self.get_responses()[key]])
            Tmax_array = []
            if self.get_doses()[key][0] == 0. or self.get_doses()[key][0] == 0:
                for t in enumerate(self.get_times()[key]):
                    Tmax_array.append((t[1], max(response_array[t[0]][1:])))
            else:
                for t in enumerate(self.get_times()[key]):
                    Tmax_array.append((t[1], max(response_array[t[0]])))

            Tmax_dict[key] = Tmax_array
        return Tmax_dict

class DataAlignment:
    """
     Documentation - A DataAlignment object contains several IfnData objects and the scale factors needed to align them.
                   - Data must have the same Cytokine, Dose, and Time labels (ie. must be identical except for measured
                     values.
     Parameters
     ----------
     None

     Attributes
     ----------
     data (list of IfnData objects): the data sets to fit
     scale_factors (list): a float to scale each dataset by in order to optimally align them all
     scaled_data (list of IfnData objects): copies of the original IfnData objects, scaled to optimally align
     Methods
     -------
     add_data(): add an IfnData object or list of such objects to the DataAlignment object for fitting later
     align(): find the optimal scale factors for all IfnData objects in self.data
     get_scaled_data(): returns a list of pandas dataframes containing all the data objects with their transformed
                        values as per the alignment
     summarize_data(): finds the mean and variance of each data point and returns the IfnData object continaing this
                        computed information
     get_ec50s(): finds the ec50 at each time point for each data set and returns the ec50 vs time curve with error bars
                    computed by comparing between data sets
     """

    # Initializer / Instance Attributes
    def __init__(self):
        self.data = []
        self.scale_factors = []
        self.scaled_data = []

    # Private methods
    # ---------------
    def __score_sf__(self, scf, data, reftable):
        diff_table = np.subtract(reftable, np.multiply(data, scf[0]))
        return np.sum(np.square(diff_table))

    # Public methods
    # --------------
    def add_data(self, data_object):
        if isinstance(data_object, IfnData):
            self.data.append(data_object)
            if len(self.data) == 1:
                self.scale_factors = [1.]
        elif type(data_object) == list:
            for d in data_object:
                if isinstance(d, IfnData):
                    self.data.append(d)
                    if len(self.data) == 1:
                        self.scale_factors = [1.]
                else:
                    raise TypeError("Must add IfnData instance or a list of such instances")
        else:
            raise TypeError("Must add IfnData instance or a list of such instances")

    def save(self, fname, save_dir=os.getcwd()):
        ifn_name_list = []
        if save_dir != os.getcwd():
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        for ifndata_obj in self.data:
            ifn_name_list.append(ifndata_obj.name)
            ifndata_obj.data_set.to_hdf(os.path.join(save_dir, ifndata_obj.name + '.h5'), key='data_set', mode='w')
            with open(os.path.join(save_dir, ifndata_obj.name + '.p'), 'wb') as fp:
                pickle.dump(ifndata_obj.conditions, fp)
        for ifndata_obj in self.scaled_data:
            ifndata_obj.data_set.to_hdf(os.path.join(save_dir, ifndata_obj.name + '_scaled.h5'), key='data_set', mode='w')
            with open(os.path.join(save_dir, ifndata_obj.name + '.p'), 'wb') as fp:
                pickle.dump(ifndata_obj.conditions, fp)

        with open(os.path.join(save_dir, fname + '_scalefactors.p'), 'wb') as fp:
            pickle.dump(self.scale_factors, fp)
        with open(os.path.join(save_dir, 'IfnData_names.p'), 'wb') as fp:
            pickle.dump(ifn_name_list, fp)

    def load_from_save_file(self, fname, save_dir):
        with open(os.path.join(save_dir, 'IfnData_names.p'), 'rb') as fp:
            ifn_name_list = pickle.load(fp)
        with open(os.path.join(save_dir, fname + '_scalefactors.p'), 'rb') as fp:
            self.scale_factors = pickle.load(fp)
        for name in ifn_name_list:
            temp = IfnData('custom', df=pd.read_hdf(os.path.join(save_dir, name + '.h5'), 'data_set'))
            temp.name = name
            self.data.append(temp)
        for name in ifn_name_list:
            temp = IfnData('custom', df=pd.read_hdf(os.path.join(save_dir, name + '_scaled.h5'), 'data_set'))
            temp.name = name
            self.data.append(temp)

    def align(self):
        self.scale_factors = np.zeros(len(self.data))
        self.scale_factors[0] = 1
        reference_table = [[el[0] for el in row] for row in self.data[0].data_set.values]
        datatable = [[[el[0] for el in row] for row in d.data_set.values] for d in self.data[1:]]
        for d in range(len(datatable)):
            opt = minimize(self.__score_sf__, [0.1], args=(datatable[d], reference_table))
            self.scale_factors[d+1] = opt['x']
        return self.scale_factors

    def get_scaled_data(self):
        self.scaled_data = [d.copy() for d in self.data]
        for d in range(1, len(self.scaled_data)):
            current_IfnData_object = self.scaled_data[d]
            scale_factor = self.scale_factors[d]
            scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])

            for spec in current_IfnData_object.get_dose_species():
                num_times = len(current_IfnData_object.get_times()[spec])
                for i in range(num_times):
                    current_IfnData_object.data_set.loc[spec].iloc[:, i] = current_IfnData_object.data_set.loc[spec].iloc[:, i].apply(scale_data)
        return self.scaled_data

    def summarize_data(self):
        data_list = {}
        for spec in self.scaled_data[0].get_dose_species():
            data_list.update({spec: [[[el[0] for el in row] for row in self.scaled_data[i].data_set.loc[spec].values]
                                     for i in range(len(self.scaled_data))]})
        mean_data = {key: np.mean(data_list[key], axis=0) for key in self.scaled_data[0].get_dose_species()}
        stddev = {key: np.std(data_list[key], axis=0) for key in self.scaled_data[0].get_dose_species()}
        dose_species_list = self.scaled_data[0].get_dose_species()
        row_length = len(mean_data[dose_species_list[0]][0])
        column_length = len(mean_data[dose_species_list[0]])
        zipped_data = []
        for key in dose_species_list:
            for row_idx, dose in enumerate(self.scaled_data[0].get_doses(key)):
                zipped_data.append(
                    [key, dose, *[(mean_data[key][row_idx][i], stddev[key][row_idx][i]) for i in range(row_length)]])
        df = pd.DataFrame.from_records(zipped_data,
                                       columns=['Dose_Species', 'Dose (pM)', *self.scaled_data[0].get_times(key)])
        df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
        summary_IfnData = IfnData('custom', df=df, conditions=self.scaled_data[0].conditions)
        return summary_IfnData

    def get_ec50s(self):
        #if self.scaled_data == []:
        #    raise AttributeError('Align data first using self.align() and then self.get_scaled_data()')
        temp = self.data[0].get_ec50s()
        times = [el[0] for el in temp[list(temp.keys())[0]]]
        mean_ec50 = {key: [] for key in temp.keys()}
        error_bars = {key: [] for key in temp.keys()}
        for d in self.data:
            ec50 = d.get_ec50s()
            for key in ec50.keys():
                mean_ec50[key].append([el[1] for el in ec50[key]])
        for key in mean_ec50.keys():
            error_bars[key] = list(zip(times, np.std(mean_ec50[key], axis=0)))
            mean_ec50[key] = list(zip(times, np.mean(mean_ec50[key], axis=0)))
        return mean_ec50, error_bars