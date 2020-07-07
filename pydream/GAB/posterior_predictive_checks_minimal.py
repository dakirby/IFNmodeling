import numpy as np
import os
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ifnclass.ifnfit import IfnModel
from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnplot import DoseresponsePlot

# For use in directory /PyDREAM_07-07-2020_10000
# User input
output_dir = os.path.join(os.getcwd(),'PyDREAM_07-07-2020_10000')
sim_name = 'mixed_IFN'
num_checks = 50

# Preparation of parameter ensemble
log10_parameters = np.load(output_dir + os.sep + sim_name + '_samples.npy')
parameters = np.power(10, log10_parameters)

parameter_names = pd.read_csv(output_dir + os.sep + 'descriptive_statistics.csv').columns.values[1:]

parameters_to_check = [parameters[i] for i in list(np.random.randint(0, high=len(parameters), size=num_checks))]

# Set up model
Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
#Mixed_Model.set_parameters({'k_a2': 0.5*Mixed_Model.parameters['k_a2'], 'ka2': 0.5*Mixed_Model.parameters['ka2']})
sf = 2.0

times = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
dose_range = np.logspace(-1,5,15)

# Compute posterior sample trajectories
def posterior_prediction(parameter_vector, parameter_names=parameter_names):
    # Make predictions
    Mixed_Model.set_parameters(parameter_vector)
    dradf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', dose_range, parameters={'Ib': 0}, 
                                scale_factor=sf, return_type='dataframe', dataframe_labels='Alpha')
    drbdf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', dose_range, parameters={'Ia': 0}, 
                                scale_factor=sf, return_type='dataframe', dataframe_labels='Beta')

    posterior = IfnData('custom', df=pd.concat((dradf, drbdf)), conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})
    posterior.drop_sigmas()
    return posterior

posterior_trajectories = []
for p in parameters_to_check:
    param_dict = {key: value for key, value in zip(parameter_names, p)}
    pp = posterior_prediction(param_dict)
    posterior_trajectories.append(pp)

# Make aggregate predicitions
mean_alpha_predictions = np.mean([posterior_trajectories[i].data_set.loc['Alpha'].values for i in
                                  range(len(posterior_trajectories))], axis=0)
mean_beta_predictions = np.mean([posterior_trajectories[i].data_set.loc['Beta'].values for i in
                                 range(len(posterior_trajectories))], axis=0)

std_alpha_predictions = np.std([posterior_trajectories[i].data_set.loc['Alpha'].values.astype(np.float64) for i in
                                range(len(posterior_trajectories))], axis=0)
std_beta_predictions = np.std([posterior_trajectories[i].data_set.loc['Beta'].values.astype(np.float64) for i in
                               range(len(posterior_trajectories))], axis=0)

std_predictions = {'Alpha': std_alpha_predictions, 'Beta': std_beta_predictions}
mean_predictions = {'Alpha': mean_alpha_predictions, 'Beta': mean_beta_predictions}

mean_model = copy.deepcopy(posterior_trajectories[0])
for s in ['Alpha', 'Beta']:
    for didx, d in enumerate(mean_model.get_doses()[s]):
        for tidx, t in enumerate(mean_model.get_times()[s]):
            mean_model.data_set.loc[s][str(t)].loc[d] = (mean_predictions[s][didx][tidx], std_predictions[s][didx][tidx])

# Get aligned data
newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")

# Aligned data, to get scale factors for each data set
alignment = DataAlignment()
alignment.add_data([newdata_4, newdata_3, newdata_2, newdata_1])
alignment.align()
alignment.get_scaled_data()
mean_data = alignment.summarize_data()

# Plot posterior samples
alpha_palette = sns.color_palette("deep", 6)
beta_palette = sns.color_palette("deep", 6)

new_fit = DoseresponsePlot((1, 2))

alpha_mask = [2.5, 7.5, 10.0] #[2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
beta_mask = [2.5, 7.5, 10.0]
plot_data = True
# Add fits
for idx, t in enumerate(times):
    if t not in alpha_mask:
        new_fit.add_trajectory(mean_model, t, 'envelope', alpha_palette[idx], (0, 0), 'Alpha', label='{} min'.format(t),
                               linewidth=2, alpha=0.2)
        if plot_data == True:
            new_fit.add_trajectory(mean_data, t, 'errorbar', 'o', (0, 0), 'Alpha', color=alpha_palette[idx])
    if t not in beta_mask:
        new_fit.add_trajectory(mean_model, t, 'envelope', beta_palette[idx], (0, 1), 'Beta', label='{} min'.format(t),
                               linewidth=2, alpha=0.2)
        if plot_data == True:
            new_fit.add_trajectory(mean_data, t, 'errorbar', 'o', (0, 1), 'Beta', color=beta_palette[idx])

# Change legend transparency
leg = new_fit.fig.legend()
for lh in leg.legendHandles:
    lh._legmarker.set_alpha(1)

dr_fig, dr_axes = new_fit.show_figure()
dr_fig.set_size_inches(14, 6)
dr_axes[0].set_title(r'IFN$\alpha$')
dr_axes[1].set_title(r'IFN$\beta$')

if plot_data == True:
    dr_fig.savefig(os.path.join(os.getcwd(), output_dir, 'posterior_predictions_with_data.pdf'))
else:
    dr_fig.savefig(os.path.join(os.getcwd(), output_dir, 'posterior_predictions.pdf'))
