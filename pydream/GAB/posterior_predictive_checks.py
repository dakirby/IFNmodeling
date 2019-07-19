import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ifnclass.ifnfit import DualMixedPopulation
from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnplot import DoseresponsePlot

# User input
output_dir = os.path.join(os.getcwd(), 'PyDREAM_27-06-2019_10000')
sim_name = 'mixed_IFN'
num_checks = 30

# Preparation of parameter ensemble
log10_parameters = np.load(output_dir + os.sep + sim_name + '_samples.npy')
parameters = np.power(10, log10_parameters)

parameter_names = pd.read_csv(output_dir + os.sep + 'descriptive_statistics.csv').columns.values[1:]

parameters_to_check = [parameters[i] for i in list(np.random.randint(0, high=len(parameters), size=num_checks))]

# Set up model
Mixed_Model = DualMixedPopulation('Mixed_IFN_ppCompatible', 1.0, 0.0)
opt_params = {'R2': 4920, 'R1': 1200,
                'k_a1': 2.0e-13, 'k_a2': 1.328e-12, 'k_d3': 1.13e-4, 'k_d4': 0.9,
                'kSOCSon': 5e-08, 'kpu': 0.0022, 'kpa': 2.36e-06,
                'ka1': 3.3e-15, 'ka2': 1.85e-12, 'kd4': 2.0,
                'kd3': 6.52e-05,
                'kint_a': 0.0015, 'kint_b': 0.002,
                'krec_a1': 0.01, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05}
Mixed_Model.set_global_parameters(opt_params)
Mixed_Model.model_1.default_parameters.update(opt_params)
Mixed_Model.model_2.default_parameters.update(opt_params)
scale_factor = 1.46182313424
times = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]


# Compute posterior sample trajectories
def posterior_prediction(parameter_vector):
    # Make predictions

    dradf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia', list(np.logspace(1, 5.2)),
                                            parameters=dict({'Ib': 0}, **parameter_vector), sf=scale_factor)
    drbdf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ib', list(np.logspace(-1, 4)),
                                            parameters=dict({'Ia': 0}, **parameter_vector), sf=scale_factor)

    posterior = IfnData('custom', df=pd.concat((dradf, drbdf)), conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})

    return posterior

posterior_trajectories = []
for p in parameters_to_check:
    param_dict = {key: value for key, value in zip(parameter_names, p)}
    pp = posterior_prediction(param_dict)
    posterior_trajectories.append(pp)

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
        for p in posterior_trajectories[1:]:
            new_fit.add_trajectory(p, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='', linewidth=2, alpha=0.2)
            if plot_data == True:
                new_fit.add_trajectory(mean_data, t, 'errorbar', 'o', (0, 0), 'Alpha', color=alpha_palette[idx])
    if t not in beta_mask:
        for p in posterior_trajectories[1:]:
            new_fit.add_trajectory(p, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label='', linewidth=2, alpha=0.2)
            if plot_data == True:
                new_fit.add_trajectory(mean_data, t, 'errorbar', 'o', (0, 1), 'Beta', color=beta_palette[idx])
for idx, t in enumerate(times):
    if t not in alpha_mask:
        new_fit.add_trajectory(posterior_trajectories[0], t, 'plot', alpha_palette[idx], (0, 0), 'Alpha',
                               label='{} min'.format(t), linewidth=2, alpha=0.2)
    if t not in beta_mask:
        new_fit.add_trajectory(posterior_trajectories[0], t, 'plot', beta_palette[idx], (0, 1), 'Beta',
                               label='{} min'.format(t), linewidth=2, alpha=0.2)

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

