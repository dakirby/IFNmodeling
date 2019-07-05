import numpy as np
import os
import pandas as pd
import seaborn as sns
from ifnclass.ifnfit import DualMixedPopulation
from ifnclass.ifndata import IfnData
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
Mixed_Model.set_parameters(opt_params)
Mixed_Model.default_parameters.update(opt_params)
scale_factor = 1.46182313424
times = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]


# Compute posterior sample trajectories
def posterior_prediction(parameter_vector):
    # Make predictions

    dradf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia', list(np.logspace(1, 5.2)),
                                            parameters=dict({'Ib': 0}, **parameter_vector), sf=scale_factor)
    drbdf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ib', list(np.logspace(-1, 4)),
                                            parameters=dict({'Ia': 0}, **parameter_vector), sf=np.scale_factor)

    posterior = IfnData('custom', df=pd.concat(dradf,drbdf), conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})

    return posterior

posterior_trajectories = []
for p in parameters_to_check:
    param_dict = {key: value for key, value in zip(parameter_names, p)}
    pp = posterior_prediction(param_dict)
    posterior_trajectories.append(pp)

# Plot posterior samples
    alpha_palette = sns.color_palette("deep", 6)
    beta_palette = sns.color_palette("deep", 6)

    new_fit = DoseresponsePlot((1, 2))

alpha_mask = [2.5, 7.5, 10.0] #[2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
beta_mask = [2.5, 7.5, 10.0]
# Add fits
for idx, t in enumerate(times):
    if t not in alpha_mask:
        new_fit.add_trajectory(posterior_trajectories[idx], t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label=str(t)+' min',
                               linewidth=2)
    if t not in beta_mask:
        new_fit.add_trajectory(posterior_trajectories[idx], t, 'plot', beta_palette[idx], (0, 1), 'Beta', label=str(t) +' min',
                               linewidth=2)

dr_fig, dr_axes = new_fit.show_figure()
dr_fig.set_size_inches(14.75, 8)
dr_fig.tight_layout()
dr_fig.savefig(os.path.join(os.getcwd(), 'posterior_predictions.pdf'))




