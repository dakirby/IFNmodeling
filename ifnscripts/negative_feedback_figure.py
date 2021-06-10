from ifnclass.ifndata import IfnData
from ifnclass.ifnplot import DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import numpy as np
import seaborn as sns
import load_model as lm
import copy
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def sample_dist_params(parameter_dict, nsamples):
    # find all distribution parameters
    dist_param_names = []
    for key in parameter_dict.keys():
        if key.endswith('_mu*'):
            dist_param_names.append(key[:-4])
    # sample according to mu, std
    params_list = []
    for _ in range(nsamples):
        temp = copy.deepcopy(parameter_dict)
        dist_param_dict = {}
        for pname in dist_param_names:
            mu = log10(parameter_dict[pname + '_mu*'])
            std = parameter_dict[pname + '_std*']
            sample = 10 ** np.random.normal(loc=mu, scale=std)
            dist_param_dict.update({pname: sample})
            # remove distribution parameters
            temp.pop(pname + '_mu*')
            temp.pop(pname + '_std*')
        # add sample to parameter_dict
        temp.update(dist_param_dict)
        temp = {key: val for key, val in temp.items() if key in dist_param_names}
        params_list.append(copy.deepcopy(temp))
    return params_list


def aggregate_response(model, method, model_type, plist, times=[60], scale_factor=1.5):
    if model_type == 'MEDIAN':
        alpha_traj = []
        beta_traj = []
        for i in tqdm(range(len(plist))):
            p = plist[i]
            # simulate IFN alpha-2
            tempP = copy.deepcopy(p)
            tempP.update({'Ib': 0})
            tempA = method(times, 'TotalpSTAT', 'Ia', list(logspace(-2, 8, num=30)),
                           parameters=tempP, return_type='DataFrame', dataframe_labels='Alpha',
                           scale_factor=scale_factor, no_sigma=True)

            # simulate IFN beta
            tempP = copy.deepcopy(p)
            tempP.update({'Ia': 0})
            tempB = method(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 8, num=30)),
                           parameters=tempP, return_type='DataFrame', dataframe_labels='Beta',
                           scale_factor=scale_factor, no_sigma=True)

            # add to record of trajectories
            alpha_traj.append(copy.deepcopy(tempA))
            beta_traj.append(copy.deepcopy(tempB))

        # get aggregate predictions
        dose_species = 'Alpha'
        dra60 = IfnData('custom', df=copy.deepcopy(alpha_traj[0]), conditions={'Alpha': {'Ib': 0}})
        mean_alpha_predictions = np.mean([alpha_traj[i].loc[dose_species].values for i in range(len(alpha_traj))], axis=0)
        for didx, d in enumerate(dra60.get_doses()['Alpha']):
            for tidx, t in enumerate(dra60.get_times()['Alpha']):
                dra60.data_set.loc['Alpha'][str(t)].loc[d] = mean_alpha_predictions[didx][tidx]

        dose_species = 'Beta'
        drb60 = IfnData('custom', df=copy.deepcopy(beta_traj[0]), conditions={'Beta': {'Ia': 0}})
        mean_beta_predictions = np.mean([beta_traj[i].loc[dose_species].values for i in range(len(beta_traj))], axis=0)
        for didx, d in enumerate(drb60.get_doses()['Beta']):
            for tidx, t in enumerate(drb60.get_times()['Beta']):
                drb60.data_set.loc['Beta'][str(t)].loc[d] = mean_beta_predictions[didx][tidx]

    elif model_type == 'SINGLE_CELL':
        dradf = method(times, 'TotalpSTAT', 'Ia', list(logspace(-2, 8)),
                       parameters={'Ib': 0}, return_type='DataFrame', dataframe_labels='Alpha',
                       scale_factor=scale_factor)

        drbdf = method(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 8)),
                       parameters={'Ia': 0}, return_type='DataFrame', dataframe_labels='Beta',
                       scale_factor=scale_factor)
        dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
        drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    else:
        raise ValueError("Model type must be MEDIAN or SINGLE_CELL")

    return dra60, drb60


if __name__ == '__main__':
    # --------------------
    # User controls
    # --------------------
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    out_dir = os.path.join(os.getcwd(), 'results', 'Figures', 'Figure_3')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fname = out_dir + os.sep + 'negative_feedback_figure.pdf'

    DR_KWARGS = {'return_type': 'IfnData'}
    PLOT_KWARGS = {'line_type': 'plot', 'alpha': 1}
    MODEL_TYPE = 'MEDIAN'  # 'SINGLE_CELL'
    NSAMPLES = 30

    # --------------------
    # Set up Model
    # --------------------
    assert MODEL_TYPE in ['MEDIAN', 'SINGLE_CELL']

    # Overide MODEL_TYPE for now; sampling will be done later to ensure
    # populations are synchronized
    Mixed_Model, DR_method = lm.load_model(MODEL_TYPE='SINGLE_CELL')
    Mixed_Model.set_default_parameters(Mixed_Model.get_parameters())

    if MODEL_TYPE == 'MEDIAN':
        prior_model, _ = lm.load_model(MODEL_TYPE='MEDIAN')
        ptest_template = dict(zip(prior_model.parameter_names, prior_model.parameters.tolist()[0]))
        R_samples = sample_dist_params(ptest_template, NSAMPLES)
    else:
        R_samples = [Mixed_Model.parameters]

    # --------------------
    # Run Simulations
    # --------------------
    times = [60]
    # Control Dose-Response
    Mixed_Model.set_parameters({'kSOCSon': 0, 'kIntBasal_r1': 0, 'kIntBasal_r2': 0, 'kint_a': 0, 'kint_b': 0})
    dra60, drb60 = aggregate_response(Mixed_Model, DR_method, MODEL_TYPE, R_samples)

    # Show internalization effects
    Mixed_Model.reset_parameters()
    Mixed_Model.set_parameters({'kSOCSon': 0})
    dra60_int, drb60_int = aggregate_response(Mixed_Model, DR_method, MODEL_TYPE, R_samples)

    # Show SOCS effects
    Mixed_Model.reset_parameters()
    Mixed_Model.set_parameters({'kIntBasal_r1': 0, 'kIntBasal_r2': 0, 'kint_a': 0, 'kint_b': 0})
    dra60_SOCS, drb60_SOCS = aggregate_response(Mixed_Model, DR_method, MODEL_TYPE, R_samples)

    # --------------------
    # Make Plot
    # --------------------
    dr_plot = DoseresponsePlot((1, 1))
    alpha_mask = []
    beta_mask = []
    for idx, t in enumerate([el for el in times]):
        if t not in alpha_mask:
            dr_plot.add_trajectory(dra60, t, 'plot', alpha_palette[5], (0, 0), 'Alpha',  linewidth=2.0)
            dr_plot.add_trajectory(dra60_int, t, 'plot', '--', (0, 0), 'Alpha', color=alpha_palette[5], linewidth=2.0)
            dr_plot.add_trajectory(dra60_SOCS, t, 'plot', ':', (0, 0), 'Alpha', color=alpha_palette[5], linewidth=2.0)
        if t not in beta_mask:
            dr_plot.add_trajectory(drb60, t, 'plot', beta_palette[5], (0, 0), 'Beta', linewidth=2.0)
            dr_plot.add_trajectory(drb60_int, t, 'plot', '--', (0, 0), 'Beta', color=beta_palette[5], linewidth=2.0)
            dr_plot.add_trajectory(drb60_SOCS, t, 'plot', ':', (0, 0), 'Beta', color=beta_palette[5], linewidth=2.0)
    # Legend:
    plt.scatter([], [], color=alpha_palette[5], label=r'IFN$\alpha$2', figure=dr_plot.fig)
    plt.scatter([], [], color=beta_palette[5], label=r'IFN$\beta$', figure=dr_plot.fig)
    plt.plot([], [], c='grey', label='No Feedback', linewidth=2.0, figure=dr_plot.fig)
    plt.plot([], [], '--', c='grey', label='Effect of Internalization', linewidth=2.0, figure=dr_plot.fig)
    plt.plot([], [], ':', c='grey', label='Effect of SOCS', linewidth=2.0, figure=dr_plot.fig)

    # Plot formatting
    dr_plot.fig.set_size_inches((5, 4))
    dr_plot.axes.set_title('Effects of Negative Feedback')
    dr_plot.axes.set_ylabel('pSTAT')
    dr_plot.axes.spines['top'].set_visible(False)
    dr_plot.axes.spines['right'].set_visible(False)
    dr_plot.save_figure(save_dir=fname, tight=True)
