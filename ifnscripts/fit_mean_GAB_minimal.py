from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import DoseresponsePlot, TimecoursePlot
import seaborn as sns
import numpy as np
import pandas as pd
from ifnclass.ifnfit import StepwiseFit


if __name__ == '__main__':
    # ------------------------------
    # Align all data
    # ------------------------------
    newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
    newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
    newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
    newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")

    alignment = DataAlignment()
    alignment.add_data([newdata_4, newdata_3, newdata_2, newdata_1])
    alignment.align()
    alignment.get_scaled_data()
    mean_data = alignment.summarize_data()

    # -------------------------------
    # Initialize model
    # -------------------------------
    times = newdata_4.get_times('Alpha')
    doses_alpha = newdata_4.get_doses('Alpha')
    doses_beta = newdata_4.get_doses('Beta')
    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')
    Mixed_Model.set_parameters({'R2': 4920, 'R1': 1200,
                                'kSOCSon': 6.4e-08, 'kpu': 0.0022, 'kpa': 1e-06,
                                'kint_a':  0.0001, 'kint_b': 0.00086,
                                'krec_a1': 0.05, 'krec_a2': 0.01, 'krec_b1': 0.005, 'krec_b2': 0.05})
    scale_factor = 1.46182313424
    scale_data = lambda q: (scale_factor * q[0], scale_factor * q[1])

    # -------------------------------
    # Perform stepwise fit
    # -------------------------------
    #stepfit = StepwiseFit(Mixed_Model, mean_data,
    #                      {'R2': (200, 12000), 'R1': (200, 12000),
    #                       'kSOCSon': (1e-08, 1e-07), 'kpu': (0.001, 0.01), 'kpa': (1e-06, 1e-05),
    #                       'kint_a':  (0.0001, 0.001), 'kint_b': (0.0001, 0.001),
    #                       'krec_a1': (0.005, 0.05), 'krec_a2': (0.005, 0.05),
    #                       'krec_b1': (0.001, 0.01), 'krec_b2': (0.001, 0.01)}, n=6)
    #best_parameters, scale_factor = stepfit.fit()
    #print(best_parameters)
    #print(scale_factor)
    #Mixed_Model = stepfit.model

    # -------------------------------
    # Make predictions
    # -------------------------------
    dradf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia',
                                     list(np.logspace(np.log10(doses_alpha[0]), np.log10(doses_alpha[-1]))),
                                     parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib',
                                     list(np.logspace(np.log10(doses_beta[0]), np.log10(doses_beta[-1]))),
                                     parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')
    for i in range(len(times)):
        dradf.loc['Alpha'].iloc[:, i] = dradf.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf.loc['Beta'].iloc[:, i] = drbdf.loc['Beta'].iloc[:, i].apply(scale_data)

    dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    # -------------------------
    # Plot mean trajectory
    # -------------------------
    alpha_palette = sns.color_palette("Reds", 1)
    alpha_model_palette = sns.color_palette("Greys", 1)
    beta_palette = sns.color_palette("Greens", 1)
    beta_model_palette = sns.color_palette("Greys", 1)

    dr_plot = DoseresponsePlot((6, 2))
    for idx, t in enumerate(newdata_4.get_times('Alpha')):
        dr_plot.add_trajectory(mean_data, t, 'errorbar', alpha_palette[0], (idx, 0), 'Alpha')
        dr_plot.add_trajectory(mean_data, t, 'scatter', 'ro', (idx, 0), 'Alpha', color=alpha_palette[0], label='Alpha ' + str(t))
    for idx, t in enumerate(newdata_4.get_times('Beta')):
        dr_plot.add_trajectory(mean_data, t, 'errorbar', beta_palette[0], (idx, 1), 'Beta')
        dr_plot.add_trajectory(mean_data, t, 'scatter', 'go', (idx, 1), 'Beta', color=beta_palette[0], label='Beta ' + str(t))

    # ----------------------------------
    # Plot best fit for each time slice
    # ----------------------------------
    for idx, t in enumerate(newdata_4.get_times('Alpha')):
        dr_plot.add_trajectory(dra60, t, 'plot', alpha_model_palette[0], (idx, 0), 'Alpha', label='Alpha model ' + str(t))
    for idx, t in enumerate(newdata_4.get_times('Beta')):
        dr_plot.add_trajectory(drb60, t, 'plot', beta_model_palette[0], (idx, 1), 'Beta', label='Beta model ' + str(t))

    dr_fig, dr_axes = dr_plot.show_figure(save_flag=False)
    dr_fig.set_size_inches(10.5, 7.1*6)
    dr_fig.savefig('results/GAB_NewData/fit_mean_minimal.pdf')

    # -------------------------------
    # Plot dose response paper figure
    # -------------------------------
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    dr_plot_mean_fit = DoseresponsePlot((1, 2))
    alpha_mask = [7.5]
    beta_mask = [7.5]
    # Add fits
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            dr_plot_mean_fit.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='Alpha ' + str(t))
        if t not in beta_mask:
            dr_plot_mean_fit.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label='Beta ' + str(t))
    # Add data
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            dr_plot_mean_fit.add_trajectory(mean_data, t, 'plot', '--', (0, 0), 'Alpha', label='Alpha ' + str(t),
                                            color=alpha_palette[idx])
            dr_plot_mean_fit.add_trajectory(mean_data, t, 'scatter', 'ro', (0, 0), 'Alpha', label='', color=alpha_palette[idx])
        if t not in beta_mask:
            dr_plot_mean_fit.add_trajectory(mean_data, t, 'plot', '--', (0, 1), 'Beta', label='Beta ' + str(t),
                                            color=beta_palette[idx])
            dr_plot_mean_fit.add_trajectory(mean_data, t, 'scatter', 'go', (0, 1), 'Beta', label='', color=beta_palette[idx])

    dr_plot_mean_fit.show_figure(save_flag=False)

    # -------------------------------
    # Plot time course paper figure
    # -------------------------------
    alpha_palette = sns.color_palette("Reds", 8)
    beta_palette = sns.color_palette("Greens", 8)

    # Simulate time courses
    alpha_time_courses = []
    for d in doses_alpha:
        alpha_time_courses.append(Mixed_Model.timecourse(list(np.linspace(0, 60, 30)), 'TotalpSTAT',
                                                         {'Ia': d * 6.022E23 * 1E-5 * 1E-12, 'Ib': 0},
                                                         return_type='dataframe', dataframe_labels=['Alpha', d]))
    beta_time_courses = []
    for d in doses_beta:
        beta_time_courses.append(Mixed_Model.timecourse(list(np.linspace(0, 60, 30)), 'TotalpSTAT',
                                                        {'Ib': d * 6.022E23 * 1E-5 * 1E-12, 'Ia': 0},
                                                        return_type='dataframe', dataframe_labels=['Beta', d]))
    # Scale simulations
    for i in range(30):
        for j in range(len(doses_alpha)):
            alpha_time_courses[j].loc['Alpha'].iloc[:, i] = alpha_time_courses[j].loc['Alpha'].iloc[:, i].apply(
                scale_data)
        for j in range(len(doses_beta)):
            beta_time_courses[j].loc['Beta'].iloc[:, i] = beta_time_courses[j].loc['Beta'].iloc[:, i].apply(scale_data)
    # Turn into IfnData objects
    alpha_IfnData_objects = []
    beta_IfnData_objects = []
    for j in range(len(doses_alpha)):
        alpha_IfnData_objects.append(IfnData('custom', df=alpha_time_courses[j], conditions={'Alpha': {'Ib': 0}}))
    for j in range(len(doses_beta)):
        beta_IfnData_objects.append(IfnData('custom', df=beta_time_courses[j], conditions={'Beta': {'Ia': 0}}))
    # Generate plot
    time_course_paper_fig = TimecoursePlot((1, 2))
    alpha_mask = [0]
    beta_mask = [0]

    # Add fits
    for j, dose in enumerate(doses_alpha):
        if dose not in alpha_mask:
            time_course_paper_fig.add_trajectory(alpha_IfnData_objects[j], 'plot', alpha_palette[j], (0, 0),
                                                 label='Alpha ' + str(dose))
    for j, dose in enumerate(doses_beta):
        if dose not in beta_mask:
            time_course_paper_fig.add_trajectory(beta_IfnData_objects[j], 'plot', beta_palette[j], (0, 1),
                                                 label='Beta ' + str(dose))
    # Add data
    for idx, d in enumerate(doses_alpha):
        # Optional mask:
        if d not in alpha_mask:
            atc = IfnData('custom', df=mean_data.data_set.loc['Alpha', d, :])
            time_course_paper_fig.add_trajectory(atc, 'scatter', 'o', (0, 0), label='Alpha ' + str(d), color=alpha_palette[idx])
            time_course_paper_fig.add_trajectory(atc, 'errorbar', alpha_palette[idx], (0, 0), color=alpha_palette[idx])
    for idx, d in enumerate(doses_beta):
        if d not in beta_mask:
            btc = IfnData('custom', df=mean_data.data_set.loc['Beta', d, :])
            time_course_paper_fig.add_trajectory(btc, 'scatter', 'o', (0, 1), label='Beta ' + str(d), color=beta_palette[idx])
            time_course_paper_fig.add_trajectory(btc, 'errorbar', beta_palette[idx], (0, 1), color=beta_palette[idx])

    time_course_paper_fig.show_figure()

