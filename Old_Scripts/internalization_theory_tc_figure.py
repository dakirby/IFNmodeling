from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import seaborn as sns


if __name__ == '__main__':
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    Mixed_Model = IfnModel('Mixed_IFN_ppCompatible')

    # This is the fitting_2_5 script best fit parameters, best at fitting Ratnadeep's B cell data
    Mixed_Model.set_parameters(
        {'R2': 2300 * 2.5,
         'R1': 1800 * 1.8, 'k_d4': 0.06, 'kint_b': 0.0003,
         'kpu': 0.0028,
         'krec_b1': 0.001, 'krec_b2': 0.01,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 4, 'kSOCSon': 0.9e-8,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3, 'kd4': 1.0, 'kd3': 0.001,
         'kint_a': 0.0014, 'krec_a1': 9e-03, 'krec_a2': 0.05})
    scale_factor = 0.036
    scale_data = lambda q: (scale_factor*q[0], scale_factor*q[1])

    # Produce plots
    times = [5, 15, 30, 60]
    # Baseline
    dradf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-2, 8)),
                                           parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 8)),
                                           parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')
    # Dramatically increased internalization to show effect
    internalization_sf = 5
    Mixed_Model.set_parameters({'kint_b': 0.0003 * internalization_sf, 'kint_a': 0.0014 * internalization_sf})
    dradf_int = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-2, 8)),
                                           parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf_int = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 8)),
                                           parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')

    # Change recycling asymmetrically to demonstrate effect
    recycle_sf = 10000000
    #Mixed_Model.set_parameters({'krec_a2': 0.05 * recycle_sf, 'krec_b2': 0.01 * recycle_sf})
    Mixed_Model.set_parameters({'krec_a1': 9e-03 * recycle_sf, 'krec_b1': 0.001 * recycle_sf})
    dradf_rec = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ia', list(logspace(-2, 8)),
                                           parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')
    drbdf_rec = Mixed_Model.doseresponse(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 8)),
                                           parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')
    for i in range(len(times)):
        dradf.loc['Alpha'].iloc[:, i] = dradf.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf.loc['Beta'].iloc[:, i] = drbdf.loc['Beta'].iloc[:, i].apply(scale_data)
        dradf_int.loc['Alpha'].iloc[:, i] = dradf_int.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf_int.loc['Beta'].iloc[:, i] = drbdf_int.loc['Beta'].iloc[:, i].apply(scale_data)
        dradf_rec.loc['Alpha'].iloc[:, i] = dradf_rec.loc['Alpha'].iloc[:, i].apply(scale_data)
        drbdf_rec.loc['Beta'].iloc[:, i] = drbdf_rec.loc['Beta'].iloc[:, i].apply(scale_data)

    dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})
    dra60_int = IfnData('custom', df=dradf_int, conditions={'Alpha': {'Ib': 0}})
    drb60_int = IfnData('custom', df=drbdf_int, conditions={'Beta': {'Ia': 0}})
    dra60_rec = IfnData('custom', df=dradf_rec, conditions={'Alpha': {'Ib': 0}})
    drb60_rec = IfnData('custom', df=drbdf_rec, conditions={'Beta': {'Ia': 0}})

    dr_plot = DoseresponsePlot((3, 2))
    alpha_mask = []
    beta_mask = []
    # Add fits
    for idx, t in enumerate([el for el in times]):
        if t not in alpha_mask:
            dr_plot.add_trajectory(dra60, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', label='Alpha '+str(t), linewidth=2.0)
            dr_plot.add_trajectory(dra60_int, t, 'plot', alpha_palette[idx], (1, 0), 'Alpha', label='Alpha ' + str(t), linewidth=2.0)
            dr_plot.add_trajectory(dra60_rec, t, 'plot', alpha_palette[idx], (2, 0), 'Alpha', label='Alpha ' + str(t), linewidth=2.0)
        if t not in beta_mask:
            dr_plot.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1), 'Beta', label='Beta '+str(t), linewidth=2.0)
            dr_plot.add_trajectory(drb60_int, t, 'plot', beta_palette[idx], (1, 1), 'Beta', label='Beta ' + str(t), linewidth=2.0)
            dr_plot.add_trajectory(drb60_rec, t, 'plot', beta_palette[idx], (2, 1), 'Beta', label='Beta ' + str(t), linewidth=2.0)

    dr_plot.axes[0][0].set_title(r"Baseline")
    dr_plot.axes[0][1].set_title(r"Baseline")
    dr_plot.axes[1][0].set_title(r"Internalization Rate Increased by {}".format(internalization_sf))
    dr_plot.axes[1][1].set_title(r"Internalization Rate Increased by {}".format(internalization_sf))
    dr_plot.axes[2][0].set_title(r"R2 Recycling Rate x {}".format(recycle_sf))
    dr_plot.axes[2][1].set_title(r"R2 Recycling Rate x {}".format(recycle_sf))

    dr_plot.show_figure(save_flag=False)

    # ----------------------------------
    # Time course plot
    # ----------------------------------
    Mixed_Model.reset_parameters()
    Mixed_Model.set_parameters(
        {'R2': 2300 * 2.5,
         'R1': 1800 * 1.8, 'k_d4': 0.06, 'kint_b': 0.0003,
         'kpu': 0.0028,
         'krec_b1': 0.001, 'krec_b2': 0.01,
         'k_a1': 4.98E-14, 'k_a2': 8.30e-13 * 4, 'kSOCSon': 0.9e-8,
         'ka1': 3.321155762205247e-14 * 0.3, 'ka2': 4.98173364330787e-13 * 0.3, 'kd4': 1.0, 'kd3': 0.001,
         'kint_a': 0.0014, 'krec_a1': 9e-03, 'krec_a2': 0.05})

   # Simulate time courses
    alpha_time_courses = []
    alpha_time_courses_int = []
    alpha_time_courses_rec = []
    alpha_doses = [10, 90, 600, 4000, 8000]

    beta_time_courses = []
    beta_time_courses_int = []
    beta_time_courses_rec = []
    beta_doses = [10, 90, 600, 2000, 11000]

    # Baseline
    for d in alpha_doses:
        alpha_time_courses.append(Mixed_Model.timecourse(list(linspace(0, 60, 30)), 'TotalpSTAT',
                                                         {'Ia': d * 6.022E23 * 1E-5 * 1E-12, 'Ib': 0},
                                                         return_type='dataframe', dataframe_labels=['Alpha', d]))
    for d in beta_doses:
        beta_time_courses.append(Mixed_Model.timecourse(list(linspace(0, 60, 30)), 'TotalpSTAT',
                                                        {'Ib': d * 6.022E23 * 1E-5 * 1E-12, 'Ia': 0},
                                                        return_type='dataframe', dataframe_labels=['Beta', d]))
    # Increase internalization
    Mixed_Model.set_parameters({'kint_b': 0.0003 * internalization_sf, 'kint_a': 0.0014 * internalization_sf, 'kdeg':0.8})
    for d in alpha_doses:
        alpha_time_courses_int.append(Mixed_Model.timecourse(list(linspace(0, 60, 30)), 'TotalpSTAT',
                                                         {'Ia': d * 6.022E23 * 1E-5 * 1E-12, 'Ib': 0},
                                                         return_type='dataframe', dataframe_labels=['Alpha', d]))
    for d in beta_doses:
        beta_time_courses_int.append(Mixed_Model.timecourse(list(linspace(0, 60, 30)), 'TotalpSTAT',
                                                        {'Ib': d * 6.022E23 * 1E-5 * 1E-12, 'Ia': 0},
                                                        return_type='dataframe', dataframe_labels=['Beta', d]))
    # Alter recycling
    Mixed_Model.set_parameters({'krec_a1': 9e-03 * recycle_sf, 'krec_b1': 0.001 * recycle_sf})
    for d in alpha_doses:
        alpha_time_courses_rec.append(Mixed_Model.timecourse(list(linspace(0, 60, 30)), 'TotalpSTAT',
                                                         {'Ia': d * 6.022E23 * 1E-5 * 1E-12, 'Ib': 0},
                                                         return_type='dataframe', dataframe_labels=['Alpha', d]))
    for d in beta_doses:
        beta_time_courses_rec.append(Mixed_Model.timecourse(list(linspace(0, 60, 30)), 'TotalpSTAT',
                                                        {'Ib': d * 6.022E23 * 1E-5 * 1E-12, 'Ia': 0},
                                                        return_type='dataframe', dataframe_labels=['Beta', d]))

    # Scale simulations
    for i in range(30):
        for j in range(len(alpha_doses)):
            alpha_time_courses[j].loc['Alpha'].iloc[:, i] = alpha_time_courses[j].loc['Alpha'].iloc[:, i].apply(scale_data)
            alpha_time_courses_int[j].loc['Alpha'].iloc[:, i] = alpha_time_courses_int[j].loc['Alpha'].iloc[:, i].apply(scale_data)
            alpha_time_courses_rec[j].loc['Alpha'].iloc[:, i] = alpha_time_courses_rec[j].loc['Alpha'].iloc[:, i].apply(scale_data)

        for j in range(len(beta_doses)):
            beta_time_courses[j].loc['Beta'].iloc[:, i] = beta_time_courses[j].loc['Beta'].iloc[:, i].apply(scale_data)
            beta_time_courses_int[j].loc['Beta'].iloc[:, i] = beta_time_courses_int[j].loc['Beta'].iloc[:, i].apply(scale_data)
            beta_time_courses_rec[j].loc['Beta'].iloc[:, i] = beta_time_courses_rec[j].loc['Beta'].iloc[:, i].apply(scale_data)

    # Turn into IfnData objects
    alpha_IfnData_objects = []
    beta_IfnData_objects = []
    alpha_IfnData_objects_int = []
    beta_IfnData_objects_int = []
    alpha_IfnData_objects_rec = []
    beta_IfnData_objects_rec = []

    for j in range(len(alpha_doses)):
        alpha_IfnData_objects.append(IfnData('custom', df=alpha_time_courses[j], conditions={'Alpha': {'Ib': 0}}))
        alpha_IfnData_objects_int.append(IfnData('custom', df=alpha_time_courses_int[j], conditions={'Alpha': {'Ib': 0}}))
        alpha_IfnData_objects_rec.append(IfnData('custom', df=alpha_time_courses_rec[j], conditions={'Alpha': {'Ib': 0}}))

    for j in range(len(beta_doses)):
        beta_IfnData_objects.append(IfnData('custom', df=beta_time_courses[j], conditions={'Beta': {'Ia': 0}}))
        beta_IfnData_objects_int.append(IfnData('custom', df=beta_time_courses_int[j], conditions={'Beta': {'Ia': 0}}))
        beta_IfnData_objects_rec.append(IfnData('custom', df=beta_time_courses_rec[j], conditions={'Beta': {'Ia': 0}}))

    # Generate plot
    tc_plot = TimecoursePlot((3, 2))

    # Add fits
    for j, dose in enumerate(alpha_doses):
        if dose not in alpha_mask:
            tc_plot.add_trajectory(alpha_IfnData_objects[j], 'plot', alpha_palette[j], (0, 0), label='Alpha ' + str(dose), linewidth=2.0)
            tc_plot.add_trajectory(alpha_IfnData_objects_int[j], 'plot', alpha_palette[j], (1, 0), label='Alpha ' + str(dose), linewidth=2.0)
            tc_plot.add_trajectory(alpha_IfnData_objects_rec[j], 'plot', alpha_palette[j], (2, 0), label='Alpha ' + str(dose), linewidth=2.0)

    for j, dose in enumerate(beta_doses):
        if dose not in beta_mask:
            tc_plot.add_trajectory(beta_IfnData_objects[j], 'plot', beta_palette[j], (0, 1), label='Beta ' + str(dose), linewidth=2.0)
            tc_plot.add_trajectory(beta_IfnData_objects_int[j], 'plot', beta_palette[j], (1, 1), label='Beta ' + str(dose), linewidth=2.0)
            tc_plot.add_trajectory(beta_IfnData_objects_rec[j], 'plot', beta_palette[j], (2, 1), label='Beta ' + str(dose), linewidth=2.0)

    tc_plot.axes[0][0].set_title(r"Baseline")
    tc_plot.axes[0][1].set_title(r"Baseline")
    tc_plot.axes[1][0].set_title(r"Internalization Rate Increased by {}".format(internalization_sf))
    tc_plot.axes[1][1].set_title(r"Internalization Rate Increased by {}".format(internalization_sf))
    tc_plot.axes[2][0].set_title(r"R2 Recycling Rate x {}".format(recycle_sf))
    tc_plot.axes[2][1].set_title(r"R2 Recycling Rate x {}".format(recycle_sf))

    tc_plot.show_figure()
