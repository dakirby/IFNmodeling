from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnfit import DualMixedPopulation
from numpy import logspace
import seaborn as sns
import os
from ifnclass.ifnplot import DoseresponsePlot


if __name__ == '__main__':
    # --------------------
    # Set up EC50 figures
    # --------------------
    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)
    data_palette = sns.color_palette("muted", 6)
    marker_shape = ["o", "v", "s", "P", "d", "1", "x", "*"]
    dataset_names = ["20190108", "20190119", "20190121", "20190214"]

    # --------------------
    # Set up Model
    # --------------------
    # Parameters found by stepwise fitting GAB mean data
    # Note: can remove multiplicative factors on all K1, K2, K4 and still get
    # very good fit to data (worst is 5 min beta)
    initial_parameters = {'k_a1': 4.98E-14 * 2, 'k_a2': 8.30e-13 * 2,
                          'k_d4': 0.006 * 3.8,
                          'kpu': 0.00095,
                          'ka2': 4.98e-13 * 2.45, 'kd4': 0.3 * 2.867,
                          'kint_a': 0.000124, 'kint_b': 0.00086,
                          'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005,
                          'krec_b2': 0.05}
    dual_parameters = {'kint_a': 0.00052, 'kSOCSon': 6e-07, 'kint_b': 0.00052,
                       'krec_a1': 0.001, 'krec_a2': 0.1,
                       'krec_b1': 0.005, 'krec_b2': 0.05}
    scale_factor = 1.227

    Mixed_Model = DualMixedPopulation('Mixed_IFN_ppCompatible', 0.8, 0.2)
    Mixed_Model.model_1.set_parameters(initial_parameters)
    Mixed_Model.model_1.set_parameters(dual_parameters)
    Mixed_Model.model_1.set_parameters({'R1': 12000.0, 'R2': 1511.1})
    Mixed_Model.model_2.set_parameters(initial_parameters)
    Mixed_Model.model_2.set_parameters(dual_parameters)
    Mixed_Model.model_2.set_parameters({'R1': 6755.56, 'R2': 1511.1})

    # Make predictions
    times = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
    alpha_doses_20190108 = [0, 10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses_20190108 = [0, 0.2, 6, 20, 60, 200, 600, 2000]

    dradf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia',
                                            list(logspace(1, 5.2)),
                                            parameters={'Ib': 0},
                                            sf=scale_factor)
    drbdf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ib',
                                            list(logspace(-1, 4)),
                                            parameters={'Ia': 0},
                                            sf=scale_factor)

    dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
    drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

    # ----------------------------------
    # Get all data set EC50 time courses
    # ----------------------------------
    newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
    newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
    newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
    newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")

    # 20190108
    ec50_20190108 = newdata_1.get_ec50s()

    # 20190119
    ec50_20190119 = newdata_2.get_ec50s()

    # 20190121
    ec50_20190121 = newdata_3.get_ec50s()

    # 20190214
    ec50_20190214 = newdata_4.get_ec50s()

    # Aligned data, to get scale factors for each data set
    alignment = DataAlignment()
    alignment.add_data([newdata_4, newdata_3, newdata_2, newdata_1])
    alignment.align()
    alignment.get_scaled_data()
    mean_data = alignment.summarize_data()

    # -------------------------------
    # Plot model dose response curves
    # -------------------------------
    alpha_palette = sns.color_palette("rocket_r", 6)
    beta_palette = sns.color_palette("rocket_r", 6)

    new_fit = DoseresponsePlot((1, 2))
    new_fit.fig.set_size_inches(14.75, 8)
    new_fit.axes[0].set_title(r'IFN$\alpha$')
    new_fit.axes[1].set_title(r'IFN$\beta$')

    t_mask = [2.5, 7.5, 10.0]
    # Add fits
    for idx, t in enumerate(times):
        if t not in t_mask:
            new_fit.add_trajectory(dra60, t, 'plot', alpha_palette[idx],
                                   (0, 0), 'Alpha', label=str(t)+' min',
                                   linewidth=2)
            new_fit.add_trajectory(mean_data, t, 'errorbar', 'o', (0, 0),
                                   'Alpha', color=alpha_palette[idx])
        if t not in t_mask:
            new_fit.add_trajectory(drb60, t, 'plot', beta_palette[idx], (0, 1),
                                   'Beta', label=str(t) + ' min',
                                   linewidth=2)
            new_fit.add_trajectory(mean_data, t, 'errorbar', 'o', (0, 1),
                                   'Beta', color=beta_palette[idx])

    # -------------------------------
    # Jacknife change parameters 20%
    # -------------------------------
    param_names = list(Mixed_Model.model_1.parameters.keys())
    for perturbation in [0.8, 1.2]:
        for pname in param_names:
            for model in [Mixed_Model.model_1, Mixed_Model.model_2]:
                model.set_parameters({pname: model.parameters[pname]*perturbation})
            # Predict
            dradf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia',
                                                    list(logspace(1, 5.2)),
                                                    parameters={'Ib': 0},
                                                    sf=scale_factor)
            drbdf = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ib',
                                                    list(logspace(-1, 4)),
                                                    parameters={'Ia': 0},
                                                    sf=scale_factor)

            dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
            drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})

            # reset parameters
            for model in [Mixed_Model.model_1, Mixed_Model.model_2]:
                model.set_parameters({pname: model.parameters[pname] / perturbation})

            # Plot
            for idx, t in enumerate(times):
                if t not in t_mask:
                    new_fit.add_trajectory(dra60, t, 'plot', alpha_palette[idx],
                                           (0, 0), 'Alpha',
                                           linewidth=1, alpha=0.1)
                if t not in t_mask:
                    new_fit.add_trajectory(drb60, t, 'plot', beta_palette[idx],
                                           (0, 1), 'Beta',
                                           linewidth=1, alpha=0.1)

    new_fit.save_figure(save_dir=os.path.join(os.getcwd(),
                                              'results', 'sensitivity.pdf'))
