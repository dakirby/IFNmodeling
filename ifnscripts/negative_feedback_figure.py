from ifnclass.ifndata import IfnData
from ifnclass.ifnplot import DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import seaborn as sns
import load_model as lm
import copy
import os
import matplotlib.pyplot as plt


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

    scale_factor = 1.5
    DR_KWARGS = {'return_type': 'IfnData'}
    PLOT_KWARGS = {'line_type': 'plot', 'alpha': 1}
    MODEL_TYPE = 'MEDIAN'  # 'SINGLE_CELL'

    # --------------------
    # Set up Model
    # --------------------
    assert MODEL_TYPE in ['MEDIAN', 'SINGLE_CELL']
    Mixed_Model, DR_method = lm.load_model(MODEL_TYPE=MODEL_TYPE)
    Mixed_Model.set_default_parameters(Mixed_Model.get_parameters())

    # --------------------
    # Run Simulations
    # --------------------
    times = [60]
    # Control Dose-Response
    Mixed_Model.set_parameters({'kSOCSon': 0, 'kIntBasal_r1': 0, 'kIntBasal_r2': 0, 'kint_a': 0, 'kint_b': 0})

    dradf = DR_method(times, 'TotalpSTAT', 'Ia', list(logspace(-2, 8)),
                      parameters={'Ib': 0}, return_type='DataFrame', dataframe_labels='Alpha',
                      scale_factor=scale_factor)

    drbdf = DR_method(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 8)),
                      parameters={'Ia': 0}, return_type='DataFrame', dataframe_labels='Beta',
                      scale_factor=scale_factor)

    # Show internalization effects
    Mixed_Model.reset_parameters()
    Mixed_Model.set_parameters({'kSOCSon': 0})
    dradf_int = DR_method(times, 'TotalpSTAT', 'Ia', list(logspace(-2, 8)),
                          parameters={'Ib': 0}, return_type='DataFrame', dataframe_labels='Alpha',
                          scale_factor=scale_factor)
    drbdf_int = DR_method(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 8)),
                          parameters={'Ia': 0}, return_type='DataFrame', dataframe_labels='Beta',
                          scale_factor=scale_factor)

    # Show SOCS effects
    Mixed_Model.reset_parameters()
    Mixed_Model.set_parameters({'kIntBasal_r1': 0, 'kIntBasal_r2': 0, 'kint_a': 0, 'kint_b': 0})
    dradf_SOCS = DR_method(times, 'TotalpSTAT', 'Ia', list(logspace(-2, 8)),
                           parameters={'Ib': 0}, return_type='DataFrame', dataframe_labels='Alpha',
                           scale_factor=scale_factor)
    drbdf_SOCS = DR_method(times, 'TotalpSTAT', 'Ib', list(logspace(-2, 8)),
                           parameters={'Ia': 0}, return_type='DataFrame', dataframe_labels='Beta',
                           scale_factor=scale_factor)

    # Make IfnData objects
    if MODEL_TYPE == 'SINGLE_CELL':
        dra60 = IfnData('custom', df=dradf, conditions={'Alpha': {'Ib': 0}})
        drb60 = IfnData('custom', df=drbdf, conditions={'Beta': {'Ia': 0}})
        dra60_int = IfnData('custom', df=dradf_int, conditions={'Alpha': {'Ib': 0}})
        drb60_int = IfnData('custom', df=drbdf_int, conditions={'Beta': {'Ia': 0}})
        dra60_SOCS = IfnData('custom', df=dradf_SOCS, conditions={'Alpha': {'Ib': 0}})
        drb60_SOCS = IfnData('custom', df=drbdf_SOCS, conditions={'Beta': {'Ia': 0}})
    else:
        dra60 = dradf
        drb60 = drbdf
        dra60_int = dradf_int
        drb60_int = drbdf_int
        dra60_SOCS = dradf_SOCS
        drb60_SOCS = drbdf_SOCS

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
