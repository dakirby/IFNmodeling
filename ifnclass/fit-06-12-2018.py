from ifndata import IfnData
from ifnmodel import IfnModel
from ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace
import seaborn as sns


if __name__ == '__main__':
    newdata = IfnData("20181113_B6_IFNs_Dose_Response_Bcells")
    testModel = IfnModel('Mixed_IFN_ppCompatible')

    dra = testModel.doseresponse([0, 5, 15, 30, 60], 'TotalpSTAT', 'Ia', list(logspace(-3, 4)),
                                parameters={'Ib':0}, return_type='dataframe', dataframe_labels='Alpha')

    smooth_plot = DoseresponsePlot((2, 2))

    alpha_palette = sns.color_palette("Reds", 6)
    beta_palette = sns.color_palette("Greens", 6)

    for idx, t in enumerate([2.5, 5, 7.5, 10, 20, 60]):
        #smooth_plot.add_trajectory(newdata, t, 'plot', alpha_palette[idx], (0, 0), 'Alpha', dn=1)
        smooth_plot.add_trajectory(newdata, t, 'scatter', alpha_palette[idx], (0, 0), 'Alpha', dn=1)
        #smooth_plot.add_trajectory(newdata, t, 'plot', beta_palette[idx], (0, 1), 'Beta', dn=1)
        smooth_plot.add_trajectory(newdata, t, 'scatter', beta_palette[idx], (0, 1), 'Beta', dn=1)

    testtraj = smooth_plot.show_figure()