from ifndata import IfnData
from ifnmodel import IfnModel
from ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace


if __name__ == '__main__':
    newdata = IfnData("20181113_B6_IFNs_Dose_Response_Bcells")
    testModel = IfnModel('Mixed_IFN_ppCompatible')

    dra = testModel.doseresponse([0, 5, 15, 30, 60], 'TotalpSTAT', 'Ia', list(logspace(-3, 4)),
                                parameters={'Ib':0}, return_type='dataframe', dataframe_labels='Alpha')

    smooth_plot = DoseresponsePlot((2, 2))

    alpha_indices = [not bool(el) for el in list(newdata.data_set.index.labels[0])]
    beta_indices = [not el for el in alpha_indices]

    smooth_plot.add_trajectory(newdata.data_set.iloc[alpha_indices], 2.5, 'plot', 'r', (0, 0), 'Alpha', dn=1)
    smooth_plot.add_trajectory(newdata.data_set.iloc[beta_indices], 15, 'plot', 'g', (0, 0), 'Beta', dn=1)
    testtraj = smooth_plot.show_figure()