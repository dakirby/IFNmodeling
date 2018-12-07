from numpy import ndarray, shape
import matplotlib.pyplot as plt
from ifndata import IfnData
from ifnmodel import IfnModel
from numpy import linspace, logspace, float64, divide
import time

class Trajectory:
    """
    Documentation - A Trajectory object is an augmented IfnData object which simply includes metainformation on the
                    desired plotting style
    Parameters
    ----------
    data (IfnData): the data to be plotted
    plot_type (string): the type of matplotlib plotting function to call - can be one of 'plot', 'scatter', 'errorbar'
    line_style (string): the argument to pass to the plotting function for colour and style (eg. line_style = 'k--')
                         if plot_type == 'scatter' then line_style can be up to two characters, one for color and the
                         other for marker shape. line_style[0] will be passed as colour and line_style[1] as marker
                         shape in scatter()
    label (string): the label to use in the plot legend for this trajectory; default is None, in which case default
                    plotting choices will be used
    Methods
    -------
    t(): returns times for a TimecoursePlot
    y(): returns responses for a TimecoursePlot
    d(): return the doses for a DoseresponsePlot
    z(): returns the responses for a DoseresponsePlot
    """

    # Initializer / Instance Attributes
    def __init__(self, data: IfnData, plot_type: str, line_style, label='', **kwargs):
        self.data = data
        self.plot_type = plot_type
        self.line_style = line_style
        self.label = label
        self.timeslice = kwargs.get('timeslice', None)
        self.dose_species = kwargs.get('dose_species', None)
        self.dose_norm = kwargs.get('dose_norm', 1)

    def t(self):  # times
        if self.timeslice is None:
            return [el for el in self.data.data_set.columns if
                    (type(el) == int or type(el) == float or isinstance(el, float64))]
        else:
            return [self.timeslice]

    def y(self):  # timecourse response values
        idx = self.t()
        return self.data.data_set[idx].values[0]

    def d(self):  # doses
        if self.timeslice is None:
            return divide(self.data.data_set.loc[self.dose_species].index.values, self.dose_norm)
        else:
            return divide(self.data.data_set.loc[self.dose_species].index.values, self.dose_norm)

    def z(self):  # dose-response response values
        if self.timeslice is None:
            return self.data.data_set.xs(self.dose_species).values
        else:
            try:
                return self.data.data_set.xs(self.dose_species).loc[:, self.timeslice].values
            except KeyError:
                try:
                    temp = [str(el) for el in self.timeslice]
                    return self.data.data_set.xs(self.dose_species).loc[:, temp].values
                except KeyError:
                    print(self.data.data_set.xs(self.dose_species).columns.values)
                    return self.data.data_set.xs(self.dose_species).loc[:, [float(el) for el in self.timeslice][0]].values


class TimecoursePlot:
    """
    Documentation - A TimecoursePlot holds IfnModels and IfnData instances which are to be plotted in a
                    Matplotlib figure.

    Parameters
    ----------
    shape (tuple): tuple of ints for nrows and ncols in the figure

    Attributes
    ----------
    nrows (int): the number of subplot rows in the figure
    ncols (int): the number of subplot columns in the the figure
    fig (matplotlib figure): the figure to plot
    axes (matplotlib axes): the axes object to plot
    trajectories (list): a list of Trajectory objects describing trajectories, experimental or simulated
    subplot_indices (list): a list of subplot locations
    Methods
    -------
    add_trajectory(self, data: IfnData, plot_type: str, line_style, subplot_idx, label='')
        Builds a Trajectory instance and adds it to the TimecoursePlot object's list of trajectories. Remembers which
        subplot to put this trajectory in, given by subplot_idx = (row, col)
            - row and col in subplot_idx are 0-indexed

    remove_trajectory(index: int)
        Removes a trajectory from the TimecoursePlot instance, as indicated by its index in self.trajectories list
    """

    # Initializer / Instance Attributes
    def __init__(self, shape):
        self.nrows = shape[0]
        self.ncols = shape[1]
        self.fig, self.axes = plt.subplots(nrows=self.nrows, ncols=self.ncols)
        self.trajectories = []
        self.subplot_indices = []

    # Instance methods
    def add_trajectory(self, data: IfnData, plot_type: str, line_style, subplot_idx: tuple, label=''):
        t = Trajectory(data, plot_type, line_style, label=label)
        self.trajectories.append(t)
        if self.nrows == 1 and self.ncols == 1:
            self.subplot_indices.append((None, None))
        elif self.nrows == 1:
            self.subplot_indices.append((None, subplot_idx[1]))
        elif self.ncols == 1:
            self.subplot_indices.append((subplot_idx[0], None))
        else:
            self.subplot_indices.append(subplot_idx)

    def get_axis_object(self, idx):
        if idx == (None, None):
            return self.axes
        elif idx[0] == None:
            return self.axes[idx[1]]
        elif idx[1] == None:
            return self.axes[idx[0]]
        else:
            return self.axes[idx[0]][idx[1]]

    def remove_trajectory(self, index):
        del self.trajectories[index]
        del self.subplot_indices[index]

    def show_figure(self):
        for trajectory_idx in range(len(self.trajectories)):
            trajectory = self.trajectories[trajectory_idx]
            plt_idx = self.subplot_indices[trajectory_idx]
            ax = self.get_axis_object(plt_idx)
            if trajectory.plot_type == 'plot':
                ax.plot(trajectory.t(), [el[0] for el in trajectory.y()], trajectory.line_style, label=trajectory.label)
            elif trajectory.plot_type == 'scatter':
                ax.scatter(trajectory.t(), [el[0] for el in trajectory.y()], c=trajectory.line_style[0],
                           marker=trajectory.line_style[1], label=trajectory.label)
            elif trajectory.plot_type == 'errorbar':
                ax.errorbar(trajectory.t(), [el[0] for el in trajectory.y()], yerr=[el[1] for el in trajectory.y()],
                            fmt=trajectory.line_style, label=trajectory.label)
        plt.show()
        return self.fig


class DoseresponsePlot:
    """
    Documentation - A DoseresponsePlot holds IfnModels and IfnData instances which are to be plotted in a
                    Matplotlib figure.

    Parameters
    ----------
    shape (tuple): tuple of ints for nrows and ncols in the figure

    Attributes
    ----------
    nrows (int): the number of subplot rows in the figure
    ncols (int): the number of subplot columns in the the figure
    fig (matplotlib figure): the figure to plot
    axes (matplotlib axes): the axes object to plot
    trajectories (list): a list of Trajectory objects describing trajectories, experimental or simulated
    subplot_indices (list): a list of subplot locations
    Methods
    -------
    add_trajectory(self, data: IfnData, time: float, plot_type: str, line_style, subplot_idx, observable_species, label='')
        Builds a Trajectory instance and adds it to the DoseresponsePlot object's list of trajectories. Remembers which
        subplot to put this trajectory in, given by subplot_idx = (row, col). Time for dose-response curve must be given
            - row and col in subplot_idx are 0-indexed

    remove_trajectory(index: int)
        Removes a trajectory from the DoseresponsePlot instance, as indicated by its index in self.trajectories list
    """

    # Initializer / Instance Attributes
    def __init__(self, shape):
        self.nrows = shape[0]
        self.ncols = shape[1]
        self.fig, self.axes = plt.subplots(nrows=self.nrows, ncols=self.ncols)
        if self.nrows > 1 and self.ncols > 1:
            for row in range(self.nrows):
                for column in range(self.ncols):
                    self.axes[row][column].set(xscale='log', yscale='linear')
                    if row == self.nrows - 1:
                        self.axes[row][column].set_xlabel('Dose (pM)')
                    if column == 0:
                        self.axes[row][column].set_ylabel('Response')
        elif self.ncols > 1:
            for column in range(self.ncols):
                self.axes[column].set(xscale='log', yscale='linear')
                self.axes[column].set_xlabel('Dose (pM)')
                if column == 0:
                    self.axes[column].set_ylabel('Response')

        else:
            self.axes.set(xscale='log', yscale='linear')
            self.axes.set_xlabel('Dose (pM)')
            self.axes.set_ylabel('Response')
        self.trajectories = []
        self.subplot_indices = []

    # Instance methods
    def add_trajectory(self, data: IfnData, time, plot_type: str, line_style, subplot_idx: tuple,
                       dose_species: str, label='', dn: float = 1.):
        t = Trajectory(data, plot_type, line_style, label=label, timeslice=time, dose_species=dose_species,
                       dose_norm=dn)
        self.trajectories.append(t)
        if self.nrows == 1 and self.ncols == 1:
            self.subplot_indices.append((None, None))
        elif self.nrows == 1:
            self.subplot_indices.append((None, subplot_idx[1]))
        elif self.ncols == 1:
            self.subplot_indices.append((subplot_idx[0], None))
        else:
            self.subplot_indices.append(subplot_idx)

    def get_axis_object(self, idx):
        if idx == (None, None):
            return self.axes
        elif idx[0] == None:
            return self.axes[idx[1]]
        elif idx[1] == None:
            return self.axes[idx[0]]
        else:
            return self.axes[idx[0]][idx[1]]

    def remove_trajectory(self, index):
        del self.trajectories[index]
        del self.subplot_indices[index]

    def show_figure(self, show_flag=True, save_flag=False):
        for trajectory_idx in range(len(self.trajectories)):
            trajectory = self.trajectories[trajectory_idx]
            plt_idx = self.subplot_indices[trajectory_idx]
            ax = self.get_axis_object(plt_idx)
            if trajectory.plot_type == 'plot':
                x = trajectory.d()
                z = [el[0] for el in trajectory.z()]
                if x[0] == 0:
                    x = x[1:]
                    z = z[1:]
                if type(trajectory.line_style) == str:
                    ax.plot(x, z, trajectory.line_style, label=trajectory.label)
                else:
                    ax.plot(x, z, c=trajectory.line_style, label=trajectory.label)
            elif trajectory.plot_type == 'scatter':
                x = trajectory.d()
                z = [el[0] for el in trajectory.z()]
                if x[0] == 0:
                    x = x[1:]
                    z = z[1:]
                if len(trajectory.line_style) == 2:
                    ax.scatter(x, z, c=trajectory.line_style[0],
                               marker=trajectory.line_style[1], label=trajectory.label)
                elif len(trajectory.line_style) == 1:
                    ax.scatter(x, z, c=trajectory.line_style[0], label=trajectory.label)
                else:
                    try:
                        ax.scatter(x, z, c=[trajectory.line_style], label=trajectory.label)
                    except:
                        print("Could not interpret line style")
                        raise
            elif trajectory.plot_type == 'errorbar':
                x = trajectory.d()
                z = [el[0] for el in trajectory.z()]
                if x[0] == 0:
                    x = x[1:]
                    z = z[1:]
                sigmas = [el[1] for el in trajectory.z()]
                ax.errorbar(x, z, yerr=sigmas, fmt=trajectory.line_style, label=trajectory.label)
        if save_flag:
            plt.savefig('fig{}.pdf'.format(int(time.time())))
        if show_flag:
            plt.show()
        return self.fig, self.axes

    def save_figure(self):
        self.show_figure(show_flag=False, save_flag=True)

if __name__ == '__main__':
    testData = IfnData("20181113_B6_IFNs_Dose_Response_Bcells")
    testModel = IfnModel('Mixed_IFN_ppCompatible')
    # testParameters = {'kpa': 4.686e-05, 'kSOCSon': 2.687e-06, 'kd4': 0.236, 'k_d4': 0.2809, 'R1': 108, 'R2': 678}
    best_fit_old20min = {'kpa': 1e-07, 'kSOCSon': 1e-07, 'krec_a1': 0.0001, 'krec_a2': 0.001, 'R1': 1272, 'R2': 1272, 'krec_b1': 1.0e-05, 'krec_b2': 0.0001}

    testModel.set_parameters(best_fit_old20min)
    tc = testModel.timecourse(list(linspace(0, 30)), 'TotalpSTAT', return_type='dataframe',
                              dataframe_labels=['Alpha', 1E-9])
    tcIfnData = IfnData('custom', df=tc, conditions={'Alpha': {'Ib': 0}})

    #testplot = TimecoursePlot((1, 1))
    #testplot.add_trajectory(tc, 'plot', 'r', (0, 0))
    #testplot.show_figure()

    dra = testModel.doseresponse([0, 2.5, 5, 15, 30, 60], 'TotalpSTAT', 'Ia', list(logspace(-3, 4)),
                                 parameters={'Ib': 0}, return_type='dataframe', dataframe_labels='Alpha')

    drb = testModel.doseresponse([0, 2.5, 5, 15, 30, 60], 'TotalpSTAT', 'Ib', list(logspace(-3, 4)),
                                 parameters={'Ia': 0}, return_type='dataframe', dataframe_labels='Beta')
    draIfnData = IfnData('custom', df=dra, conditions={'Alpha': {'Ib': 0}})
    drbIfnData = IfnData('custom', df=drb, conditions={'Beta': {'Ia': 0}})

    testplot = DoseresponsePlot((1, 1))
    testplot.add_trajectory(draIfnData, '15', 'plot', 'r', (0, 0), 'Alpha', dn=1)
    testplot.add_trajectory(drbIfnData, '15', 'plot', 'g', (0, 0), 'Beta', dn=1)
    testtraj = testplot.show_figure()
