from numpy import multiply
import matplotlib.pyplot as plt
from ifndata import IfnData
from ifnmodel import IfnModel
from numpy import linspace, logspace, float64, divide

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
        self.observable = kwargs.get('observable', None)
        self.dose_norm = kwargs.get('dose_norm', 1)

    def t(self):
        return [el for el in self.data.columns if (type(el) == int or type(el) == float or isinstance(el, float64))]

    def y(self):
        idx = self.t()
        return self.data[idx].values[0]

    def d(self):
        return divide(self.data.loc[self.observable].index.values, self.dose_norm)

    def z(self):
        return self.data.loc[self.observable][:][str(self.timeslice)].values


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
        if type(self.axes) == list:
            for ax in self.axes:
                ax.set(xscale='log', yscale='linear')
        else:
            self.axes.set(xscale='log', yscale='linear')
        self.trajectories = []
        self.subplot_indices = []

    # Instance methods
    def add_trajectory(self, data: IfnData, time, plot_type: str, line_style, subplot_idx: tuple,
                       observable_species: str, label='', dn: float=1.):
        t = Trajectory(data, plot_type, line_style, label=label, timeslice = time, observable=observable_species, dose_norm=dn)
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
                x = trajectory.d()
                z = [el[0] for el in trajectory.z()]
                ax.plot(x, z, trajectory.line_style, label=trajectory.label)
            elif trajectory.plot_type == 'scatter':
                x = trajectory.data.loc[:, 'Dose (pM)'].values
                y = [el[0] for el in trajectory.data.loc[:, trajectory.metadata['time']].values]
                ax.scatter(x, y, c=trajectory.line_style[0],
                           marker=trajectory.line_style[1], label=trajectory.label)
            elif trajectory.plot_type == 'errorbar':
                x = trajectory.data.loc[:, 'Dose (pM)'].values
                y = [el[0] for el in trajectory.data.loc[:, trajectory.metadata['time']].values]
                sigmas = [el[1] for el in trajectory.data.loc[:, trajectory.metadata['time']].values]
                ax.errorbar(x, y, yerr=sigmas, fmt=trajectory.line_style, label=trajectory.label)
        plt.show()
        return self.fig


if __name__ == '__main__':
    testClass = IfnData("Experimental_Data")
    testModel = IfnModel('IFN_alpha_altSOCS_ppCompatible')
    tc = testModel.timecourse(list(linspace(0,30)), 'TotalpSTAT', return_type='dataframe', dataframe_labels=['Alpha', 1E-9])
    testplot = TimecoursePlot((1,1))
    testplot.add_trajectory(tc, 'plot', 'r', (0,0))
    testplot.show_figure()

    dr = testModel.doseresponse([0, 5, 15, 30], 'TotalpSTAT', 'I', multiply(list(logspace(-11,-2)), 6.022E18),
                                return_type='dataframe', dataframe_labels='Alpha')
    testplot = DoseresponsePlot((1,1))
    testplot.add_trajectory(dr, 5, 'plot', 'r', (0,0), 'Alpha', dn=6.022E23*1E-9*1E-5)
    testtraj = testplot.show_figure()
