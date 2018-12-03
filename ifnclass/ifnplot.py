from numpy import multiply
import matplotlib.pyplot as plt
from ifndata import IfnData
from ifnmodel import IfnModel

class Trajectory:
    """
    Documentation - A Trajectory object is an augmented IfnData object which simply includes metainformation on the
                    desired plotting style
    Parameters
    ----------
    data (IfnData): the data to be plotted
    plot_type (string): the type of matplotlib plotting function to call - can be one of 'plot', 'scatter', 'errorbar'
    line_style (string): the argument to pass to the plotting function for colour and style (eg. line_style = 'k--')
    label (string): the label to use in the plot legend for this trajectory; default is None, in which case default
                    plotting choices will be used
    """
    # Initializer / Instance Attributes
    def __init__(self, data: IfnData, plot_type: str, line_style, label=''):
        self.data = data
        self.plot_type = plot_type
        self.line_style = line_style
        self.label = label

    def x(self):
        return self.data.columns[2:]

    def y(self):
        return self.data.loc[self.data.columns[2:]]

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
        self.subplot_indices.append(subplot_idx)

    def remove_trajectory(self, index):
        del self.trajectories[index]
        del self.subplot_indices[index]

    def show_figure(self):
        for trajectory_idx in range(len(self.trajectories)):
            trajectory = self.trajectories[t]
            subplot_location = self.subplot_indices[t]
            if trajectory.plot_type == 'plot':
                self.axes[trajectory][subplot_location[0]][subplot_location[1]].plot(trajectory.x(), trajectory.y(),
                                                                        trajectory.line_style, label=trajectory.label)
        self.fig.show()
        return self.fig



if __name__ == '__main__':
    testClass = IfnData("Experimental_Data")
    print(testClass.data_set)
    testModel = IfnModel('IFN_alpha_altSOCS_ppCompatible')
    #dr = testModel.doseresponse([0, 5, 15, 30], ['T', 'TotalpSTAT'], 'I', multiply([1E-9, 1E-8, 1E-7], 6.022E18),
    #                            return_type='dataframe', dataframe_labels='Alpha')
    tc = testModel.timecourse([0, 5, 15, 30], 'TotalpSTAT', return_type='dataframe', dataframe_labels=['Alpha',1E-9])
    testplot = TimecoursePlot((1,1))
    testplot.add_trajectory(tc, 'plot', 'r')

