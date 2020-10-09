# IFNmodeling
A collection of Python scripts for modelling interferon signaling.

This is primarily my working directory for exploring models of immune cell signaling. 
- Class files and additional documentation on class methods can be found in /ifnclass
- Scripts to produce figures for upcoming publication are in /ifnscripts with file names of the form figure_X.py where X is the figure from the draft paper

## Basic usage:

The `IfnModel` class is used to construct an ODE system from a model file written with the package PySB.
The class has two major methods, `IfnModel.timecourse()` and `IfnModel.doseresponse()`. These are the primary methods to simulate a model and fit simulations to experimental data.

The `IfnData` class provides a standardized interface to compare experimental data with simulations. 
Experimental data can be loaded directly into an `IfnData` instance from /ifndatabase while the Pandas dataframe output from an `IfnModel` instance can be turned into an `IfnData`
 instance using `IfnData('custom', df=<simulation output>)`. Several other useful methods in the IfnData class are `IfnData.drop_sigmas()`, `IfnData.get_ec50s()`, and `IfnData.get_max_responses()`.

The `DataAlignment` class provides a convenient way to combine several `IfnData` instances from biological replicates of an experiment.

The main advantage of standardizing the interface for all data and simulations is that this allows high quality plots involving both model and experiments to be reliably built.
To this end, the `TimecoursePlot` and `DoseresponsePlot` classes are extremely handy. Basic usage requires each trajectory in your plot to be added from an `IfnData` instance 
using the `add_trajectory()` method from either plotting class. Generating plots is then as simple as calling, for example, `DoseresponsePlot.show_figure()`.

## Minimal Example
A script which uses most of the above features can be found at /ifnscripts/minimal_working_example.py 
