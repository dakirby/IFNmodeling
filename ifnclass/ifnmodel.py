from collections.__init__ import OrderedDict
from typing import Any

from pysb.export import export
from collections import OrderedDict
import copy

class IfnModel:
    """
    Documentation - An IfnModel object is the primary object for modelling
    experimental IFN dose-response or timecourse data. This is the expected
    model object used for plotting and fitting within the IFNmodeling module.

    Parameters
    ----------
    name : string
        The name of the ifnmodels model file written using PySB, including
        file extension.
    Attributes
    -------
    name : string
        The filename used to find source files for this IfnData instance
    model : PySB standalone python model
        An instance of the python Model class, specific for this model
    parameters : ordered dict
        A dictionary with all the Model parameters. The dictionary is
        ordered to match the order in the PySB model.
    default_parameters : ordered dict
        A dictionary containing all the original imported Model parameters
    Methods
    -------
    build_model -> Model = PySB model instance
    """

    # Initializer / Instance Attributes
    def __init__(self, name):
        self.name = name
        self.model = self.build_model(self.name)
        self.parameters = self.build_parameters(self.model)
        self.default_parameters = copy.deepcopy(self.parameters)

    # Instance methods
    def build_model(self, name):
        model_code = __import__('ifnmodels.'+name, fromlist=['ifnmodels'])
        py_output = export(model_code.model, 'python')
        with open("ODE_system.py", 'w') as f:
            f.write(py_output)
        import ODE_system
        model_obj = ODE_system.Model()
        return model_obj

    def build_parameters(self, pysb_model):
        parameter_dict = OrderedDict({})
        for p in pysb_model.parameters:
            parameter_dict.update({p[0]: [1]})
        return parameter_dict

    def set_parameters(self, new_parameters: dict):
        self.parameters.update(new_parameters)

testModel = IfnModel('IFN_alpha_altSOCS_ppCompatible')
