# Minimal working Interferon model
The main advantage of using this module over the vanilla installation of PySB is
that for Type I Interferon (IFN) signaling we require that detailed balance be maintained
between the dissociation constants for ternary complex formation, i.e.
$$
K_1 K_3 = K_2 K_4
$$
where $K_i = \frac{k^+_i}{k^-_i}$.

The basic PySB model of Type I IFN signaling is defined in IFN_model.py

The default parameter values are biophysically reasonable choices for parameters.
However, a choice of parameters found to best fit experimental data on primary mouse B cells is provided in runfile.py   

The `IfnModel` class is defined in model_class_file.py
Detailed balance is always checked when changing parameters in the IfnModel class.
Additional functionality is provided by the timecourse() and doseresponse() methods,
which provide convenient formatting of the simulation results and allow for easy
concatenation of many doses and times.
