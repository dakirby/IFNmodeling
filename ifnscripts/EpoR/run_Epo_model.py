from ifnclass.ifndata import IfnData, DataAlignment
from ifnclass.ifnmodel import IfnModel
from numpy import linspace, logspace, transpose
import numpy as np
import seaborn as sns
from ifnclass.ifnplot import Trajectory, DoseresponsePlot


if __name__ == '__main__':
    # --------------------
    # Import data
    #---------------------
    IfnData("20190108_pSTAT1_IFN_Bcell")
    # --------------------
    # Set up Model
    # --------------------
    model = IfnModel('Epo_model')

    # ---------------------------------
    # Make theory dose response curves
    # ---------------------------------
    # Make predictions
    times = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0] # min
    Epo_doses = np.array(np.logspace(-2,3,30)) # pM

    dr_Epo = IfnData('custom',
                     df=model.doseresponse(times, 'T_Epo', 'Epo_IC', Epo_doses,
                                           parameters={'EMP1_IC': 0, 'EMP33_IC': 0}, return_type='dataframe',
                                           dataframe_labels='T_Epo'),
                     conditions={})

    # -------------------------------
    # Plot model dose response curves
    # -------------------------------
    alpha_palette = sns.color_palette("deep", 6)
    beta_palette = sns.color_palette("deep", 6)

    new_fit = DoseresponsePlot((1, 1))

    alpha_mask = [7.5]
    beta_mask = [7.5]
    # Add fits
    for idx, t in enumerate(times):
        if t not in alpha_mask:
            new_fit.add_trajectory(dr_Epo, t, 'plot', alpha_palette[idx], (0, 0), 'T_Epo', label=str(t)+' min',
                                   linewidth=2)
    new_fit.show_figure()