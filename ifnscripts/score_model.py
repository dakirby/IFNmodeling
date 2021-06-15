from ifnclass.ifndata import IfnData, DataAlignment
from numpy import logspace
import numpy as np
import load_model as lm
import pandas as pd


def RMSE(x, y):
    "Assumes x is the 'true' values to normalize by"
    x_norm = np.divide(x, x)
    y_norm = np.divide(y, x)
    diff = np.subtract(x_norm, y_norm)
    return np.sqrt(np.mean(np.square(diff)))

def MAE(x, y):
    "Assumes x is the 'true' values to normalize by"
    x_norm = np.divide(x, x)
    y_norm = np.divide(y, x)
    diff = np.subtract(x_norm, y_norm)
    return np.mean(np.abs(diff))


if __name__ == '__main__':
    # --------------------
    # Set up Model
    # --------------------
    Mixed_Model, DR_method = lm.load_model()
    scale_factor, DR_KWARGS = lm.SCALE_FACTOR, lm.DR_KWARGS

    # Make predictions
    times = [2.5, 5.0, 7.5, 10.0, 20.0, 60.0]
    alpha_doses = [10, 100, 300, 1000, 3000, 10000, 100000]
    beta_doses = [0.2, 6, 20, 60, 200, 600, 2000]

    dra60 = DR_method(times, 'TotalpSTAT', 'Ia',
                                            alpha_doses,
                                            parameters={'Ib': 0},
                                            sf=scale_factor,
                                            **DR_KWARGS)

    drb60 = DR_method(times, 'TotalpSTAT', 'Ib',
                                            beta_doses,
                                            parameters={'Ia': 0},
                                            sf=scale_factor,
                                            **DR_KWARGS)
    sim_df = IfnData('custom', df=pd.concat((dra60.data_set, drb60.data_set)),
                     conditions={'Alpha': {'Ib': 0}, 'Beta': {'Ia': 0}})

    # --------------------
    # Set up Data
    # --------------------
    newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
    newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
    newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
    newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")

    # Aligned data, to get scale factors for each data set
    alignment = DataAlignment()
    alignment.add_data([newdata_4, newdata_3, newdata_2, newdata_1])
    alignment.align()
    alignment.get_scaled_data()
    mean_data = alignment.summarize_data()

    # --------------------
    # Score model
    # --------------------
    sim_df.drop_sigmas()
    mean_data.drop_sigmas()

    rmse = RMSE(mean_data.data_set.values, sim_df.data_set.values)
    mae = MAE(mean_data.data_set.values, sim_df.data_set.values)
    print("RMSE = ", rmse)
    print("MAE = ", mae)
