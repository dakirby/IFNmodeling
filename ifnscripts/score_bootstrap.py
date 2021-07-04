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

def score_params(params):
    # --------------------
    # Set up Model
    # --------------------
    Mixed_Model, DR_method = lm.load_model()
    scale_factor, DR_KWARGS = lm.SCALE_FACTOR, lm.DR_KWARGS
    Mixed_Model.set_parameters(params)

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

    # rmse = RMSE(mean_data.data_set.values, sim_df.data_set.values)
    mae = MAE(mean_data.data_set.values, sim_df.data_set.values)
    return mae

if __name__ == '__main__':
    # From PyDREAM_10-06-2021_BOOTSTRAP/Batch/Mixed_IFN_ML_params.txt
    p_list = [{'kpa': 1.3652412309889275e-07, 'kSOCSon': 6.497004276614423e-06, 'R1_mu*': 3792.944125800926, 'R1_std*': 0.20751329309214633, 'R2_mu*': 1411.1529712294496, 'R2_std*': 0.2005623553441309, 'kint_a': 0.00017828025840610948, 'kint_b': 0.00038537833426453133, 'krec_a1': 0.0009201191282383926, 'krec_a2': 0.014280261624268589, 'krec_b1': 0.0005251437364197522, 'krec_b2': 0.0037923487326169377},
              {'kpa': 1.5668202279264452e-06, 'kSOCSon': 7.255280543489566e-06, 'R1_mu*': 1414.0807226091174, 'R1_std*': 0.185421809029744, 'R2_mu*': 536.3894113706257, 'R2_std*': 0.2091982168989663, 'kint_a': 0.004354543520398367, 'kint_b': 2.7157652274983732e-05, 'krec_a1': 0.0011878772971791144, 'krec_a2': 0.002542839783901077, 'krec_b1': 0.0001382621398243896, 'krec_b2': 0.0008217522572414162},
              {'kpa': 4.8802953534339715e-06, 'kSOCSon': 7.786820128972969e-07, 'R1_mu*': 353.6364940270348, 'R1_std*': 0.18983945024509274, 'R2_mu*': 720.5110713054098, 'R2_std*': 0.21359491869748534, 'kint_a': 0.00048475147224099653, 'kint_b': 0.0005045158300672412, 'krec_a1': 0.001861824485419296, 'krec_a2': 0.008776056738452264, 'krec_b1': 2.316607046909074e-05, 'krec_b2': 0.0004619439982587077},
              {'kpa': 2.4724566529349485e-06, 'kSOCSon': 3.1311278264790576e-06, 'R1_mu*': 4264.421138808471, 'R1_std*': 0.20845817294573565, 'R2_mu*': 1843.124365951139, 'R2_std*': 0.16706183723995102, 'kint_a': 0.0004320686309297391, 'kint_b': 3.904410615334764e-05, 'krec_a1': 0.0003223921531506014, 'krec_a2': 0.002362241339325204, 'krec_b1': 9.676918767751533e-05, 'krec_b2': 0.0002987866277659682},
              {'kpa': 6.289938841307757e-07, 'kSOCSon': 7.590996156147515e-07, 'R1_mu*': 761.1079345360049, 'R1_std*': 0.20356562725845906, 'R2_mu*': 1039.4955375844238, 'R2_std*': 0.2019756144438121, 'kint_a': 0.001716271574723621, 'kint_b': 0.000256923812651761, 'krec_a1': 0.0010817171778677091, 'krec_a2': 0.002352467552656415, 'krec_b1': 0.00013019679444894886, 'krec_b2': 0.0009240954116252316}]
    # Compute average mean absolute error
    avg_mae = 0
    for p in p_list:
        avg_mae += score_params(p) 
    avg_mae = avg_mae / len(p_list)
    print("Average MAE = ", avg_mae)
