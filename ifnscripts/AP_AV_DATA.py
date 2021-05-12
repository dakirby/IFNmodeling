import numpy as np
import pandas as pd
import os

AVcsv = pd.read_csv(os.getcwd() + os.sep + '2011_Thomas_IFN_Antiviral_Activity.csv')
APcsv = pd.read_csv(os.getcwd() + os.sep + '2011_Thomas_IFN_Cell_Proliferation.csv')

# No error bars
Thomas2011IFNalpha2AP = APcsv.loc[APcsv['Dose_Species'] == 'IFNa2'].values[:,1:3]

Thomas2011IFNalpha2AV = AVcsv.loc[AVcsv['Dose_Species'] == 'IFNa2'].values[:,1:3]

Thomas2011IFNalpha7AP = APcsv.loc[APcsv['Dose_Species'] == 'IFNa7'].values[:,1:3]

Thomas2011IFNalpha7AV = AVcsv.loc[AVcsv['Dose_Species'] == 'IFNa7'].values[:,1:3]

Thomas2011IFNomegaAP = APcsv.loc[APcsv['Dose_Species'] == 'IFNw'].values[:,1:3]

Thomas2011IFNomegaAV = AVcsv.loc[AVcsv['Dose_Species'] == 'IFNw'].values[:,1:3]

Thomas2011IFNalpha2YNSAP = APcsv.loc[APcsv['Dose_Species'] == 'IFNa2_YNS'].values[:,1:3]

Thomas2011IFNalpha2YNSAV = AVcsv.loc[AVcsv['Dose_Species'] == 'IFNa2_YNS'].values[:,1:3]

# With error bars
Thomas2011IFNalpha2AP_s = APcsv.loc[APcsv['Dose_Species'] == 'IFNa2'].values[:,1:4]

Thomas2011IFNalpha2AV_s = AVcsv.loc[AVcsv['Dose_Species'] == 'IFNa2'].values[:,1:4]

Thomas2011IFNalpha7AP_s = APcsv.loc[APcsv['Dose_Species'] == 'IFNa7'].values[:,1:4]

Thomas2011IFNalpha7AV_s = AVcsv.loc[AVcsv['Dose_Species'] == 'IFNa7'].values[:,1:4]

Thomas2011IFNomegaAP_s = APcsv.loc[APcsv['Dose_Species'] == 'IFNw'].values[:,1:4]

Thomas2011IFNomegaAV_s = AVcsv.loc[AVcsv['Dose_Species'] == 'IFNw'].values[:,1:4]

Thomas2011IFNalpha2YNSAP_s = APcsv.loc[APcsv['Dose_Species'] == 'IFNa2_YNS'].values[:,1:4]

Thomas2011IFNalpha2YNSAV_s = AVcsv.loc[AVcsv['Dose_Species'] == 'IFNa2_YNS'].values[:,1:4]

Schreiber2017AV = np.array(
                  [[0.0001, 0.0015], [0.00034, 0.0025], [0.0014, 0.01],
                   [0.0051, 0.02], [0.01, 0.027], [0.019, 0.05],
                   [0.024, 0.042], [0.028, 0.17], [0.073, 0.077],
                   [0.078, 0.1], [0.104, 0.33], [0.12, 0.22], [0.18, 0.3],
                   [0.2, 0.58], [0.34, 0.74], [0.32, 0.63], [0.58, 0.75],
                   [1., 0.45], [1.33, 0.29], [2.98, 2.12], [7.5, 1.06],
                   [14.8, 4], [25., 2.1], [39., 2.1], [74., 2.1]])

Schreiber2017AP = np.array(
                  [[0.0001, 0.00022], [0.00034, 0.00096], [0.0014, 0.0025],
                   [0.0012, 0.0081], [0.0051, 0.015], [0.01, 0.045],
                   [0.019, 0.05], [0.017, 0.13], [0.024, 0.016], [0.028, 0.13],
                   [0.073, 0.277], [0.078, 0.3], [0.104, 0.24], [0.12, 0.22],
                   [0.18, 0.44], [0.2, 0.36], [0.34, 0.39], [0.32, 0.58],
                   [0.58, 0.3], [1., 0.9], [2.98, 3.24], [7.5, 2.38],
                   [14.8, 9.3], [25., 25.8], [39., 41.], [74., 71.3]])
