# suggested python code:
# Import the required libraries 
import pystan
import numpy as np
import scipy
import os
import pickle
from matplotlib import pyplot as plt
# Check to see if we can avoid compiling the model:
cwd=os.getcwd()
if os.path.isfile(cwd+'/STAN_alpha.pkl'):
    print('Model has been compiled')
    sm = pickle.load(open('STAN_alpha.pkl','rb'))
else:
    sm = pystan.StanModel(file='STAN_alpha.stan')
    # save the compiled model for later use
    with open('STAN_alpha.pkl','wb') as f:
        pickle.dump(sm,f)
# Simulate the model
deltaT = 1
t_end = 3600
t0 = -deltaT
ts = np.arange(0,t_end*deltaT,deltaT)
y0_input=[0 for i in range(15)];
y0_input[0] = 6022000000;
y0_input[1] = 2000;
y0_input[2] = 2000;
y0_input[3] = 10000;
p=[0 for i in range(36)];
p[0] = 6.022e+23; # 'NA'
p[1] = 3.142; # 'PI'
p[2] = 3e-05; # 'rad_cell'
p[3] = 8e-06; # 'cell_thickness'
p[4] = 100000; # 'cell_dens'
p[5] = 1e-06; # 'width_PM'
p[6] = 1e-05; # 'volEC'
p[7] = 2.76e-09; # 'volPM'
p[8] = 7.2e-15; # 'volCP'
p[9] = 1e-09; # 'IFN'
p[10] = 6.022e+09; # 'I'
p[11] = 2000; # 'R1'
p[12] = 2000; # 'R2'
p[13] = 10000; # 'S'
p[14] = 3.3211558e-14; # 'ka1'
p[15] = 1; # 'kd1'
p[16] = 4.9817336e-13; # 'ka2'
p[17] = 0.015; # 'kd2'
p[18] = 0.0003623188; # 'ka4'
p[19] = 0.3; # 'kd4'
p[20] = 0.0003623188; # 'ka3'
p[21] = 0.0003; # 'kd3'
p[22] = 1e-06; # 'kpa'
p[23] = 0.001; # 'kpu'
p[24] = 0.0001; # 'kIntBasal_r1'
p[25] = 2e-05; # 'kIntBasal_r2'
p[26] = 0.0001; # 'krec_r1'
p[27] = 0.0001; # 'krec_r2'
p[28] = 0.0005; # 'kint_IFN'
p[29] = 0.0008; # 'kdeg_IFN'
p[30] = 0.0003; # 'krec_a1'
p[31] = 0.005; # 'krec_a2'
p[32] = 0.004; # 'kSOCS'
p[33] = 0.0025; # 'SOCSdeg'
p[34] = 1e-06; # 'kSOCSon'
p[35] = 0.00055; # 'kSOCSoff'

samples = sm.sampling(data={'T':t_end,'y0':y0_input,'t0':t0,'ts':ts,'p':p},
                        algorithm='Fixed_param',
                        seed=42,
                        chains=1,
                        iter=1)
                  
samples.plot()
plt.show()
