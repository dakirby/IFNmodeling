# exported from PySB model 'ifnmodels.IFN_beta_altSOCS_Internalization_ppCompatible'
stan_code = r"""functions{    
    real[] ode_rhs(real t,
                    real[] y,
                    real[] p,
                    real[] x_r,
                    int[] x_i){
        real ydot[15];

        ydot[1] = y[14]*p[18] + y[5]*p[16] + y[6]*p[18] + (y[1]*y[2]*p[15])*(-1) + (y[1]*y[13]*p[17])*(-1) + (y[1]*y[3]*p[17])*(-1);
        ydot[2] = y[15]*p[21] + y[5]*p[16] + y[7]*p[31] + y[7]*p[27] + y[9]*p[21] + (y[2]*p[25])*(-1) + (y[1]*y[2]*p[15])*(-1) + (y[2]*y[14]*p[20])*(-1) + (y[2]*y[6]*p[20])*(-1);
        ydot[3] = y[13]*p[36] + y[6]*p[18] + y[8]*p[32] + y[8]*p[28] + y[9]*p[22] + (y[3]*p[26])*(-1) + (y[1]*y[3]*p[17])*(-1) + (y[12]*y[3]*p[35])*(-1) + (y[3]*y[5]*p[19])*(-1);
        ydot[4] = y[10]*p[24] + (y[4]*y[9]*p[23])*(-1);
        ydot[5] = y[1]*y[2]*p[15] + y[9]*p[22] + (y[5]*p[16])*(-1) + (y[3]*y[5]*p[19])*(-1);
        ydot[6] = y[1]*y[3]*p[17] + y[14]*p[36] + y[9]*p[21] + (y[6]*p[18])*(-1) + (y[2]*y[6]*p[20])*(-1) + (y[12]*y[6]*p[35])*(-1);
        ydot[7] = y[2]*p[25] + y[11]*p[30] + (y[7]*p[31])*(-1) + (y[7]*p[27])*(-1);
        ydot[8] = y[11]*p[30] + y[3]*p[26] + (y[8]*p[32])*(-1) + (y[8]*p[28])*(-1);
        ydot[9] = y[2]*y[6]*p[20] + y[15]*p[36] + y[3]*y[5]*p[19] + (y[9]*p[22])*(-1) + (y[9]*p[21])*(-1) + (y[9]*p[29])*(-1) + (y[12]*y[9]*p[35])*(-1);
        ydot[10] = y[4]*y[9]*p[23] + (y[10]*p[24])*(-1);
        ydot[11] = y[9]*p[29] + (y[11]*p[30])*(-1);
        ydot[12] = y[13]*p[36] + y[14]*p[36] + y[15]*p[36] + y[10]*p[33] + (y[12]*p[34])*(-1) + (y[12]*y[3]*p[35])*(-1) + (y[12]*y[6]*p[35])*(-1) + (y[12]*y[9]*p[35])*(-1);
        ydot[13] = y[12]*y[3]*p[35] + y[14]*p[18] + (y[13]*p[36])*(-1) + (y[1]*y[13]*p[17])*(-1);
        ydot[14] = y[1]*y[13]*p[17] + y[12]*y[6]*p[35] + y[15]*p[21] + (y[14]*p[36])*(-1) + (y[14]*p[18])*(-1) + (y[2]*y[14]*p[20])*(-1);
        ydot[15] = y[2]*y[14]*p[20] + y[12]*y[9]*p[35] + (y[15]*p[36])*(-1) + (y[15]*p[21])*(-1);
        return ydot;
    }
}
            
data {
    real y0[15];
    real p[36];
    int<lower=1> T;
    real t0;
    real ts[T];
}
transformed data {
    real x_r[0];
    int x_i[0];
}
parameters {
}
model {
    real y[T,15];
    y = integrate_ode_rk45(ode_rhs, y0, t0, ts, p, x_r, x_i);
}"""
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
    sm = pystan.StanModel(model_code='stan_code')
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
p[14] = 4.98e-14; # 'k_a1'
p[15] = 0.03; # 'k_d1'
p[16] = 8.3e-13; # 'k_a2'
p[17] = 0.002; # 'k_d2'
p[18] = 0.000362; # 'k_a3'
p[19] = 0.000362; # 'k_a4'
p[20] = 0.006; # 'k_d4'
p[21] = 2.4e-05; # 'k_d3'
p[22] = 1e-06; # 'kpa'
p[23] = 0.001; # 'kpu'
p[24] = 0.0001; # 'kIntBasal_r1'
p[25] = 2e-05; # 'kIntBasal_r2'
p[26] = 0.0001; # 'krec_r1'
p[27] = 0.0001; # 'krec_r2'
p[28] = 0.0002; # 'kint_IFN'
p[29] = 0.0008; # 'kdeg_IFN'
p[30] = 0.0001; # 'krec_b1'
p[31] = 0.001; # 'krec_b2'
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