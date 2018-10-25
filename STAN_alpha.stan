# exported from PySB model 'IFN_Models.IFN_alpha_altSOCS_Internalization_ppCompatible'
functions{    
    real[] ode_rhs(real t,
                    real[] y,
                    real[] p,
                    real[] x_r,
                    int[] x_i){
        real ydot[15];

        ydot[1] = y[14]*p[18] + y[5]*p[16] + y[6]*p[18] + (y[1]*y[2]*p[15])*(-1) + (y[1]*y[13]*p[17])*(-1) + (y[1]*y[3]*p[17])*(-1);
        ydot[2] = y[15]*p[20] + y[5]*p[16] + y[7]*p[31] + y[7]*p[27] + y[9]*p[20] + (y[2]*p[25])*(-1) + (y[1]*y[2]*p[15])*(-1) + (y[2]*y[14]*p[19])*(-1) + (y[2]*y[6]*p[19])*(-1);
        ydot[3] = y[13]*p[36] + y[6]*p[18] + y[8]*p[32] + y[8]*p[28] + y[9]*p[22] + (y[3]*p[26])*(-1) + (y[1]*y[3]*p[17])*(-1) + (y[12]*y[3]*p[35])*(-1) + (y[3]*y[5]*p[21])*(-1);
        ydot[4] = y[10]*p[24] + (y[4]*y[9]*p[23])*(-1);
        ydot[5] = y[1]*y[2]*p[15] + y[15]*p[22] + y[9]*p[22] + (y[5]*p[16])*(-1) + (y[13]*y[5]*p[21])*(-1) + (y[3]*y[5]*p[21])*(-1);
        ydot[6] = y[1]*y[3]*p[17] + y[14]*p[36] + y[9]*p[20] + (y[6]*p[18])*(-1) + (y[2]*y[6]*p[19])*(-1) + (y[12]*y[6]*p[35])*(-1);
        ydot[7] = y[2]*p[25] + y[11]*p[30] + (y[7]*p[31])*(-1) + (y[7]*p[27])*(-1);
        ydot[8] = y[11]*p[30] + y[3]*p[26] + (y[8]*p[32])*(-1) + (y[8]*p[28])*(-1);
        ydot[9] = y[2]*y[6]*p[19] + y[15]*p[36] + y[3]*y[5]*p[21] + (y[9]*p[22])*(-1) + (y[9]*p[20])*(-1) + (y[9]*p[29])*(-1) + (y[12]*y[9]*p[35])*(-1);
        ydot[10] = y[4]*y[9]*p[23] + (y[10]*p[24])*(-1);
        ydot[11] = y[9]*p[29] + (y[11]*p[30])*(-1);
        ydot[12] = y[13]*p[36] + y[14]*p[36] + y[15]*p[36] + y[10]*p[33] + (y[12]*p[34])*(-1) + (y[12]*y[3]*p[35])*(-1) + (y[12]*y[6]*p[35])*(-1) + (y[12]*y[9]*p[35])*(-1);
        ydot[13] = y[12]*y[3]*p[35] + y[14]*p[18] + y[15]*p[22] + (y[13]*p[36])*(-1) + (y[1]*y[13]*p[17])*(-1) + (y[13]*y[5]*p[21])*(-1);
        ydot[14] = y[1]*y[13]*p[17] + y[12]*y[6]*p[35] + y[15]*p[20] + (y[14]*p[36])*(-1) + (y[14]*p[18])*(-1) + (y[2]*y[14]*p[19])*(-1);
        ydot[15] = y[2]*y[14]*p[19] + y[12]*y[9]*p[35] + y[13]*y[5]*p[21] + (y[15]*p[36])*(-1) + (y[15]*p[22])*(-1) + (y[15]*p[20])*(-1);
        return ydot;
    }
}
            
data {
    int<lower=1> T;
    real t0;
    real ts[72];
}
transformed data {
    real x_r[0];
    int x_i[0];
}
parameters{
    real kd4[1];
}
transformed parameters {
    real y0[15];
    real p[36];
    
    y0[1] = 6022000000;
    y0[2] = 2000;
    y0[3] = 2000;
    y0[4] = 10000;
    y0[5] = 0;
    y0[6] = 0;
    y0[7] = 0;
    y0[8] = 0;
    y0[9] = 0;
    y0[10] = 0;
    y0[11] = 0;
    y0[12] = 0;
    y0[13] = 0;
    y0[14] = 0;
    y0[15] = 0;
    p[1] = 6.022e+23; # 'NA'
    p[2] = 3.142; # 'PI'
    p[3] = 3e-05; # 'rad_cell'
    p[4] = 8e-06; # 'cell_thickness'
    p[5] = 100000; # 'cell_dens'
    p[6] = 1e-06; # 'width_PM'
    p[7] = 1e-05; # 'volEC'
    p[8] = 2.76e-09; # 'volPM'
    p[9] = 7.2e-15; # 'volCP'
    p[10] = 1e-09; # 'IFN'
    p[11] = 6.022e+09; # 'I'
    p[12] = 2000; # 'R1'
    p[13] = 2000; # 'R2'
    p[14] = 10000; # 'S'
    p[15] = 3.3211558e-14; # 'ka1'
    p[16] = 1; # 'kd1'
    p[17] = 4.9817336e-13; # 'ka2'
    p[18] = 0.015; # 'kd2'
    p[19] = 0.0003623188; # 'ka4'
    p[20] = kd4[1]; # 'kd4'
    p[21] = 0.0003623188; # 'ka3'
    p[22] = 0.0003; # 'kd3'
    p[23] = 1e-06; # 'kpa'
    p[24] = 0.001; # 'kpu'
    p[25] = 0.0001; # 'kIntBasal_r1'
    p[26] = 2e-05; # 'kIntBasal_r2'
    p[27] = 0.0001; # 'krec_r1'
    p[28] = 0.0001; # 'krec_r2'
    p[29] = 0.0005; # 'kint_IFN'
    p[30] = 0.0008; # 'kdeg_IFN'
    p[31] = 0.0003; # 'krec_a1'
    p[32] = 0.005; # 'krec_a2'
    p[33] = 0.004; # 'kSOCS'
    p[34] = 0.0025; # 'SOCSdeg'
    p[35] = 1e-06; # 'kSOCSon'
    p[36] = 0.00055; # 'kSOCSoff'

}
model{
    kd4 ~ lognormal(log(0.3),0.2);
}
generated quantities {
    real y[72,15];
    real observables[72,19];
    y = integrate_ode_rk45(ode_rhs, y0, t0, ts, p, x_r, x_i);
    for (t in 1:72) {
        observables[t,1] = y[t,1]; # 'Free_Ia'
        observables[t,2] = y[t,2]; # 'Free_R1'
        observables[t,3] = y[t,3]; # 'Free_R2'
        observables[t,4] = y[t,5]; # 'R1Ia'
        observables[t,5] = y[t,6]; # 'R2Ia'
        observables[t,6] = y[t,7]+y[t,11]; # 'IntR1'
        observables[t,7] = y[t,8]+y[t,11]; # 'IntR2'
        observables[t,8] = y[t,2]+y[t,5]+y[t,9]+y[t,15]; # 'R1surface'
        observables[t,9] = y[t,3]+y[t,6]+y[t,9]+y[t,13]+y[t,14]+y[t,15]; # 'R2surface'
        observables[t,10] = y[t,9]; # 'T'
        observables[t,11] = y[t,4]+y[t,10]; # 'TotalSTAT'
        observables[t,12] = y[t,10]; # 'pSTATCyt'
        observables[t,13] = 0; # 'pSTATNuc'
        observables[t,14] = y[t,10]; # 'TotalpSTAT'
        observables[t,15] = y[t,12]+y[t,13]+y[t,14]+y[t,15]; # 'SOCSAvail'
        observables[t,16] = 0; # 'SOCSmRNANuc'
        observables[t,17] = 0; # 'SOCSmRNACyt'
        observables[t,18] = y[t,13]; # 'BoundSOCS'
        observables[t,19] = y[t,15]; # 'TSOCS'
    }
}
# suggested python code:
/*

*/