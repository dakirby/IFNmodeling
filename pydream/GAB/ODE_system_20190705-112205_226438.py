# exported from PySB model 'ifnmodels.Mixed_IFN_ppCompatible'

import numpy
import scipy.integrate
import collections
import itertools
import distutils.errors


_use_inline = False

Parameter = collections.namedtuple('Parameter', 'name value')
Observable = collections.namedtuple('Observable', 'name species coefficients')
Initial = collections.namedtuple('Initial', 'param_index species_index')


class Model(object):
    
    def __init__(self):
        self.y = None
        self.yobs = None
        self.integrator = scipy.integrate.ode(self.ode_rhs)
        self.integrator.set_integrator('vode', method='bdf',
                                       with_jacobian=True)
        self.y0 = numpy.empty(22)
        self.ydot = numpy.empty(22)
        self.sim_param_values = numpy.empty(49)
        self.parameters = [None] * 49
        self.observables = [None] * 24
        self.initial_conditions = [None] * 6
    
        self.parameters[0] = Parameter('NA', 6.0220000000000003e+23)
        self.parameters[1] = Parameter('PI', 3.1419999999999999)
        self.parameters[2] = Parameter('rad_cell', 3.0000000000000001e-05)
        self.parameters[3] = Parameter('cell_thickness', 7.9999999999999996e-06)
        self.parameters[4] = Parameter('cell_dens', 100000)
        self.parameters[5] = Parameter('width_PM', 9.9999999999999995e-07)
        self.parameters[6] = Parameter('volEC', 1.0000000000000001e-05)
        self.parameters[7] = Parameter('volPM', 2.76e-09)
        self.parameters[8] = Parameter('volCP', 7.2000000000000002e-15)
        self.parameters[9] = Parameter('Ia', 6022000000)
        self.parameters[10] = Parameter('Ib', 6022000000)
        self.parameters[11] = Parameter('R1', 2000)
        self.parameters[12] = Parameter('R2', 2000)
        self.parameters[13] = Parameter('S', 10000)
        self.parameters[14] = Parameter('pS', 0)
        self.parameters[15] = Parameter('ka1', 3.3211557622052469e-14)
        self.parameters[16] = Parameter('kd1', 1)
        self.parameters[17] = Parameter('ka2', 4.9817336433078698e-13)
        self.parameters[18] = Parameter('kd2', 0.014999999999999999)
        self.parameters[19] = Parameter('ka4', 0.00036231879999999998)
        self.parameters[20] = Parameter('kd4', 0.29999999999999999)
        self.parameters[21] = Parameter('ka3', 0.00036231879999999998)
        self.parameters[22] = Parameter('kd3', 0.00029999999999999997)
        self.parameters[23] = Parameter('k_a1', 4.98e-14)
        self.parameters[24] = Parameter('k_d1', 0.029999999999999999)
        self.parameters[25] = Parameter('k_a2', 8.3e-13)
        self.parameters[26] = Parameter('k_d2', 0.002)
        self.parameters[27] = Parameter('k_a3', 0.00036200000000000002)
        self.parameters[28] = Parameter('k_d3', 2.4000000000000001e-05)
        self.parameters[29] = Parameter('k_a4', 0.00036200000000000002)
        self.parameters[30] = Parameter('k_d4', 0.0060000000000000001)
        self.parameters[31] = Parameter('kpa', 9.9999999999999995e-07)
        self.parameters[32] = Parameter('kpu', 0.001)
        self.parameters[33] = Parameter('kIntBasal_r1', 0.0001)
        self.parameters[34] = Parameter('kIntBasal_r2', 2.0000000000000002e-05)
        self.parameters[35] = Parameter('krec_r1', 0.0001)
        self.parameters[36] = Parameter('krec_r2', 0.0001)
        self.parameters[37] = Parameter('kint_a', 0.00050000000000000001)
        self.parameters[38] = Parameter('kint_b', 0.00020000000000000001)
        self.parameters[39] = Parameter('kdeg_a', 0.00080000000000000004)
        self.parameters[40] = Parameter('kdeg_b', 0.00080000000000000004)
        self.parameters[41] = Parameter('krec_a1', 0.00029999999999999997)
        self.parameters[42] = Parameter('krec_a2', 0.0050000000000000001)
        self.parameters[43] = Parameter('krec_b1', 0.0001)
        self.parameters[44] = Parameter('krec_b2', 0.001)
        self.parameters[45] = Parameter('kSOCS', 0.0040000000000000001)
        self.parameters[46] = Parameter('SOCSdeg', 0.0025000000000000001)
        self.parameters[47] = Parameter('kSOCSon', 9.9999999999999995e-07)
        self.parameters[48] = Parameter('kSOCSoff', 0.00055000000000000003)

        self.observables[0] = Observable('Free_Ia', [0], [1])
        self.observables[1] = Observable('Free_Ib', [1], [1])
        self.observables[2] = Observable('Free_R1', [2], [1])
        self.observables[3] = Observable('Free_R2', [3], [1])
        self.observables[4] = Observable('R1Ia', [6], [1])
        self.observables[5] = Observable('R2Ia', [7], [1])
        self.observables[6] = Observable('R1Ib', [8], [1])
        self.observables[7] = Observable('R2Ib', [9], [1])
        self.observables[8] = Observable('IntR1', [11, 20, 21], [1, 1, 1])
        self.observables[9] = Observable('IntR2', [12, 20, 21], [1, 1, 1])
        self.observables[10] = Observable('R1surface', [2, 6, 8, 13, 14, 18, 19], [1, 1, 1, 1, 1, 1, 1])
        self.observables[11] = Observable('R2surface', [3, 7, 9, 13, 14, 15, 16, 17, 18, 19], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.observables[12] = Observable('Ta', [13], [1])
        self.observables[13] = Observable('Tb', [14], [1])
        self.observables[14] = Observable('TotalSTAT', [4, 5], [1, 1])
        self.observables[15] = Observable('pSTATCyt', [5], [1])
        self.observables[16] = Observable('pSTATNuc', [], [])
        self.observables[17] = Observable('TotalpSTAT', [5], [1])
        self.observables[18] = Observable('SOCSAvail', [10, 15, 16, 17, 18, 19], [1, 1, 1, 1, 1, 1])
        self.observables[19] = Observable('SOCSmRNANuc', [], [])
        self.observables[20] = Observable('SOCSmRNACyt', [], [])
        self.observables[21] = Observable('BoundSOCS', [15], [1])
        self.observables[22] = Observable('TSOCSa', [18], [1])
        self.observables[23] = Observable('TSOCSb', [19], [1])

        self.initial_conditions[0] = Initial(9, 0)
        self.initial_conditions[1] = Initial(10, 1)
        self.initial_conditions[2] = Initial(11, 2)
        self.initial_conditions[3] = Initial(12, 3)
        self.initial_conditions[4] = Initial(13, 4)
        self.initial_conditions[5] = Initial(14, 5)

    if _use_inline:
        
        def ode_rhs(self, t, y, p):
            ydot = self.ydot
            weave.inline(r'''                
                ydot[0] = __y[16]*p[18] + __y[6]*p[16] + __y[7]*p[18] + (__y[0]*__y[15]*p[17])*(-1) + (__y[0]*__y[2]*p[15])*(-1) + (__y[0]*__y[3]*p[17])*(-1);
                ydot[1] = __y[17]*p[26] + __y[8]*p[24] + __y[9]*p[26] + (__y[1]*__y[15]*p[25])*(-1) + (__y[1]*__y[2]*p[23])*(-1) + (__y[1]*__y[3]*p[25])*(-1);
                ydot[2] = 2.0*__y[11]*p[41] + __y[11]*p[35] + __y[13]*p[20] + __y[14]*p[30] + __y[18]*p[20] + __y[19]*p[30] + __y[6]*p[16] + __y[8]*p[24] + (__y[2]*p[33])*(-1) + (__y[0]*__y[2]*p[15])*(-1) + (__y[1]*__y[2]*p[23])*(-1) + (__y[16]*__y[2]*p[19])*(-1) + (__y[17]*__y[2]*p[29])*(-1) + (__y[2]*__y[7]*p[19])*(-1) + (__y[2]*__y[9]*p[29])*(-1);
                ydot[3] = 2.0*__y[12]*p[42] + __y[12]*p[36] + __y[13]*p[22] + __y[14]*p[28] + __y[15]*p[48] + __y[7]*p[18] + __y[9]*p[26] + (__y[3]*p[34])*(-1) + (__y[0]*__y[3]*p[17])*(-1) + (__y[1]*__y[3]*p[25])*(-1) + (__y[10]*__y[3]*p[47])*(-1) + (__y[3]*__y[6]*p[21])*(-1) + (__y[3]*__y[8]*p[27])*(-1);
                ydot[4] = __y[5]*p[32] + (__y[13]*__y[4]*p[31])*(-1) + (__y[14]*__y[4]*p[31])*(-1);
                ydot[5] = __y[13]*__y[4]*p[31] + __y[14]*__y[4]*p[31] + (__y[5]*p[32])*(-1);
                ydot[6] = __y[0]*__y[2]*p[15] + __y[13]*p[22] + __y[18]*p[22] + (__y[6]*p[16])*(-1) + (__y[15]*__y[6]*p[21])*(-1) + (__y[3]*__y[6]*p[21])*(-1);
                ydot[7] = __y[0]*__y[3]*p[17] + __y[13]*p[20] + __y[16]*p[48] + (__y[7]*p[18])*(-1) + (__y[10]*__y[7]*p[47])*(-1) + (__y[2]*__y[7]*p[19])*(-1);
                ydot[8] = __y[1]*__y[2]*p[23] + __y[14]*p[28] + (__y[8]*p[24])*(-1) + (__y[3]*__y[8]*p[27])*(-1);
                ydot[9] = __y[1]*__y[3]*p[25] + __y[14]*p[30] + __y[17]*p[48] + (__y[9]*p[26])*(-1) + (__y[10]*__y[9]*p[47])*(-1) + (__y[2]*__y[9]*p[29])*(-1);
                ydot[10] = __y[15]*p[48] + __y[16]*p[48] + __y[17]*p[48] + __y[18]*p[48] + __y[19]*p[48] + __y[5]*p[45] + (__y[10]*p[46])*(-1) + (__y[10]*__y[13]*p[47])*(-1) + (__y[10]*__y[14]*p[47])*(-1) + (__y[10]*__y[3]*p[47])*(-1) + (__y[10]*__y[7]*p[47])*(-1) + (__y[10]*__y[9]*p[47])*(-1);
                ydot[11] = __y[2]*p[33] + __y[20]*p[39] + __y[21]*p[40] + (2.0*__y[11]*p[41])*(-1) + (__y[11]*p[35])*(-1);
                ydot[12] = __y[20]*p[39] + __y[21]*p[40] + __y[3]*p[34] + (2.0*__y[12]*p[42])*(-1) + (__y[12]*p[36])*(-1);
                ydot[13] = __y[18]*p[48] + __y[2]*__y[7]*p[19] + __y[3]*__y[6]*p[21] + (__y[13]*p[22])*(-1) + (__y[13]*p[20])*(-1) + (__y[13]*p[37])*(-1) + (__y[10]*__y[13]*p[47])*(-1);
                ydot[14] = __y[19]*p[48] + __y[2]*__y[9]*p[29] + __y[3]*__y[8]*p[27] + (__y[14]*p[28])*(-1) + (__y[14]*p[30])*(-1) + (__y[14]*p[38])*(-1) + (__y[10]*__y[14]*p[47])*(-1);
                ydot[15] = __y[10]*__y[3]*p[47] + __y[16]*p[18] + __y[17]*p[26] + __y[18]*p[22] + (__y[15]*p[48])*(-1) + (__y[0]*__y[15]*p[17])*(-1) + (__y[1]*__y[15]*p[25])*(-1) + (__y[15]*__y[6]*p[21])*(-1);
                ydot[16] = __y[0]*__y[15]*p[17] + __y[10]*__y[7]*p[47] + __y[18]*p[20] + (__y[16]*p[48])*(-1) + (__y[16]*p[18])*(-1) + (__y[16]*__y[2]*p[19])*(-1);
                ydot[17] = __y[1]*__y[15]*p[25] + __y[10]*__y[9]*p[47] + __y[19]*p[30] + (__y[17]*p[48])*(-1) + (__y[17]*p[26])*(-1) + (__y[17]*__y[2]*p[29])*(-1);
                ydot[18] = __y[10]*__y[13]*p[47] + __y[15]*__y[6]*p[21] + __y[16]*__y[2]*p[19] + (__y[18]*p[48])*(-1) + (__y[18]*p[22])*(-1) + (__y[18]*p[20])*(-1);
                ydot[19] = __y[10]*__y[14]*p[47] + __y[17]*__y[2]*p[29] + (__y[19]*p[48])*(-1) + (__y[19]*p[30])*(-1);
                ydot[20] = __y[13]*p[37] + (__y[20]*p[39])*(-1);
                ydot[21] = __y[14]*p[38] + (__y[21]*p[40])*(-1);
                ''', ['ydot', 't', 'y', 'p'])
            return ydot
        
    else:
        
        def ode_rhs(self, t, y, p):
            ydot = self.ydot
            ydot[0] = y[16]*p[18] + y[6]*p[16] + y[7]*p[18] + (y[0]*y[15]*p[17])*(-1) + (y[0]*y[2]*p[15])*(-1) + (y[0]*y[3]*p[17])*(-1)
            ydot[1] = y[17]*p[26] + y[8]*p[24] + y[9]*p[26] + (y[1]*y[15]*p[25])*(-1) + (y[1]*y[2]*p[23])*(-1) + (y[1]*y[3]*p[25])*(-1)
            ydot[2] = 2.0*y[11]*p[41] + y[11]*p[35] + y[13]*p[20] + y[14]*p[30] + y[18]*p[20] + y[19]*p[30] + y[6]*p[16] + y[8]*p[24] + (y[2]*p[33])*(-1) + (y[0]*y[2]*p[15])*(-1) + (y[1]*y[2]*p[23])*(-1) + (y[16]*y[2]*p[19])*(-1) + (y[17]*y[2]*p[29])*(-1) + (y[2]*y[7]*p[19])*(-1) + (y[2]*y[9]*p[29])*(-1)
            ydot[3] = 2.0*y[12]*p[42] + y[12]*p[36] + y[13]*p[22] + y[14]*p[28] + y[15]*p[48] + y[7]*p[18] + y[9]*p[26] + (y[3]*p[34])*(-1) + (y[0]*y[3]*p[17])*(-1) + (y[1]*y[3]*p[25])*(-1) + (y[10]*y[3]*p[47])*(-1) + (y[3]*y[6]*p[21])*(-1) + (y[3]*y[8]*p[27])*(-1)
            ydot[4] = y[5]*p[32] + (y[13]*y[4]*p[31])*(-1) + (y[14]*y[4]*p[31])*(-1)
            ydot[5] = y[13]*y[4]*p[31] + y[14]*y[4]*p[31] + (y[5]*p[32])*(-1)
            ydot[6] = y[0]*y[2]*p[15] + y[13]*p[22] + y[18]*p[22] + (y[6]*p[16])*(-1) + (y[15]*y[6]*p[21])*(-1) + (y[3]*y[6]*p[21])*(-1)
            ydot[7] = y[0]*y[3]*p[17] + y[13]*p[20] + y[16]*p[48] + (y[7]*p[18])*(-1) + (y[10]*y[7]*p[47])*(-1) + (y[2]*y[7]*p[19])*(-1)
            ydot[8] = y[1]*y[2]*p[23] + y[14]*p[28] + (y[8]*p[24])*(-1) + (y[3]*y[8]*p[27])*(-1)
            ydot[9] = y[1]*y[3]*p[25] + y[14]*p[30] + y[17]*p[48] + (y[9]*p[26])*(-1) + (y[10]*y[9]*p[47])*(-1) + (y[2]*y[9]*p[29])*(-1)
            ydot[10] = y[15]*p[48] + y[16]*p[48] + y[17]*p[48] + y[18]*p[48] + y[19]*p[48] + y[5]*p[45] + (y[10]*p[46])*(-1) + (y[10]*y[13]*p[47])*(-1) + (y[10]*y[14]*p[47])*(-1) + (y[10]*y[3]*p[47])*(-1) + (y[10]*y[7]*p[47])*(-1) + (y[10]*y[9]*p[47])*(-1)
            ydot[11] = y[2]*p[33] + y[20]*p[39] + y[21]*p[40] + (2.0*y[11]*p[41])*(-1) + (y[11]*p[35])*(-1)
            ydot[12] = y[20]*p[39] + y[21]*p[40] + y[3]*p[34] + (2.0*y[12]*p[42])*(-1) + (y[12]*p[36])*(-1)
            ydot[13] = y[18]*p[48] + y[2]*y[7]*p[19] + y[3]*y[6]*p[21] + (y[13]*p[22])*(-1) + (y[13]*p[20])*(-1) + (y[13]*p[37])*(-1) + (y[10]*y[13]*p[47])*(-1)
            ydot[14] = y[19]*p[48] + y[2]*y[9]*p[29] + y[3]*y[8]*p[27] + (y[14]*p[28])*(-1) + (y[14]*p[30])*(-1) + (y[14]*p[38])*(-1) + (y[10]*y[14]*p[47])*(-1)
            ydot[15] = y[10]*y[3]*p[47] + y[16]*p[18] + y[17]*p[26] + y[18]*p[22] + (y[15]*p[48])*(-1) + (y[0]*y[15]*p[17])*(-1) + (y[1]*y[15]*p[25])*(-1) + (y[15]*y[6]*p[21])*(-1)
            ydot[16] = y[0]*y[15]*p[17] + y[10]*y[7]*p[47] + y[18]*p[20] + (y[16]*p[48])*(-1) + (y[16]*p[18])*(-1) + (y[16]*y[2]*p[19])*(-1)
            ydot[17] = y[1]*y[15]*p[25] + y[10]*y[9]*p[47] + y[19]*p[30] + (y[17]*p[48])*(-1) + (y[17]*p[26])*(-1) + (y[17]*y[2]*p[29])*(-1)
            ydot[18] = y[10]*y[13]*p[47] + y[15]*y[6]*p[21] + y[16]*y[2]*p[19] + (y[18]*p[48])*(-1) + (y[18]*p[22])*(-1) + (y[18]*p[20])*(-1)
            ydot[19] = y[10]*y[14]*p[47] + y[17]*y[2]*p[29] + (y[19]*p[48])*(-1) + (y[19]*p[30])*(-1)
            ydot[20] = y[13]*p[37] + (y[20]*p[39])*(-1)
            ydot[21] = y[14]*p[38] + (y[21]*p[40])*(-1)
            return ydot
        
    
    def simulate(self, tspan, param_values=None, view=False):
        if param_values is not None:
            # accept vector of parameter values as an argument
            if len(param_values) != len(self.parameters):
                raise Exception("param_values must have length %d" %
                                len(self.parameters))
            self.sim_param_values[:] = param_values
        else:
            # create parameter vector from the values in the model
            self.sim_param_values[:] = [p.value for p in self.parameters]
        self.y0.fill(0)
        for ic in self.initial_conditions:
            self.y0[ic.species_index] = self.sim_param_values[ic.param_index]
        if self.y is None or len(tspan) != len(self.y):
            self.y = numpy.empty((len(tspan), len(self.y0)))
            if len(self.observables):
                self.yobs = numpy.ndarray(len(tspan),
                                list(zip((obs.name for obs in self.observables),
                                    itertools.repeat(float))))
            else:
                self.yobs = numpy.ndarray((len(tspan), 0))
            self.yobs_view = self.yobs.view(float).reshape(len(self.yobs),
                                                           -1)
        # perform the actual integration
        self.integrator.set_initial_value(self.y0, tspan[0])
        self.integrator.set_f_params(self.sim_param_values)
        self.y[0] = self.y0
        t = 1
        while self.integrator.successful() and self.integrator.t < tspan[-1]:
            self.y[t] = self.integrator.integrate(tspan[t])
            t += 1
        for i, obs in enumerate(self.observables):
            self.yobs_view[:, i] = \
                (self.y[:, obs.species] * obs.coefficients).sum(1)
        if view:
            y_out = self.y.view()
            yobs_out = self.yobs.view()
            for a in y_out, yobs_out:
                a.flags.writeable = False
        else:
            y_out = self.y.copy()
            yobs_out = self.yobs.copy()
        return (y_out, yobs_out)
    
