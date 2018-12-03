"""
A module containing a class that produces STAN code for simulating a PySB
model and fitting ODE parameters.
Structure of the STAN code
=======================================
The STAN code defines a STAN model and writes the outline of how to run this
simulation using PySTAN.
Using the standalone Python model
=================================
An example usage pattern for the standalone Robertson model, once generated::
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
    y0_input=[6022000000,2000,2000,10000]
    samples = sm.sampling(data={'T':t_end,'y0':y0_input,'t0':t0,'ts':ts},
                            seed=42,
                            chains=1,
                            iter=1000,
                            warmup=500,
                            refresh=-1)
                      
    samples.plot()
    plt.show()
"""
import pysb
import pysb.bng
import sympy
import textwrap
from pysb.export import Exporter, pad
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
import re
class StanExporter(Exporter):
    """A class for returning the standalone Python code for a given PySB model.
    Inherits from :py:class:`pysb.export.Exporter`, which implements
    basic functionality for all exporters.
    """
    def export(self):
        """Export STAN code for simulation of a model using STAN
        Returns
        -------
        string
            String containing the STAN code.
        """
        output = StringIO()
        pysb.bng.generate_equations(self.model)
        # Note: This has a lot of duplication from pysb.integrate.
        # Can that be helped?
        code_eqs = '\n'.join(['ydot[%d] = %s;' %
                                 (i+1, sympy.ccode(self.model.odes[i]))
                              for i in range(len(self.model.odes))])
        code_eqs = re.sub(r's(\d+)',
                          lambda m: 'y[%s]' % (int(m.group(1))+1), code_eqs)
						  
        for i, p in enumerate(self.model.parameters):
            code_eqs = re.sub(r'\b(%s)\b' % p.name, 'p[%s]' % str(i+1), code_eqs)
        init_data = {
            'num_species': len(self.model.species),
            'num_params': len(self.model.parameters),
            'num_observables': len(self.model.observables),
            'num_ics': len(self.model.initial_conditions),
            }            
        if self.docstring:
            output.write('"""')
            output.write(self.docstring)
            output.write('"""\n\n')
        output.write("// exported from PySB model '%s'\n" % self.model.name)
        # output independent STAN model and simulation code
        # first write the STAN code as a string
        
        #output.write('stan_code = r"""')
        output.write("functions{")
		# EDIT BY DUNCAN KIRBY: REMOVE '_' FROM code_eqs TO GET SIMULATIONS TO RUN 
        code_eqs = re.sub('_','',code_eqs)
        output.write(pad(r"""
        real[] ode_rhs(real t,
                        real[] y,
                        real[] p,
                        real[] x_r,
                        int[] x_i){
            real ydot[%d];""",4) % init_data['num_species'])
        output.write(r"""
        %s
        return ydot;
    }
}
            """ % pad('\n' + code_eqs, 8).strip())
			# note the simulate method is fixed, i.e. it doesn't require any templating
        output.write(r"""
data {
    int<lower=1> T;
    real t0;
    real ts[%d];
}
transformed data {
    real x_r[0];
    int x_i[0];
}
parameters{
}
transformed parameters {
    real y0[%d];
    real p[%d];
    
""" % (72,init_data['num_species'],init_data['num_params']))

        # initial conditions
        species_stringList = [str(el) for el in self.model.species]
        used_indices=[]
        for ic in self.model.initial_conditions:
            y_index = species_stringList.index(str(ic[0]))+1
            used_indices.append(y_index)
            output.write("    y0[%d] = %d;\n" % (y_index, ic[1].value))
        for i in range(1,init_data['num_species']+1):
            if i not in used_indices:
                output.write("    y0[%d] = 0;\n" % i)
        # model parameter vector
        for i, p in enumerate(self.model.parameters):
            p_data = (i+1, p.value, repr(p.name))
            output.write("    p[%d] = %.8g; // %s\n" % p_data)    

        output.write(r"""
}
model {
    real y[72,%d];
    real observables[72,%d];
    y = integrate_ode_rk45(ode_rhs, y0, t0, ts, p, x_r, x_i);
""" % (init_data['num_species'],init_data['num_observables']))
        #   observables
        output.write("""    for (t in 1:72) {""")
        output.write("\n")
        for i, obs in enumerate(self.model.observables):
            ycoords = [j for j in obs.species]
            for ind in range(len(ycoords)):# offset indices by 1
                ycoords[ind]+=1
            if ycoords == []:# pass 0 to observables which don't appear in ODE equations
                ycoords = 0
            obs_data = (i+1, str(ycoords).replace(", ","]+y[t,")[1:], repr(obs.name))
            output.write(" " * 8)
            if ycoords==0:
                output.write("observables[t,%d] =%s 0; // %s\n" % obs_data)
            else:
                output.write("observables[t,%d] = y[t,%s; // %s\n" % obs_data)         
        output.write("""    }
}""")
        #output.write('"""')
        
        # now write the simulation code
        output.write(r"""
# suggested python code:
/*
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
deltaT = 50
t_end = 3600
t0 = -1
ts = np.arange(0,t_end*deltaT,deltaT)""")


        
        output.write(r"""
samples = sm.sampling(data={'T':t_end,'t0':t0,'ts':ts},
                        algorithm='Fixed_param',
                        seed=42,
                        chains=1,
                        iter=1)
                  
#samples.plot()
#plt.show()

# Plot TotalpSTAT
res = samples.extract("observables",permuted=False)['observables'][0][0]
plt.plot(range(0,t_end,deltaT),res[:,11])
plt.show()
*/""")
        
        return output.getvalue()
