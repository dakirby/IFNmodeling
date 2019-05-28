from numpy import *
import matplotlib.pyplot as mpl
#from pysb.simulator import ScipyOdeSimulator
T = 800
t = 0

## Parameters ##
k1a = 4.98173364330787e-13  # =3e6/NA*volEC
k_1a = 0.015  # *10 s-1
k2a = 3.623188E-4  # =2E5/NA*volEC
k_2a = 0.3  # s-1
k3a = 3.623188E-4  # =2E5/NA*volEC
k_3a = 3E-4  # s-1
k4a = 3.321155762205247e-14  # =2e5/NA*volEC
k_4a = 1  # *10 s-1

## Initial pop ##
IFNa = 100e-12 * 6.022e23 * 1e-5
R1 = 2000
R2 = 2000
IFNaR1 = 0
IFNaR2 = 0
IFNaR1R2 = 0

# initialize results list
iterations = 3
ternary = []
data = []
time = []
ternary_t = [[] for i in range(iterations)]
data_t = [[] for i in range(iterations)]
time_t = [[] for i in range(iterations)]
time.append(t)
data.append((IFNa, R1, R2, IFNaR1, IFNaR2))
ternary.append(IFNaR1R2)
traj_t = []
mean_time = linspace(0, T, T)

# main loop
for i in range(iterations):

    IFNa = 100e-12 * 6.022e23 * 1e-5
    R1 = 2000
    R2 = 2000
    IFNaR1 = 0
    IFNaR2 = 0
    IFNaR1R2 = 0
    t = 0
    ternary = []
    data = []
    time = []
    time.append(t)
    data.append((IFNa, R1, R2, IFNaR1, IFNaR2))
    ternary.append(IFNaR1R2)

    while t < T:
        r1 = k4a * IFNa * R1  # d[IFNaR1]
        r2 = k1a * IFNa * R2  # d[IFNaR2]
        r_1 = k_4a * IFNaR1  # + k_2a*IFNaR1R2 #d[R1]
        r_2 = k_1a * IFNaR2  # + k_3a*IFNaR1R2 #d[R2]
        rT1 = k3a * IFNaR1 * R2  # d[Ternary Complex] from IFNaR1
        rT2 = k2a * IFNaR2 * R1  # d[Ternary Complex] from IFNaR2
        R_T1 = k_3a * IFNaR1R2
        R_T2 = k_2a * IFNaR1R2
        RTOT = r1 + r2 + r_1 + r_2 + rT1 + rT2 + R_T1 + R_T2
        dt = -log((random.random_sample())) / RTOT
        t = t + dt
        rxn_number = random.random_sample()

        if rxn_number < r1 / RTOT:
            IFNa = IFNa - 1
            R1 = R1 - 1
            IFNaR1 = IFNaR1 + 1
        elif rxn_number < (r1+r_1)/RTOT:
            IFNa = IFNa + 1
            R1 = R1 + 1
            IFNaR1 = IFNaR1 - 1
        elif rxn_number < (r1+r_1+r_2)/RTOT:
            IFNa = IFNa + 1
            R2 = R2 + 1
            IFNaR2 = IFNaR2 - 1
        elif rxn_number < (r1+r_1+r_2+r2) / RTOT:
            IFNa = IFNa - 1
            R2 = R2 - 1
            IFNaR2 = IFNaR2 + 1
        elif rxn_number < (r1 + r_1 + r_2 + r2 + rT1) / RTOT:
            IFNaR1 -= 1
            R2 -= 1
            IFNaR1R2 += 1
        elif rxn_number < (r1 + r_1 + r_2 + r2 + rT1 + rT2) / RTOT:
            IFNaR2 -= 1
            R1 -= 1
            IFNaR1R2 += 1
        elif rxn_number < (r1 + r_1 + r_2 + r2 + rT1 + rT2 + R_T1) / RTOT:
            IFNaR1 += 1
            R2 += 1
            IFNaR1R2 -= 1
        elif rxn_number < (r1 + r_1 + r_2 + r2 + rT1 + rT2 + R_T1 + R_T2) / RTOT:
            IFNaR2 += 1
            R1 += 1
            IFNaR1R2 -= 1

        data.append((IFNa, R1, R2, IFNaR1, IFNaR2))
        time.append(t)
        ternary.append(IFNaR1R2)

    data_t[i] = data
    time_t[i] = time
    ternary_t[i] = ternary

mpl.figure(1)
for r in range(iterations):
    mpl.plot(time_t[r], ternary_t[r], alpha=0.05)
mpl.xlabel("Time (s)")
mpl.ylabel("# Ternary Complex")

# MEAN TRAJECTORY ---------------
traj_var = []
for j in range(T):
    A = 0
    B = 0
    time_value = 0
    time_index = 0
    traj_value = 0
    for i in range(iterations):
        time_value = next(x for x in time_t[i] if x > j)
        time_index = time_t[i].index(time_value)
        traj_value = ternary_t[i][time_index]
        A += traj_value / iterations

    traj_t.append(A)
    for i in range(iterations):
        time_value = next(x for x in time_t[i] if x > j)
        time_index = time_t[i].index(time_value)
        traj_value = ternary_t[i][time_index]
        B += ((traj_t[j] - traj_value) ** 2) / iterations
    traj_var.append(B)

mpl.figure(1)
mpl.plot(mean_time, traj_t, label="Mean of 20 simulations")
mpl.show()
exit()
import ifn_test_model

det_t = linspace(0, T, T)


def timecourse(modelfile, t, spec, axes_labels=['', ''], title='',
               Norm=1, suppress=False, parameters=1):
    # Run simulation
    if parameters == 1:
        simres = ScipyOdeSimulator(modelfile.model, tspan=t, compiler='python').run()
        timecourse = simres.all
    elif type(parameters) == dict:
        simres = ScipyOdeSimulator(modelfile.model, tspan=t,
                                   param_values=parameters, compiler='python').run()
        timecourse = simres.all
    else:
        print("Expected parameters to be a dictionary.")
        return 1
    # Plot results
    if suppress == False:

        mpl.figure(1)

        if Norm == 1:
            for species in spec:
                mpl.plot(t, timecourse[species[0]], label=species[1], linewidth=2.0)
        elif type(Norm) == int or type(Norm) == float:
            for species in spec:
                mpl.plot(t, timecourse[species[0]] / Norm, label=species[1], linewidth=2.0)
        elif type(Norm) == list:
            if type(Norm[0]) == int or type(Norm[0]) == float:  # species-specific normalization factors
                spec_count = 0
                for species in spec:
                    mpl.plot(t, timecourse[species[0]] / Norm[spec_count], label=species[1], linewidth=2.0)
                    spec_count += 1
        else:  # species-specific trajectory to normalize by (ie. norm-factor is a function of time)
            if len(t) != len(Norm[0]):
                print("Expected a trajectory to normalize each species by, but len(t) != len(Norm[0])")
                return 1
            else:
                spec_count = 0
                for species in spec:
                    mpl.plot(t, divide(timecourse[species[0]], Norm[spec_count]),
                             label=species[1], linewidth=2.0)
                    spec_count += 1
        mpl.legend()
    return timecourse


deterministic = timecourse(ifn_test_model, det_t, [['obsIFNaR1R2', 'Deterministic model']])

timecourse_traj = []
for i in range(T):
    timecourse_traj.append(deterministic[i][13])

# difference between stoch and det
mean_diff = abs(traj_t - asarray(timecourse_traj))

mpl.figure(2)
mpl.plot(det_t, mean_diff)
mpl.xlabel('Time (s)')
mpl.ylabel('# Ternary Complex (|Mean-Deterministic|)')

mpl.figure(3)
mpl.plot(mean_time, (asarray(traj_var) / ((asarray(traj_t)) ** 2)))
mpl.xlabel("Time (s)")
mpl.ylabel("$\sigma^2$/T$^2$")