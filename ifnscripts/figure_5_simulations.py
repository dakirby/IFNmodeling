import load_model
import numpy as np
import copy
import os


def figure_5_simulations(USP18_sf, times, test_doses, dir, tag=''):
    Mixed_Model = load_model.load_model()
    scale_factor = load_model.scale_factor

    params = copy.deepcopy(Mixed_Model.get_parameters())

    np.save(dir + os.sep + 'doses' + tag + '.npy', test_doses)
    # ---------------------------------------------------------------
    # IFNa2-YNS
    # ---------------------------------------------------------------
    # Use IFNbeta parameters for YNS since it is a beta mimic
    pSTAT_a2YNS = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ib',
                                                  test_doses,
                                                  parameters={'Ia': 0},
                                                  sf=scale_factor)
    pSTAT_a2YNS = np.array([el[0][0] for el in pSTAT_a2YNS.values])

    np.save(dir + os.sep + 'pSTAT_a2YNS' + tag + '.npy', pSTAT_a2YNS)

    pSTAT_a2YNS_refractory = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ib',
                                                             test_doses,
                                                             parameters={'Ia': 0, 'k_d4': params['k_d4'] * USP18_sf},
                                                             sf=scale_factor)
    pSTAT_a2YNS_refractory = np.array([el[0][0] for el in pSTAT_a2YNS_refractory.values])

    np.save(dir + os.sep + 'pSTAT_a2YNS_refractory' + tag + '.npy', pSTAT_a2YNS_refractory)

    # response('pSTAT_a2YNS', 1 / 53., 1 / 1.4)
    # response('pSTAT_a2YNS_refractory', 1 / 53., 1 / 1.4, refractory=True)

    # ---------------------------------------------------------------
    # IFN omega
    # ---------------------------------------------------------------
    # IFNw has K1 = 0.08 * K1 of IFNa2  and K2 = 0.4 * K2 of IFNa2, but no change to K4
    custom_params_w = {'Ib': 0, 'kd1': params['kd1'] * 0.08, 'kd2': params['kd2'] * 0.4}
    pSTAT_w = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia',
                                            test_doses,
                                            parameters=custom_params_w,
                                            sf=scale_factor)
    pSTAT_w = np.array([el[0][0] for el in pSTAT_w.values])
    np.save(dir + os.sep + 'pSTAT_w' + tag + '.npy', pSTAT_w)

    # now refractory
    custom_params_w.update({'kd4': params['kd4'] * USP18_sf})
    pSTAT_w_ref = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia',
                                            test_doses,
                                            parameters=custom_params_w,
                                            sf=scale_factor)
    pSTAT_w_ref = np.array([el[0][0] for el in pSTAT_w_ref.values])
    np.save(dir + os.sep + 'pSTAT_w_refractory' + tag + '.npy', pSTAT_w_ref)

    # response('pSTAT_w', 0.4 / 5, 2. / 5.)
    # response('pSTAT_w_refractory', 0.4 / 5, 2. / 5., refractory=True)

    # ---------------------------------------------------------------
    # The rest of the IFNs
    # ---------------------------------------------------------------
    # The rest of the interferons will use the IFNalpha2 model as a baseline
    np.save(dir + os.sep + 'doses' + tag + '.npy', test_doses)
    def response(filename, kd1_sf, kd2_sf, refractory=False):
        custom_params = {'Ib': 0, 'kd1': params['kd1'] * kd1_sf, 'kd2': params['kd2'] * kd2_sf, 'kd4': params['kd4'] * kd1_sf}
        if refractory:
            custom_params.update({'kd4': params['kd4'] * kd1_sf * USP18_sf})
        pSTAT = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT', 'Ia',
                                                test_doses,
                                                parameters=custom_params,
                                                sf=scale_factor)
        pSTAT = np.array([el[0][0] for el in pSTAT.values])
        np.save(dir + os.sep + filename + tag + '.npy', pSTAT)

    # Use the fit IFNa2 parameters
    response('pSTAT_a2', 1., 1.)
    response('pSTAT_a2_refractory', 1., 1., refractory=True)

    # IFNa7 has K1 and K2 half that of IFNa2  (taken from Mathematica notebook)
    response('pSTAT_a7', 0.5, 0.5)
    response('pSTAT_a7_refractory', 0.5, 0.5, refractory=True)  # repeat for refractory response


    # IFNa2-R149A
    response('pSTAT_R149A', 0.096, 1000.)
    response('pSTAT_R149A_refractory', 0.096, 1000., refractory=True)  # repeat for refractory response

    # IFNa2-A145G
    response('pSTAT_A145G', 0.03*32, 1. / 0.03)
    response('pSTAT_A145G_refractory', 0.03*32, 1. / 0.03, refractory=True)  # repeat for refractory response

    # IFNa2-L26A
    response('pSTAT_L26A', 0.22*4.5, 1. / 0.22)
    response('pSTAT_L26A_refractory', 0.22*4.5, 1. / 0.22, refractory=True)  # repeat for refractory response

    # IFNa2-L30A
    response('pSTAT_L30A', 0.0013*742., 1. / 0.0013)
    response('pSTAT_L30A_refractory', 0.0013*742., 1. / 0.0013, refractory=True)  # repeat for refractory response

    # IFNa2-YNS, M148A, scaling factors taken from Thomas 2011
    response('pSTAT_YNSM148A', 1 / 43., 1. / 0.023)
    response('pSTAT_YNSM148A_refractory', 1 / 43., 1. / 0.023, refractory=True)  # repeat for refractory response

    # IFNa2-YNS, L153A, same source as YNS M148A
    response('pSTAT_YNSL153A', 1 / 70., 1. / 0.11)
    response('pSTAT_YNSL153A_refractory', 1 / 70., 1. / 0.11, refractory=True)  # repeat for refractory response

    print("Finished simulating responses")
