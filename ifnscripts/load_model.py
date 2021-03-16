from ifnclass.ifnfit import DualMixedPopulation

initial_parameters = {'k_a1': 4.98E-14 * 1.33, 'k_a2': 8.30e-13 * 2,
                      'k_d4': 0.006 * 3.8,
                      'kpu': 0.00095,
                      'ka2': 4.98e-13 * 1.33, 'kd4': 0.3 * 2.867,
                      'kint_a': 0.000124, 'kint_b': 0.00086,
                      'krec_a1': 0.0028, 'krec_a2': 0.01, 'krec_b1': 0.005,
                      'krec_b2': 0.05}
dual_parameters = {'kint_a': 0.00052, 'kSOCSon': 6e-07, 'kint_b': 0.00052,
                   'krec_a1': 0.001, 'krec_a2': 0.1,
                   'krec_b1': 0.005, 'krec_b2': 0.05}
scale_factor = 1.227


def load_model():
    # Parameters found by stepwise fitting GAB mean data
    # Note: can remove multiplicative factors on all K1, K2, K4 and still get
    # very good fit to data (worst is 5 min beta)

    Mixed_Model = DualMixedPopulation('Mixed_IFN_ppCompatible', 0.8, 0.2)
    Mixed_Model.model_1.set_parameters(initial_parameters)
    Mixed_Model.model_1.set_parameters(dual_parameters)
    Mixed_Model.model_1.set_parameters({'R1': 12000.0, 'R2': 1511.1})
    Mixed_Model.model_2.set_parameters(initial_parameters)
    Mixed_Model.model_2.set_parameters(dual_parameters)
    Mixed_Model.model_2.set_parameters({'R1': 6755.56, 'R2': 1511.1})

    return Mixed_Model, Mixed_Model.mixed_dose_response
