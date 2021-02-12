test_affinities = list(logspace(-2, 4, num=10))  # used if simulate_scaling
reference_affinity_idx = find_nearest(test_affinities, 1.0, idx_flag=True)
assert(test_affinities[reference_affinity_idx] == 1.0)
typicalDose = 1  # (pM), to use for plotting how activity scales with K1


if simulate_scaling:
    typicalDose = find_nearest(test_doses, typicalDose)
    print("Using typical dose of {} pM".format(typicalDose))
    AV_EC50_record = []
    AP_EC50_record = []
    AV_typical_record = []
    AP_typical_record = []
    AV_typical_ref = 0
    AP_typical_ref = 0
    # ----------------------------------------------------------------------
    # Get anti-viral and anti-proliferative EC50 for a variety of K1 and K4
    # ----------------------------------------------------------------------
    for affinity_idx, i in enumerate(test_affinities):
        # Simulate dose-response curve
        test_params = {'Ib': 0, 'kd1': params['kd1'] / i, 'kd4': params['kd4'] / i}
        pSTAT_primary = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT',
                                                        'Ia', test_doses,
                                                        parameters=test_params,
                                                        sf=scale_factor)
        pSTAT_primary = np.array([el[0][0] for el in pSTAT_primary.values])
        AV_dose_response = antiViralActivity(pSTAT_primary)
        AV_df = pd.DataFrame.from_dict({'Dose_Species': ['Alpha']*len(AV_dose_response),
                                        'Dose (pM)': test_doses,
                                        times[0]: AV_dose_response})
        AV_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
        AV_data = IfnData('custom', df=AV_df, conditions={'Alpha': {'Ib': 0}})

        test_params.update({'kd4': USP18_sf * params['kd4'] / i})
        pSTAT_refractory = Mixed_Model.mixed_dose_response(times, 'TotalpSTAT',
                                                        'Ia', test_doses,
                                                        parameters=test_params,
                                                        sf=scale_factor)
        pSTAT_refractory = np.array([el[0][0] for el in pSTAT_refractory.values])
        AP_dose_response = antiProliferativeActivity(pSTAT_refractory)
        AP_df = pd.DataFrame.from_dict({'Dose_Species': ['Alpha']*len(AP_dose_response),
                                        'Dose (pM)': test_doses,
                                        times[0]: AP_dose_response})
        AP_df.set_index(['Dose_Species', 'Dose (pM)'], inplace=True)
        AP_data = IfnData('custom', df=AP_df, conditions={'Alpha': {'Ib': 0}})

        # Get EC50
        AV_EC50 = AV_data.get_ec50s()['Alpha'][0][1] # species: DR curve, (time, EC50)
        AP_EC50 = AP_data.get_ec50s()['Alpha'][0][1]

        # Add to record
        AV_EC50_record.append([i, AV_EC50])
        AP_EC50_record.append([i, AP_EC50])

        # Add response at typicalDose
        AV_typical_record.append([i, AV_data.data_set.loc['Alpha', typicalDose][60]])
        AP_typical_record.append([i, AP_data.data_set.loc['Alpha', typicalDose][60]])
        if affinity_idx == reference_affinity_idx:
            AV_typical_ref = AV_data.data_set.loc['Alpha', typicalDose][60]
            AP_typical_ref = AP_data.data_set.loc['Alpha', typicalDose][60]

    # Make typical response relative to reference response
    for i in range(len(AV_typical_record)):
        AV_typical_record[i][1] = AV_typical_record[i][1] / AV_typical_ref
        AP_typical_record[i][1] = AP_typical_record[i][1] / AP_typical_ref

    # Save to dir
    np.save(dir + os.sep + 'AV_EC50.npy', np.array(AV_EC50_record))
    np.save(dir + os.sep + 'AP_EC50.npy', np.array(AP_EC50_record))
    np.save(dir + os.sep + 'AV_typical.npy', np.array(AV_typical_record))
    np.save(dir + os.sep + 'AP_typcial.npy', np.array(AP_typical_record))

else:
    AV_EC50_record = np.array(np.load(dir + os.sep + 'AV_EC50.npy'))
    AP_EC50_record = np.array(np.load(dir + os.sep + 'AP_EC50.npy'))
    AV_typical_record = np.array(np.load(dir + os.sep + 'AV_typical.npy'))
    AP_typical_record = np.array(np.load(dir + os.sep + 'AP_typcial.npy'))


if plot_scaling:
    red = sns.color_palette("tab10")[3]
    blue = sns.color_palette("tab10")[0]

    # EC50 scaling
    ax = panelD  # all_axes[1][0]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Binding Affinity\n' + r'(Relative to IFN$\alpha$2)')
    ax.set_ylabel(r'$EC_{50}$ (Relative to IFN$\alpha$2)')
    # ax.scatter(Schreiber2017AV[:,0], Schreiber2017AV[:,1], color=red, label='Anti-viral')
    # ax.scatter(Schreiber2017AP[:,0], Schreiber2017AP[:,1], color=blue, label='Anti-proliferative')
    ax.plot(AV_EC50_record[:, 0], (AV_EC50_record[:, 1]/AV_EC50_record[reference_affinity_idx, 1]), color=red, linewidth=3, label='Anti-viral')
    ax.plot(AP_EC50_record[:, 0], (AP_EC50_record[:, 1]/AP_EC50_record[reference_affinity_idx, 1]), color=blue, linewidth=3, label='Anti-proliferative')
    ax.legend()

    # Typical response scaling
    ax = panelE  # all_axes[1][1]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Binding Affinity\n' + r'(Relative to IFN$\alpha$2)')
    ax.set_ylabel('Biological Activity\n' + r'(Relative to IFN$\alpha$2)')
    ax.plot(AV_typical_record[:, 0], AV_typical_record[:, 1], color=red, linewidth=3)
    ax.plot(AP_typical_record[:, 0], AP_typical_record[:, 1], color=blue, linewidth=3)
