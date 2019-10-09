from pysb import Model, Parameter, Expression, Initial, Monomer, Observable, Rule, WILD, ANY
# Begin Model
Model()
# =============================================================================
# # Parameters
# =============================================================================
Parameter('volEC', 1E-5)
Parameter('NA', 6.022E23)
Parameter('volPM', 4 * 3.14159 * (3.6E-6)**2)

# Initial conditions
Parameter('IL10_IC', 0)
Parameter('IL10RA_IC', 1000)
Parameter('IL10RB_IC', 1000)
Parameter('IL20_IC', 0)
Parameter('IL20RA_IC', 1000)
Parameter('IL20RB_IC', 1000)
Parameter('IL22RA1_IC', 1000)
Parameter('IL12_IC', 0)
Parameter('IL12RB1_IC', 1000)
Parameter('IL12RB2_IC', 1000)
Parameter('IL23RA_IC', 1000)
Parameter('IFN_alpha2_IC', 0)
Parameter('IFN_beta_IC', 0)
Parameter('IFNAR1_IC', 1000)
Parameter('IFNAR2_IC', 1000)
Parameter('IFN_gamma_IC', 0)
Parameter('IFNGR1_IC', 1000)
Parameter('IFNGR2_IC', 1000)
Parameter('Gamma_c_IC', 1000)
Parameter('IL21_IC', 0)
Parameter('IL21R_IC', 1000)
Parameter('IL15_IC', 0)
Parameter('IL15RA_IC', 1000)
Parameter('IL9_IC', 0)
Parameter('IL9RA_IC', 1000)
Parameter('IL7_IC', 0)
Parameter('IL7RA_IC', 1000)
Parameter('IL4_IC', 0)
Parameter('IL4RA_IC', 1000)
Parameter('IL13_IC', 0)
Parameter('IL13RA1_IC', 1000)
Parameter('IL13RA2_IC', 1000)
Parameter('IL2_IC', 0)
Parameter('IL2RA_IC', 1000)
Parameter('IL2RB_IC', 1000)
Parameter('Beta_c_IC', 1000)
Parameter('GMCSF_IC', 0)
Parameter('GMCSFRA_IC', 1000)
Parameter('IL3_IC', 0)
Parameter('IL3RA_IC', 1000)
Parameter('IL5_IC', 0)
Parameter('IL5RA_IC', 1000)
Parameter('GP130_IC', 1000)
Parameter('IL6_IC', 0)
Parameter('IL6RA_IC', 1000)
Parameter('GCSF_IC', 0)
Parameter('GCSFR_IC', 1000)
Parameter('IL11_IC', 1000)
Parameter('IL11RA_IC', 1000)
Parameter('IL27_IC', 1000)
Parameter('IL27RA_IC', 1000)

Parameter('Jak1_IC', 1000)
Parameter('Jak2_IC', 1000)
Parameter('Jak3_IC', 1000)
Parameter('Tyk2_IC', 1000)

Parameter('STAT1_IC', 1000)
Parameter('STAT2_IC', 1000)
Parameter('STAT3_IC', 1000)
Parameter('STAT4_IC', 1000)
Parameter('STAT5_IC', 1000)

# Phosphorylation rates
Parameter('kp_STAT1_Jak1', 5.5E-4)
Parameter('kp_STAT2_Jak1', 5.5E-4)
Parameter('kp_STAT3_Jak1', 5.5E-4)
Parameter('kp_STAT4_Jak1', 5.5E-4)
Parameter('kp_STAT5_Jak1', 5.5E-4)

Parameter('kp_STAT1_Jak2', 5.5E-4)
Parameter('kp_STAT2_Jak2', 5.5E-4)
Parameter('kp_STAT3_Jak2', 5.5E-4)
Parameter('kp_STAT4_Jak2', 5.5E-4)
Parameter('kp_STAT5_Jak2', 5.5E-4)

Parameter('kp_STAT1_Jak3', 5.5E-4)
Parameter('kp_STAT2_Jak3', 5.5E-4)
Parameter('kp_STAT3_Jak3', 5.5E-4)
Parameter('kp_STAT4_Jak3', 5.5E-4)
Parameter('kp_STAT5_Jak3', 5.5E-4)

Parameter('kp_STAT1_Tyk2', 5.5E-4)
Parameter('kp_STAT2_Tyk2', 5.5E-4)
Parameter('kp_STAT3_Tyk2', 5.5E-4)
Parameter('kp_STAT4_Tyk2', 5.5E-4)
Parameter('kp_STAT5_Tyk2', 5.5E-4)

Parameter('ku_STAT1', 5.5E-4)
Parameter('ku_STAT2', 5.5E-4)
Parameter('ku_STAT3', 5.5E-4)
Parameter('ku_STAT4', 5.5E-4)
Parameter('ku_STAT5', 5.5E-4)

# IL10
Parameter('ka_Jak1_IL10RA', 5.5E-4)
Parameter('kd_Jak1_IL10RA', 5.5E-4)
Parameter('ka_Tyk2_IL10RB', 5.5E-4)
Parameter('kd_Tyk2_IL10RB', 5.5E-4)
Parameter('ka_IL10_IL10RA', 5.5E-4)
Parameter('kd_IL10_IL10RA', 5.5E-4)

Parameter('ka_IL10RAIL10_IL10RA', 5.5E-4)
Parameter('kd_IL10RAIL10_IL10RA', 5.5E-4)
Parameter('ka_RAIL10RA_IL10RB', 5.5E-4)
Parameter('kd_RAIL10RA_IL10RB', 5.5E-4)
Parameter('IL10_activation', 5.5E-4)
Parameter('IL10_deactivation', 5.5E-4)

# IL20
Parameter('ka_IL20RA_Jak2', 5.5E-4)
Parameter('kd_IL20RA_Jak2', 5.5E-4)
Parameter('ka_IL20_IL20RA', 5.5E-4)
Parameter('kd_IL20_IL20RA', 5.5E-4)
Parameter('ka_ANYIL20_IL20RA', 5.5E-4)
Parameter('kd_ANYIL20_IL20RA', 5.5E-4)
Parameter('ka_IL20_IL20RB', 5.5E-4)
Parameter('kd_IL20_IL20RB', 5.5E-4)
Parameter('ka_RAIL20_IL20RB', 5.5E-4)
Parameter('kd_RAIL20_IL20RB', 5.5E-4)
Parameter('ka_IL20_IL22RA1', 5.5E-4)
Parameter('kd_IL20_IL22RA1', 5.5E-4)
Parameter('ka_22RA1IL20', 5.5E-4)
Parameter('kd_22RA1IL20', 5.5E-4)
Parameter('IL20_activation', 5.5E-4)
Parameter('IL20_deactivation', 5.5E-4)

# IL12
Parameter('ka_IL12RB1_Tyk2', 5.5E-4)
Parameter('kd_IL12RB1_Tyk2', 5.5E-4)
Parameter('ka_IL12RB2_Jak2', 5.5E-4)
Parameter('kd_IL12RB2_Jak2', 5.5E-4)
Parameter('ka_IL12_IL12RB1', 5.5E-4)
Parameter('kd_IL12_IL12RB1', 5.5E-4)
Parameter('ka_IL12_IL12RB2', 5.5E-4)
Parameter('kd_IL12_IL12RB2', 5.5E-4)
Parameter('ka_RB1IL12_RB2', 5.5E-4)
Parameter('kd_RB1IL12_RB2', 5.5E-4)
Parameter('ka_RB2IL12_RB1', 5.5E-4)
Parameter('kd_RB2IL12_RB1', 5.5E-4)
Parameter('IL12_activation', 5.5E-4)
Parameter('IL12_deactivation', 5.5E-4)

# IL23
Parameter('ka_IL23RA_Jak2', 5.5E-4)
Parameter('kd_IL23RA_Jak2', 5.5E-4)
Parameter('ka_IL23_IL23RA', 5.5E-4)
Parameter('kd_IL23_IL23RA', 5.5E-4)
Parameter('ka_IL23_IL12RB1', 5.5E-4)
Parameter('kd_IL23_IL12RB1', 5.5E-4)
Parameter('ka_RAIL23_IL12RB1', 5.5E-4)
Parameter('kd_RAIL23_IL12RB1', 5.5E-4)
Parameter('ka_RB1IL23_IL23RA', 5.5E-4)
Parameter('kd_RB1IL23_IL23RA', 5.5E-4)
Parameter('IL23_activation', 5.5E-4)
Parameter('IL23_deactivation', 5.5E-4)

# Type 1 IFN
Parameter('ka_IFNAR1_Tyk2', 5.5E-4)
Parameter('kd_IFNAR1_Tyk2', 5.5E-4)
Parameter('ka_IFNAR2_Jak1', 5.5E-4)
Parameter('kd_IFNAR2_Jak1', 5.5E-4)
Parameter('IFNa_activation', 5.5E-4)
Parameter('IFNa_deactivation', 5.5E-4)
Parameter('IFNb_activation', 5.5E-4)
Parameter('IFNb_deactivation', 5.5E-4)

Parameter('ka_IFNa_R1', 3.321e-14)
Parameter('kd_IFNa_R1', 1)
Parameter('ka_IFNa_R2', 4.9817E-13)
Parameter('kd_IFNa_R2', 0.015)
Parameter('ka_IFNaR1_R2', 3.623E-4)
Parameter('kd_IFNaR1_R2', 3E-4)
Parameter('ka_IFNaR2_R1', 3.623E-4)
Parameter('kd_IFNaR2_R1', 0.3)

Parameter('ka_IFNb_R1', 4.98E-14)
Parameter('kd_IFNb_R1', 0.030)
Parameter('ka_IFNb_R2', 8.30e-13)
Parameter('kd_IFNb_R2', 0.002)
Parameter('ka_IFNbR1_R2', 3.62e-4)
Parameter('kd_IFNbR1_R2', 2.4e-5)
Parameter('ka_IFNbR2_R1', 3.62e-4)
Parameter('kd_IFNbR2_R1', 0.006)

# Type 2 IFN
Parameter('ka_IFNGR1_Jak1', 5.5E-4)
Parameter('kd_IFNGR1_Jak1', 5.5E-4)
Parameter('ka_IFNGR2_Tyk2', 5.5E-4)
Parameter('kd_IFNGR2_Tyk2', 5.5E-4)
Parameter('ka_IFNg_R1', 5.5E-4)
Parameter('kd_IFNg_R1', 5.5E-4)

Parameter('ka_R1_bind_R1', 4 * 3.14157 * 0.5E10 / (6.022E23 * 1E-5))
Parameter('kd_R1_bind_R1', 1E-4)
Parameter('ka_IFNgR1_R2', 4 * 3.14157 * 0.55E10 / (6.022E23 * 1.6286e-10))
Parameter('kd_IFNgR1_R2', 5.5E-4)
Parameter('IFNg_activation', 5.5E-4)
Parameter('IFNg_deactivation', 5.5E-4)

# Gamma chain
Parameter('ka_Gamma_Jak3', 5.5E-4)
Parameter('kd_Gamma_Jak3', 5.5E-4)
# IL23
Parameter('ka_IL21R_Jak1', 5.5E-4)
Parameter('kd_IL21R_Jak1', 5.5E-4)
Parameter('ka_IL12R_IL21', 5.5E-4)
Parameter('kd_IL12R_IL21', 5.5E-4)
Parameter('ka_RIL21_Gamma', 5.5E-4)
Parameter('kd_RIL21_Gamma', 5.5E-4)
Parameter('IL21_activation', 5.5E-4)
Parameter('IL21_deactivation', 5.5E-4)
# IL15
Parameter('ka_IL15RA_Jak1', 5.5E-4)
Parameter('kd_IL15RA_Jak1', 5.5E-4)
Parameter('ka_IL15RA_IL15', 5.5E-4)
Parameter('kd_IL15RA_IL15', 5.5E-4)
Parameter('ka_RIL15_Gamma', 5.5E-4)
Parameter('kd_RIL15_Gamma', 5.5E-4)
Parameter('IL15_activation', 5.5E-4)
Parameter('IL15_deactivation', 5.5E-4)

# IL9
Parameter('ka_IL9RA_Jak1', 5.5E-4)
Parameter('kd_IL9RA_Jak1', 5.5E-4)
Parameter('ka_IL9RA_IL9', 5.5E-4)
Parameter('kd_IL9RA_IL9', 5.5E-4)
Parameter('ka_RIL9_Gamma', 5.5E-4)
Parameter('kd_RIL9_Gamma', 5.5E-4)
Parameter('IL9_activation', 5.5E-4)
Parameter('IL9_deactivation', 5.5E-4)

# IL7
Parameter('ka_IL7RA_Jak1', 5.5E-4)
Parameter('kd_IL7RA_Jak1', 5.5E-4)
Parameter('ka_IL7RA_IL7', 5.5E-4)
Parameter('kd_IL7RA_IL7', 5.5E-4)
Parameter('ka_RIL7_Gamma', 5.5E-4)
Parameter('kd_RIL7_Gamma', 5.5E-4)
Parameter('IL7_activation', 5.5E-4)
Parameter('IL7_deactivation', 5.5E-4)

# IL4
Parameter('ka_IL4RA_Jak1', 5.5E-4)
Parameter('kd_IL4RA_Jak1', 5.5E-4)
Parameter('ka_IL4RA_IL4', 5.5E-4)
Parameter('kd_IL4RA_IL4', 5.5E-4)
Parameter('ka_RIL4_Gamma', 5.5E-4)
Parameter('kd_RIL4_Gamma', 5.5E-4)
Parameter('IL4_activation', 5.5E-4)
Parameter('IL4_deactivation', 5.5E-4)
# IL13
Parameter('ka_IL13RA1_Jak3', 5.5E-4)
Parameter('kd_IL13RA1_Jak3', 5.5E-4)
Parameter('ka_RIL4_IL13RA1', 5.5E-4)
Parameter('kd_RIL4_IL13RA1', 5.5E-4)
Parameter('IL413RA1_activation', 5.5E-4)
Parameter('IL413RA1_deactivation', 5.5E-4)

Parameter('ka_IL13RA1_Jak2', 5.5E-4)
Parameter('kd_IL13RA1_Jak2', 5.5E-4)
Parameter('ka_IL13RA1_Tyk2', 5.5E-4)
Parameter('kd_IL13RA1_Tyk2', 5.5E-4)
Parameter('ka_IL13RA1_IL13', 5.5E-4)
Parameter('kd_IL13RA1_IL13', 5.5E-4)
Parameter('ka_RA1IL13_IL4RA', 5.5E-4)
Parameter('kd_RA1IL13_IL4RA', 5.5E-4)
Parameter('IL13_Jak2_receptor_activation', 5.5E-4)
Parameter('IL13_Jak2_receptor_deactivation', 5.5E-4)
Parameter('IL13_Tyk2_receptor_activation', 5.5E-4)
Parameter('IL13_Tyk2_receptor_deactivation', 5.5E-4)

Parameter('ka_RA1IL13_IL13RA2', 5.5E-4)
Parameter('kd_RA1IL13_IL13RA2', 5.5E-4)
# IL2
Parameter('ka_IL2RB_Jak1', 5.5E-4)
Parameter('kd_IL2RB_Jak1', 5.5E-4)
Parameter('ka_IL2RB_Gamma', 5.5E-4)
Parameter('kd_IL2RB_Gamma', 5.5E-4)
Parameter('ka_IL2RBGamma_IL2RA', 5.5E-4)
Parameter('kd_IL2RBGamma_IL2RA', 5.5E-4)
Parameter('ka_IL2RABGamma_IL2', 5.5E-4)
Parameter('kd_IL2RABGamma_IL2', 5.5E-4)
Parameter('ka_IL2RBGamma_IL2', 5.5E-4)
Parameter('kd_IL2RBGamma_IL2', 5.5E-4)
Parameter('ka_IL2RA_IL2', 5.5E-4)
Parameter('kd_IL2RA_IL2', 5.5E-4)
Parameter('IL2_activation', 5.5E-4)
Parameter('IL2_deactivation', 5.5E-4)

# Beta_c
Parameter('ka_BetaC_Jak2', 5.5E-4)
Parameter('kd_BetaC_Jak2', 5.5E-4)
Parameter('ka_BetaC_BetaC', 5.5E-4)
Parameter('kd_BetaC_BetaC', 5.5E-4)

# GM-CSF
Parameter('ka_GMCSFRA_GMCSF', 5.5E-4)
Parameter('kd_GMCSFRA_GMCSF', 5.5E-4)
Parameter('ka_RAGMCSF_BetaCdimer', 5.5E-4)
Parameter('kd_RAGMCSF_BetaCdimer', 5.5E-4)
Parameter('ka_RAGMCSF_hexamerisation', 5.5E-4)
Parameter('kd_RAGMCSF_hexamerisation', 5.5E-4)
Parameter('ka_GMCSF_dodecamerisation', 5.5E-4)
Parameter('kd_GMCSF_dodecamerisation', 5.5E-4)
Parameter('GMCSF_activation', 5.5E-4)
Parameter('GMCSF_deactivation', 5.5E-4)

# IL3
Parameter('ka_IL3RA_IL3', 5.5E-4)
Parameter('kd_IL3RA_IL3', 5.5E-4)
Parameter('ka_RAIL3_BetaCdimer', 5.5E-4)
Parameter('kd_RAIL3_BetaCdimer', 5.5E-4)
Parameter('ka_RAIL3_hexamerisation', 5.5E-4)
Parameter('kd_RAIL3_hexamerisation', 5.5E-4)
Parameter('ka_IL3_dodecamerisation', 5.5E-4)
Parameter('kd_IL3_dodecamerisation', 5.5E-4)
Parameter('IL3_activation', 5.5E-4)
Parameter('IL3_deactivation', 5.5E-4)

# IL5
Parameter('ka_IL5RA_IL5', 5.5E-4)
Parameter('kd_IL5RA_IL5', 5.5E-4)
Parameter('ka_RAIL5_BetaCdimer', 5.5E-4)
Parameter('kd_RAIL5_BetaCdimer', 5.5E-4)
Parameter('ka_RAIL5_hexamerisation', 5.5E-4)
Parameter('kd_RAIL5_hexamerisation', 5.5E-4)
Parameter('ka_IL5_dodecamerisation', 5.5E-4)
Parameter('kd_IL5_dodecamerisation', 5.5E-4)
Parameter('IL5_activation', 5.5E-4)
Parameter('IL5_deactivation', 5.5E-4)

# IL6
Parameter('ka_IL6RA_Jak1', 5.5E-4)
Parameter('kd_IL6RA_Jak1', 5.5E-4)
Parameter('ka_IL6_IL6RA', 5.5E-4)
Parameter('kd_IL6_IL6RA', 5.5E-4)
Parameter('ka_RAIL6_GP130', 5.5E-4)
Parameter('kd_RAIL6_GP130', 5.5E-4)
Parameter('ka_RAIL6GP130_dimerisation', 5.5E-4)
Parameter('kd_RAIL6GP130_dimerisation', 5.5E-4)
Parameter('IL6_activation', 5.5E-4)
Parameter('IL6_deactivation', 5.5E-4)

# GCSF
Parameter('ka_GCSFR_Jak2', 5.5E-4)
Parameter('kd_GCSFR_Jak2', 5.5E-4)
Parameter('ka_GCSF_GCSFR', 5.5E-4)
Parameter('kd_GCSF_GCSFR', 5.5E-4)
Parameter('ka_RGCSF_dimerisation', 5.5E-4)
Parameter('kd_RGCSF_dimerisation', 5.5E-4)
Parameter('GCSF_deactivation', 5.5E-4)
Parameter('GCSF_activation', 5.5E-4)

# IL11
Parameter('ka_IL11RA_Jak2', 5.5E-4)
Parameter('kd_IL11RA_Jak2', 5.5E-4)
Parameter('ka_IL11_IL11RA', 5.5E-4)
Parameter('kd_IL11_IL11RA', 5.5E-4)
Parameter('ka_RAIL11_GP130', 5.5E-4)
Parameter('kd_RAIL11_GP130', 5.5E-4)
Parameter('ka_RAIL11GP130_dimerisation', 5.5E-4)
Parameter('kd_RAIL11GP130_dimerisation', 5.5E-4)
Parameter('IL11_activation', 5.5E-4)
Parameter('IL11_deactivation', 5.5E-4)

# IL27
Parameter('ka_IL27RA_Jak2', 5.5E-4)
Parameter('kd_IL27RA_Jak2', 5.5E-4)
Parameter('ka_IL27_IL27RA', 5.5E-4)
Parameter('kd_IL27_IL27RA', 5.5E-4)
Parameter('ka_RAIL27_GP130', 5.5E-4)
Parameter('kd_RAIL27_GP130', 5.5E-4)
Parameter('ka_RAIL27GP130_dimerisation', 5.5E-4)
Parameter('kd_RAIL27GP130_dimerisation', 5.5E-4)
Parameter('IL27_activation', 5.5E-4)
Parameter('IL27_deactivation', 5.5E-4)

# =============================================================================
# # Molecules
# =============================================================================
# IL10
Monomer('IL10', ['s1', 's2'])
Monomer('IL10RA', ['r1', 'r2', 'r3'])
Monomer('IL10RB', ['r1',       'r3'])

# IL20 Family
Monomer('IL20', ['s1', 's2'])
Monomer('IL20RA', ['r1',       'r3'])
Monomer('IL20RB', ['r1', 'r2'])
Monomer('IL22RA1', ['r1', 'r2'])

# IL12 Family
Monomer('IL12', ['s1','s2'])
Monomer('IL12RB1', ['r1',       'r3'])
Monomer('IL12RB2', ['r1',       'r3'])
# IL23
Monomer('IL23', ['s1','s2'])
Monomer('IL23RA', ['r1',       'r3'])

# Type 1 IFN
Monomer('IFN_alpha2',['s1','s2'])
Monomer('IFN_beta',['s1','s2'])
Monomer('IFNAR1',['r1', 'r3'])
Monomer('IFNAR2',['r1', 'r3'])

# Type 2 IFN
Monomer('IFN_gamma',['s1','s2'])
Monomer('IFNGR1', ['r1', 'r2', 'r3', 'r4'])
Monomer('IFNGR2', ['r1', 'r2', 'r3', 'r4'])

# Gamma Chain (shared)
Monomer('Gamma_c', ['r1',   'r3'])
# IL13 interacts with IL4
Monomer('IL21', ['s1', 's2'])
Monomer('IL21R', ['r1',     'r3'])
Monomer('IL15', ['s1', 's2'])
Monomer('IL15RA', ['r1',     'r3'])
Monomer('IL9', ['s1', 's2'])
Monomer('IL9RA', ['r1',     'r3'])
Monomer('IL7', ['s1', 's2'])
Monomer('IL7RA', ['r1',     'r3'])
Monomer('IL4', ['s1', 's2'])
Monomer('IL4RA', ['r1',     'r3'])
Monomer('IL13', ['s1', 's2'])
Monomer('IL13RA1', ['r1',     'r3'])
Monomer('IL13RA2', ['r1'])
Monomer('IL2', ['s1', 's2'])
Monomer('IL2RA', ['r1','r2','r3'])
Monomer('IL2RB', ['r1','r2','r3','r4'])

# Granulocyte CSF receptor (promotes neutrophil proliferation)
Monomer('GCSF', ['s1', 's2'])
Monomer('GCSFR', ['r1', 'r2', 'r3'])

# GP130 (shared)
Monomer('GP130', ['r1','r2'])
Monomer('IL11', ['s1', 's2'])
Monomer('IL11RA', ['r1','r2','r3'])
Monomer('IL6', ['s1', 's2'])
Monomer('IL6RA', ['r1','r2','r3'])
Monomer('IL27', ['s1', 's2'])
Monomer('IL27RA', ['r1','r2','r3'])

# Beta_c (shared)
Monomer('Beta_c', ['r1', 'r2', 'r3'])
Monomer('GMCSF', ['s1', 's2', 's3'])
Monomer('GMCSFRB', ['r1', 'r3'])
Monomer('GMCSFRA', ['r1', 'r3'])
Monomer('IL3', ['s1', 's2', 's3'])
Monomer('IL3RA', ['r1', 'r3'])
Monomer('IL5', ['s1', 's2', 's3'])
Monomer('IL5RA', ['r1', 'r3'])


# STATs
Monomer('STAT1', ['p1'], {'p1':['U', 'P']})
Monomer('STAT2', ['p1'], {'p1':['U', 'P']})
Monomer('STAT3', ['p1'], {'p1':['U', 'P']})
Monomer('STAT4', ['p1'], {'p1':['U', 'P']})
Monomer('STAT5', ['p1'], {'p1':['U', 'P']}) # I suspect that this is STAT5a, not STAT5b

# Kinases
Monomer('Jak1', ['s1', 'status'], {'status':['inactive','active']})
Monomer('Jak2', ['s1', 'status'], {'status':['inactive','active']})
Monomer('Jak3', ['s1', 'status'], {'status':['inactive','active']})
Monomer('Tyk2', ['s1', 'status'], {'status':['inactive','active']})

# =============================================================================
# # Seed Species
# =============================================================================
Initial(IL10(s1=None,s2=None), IL10_IC)
Initial(IL10RA(r1=None,r2=None,r3=None), IL10RA_IC)
Initial(IL10RB(r1=None,r3=None), IL10RB_IC)

Initial(IL20(s1=None,s2=None), IL20_IC)
Initial(IL20RA(r1=None,r3=None), IL20RA_IC)
Initial(IL20RB(r1=None,r2=None), IL20RB_IC)
Initial(IL22RA1(r1=None,r2=None), IL22RA1_IC)

Initial(IL12(s1=None,s2=None), IL12_IC)
Initial(IL12RB1(r1=None,r3=None), IL12RB1_IC)
Initial(IL12RB2(r1=None,r3=None), IL12RB2_IC)
Initial(IL23RA(r1=None,r3=None), IL23RA_IC)

Initial(IFN_alpha2(s1=None,s2=None), IFN_alpha2_IC)
Initial(IFN_beta(s1=None,s2=None), IFN_beta_IC)
Initial(IFNAR1(r1=None, r3=None), IFNAR1_IC)
Initial(IFNAR2(r1=None, r3=None), IFNAR2_IC)

Initial(IFN_gamma(s1=None,s2=None), IFN_gamma_IC)
Initial(IFNGR1(r1=None, r2=None, r3=None, r4=None), IFNGR1_IC)
Initial(IFNGR2(r1=None, r2=None, r3=None, r4=None), IFNGR2_IC)

Initial(Gamma_c(r1=None,r3=None), Gamma_c_IC)
Initial(IL21(s1=None,s2=None), IL21_IC)
Initial(IL21R(r1=None,r3=None), IL21R_IC)
Initial(IL15(s1=None,s2=None), IL15_IC)
Initial(IL15RA(r1=None,r3=None), IL15RA_IC)
Initial(IL9(s1=None,s2=None), IL9_IC)
Initial(IL9RA(r1=None,r3=None), IL9RA_IC)
Initial(IL7(s1=None,s2=None), IL7_IC)
Initial(IL7RA(r1=None,r3=None), IL7RA_IC)
Initial(IL4(s1=None,s2=None), IL4_IC)
Initial(IL4RA(r1=None,r3=None), IL4RA_IC)
Initial(IL13RA1(r1=None,r3=None), IL13RA1_IC)
Initial(IL13RA2(r1=None), IL13RA2_IC)
Initial(IL13(s1=None,s2=None), IL13_IC)
Initial(IL2(s1=None,s2=None), IL2_IC)
Initial(IL2RA(r1=None,r2=None,r3=None), IL2RA_IC)
Initial(IL2RB(r1=None,r2=None,r3=None,r4=None), IL2RB_IC)
Initial(Beta_c(r1=None,r2=None,r3=None), Beta_c_IC)
Initial(GMCSF(s1=None, s2=None,s3=None), GMCSF_IC)
Initial(GMCSFRA(r1=None,r3=None), GMCSFRA_IC)
Initial(IL3(s1=None, s2=None,s3=None), IL3_IC)
Initial(IL3RA(r1=None,r3=None), IL3RA_IC)
Initial(IL5(s1=None, s2=None,s3=None), IL5_IC)
Initial(IL5RA(r1=None,r3=None), IL5RA_IC)
Initial(GP130(r1=None,r2=None), GP130_IC)
Initial(IL6(s1=None, s2=None), IL6_IC)
Initial(IL6RA(r1=None,r2=None,r3=None), IL6RA_IC)
Initial(GCSF(s1=None, s2=None), GCSF_IC)
Initial(GCSFR(r1=None,r2=None,r3=None), GCSFR_IC)
Initial(IL11(s1=None, s2=None), IL11_IC)
Initial(IL11RA(r1=None,r2=None,r3=None), IL11RA_IC)
Initial(IL27(s1=None, s2=None), IL27_IC)
Initial(IL27RA(r1=None,r2=None,r3=None), IL27RA_IC)

Initial(Jak1(s1=None,status='inactive'), Jak1_IC)
Initial(Jak2(s1=None,status='inactive'), Jak2_IC)
Initial(Jak3(s1=None,status='inactive'), Jak3_IC)
Initial(Tyk2(s1=None,status='inactive'), Tyk2_IC)

Initial(STAT1(p1='U'), STAT1_IC)
Initial(STAT2(p1='U'), STAT2_IC)
Initial(STAT3(p1='U'), STAT3_IC)
Initial(STAT4(p1='U'), STAT4_IC)
Initial(STAT5(p1='U'), STAT5_IC)
# =============================================================================
# # Observables
# Use 'WILD' for ?, use 'ANY' for +
# =============================================================================
# IL10
Observable('Active_IL10_Receptor', Tyk2(s1=1,status='active')%IL10RB(r1=2, r3=1)%Jak1(s1=3,status='active')%IL10RA(r1=4, r2=2, r3=3)%IL10(s1=4, s2=5)%IL10RA(r1=5, r2=6, r3=7)%Jak1(s1=7,status='active')%IL10RB(r1=6, r3=8)%Tyk2(s1=8,status='active'))
# IL20
Observable('Active_IL20_Type1', Jak2(s1=3, status='active')%IL20RA(r1=1, r3=3)%IL20(s1=1, s2=2)%IL20RB(r1=2))
Observable('Active_IL20_Type2', Jak2(s1=3, status='active')%IL20RA(r1=1, r3=3)%IL20(s1=1, s2=2)%IL22RA1(r1=2))
# IL12
Observable('Active_IL12_Receptor', Tyk2(s1=3, status='active')%IL12RB1(r1=1,r3=3)%IL12(s1=1,s2=2)%IL12RB2(r1=2,r3=4)%Jak2(s1=4, status='active'))
# IL23
Observable('Active_IL23_Receptor', Tyk2(s1=3, status='active')%IL12RB1(r1=1,r3=3)%IL12(s1=1,s2=2)%IL23RA(r1=2,r3=4)%Jak2(s1=4, status='active'))
# Type 1 IFN
Observable('Active_IFNa_Receptor', Tyk2(s1=3)%IFNAR1(r1=1, r3=3)%IFN_alpha2(s1=1,s2=2)%IFNAR2(r1=2, r3=4)%Jak1(s1=4))
Observable('Active_IFNb_Receptor', Tyk2(s1=3)%IFNAR1(r1=1, r3=3)%IFN_beta(s1=1,s2=2)%IFNAR2(r1=2, r3=4)%Jak1(s1=4))
# Type 2 IFN
Observable('Active_IFNg_Receptor', Tyk2(s1=8,status='active') % Tyk2(s1=7,status='active') % Jak1(s1=6,status='active') % Jak1(s1=5,status='active') % IFNGR2(r1=3, r3=7) % IFNGR1(r1=1, r2=3, r3=5, r4=20) % IFN_gamma(s1=1, s2=2) % IFNGR1(r1=2, r2=4, r3=6, r4=20) % IFNGR2(r1=4, r3=8))
# IL21
Observable('Active_IL21_Receptor', Jak1(s1=3,status='active')%IL21R(r1=1, r3=3)%IL21(s1=1, s2=2)%Gamma_c(r1=2, r3=4)%Jak3(s1=4,status='active'))
# IL15
Observable('Active_IL15_Receptor', Jak1(s1=3,status='active')%IL15RA(r1=1, r3=3)%IL15(s1=1, s2=2)%Gamma_c(r1=2, r3=4)%Jak3(s1=4,status='active'))
# IL9
Observable('Active_IL9_Receptor', Jak1(s1=3,status='active')%IL9RA(r1=1, r3=3)%IL9(s1=1, s2=2)%Gamma_c(r1=2, r3=4)%Jak3(s1=4,status='active'))
# IL7
Observable('Active_IL7_Receptor', Jak1(s1=3,status='active')%IL7RA(r1=1, r3=3)%IL7(s1=1, s2=2)%Gamma_c(r1=2, r3=4)%Jak3(s1=4,status='active'))
# IL4
Observable('Active_IL4_Receptor', Jak1(s1=3,status='active')%IL4RA(r1=1, r3=3)%IL4(s1=1, s2=2)%Gamma_c(r1=2, r3=4)%Jak3(s1=4,status='active'))
Observable('Active_IL4_IL13RA1_Receptor', Jak1(s1=3,status='active')%IL4RA(r1=1, r3=3)%IL4(s1=1, s2=2)%IL13RA1(r1=2, r3=4)%Jak3(s1=4,status='active'))
# IL13
Observable('Active_Tyk2_IL13_Receptor', Tyk2(s1=4,status='active')%IL13RA1(r1=1,r3=4)%IL13(s1=1,s2=2)%IL4RA(r1=2,r3=3)%Jak1(s1=3,status='active'))
Observable('Active_Jak2_IL13_Receptor', Jak2(s1=4,status='active')%IL13RA1(r1=1,r3=4)%IL13(s1=1,s2=2)%IL4RA(r1=2,r3=3)%Jak1(s1=3,status='active'))
Observable('Inactive_IL13_Receptor', Jak2(s1=3,status='inactive')%IL13RA1(r1=1, r3=3)%IL13(s1=1,s2=2)%IL13RA2(r1=2))
# IL2
Observable('Active_IL2_Receptor', Jak1(s1=5,status='active')%IL2RB(r1=2,r3=5,r4=1)%Gamma_c(r1=1,r3=6)%IL2(s2=2)%Jak3(s1=6,status='active'))
# Beta_c
Observable('Beta_c_dimer', Jak2(s1=2, status='inactive')%Beta_c(r2=1,r3=2)%Beta_c(r2=1,r3=3)%Jak2(s1=3, status='inactive'))
# GM-CSF
Observable('GM_CSF_BetaC_dimer_GM_CSF', GMCSFRA(r1=2)%GMCSF(s1=2, s2=3)%Jak2(s1=5, status='inactive')%Beta_c(r1=3, r2=1,r3=5)%Beta_c(r2=1,r3=4)%Jak2(s1=4, status='inactive'))
Observable('GM_CSF_Hexamer', GMCSFRA(r1=1)%GMCSF(s1=1,s2=2)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3)%GMCSFRA(r1=5)%GMCSF(s1=5,s2=4,s3=None))
Observable('GM_CSF_Dodecamer', GMCSFRA(r1=1)%GMCSF(s1=1,s2=2)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3,r3=21)%Jak2(s1=21, status='active')%GMCSFRA(r1=5)%GMCSF(s1=5,s2=4,s3=12)%GMCSFRA(r1=6)%GMCSF(s1=6,s2=7,s3=12)%Beta_c(r1=7,r2=8,r3=20)%Jak2(s1=20, status='active')%Beta_c(r1=9,r2=8)%GMCSFRA(r1=10)%GMCSF(s1=10,s2=9))
# IL3
Observable('IL3_Dodecamer', IL3RA(r1=1)%IL3(s1=1,s2=2)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3,r3=21)%Jak2(s1=21, status='active')%IL3RA(r1=5)%IL3(s1=5,s2=4,s3=12)%IL3RA(r1=6)%IL3(s1=6,s2=7,s3=12)%Beta_c(r1=7,r2=8,r3=20)%Jak2(s1=20, status='active')%Beta_c(r1=9,r2=8)%IL3RA(r1=10)%IL3(s1=10,s2=9))
# IL5
Observable('IL5_Dodecamer', IL5RA(r1=1)%IL5(s1=1,s2=2)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3,r3=21)%Jak2(s1=21, status='active')%IL5RA(r1=5)%IL5(s1=5,s2=4,s3=12)%IL5RA(r1=6)%IL5(s1=6,s2=7,s3=12)%Beta_c(r1=7,r2=8,r3=20)%Jak2(s1=20, status='active')%Beta_c(r1=9,r2=8)%IL5RA(r1=10)%IL5(s1=10,s2=9))
# IL6
Observable('IL6_Receptor', Jak1(s1=5,status='active')%IL6(s1=1, s2=2)%IL6RA(r1=1,r2=4,r3=5)%GP130(r1=2)%IL6(s1=11, s2=12)%IL6RA(r1=11,r2=4,r3=16)%GP130(r1=12)%Jak1(s1=16,status='active'))
# GCSF
Observable('GCSF_Receptor', Jak2(s1=15,status='active')%GCSF(s1=1)%GCSFR(r1=1,r2=4,r3=15)%GCSF(s1=3)%GCSFR(r1=3,r2=4,r3=16)%Jak2(s1=16,status='active'))
# IL11
Observable('IL11_Receptor', Jak2(s1=5,status='active')%IL11(s1=1, s2=2)%IL11RA(r1=1,r2=4,r3=5)%GP130(r1=2)%IL11(s1=11, s2=12)%IL11RA(r1=11,r2=4,r3=16)%GP130(r1=12)%Jak2(s1=16,status='active'))
# IL27
Observable('IL27_Receptor', Jak2(s1=5,status='active')%IL27(s1=1, s2=2)%IL27RA(r1=1,r2=4,r3=5)%GP130(r1=2)%IL27(s1=11, s2=12)%IL27RA(r1=11,r2=4,r3=16)%GP130(r1=12)%Jak2(s1=16,status='active'))

# STATs
Observable('pSTAT1', STAT1(p1='P'))
Observable('pSTAT2', STAT2(p1='P'))
Observable('pSTAT3', STAT3(p1='P'))
Observable('pSTAT4', STAT4(p1='P'))
Observable('pSTAT5', STAT5(p1='P'))
# =============================================================================
# # Reaction rules
# =============================================================================
# ---------------------
# STAT phosphorylation
# ---------------------
Rule('Jak1_STAT1', Jak1(status='active') + STAT1(p1='U') >> Jak1(status='active') + STAT1(p1='P'), kp_STAT1_Jak1)
Rule('Jak1_STAT2', Jak1(status='active') + STAT2(p1='U') >> Jak1(status='active') + STAT2(p1='P'), kp_STAT2_Jak1)
Rule('Jak1_STAT3', Jak1(status='active') + STAT3(p1='U') >> Jak1(status='active') + STAT3(p1='P'), kp_STAT3_Jak1)
Rule('Jak1_STAT4', Jak1(status='active') + STAT4(p1='U') >> Jak1(status='active') + STAT4(p1='P'), kp_STAT4_Jak1)
Rule('Jak1_STAT5', Jak1(status='active') + STAT5(p1='U') >> Jak1(status='active') + STAT5(p1='P'), kp_STAT5_Jak1)

Rule('Jak2_STAT1', Jak2(status='active') + STAT1(p1='U') >> Jak2(status='active') + STAT1(p1='P'), kp_STAT1_Jak2)
Rule('Jak2_STAT2', Jak2(status='active') + STAT2(p1='U') >> Jak2(status='active') + STAT2(p1='P'), kp_STAT2_Jak2)
Rule('Jak2_STAT3', Jak2(status='active') + STAT3(p1='U') >> Jak2(status='active') + STAT3(p1='P'), kp_STAT3_Jak2)
Rule('Jak2_STAT4', Jak2(status='active') + STAT4(p1='U') >> Jak2(status='active') + STAT4(p1='P'), kp_STAT4_Jak2)
Rule('Jak2_STAT5', Jak2(status='active') + STAT5(p1='U') >> Jak2(status='active') + STAT5(p1='P'), kp_STAT5_Jak2)

Rule('Jak3_STAT1', Jak3(status='active') + STAT1(p1='U') >> Jak3(status='active') + STAT1(p1='P'), kp_STAT1_Jak3)
Rule('Jak3_STAT2', Jak3(status='active') + STAT2(p1='U') >> Jak3(status='active') + STAT2(p1='P'), kp_STAT2_Jak3)
Rule('Jak3_STAT3', Jak3(status='active') + STAT3(p1='U') >> Jak3(status='active') + STAT3(p1='P'), kp_STAT3_Jak3)
Rule('Jak3_STAT4', Jak3(status='active') + STAT4(p1='U') >> Jak3(status='active') + STAT4(p1='P'), kp_STAT4_Jak3)
Rule('Jak3_STAT5', Jak3(status='active') + STAT5(p1='U') >> Jak3(status='active') + STAT5(p1='P'), kp_STAT5_Jak3)

Rule('Tyk2_STAT1', Tyk2(status='active') + STAT1(p1='U') >> Tyk2(status='active') + STAT1(p1='P'), kp_STAT1_Tyk2)
Rule('Tyk2_STAT2', Tyk2(status='active') + STAT2(p1='U') >> Tyk2(status='active') + STAT2(p1='P'), kp_STAT2_Tyk2)
Rule('Tyk2_STAT3', Tyk2(status='active') + STAT3(p1='U') >> Tyk2(status='active') + STAT3(p1='P'), kp_STAT3_Tyk2)
Rule('Tyk2_STAT4', Tyk2(status='active') + STAT4(p1='U') >> Tyk2(status='active') + STAT4(p1='P'), kp_STAT4_Tyk2)
Rule('Tyk2_STAT5', Tyk2(status='active') + STAT5(p1='U') >> Tyk2(status='active') + STAT5(p1='P'), kp_STAT5_Tyk2)

Rule('STAT1_dephos', STAT1(p1='P') >> STAT1(p1='U'), ku_STAT1)
Rule('STAT2_dephos', STAT2(p1='P') >> STAT2(p1='U'), ku_STAT2)
Rule('STAT3_dephos', STAT3(p1='P') >> STAT3(p1='U'), ku_STAT3)
Rule('STAT4_dephos', STAT4(p1='P') >> STAT4(p1='U'), ku_STAT4)
Rule('STAT5_dephos', STAT5(p1='P') >> STAT5(p1='U'), ku_STAT5)

# --------------
# GP130 family
# --------------
# G-CSFR + G-CSF <-> 2x(G-CSFR.G-CSF) -> active rececptor + JAK2 -> pSTAT3
##Rule('GCSFR_binds_Jak2', GCSFR(r3=None) + Jak2(s1=None, status='inactive') | GCSFR(r3=1)%Jak2(s1=1, status='inactive'), ka_GCSFR_Jak2, kd_GCSFR_Jak2)
##Rule('GCSF_binds_GCSFR', GCSF(s1=None) + GCSFR(r1=None,r2=None) | GCSF(s1=1)%GCSFR(r1=1,r2=None), ka_GCSF_GCSFR, kd_GCSF_GCSFR)
##Rule('RGCSF_dimerisation', GCSF(s1=1)%GCSFR(r1=1,r2=None) + GCSF(s1=1)%GCSFR(r1=1,r2=None) | GCSF(s1=1)%GCSFR(r1=1,r2=4)%GCSF(s1=3)%GCSFR(r1=3,r2=4), ka_RGCSF_dimerisation, kd_RGCSF_dimerisation)
##Rule('GCSF_receptor_state_change', Jak2(s1=15,status='inactive')%GCSF(s1=1)%GCSFR(r1=1,r2=4,r3=15)%GCSF(s1=3)%GCSFR(r1=3,r2=4,r3=16)%Jak2(s1=16,status='inactive') | Jak2(s1=15,status='active')%GCSF(s1=1)%GCSFR(r1=1,r2=4,r3=15)%GCSF(s1=3)%GCSFR(r1=3,r2=4,r3=16)%Jak2(s1=16,status='active'), GCSF_activation, GCSF_deactivation)

# IL6RA + IL6 <-> IL6RA.IL6 + GP130 <-> 2x(IL6RA.IL6.GP130) + Jak1 -> pSTAT3
#Rule('IL6RA_binds_Jak1', IL6RA(r3=None) + Jak1(s1=None, status='inactive') | IL6RA(r3=1)%Jak1(s1=1, status='inactive'), ka_IL6RA_Jak1, kd_IL6RA_Jak1)
#Rule('IL6_binds_IL6RA', IL6(s1=None) + IL6RA(r1=None,r2=None) | IL6(s1=1)%IL6RA(r1=1,r2=None), ka_IL6_IL6RA, kd_IL6_IL6RA)
#Rule('RAIL6_binds_GP130', IL6(s1=1, s2=None)%IL6RA(r1=1,r2=None) + GP130(r1=None) | IL6(s1=1, s2=2)%IL6RA(r1=1,r2=None)%GP130(r1=2), ka_RAIL6_GP130, kd_RAIL6_GP130)
#Rule('RAIL6GP130_dimerisation', IL6(s1=1, s2=2)%IL6RA(r1=1,r2=None)%GP130(r1=2) + IL6(s1=1, s2=2)%IL6RA(r1=1,r2=None)%GP130(r1=2) | IL6(s1=1, s2=2)%IL6RA(r1=1,r2=4)%GP130(r1=2)%IL6(s1=6, s2=7)%IL6RA(r1=6,r2=4)%GP130(r1=7), ka_RAIL6GP130_dimerisation, kd_RAIL6GP130_dimerisation)
#Rule('IL6_receptor_state_change', Jak1(s1=5,status='inactive')%IL6(s1=1, s2=2)%IL6RA(r1=1,r2=4,r3=5)%GP130(r1=2)%IL6(s1=11, s2=12)%IL6RA(r1=11,r2=4,r3=16)%GP130(r1=12)%Jak1(s1=16,status='inactive') | Jak1(s1=5,status='active')%IL6(s1=1, s2=2)%IL6RA(r1=1,r2=4,r3=5)%GP130(r1=2)%IL6(s1=11, s2=12)%IL6RA(r1=11,r2=4,r3=16)%GP130(r1=12)%Jak1(s1=16,status='active'), IL6_activation, IL6_deactivation)

# IL11RA + IL11 <-> IL11RA.IL11 + GP130 <-> 2x(IL11RA.IL11.GP130) + Jak2 -> pSTAT1
#Rule('IL11RA_binds_Jak2', IL11RA(r3=None) + Jak2(s1=None, status='inactive') | IL11RA(r3=1)%Jak2(s1=1, status='inactive'), ka_IL11RA_Jak2, kd_IL11RA_Jak2)
#Rule('IL11_binds_IL11RA', IL11(s1=None) + IL11RA(r1=None,r2=None) | IL11(s1=1)%IL11RA(r1=1,r2=None), ka_IL11_IL11RA, kd_IL11_IL11RA)
#Rule('RAIL11_binds_GP130', IL11(s1=1, s2=None)%IL11RA(r1=1,r2=None) + GP130(r1=None) | IL11(s1=1, s2=2)%IL11RA(r1=1,r2=None)%GP130(r1=2), ka_RAIL11_GP130, kd_RAIL11_GP130)
#Rule('RAIL11GP130_dimerisation', IL11(s1=1, s2=2)%IL11RA(r1=1,r2=None)%GP130(r1=2) + IL11(s1=1, s2=2)%IL11RA(r1=1,r2=None)%GP130(r1=2) | IL11(s1=1, s2=2)%IL11RA(r1=1,r2=4)%GP130(r1=2)%IL11(s1=6, s2=7)%IL11RA(r1=6,r2=4)%GP130(r1=7), ka_RAIL11GP130_dimerisation, kd_RAIL11GP130_dimerisation)
#Rule('IL11_receptor_state_change', Jak2(s1=5,status='inactive')%IL11(s1=1, s2=2)%IL11RA(r1=1,r2=4,r3=5)%GP130(r1=2)%IL11(s1=11, s2=12)%IL11RA(r1=11,r2=4,r3=16)%GP130(r1=12)%Jak2(s1=16,status='inactive') | Jak2(s1=5,status='active')%IL11(s1=1, s2=2)%IL11RA(r1=1,r2=4,r3=5)%GP130(r1=2)%IL11(s1=11, s2=12)%IL11RA(r1=11,r2=4,r3=16)%GP130(r1=12)%Jak2(s1=16,status='active'), IL11_activation, IL11_deactivation)

# IL27RA + IL27 <-> IL27RA.IL27 + GP130 <-> 2x(IL27RA.IL27.GP130) + Jak2 -> pSTAT1 (?)
#Rule('IL27RA_binds_Jak2', IL27RA(r3=None) + Jak2(s1=None, status='inactive') | IL27RA(r3=1)%Jak2(s1=1, status='inactive'), ka_IL27RA_Jak2, kd_IL27RA_Jak2)
#Rule('IL27_binds_IL27RA', IL27(s1=None) + IL27RA(r1=None,r2=None) | IL27(s1=1)%IL27RA(r1=1,r2=None), ka_IL27_IL27RA, kd_IL27_IL27RA)
#Rule('RAIL27_binds_GP130', IL27(s1=1, s2=None)%IL27RA(r1=1,r2=None) + GP130(r1=None) | IL27(s1=1, s2=2)%IL27RA(r1=1,r2=None)%GP130(r1=2), ka_RAIL27_GP130, kd_RAIL27_GP130)
#Rule('RAIL27GP130_dimerisation', IL27(s1=1, s2=2)%IL27RA(r1=1,r2=None)%GP130(r1=2) + IL27(s1=1, s2=2)%IL27RA(r1=1,r2=None)%GP130(r1=2) | IL27(s1=1, s2=2)%IL27RA(r1=1,r2=4)%GP130(r1=2)%IL27(s1=6, s2=7)%IL27RA(r1=6,r2=4)%GP130(r1=7), ka_RAIL27GP130_dimerisation, kd_RAIL27GP130_dimerisation)
#Rule('IL27_receptor_state_change', Jak2(s1=5,status='inactive')%IL27(s1=1, s2=2)%IL27RA(r1=1,r2=4,r3=5)%GP130(r1=2)%IL27(s1=11, s2=12)%IL27RA(r1=11,r2=4,r3=16)%GP130(r1=12)%Jak2(s1=16,status='inactive') | Jak2(s1=5,status='active')%IL27(s1=1, s2=2)%IL27RA(r1=1,r2=4,r3=5)%GP130(r1=2)%IL27(s1=11, s2=12)%IL27RA(r1=11,r2=4,r3=16)%GP130(r1=12)%Jak2(s1=16,status='active'), IL27_activation, IL27_deactivation)

# Beta_c family
#Rule('BetaC_binds_Jak2', Beta_c(r3=None) + Jak2(s1=None, status='inactive') | Beta_c(r3=1)%Jak2(s1=1, status='inactive'), ka_BetaC_Jak2, kd_BetaC_Jak2)
#Rule('BetaC_binds_BetaC', Beta_c(r1=None,r2=None) + Beta_c(r1=None,r2=None) | Beta_c(r1=None,r2=1)%Beta_c(r1=None,r2=1), ka_BetaC_BetaC, kd_BetaC_BetaC)
# GM-CSFRA binds GM-CSF, then binds (Jak2.Beta_c.Beta_c.Jak2), then
#   binds another GM-CSFRA.GM-CSF, then this hexamer binds another
#   hexamer to form a dodecamer which phosphorylates STAT5
#Rule('GMCSFRA_binds_GMCSF', GMCSFRA(r1=None) + GMCSF(s1=None, s2=None) | GMCSFRA(r1=1)%GMCSF(s1=1, s2=None), ka_GMCSFRA_GMCSF, kd_GMCSFRA_GMCSF)
#Rule('RAGMCSF_binds_BetaCdimer', Beta_c(r1=None, r2=1)%Beta_c(r1=None,r2=1) + GMCSFRA(r1=2)%GMCSF(s1=2, s2=None) | GMCSFRA(r1=2)%GMCSF(s1=2, s2=3)%Beta_c(r1=3, r2=1)%Beta_c(r1=None,r2=1), ka_RAGMCSF_BetaCdimer, kd_RAGMCSF_BetaCdimer)
#Rule('RAGMCSF_hexamerisation', GMCSFRA(r1=2)%GMCSF(s1=2, s2=3,s3=None)%Beta_c(r1=3, r2=1)%Beta_c(r1=None,r2=1) + GMCSFRA(r1=2)%GMCSF(s1=2, s2=None,s3=None) | GMCSFRA(r1=1)%GMCSF(s1=1,s2=2,s3=None)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3)%GMCSFRA(r1=5)%GMCSF(s1=5,s2=4,s3=None), ka_RAGMCSF_hexamerisation, kd_RAGMCSF_hexamerisation)
#Rule('GMCSF_dodecamerisation', GMCSFRA(r1=1)%GMCSF(s1=1,s2=2,s3=None)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3)%GMCSFRA(r1=5)%GMCSF(s1=5,s2=4,s3=None) + GMCSFRA(r1=1)%GMCSF(s1=1,s2=2,s3=None)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3)%GMCSFRA(r1=5)%GMCSF(s1=5,s2=4,s3=None)| GMCSFRA(r1=1)%GMCSF(s1=1,s2=2,s3=None)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3)%GMCSFRA(r1=5)%GMCSF(s1=5,s2=4,s3=12)%GMCSFRA(r1=6)%GMCSF(s1=6,s2=7,s3=12)%Beta_c(r1=7,r2=8)%Beta_c(r1=9,r2=8)%GMCSFRA(r1=10)%GMCSF(s1=10,s2=9,s3=None), ka_GMCSF_dodecamerisation, kd_GMCSF_dodecamerisation)
#Rule('GMCSF_receptor_state_change', GMCSFRA(r1=1)%GMCSF(s1=1,s2=2)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3,r3=21)%Jak2(s1=21, status='inactive')%GMCSFRA(r1=5)%GMCSF(s1=5,s2=4,s3=12)%GMCSFRA(r1=6)%GMCSF(s1=6,s2=7,s3=12)%Beta_c(r1=7,r2=8,r3=20)%Jak2(s1=20, status='inactive')%Beta_c(r1=9,r2=8)%GMCSFRA(r1=10)%GMCSF(s1=10,s2=9) | GMCSFRA(r1=1)%GMCSF(s1=1,s2=2)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3,r3=21)%Jak2(s1=21, status='active')%GMCSFRA(r1=5)%GMCSF(s1=5,s2=4,s3=12)%GMCSFRA(r1=6)%GMCSF(s1=6,s2=7,s3=12)%Beta_c(r1=7,r2=8,r3=20)%Jak2(s1=20, status='active')%Beta_c(r1=9,r2=8)%GMCSFRA(r1=10)%GMCSF(s1=10,s2=9), GMCSF_activation, GMCSF_deactivation)
# IL3RA + IL3 <-> 4x(IL3RA.IL3) + 2x(Jak2.Beta_c.Beta_c.Jak2) <-> Active receptor + STAT5 # same idea as GM-CSF
#Rule('IL3RA_binds_IL3', IL3RA(r1=None) + IL3(s1=None, s2=None) | IL3RA(r1=1)%IL3(s1=1, s2=None), ka_IL3RA_IL3, kd_IL3RA_IL3)
#Rule('RAIL3_binds_BetaCdimer', Beta_c(r1=None, r2=1)%Beta_c(r1=None,r2=1) + IL3RA(r1=2)%IL3(s1=2, s2=None) | IL3RA(r1=2)%IL3(s1=2, s2=3)%Beta_c(r1=3, r2=1)%Beta_c(r1=None,r2=1), ka_RAIL3_BetaCdimer, kd_RAIL3_BetaCdimer)
#Rule('RAIL3_hexamerisation', IL3RA(r1=2)%IL3(s1=2, s2=3,s3=None)%Beta_c(r1=3, r2=1)%Beta_c(r1=None,r2=1) + IL3RA(r1=2)%IL3(s1=2, s2=None,s3=None) | IL3RA(r1=1)%IL3(s1=1,s2=2,s3=None)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3)%IL3RA(r1=5)%IL3(s1=5,s2=4,s3=None), ka_RAIL3_hexamerisation, kd_RAIL3_hexamerisation)
#Rule('IL3_dodecamerisation', IL3RA(r1=1)%IL3(s1=1,s2=2,s3=None)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3)%IL3RA(r1=5)%IL3(s1=5,s2=4,s3=None) + IL3RA(r1=1)%IL3(s1=1,s2=2,s3=None)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3)%IL3RA(r1=5)%IL3(s1=5,s2=4,s3=None)| IL3RA(r1=1)%IL3(s1=1,s2=2,s3=None)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3)%IL3RA(r1=5)%IL3(s1=5,s2=4,s3=12)%IL3RA(r1=6)%IL3(s1=6,s2=7,s3=12)%Beta_c(r1=7,r2=8)%Beta_c(r1=9,r2=8)%IL3RA(r1=10)%IL3(s1=10,s2=9,s3=None), ka_IL3_dodecamerisation, kd_IL3_dodecamerisation)
#Rule('IL3_receptor_state_change', IL3RA(r1=1)%IL3(s1=1,s2=2)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3,r3=21)%Jak2(s1=21, status='inactive')%IL3RA(r1=5)%IL3(s1=5,s2=4,s3=12)%IL3RA(r1=6)%IL3(s1=6,s2=7,s3=12)%Beta_c(r1=7,r2=8,r3=20)%Jak2(s1=20, status='inactive')%Beta_c(r1=9,r2=8)%IL3RA(r1=10)%IL3(s1=10,s2=9) | IL3RA(r1=1)%IL3(s1=1,s2=2)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3,r3=21)%Jak2(s1=21, status='active')%IL3RA(r1=5)%IL3(s1=5,s2=4,s3=12)%IL3RA(r1=6)%IL3(s1=6,s2=7,s3=12)%Beta_c(r1=7,r2=8,r3=20)%Jak2(s1=20, status='active')%Beta_c(r1=9,r2=8)%IL3RA(r1=10)%IL3(s1=10,s2=9), IL3_activation, IL3_deactivation)

# IL5RA + IL5 <-> 4x(IL5RA.IL5) + 2x(Jak2.Beta_c.Beta_c.Jak2) <-> Active receptor + STAT5 # same idea as GM-CSF
#Rule('IL5RA_binds_IL5', IL5RA(r1=None) + IL5(s1=None, s2=None) | IL5RA(r1=1)%IL5(s1=1, s2=None), ka_IL5RA_IL5, kd_IL5RA_IL5)
#Rule('RAIL5_binds_BetaCdimer', Beta_c(r1=None, r2=1)%Beta_c(r1=None,r2=1) + IL5RA(r1=2)%IL5(s1=2, s2=None) | IL5RA(r1=2)%IL5(s1=2, s2=3)%Beta_c(r1=3, r2=1)%Beta_c(r1=None,r2=1), ka_RAIL5_BetaCdimer, kd_RAIL5_BetaCdimer)
#Rule('RAIL5_hexamerisation', IL5RA(r1=2)%IL5(s1=2, s2=3,s3=None)%Beta_c(r1=3, r2=1)%Beta_c(r1=None,r2=1) + IL5RA(r1=2)%IL5(s1=2, s2=None,s3=None) | IL5RA(r1=1)%IL5(s1=1,s2=2,s3=None)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3)%IL5RA(r1=5)%IL5(s1=5,s2=4,s3=None), ka_RAIL5_hexamerisation, kd_RAIL5_hexamerisation)
#Rule('IL5_dodecamerisation', IL5RA(r1=1)%IL5(s1=1,s2=2,s3=None)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3)%IL5RA(r1=5)%IL5(s1=5,s2=4,s3=None) + IL5RA(r1=1)%IL5(s1=1,s2=2,s3=None)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3)%IL5RA(r1=5)%IL5(s1=5,s2=4,s3=None)| IL5RA(r1=1)%IL5(s1=1,s2=2,s3=None)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3)%IL5RA(r1=5)%IL5(s1=5,s2=4,s3=12)%IL5RA(r1=6)%IL5(s1=6,s2=7,s3=12)%Beta_c(r1=7,r2=8)%Beta_c(r1=9,r2=8)%IL5RA(r1=10)%IL5(s1=10,s2=9,s3=None), ka_IL5_dodecamerisation, kd_IL5_dodecamerisation)
#Rule('IL5_receptor_state_change', IL5RA(r1=1)%IL5(s1=1,s2=2)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3,r3=21)%Jak2(s1=21, status='inactive')%IL5RA(r1=5)%IL5(s1=5,s2=4,s3=12)%IL5RA(r1=6)%IL5(s1=6,s2=7,s3=12)%Beta_c(r1=7,r2=8,r3=20)%Jak2(s1=20, status='inactive')%Beta_c(r1=9,r2=8)%IL5RA(r1=10)%IL5(s1=10,s2=9) | IL5RA(r1=1)%IL5(s1=1,s2=2)%Beta_c(r1=2,r2=3)%Beta_c(r1=4,r2=3,r3=21)%Jak2(s1=21, status='active')%IL5RA(r1=5)%IL5(s1=5,s2=4,s3=12)%IL5RA(r1=6)%IL5(s1=6,s2=7,s3=12)%Beta_c(r1=7,r2=8,r3=20)%Jak2(s1=20, status='active')%Beta_c(r1=9,r2=8)%IL5RA(r1=10)%IL5(s1=10,s2=9), IL5_activation, IL5_deactivation)

# Gamma_chain family
#Rule('Gamma_binds_Jak3', Gamma_c(r3=None) + Jak3(s1=None, status='inactive') | Gamma_c(r3=1)%Jak3(s1=1, status='inactive'), ka_Gamma_Jak3, kd_Gamma_Jak3)

# IL2 has many receptor forms, not well identified experimentally
# IL2RA + IL2 <-> IL2RA.IL2 + IL2RB <-> IL2RA.IL2.IL2RB + Gamma_chain <-> IL2RA.IL2.IL2RB.Gamma_chain + Jak1 + Jak3 -> pSTAT5
#Rule('IL2RB_binds_Jak1', IL2RB(r3=None) + Jak1(s1=None, status='inactive') | IL2RB(r3=1)%Jak1(s1=1, status='inactive'), ka_IL2RB_Jak1, kd_IL2RB_Jak1)
#Rule('IL2RB_binds_Gamma', IL2RB(r4=None) + Gamma_c(r1=None) | IL2RB(r4=1)%Gamma_c(r1=1), ka_IL2RB_Gamma, kd_IL2RB_Gamma)
#Rule('IL2RBGamma_binds_IL2RA', IL2RB(r2=None,r4=1)%Gamma_c(r1=1) + IL2RA(r2=None) | IL2RA(r2=2)%IL2RB(r2=2,r4=1)%Gamma_c(r1=1), ka_IL2RBGamma_IL2RA, kd_IL2RBGamma_IL2RA)
    # High affinity receptor
#Rule('IL2RABGamma_binds_IL2', IL2RA(r1=None,r2=2)%IL2RB(r1=None,r2=2,r4=1)%Gamma_c(r1=1) + IL2(s1=None, s2=None) | IL2RA(r1=4,r2=2)%IL2(s1=4,s2=3)%IL2RB(r1=3,r2=2,r4=1)%Gamma_c(r1=1), ka_IL2RABGamma_IL2, kd_IL2RABGamma_IL2)
    # Medium affinity receptor
#Rule('IL2RBGamma_binds_IL2', IL2RB(r1=None,r4=1)%Gamma_c(r1=1) + IL2(s1=None, s2=None) | IL2RB(r1=3,r4=1)%Gamma_c(r1=1)%IL2(s1=None, s2=3), ka_IL2RBGamma_IL2, kd_IL2RBGamma_IL2)
    # Low affinity receptor
#Rule('IL2RA_binds_IL2', IL2RA(r1=None) + IL2(s1=None, s2=None) | IL2RA(r1=3)%IL2(s1=3, s2=None), ka_IL2RA_IL2, kd_IL2RA_IL2)
    # Receptor can activate whether or not IL2RA is bound
#Rule('IL2_receptor_state_change', Jak1(s1=5,status='inactive')%IL2RB(r1=2,r3=5,r4=1)%Gamma_c(r1=1,r3=6)%IL2(s2=2)%Jak3(s1=6,status='inactive') | Jak1(s1=5,status='active')%IL2RB(r1=2,r3=5,r4=1)%Gamma_c(r1=1,r3=6)%IL2(s2=2)%Jak3(s1=6,status='active'), IL2_activation, IL2_deactivation)

# IL4RA + IL4 + Gamma_chain <-> Jak1 + IL4RA.IL4.Gamma_chain + Jak3 -> STAT6 # Jak1 binds to IL4RA and Jak3 to Gamma_c
#Rule('IL4RA_binds_Jak1', IL4RA(r3=None) + Jak1(s1=None, status='inactive') | IL4RA(r3=1)%Jak1(s1=1, status='inactive'), ka_IL4RA_Jak1, kd_IL4RA_Jak1)
#Rule('IL4_binds_IL4RA', IL4RA(r1=None) + IL4(s1=None) | IL4RA(r1=1)%IL4(s1=1), ka_IL4RA_IL4, kd_IL4RA_IL4)
#Rule('RIL4_binds_Gamma', IL4RA(r1=1)%IL4(s1=1, s2=None) + Gamma_c(r1=None) | IL4RA(r1=1)%IL4(s1=1, s2=2)%Gamma_c(r1=2), ka_RIL4_Gamma, kd_RIL4_Gamma)
#Rule('IL4_receptor_state_change', Jak1(s1=3,status='inactive')%IL4RA(r1=1, r3=3)%IL4(s1=1, s2=2)%Gamma_c(r1=2, r3=4)%Jak3(s1=4,status='inactive') | Jak1(s1=3,status='active')%IL4RA(r1=1, r3=3)%IL4(s1=1, s2=2)%Gamma_c(r1=2, r3=4)%Jak3(s1=4,status='active'), IL4_activation, IL4_deactivation)

# IL4RA + IL4 + IL13RA1 <-> Jak1 + IL4RA.IL4.IL13RA1 + Jak3 -> STAT6
#Rule('IL13RA1_binds_Jak3', IL13RA1(r3=None) + Jak3(s1=None, status='inactive') | IL13RA1(r3=1)%Jak3(s1=1, status='inactive'), ka_IL13RA1_Jak3, kd_IL13RA1_Jak3)
#Rule('RIL4_binds_IL13RA1', IL4RA(r1=1)%IL4(s1=1, s2=None) + IL13RA1(r1=None) | IL4RA(r1=1)%IL4(s1=1, s2=2)%IL13RA1(r1=2), ka_RIL4_IL13RA1, kd_RIL4_IL13RA1)
#Rule('IL413RA1_receptor_state_change', Jak1(s1=3,status='inactive')%IL4RA(r1=1, r3=3)%IL4(s1=1, s2=2)%IL13RA1(r1=2, r3=4)%Jak3(s1=4,status='inactive') | Jak1(s1=3,status='active')%IL4RA(r1=1, r3=3)%IL4(s1=1, s2=2)%IL13RA1(r1=2, r3=4)%Jak3(s1=4,status='active'), IL413RA1_activation, IL413RA1_deactivation)

# IL13RA1 + IL13 <-> Jak1 + IL13RA1.IL13 + IL4RA <-> IL13RA1.IL13.IL4RA + Jak1 + Tyk2/Jak2 -> STAT6
#Rule('IL13RA1_binds_Jak2', IL13RA1(r3=None) + Jak2(s1=None,status='inactive') | IL13RA1(r3=1)%Jak2(s1=1,status='inactive'), ka_IL13RA1_Jak2, ka_IL13RA1_Jak2)
#Rule('IL13RA1_binds_Tyk2', IL13RA1(r3=None) + Tyk2(s1=None,status='inactive') | IL13RA1(r3=1)%Tyk2(s1=1,status='inactive'), ka_IL13RA1_Tyk2, ka_IL13RA1_Tyk2)
#Rule('IL13RA1_binds_IL13', IL13RA1(r1=None) + IL13(s1=None) | IL13RA1(r1=1)%IL13(s1=1), ka_IL13RA1_IL13, kd_IL13RA1_IL13)
#Rule('RA1IL13_binds_IL4RA', IL13RA1(r1=1)%IL13(s1=1,s2=None) + IL4RA(r1=None) | IL13RA1(r1=1)%IL13(s1=1,s2=2)%IL4RA(r1=2), ka_RA1IL13_IL4RA, kd_RA1IL13_IL4RA)
#Rule('IL13_Jak2_receptor_state_change', Jak2(s1=4,status='inactive')%IL13RA1(r1=1,r3=4)%IL13(s1=1,s2=2)%IL4RA(r1=2,r3=3)%Jak1(s1=3,status='inactive') | Jak2(s1=4,status='active')%IL13RA1(r1=1,r3=4)%IL13(s1=1,s2=2)%IL4RA(r1=2,r3=3)%Jak1(s1=3,status='active'), IL13_Jak2_receptor_activation, IL13_Jak2_receptor_deactivation)
#Rule('IL13_Tyk2_receptor_state_change', Tyk2(s1=4,status='inactive')%IL13RA1(r1=1,r3=4)%IL13(s1=1,s2=2)%IL4RA(r1=2,r3=3)%Jak1(s1=3,status='inactive') | Tyk2(s1=4,status='active')%IL13RA1(r1=1,r3=4)%IL13(s1=1,s2=2)%IL4RA(r1=2,r3=3)%Jak1(s1=3,status='active'), IL13_Tyk2_receptor_activation, IL13_Tyk2_receptor_deactivation)

# IL13RA1 + IL13 <-> IL13RA1.IL13 + IL13RA2 <-> IL13RA1.IL13.IL13RA2 # Does not activate any STATs
#Rule('RA1IL13_binds_IL13RA2', IL13RA1(r1=1)%IL13(s1=1,s2=None) + IL13RA2(r1=None) | IL13RA1(r1=1)%IL13(s1=1,s2=2)%IL13RA2(r1=2), ka_RA1IL13_IL13RA2, kd_RA1IL13_IL13RA2)

# IL7RA + IL7 + Gamma_chain <-> Jak1 + IL7RA.IL7.Gamma_chain + Jak3 -> STAT5 # Jak1 binds to IL7RA and Jak3 to Gamma_c
#Rule('IL7RA_binds_Jak1', IL7RA(r3=None) + Jak1(s1=None, status='inactive') | IL7RA(r3=1)%Jak1(s1=1, status='inactive'), ka_IL7RA_Jak1, kd_IL7RA_Jak1)
#Rule('IL7_binds_IL7RA', IL7RA(r1=None) + IL7(s1=None) | IL7RA(r1=1)%IL7(s1=1), ka_IL7RA_IL7, kd_IL7RA_IL7)
#Rule('RIL7_binds_Gamma', IL7RA(r1=1)%IL7(s1=1, s2=None) + Gamma_c(r1=None) | IL7RA(r1=1)%IL7(s1=1, s2=2)%Gamma_c(r1=2), ka_RIL7_Gamma, kd_RIL7_Gamma)
#Rule('IL7_receptor_state_change', Jak1(s1=3,status='inactive')%IL7RA(r1=1, r3=3)%IL7(s1=1, s2=2)%Gamma_c(r1=2, r3=4)%Jak3(s1=4,status='inactive') | Jak1(s1=3,status='active')%IL7RA(r1=1, r3=3)%IL7(s1=1, s2=2)%Gamma_c(r1=2, r3=4)%Jak3(s1=4,status='active'), IL7_activation, IL7_deactivation)

# IL9RA + IL9 + Gamma_chain <-> Jak1 + IL9RA.IL9.Gamma_chain + Jak3 -> STAT5 # Jak1 binds to IL9RA and Jak3 to Gamma_c
#Rule('IL9RA_binds_Jak1', IL9RA(r3=None) + Jak1(s1=None, status='inactive') | IL9RA(r3=1)%Jak1(s1=1, status='inactive'), ka_IL9RA_Jak1, kd_IL9RA_Jak1)
#Rule('IL9_binds_IL9RA', IL9RA(r1=None) + IL9(s1=None) | IL9RA(r1=1)%IL9(s1=1), ka_IL9RA_IL9, kd_IL9RA_IL9)
#Rule('RIL9_binds_Gamma', IL9RA(r1=1)%IL9(s1=1, s2=None) + Gamma_c(r1=None) | IL9RA(r1=1)%IL9(s1=1, s2=2)%Gamma_c(r1=2), ka_RIL9_Gamma, kd_RIL9_Gamma)
#Rule('IL9_receptor_state_change', Jak1(s1=3,status='inactive')%IL9RA(r1=1, r3=3)%IL9(s1=1, s2=2)%Gamma_c(r1=2, r3=4)%Jak3(s1=4,status='inactive') | Jak1(s1=3,status='active')%IL9RA(r1=1, r3=3)%IL9(s1=1, s2=2)%Gamma_c(r1=2, r3=4)%Jak3(s1=4,status='active'), IL9_activation, IL9_deactivation)

# IL15RA + IL15 + Gamma_chain <-> Jak1 + IL15RA.IL15.Gamma_chain + Jak3 -> STAT5 # Jak1 binds to IL15RA and Jak3 to Gamma_c
#Rule('IL15RA_binds_Jak1', IL15RA(r3=None) + Jak1(s1=None, status='inactive') | IL15RA(r3=1)%Jak1(s1=1, status='inactive'), ka_IL15RA_Jak1, kd_IL15RA_Jak1)
#Rule('IL15_binds_IL15RA', IL15RA(r1=None) + IL15(s1=None) | IL15RA(r1=1)%IL15(s1=1), ka_IL15RA_IL15, kd_IL15RA_IL15)
#Rule('RIL15_binds_Gamma', IL15RA(r1=1)%IL15(s1=1, s2=None) + Gamma_c(r1=None) | IL15RA(r1=1)%IL15(s1=1, s2=2)%Gamma_c(r1=2), ka_RIL15_Gamma, kd_RIL15_Gamma)
#Rule('IL15_receptor_state_change', Jak1(s1=3,status='inactive')%IL15RA(r1=1, r3=3)%IL15(s1=1, s2=2)%Gamma_c(r1=2, r3=4)%Jak3(s1=4,status='inactive') | Jak1(s1=3,status='active')%IL15RA(r1=1, r3=3)%IL15(s1=1, s2=2)%Gamma_c(r1=2, r3=4)%Jak3(s1=4,status='active'), IL15_activation, IL15_deactivation)
# IL21R + IL21 + Gamma_chain <-> Jak1 + IL21R.IL21.Gamma_chain + Jak3 -> STAT3 (STAT1, STAT5) # Jak1 binds to IL21R and Jak3 to Gamma_c
#Rule('IL21R_binds_Jak1', IL21R(r3=None) + Jak1(s1=None, status='inactive') | IL21R(r3=1)%Jak1(s1=1, status='inactive'), ka_IL21R_Jak1, kd_IL21R_Jak1)
#Rule('IL21_binds_IL21R', IL21R(r1=None) + IL21(s1=None) | IL21R(r1=1)%IL21(s1=1), ka_IL12R_IL21, kd_IL12R_IL21)
#Rule('RIL21_binds_Gamma', IL21R(r1=1)%IL21(s1=1, s2=None) + Gamma_c(r1=None) | IL21R(r1=1)%IL21(s1=1, s2=2)%Gamma_c(r1=2), ka_RIL21_Gamma, kd_RIL21_Gamma)
#Rule('IL21_receptor_state_change', Jak1(s1=3,status='inactive')%IL21R(r1=1, r3=3)%IL21(s1=1, s2=2)%Gamma_c(r1=2, r3=4)%Jak3(s1=4,status='inactive') |\
#     Jak1(s1=3,status='active')%IL21R(r1=1, r3=3)%IL21(s1=1, s2=2)%Gamma_c(r1=2, r3=4)%Jak3(s1=4,status='active'), IL21_activation, IL21_deactivation)

# Type 1 IFN
# IFNAR2 + IFNa + IFNAR1 <-> Jak1.IFNAR2.IFNa.IFNAR1.Tyk2 -> STAT1  # Same activation for IFNb
#Rule('IFNAR1_binds_Tyk2', IFNAR1(r3=None) + Tyk2(s1=None, status='inactive') | IFNAR1(r3=1)%Tyk2(s1=1, status='inactive'), ka_IFNAR1_Tyk2, kd_IFNAR1_Tyk2)
#Rule('IFNAR2_binds_Jak1', IFNAR2(r3=None) + Jak1(s1=None, status='inactive') | IFNAR2(r3=1)%Jak1(s1=1, status='inactive'), ka_IFNAR2_Jak1, kd_IFNAR2_Jak1)

#Rule('IFNa_bind_R1', IFNAR1(r1=None) + IFN_alpha2(s1=None,s2=None) | IFNAR1(r1=1)%IFN_alpha2(s1=1,s2=None), ka_IFNa_R1, kd_IFNa_R1)
#Rule('IFNa_bind_R2', IFNAR2(r1=None) + IFN_alpha2(s1=None,s2=None) | IFNAR2(r1=1)%IFN_alpha2(s1=1,s2=None), ka_IFNa_R2, kd_IFNa_R2)
#Rule('IaR1_bind_R2', IFNAR1(r1=1)%IFN_alpha2(s1=1,s2=None) + IFNAR2(r1=None) | IFNAR1(r1=1)%IFN_alpha2(s1=1,s2=2)%IFNAR2(r1=2), ka_IFNaR1_R2, kd_IFNaR1_R2)
#Rule('IaR2_bind_R1', IFNAR2(r1=1)%IFN_alpha2(s1=1,s2=None) + IFNAR1(r1=None) | IFNAR1(r1=1)%IFN_alpha2(s1=1,s2=2)%IFNAR2(r1=2), ka_IFNaR2_R1, kd_IFNaR2_R1)

#Rule('IFNb_bind_R1', IFNAR1(r1=None) + IFN_beta(s1=None,s2=None) | IFNAR1(r1=1)%IFN_beta(s1=1,s2=None), ka_IFNb_R1, kd_IFNb_R1)
#Rule('IFNb_bind_R2', IFNAR2(r1=None) + IFN_beta(s1=None,s2=None) | IFNAR2(r1=1)%IFN_beta(s1=1,s2=None), ka_IFNb_R2, kd_IFNb_R2)
#Rule('IbR1_bind_R2', IFNAR1(r1=1)%IFN_beta(s1=1,s2=None) + IFNAR2(r1=None) | IFNAR1(r1=1)%IFN_beta(s1=1,s2=2)%IFNAR2(r1=2), ka_IFNbR1_R2, kd_IFNbR1_R2)
#Rule('IbR2_bind_R1', IFNAR2(r1=1)%IFN_beta(s1=1,s2=None) + IFNAR1(r1=None) | IFNAR1(r1=1)%IFN_beta(s1=1,s2=2)%IFNAR2(r1=2), ka_IFNbR2_R1, kd_IFNbR2_R1)

#Rule('IFNa_receptor_state_change', Tyk2(s1=3, status='inactive')%IFNAR1(r1=1, r3=3)%IFN_alpha2(s1=1,s2=2)%IFNAR2(r1=2, r3=4)%Jak1(s1=4, status='inactive') |\
#     Tyk2(s1=3, status='active')%IFNAR1(r1=1, r3=3)%IFN_alpha2(s1=1,s2=2)%IFNAR2(r1=2, r3=4)%Jak1(s1=4, status='active'), IFNa_activation, IFNa_deactivation)
#Rule('IFNb_receptor_state_change', Tyk2(s1=3, status='inactive')%IFNAR1(r1=1, r3=3)%IFN_beta(s1=1,s2=2)%IFNAR2(r1=2, r3=4)%Jak1(s1=4, status='inactive') |\
#     Tyk2(s1=3, status='active')%IFNAR1(r1=1, r3=3)%IFN_beta(s1=1,s2=2)%IFNAR2(r1=2, r3=4)%Jak1(s1=4, status='active'), IFNb_activation, IFNb_deactivation)

# Type 2 IFN
# IFNg + 2x(IFNGR1.Jak1) <-> Jak1.IFNGR1.IFNg.IFNGR1.Jak1 + 2x(Jak2.IFNGR2) -> STAT1
Rule('IFNGR1_binds_Jak1', IFNGR1(r3=None) + Jak1(s1=None, status='inactive') | IFNGR1(r3=1)%Jak1(s1=1, status='inactive'), ka_IFNGR1_Jak1, kd_IFNGR1_Jak1)
Rule('IFNGR2_binds_Tyk2', IFNGR2(r3=None) + Tyk2(s1=None, status='inactive') | IFNGR2(r3=1)%Tyk2(s1=1, status='inactive'), ka_IFNGR2_Tyk2, kd_IFNGR2_Tyk2)

Rule('R1_bind_R1', IFNGR1(r1=None, r2=None, r4=None) + IFNGR1(r1=None, r2=None, r4=None) | IFNGR1(r1=None, r2=None, r4=1)%IFNGR1(r1=None, r2=None, r4=1), ka_R1_bind_R1, kd_R1_bind_R1)
Rule('IFNg_bind_R1', IFNGR1(r1=None, r2=None, r4=20)%IFNGR1(r1=None, r2=None, r4=20) + IFN_gamma(s1=None,s2=None) | IFNGR1(r1=1, r2=None, r4=20)%IFN_gamma(s1=1,s2=2)%IFNGR1(r1=2, r2=None, r4=20), ka_IFNg_R1, kd_IFNg_R1)
Rule('IFNgR1_bind_R2', IFNGR1(r1=ANY, r2=None, r4=ANY) + IFNGR2(r1=None) | IFNGR1(r1=ANY, r2=2, r4=ANY)%IFNGR2(r1=2), ka_IFNgR1_R2, kd_IFNgR1_R2)
Rule('IFNg_receptor_state_change', Tyk2(s1=8,status='inactive')%Tyk2(s1=7,status='inactive')%Jak1(s1=6,status='inactive')%Jak1(s1=5,status='inactive')%IFNGR2(r1=3, r3=7)%IFNGR1(r1=1, r2=3, r3=5, r4=20)%IFN_gamma(s1=1,s2=2)%IFNGR1(r1=2, r2=4, r3=6, r4=20)%IFNGR2(r1=4, r3=8) | Tyk2(s1=8,status='active') % Tyk2(s1=7,status='active') % Jak1(s1=6,status='active') % Jak1(s1=5,status='active') % IFNGR2(r1=3, r3=7) % IFNGR1(r1=1, r2=3, r3=5, r4=20) % IFN_gamma(s1=1, s2=2) % IFNGR1(r1=2, r2=4, r3=6, r4=20) % IFNGR2(r1=4, r3=8), IFNg_activation, IFNg_deactivation)
# -----------
# IL12 family
# -----------
# IL12RB1 + IL12 + IL12RB2 <-> Tyk2.IL12RB1.IL12.IL12RB2.Jak2 -> STAT3
#Rule('IL12RB1_binds_Tyk2', IL12RB1(r3=None) + Tyk2(s1=None, status='inactive') | IL12RB1(r3=1)%Tyk2(s1=1, status='inactive'), ka_IL12RB1_Tyk2, kd_IL12RB1_Tyk2)
#Rule('IL12RB2_binds_Jak2', IL12RB2(r3=None) + Jak2(s1=None, status='inactive') | IL12RB2(r3=1)%Jak2(s1=1, status='inactive'), ka_IL12RB2_Jak2, kd_IL12RB2_Jak2)

#Rule('RB1_IL12', IL12RB1(r1=None) + IL12(s1=None) | IL12RB1(r1=1)%IL12(s1=1), ka_IL12_IL12RB1, kd_IL12_IL12RB1)
#Rule('RB2_IL12', IL12RB2(r1=None) + IL12(s2=None) | IL12RB2(r1=1)%IL12(s2=1), ka_IL12_IL12RB2, kd_IL12_IL12RB2)
#Rule('RB1IL12_RB2', IL12RB1(r1=1)%IL12(s1=1,s2=None) + IL12RB2(r1=None) | IL12RB2(r1=1)%IL12(s1=1,s2=2)%IL12RB1(r1=2), ka_RB1IL12_RB2, kd_RB1IL12_RB2)
#Rule('RB2IL12_RB1', IL12RB2(r1=2)%IL12(s1=None,s2=2) + IL12RB1(r1=None) | IL12RB2(r1=2)%IL12(s1=1,s2=2)%IL12RB1(r1=1), ka_RB2IL12_RB1, kd_RB2IL12_RB1)

#Rule('IL12_receptor_activation', Tyk2(s1=3, status='inactive')%IL12RB1(r1=1,r3=3)%IL12(s1=1,s2=2)%IL12RB2(r1=2,r3=4)%Jak2(s1=4, status='inactive') |\
#     Tyk2(s1=3, status='active')%IL12RB1(r1=1,r3=3)%IL12(s1=1,s2=2)%IL12RB2(r1=2,r3=4)%Jak2(s1=4, status='active'), IL12_activation, IL12_deactivation)

# IL23RA + IL23 + IL12RB1 <-> Jak2.IL23RA.IL23.IL12RB1.Tyk2 -> STAT3
#Rule('IL23RA_binds_Jak2', IL23RA(r3=None) + Jak2(s1=None, status='inactive') | IL23RA(r3=1)%Jak2(s1=1, status='inactive'), ka_IL23RA_Jak2, kd_IL23RA_Jak2)
#Rule('IL23RA_IL23', IL23RA(r1=None) + IL23(s1=None,s2=None) | IL23RA(r1=1)%IL23(s1=None,s2=1), ka_IL23_IL23RA, kd_IL23_IL23RA)
#Rule('IL12RB1_IL23', IL12RB1(r1=None) + IL23(s1=None,s2=None) | IL12RB1(r1=1)%IL23(s1=None,s2=1), ka_IL23_IL12RB1, kd_IL23_IL12RB1)

#Rule('RAIL23_IL12RB1', IL23RA(r1=2)%IL23(s1=None,s2=2) + IL12RB1(r1=None) | IL23RA(r1=2)%IL23(s1=1,s2=2)%IL12RB1(r1=1), ka_RAIL23_IL12RB1, kd_RAIL23_IL12RB1)
#Rule('RB1IL23_IL23RA', IL12RB1(r1=1)%IL23(s1=None,s2=1) + IL23RA(r1=None) | IL23RA(r1=2)%IL23(s1=2,s2=1)%IL12RB1(r1=1), ka_RB1IL23_IL23RA, kd_RB1IL23_IL23RA)

#Rule('IL23_receptor_activation', Tyk2(s1=3, status='inactive')%IL12RB1(r1=1,r3=3)%IL23(s1=1,s2=2)%IL23RA(r1=2,r3=4)%Jak2(s1=4, status='inactive') |\
#     Tyk2(s1=3, status='active')%IL12RB1(r1=1,r3=3)%IL23(s1=1,s2=2)%IL23RA(r1=2,r3=4)%Jak2(s1=4, status='active'), IL23_activation, IL23_deactivation)

# -----------
# IL20 family
# -----------
# IL20 + IL20RA + IL20RB <-> IL20RA.IL20.IL20RB -> STAT3
#Rule('IL20RA_binds_Jak2', IL20RA(r3=None) + Jak2(s1=None, status='inactive') | IL20RA(r3=1)%Jak2(s1=1, status='inactive'), ka_IL20RA_Jak2, kd_IL20RA_Jak2)
#Rule('IL20_IL20RA', IL20(s1=None) + IL20RA(r1=None) | IL20(s1=1)%IL20RA(r1=1), ka_IL20_IL20RA, kd_IL20_IL20RA)
#Rule('ANYIL20_IL20RA', IL20(s1=None,s2=ANY) + IL20RA(r1=None) | IL20(s1=1,s2=ANY)%IL20RA(r1=1), ka_ANYIL20_IL20RA, kd_ANYIL20_IL20RA)

# IL20 family Type 1 receptor (IL20RA + IL20RB) senses IL19, 20, and 24
#Rule('IL20_Type1', IL20(s1=None,s2=None) + IL20RB(r1=None) | IL20(s1=None,s2=1)%IL20RB(r1=1), ka_IL20_IL20RB, kd_IL20_IL20RB)
#Rule('RAIL20_Type1', IL20(s1=1,s2=None)%IL20RA(r1=1) + IL20RB(r1=None) | IL20(s1=1, s2=2)%IL20RA(r1=1)%IL20RB(r1=2), ka_RAIL20_IL20RB, kd_RAIL20_IL20RB)

# IL20 family Type 2 receptor (IL20RA + IL22RA1) senses IL20 and IL24
# IL20 + IL20RA + IL22RA1 <-> IL20RA.IL20.IL22RA1 -> STAT3
#Rule('IL20_Type2', IL20(s1=None,s2=None) + IL22RA1(r1=None) | IL20(s1=None,s2=1)%IL22RA1(r1=1), ka_IL20_IL22RA1, kd_IL20_IL22RA1)
#Rule('IL22RA1IL20_Type2', IL20(s1=1,s2=None)%IL20RA(r1=1) + IL22RA1(r1=None) | IL20RA(r1=1)%IL20(s1=1,s2=2)%IL22RA1(r1=2), ka_22RA1IL20, kd_22RA1IL20)

#Rule('IL20_Type1_activation', Jak2(s1=3, status='inactive')%IL20RA(r1=1, r3=3)%IL20(s1=1, s2=2)%IL20RB(r1=2) | Jak2(s1=3, status='active')%IL20RA(r1=1, r3=3)%IL20(s1=1, s2=2)%IL20RB(r1=2), IL20_activation, IL20_deactivation)
#Rule('IL20_Type2_activation', Jak2(s1=3, status='inactive')%IL20RA(r1=1, r3=3)%IL20(s1=1, s2=2)%IL22RA1(r1=2) | Jak2(s1=3, status='active')%IL20RA(r1=1, r3=3)%IL20(s1=1, s2=2)%IL22RA1(r1=2), IL20_activation, IL20_deactivation)

# -----------
# IL10 family
# -----------
# IL10 + 2x(IL10RA) <-> IL10RA.IL10.IL10RA + 2x(IL10RB) <-> IL10RB.IL10RA.IL10.IL10RA.IL10RB + Jak1 + Tyk2 -> STAT3 # Jak1 is actually bound to IL10RA and Tyk2 to IL10RB
#Rule('IL10RA_binds_Jak1', IL10RA(r3=None) + Jak1(s1=None,status='inactive') | IL10RA(r3=1)%Jak1(s1=1,status='inactive'), ka_Jak1_IL10RA, kd_Jak1_IL10RA)
#Rule('IL10RB_binds_Tyk2', IL10RB(r3=None) + Tyk2(s1=None,status='inactive') | IL10RB(r3=1)%Tyk2(s1=1,status='inactive'), ka_Tyk2_IL10RB, kd_Tyk2_IL10RB)
#Rule('IL10_binds_RA', IL10(s1=None, s2=None) + IL10RA(r1=None, r2=None) | IL10(s1=1, s2=None)%IL10RA(r1=1, r2=None), ka_IL10_IL10RA, kd_IL10_IL10RA)
#Rule('IL10RA_IL10_binds_RA', IL10(s1=1, s2=None)%IL10RA(r1=1, r2=None) + IL10RA(r1=None, r2=None) | IL10RA(r1=1, r2=None)%IL10(s1=1, s2=2)%IL10RA(r1=2, r2=None), ka_IL10RAIL10_IL10RA, kd_IL10RAIL10_IL10RA)
#Rule('IL10_binds_RB', IL10()%IL10RA(r2=None) + IL10RB(r1=None) | IL10()%IL10RA(r2=1)%IL10RB(r1=1), ka_RAIL10RA_IL10RB, kd_RAIL10RA_IL10RB)
#Rule('IL10_activate_Jak', Tyk2(s1=1,status='inactive')%IL10RB(r1=2, r3=1)%Jak1(s1=3,status='inactive')%IL10RA(r1=4, r2=2, r3=3)%IL10(s1=4, s2=5)%IL10RA(r1=5, r2=6, r3=7)%Jak1(s1=7,status='inactive')%IL10RB(r1=6, r3=8)%Tyk2(s1=8,status='inactive') |\
#     Tyk2(s1=1,status='active')%IL10RB(r1=2, r3=1)%Jak1(s1=3,status='active')%IL10RA(r1=4, r2=2, r3=3)%IL10(s1=4, s2=5)%IL10RA(r1=5, r2=6, r3=7)%Jak1(s1=7,status='active')%IL10RB(r1=6, r3=8)%Tyk2(s1=8,status='active'), IL10_activation, IL10_deactivation)
