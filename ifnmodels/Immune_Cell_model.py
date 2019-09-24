from pysb import Model, Parameter, Expression, Initial, Monomer, Observable, Rule, WILD
# Begin Model
Model()
# =============================================================================
# # Parameters
# =============================================================================

# =============================================================================
# # Molecules
# =============================================================================
# Granulocyte CSF receptor (promotes neutrophil proliferation)
Monomer('G-CSF', ['s1'])
Monomer('G-CSFR', ['r1', 'r2'])
# IL10
Monomer('IL10', ['s1', 's2'])
Monomer('IL10RA')
Monomer('IL10RB')
# IL12
Monomer('IL12RB1')
Monomer('IL12RB2')
# IL13
Monomer('IL13RA1')
Monomer('IL13RA2')
# IL20
Monomer('IL20RB')
Monomer('IL20RA')
# IL22
Monomer('IL22RA1')
# IL23
Monomer('IL23RA')
# IL27
Monomer('IL27RA')
# Type 1 IFN
Monomer('IFNAR2', ['r1', 'r2'])
Monomer('IFNAR1', ['r1', 'r2'])
# Type 2 IFN
Monomer('IFNGR1', ['r1', 'r2'])
Monomer('IFNGR2', ['r1', 'r2'])

# Gamma Chain (shared)
Monomer('Gamma_Chain')
Monomer('IL2RA')
Monomer('IL2RB')
Monomer('IL4RA')
Monomer('IL7R')
Monomer('IL9R')
Monomer('IL15RA')
Monomer('IL21R')

# GP130 (shared)
Monomer('GP130')
Monomer('IL11RA')
Monomer('IL6RA')

# Beta_c (shared)
Monomer('Beta_c', ['r1', 'r2'])
Monomer('GM-CSFRB', ['r1', 'r2'])
Monomer('GM-CSFRA', ['r1', 'r2'])
Monomer('IL3RA')
Monomer('IL5RA')


# STATs
Monomer('STAT1')
Monomer('STAT2')
Monomer('STAT3')
Monomer('STAT4')
Monomer('STAT5')

# Kinases
Monomer('Jak1')
Monomer('Jak2')
Monomer('Jak3')
Monomer('Tyk2')

# =============================================================================
# # Reaction rules
# =============================================================================
# G-CSFR + G-CSF + JAK2 -> pSTAT3                                           (?)
# 2x(GM-CSFRA.GM-CSF.Beta_c.GM-CSF.GM-CSFRB) + 2x(Jak2) -> pSTAT5            (?)
# Beta_c + IL3RA + IL3 + Jak2 -> pSTAT5
# 2x(IL10RA.IL10.IL10RB) + Tyk2 + Jak1 -> pSTAT1 + pSTAT3

