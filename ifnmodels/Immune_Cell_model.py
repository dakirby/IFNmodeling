from pysb import Model, Parameter, Expression, Initial, Monomer, Observable, Rule, WILD
# Begin Model
Model()
# =============================================================================
# # Parameters
# =============================================================================

# =============================================================================
# # Molecules
# =============================================================================
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

# Granulocyte CSF receptor (promotes neutrophil proliferation)
Monomer('G-CSF', ['s1'])
Monomer('G-CSFR', ['r1', 'r2'])

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
Monomer('STAT5') # I suspect that this is STAT5a, not STAT5b

# Kinases
Monomer('Jak1', ['s1'])
Monomer('Jak2', ['s1'])
Monomer('Jak3', ['s1'])
Monomer('Tyk2', ['s1'])

# =============================================================================
# # Reaction rules
# =============================================================================
# GP130 family
# G-CSFR + G-CSF <-> 2x(G-CSFR.G-CSF) -> active rececptor + JAK2 -> pSTAT3
# IL6RA + IL6 <-> IL6RA.IL6 + GP130 <-> 2x(IL6RA.IL6.GP130) + Jak1 -> pSTAT3
# IL11RA + IL11 <-> IL11RA.IL11 + GP130 <-> 2x(IL11RA.IL11.GP130) + Jak2 -> pSTAT1

# Beta_c family
# GM-CSFRA binds GM-CSF, then binds (Jak2.Beta_c.Beta_c.Jak2), then
#   binds another GM-CSFRA.GM-CSF, then this hexamer binds another
#   hexamer to form a dodecamer which phosphorylates STAT5
# IL3RA + IL3 <-> 4x(IL3RA.IL3) + 2x(Jak2.Beta_c.Beta_c.Jak2) <-> Active receptor + STAT5 # same idea as GM-CSF
# IL5RA + IL5 <-> 4x(IL5RA.IL5) + 2x(Jak2.Beta_c.Beta_c.Jak2) <-> Active receptor + STAT5 # same idea as GM-CSF

# Gamma_chain family
# multiple receptor assembly methods available to IL2
# IL2RA + IL2 <-> IL2RA.IL2 + IL2RB <-> IL2RA.IL2.IL2RB + Gamma_chain <-> IL2RA.IL2.IL2RB.Gamma_chain + Jak1 + Jak3 -> pSTAT5
# IL4RA + IL4 + Gamma_chain <-> Jak1 + IL4RA.IL4.Gamma_chain + Jak3 -> STAT6 # Jak1 binds to IL4RA and Jak3 to Gamma_c
# IL7RA + IL7 + Gamma_chain <-> Jak1 + IL7RA.IL7.Gamma_chain + Jak3 -> STAT5 # Jak1 binds to IL4RA and Jak3 to Gamma_c

# 2x(IL10RA.IL10.IL10RB) + Tyk2 + Jak1 -> pSTAT1 + pSTAT3
# IL12RB1 + IL12RB2 + IL12 + Jak2 + Tyk2 -> pSTAT4 dimers
# IL13RA1 + IL13 + IL13RA2 <-> IL13RA1.IL13.IL13RA2
# IL13RA1 + IL13 <-> IL13RA1.IL13 + IL4RA -> IL13RA1.IL13.IL4RA + Jak1 -> STAT6
# IL4RA + IL4 <-> IL4RA.IL4 + IL13RA1 -> IL13RA1.IL4.IL4RA + Jak1 -> STAT6
# IL20RA + IL20 + IL20RB + Jak1 + Jak2 -> STAT3
# IL22RA1 + IL22 + IL10RA + Jak1 + Jak2 -> STAT3


