# Import of \IFN January 2018\Simplified SOCS model - beta model from 
#   Rulebender to PySB. This is just a model file - must be run from a run file
from pysb import *
Model() 
# =============================================================================
# # Parameters
# =============================================================================
Parameter('NA', 6.022E23)   # Avogadro's number (molecues/mol)
Parameter('PI', 3.142)      # no unit

Parameter('rad_cell', 30E-6)      # radius of cell in m approx 10 micron
Parameter('cell_thickness', 8E-6) # height, m
Parameter('cell_dens', 1E5) # density of cells , /L
Parameter('width_PM', 1E-6) # effective width of membrane , m

#vol. extracellular space , L
Parameter('volEC', 1E-5) # = 1/cell_dens 

# virtual vol. of plasma membrane , L 
Parameter('volPM', 2.76e-09 ) # = 2*rad_cell**2 + rad_cell*cell_thickness*4

# vol. of cytoplasm , L 
Parameter('volCP', 7.2e-15) # = cell_thickness*rad_cell**2  

Parameter('IFN', 1E-9)    # initial concentration in Molar
#number of copies per cell 
Parameter('I', 6.022E9) # = IFN*volEC*NA
Parameter('Ib', 6.022E9) # = I

Parameter('R1', 2000) #(2000/7.2e-15)*volCP#(8000/2.76e-9)*volPM#R -r#
Parameter('R2', 2000)#(2000/7.2e-15)*volCP#(8000/2.76e-9)*volPM#(8000/2.76e-9)*volPM#R +r#

Parameter('S', 1E4)#(1e4/7.2e-15)*volCP#

# Rate constants
# Divide by NA*V to convert bimolecular rate constants
# from /M/sec to /(molecule/cell)/sec

# Beta block
Parameter('k_a1', 4.98E-14)
Parameter('k_d1', 0.030)#*10

Parameter('k_a2', 8.30e-13)   # ligand-monomer binding  (scaled)
Parameter('k_d2', 0.001) #*10              # ligand-monomer dissociation

#NEW PARAMETERS IN VIVO
Parameter('k_a3', 3.62e-4)

Parameter('k_a4', 3.62e-4) #*20#e-12
Parameter('k_d4', 0.006) #*20

Parameter('k_d3', 1.2e-5) #(ka3)/(q3)

Parameter('kpa', 1E-6)#6e-5##OLD VALUE was (1e6)/(NA*volCP)=1e-6
Parameter('kpu', 1E-3)#1e-3

#Internalization: 
# Basal:
#Parameter('kIntBasal_r1', 0.0001)#0.0002
#Parameter('kIntBasal_r2', 0.00002)#0.000012
#Parameter('krec_r1', 0.0001)
#Parameter('krec_r2', 0.0001)
# Beta:
# Asymmetric:
#Parameter('kint_b', 0.0002)
#Parameter('kdeg_b', 0.0008)
#Parameter('krec_b1', 0.0001)
#Parameter('krec_b2', 0.001)

#SOCS Feedback Inhibition
Parameter('kSOCS', 4E-3) # 4e-3 was old value #Should sufficiently separate peak pSTAT from peak SOCS
Parameter('SOCSdeg', (5e-4)*5)	#Maiwald*form factor
Parameter('kSOCSon', 1e-6)
Parameter('kSOCSoff', 5.5E-4)#1.5e-3	#Rate of SOCS unbinding ternary complex. Very fudged. Was 1.5e-3 


# =============================================================================
# # Molecules
# =============================================================================
Monomer('IFN_beta',['r1','r2']) 

Monomer('IFNAR1',['re','ri','loc'],{'loc':['in','out']}) 
Monomer('IFNAR2',['re','ri','loc'],{'loc':['in','out']}) 

Monomer('STAT',['j','loc','fdbk'],{'j':['U','P'],'loc':['Cyt','Nuc']})
Monomer('SOCSmRNA',['loc','reg'],{'loc':['Nuc','Cyt']})
Monomer('SOCS',['site'])			


# =============================================================================
# # Seed Species
# =============================================================================
Initial(IFN_beta(r1=None,r2=None), Ib)

Initial(IFNAR1(re=None, ri=None, loc='out'), R1)
Initial(IFNAR2(re=None, ri=None, loc='out'), R2)

Initial(STAT(j='U',loc='Cyt',fdbk=None), S)

# =============================================================================
# # Observables
# Use 'WILD' for ?, use 'ANY' for +
# =============================================================================
Observable('Free_Ib', IFN_beta(r1=None, r2=None))
Observable('Free_R1', IFNAR1(re=None, ri=None, loc='out'))
Observable('Free_R2', IFNAR2(re=None, ri=None, loc='out'))

Observable('R1Ib', IFNAR1(re=1,  ri=None, loc='out')%IFN_beta(r1=1, r2=None))
Observable('R2Ib', IFNAR2(re=1, ri=None, loc='out')%IFN_beta(r1=1, r2=None))

Observable('IntR1', IFNAR1(re=WILD, ri=WILD, loc='in'))
Observable('IntR2', IFNAR2(re=WILD, ri=WILD, loc='in'))
Observable('R1surface', IFNAR1(re=WILD, ri=WILD, loc='out'))
Observable('R2surface', IFNAR2(re=WILD, ri=WILD, loc='out'))
Observable('T',IFNAR1(re=1, ri=None, loc='out')%IFN_beta(r1=1,r2=2)%IFNAR2(re=2, ri=None, loc='out'))

Observable('TotalSTAT', STAT(j=WILD,loc=WILD,fdbk=None))
Observable('pSTATCyt', STAT(j='P',loc='Cyt',fdbk=None))
Observable('pSTATNuc', STAT(j='P',loc='Nuc',fdbk=None))
Observable('TotalpSTAT', STAT(j='P',loc=WILD,fdbk=None))
Observable('SOCSAvail', SOCS(site=WILD))
Observable('SOCSmRNANuc', SOCSmRNA(loc='Nuc'))
Observable('SOCSmRNACyt', SOCSmRNA(loc='Cyt'))
Observable('BoundSOCS', IFNAR1(re=1, ri=None, loc='out')%IFN_beta(r1=1,r2=2)%IFNAR2(re=2,ri=3,loc='out')%SOCS(site=3))

# =============================================================================
# # Reaction rules
# =============================================================================
# Alpha block
Rule('IFN_bind_R1', IFNAR1(re=None,ri=None,loc='out') + IFN_beta(r1=None,r2=None) | IFNAR1(re=1,ri=None,loc='out')%IFN_beta(r1=1,r2=None), k_a1, k_d1 )
Rule('IFN_bind_R2', IFNAR2(re=None,ri=None,loc='out') + IFN_beta(r1=None,r2=None) | IFNAR2(re=1,ri=None,loc='out')%IFN_beta(r1=1,r2=None), k_a2, k_d2 )

Rule('IR1_bind_R2', IFNAR1(re=1,ri=None,loc='out')%IFN_beta(r1=1,r2=None) + IFNAR2(re=None,ri=None,loc='out') | IFNAR1(re=1,ri=None,loc='out')%IFN_beta(r1=1,r2=2)%IFNAR2(re=2,ri=None,loc='out'), k_a3, k_d3)
Rule('IR2_bind_R1', IFNAR2(re=1,ri=None,loc='out')%IFN_beta(r1=1,r2=None) + IFNAR1(re=None,ri=None,loc='out') | IFNAR1(re=1,ri=None,loc='out')%IFN_beta(r1=1,r2=2)%IFNAR2(re=2,ri=None,loc='out'), k_a4, k_d4)

#  STAT Block
# Alpha:
Rule('activate_STAT', IFNAR1(re=1,ri=None,loc='out')%IFN_beta(r1=1,r2=2)%IFNAR2(re=2,ri=None,loc='out') + STAT(j='U',loc='Cyt',fdbk=None) >> IFNAR1(re=1,ri=None,loc='out')%IFN_beta(r1=1,r2=2)%IFNAR2(re=2,ri=None,loc='out') + STAT(j='P',loc='Cyt',fdbk=None), kpa )

Rule('deactivate_STAT', STAT(j='P', loc='Cyt',fdbk=None) >> STAT(j='U', loc='Cyt',fdbk=None), kpu )

# SOCS Block
Rule('synth_SOCS', STAT(j='P', loc='Cyt',fdbk=None) >> STAT(j='P', loc='Cyt',fdbk=None) + SOCS(site=None), kSOCS)
Rule('degrade_SOCS', SOCS(site=None) >> None, SOCSdeg)
# SOCS Inhibition Feedback
# Alpha
Rule('SOCS_inhibition', SOCS(site=None) + IFNAR1(re=1, ri=None, loc='out')%IFN_beta(r1=1, r2=2)%IFNAR2(re=2, ri=None, loc='out') | IFNAR1(re=1, ri=None, loc='out')%IFN_beta(r1=1, r2=2)%IFNAR2(re=2, ri=3, loc='out')%SOCS(site=3), kSOCSon, kSOCSoff)
  
# Internalization Block
# Basal:
#Rule('Basal_int1', IFNAR1(re=None, ri=None, loc='out') | IFNAR1(re=None, ri=None, loc='in'), kIntBasal_r1, krec_r1)
#Rule('Basal_int2', IFNAR2(re=None, ri=None, loc='out') | IFNAR2(re=None, ri=None, loc='in'), kIntBasal_r2, krec_r2)
#Rule('Basal_intT', IFNAR1(re=1, ri=None, loc='in')%IFN_beta(r1=1,r2=2)%IFNAR2(re=2, ri=None, loc='in') >> IFNAR1(re=None, ri=None, loc='in') + IFNAR2(re=None, ri=None, loc='in'), kdeg_b)
# Alpha Block:
#Rule('IFNb_intT', IFNAR1(re=1, ri=None, loc='out')%IFN_beta(r1=1,r2=2)%IFNAR2(re=2, ri=None, loc='out') >> IFNAR1(re=1,ri=None,loc='in')%IFN_beta(r1=1,r2=2)%IFNAR2(re=2,ri=None,loc='in'), kint_b)
#Rule('Rec_1', IFNAR1(re=None, ri=None, loc='in')>>IFNAR1(re=None, ri=None, loc='out'), krec_b1)
#Rule('Rec_2', IFNAR2(re=None, ri=None, loc='in')>>IFNAR2(re=None, ri=None, loc='out'), krec_b2)

