# Import of \IFN January 2018\Simplified SOCS model - alpha model from 
#   Rulebender to PySB. This is just a model file - must be run from a run file
from pysb import Model, Parameter, Expression, Initial, Monomer, Observable, Rule, WILD
# Begin Model
Model() 
# =============================================================================
# # Parameters
# =============================================================================
Parameter('NA', 6.022E23)   # Avogadro's number (molecues/mol)
Parameter('PI', 3.142)      # no unit

Parameter('rad_cell', 30E-6)      # radius of cell in m approx 30 micron
Parameter('cell_thickness', 8E-6) # height, m
Parameter('cell_dens', 1E5) # density of cells , /L
Parameter('width_PM', 1E-6) # effective width of membrane , m

Expression('volEC', 1/cell_dens) # vol. extracellular space , L
Expression('volPM', 2*rad_cell**2 + rad_cell*cell_thickness*4 ) # virtual vol. of plasma membrane , L
Expression('volCP', cell_thickness*rad_cell**2) # vol. of cytoplasm , L 

Parameter('IFN', 1E-9)    # initial concentration in Molar
Expression('I', IFN*volEC*NA) # number of copies per cell (~ 6.022e8 copies per cell)
Expression('Ia', I)
#Expression('Ib', I) 
#Parameter('r', 0) # Receptor assymmetry
#Parameter('R', 0) #
Parameter('R1', 2000) #(2000/7.2e-15)*volCP#(8000/2.76e-9)*volPM#R -r#
Parameter('R2', 2000)#(2000/7.2e-15)*volCP#(8000/2.76e-9)*volPM#(8000/2.76e-9)*volPM#R +r#

Parameter('S', 1E4)#(1e4/7.2e-15)*volCP#

# Rate constants
# Divide by NA*V to convert bimolecular rate constants
# from /M/sec to /(molecule/cell)/sec

# Alpha block
Expression('ka1', (2E5)/(NA*volEC))
Parameter('kd1', 1)#*10

Expression('ka2', (3e6)/(NA*volEC))   # ligand-monomer binding  (scaled)
Parameter('kd2', 0.015) #*10              # ligand-monomer dissociation

#OLD PARAMETERS
#ka4            (33*1e-6/(1e-15*NA*(rad_cell)^2))#*44*10#*200#*200#0.0241116#ka4 in mm^2/ fmol/sec divide by area
#kd4            4.4/100#ka4*kd1*(2*2.6e-11)/ka1#
#NEW PARAMETERS IN VIVO
Expression('ka4', 1e-12/(volPM)) #*20#e-12
Parameter('kd4', 0.3) #*20

#USP18-bound dissociation constants are higher
Parameter('USP18modfac',15) # =15 literature
Expression('kd4_USP18',kd4*USP18modfac)

Expression('q1', (ka1/kd1))
Expression('q2', (ka2/kd2))
Expression('q4', (ka4/kd4))
Expression('q3', (q2*q4)/(q1))

Expression('ka3', ka4)
Expression('kd3', ka3/q3) #(ka3)/(q3)
Expression('kd3_USP18',kd3*USP18modfac)

Parameter('kpa', 1E-6)#6e-5##OLD VALUE was (1e6)/(NA*volCP)=1e-6
Parameter('kpu', 1E-3)#1e-3

#Internalization: 
Parameter('Internalization_switch', 0)
# Basal:
Expression('kIntBasal_r1', 0.0001*Internalization_switch)#0.0002
Expression('kIntBasal_r2', 0.00002*Internalization_switch)#0.000012
Expression('krec_r1', 0.0001*Internalization_switch)
Expression('krec_r2', 0.0001*Internalization_switch)
# Alpha:
# Asymmetric:
Expression('kint_a', 0.0005*Internalization_switch)
Expression('kdeg_a', 0.0008*Internalization_switch)
Expression('krec_a1', 0.0003*Internalization_switch)
Expression('krec_a2', 0.005*Internalization_switch)
# Symmetric:
#kint_a				0.00025
#kdeg_a				0.00072
#krec_a1			0.0002
#krec_a2			0.0002

#SOCS Feedback Inhibition
Parameter('kSOCS', 4E-3) # 4e-3 was old value #Should sufficiently separate peak pSTAT from peak SOCS
Parameter('SOCSdeg', (5e-4)*5)	#Maiwald*form factor
Parameter('kSOCSonModifier',1) # this times kSOCSon is sometimes used
Parameter('kSOCSon', 1e-6)
Parameter('kSOCSoff', 5.5E-4)#1.5e-3	#Rate of SOCS unbinding ternary complex. Very fudged. Was 1.5e-3 


# =============================================================================
# # Molecules
# =============================================================================
Monomer('IFN_alpha2',['r1','r2']) 

Monomer('IFNAR1',['re','ri','loc'],{'loc':['in','out']}) 
Monomer('IFNAR2',['re','ri','loc'],{'loc':['in','out']}) 

Monomer('STAT',['j','loc','fdbk'],{'j':['U','P'],'loc':['Cyt','Nuc']})
Monomer('SOCSmRNA',['loc','reg'],{'loc':['Nuc','Cyt']})
Monomer('SOCS',['site'])			

Monomer('R2USP18',['re','ri','loc'],{'loc':['in','out']}) 

# =============================================================================
# # Seed Species
# =============================================================================
Initial(IFN_alpha2(r1=None,r2=None), Ia)

Initial(IFNAR1(re=None, ri=None, loc='out'), R1)

Parameter('fracUSP18',0) # 0.6 is reasonable when USP18 turned on
Expression('R2_0',(1-fracUSP18)*R2)
Expression('R2USP18_0',fracUSP18*R2)
Initial(IFNAR2(re=None, ri=None, loc='out'), R2_0)
Initial(R2USP18(re=None, ri=None, loc='out'), R2USP18_0)

Initial(STAT(j='U',loc='Cyt',fdbk=None), S)

# =============================================================================
# # Observables
# Use 'WILD' for ?, use 'ANY' for +
# =============================================================================
Observable('Free_Ia', IFN_alpha2(r1=None, r2=None))
Observable('Free_R1', IFNAR1(re=None, ri=None, loc='out'))
Observable('Free_R2', IFNAR2(re=None, ri=None, loc='out'))

Observable('R1Ia', IFNAR1(re=1,  ri=None, loc='out')%IFN_alpha2(r1=1, r2=None))
Observable('R2Ia', IFNAR2(re=1, ri=None, loc='out')%IFN_alpha2(r1=1, r2=None))

Observable('IntR1', IFNAR1(re=WILD, ri=WILD, loc='in'))
Observable('IntR2', IFNAR2(re=WILD, ri=WILD, loc='in'))
Observable('R1surface', IFNAR1(re=WILD, ri=WILD, loc='out'))
Observable('R2surface', IFNAR2(re=WILD, ri=WILD, loc='out'))
Observable('T',IFNAR1(re=1, ri=None, loc='out')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2, ri=None, loc='out'))

Observable('TotalSTAT', STAT(j=WILD,loc=WILD,fdbk=None))
Observable('pSTATCyt', STAT(j='P',loc='Cyt',fdbk=None))
Observable('pSTATNuc', STAT(j='P',loc='Nuc',fdbk=None))
Observable('TotalpSTAT', STAT(j='P',loc=WILD,fdbk=None))
Observable('SOCSAvail', SOCS(site=WILD))
Observable('SOCSmRNANuc', SOCSmRNA(loc='Nuc'))
Observable('SOCSmRNACyt', SOCSmRNA(loc='Cyt'))
Observable('BoundSOCS', IFNAR2(re=None, ri=3, loc='out')%SOCS(site=3))
Observable('TSOCS', IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2, ri=3, loc='out')%SOCS(site=3))

Observable('TUSP18',IFNAR1(re=1, ri=None, loc='out')%IFN_alpha2(r1=1,r2=2)%R2USP18(re=2, ri=None, loc='out'))

# =============================================================================
# # Reaction rules
# =============================================================================
# Alpha block
Rule('IFN_bind_R1', IFNAR1(re=None,ri=None,loc='out') + IFN_alpha2(r1=None,r2=None) | IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=None), ka1, kd1 )
Rule('IFN_bind_R2', IFNAR2(re=None,ri=WILD,loc='out') + IFN_alpha2(r1=None,r2=None) | IFNAR2(re=1,ri=WILD,loc='out')%IFN_alpha2(r1=1,r2=None), ka2, kd2 )

Rule('IFN_bind_R2USP18', R2USP18(re=None,ri=WILD,loc='out') + IFN_alpha2(r1=None,r2=None) | R2USP18(re=1,ri=WILD,loc='out')%IFN_alpha2(r1=1,r2=None), ka2, kd2 )


Rule('IR1_bind_R2', IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=None) + IFNAR2(re=None,ri=WILD,loc='out') | IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2,ri=WILD,loc='out'), ka3, kd3)
Rule('IR2_bind_R1', IFNAR2(re=1,ri=WILD,loc='out')%IFN_alpha2(r1=1,r2=None) + IFNAR1(re=None,ri=None,loc='out') | IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2,ri=WILD,loc='out'), ka4, kd4)

Rule('IR1_bind_R2USP18', IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=None) + R2USP18(re=None,ri=WILD,loc='out') | IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=2)%R2USP18(re=2,ri=WILD,loc='out'), ka3, kd3_USP18)
Rule('IR2USP18_bind_R1', R2USP18(re=1,ri=WILD,loc='out')%IFN_alpha2(r1=1,r2=None) + IFNAR1(re=None,ri=None,loc='out') | IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=2)%R2USP18(re=2,ri=WILD,loc='out'), ka4, kd4_USP18)

#  STAT Block
# Alpha:
Rule('activate_STAT', IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2,ri=None,loc='out') + STAT(j='U',loc='Cyt',fdbk=None) >> IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2,ri=None,loc='out') + STAT(j='P',loc='Cyt',fdbk=None), kpa )
Rule('USP18activate_STAT', IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=2)%R2USP18(re=2,ri=None,loc='out') + STAT(j='U',loc='Cyt',fdbk=None) >> IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=2)%R2USP18(re=2,ri=None,loc='out') + STAT(j='P',loc='Cyt',fdbk=None), kpa )

Rule('deactivate_STAT', STAT(j='P', loc='Cyt',fdbk=None) >> STAT(j='U', loc='Cyt',fdbk=None), kpu )

# SOCS Block
Rule('synth_SOCS', STAT(j='P', loc='Cyt',fdbk=None) >> STAT(j='P', loc='Cyt',fdbk=None) + SOCS(site=None), kSOCS)
Rule('degrade_SOCS', SOCS(site=None) >> None, SOCSdeg)
# SOCS Inhibition Feedback
# Alpha
Rule('SOCS_inhibition1', SOCS(site=None) + IFNAR2(re=WILD, ri=None, loc='out') | IFNAR2(re=WILD, ri=3, loc='out')%SOCS(site=3), kSOCSon, kSOCSoff)
Rule('USP18_SOCS_inhibition', SOCS(site=None) + R2USP18(re=WILD, ri=None, loc='out') | R2USP18(re=WILD, ri=3, loc='out')%SOCS(site=3), kSOCSon, kSOCSoff)

# Internalization Block
# Basal:
Rule('Basal_int1', IFNAR1(re=WILD, ri=WILD, loc='out') | IFNAR1(re=WILD, ri=WILD, loc='in'), kIntBasal_r1, krec_r1)
Rule('Basal_int2', IFNAR2(re=WILD, ri=WILD, loc='out') | IFNAR2(re=WILD, ri=WILD, loc='in'), kIntBasal_r2, krec_r2)
Rule('Basal_int3', R2USP18(re=WILD, ri=WILD, loc='out') | R2USP18(re=WILD, ri=WILD, loc='in'), kIntBasal_r2, krec_r2)
# Alpha Block:
Rule('IFNb_intT', IFNAR1(re=1, ri=WILD, loc='out')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2, ri=WILD, loc='out') >> IFNAR1(re=1,ri=None,loc='in')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2,ri=WILD,loc='in'), kint_a)
Rule('IFNb_intT_USP18', IFNAR1(re=1, ri=WILD, loc='out')%IFN_alpha2(r1=1,r2=2)%R2USP18(re=2, ri=WILD, loc='out') >> IFNAR1(re=1,ri=None,loc='in')%IFN_alpha2(r1=1,r2=2)%R2USP18(re=2,ri=WILD,loc='in'), kint_a)
Rule('Rec_1', IFNAR1(re=None, ri=WILD, loc='in')>>IFNAR1(re=None, ri=WILD, loc='out'), krec_a1)
Rule('Rec_2', IFNAR2(re=None, ri=WILD, loc='in')>>IFNAR2(re=None, ri=WILD, loc='out'), krec_a2)
Rule('Rec_3', R2USP18(re=None, ri=WILD, loc='in')>>R2USP18(re=None, ri=WILD, loc='out'), krec_a2)

