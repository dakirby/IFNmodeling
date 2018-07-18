# Import of \IFN January 2018\Detailed IFN model - alpha model from 
# Rulebender to PySB. This is just a model file - must be run from a run file
# Importantly, there cannot be any use of PySB Expressions in order to use 
# the export functions from PySB.
# =============================================================================
# The major difference in usage for this file is that one cannot simply pass the 
# concentration of IFN as a parameter for the simulation any more. Instead, pass
# the pre-calculated value of IFN*volEC*NA to the parameter I
# =============================================================================

from pysb import Model, Parameter, Rule, Monomer, Initial, Observable, WILD
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

#vol. extracellular space , L
Parameter('volEC', 1E-5) # = 1/cell_dens 

# virtual vol. of plasma membrane , L 
Parameter('volPM', 2.76e-09 ) # = 2*rad_cell**2 + rad_cell*cell_thickness*4

# vol. of cytoplasm , L 
Parameter('volCP', 7.2e-15) # = cell_thickness*rad_cell**2  

Parameter('IFN', 1E-9)    # initial concentration in Molar
#number of copies per cell 
Parameter('I', 6.022E9) # = IFN*volEC*NA

Parameter('R1', 2000) #(2000/7.2e-15)*volCP#(8000/2.76e-9)*volPM#R -r#
Parameter('R2', 2000)#(2000/7.2e-15)*volCP#(8000/2.76e-9)*volPM#(8000/2.76e-9)*volPM#R +r#

Parameter('S', 1E4)#(1e4/7.2e-15)*volCP#

# Rate constants
# Divide by NA*V to convert bimolecular rate constants
# from /M/sec to /(molecule/cell)/sec

# Alpha block
Parameter('ka1', 3.321155762205247e-14) # = (2E5)/(NA*volEC)
Parameter('kd1', 1)#*10

# ligand-monomer binding  (scaled)
Parameter('ka2', 4.98173364330787e-13) # = (3e6)/(NA*volEC)
Parameter('kd2', 0.015) #*10              # ligand-monomer dissociation

#OLD PARAMETERS
#ka4            (33*1e-6/(1e-15*NA*(rad_cell)^2))#*44*10#*200#*200#0.0241116#ka4 in mm^2/ fmol/sec divide by area
#kd4            4.4/100#ka4*kd1*(2*2.6e-11)/ka1#
#NEW PARAMETERS IN VIVO
Parameter('ka4', 3.623188E-4) # = 1e-12/volPM
Parameter('kd4', 0.3) #*20

#Expression('q1', (ka1/kd1))
#Expression('q2', (ka2/kd2))
#Expression('q4', (ka4/kd4))
#Expression('q3', (q2*q4)/(q1))

Parameter('ka3', 3.623188E-4) # = ka4
Parameter('kd3', 3E-4) # = (ka3)/(q3)

Parameter('kSTATbinding',1e-6)
Parameter('kSTATunbinding',4)
Parameter('kpa', 1)#6e-5##OLD VALUE was (1e6)/(NA*volCP)=1e-6
Parameter('kpu', 1E-3)#1e-3

#Internalization: 
# Basal:
#Parameter('kIntBasal_r1', 0.0001)#0.0002
#Parameter('kIntBasal_r2', 0.00002)#0.000012
#Parameter('krec_r1', 0.0001)
#Parameter('krec_r2', 0.0001)
# Alpha:
# Asymmetric:
#Parameter('kint_a', 0.0005)
#Parameter('kdeg_a', 0.0008)
#Parameter('krec_a1', 0.0003)
#Parameter('krec_a2', 0.005)

#SOCS Feedback Inhibition
Parameter('kloc', 1.25e-3)
Parameter('kdeloc', 1e-2)
Parameter('kSOCSmRNA', 1e-3)
Parameter('mRNAdeg', 5e-4)
Parameter('mRNAtrans',1e-3)

Parameter('kSOCS', 5E-3) # 4e-3 was old value #Should sufficiently separate peak pSTAT from peak SOCS
Parameter('SOCSdeg', 5e-4*5)	#Maiwald*form factor
Parameter('kSOCSon', 1E-3) # = kpa
Parameter('kSOCSoff', 5E-4)#1.5e-3	#Rate of SOCS unbinding ternary complex. Very fudged. Was 1.5e-3 


# =============================================================================
# # Molecules
# =============================================================================
Monomer('IFN_alpha2',['r1','r2']) 

Monomer('IFNAR1',['re','ri','loc'],{'loc':['in','out']}) 
Monomer('IFNAR2',['re','ri','rs','loc'],{'loc':['in','out']}) 

Monomer('STAT',['j','loc','fdbk'],{'j':['U','P'],'loc':['Cyt','Nuc']})
Monomer('SOCSmRNA',['loc','reg'],{'loc':['Nuc','Cyt']})
Monomer('SOCS',['site'])			


# =============================================================================
# # Seed Species
# =============================================================================
Initial(IFN_alpha2(r1=None,r2=None), Ia)

Initial(IFNAR1(re=None, ri=None, loc='out'), R1)
Initial(IFNAR2(re=None, ri=None, rs=None, loc='out'), R2)

Initial(STAT(j='U',loc='Cyt',fdbk=None), S)

# =============================================================================
# # Observables
# Use 'WILD' for ?, use 'ANY' for +
# =============================================================================
Observable('Free_Ia', IFN_alpha2(r1=None, r2=None))
Observable('Free_R1', IFNAR1(re=None, ri=None, loc='out'))
Observable('Free_R2', IFNAR2(re=None, ri=None, rs=None, loc='out'))

Observable('R1Ia', IFNAR1(re=1,  ri=None, loc='out')%IFN_alpha2(r1=1, r2=None))
Observable('R2Ia', IFNAR2(re=1, ri=None, rs=None, loc='out')%IFN_alpha2(r1=1, r2=None))

Observable('IntR1', IFNAR1(re=WILD, ri=WILD, loc='in'))
Observable('IntR2', IFNAR2(re=WILD, ri=WILD, rs=WILD, loc='in'))
Observable('R1surface', IFNAR1(re=WILD, ri=WILD, loc='out'))
Observable('R2surface', IFNAR2(re=WILD, ri=WILD, rs=WILD, loc='out'))
Observable('T',IFNAR1(re=1, ri=None, loc='out')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2, ri=None, rs=None, loc='out'))

Observable('TotalSTAT', STAT(j=WILD,loc=WILD,fdbk=None))
Observable('pSTATCyt', STAT(j='P',loc='Cyt',fdbk=None))
Observable('pSTATNuc', STAT(j='P',loc='Nuc',fdbk=None))
Observable('TotalpSTAT', STAT(j='P',loc=WILD,fdbk=None))
Observable('SOCSAvail', SOCS(site=WILD))
Observable('SOCSmRNANuc', SOCSmRNA(loc='Nuc'))
Observable('SOCSmRNACyt', SOCSmRNA(loc='Cyt'))
Observable('BoundSOCS', IFNAR1(re=1, ri=None, loc='out')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2,ri=3,rs=None,loc='out')%SOCS(site=3))

# =============================================================================
# # Reaction rules
# =============================================================================
# Alpha block
Rule('IFN_bind_R1', IFNAR1(re=None,ri=None,loc='out') + IFN_alpha2(r1=None,r2=None) | IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=None), ka1, kd1 )
Rule('IFN_bind_R2', IFNAR2(re=None,ri=None,rs=None,loc='out') + IFN_alpha2(r1=None,r2=None) | IFNAR2(re=1,ri=None,rs=None,loc='out')%IFN_alpha2(r1=1,r2=None), ka2, kd2 )

Rule('IR1_bind_R2', IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=None) + IFNAR2(re=None,ri=None,loc='out') | IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2,ri=None,loc='out'), ka3, kd3)
Rule('IR2_bind_R1', IFNAR2(re=1,ri=None,rs=None,loc='out')%IFN_alpha2(r1=1,r2=None) + IFNAR1(re=None,ri=None,loc='out') | IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2,ri=None,rs=None,loc='out'), ka4, kd4)

#  STAT Block
# Alpha:
Rule('STAT_binding', IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2,ri=None,rs=None,loc='out') + STAT(j='U',loc='Cyt',fdbk=None) | IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2,ri=None,rs=3,loc='out')%STAT(j='U',loc='Cyt',fdbk=3), kSTATbinding, kSTATunbinding )
Rule('STAT_activation', IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2,ri=None,rs=3,loc='out')%STAT(j='U',loc='Cyt',fdbk=3) >> IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2,ri=None,rs=None,loc='out') +  STAT(j='P',loc='Cyt',fdbk=None), kpa)
Rule('STAT_dephos', STAT(j='P',loc='Cyt',fdbk=None) >> STAT(j='U',loc='Cyt',fdbk=None), kpu)
Rule('transport_STAT', STAT(j='P',loc='Cyt',fdbk=None) | STAT(j='P',loc='Nuc',fdbk=None), kloc, kdeloc)
# SOCS Block
Rule('synth_mRNA', STAT(j='P',loc='Nuc',fdbk=None) >> STAT(j='P',loc='Nuc',fdbk=None) + SOCSmRNA(loc='Nuc',reg=None), kSOCSmRNA)
Rule('transport_mRNA', SOCSmRNA(loc='Nuc',reg=None) >> SOCSmRNA(loc='Cyt',reg=None), mRNAtrans)
Rule('synth_SOCS', SOCSmRNA(loc='Cyt',reg=None) >> SOCSmRNA(loc='Cyt',reg=None) + SOCS(site=None), kSOCS)
Rule('degrade_SOCS', SOCS(site=None) >> None, SOCSdeg)
Rule('degrade_mRNA', SOCSmRNA(loc='Cyt',reg=None) >> None, mRNAdeg)

Rule('SOCS_inhibition', SOCS(site=None) + IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2,ri=None,rs=None,loc='out') | IFNAR1(re=1,ri=None,loc='out')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2,ri=3,rs=None,loc='out')%SOCS(site=3), kSOCSon, kSOCSoff)
   
# Internalization Block
# Basal:
#Rule('Basal_int1', IFNAR1(re=None, ri=None, loc='out') | IFNAR1(re=None, ri=None, loc='in'), kIntBasal_r1, krec_r1)
#Rule('Basal_int2', IFNAR2(re=None, ri=None, loc='out') | IFNAR2(re=None, ri=None, loc='in'), kIntBasal_r2, krec_r2)
#Rule('Basal_intT', IFNAR1(re=1, ri=None, loc='in')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2, ri=None, loc='in') >> IFNAR1(re=None, ri=None, loc='in') + IFNAR2(re=None, ri=None, loc='in'), kdeg_a)
# Alpha Block:
#Rule('IFNa_intT', IFNAR1(re=1, ri=None, loc='out')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2, ri=None, loc='out') >> IFNAR1(re=1,ri=None,loc='in')%IFN_alpha2(r1=1,r2=2)%IFNAR2(re=2,ri=None,loc='in'), kint_a)
#Rule('Rec_1', IFNAR1(re=None, ri=None, loc='in')>>IFNAR1(re=None, ri=None, loc='out'), krec_a1)
#Rule('Rec_2', IFNAR2(re=None, ri=None, loc='in')>>IFNAR2(re=None, ri=None, loc='out'), krec_a2)

