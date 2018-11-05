from mcmc import *
from MCMC_reanalysis import *

def main():
    plt.close('all')
#    modelfiles = ['IFN_Models.IFN_alpha_altSOCS_ppCompatible','IFN_Models.IFN_beta_altSOCS_ppCompatible']
    modelfiles = ['IFN_Models.IFN_alpha_altSOCS_Internalization_ppCompatible','IFN_Models.IFN_beta_altSOCS_Internalization_ppCompatible']
# Write modelfiles
    print("Importing models")
    alpha_model = __import__(modelfiles[0],fromlist=['IFN_Models'])
    py_output = export(alpha_model.model, 'python')
    with open('ODE_system_alpha.py','w') as f:
        f.write(py_output)
    beta_model = __import__(modelfiles[1],fromlist=['IFN_Models'])
    py_output = export(beta_model.model, 'python')
    with open('ODE_system_beta.py','w') as f:
        f.write(py_output)
# =============================================================================
#   altSOCS model:        
#     p0=[['kpa',1.79E-5,0.1,'log'],['kSOCSon',1.70E-6,0.1,'log'],['kd4',0.87,0.2,'log'],
#         ['k_d4',0.86,0.5,'log'],['delR',-1878,500,'linear'],['meanR',2000,300,'linear']]
# 
#     our_priors_dict={'R1':[100,12000,None,None],'R2':[100,12000,None,None],
#              'kpa':[1.5E-9,1,np.log(1),4],'kSOCSon':[1.5E-11,0.07,np.log(1E-6),4],
#              'k_d4':[4E-5,0.9,np.log(0.006),1.8],'kd4':[0.002,44,np.log(0.3),1.8]}
# =============================================================================

#   altSOCS model with internalization        
    p0_int=[['kpa',1.79E-5,0.1,'log'],['kSOCSon',1.70E-6,0.2,'log'],['kd4',0.87,0.2,'log'],
        ['k_d4',0.86,0.5,'log'],['delR',-1878,500,'linear'],['meanR',2000,300,'linear'],
        ['kIntBasal_r1',1E-4,0.1,'log'],['kIntBasal_r2',2E-4,0.1,'log'],
        ['kint_IFN',5E-4,0.1,'log'],['krec_a1',3E-4,0.1,'log'],['krec_a2',5E-3,0.1,'log'],
        ['krec_b1',1E-4,0.1,'log'],['krec_b2',1E-3,0.1,'log']]

    int_priors_dict={'R1':[100,12000,None,None],'R2':[100,12000,None,None],
             'kpa':[1.5E-9,1,np.log(1),4],'kSOCSon':[1.5E-11,0.07,np.log(1E-6),4],
             'k_d4':[4E-5,0.9,np.log(0.006),1.8],'kd4':[0.002,44,np.log(0.3),1.8],
             'kIntBasal_r1':[1E-7,1E-1,None,None],'kIntBasal_r2':[2E-7,2E-1,None,None],
        'kint_IFN':[5E-7,5E-1,None,None],'krec_a1':[3E-7,3E-1,None,None],'krec_a2':[5E-6,5E0,None,None],
        'krec_b1':[1E-7,1E-1,None,None],'krec_b2':[1E-6,1E0,None,None]}
        
    #   (n, theta_0, beta, rho, chains, burn_rate=0.1, down_sample=1, max_attempts=6,
    #    pflag=True, cpu=None, randomize=True)
    MCMC(100, p0_int, int_priors_dict, 5, 1, 1, burn_rate=0.0, down_sample=1, max_attempts=0, cpu=1)
    #continue_sampling(3, 500, 0.1, 1)
# Testing functions
    #                    1E-6, 1E-6, 0.3, 0.006, 2E3, 2E3, 4
    #print(get_prior_logp(4E-3, 4E-3, 20, 0.1, 4E3, 1E3, 6)) 
    #print(get_likelihood_logp(4E-3, 4E-3, 20, 0.1, 4E3, 1E3, 6))
    #print(get_prior_logp(1E-6, 1E-6, 0.3, 0.006, 2E3, 2E3, 4)) 
    #print(get_likelihood_logp(1E-6, 1E-5, 0.3, 0.06, 2E3, 2E3, 4))
    
    #plot_parameter_distributions(df, title='parameter_distributions.pdf', save=True)
    #profile([1,2,3])
    
#if __name__ == '__main__':
#    main()
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'MCMC_Results-15-10-2018/')
sample_incomplete_sim(results_dir, 0.2, 30)
#resample_simulation(results_dir, 0.25, 60,check_convergence=True, plot_autocorr=True, check_corr=True)  

