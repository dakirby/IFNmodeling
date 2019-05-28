def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
import re
# =============================================================================
# score_k4_models() takes output files from fit_pysb_model and aggregates models
# which differ only in their choice of K4, so that IFNalpha and IFNbeta can be
# fit with the same parameters except for choice of K4
# Inputs:
#     alpha_modfiles: a string or list of modelfiles for fitting IFNalpha models
#     beta_modfiles: a string or list of modelfiles for fitting IFNbeta models
# Outputs:
#     alpha_and_beta_k4.txt : a text file containing ranked models (from best to worst)
#                             and the model parameters to get the best score        
# =============================================================================
def score_k4_models(alpha_modfiles, beta_modfiles):
    if type(alpha_modfiles)==list:
        if type(beta_modfiles)!=list or len(beta_modfiles)!=len(alpha_modfiles):
            print("Provide an equal number of alpha and beta model files")
            return 1
    scoreboard = {}
    outfile_header=""
    if type(alpha_modfiles)==str:
        with open(alpha_modfiles,'r') as f:
            f.readline()
            f.readline()
            header = f.readline()
            outfile_header=header
            header=header.split()
            score_index = header.index('score')
            k4_index = header.index('kd4')
            numMods = file_len(alpha_modfiles)-3
            print("Ranking {} models for best K4".format(numMods))
            for l in range(numMods):
                line = f.readline().split()
                key = [[header[i],float(line[i])] for i in range(len(line)) if(header[i]!='kd4' and header[i]!='score')]
                score = float(line[score_index])
                k4 = float(line[k4_index])
                current_score = scoreboard.setdefault(str(key), [1E8,1E8,0.3,0.006]) # adds key to dictionary with default values, unless key already exists (in which case it returns value) 
                if current_score[0] > score:
                    scoreboard[str(key)][0]=score
                    scoreboard[str(key)][2]=k4*0.3
        with open(beta_modfiles,'r') as f:
            f.readline()
            f.readline()
            header = f.readline()
            header=header.split()
            score_index = header.index('score')
            k4_index = header.index('k_d4')
            numMods = file_len(beta_modfiles)-3
            for l in range(numMods):
                line = f.readline().split()
                key = [[header[i],float(line[i])] for i in range(len(line)) if(header[i]!='k_d4' and header[i]!='score')]
                score = float(line[score_index])
                k4 = float(line[k4_index])
                current_score = scoreboard.setdefault(str(key), [1E8,1E8,0.3,0.006]) # adds key to dictionary with default values, unless key already exists (in which case it returns value) 
                if current_score[1] > score:
                    scoreboard[str(key)][1]=score
                    scoreboard[str(key)][3]=k4*0.006
    else:
        for mod in alpha_modfiles:
            with open(mod,'r') as f:
                f.readline()
                f.readline()
                header = f.readline()
                outfile_header=header
                header=header.split()
                score_index = header.index('score')
                k4_index = header.index('kd4')
                numMods = file_len(mod)-3
                for l in range(numMods):
                    line = f.readline().split()
                    key = [[header[i],float(line[i])] for i in range(len(line)) if(header[i]!='kd4' and header[i]!='score')]
                    score = float(line[score_index])
                    k4 = float(line[k4_index])
                    current_score = scoreboard.setdefault(str(key), [1E8,1E8,0.3,0.006]) # adds key to dictionary with default values, unless key already exists (in which case it returns value) 
                    if current_score[0] > score:
                        scoreboard[str(key)][0]=score
                        scoreboard[str(key)][2]=k4*0.3
        for mod in beta_modfiles:
            with open(mod,'r') as f:
                f.readline()
                f.readline()
                header = f.readline()
                header=header.split()
                score_index = header.index('score')
                k4_index = header.index('k_d4')
                numMods = file_len(mod)-3
                for l in range(numMods):
                    line = f.readline().split()
                    key = [[header[i],float(line[i])] for i in range(len(line)) if(header[i]!='k_d4' and header[i]!='score')]
                    score = float(line[score_index])
                    k4 = float(line[k4_index])
                    current_score = scoreboard.setdefault(str(key), [1E8,1E8,0.3,0.006]) # adds key to dictionary with default values, unless key already exists (in which case it returns value) 
                    if current_score[1] > score:
                        scoreboard[str(key)][1]=score
                        scoreboard[str(key)][3]=k4*0.006

    print("Reduced list from {} models to {} models".format(numMods*len(beta_modfiles),len(scoreboard)))
    leaderboard = [(k[0], scoreboard[k[0]]) for k in sorted(scoreboard.items(), key=lambda e: e[1][0]+e[1][1])]#THIS WILL NEED TO CHANGE SO THAT THESE ARE RANKED BY TOTAL SCORE
    #Write results to output file
    f = open('K4_fit.txt', 'w')
    f.close()
    with open('K4_fit.txt', 'a') as outfile:
        outfile.write("# models: "+str(len(leaderboard))+"\n") 
        outfile.write("---------------------------------------------------------\n")
        outfile_header = outfile_header.replace('kd4          ','')
        outfile_header = outfile_header.replace('score','kd4          k_d4          score')
        outfile.write(outfile_header)
        kd4_index = outfile_header.split().index('kd4')
        for key, score in leaderboard:
            # Example string below shows what happens before and after each line of code
            #[['kpa', 0.001], ['kSOCSon', 3.1622776601683792e-08]]
            key = key[3:-2]
            #kpa', 0.001], ['kSOCSon', 3.1622776601683792e-08
            key = re.split("', |\], \['", key)
            #["kpa" , "0.001" , "kSOCSon" , "3.1622776601683792e-08"]
            key = key[0:kd4_index*2]+['kd4',score[2],'k_d4',score[3]]+key[kd4_index*2+1:-1]
            mod_p = ""
            for i in range(1,len(key),2):
                mod_p += '{:.3e}'.format(float(key[i]))+"    "
            outfile.write(mod_p+str(score[0]+score[1])+"\n")
       
        
score_k4_models(['modelfit_alpha_half.txt','modelfit_alpha_2half.txt'],['modelfit_beta_half.txt','modelfit_beta_2half.txt'])
        
        