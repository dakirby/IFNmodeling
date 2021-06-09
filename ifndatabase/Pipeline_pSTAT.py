#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:02:02 2019
Modified May 2019 by dkirby to separate cells based on FSC

@author: gbonnet
@author: dkirby
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import os
import warnings
import fcsparser

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

path=os.getcwd()
path_RawFiles=os.path.join(os.path.dirname(os.getcwd()), "Compensated Export\\")

Experiment='pSTAT1 kinetics [20190214]'

FileNames=[]
for ii in range(96):
    FileNames.append([x for x in os.listdir(path_RawFiles) if x.find('0'+str(ii+1)+'_Lymphocytes')>-1][0])

IndexFiles=[x[20:20+x[20:].find('_')] for x in FileNames]
Rows=['A','B','C','D','E','F','G','H']
Columns=['01','02','03','04','05','06','07','08','09','10','11','12']

Concentrations_IFNa=(np.array([1e-7,1e-8,3e-9,1e-9,3e-10,1e-10,1e-11,0]))
Concentrations_IFNb=(np.array([2e-9,6e-10,2e-10,6e-11,2e-11,6e-12,2e-13,0]))

Timepoints=([2.5,5,7.5,10,20,60])
Cytokines=['IFN-alpha','IFN-beta']
Concentrations=pd.DataFrame(np.array([Concentrations_IFNa,Concentrations_IFNb]).T,columns=Cytokines)
Readout=['pSTAT1 in B cells','pSTAT1 in CD8+ T cells','pSTAT1 in CD4+ T cells']
df1=pd.DataFrame(np.zeros((96,)))
df1.index=FileNames
df1.columns=['Concentration (Mol)']
df2=[]#=pd.DataFrame(np.zeros((96,)),index=FileNames,columns=['Cytokine'])
df3=pd.DataFrame(np.zeros((96,)),index=FileNames,columns=['Time (min)'])

for ff in FileNames:
    ii=ff[20:20+ff[20:].find('_')]
    if int(ii[1:]) %2 ==1:
        df1.loc[ff]['Concentration (Mol)']=Concentrations_IFNa[Rows.index(ii[0])]
        df2.append('IFN-alpha')
        df3.loc[ff]['Time (min)']=Timepoints[int(np.floor((int(ii[1:])-1)/2))]
    else:
        df1.loc[ff]['Concentration (Mol)']=Concentrations_IFNb[Rows.index(ii[0])]
        df2.append('IFN-beta')
        df3.loc[ff]['Time (min)']=Timepoints[int(np.floor((int(ii[1:])-1)/2))]

df2=pd.DataFrame(df2,index=FileNames,columns=['Cytokine'])
df0=pd.DataFrame([ff[20:20+ff[20:].find('_')] for ff in FileNames],columns=['Well'],index=FileNames)
Conditions=pd.concat([df0,df1,df2,df3],axis=1)


AllData = {}
markers=[]

for ff in FileNames:
    meta,df=fcsparser.parse(path_RawFiles+ff,reformat_meta=True)
    if len(AllData)==0:
        Fluorophores=[x for x in meta['_channel_names_'] if (x.find('FJComp')>-1) or (x.find('FSC')>-1)]
        channels=list(meta['_channels_']['$PnN'].values)
        for mm in Fluorophores:
            if mm.find('SC')>-1:
                markers.append(mm)
            else:
                identifier='$P'+str(channels.index(mm)+1)+'S'
                markers.append(meta[identifier]+' ('+mm[mm.find('-')+1:-2]+')')
    AllData[len(AllData)]=df[Fluorophores]
    AllData[len(AllData)-1].columns=markers

pSTAT1=pd.DataFrame(np.zeros((96,5)),columns=['pSTAT1 in B cells','pSTAT1 in CD8+ T cells','pSTAT1 in CD4+ T cells',
                                              'pSTAT1 in Small B cells','pSTAT1 in Large B cells'],index=FileNames)

for ff in FileNames:
        index=(AllData[FileNames.index(ff)]['CD19 (PE ( 561 ))'] > 600) & \
                        (AllData[FileNames.index(ff)]['IA_IE (BV510)'] > 2000)
        pSTAT1.loc[ff]['pSTAT1 in B cells']=np.sinh(np.mean(np.arcsinh(AllData[FileNames.index(ff)][index]['pSTAT1 (FITC)'])))

        index=(AllData[FileNames.index(ff)]['CD3 (PE-Cy5)'] > 200) & \
                        (AllData[FileNames.index(ff)]['CD8 (APC-Cy7)'] > 3000)
        pSTAT1.loc[ff]['pSTAT1 in CD8+ T cells']=np.sinh(np.mean(np.arcsinh(AllData[FileNames.index(ff)][index]['pSTAT1 (FITC)'])))

        index=(AllData[FileNames.index(ff)]['CD3 (PE-Cy5)'] > 10) & \
                        (AllData[FileNames.index(ff)]['CD4 (APC)'] > 1000)
        pSTAT1.loc[ff]['pSTAT1 in CD4+ T cells']=np.sinh(np.mean(np.arcsinh(AllData[FileNames.index(ff)][index]['pSTAT1 (FITC)'])))

        # Small and large B Cells
            # Get average cell size (average FSC) to use as a threshold for small and large cells
        index = (AllData[FileNames.index(ff)]['CD19 (PE ( 561 ))'] > 600) & \
                (AllData[FileNames.index(ff)]['IA_IE (BV510)'] > 2000)
        well_threshold_size = AllData[FileNames.index(ff)][index]['FSC-A'].quantile(q=0.8)

        small_index = (AllData[FileNames.index(ff)]['CD19 (PE ( 561 ))'] > 600) & \
                      (AllData[FileNames.index(ff)]['IA_IE (BV510)'] > 2000) & \
                      (AllData[FileNames.index(ff)]['FSC-A'] <= well_threshold_size)

        pSTAT1.loc[ff]['pSTAT1 in Small B cells'] = np.sinh(np.mean(np.arcsinh(AllData[FileNames.index(ff)][small_index]['pSTAT1 (FITC)'])))
        pSTAT1.loc[ff]['pSTAT1 in Large B cells'] = np.sinh(np.mean(np.arcsinh(AllData[FileNames.index(ff)][~small_index]['pSTAT1 (FITC)'])))

pSTAT1=pd.concat([Conditions,pSTAT1],axis=1)


pickle.dump(pSTAT1,open('pSTAT1.pkl','wb'))
pSTAT1.to_csv('pSTAT1.csv')
#%%
fig=plt.figure(num=10,figsize=(10,10))
for ii in range(3):
    ax=fig.add_subplot(2,2,ii+1)
    ax.matshow(np.log10(pSTAT1.iloc[:,ii+4].values.reshape((8,12))))
    ax.set_title(pSTAT1.columns[ii+4]+'\n')


#%%
couleur_conc=sns.color_palette("plasma", Concentrations.shape[0])
for measurement in Readout:

    for cyt in Cytokines:

        fig=plt.figure(figsize=(15,10))
        for conc in Concentrations[cyt]:

            index_cyt=(pSTAT1['Cytokine']==cyt).values
            index_conc=(pSTAT1['Concentration (Mol)']==conc).values

            ax=fig.add_subplot(111)
            temp=pSTAT1[index_cyt&index_conc][['Time (min)',measurement]].values
            plt.plot(temp[:,0],temp[:,1],'-o',c=couleur_conc[list(Concentrations[cyt]).index(conc)],label='['+cyt+']='+str(conc)+' Mol')
            plt.legend(loc=0)
            ax.set_xlabel('Time(min)')
            ax.set_ylabel('pSTAT1')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='top left', bbox_to_anchor=(1, 1))
        plt.title(measurement +' in response to '+cyt)

        plt.savefig('Kinetics - '+measurement +' in response to '+cyt+'.png')
        plt.close(fig)

#%%
couleur_time=sns.color_palette("viridis", Concentrations.shape[0])
for measurement in Readout:

    for cyt in Cytokines:

        fig=plt.figure(figsize=(15,10))
        for time in Timepoints:

            index_cyt=(pSTAT1['Cytokine']==cyt).values
            index_time=(pSTAT1['Time (min)']==time).values

            ax=fig.add_subplot(111)
            temp=np.sort(pSTAT1[index_cyt&index_time][['Concentration (Mol)',measurement]].values,axis=0)
            plt.semilogx(temp[:,0],temp[:,1],'-o',c=couleur_time[list(Timepoints).index(time)],label='Time = '+str(time)+' min')
            plt.legend(loc=0)
            ax.set_xlabel('Concentration (Mol)')
            ax.set_ylabel('pSTAT1')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='top left', bbox_to_anchor=(1, 1))
        plt.title(measurement +' in response to '+cyt)

        plt.savefig('Dose response - '+measurement +' in response to '+cyt+'.png')
        plt.close(fig)
