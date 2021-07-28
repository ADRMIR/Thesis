#This is the code that performs feature selection using several procedures 
#Already implemented:
#LAsso
#Pearson Correlation
#Spearman Correleation
#Average of methods
##
##
#Python 3.7 by André Miranda @ andre.lima.miranda@tecnico.ulisboa.pt

###################################################################################################
#######SET UP
###################################################################################################
import numpy as np
import matplotlib.pyplot as plt
#import skfuzzy as fuzz
import pandas as pd
from Data_load_final import inputs, outputs        #4th version of database


###################################################################################################
#######Choice of FS method
###################################################################################################
# 0 for random features as per matlab randi formula 
# 1 for Pearson
# 2 for Spearman
# 3 for chi2
# 4 for Lasso
# 5 for Mutual information
# 6 for a weighted (all methods) method


FS=6        #Choose one of the values above

###################################################################################################
#######Number of features
###################################################################################################

Max_number_of_features=20

###################################################################################################
#######Correlation matrix - Pearson
###################################################################################################
Limit=outputs.shape[1]


# Correlation just for the continuous inputs
# To_correlation=pd.concat([inputs[['idade', 'ACS altura', 'ACS peso']],outputs],axis=1)
# Correlation=To_correlation.corr()
# Correlation_no_outputs=Correlation.drop(Correlation.columns[-Limit:],axis=0)    #Drops the output rows 

To_correlation=pd.concat([inputs,outputs],axis=1)
Correlation=To_correlation.corr()
Correlation_no_outputs=Correlation.drop(Correlation.columns[-Limit:],axis=0)    #Drops the output rows 
# fig = plt.figure(figsize = (20,5))
# ax=sns.heatmap(Correlation_no_outputs,annot=True,vmin=-1,vmax=1,center=0,cmap='coolwarm',linewidths=1, linecolor='black')
# ax.set_title('Correlation heatmap inputs vs all',fontsize=12)
# ax.set_ylabel('Inputs',fontsize=12)
# ax.set_xlabel('All features',fontsize=12)


# print('______________________________________________________________________')
# print('                                                                      ')
# print('                            Pearson Correlation                       ')
# print('______________________________________________________________________')

#Most relevant features according to correlation
output_names=Correlation.columns[-Limit:]
for i in range(Limit):
    relevant=abs(Correlation_no_outputs.iloc[:,-(i+1)]).sort_values(axis=0,ascending=False,na_position='last')
    # print()
    # print()
    # print('The ' Max_number_of_features' most relevant features for "',output_names[-(i+1)] ,'" are:')
    # print()
    # print(relevant[:Max_number_of_features])
    # print()


#_=input('Proceed?')

#Retriving the top 15 features from the pearson correlation
complicação_pears_var=abs(Correlation_no_outputs['complicação pós-cirúrgica']).sort_values(axis=0,ascending=False,na_position='last')[:Max_number_of_features].index
obito_pears_var=abs(Correlation_no_outputs['óbito até 1 ano após cirurgia']).sort_values(axis=0,ascending=False,na_position='last')[:Max_number_of_features].index
dias_pears_var=abs(Correlation_no_outputs['nº dias na UCI']).sort_values(axis=0,ascending=False,na_position='last')[:Max_number_of_features].index
clavien_pears_var=abs(Correlation_no_outputs['classificação clavien-dindo']).sort_values(axis=0,ascending=False,na_position='last')[:Max_number_of_features].index
reint_pears_var=abs(Correlation_no_outputs[' reinternamento na UCI']).sort_values(axis=0,ascending=False,na_position='last')[:Max_number_of_features].index



###################################################################################################
#######Correlation matrix - Spearman
###################################################################################################


Correlation_spearman=To_correlation.corr(method='spearman')
Correlation_no_outputs_spearman=Correlation_spearman.drop(Correlation_spearman.columns[-Limit:],axis=0)    #Drops the output rows
# fig = plt.figure(figsize = (20,5))
# ax=sns.heatmap(Correlation_no_outputs_spearman,annot=True,vmin=-1,vmax=1,center=0,cmap='coolwarm',linewidths=1, linecolor='black')
# ax.set_title('Correlation heatmap (using Spearman coeeficient) inputs vs all',fontsize=12)
# ax.set_ylabel('Inputs',fontsize=12)
# ax.set_xlabel('All features',fontsize=12)


# print('______________________________________________________________________')
# print('                                                                      ')
# print('                            Spearman Correlation                      ')
# print('______________________________________________________________________')


#Most relevant features according to correlation
output_names_spearman=Correlation_spearman.columns[-Limit:]
for i in range(Limit):
    relevant_spearman=abs(Correlation_no_outputs_spearman.iloc[:,-(i+1)]).sort_values(axis=0,ascending=False,na_position='last')
    # print()
    # print()
    # print('The 15 most relevant features for "',output_names_spearman[-(i+1)] ,'" are:')
    # print()
    # print(relevant_spearman[:15])
    # print()


#_=input('Proceed?')

#Retriving the top 10 features from the spearman correlation
complicação_spear_var=abs(Correlation_no_outputs_spearman['complicação pós-cirúrgica']).sort_values(axis=0,ascending=False,na_position='last')[:Max_number_of_features].index
obito_spear_var=abs(Correlation_no_outputs_spearman['óbito até 1 ano após cirurgia']).sort_values(axis=0,ascending=False,na_position='last')[:Max_number_of_features].index
dias_spear_var=abs(Correlation_no_outputs_spearman['nº dias na UCI']).sort_values(axis=0,ascending=False,na_position='last')[:Max_number_of_features].index
clavien_spear_var=abs(Correlation_no_outputs_spearman['classificação clavien-dindo']).sort_values(axis=0,ascending=False,na_position='last')[:Max_number_of_features].index
reint_spear_var=abs(Correlation_no_outputs_spearman[' reinternamento na UCI']).sort_values(axis=0,ascending=False,na_position='last')[:Max_number_of_features].index


###################################################################################################
###Chi squared feature analysis
###################################################################################################
        
from sklearn.feature_selection import chi2 as chi2
from sklearn.feature_selection import SelectKBest

names=inputs.columns
selector=SelectKBest(chi2,k=Max_number_of_features)   #Selects the k best features based on chi2

complicação_chi2=selector.fit_transform(inputs,outputs['complicação pós-cirúrgica'].values)
columns=selector.get_support(indices=True)
complicação_chi2_var=names[columns]

obito_chi2=selector.fit_transform(inputs,outputs['óbito até 1 ano após cirurgia'].values)
columns=selector.get_support(indices=True)
obito_chi2_var=names[columns]

dias_chi2=selector.fit_transform(inputs,round(outputs['nº dias na UCI']).values)       #Rounded days
columns=selector.get_support(indices=True)
dias_chi2_var=names[columns]

clavien_chi2=selector.fit_transform(inputs,outputs['classificação clavien-dindo'].values)     
columns=selector.get_support(indices=True)
clavien_chi2_var=names[columns]

reint_chi2=selector.fit_transform(inputs,outputs[' reinternamento na UCI'].values)      
columns=selector.get_support(indices=True)
reint_chi2_var=names[columns]

###################################################################################################
###Mutual information feature analysis
###################################################################################################
        
from sklearn.feature_selection import mutual_info_classif as mutual
from sklearn.feature_selection import SelectKBest

names=inputs.columns
selector=SelectKBest(mutual,k=Max_number_of_features)   #Selects the k best features based on mutual

complicação_mutual=selector.fit_transform(inputs,outputs['complicação pós-cirúrgica'].values)
columns=selector.get_support(indices=True)
complicação_mutual_var=names[columns]

obito_mutual=selector.fit_transform(inputs,outputs['óbito até 1 ano após cirurgia'].values)
columns=selector.get_support(indices=True)
obito_mutual_var=names[columns]

dias_mutual=selector.fit_transform(inputs,round(outputs['nº dias na UCI']).values)       #Rounded days
columns=selector.get_support(indices=True)
dias_mutual_var=names[columns]

clavien_mutual=selector.fit_transform(inputs,outputs['classificação clavien-dindo'].values)     
columns=selector.get_support(indices=True)
clavien_mutual_var=names[columns]

reint_mutual=selector.fit_transform(inputs,outputs[' reinternamento na UCI'].values)      
columns=selector.get_support(indices=True)
reint_mutual_var=names[columns]

###################################################################################################
###Lasso regression
###################################################################################################

from sklearn import linear_model

def lasso_them(inputs,outputs):         #Creates the lasso model for the desired output
    lasso_model=linear_model.Lasso() 
    lasso_model.max_iter = 10000  
    lasso_betas=np.empty(shape=[0,inputs.shape[1]])
    alphaRange = []
    alpha = 0.001
    while alpha < 0.5:
        alphaRange.append(alpha)
        lasso_model.alpha = alpha
        lasso_model.fit(inputs.values, outputs.values)
        #coefs = np.hstack([lasso_model.intercept_, lasso_model.coef_])
        coefs=lasso_model.coef_
        lasso_betas = np.vstack([lasso_betas, coefs])
        alpha += 0.001
    return(lasso_betas)

lasso_betas_complicação=lasso_them(inputs,outputs['complicação pós-cirúrgica'])
lasso_betas_obito=lasso_them(inputs,outputs['óbito até 1 ano após cirurgia'])
lasso_betas_dias=lasso_them(inputs,outputs['nº dias na UCI'])
lasso_betas_clavien=lasso_them(inputs,outputs['classificação clavien-dindo'])
lasso_betas_reint=lasso_them(inputs,outputs[' reinternamento na UCI']) 


def plot_them_lassos(lasso, string):            #Plots both lasso plots
    x=np.transpose(list(range(1,51*1,1))).astype(float)/100
    
    plt.figure(figsize= (8,6))
    plt.plot(x[0:40],lasso[:40,:])
    #plt.title('Lasso coefficients across different lambda values (for {0:s})'.format(string))
    plt.xlabel('Lambda',fontsize=16)
    plt.ylabel('Coefficients',fontsize=16)
    
# plot_them_lassos(lasso_betas_complicação, "Complications")
# plot_them_lassos(lasso_betas_obito, "One-year mortality")
# plot_them_lassos(lasso_betas_dias, "nº dias na UCI")
# plot_them_lassos(lasso_betas_clavien, "classificação clavien-dindo")
# plot_them_lassos(lasso_betas_reint, " reinternamento na UCI")


def get_relevant(inputs, lasso_betas, out):    #Gets the top features of the lasso 
    index=np.nonzero(lasso_betas)
    # print("#############################################################")
    # print()
    # print('The lasso regression for "',out,'" resulted in ',inputs.columns[index].shape[0]," features, and they are: ")
    # print(inputs.columns[index])
    # print()
    return inputs.columns[index]

lasso_feat_complicaçao=get_relevant(inputs,lasso_betas_complicação[8],"complicação pós-cirúrgica")
lasso_feat_obito=get_relevant(inputs,lasso_betas_obito[2],"óbito até 1 ano após cirurgia após cirurgia")

lasso_feat_clavien=get_relevant(inputs,lasso_betas_clavien[6],"classificação clavien-dindo")
lasso_feat_reint=get_relevant(inputs,lasso_betas_reint[2]," reinternamento na UCI")
lasso_feat_dias=get_relevant(inputs,lasso_betas_dias[11],"nº dias na UCI")

###################################################################################################
##Random
###################################################################################################

seed=[15,2,36,33,33,45,8,34,19,36,15,33,18,47,20]
features=inputs.columns[seed]

###################################################################################################
##Voting system
###################################################################################################

pristine=all_scored_compli=all_scored_obito=all_scored_dias=all_scored_clavien=all_scored_reint=pd.DataFrame(np.zeros((names.shape[0],1)),names)             #Empty df with all the possible variables

def stack_scores(all_scored,var):       #Groups the scores into one
    rank=pd.DataFrame(list(range(var.size,0,-1)),var)        #Adding the weights accordinng to position
    all_scored=pd.concat([all_scored,rank],axis=0)
    all_scored=all_scored.groupby(by=all_scored.index, axis=0).sum()
    return(all_scored)

def lasso_ranked_scores(all_scored,var):       #Groups the lasso scores into one
    rank=pd.DataFrame(list(range(var.size,0,-1)),var)        #Adding the weights accordinng to position
    all_scored=pd.concat([all_scored,rank],axis=0)
    all_scored=all_scored.groupby(by=all_scored.index, axis=0).sum()
    return(all_scored)

### If statement to only choose the appropriate feature selection method
  
if FS==0:

    all_scored_compli=stack_scores(all_scored_compli,features)
    all_scored_obito=stack_scores(all_scored_obito,features)
    all_scored_dias=stack_scores(all_scored_dias,features)
    all_scored_clavien=stack_scores(all_scored_clavien,features)
    all_scored_reint=stack_scores(all_scored_reint,features)    

elif FS==1:

    ###################
    
    #Pearson feature selection
    
    all_scored_compli=stack_scores(all_scored_compli,complicação_pears_var)
    all_scored_obito=stack_scores(all_scored_obito,obito_pears_var)
    all_scored_dias=stack_scores(all_scored_dias,dias_pears_var)
    all_scored_clavien=stack_scores(all_scored_clavien,clavien_pears_var)
    all_scored_reint=stack_scores(all_scored_reint,reint_pears_var)
    
    ###################
    
elif FS==2:
    
    ###################
    
    #Spearman feature selection
    
    all_scored_compli=stack_scores(all_scored_compli,complicação_spear_var)
    all_scored_obito=stack_scores(all_scored_obito,obito_spear_var)
    all_scored_dias=stack_scores(all_scored_dias,dias_spear_var)
    all_scored_clavien=stack_scores(all_scored_clavien,clavien_spear_var)
    all_scored_reint=stack_scores(all_scored_reint,reint_spear_var)
    
    ###################

elif FS==3:    
    
    ###################
    
    #Chi2 feature selection
    
    all_scored_compli=stack_scores(all_scored_compli,complicação_chi2_var)
    all_scored_obito=stack_scores(all_scored_obito,obito_chi2_var)
    all_scored_dias=stack_scores(all_scored_dias,dias_chi2_var)
    all_scored_clavien=stack_scores(all_scored_clavien,clavien_chi2_var)
    all_scored_reint=stack_scores(all_scored_reint,reint_chi2_var)
    
    ###################

elif FS==4:    
    
    ###################
     
    #Lasso feature selection
    
    all_scored_compli=lasso_ranked_scores(all_scored_compli,lasso_feat_complicaçao)
    all_scored_obito=lasso_ranked_scores(all_scored_obito,lasso_feat_obito)
    all_scored_dias=lasso_ranked_scores(all_scored_dias,lasso_feat_dias)
    all_scored_clavien=lasso_ranked_scores(all_scored_clavien,lasso_feat_clavien)
    all_scored_reint=lasso_ranked_scores(all_scored_reint,lasso_feat_reint)
    
    ###################

elif FS==5:    
    
    ###################
     
    #Mutual information
    
    #Write the mutual information code
    
    
    all_scored_compli=stack_scores(all_scored_compli,complicação_mutual_var)
    all_scored_obito=stack_scores(all_scored_obito,obito_mutual_var)
    all_scored_dias=stack_scores(all_scored_dias,dias_mutual_var)
    all_scored_clavien=stack_scores(all_scored_clavien,clavien_mutual_var)
    all_scored_reint=stack_scores(all_scored_reint,reint_mutual_var)

    ###################

elif FS==6:    
    
    ###################
     
    #Pondered feature selection with all methods
    
    all_scored_compli=stack_scores(all_scored_compli,complicação_pears_var)
    all_scored_obito=stack_scores(all_scored_obito,obito_pears_var)
    all_scored_dias=stack_scores(all_scored_dias,dias_pears_var)
    all_scored_clavien=stack_scores(all_scored_clavien,clavien_pears_var)
    all_scored_reint=stack_scores(all_scored_reint,reint_pears_var)    
    
    all_scored_compli=stack_scores(all_scored_compli,complicação_spear_var)
    all_scored_obito=stack_scores(all_scored_obito,obito_spear_var)
    all_scored_dias=stack_scores(all_scored_dias,dias_spear_var)
    all_scored_clavien=stack_scores(all_scored_clavien,clavien_spear_var)
    all_scored_reint=stack_scores(all_scored_reint,reint_spear_var)    
    
    all_scored_compli=stack_scores(all_scored_compli,complicação_chi2_var)
    all_scored_obito=stack_scores(all_scored_obito,obito_chi2_var)
    all_scored_dias=stack_scores(all_scored_dias,dias_chi2_var)
    all_scored_clavien=stack_scores(all_scored_clavien,clavien_chi2_var)
    all_scored_reint=stack_scores(all_scored_reint,reint_chi2_var)   
    
    all_scored_compli=stack_scores(all_scored_compli,complicação_mutual_var)
    all_scored_obito=stack_scores(all_scored_obito,obito_mutual_var)
    all_scored_dias=stack_scores(all_scored_dias,dias_mutual_var)
    all_scored_clavien=stack_scores(all_scored_clavien,clavien_mutual_var)
    all_scored_reint=stack_scores(all_scored_reint,reint_mutual_var)   
    
    all_scored_compli=lasso_ranked_scores(all_scored_compli,lasso_feat_complicaçao)
    all_scored_obito=lasso_ranked_scores(all_scored_obito,lasso_feat_obito)
    all_scored_dias=lasso_ranked_scores(all_scored_dias,lasso_feat_dias)
    all_scored_clavien=lasso_ranked_scores(all_scored_clavien,lasso_feat_clavien)
    all_scored_reint=lasso_ranked_scores(all_scored_reint,lasso_feat_reint)
    

###################
#Variables
#Random

big_boi_compli=stack_scores(pristine,features)
big_boi_obito=stack_scores(pristine,features)

#Pearson feature selection

big_boi_compli=pd.concat([big_boi_compli,stack_scores(pristine,complicação_pears_var)],axis=1)
big_boi_obito=pd.concat([big_boi_obito,stack_scores(pristine,obito_pears_var)],axis=1)

#Spearman feature selection

big_boi_compli=pd.concat([big_boi_compli,stack_scores(pristine,complicação_spear_var)],axis=1)
big_boi_obito=pd.concat([big_boi_obito,stack_scores(pristine,obito_spear_var)],axis=1)

#Chi2 feature selection

big_boi_compli=pd.concat([big_boi_compli,stack_scores(pristine,complicação_chi2_var)],axis=1)
big_boi_obito=pd.concat([big_boi_obito,stack_scores(pristine,obito_chi2_var)],axis=1)

#Lasso feature selection

big_boi_compli=pd.concat([big_boi_compli,lasso_ranked_scores(pristine,lasso_feat_complicaçao)],axis=1)
big_boi_obito=pd.concat([big_boi_obito,lasso_ranked_scores(pristine,lasso_feat_obito)],axis=1)

#Mutual information

big_boi_compli=pd.concat([big_boi_compli,stack_scores(pristine,complicação_mutual_var)],axis=1)
big_boi_obito=pd.concat([big_boi_obito,stack_scores(pristine,obito_mutual_var)],axis=1)

#Pondered feature selection with all methods

pristine_compli=stack_scores(pristine,complicação_pears_var)
pristine_obito=stack_scores(pristine,obito_pears_var)

pristine_compli=stack_scores(pristine_compli,complicação_spear_var)
pristine_obito=stack_scores(pristine_obito,obito_spear_var)

pristine_compli=stack_scores(pristine_compli,complicação_chi2_var)
pristine_obito=stack_scores(pristine_obito,obito_chi2_var)

pristine_compli=stack_scores(pristine_compli,complicação_mutual_var)
pristine_obito=stack_scores(pristine_obito,obito_mutual_var)

pristine_compli=lasso_ranked_scores(pristine_compli,lasso_feat_complicaçao)
pristine_obito=lasso_ranked_scores(pristine_obito,lasso_feat_obito)

big_boi_compli=pd.concat([big_boi_compli,pristine_compli],axis=1)
big_boi_obito=pd.concat([big_boi_obito,pristine_obito],axis=1)

###################
###################


    
# print('#######################################################')
# print('Complicações final 15')

all_scored_compli=all_scored_compli.sort_values(by=0,axis=0,ascending=False,na_position='last')
# print(all_scored_compli.iloc[:15,:])
# print()
# print('Óbito final 15')

all_scored_obito=all_scored_obito.sort_values(by=0,axis=0,ascending=False,na_position='last')
# print(all_scored_obito.iloc[:15,:])
# print()
# print('Dias final 15')

all_scored_dias=all_scored_dias.sort_values(by=0,axis=0,ascending=False,na_position='last')
# print(all_scored_dias.iloc[:15,:])
# print()
# print('Clavien final 15')

all_scored_clavien=all_scored_clavien.sort_values(by=0,axis=0,ascending=False,na_position='last')
# print(all_scored_clavien.iloc[:15,:])
# print()
# print('Reinternamento final 15')

all_scored_reint=all_scored_reint.sort_values(by=0,axis=0,ascending=False,na_position='last')
# print(all_scored_reint.iloc[:15,:])



######################################################
#All of the feature selection and their results
import seaborn as sns

pears_compli=abs(Correlation_no_outputs['complicação pós-cirúrgica']).sort_values(ascending=False,na_position='last').to_frame().reset_index().rename(columns={'complicação pós-cirúrgica':'Score','index':'Feature'})
spear_compli=abs(Correlation_no_outputs_spearman['complicação pós-cirúrgica']).sort_values(ascending=False,na_position='last').to_frame().reset_index().rename(columns={'complicação pós-cirúrgica':'Score','index':'Feature'})
chi2_compli=abs(pd.DataFrame(data=chi2(inputs,outputs['complicação pós-cirúrgica'].values)[0],index=inputs.columns)).sort_values(by=0,ascending=False,na_position='last').reset_index().rename(columns={0:'Score','index':'Feature'})
mutual_compli=abs(pd.DataFrame(data=mutual(inputs,outputs['complicação pós-cirúrgica'].values),index=inputs.columns)).sort_values(by=0,ascending=False,na_position='last').reset_index().rename(columns={0:'Score','index':'Feature'})
lasso_compli=abs(pd.DataFrame(data=lasso_betas_complicação[8],index=inputs.columns)).sort_values(by=0,ascending=False,na_position='last').reset_index().rename(columns={0:'Score','index':'Feature'})
ponder_compli=all_scored_compli.reset_index().rename(columns={0:'Score','index':'Feature'})

pears_mort=abs(Correlation_no_outputs['óbito até 1 ano após cirurgia']).sort_values(ascending=False,na_position='last').to_frame().reset_index().rename(columns={'óbito até 1 ano após cirurgia':'Score','index':'Feature'})
spear_mort=abs(Correlation_no_outputs_spearman['óbito até 1 ano após cirurgia']).sort_values(ascending=False,na_position='last').to_frame().reset_index().rename(columns={'óbito até 1 ano após cirurgia':'Score','index':'Feature'})
chi2_mort=abs(pd.DataFrame(data=chi2(inputs,outputs['óbito até 1 ano após cirurgia'].values)[0],index=inputs.columns)).sort_values(by=0,ascending=False,na_position='last').reset_index().rename(columns={0:'Score','index':'Feature'})
mutual_mort=abs(pd.DataFrame(data=mutual(inputs,outputs['óbito até 1 ano após cirurgia'].values),index=inputs.columns)).sort_values(by=0,ascending=False,na_position='last').reset_index().rename(columns={0:'Score','index':'Feature'})
lasso_mort=abs(pd.DataFrame(data=lasso_betas_obito[2],index=inputs.columns)).sort_values(by=0,ascending=False,na_position='last').reset_index().rename(columns={0:'Score','index':'Feature'})
ponder_mort=all_scored_obito.reset_index().rename(columns={0:'Score','index':'Feature'})

compli_list=[pears_compli,spear_compli,chi2_compli,mutual_compli,lasso_compli,ponder_compli]
mort_list=[pears_mort,spear_mort,chi2_mort,mutual_mort,lasso_mort,ponder_mort]
method_list=['Pearson','Spearman','Chi_2','Mutual','Lasso','Pondered']

for compli,mort,method in zip(compli_list,mort_list,method_list):
    plt.figure(figsize = (5,5))
    sns.barplot(x='Feature',y='Score',data=compli.iloc[:10,:],color='orange')
    plt.xticks(rotation=90)
    plt.ylabel('Score',fontsize=16)
    plt.xlabel('Features',fontsize=16)
    plt.tight_layout()
    plt.savefig('C:/Documents/MEMec_2015-2020/Tese/Images/Feature_selection/' + 'Complications_features_' + method +'.png')
    plt.figure(figsize = (5,5))
    sns.barplot(x='Feature',y='Score',data=mort.iloc[:10,:],color='#DA70D6')
    plt.xticks(rotation=90)
    plt.ylabel('Score',fontsize=16)
    plt.xlabel('Features',fontsize=16)
    plt.tight_layout()
    plt.savefig('C:/Documents/MEMec_2015-2020/Tese/Images/Feature_selection/' + 'Death_features_' + method +'.png')

    