#Final Fuzzy decision tree code to pass on to matlab inference rules
##
##
#Python 3.7 by André Miranda @ andre.lima.miranda@tecnico.ulisboa.pt

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import pandas as pd
import seaborn as sns
import math

import sys

sys.path.append("C:\Documents\MEMec_2015-2020\Tese\Python\Feature_selection")   #Goes to the other directory and gets the desired files


# Data_load import inputs, outputs
from Data_load_final import inputs, outputs
from sklearn import preprocessing as pre
from scipy import stats

min_max_scaler = pre.MinMaxScaler()

##############################
#Membership functions creation
##############################

def membership_creation(inputs):
    

    Default_membership_x=np.arange(inputs.shape[0])
    #Standardizing
    Default_membership_x=np.reshape(min_max_scaler.fit_transform(Default_membership_x.reshape(-1,1)),inputs.shape[0])   #Reshape due to function restrictions
    

    
    #3 membership functions from [0 to 1]
    
    mf_3_small=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0,0.2),inputs.shape[0])      
    mf_3_median=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0.5,0.2),inputs.shape[0])    
    mf_3_large=np.reshape(fuzz.membership.gaussmf(Default_membership_x,1,0.2),inputs.shape[0])     
    
    # plt.figure()
    
    # plt.plot(Default_membership_x,mf_3_small, label='Small')
    # plt.plot(Default_membership_x,mf_3_median, label='Median')
    # plt.plot(Default_membership_x,mf_3_large, label='Large')
    # plt.legend()
    # plt.ylim(0,1.2)
    # plt.xlim(0,1)
    
    
    mf=pd.DataFrame(data={'Small_3':mf_3_small,'Median_3':mf_3_median,'Large_3':mf_3_large})
    
    return mf,Default_membership_x

mf,Default_membership_x=membership_creation(inputs)

##############################
#Membership interpretation
##############################


def clc_memb(Default,mf,var):       #Calculates the membership value for the 15 different membership functions
    for i in range(mf.shape[1]):
        if i==0:
            result=fuzz.interp_membership(Default,mf.iloc[:,i].values,var).reshape([var.shape[0],1])
        else:
            result=np.concatenate((result,fuzz.interp_membership(Default,mf.iloc[:,i].values,var).reshape([var.shape[0],1])),axis=1)
    return result
        
def dataframe_append(Default,mf,var):       #Calculates the membership value for all features
    for i in range(var.shape[1]):
        if i==0:
            result_data=clc_memb(Default,mf,var.iloc[:,i].values)
        else:
            result_data=np.concatenate((result_data,clc_memb(Default,mf,var.iloc[:,i].values)),axis=1)
        
    for j in range(var.shape[1]):           #Creates the names of the features
        if j==0:
            columns_names=[var.columns[j]+' '+mf.columns[k] for k in range(mf.shape[1])] #For each feature there are 15 options
        else:
            columns_names=columns_names+[var.columns[j]+' '+mf.columns[k] for k in range(mf.shape[1])]
    result=pd.DataFrame(data=result_data,columns=columns_names)
    return result
        
     
        
fuzzified_inputs=dataframe_append(Default_membership_x,mf,inputs)       #Obtains the fuzzified features


##############################
#Fuzzy Decision tree
##############################

from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from Feature_selection import all_scored_dias, all_scored_clavien, all_scored_reint, big_boi_compli, big_boi_obito


from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.utils import resample

from time import time

def scores(Y_Test,Y_Pred):
    accu=accuracy_score(Y_Test,Y_Pred)
    f1=f1_score(Y_Test,Y_Pred)
    mathew=matthews_corrcoef(Y_Test,Y_Pred)
    roc=roc_auc_score(Y_Test,Y_Pred)
    kappa=cohen_kappa_score(Y_Test,Y_Pred)
    recall=recall_score(Y_Test,Y_Pred)
    confusion=confusion_matrix(Y_Test,Y_Pred).flatten()
    
    score_values=np.hstack(([accu,f1,mathew,roc,kappa,recall],confusion))      #Stacks accuracy, F1-score, MCC, ROC, Kappa, confusion matrix in a single array
    
    score=pd.DataFrame([score_values],columns=["Accuracy","F1 Score","MCC","ROC AUC","Cohen's Kappa","Recall","CM (1,1)","CM (1,2)","CM (2,1)","CM (2,2)"])
    return score

print('Beggining fuzzy decision tree training')
print()

########################################
####Definition of desired prediction####
########################################

#Choose the feature selection method
# 0 for random features as per matlab randi formula 
# 1 for Pearson
# 2 for Spearman
# 3 for chi2
# 4 for Lasso
# 5 for Mutual information
# 6 for a weighted (all methods) method
# 7 for all of the methods

Choice=1

if Choice!=7:
    run_counter=1    #variable to choose the number of runs 
else:
    run_counter=7
    Choice=0

Number_of_features=10

#['nº dias na UCI', 'destino após UCI', 'complicação pós-cirúrgica',
#       'classificação clavien-dindo',
#       'óbito até 1 ano após cirurgia', ' reinternamento na UCI']    Pick one of these

outcome=outputs['complicação pós-cirúrgica']
#outcome=outputs['óbito até 1 ano após cirurgia']
#outcome=outputs['classificação clavien-dindo']

#method='balanced'
method='unbalanced'

####################################################################
#Data balancing


sss=StratifiedShuffleSplit(1,test_size=0.2, random_state=1)
  
for train_index, test_index in sss.split(inputs, outcome):             #Cycle to create the stratified training and testing sets
    inputs, test_inputs = inputs.iloc[train_index], inputs.iloc[test_index]
    outcome, y_test_fuzzy_FE = outcome.iloc[train_index], outcome.iloc[test_index]

if method=='balanced':
    input_ones=inputs[outcome==1]
    
    input_zeros=inputs[outcome==0]
    
    outcome_ones=outcome[outcome==1]
    
    outcome_zeros=outcome[outcome==0]
    
    random_sampling=resample(input_ones,random_state=1,replace=True,n_samples=input_zeros.shape[0])
    
    random_outcome_sampling=resample(outcome_ones,random_state=1,replace=True,n_samples=outcome_zeros.shape[0])
    
    inputs=pd.concat([input_zeros,random_sampling], ignore_index=True)
    
    outcome=pd.concat([outcome_zeros,random_outcome_sampling],ignore_index=True)
    
    mf,Default_membership_x=membership_creation(inputs)
    
    mf2,Default_membership_x2=membership_creation(test_inputs)
    
elif method=='unbalanced':
    input_ones=inputs[outcome==1]
    
    input_zeros=inputs[outcome==0]
    
    outcome_ones=outcome[outcome==1]
    
    outcome_zeros=outcome[outcome==0]
    
    random_sampling=resample(input_zeros,random_state=1,replace=False,n_samples=input_ones.shape[0])
    
    random_outcome_sampling=resample(outcome_zeros,random_state=1,replace=False,n_samples=outcome_ones.shape[0])
    
    inputs=pd.concat([input_ones,random_sampling], ignore_index=True)
    
    outcome=pd.concat([outcome_ones,random_outcome_sampling],ignore_index=True)
    
    mf,Default_membership_x=membership_creation(inputs)
    
    mf2,Default_membership_x2=membership_creation(test_inputs)

####################################################################



big_boi_mean_and_std=pd.DataFrame(index=["No Rank mean", "No Rank std", "Pearson mean", "Pearson std", "Spearman mean", "Spearman std", "Chi mean", "Chi std", "Lasso mean", "Lasso std", "Mutual mean", "Mutual std", "Pondered mean", "Pondered std"],columns=["Accuracy","F1 Score","MCC","ROC AUC","Cohen's Kappa","Recall"])
big_boi_best=pd.DataFrame(index=["No Rank", "Pearson", "Spearman", "Chi", "Lasso", "Mutual", "Pondered"],columns=["Accuracy","F1 Score","MCC","ROC AUC","Cohen's Kappa","Recall"])


def update(big_boi_mean_and_std,big_boi_best,avg_result_fuzzy_FE,std_results,best_model_result,l):    #Function to update the results with all feature selection methods
    big_boi_mean_and_std.iloc[2*l,:]=avg_result_fuzzy_FE.iloc[0,:6].values
    big_boi_mean_and_std.iloc[2*l+1,:]=std_results.iloc[0,:6].values
    big_boi_best.iloc[l,:]=best_model_result.iloc[0,:6].values
    return(big_boi_mean_and_std,big_boi_best)


for l in range(run_counter):

#all_scored_compli, all_scored_obito, all_scored_dias, all_scored_clavien,
#        all_scored_reint    choose accordingly

    if outcome.name=='complicação pós-cirúrgica':                                               #This if cycle chooses the appropriate feature selection method
        proper_feature_selection=big_boi_compli.iloc[:,Choice].sort_values(ascending=False)
    elif outcome.name=='óbito até 1 ano após cirurgia':
        proper_feature_selection=big_boi_obito.iloc[:,Choice].sort_values(ascending=False)
    elif outcome.name=='classificação clavien-dindo':
        proper_feature_selection=all_scored_clavien
    

    ########################
    
    Number_of_models=200
    
    maximum_depth=15
    
    
    percent=0.5      #Percentage of positive cases to be used in training
    
    training=0.80       #Percentage of data to use in training
    
    # j=0
    t1=time()
    
    
    avg_result=avg_result_fuzzy=avg_result_fuzzy_FE=pd.DataFrame(columns=["Avg Accuracy","Avg F1 Score","Avg MCC","Avg ROC AUC","Avg Cohen's Kappa","Avg Recall","Avg CM (1,1)","Avg CM (1,2)","Avg CM (2,1)","Avg CM (2,2)"])        #Initializes the scores dataframe
    
    best_model_result=pd.DataFrame(columns=["Accuracy","F1 Score","MCC","ROC AUC","Cohen's Kappa","Recall","CM (1,1)","CM (1,2)","CM (2,1)","CM (2,2)"])        #Initializes the scores dataframe
    std_results=pd.DataFrame(columns=["Std Accuracy","Std F1 Score","Std MCC","Std ROC AUC","Std Cohen's Kappa","Std Recall","CM (1,1)","CM (1,2)","CM (2,1)","CM (2,2)"])        #Initializes the scores dataframe
         
    best_model={0:[]}
    all_models={0:[]}
    diction=0
    diction_final=0
    # for number in range(1,30):
    
        # fuzzified_inputs_FE=dataframe_append(Default_membership_x,mf,inputs[proper_feature_selection.index[:number].values]) #Fuzzification of the "number" best features 
    
    
    fuzzified_inputs_FE=dataframe_append(Default_membership_x,mf,inputs[proper_feature_selection.index[:Number_of_features].values]) #Fuzzification of the Number_of_features best features 
      
    X_test_fuzzy_FE=dataframe_append(Default_membership_x2,mf2,test_inputs[proper_feature_selection.index[:Number_of_features].values]) #Fuzzification of the Number_of_features best features for complication
    
         
    # for j in range(1,7):
    j=5
    
    t0 = time()
    
    
    result_fuzzy_FE=pd.DataFrame(columns=["Accuracy","F1 Score","MCC","ROC AUC","Cohen's Kappa","Recall","CM (1,1)","CM (1,2)","CM (2,1)","CM (2,2)"])        #Initializes the scores dataframe
    
    
    for i in range(Number_of_models):
        
        # Standard Model
        # clf=DecisionTreeClassifier(random_state=i)
        
        # Model creation and parameter exploration
        
        # clf = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
        #                     max_depth=j, max_features=None, max_leaf_nodes=None,
        #                     min_impurity_decrease=0.0, min_impurity_split=None,
        #                     min_samples_leaf=19, min_samples_split=12,
        #                     min_weight_fraction_leaf=0.0, presort='deprecated',
        #                     random_state=None, splitter='random')
        
        # Best parameters found with GridSearchCV with enphasis on Recall for complications
        
        # clf=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
        #                     max_depth=6, max_features=None, max_leaf_nodes=None,
        #                     min_impurity_decrease=0.0, min_impurity_split=None,
        #                     min_samples_leaf=19, min_samples_split=12,
        #                     min_weight_fraction_leaf=0.0, presort='deprecated',
        #                     random_state=None, splitter='random')
        
        # Best Parameters found with GridsearchCV with emphasis on Recall for mortality
        
        # clf=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
        #                     max_depth=9, max_features=None, max_leaf_nodes=None,
        #                     min_impurity_decrease=0.0, min_impurity_split=None,
        #                     min_samples_leaf=5, min_samples_split=13,
        #                     min_weight_fraction_leaf=0.0, presort='deprecated',
        #                     random_state=None, splitter='random')
        
        # Best Parameters found with GridsearchCV with emphasis on MCC for complications with pearson feature selection
        # clf=DecisionTreeClassifier( ccp_alpha=0.0, class_weight=None, criterion='gini',
        #                   max_depth=7, max_features=None, max_leaf_nodes=None,
        #                   min_impurity_decrease=0.0, min_impurity_split=None,
        #                   min_samples_leaf=3, min_samples_split=13,
        #                   min_weight_fraction_leaf=0.0, presort='deprecated',
        #                   random_state=None, splitter='random')
        
        # Best Parameters found with GridsearchCV with emphasis on MCC for mortality with pearson feature selection
        # clf=DecisionTreeClassifier( ccp_alpha=0.0, class_weight=None, criterion='entropy',
        #                   max_depth=3, max_features=None, max_leaf_nodes=None,
        #                   min_impurity_decrease=0.0, min_impurity_split=None,
        #                   min_samples_leaf=9, min_samples_split=13,
        #                   min_weight_fraction_leaf=0.0, presort='deprecated',
        #                   random_state=None, splitter='random')



        ###########################################################################################################
        #Gridsearch results for complications with emphasis on Recall
        ###########################################################################################################
    
        if outcome.name=='complicação pós-cirúrgica':    
    
            if l==0:
            #Random
    
                clf=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                        max_depth=9, max_features=None, max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=17, min_samples_split=4,
                        min_weight_fraction_leaf=0.0, presort='deprecated',
                        random_state=None, splitter='random')

            elif l==1:
            #Pearson
    
                clf=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                        max_depth=9, max_features=None, max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=15, min_samples_split=5,
                        min_weight_fraction_leaf=0.0, presort='deprecated',
                        random_state=None, splitter='random')
        
            elif l==2:        
            #Spearman
    
                clf=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                        max_depth=6, max_features=None, max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=29, min_samples_split=25,
                        min_weight_fraction_leaf=0.0, presort='deprecated',
                        random_state=None, splitter='random')
        
            elif l==3:     
            #chi2
    
                clf=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                        max_depth=6, max_features=None, max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=20, min_samples_split=19,
                        min_weight_fraction_leaf=0.0, presort='deprecated',
                        random_state=None, splitter='random')
        
            elif l==4: 
            #Lasso
    
                clf=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                        max_depth=9, max_features=None, max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=23, min_samples_split=3,
                        min_weight_fraction_leaf=0.0, presort='deprecated',
                        random_state=None, splitter='random')
        
            elif l==5:         
            #Mutual information
    
                clf=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                        max_depth=8, max_features=None, max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=23, min_samples_split=15,
                        min_weight_fraction_leaf=0.0, presort='deprecated',
                        random_state=None, splitter='random')
        
                
            elif l==6: 
            #weighted (all methods) method
    
                clf=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                        max_depth=9, max_features=None, max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=13, min_samples_split=27,
                        min_weight_fraction_leaf=0.0, presort='deprecated',
                        random_state=None, splitter='random')


        # ###########################################################################################################
        # #Gridsearch results for mortality with emphasis on Recall
        # ###########################################################################################################
    
        elif outcome.name=='óbito até 1 ano após cirurgia':   
    
            if l==0:
            #Random
    
                clf=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                        max_depth=8, max_features=None, max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=3, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, presort='deprecated',
                        random_state=None, splitter='random')

            elif l==1:
            #Pearson
    
                clf=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                        max_depth=9, max_features=None, max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=2, min_samples_split=3,
                        min_weight_fraction_leaf=0.0, presort='deprecated',
                        random_state=None, splitter='best')
    
    
            elif l==2:         
            #Spearman
        
                clf=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                        max_depth=9, max_features=None, max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=2, min_samples_split=4,
                        min_weight_fraction_leaf=0.0, presort='deprecated',
                        random_state=None, splitter='best')
    
            elif l==3:     
            #chi2
    
                clf=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                        max_depth=8, max_features=None, max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=3, min_samples_split=3,
                        min_weight_fraction_leaf=0.0, presort='deprecated',
                        random_state=None, splitter='random')
    
            elif l==4: 
            #Lasso
    
                clf=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                        max_depth=8, max_features=None, max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=2, min_samples_split=6,
                        min_weight_fraction_leaf=0.0, presort='deprecated',
                        random_state=None, splitter='random')
    
            elif l==5:         
            #Mutual information
    
                clf=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                        max_depth=9, max_features=None, max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=2, min_samples_split=5,
                        min_weight_fraction_leaf=0.0, presort='deprecated',
                        random_state=None, splitter='best')
    
            elif l==6: 
            #weighted (all methods) method

                clf=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                        max_depth=9, max_features=None, max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=3, min_samples_split=14,
                        min_weight_fraction_leaf=0.0, presort='deprecated',
                        random_state=None, splitter='random')



    
        ##################################################################
        ##Model run with feature selection according to previous selection
        ##################################################################  
    
        #X_train_fuzzy_FE, X_test_fuzzy_FE, y_train_fuzzy_FE, y_test_fuzzy_FE = train_test_split(fuzzified_inputs_FE, outcome, test_size=1-training, random_state=i)
    
    
        X_train_fuzzy_FE=fuzzified_inputs_FE
        y_train_fuzzy_FE=outcome    
    
        # sss=StratifiedShuffleSplit(1,test_size=0.25, random_state=i)
          
        # for train_index, test_index in sss.split(fuzzified_inputs_FE, outcome):             #Cycle to create the stratified training and testing sets
        #     X_train_fuzzy_FE, X_test_fuzzy_FE = fuzzified_inputs_FE.iloc[train_index], fuzzified_inputs_FE.iloc[test_index]
        #     y_train_fuzzy_FE, y_test_fuzzy_FE = outcome.iloc[train_index], outcome.iloc[test_index]
        
    
    
        DT_fuzzy_FE_clf=clf.fit(X_train_fuzzy_FE,y_train_fuzzy_FE)      #Trains the classifier                      
        
        y_predicted_fuzzy_FE=DT_fuzzy_FE_clf.predict(X_test_fuzzy_FE)   #Obtains the predicted classes
        
        score_fuzzy_FE=scores(y_test_fuzzy_FE,y_predicted_fuzzy_FE)   #Calculates the scores
        
        result_fuzzy_FE=result_fuzzy_FE.append(score_fuzzy_FE)     #Stacks the scores in a result variable
        
        all_models[diction]=DT_fuzzy_FE_clf            #Stores all the loop models
        diction=diction+1
        
        if i==round(Number_of_models/4):
            print()
            print()
            print()
            print('25% done')
        elif i==2*round(Number_of_models/4):
            print()
            print()
            print()
            print('50% done')
        elif i==3*round(Number_of_models/4):
            print()
            print()
            print()
            print('75% done')
        elif i==4*round(Number_of_models/4):
            print()
            print()
            print()
            print('DONE')
    
    print()
    print( round((time() - t0)/60), " minutes to train ", Number_of_models,"  models with ", j+1," layers")
    print()
    diction=0
    
    result_fuzzy_FE=result_fuzzy_FE.reset_index(drop=True)      #Resets the indexes to allow finding the random seed
    
    best_model_result=best_model_result.append(result_fuzzy_FE.iloc[np.argmax(result_fuzzy_FE.iloc[:,5]),:])    #Chooses the best model based on MCC
    
    best_model[diction_final]=all_models[np.argmax(result_fuzzy_FE.iloc[:,5])]      #Chooses the best odel based on (0-Accuracy,1-F1 Score,2-MCC,3-ROC AUC,4-Cohen's Kappa,5-Recall)
    diction_final=diction_final+1
    
    std_results=std_results.append(pd.DataFrame([np.std(result_fuzzy_FE).values],columns=["Std Accuracy","Std F1 Score","Std MCC","Std ROC AUC","Std Cohen's Kappa","Std Recall","CM (1,1)","CM (1,2)","CM (2,1)","CM (2,2)"]))        #Initializes the scores dataframe
       
    avg_result_fuzzy_FE_to_pass=pd.DataFrame([(pd.DataFrame.mean(result_fuzzy_FE).values)],columns=["Avg Accuracy","Avg F1 Score","Avg MCC","Avg ROC AUC","Avg Cohen's Kappa","Avg Recall","Avg CM (1,1)","Avg CM (1,2)","Avg CM (2,1)","Avg CM (2,2)"])
    avg_result_fuzzy_FE=avg_result_fuzzy_FE.append(avg_result_fuzzy_FE_to_pass)
    
    
    print()
    print()
    print("Total elapsed time: ",round((time() - t1)/60)," minutes")  

    big_boi_mean_and_std,big_boi_best=update(big_boi_mean_and_std,big_boi_best,avg_result_fuzzy_FE,std_results,best_model_result,l)
    
    Choice+=1
    
    
# ###########################################
# #Hypertuning with GridSearchCV
# ###########################################

# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer
    
# t5=time()

# for l in range(0,1):#run_counter):

#     if outcome.name=='complicação pós-cirúrgica':
#         proper_feature_selection=big_boi_compli.iloc[:,Choice].sort_values(ascending=False)
#     elif outcome.name=='óbito até 1 ano após cirurgia':
#         proper_feature_selection=big_boi_obito.iloc[:,Choice].sort_values(ascending=False)
#     elif outcome.name=='classificação clavien-dindo':
#         proper_feature_selection=all_scored_clavien
        
#     fuzzified_inputs_FE=dataframe_append(Default_membership_x,mf,inputs[proper_feature_selection.index[:15].values]) #Fuzzification of the 15 best features for complication
    
    
#     sss=StratifiedShuffleSplit(2,test_size=0.25, random_state=1)
      
#     for train_index, test_index in sss.split(fuzzified_inputs_FE, outcome):             #Cycle to create the stratified training and testing sets
#         X_train_fuzzy_FE, X_test_fuzzy_FE = fuzzified_inputs_FE.iloc[train_index], fuzzified_inputs_FE.iloc[test_index]
#         y_train_fuzzy_FE, y_test_fuzzy_FE = outcome.iloc[train_index], outcome.iloc[test_index]

#     clf=GridSearchCV(DecisionTreeClassifier(),{'criterion':['gini','entropy'],'splitter':['best','random'],
#                                                 'max_depth':range(1,10),'min_samples_split':range(2,30),'min_samples_leaf':range(2,30)}
#                       ,scoring={'Accuracy':'accuracy','ROC AUC':'roc_auc',
#                                                                                   'MCC': make_scorer(matthews_corrcoef, greater_is_better=True,
#                                                                                                     needs_proba=False),
#                                                                                   'Recall':'recall'}, cv=5, verbose=0, refit='Recall', return_train_score=0)
    
#     t_tuning=time()           
                                                                      
#     clf.fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
    
#     Hyper_tuning_results=pd.DataFrame(clf.cv_results_)
    
#     print()
#     print()
#     print("Total tuning time: ",round((time() - t_tuning)/60)," minutes")  
    
#     y_predicted_best_grid_fuzzy_FE=clf.best_estimator_.predict(X_test_fuzzy_FE)   #Obtains the predicted classes
        
#     score_best_Grid_search_fuzzy_FE=scores(y_test_fuzzy_FE,y_predicted_best_grid_fuzzy_FE)   #Calculates the scores  


#     print('Feature selection: ',l)
#     print(clf.best_estimator_)
#     Choice+=1
     

# print("Total elapsed time: ",round((time() - t5)/60)," minutes")     
        


###########################################
#Data visualization for the trained models
###########################################


# plt.figure()

# plt.plot(range(1,result_fuzzy_FE.iloc[:,0].shape[0]+1),result_fuzzy_FE.iloc[:,0],label="Accuracy (%)")
# plt.plot(range(1,result_fuzzy_FE.iloc[:,0].shape[0]+1),result_fuzzy_FE.iloc[:,1],label="F-Score")
# plt.plot(range(1,result_fuzzy_FE.iloc[:,0].shape[0]+1),result_fuzzy_FE.iloc[:,2],label="MCC")
# plt.plot(range(1,result_fuzzy_FE.iloc[:,0].shape[0]+1),result_fuzzy_FE.iloc[:,3],label="ROC AUC")
# plt.plot(range(1,result_fuzzy_FE.iloc[:,0].shape[0]+1),result_fuzzy_FE.iloc[:,4],label="Cohen's kappa")
# plt.plot(range(1,result_fuzzy_FE.iloc[:,0].shape[0]+1),result_fuzzy_FE.iloc[:,5],label="Recall")
# plt.legend()
# plt.title("Diferent metrics across "+str(Number_of_models)+" with fuzzy and reduced features")
# plt.xlabel('Model Number')
# plt.ylabel("Metrics")
# plt.show()


##############################################
#Tree Plot ###################################
##############################################

# from sklearn.tree import plot_tree

# plt.figure(figsize=(72,15),dpi=400)
# plot_tree(best_model[0], max_depth=None, feature_names=pd.Index.tolist(fuzzified_inputs_FE.columns), class_names=['No complication','Complication'], label='all', filled=True, impurity=False, precision=3,fontsize=8)

##############################################
#Adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
#at 05/01/2021 at 10:00 am

# n_nodes = best_model[0].tree_.node_count
# children_left = best_model[0].tree_.children_left
# children_right = best_model[0].tree_.children_right
# feature = best_model[0].tree_.feature
# threshold = best_model[0].tree_.threshold

# node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
# is_leaves = np.zeros(shape=n_nodes, dtype=bool)
# stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
# while len(stack) > 0:
#     # `pop` ensures each node is only visited once
#     node_id, depth = stack.pop()
#     node_depth[node_id] = depth

#     # If the left and right child of a node is not the same we have a split
#     # node
#     is_split_node = children_left[node_id] != children_right[node_id]
#     # If a split node, append left and right children and depth to `stack`
#     # so we can loop through them
#     if is_split_node:
#         stack.append((children_left[node_id], depth + 1))
#         stack.append((children_right[node_id], depth + 1))
#     else:
#         is_leaves[node_id] = True

# print("The binary tree structure has {n} nodes and has "
#       "the following tree structure:\n".format(n=n_nodes))
# for i in range(n_nodes):
#     if is_leaves[i]:
#         print("{space}node={node} is a leaf node.".format(
#             space=node_depth[i] * "\t", node=i))
#     else:
#         print("{space}node={node} is a split node: "
#               "go to node {left} if X[:, {feature}] <= {threshold} "
#               "else to node {right}.".format(
#                   space=node_depth[i] * "\t",
#                   node=i,
#                   left=children_left[i],
#                   feature=feature[i],
#                   threshold=threshold[i],
#                   right=children_right[i]))

###################################################

# plt.figure(figsize=(35,15),dpi=400)
# plot_tree(DT_fuzzy_FE_clf, max_depth=3, feature_names=pd.Index.tolist(fuzzified_inputs_FE.columns), class_names=['No complication','Complication'], label='all', filled=True, impurity=False, precision=3,fontsize=8)