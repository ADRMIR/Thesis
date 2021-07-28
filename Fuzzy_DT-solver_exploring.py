#Fuzzy decision tree exploration code
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


######################################################################
#Pre processing with min-max
indexes_to_keep=(np.abs(stats.zscore(inputs[['idade', 'ACS altura', 'ACS peso']])) < 3).all(axis=1)
inputs=inputs[indexes_to_keep].reset_index()
outputs=outputs[indexes_to_keep].reset_index()

temporary = inputs.values #returns a numpy array
min_max_scaler = pre.MinMaxScaler()
temporary_scaled = min_max_scaler.fit_transform(temporary)
inputs_normal = pd.DataFrame(temporary_scaled,columns=inputs.columns)  #All normalized now
#######################################################################

##############################
#Membership functions creation
##############################

def membership_creation(inputs):
    

    Default_membership_x=np.arange(inputs.shape[0])
    #Standardizing
    Default_membership_x=np.reshape(min_max_scaler.fit_transform(Default_membership_x.reshape(-1,1)),inputs.shape[0])   #Reshape due to function restrictions
    
    #Don´t care membership function
    
    mf_DC=np.empty(inputs.shape[0])
    mf_DC.fill(1)
    
    # plt.figure()
    
    # plt.plot(Default_membership_x, mf_DC, label='Do not Care')
    # plt.legend()
    # plt.ylim(0,1.2)
    # plt.xlim(0,1)
    
    #2 membership functions from [0 to 1]
    
    mf_2_small=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0,0.5),inputs.shape[0])     
    mf_2_large=np.reshape(fuzz.membership.gaussmf(Default_membership_x,1,0.5),inputs.shape[0])     
    
    # plt.figure()
    
    # plt.plot(Default_membership_x,mf_2_small, label='Small')
    # plt.plot(Default_membership_x,mf_2_large, label='Large')
    # plt.legend()
    # plt.ylim(0,1.2)
    # plt.xlim(0,1)
    
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
    
    #4 membership functions from [0 to 1]
    
    mf_4_small=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0,0.15),inputs.shape[0])      #Membership function for ages with a median of 0.5 and sigma of 0.1
    mf_4_medium_small=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0.33,0.15),inputs.shape[0])      #Membership function for ages with a median of 0.5 and sigma of 0.1
    mf_4_medium_large=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0.66,0.15),inputs.shape[0])    #Membership function for ages with a median of 0.5 and sigma of 0.1
    mf_4_large=np.reshape(fuzz.membership.gaussmf(Default_membership_x,1,0.15),inputs.shape[0]) 
    
    # plt.figure()
    
    # plt.plot(Default_membership_x,mf_4_small, label='Small')
    # plt.plot(Default_membership_x,mf_4_medium_small, label='Medium Small')
    # plt.plot(Default_membership_x,mf_4_medium_large, label='Medium Large')
    # plt.plot(Default_membership_x,mf_4_large, label='Large')
    # plt.legend()
    # plt.ylim(0,1.2)
    # plt.xlim(0,1)
    
    #5 membership functions from [0 to 1]
    
    mf_5_small=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0,0.11),inputs.shape[0])     #Membership function for ages with a median of 0.5 and sigma of 0.1
    mf_5_medium_small=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0.25,0.11),inputs.shape[0])      #Membership function for ages with a median of 0.5 and sigma of 0.1
    mf_5_median=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0.5,0.11),inputs.shape[0])  
    mf_5_medium_large=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0.75,0.11),inputs.shape[0])      #Membership function for ages with a median of 0.5 and sigma of 0.1
    mf_5_large=np.reshape(fuzz.membership.gaussmf(Default_membership_x,1,0.11),inputs.shape[0]) 
    
    
    mf=pd.DataFrame(data={'Dont Care':mf_DC,
                          'Small_2':mf_2_small,'Large_2':mf_2_large,
                          'Small_3':mf_3_small,'Median_3':mf_3_median,'Large_3':mf_3_large,
                          'Small_4':mf_4_small,'Medium_small_4':mf_4_medium_small,'Medium_large_4':mf_4_medium_large,'Large_4':mf_4_large,
                          'Small_5':mf_5_small,'Medium_small_5':mf_5_medium_small,'Medium_5':mf_5_median,'Medium_large_5':mf_5_medium_large,'Large_5':mf_5_large})
    
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
        
     
        
fuzzified_inputs=dataframe_append(Default_membership_x,mf,inputs_normal)       #Obtains the fuzzified features


##############################
#Fuzzy Decision tree
##############################

from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from Feature_selection import all_scored_compli, all_scored_obito, all_scored_dias, all_scored_clavien, all_scored_reint


from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

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

########################
#Definition of desired prediction

#['nº dias na UCI', 'destino após UCI', 'complicação pós-cirúrgica',
#       'classificação clavien-dindo',
#       'óbito até 1 ano após cirurgia', ' reinternamento na UCI']    Pick one of these

#outcome=outputs['complicação pós-cirúrgica']
outcome=outputs['óbito até 1 ano após cirurgia']
#outcome=outputs['classificação clavien-dindo']


#all_scored_compli, all_scored_obito, all_scored_dias, all_scored_clavien,
#        all_scored_reint    choose accordingly

if outcome.name=='complicação pós-cirúrgica':
    proper_feature_selection=all_scored_compli
elif outcome.name=='óbito até 1 ano após cirurgia':
    proper_feature_selection=all_scored_obito
elif outcome.name=='classificação clavien-dindo':
    proper_feature_selection=all_scored_clavien

########################



fuzzified_inputs_FE=dataframe_append(Default_membership_x,mf,inputs_normal[proper_feature_selection.index[:15].values]) #Fuzzification of the 15 best features for complication


avg_result=avg_result_fuzzy=avg_result_fuzzy_FE=pd.DataFrame(columns=["Avg Accuracy","Avg F1 Score","Avg MCC","Avg ROC AUC","Avg Cohen's Kappa","Avg Recall","Avg CM (1,1)","Avg CM (1,2)","Avg CM (2,1)","Avg CM (2,2)"])        #Initializes the scores dataframe

best_model_result=pd.DataFrame(columns=["Accuracy","F1 Score","MCC","ROC AUC","Cohen's Kappa","Recall","CM (1,1)","CM (1,2)","CM (2,1)","CM (2,2)"])        #Initializes the scores dataframe
std_results=pd.DataFrame(columns=["Std Accuracy","Std F1 Score","Std MCC","Std ROC AUC","Std Cohen's Kappa","Std Recall","CM (1,1)","CM (1,2)","CM (2,1)","CM (2,2)"])        #Initializes the scores dataframe
 

maximum_range=200

maximum_depth=20


percent=0.5      #Percentage of positive cases to be used in training

training=0.7        #Percentage of data to use in training


# j=0
t1=time()

  
# for j in range(2,maximum_depth):                                      #Explores the network depth
#     print('Training with a depth of ',j,' decision tree layers')
#     print()

j=5

t0 = time()


result=result_fuzzy=result_fuzzy_FE=pd.DataFrame(columns=["Accuracy","F1 Score","MCC","ROC AUC","Cohen's Kappa","Recall","CM (1,1)","CM (1,2)","CM (2,1)","CM (2,2)"])        #Initializes the scores dataframe



for i in range(maximum_range):
    
    clf = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=3, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=9, min_samples_split=13,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='random')
    
    #Best parameters found with GridSearchCV with enphasis on Recall for complications
    # ccp_alpha=0.0, class_weight=None, criterion='entropy',
    #                    max_depth=6, max_features=None, max_leaf_nodes=None,
    #                    min_impurity_decrease=0.0, min_impurity_split=None,
    #                    min_samples_leaf=19, min_samples_split=12,
    #                    min_weight_fraction_leaf=0.0, presort='deprecated',
    #                    random_state=None, splitter='random'
    
    #Best Parameters found with GridsearchCV with emphasis on Recall for mortality
    # ccp_alpha=0.0, class_weight=None, criterion='entropy',
    #                     max_depth=9, max_features=None, max_leaf_nodes=None,
    #                     min_impurity_decrease=0.0, min_impurity_split=None,
    #                     min_samples_leaf=5, min_samples_split=13,
    #                     min_weight_fraction_leaf=0.0, presort='deprecated',
    #                     random_state=None, splitter='random'
    
    #Best Parameters found with GridsearchCV with emphasis on Recall for complications and 50% data balancing
    # ccp_alpha=0.0, class_weight=None, criterion='gini',
    #                    max_depth=5, max_features=None, max_leaf_nodes=None,
    #                    min_impurity_decrease=0.0, min_impurity_split=None,
    #                    min_samples_leaf=6, min_samples_split=24,
    #                    min_weight_fraction_leaf=0.0, presort='deprecated',
    #                    random_state=None, splitter='random'
    
    #Best Parameters found with GridsearchCV with emphasis on MCC for complications with pearson feature selection
    # ccp_alpha=0.0, class_weight=None, criterion='gini',
    #                   max_depth=7, max_features=None, max_leaf_nodes=None,
    #                   min_impurity_decrease=0.0, min_impurity_split=None,
    #                   min_samples_leaf=3, min_samples_split=13,
    #                   min_weight_fraction_leaf=0.0, presort='deprecated',
    #                   random_state=None, splitter='random'
    
    #Best Parameters found with GridsearchCV with emphasis on MCC for mortality with pearson feature selection
    # ccp_alpha=0.0, class_weight=None, criterion='entropy',
    #                   max_depth=3, max_features=None, max_leaf_nodes=None,
    #                   min_impurity_decrease=0.0, min_impurity_split=None,
    #                   min_samples_leaf=9, min_samples_split=13,
    #                   min_weight_fraction_leaf=0.0, presort='deprecated',
    #                   random_state=None, splitter='random'
    
    # ################################################################
    # #Decision tree with original inputs
    # ################################################################

    # X_train_no_fuzzy, X_test_no_fuzzy, y_train_no_fuzzy, y_test_no_fuzzy = train_test_split(inputs_normal, outcome, test_size=0.25, random_state=i)

    # X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(inputs_normal.iloc[outcome_1.index.values,:], outcome_1, test_size=1-percentage,random_state=i)   
    
    # X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(inputs_normal.iloc[outcome_0.index.values,:], outcome_0, test_size=(outcome_1.shape[0]*(1-percentage))/outcome_0.shape[0],random_state=i)   
    
    # X_train_no_fuzzy= pd.concat([X_train_1,X_train_0],ignore_index=True)
    # X_test_no_fuzzy= pd.concat([X_test_1,X_test_0],ignore_index=True) 
    # y_train_no_fuzzy= pd.concat([y_train_1,y_train_0],ignore_index=True)
    # y_test_no_fuzzy= pd.concat([y_test_1,y_test_0],ignore_index=True)    

    # DT_clf=clf.fit(X_train_no_fuzzy,y_train_no_fuzzy)               #Trains the classifier
    
    # y_predicted_no_fuzzy=DT_clf.predict(X_test_no_fuzzy)            #Obtains the predicted classes
    
    # score_no_fuzzy=scores(y_test_no_fuzzy,y_predicted_no_fuzzy)   #Calculates the scores

    # result=result.append(score_no_fuzzy)                            #Stacks the scores in a result variable

    # ################################################################
    # #FDT with fuzzy inputs 
    # ################################################################
    
    # X_train_fuzzy, X_test_fuzzy, y_train_fuzzy, y_test_fuzzy = train_test_split(fuzzified_inputs, outcome, test_size=0.25, random_state=i)
          
    
    # DT_fuzzy_clf=clf.fit(X_train_fuzzy,y_train_fuzzy)                           #Trains the classifier
    
    # y_predicted_fuzzy=DT_fuzzy_clf.predict(X_test_fuzzy)                        #Obtains the predicted classes
    
    # score=scores(y_test_fuzzy,y_predicted_fuzzy)                              #Calculates the scores

    # result_fuzzy=result_fuzzy.append(score)                         #Stacks the scores in a result variable

    ##################################################################
    ##Model run with feature selection according to previous selection
    ##################################################################  

    #X_train_fuzzy_FE, X_test_fuzzy_FE, y_train_fuzzy_FE, y_test_fuzzy_FE = train_test_split(fuzzified_inputs_FE, outcome, test_size=0.25, random_state=i)
    
    sss=StratifiedShuffleSplit(2,test_size=0.25, random_state=i)
  
    for train_index, test_index in sss.split(fuzzified_inputs_FE, outcome):             #Cycle to create the stratified training and testing sets
        X_train_fuzzy_FE, X_test_fuzzy_FE = fuzzified_inputs_FE.iloc[train_index], fuzzified_inputs_FE.iloc[test_index]
        y_train_fuzzy_FE, y_test_fuzzy_FE = outcome.iloc[train_index], outcome.iloc[test_index]
          
    
    DT_fuzzy_FE_clf=clf.fit(X_train_fuzzy_FE,y_train_fuzzy_FE)      #Trains the classifier                      
    
    y_predicted_fuzzy_FE=DT_fuzzy_FE_clf.predict(X_test_fuzzy_FE)   #Obtains the predicted classes
    
    score_fuzzy_FE=scores(y_test_fuzzy_FE,y_predicted_fuzzy_FE)   #Calculates the scores
    
    result_fuzzy_FE=result_fuzzy_FE.append(score_fuzzy_FE)     #Stacks the scores in a result variable
    
    
    if i==round(maximum_range/4):
        print()
        print()
        print()
        print('25% done')
    elif i==2*round(maximum_range/4):
        print()
        print()
        print()
        print('50% done')
    elif i==3*round(maximum_range/4):
        print()
        print()
        print()
        print('75% done')
    elif i==4*round(maximum_range/4):
        print()
        print()
        print()
        print('DONE')

print()
print( round((time() - t0)/60), " minutes to train ", maximum_range,"  models with ", j," layers")
print()

# avg_result_to_pass=pd.DataFrame([np.hstack((pd.DataFrame.mean(result).values,np.std(result.iloc[:,0].values,axis=0)))],columns=["Avg Accuracy","Avg F1 Score","Avg MCC","Avg ROC AUC","Avg Cohen's Kappa","Avg Recall","Avg CM (1,1)","Avg CM (1,2)","Avg CM (2,1)","Avg CM (2,2)","Accuracy Std"])       #Calculates mean values of the scores, and the std for accuracy
# avg_result=avg_result.append(avg_result_to_pass)

# avg_result_fuzzy_to_pass=pd.DataFrame([np.hstack((pd.DataFrame.mean(result_fuzzy).values,np.std(result_fuzzy.iloc[:,0].values,axis=0)))],columns=["Avg Accuracy","Avg F1 Score","Avg MCC","Avg ROC AUC","Avg Cohen's Kappa","Avg Recall","Avg CM (1,1)","Avg CM (1,2)","Avg CM (2,1)","Avg CM (2,2)","Accuracy Std"])
# avg_result_fuzzy=avg_result_fuzzy.append(avg_result_fuzzy_to_pass)

avg_result_fuzzy_FE_to_pass=pd.DataFrame([pd.DataFrame.mean(result_fuzzy_FE).values],columns=["Avg Accuracy","Avg F1 Score","Avg MCC","Avg ROC AUC","Avg Cohen's Kappa","Avg Recall","Avg CM (1,1)","Avg CM (1,2)","Avg CM (2,1)","Avg CM (2,2)"])
avg_result_fuzzy_FE=avg_result_fuzzy_FE.append(avg_result_fuzzy_FE_to_pass)

best_model_result=best_model_result.append(result_fuzzy_FE.iloc[np.argmax(result_fuzzy_FE.iloc[:,2]),:])

std_results=std_results.append(pd.DataFrame([np.std(result_fuzzy_FE).values],columns=["Std Accuracy","Std F1 Score","Std MCC","Std ROC AUC","Std Cohen's Kappa","Std Recall","CM (1,1)","CM (1,2)","CM (2,1)","CM (2,2)"]))        #Initializes the scores dataframe


print()
print()
print("Total elapsed time: ",round((time() - t1)/60)," minutes")  




###########################################
#Hypertuning with GridSearchCV
###########################################

# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer

# #X_train_fuzzy_FE, X_test_fuzzy_FE, y_train_fuzzy_FE, y_test_fuzzy_FE


# # accu=accuracy_score(Y_Test,Y_Pred)
# # f1=f1_score(Y_Test,Y_Pred)
# # mathew=matthews_corrcoef(Y_Test,Y_Pred)
# # roc=roc_auc_score(Y_Test,Y_Pred)
# # kappa=cohen_kappa_score(Y_Test,Y_Pred)
# # recall=recall_score(Y_Test,Y_Pred)
# # confusion=confusion_matrix(Y_Test,Y_Pred).flatten()

# clf=GridSearchCV(DecisionTreeClassifier(),{'criterion':['gini','entropy'],'splitter':['best','random'],
#                                             'max_depth':range(1,10),'min_samples_split':range(2,30),'min_samples_leaf':range(2,30)}
#                   ,scoring={'Accuracy':'accuracy','ROC AUC':'roc_auc',
#                                                                               'MCC': make_scorer(matthews_corrcoef, greater_is_better=True,
#                                                                                                 needs_proba=False),
#                                                                               'Recall':'recall'}, cv=5, verbose=2, refit='MCC', return_train_score=0)

# t_tuning=time()           
                                                                  
# clf.fit(X_train_fuzzy_FE,y_train_fuzzy_FE)

# Hyper_tuning_results=pd.DataFrame(clf.cv_results_)

# print()
# print()
# print("Total tuning time: ",round((time() - t_tuning)/60)," minutes")  

# y_predicted_best_grid_fuzzy_FE=clf.best_estimator_.predict(X_test_fuzzy_FE)   #Obtains the predicted classes
    
# score_best_Grid_search_fuzzy_FE=scores(y_test_fuzzy_FE,y_predicted_best_grid_fuzzy_FE)   #Calculates the scores

###########################################
#Data visualization for the trained models
###########################################

# plt.figure()

# plt.plot(range(1,avg_result.iloc[:,0].shape[0]+1),avg_result.iloc[:,0],label="Accuracy (%)")
# plt.plot(range(1,avg_result.iloc[:,0].shape[0]+1),avg_result.iloc[:,1],label="F-Score")
# plt.plot(range(1,avg_result.iloc[:,0].shape[0]+1),avg_result.iloc[:,2],label="MCC")
# plt.plot(range(1,avg_result.iloc[:,0].shape[0]+1),avg_result.iloc[:,3],label="ROC AUC")
# plt.plot(range(1,avg_result.iloc[:,0].shape[0]+1),avg_result.iloc[:,4],label="Cohen's kappa")
# plt.plot(range(1,avg_result.iloc[:,0].shape[0]+1),avg_result.iloc[:,5],label="Recall")
# plt.legend()
# plt.title("Evolution of metrics with increase in depth of the tree, with initial features")
# plt.xlabel('Number of layers')
# plt.ylabel("Metrics")
# plt.show()

# plt.figure()

# plt.plot(range(1,avg_result_fuzzy.iloc[:,0].shape[0]+1),avg_result_fuzzy.iloc[:,0],label="Accuracy (%)")
# plt.plot(range(1,avg_result_fuzzy.iloc[:,0].shape[0]+1),avg_result_fuzzy.iloc[:,1],label="F-Score")
# plt.plot(range(1,avg_result_fuzzy.iloc[:,0].shape[0]+1),avg_result_fuzzy.iloc[:,2],label="MCC")
# plt.plot(range(1,avg_result_fuzzy.iloc[:,0].shape[0]+1),avg_result_fuzzy.iloc[:,3],label="ROC AUC")
# plt.plot(range(1,avg_result_fuzzy.iloc[:,0].shape[0]+1),avg_result_fuzzy.iloc[:,4],label="Cohen's kappa")
# plt.legend()
# plt.title("Evolution of metrics with increase in depth of the tree, with fuzzy features")
# plt.xlabel('Number of layers')
# plt.ylabel("Metrics")
# plt.show()

plt.figure()

plt.plot(range(1,result_fuzzy_FE.iloc[:,0].shape[0]+1),result_fuzzy_FE.iloc[:,0],label="Accuracy (%)")
plt.plot(range(1,result_fuzzy_FE.iloc[:,0].shape[0]+1),result_fuzzy_FE.iloc[:,1],label="F-Score")
plt.plot(range(1,result_fuzzy_FE.iloc[:,0].shape[0]+1),result_fuzzy_FE.iloc[:,2],label="MCC")
plt.plot(range(1,result_fuzzy_FE.iloc[:,0].shape[0]+1),result_fuzzy_FE.iloc[:,3],label="ROC AUC")
plt.plot(range(1,result_fuzzy_FE.iloc[:,0].shape[0]+1),result_fuzzy_FE.iloc[:,4],label="Cohen's kappa")
plt.plot(range(1,result_fuzzy_FE.iloc[:,0].shape[0]+1),result_fuzzy_FE.iloc[:,5],label="Recall")
plt.legend()
plt.title("Diferent metrics across "+str(maximum_range)+" with fuzzy and reduced features")
plt.xlabel('Model Number')
plt.ylabel("Metrics")
plt.show()

# plt.figure()

# plt.plot(range(1,result.iloc[:,0].shape[0]+1),result.iloc[:,0],label="Raw data")
# plt.plot(range(1,result_fuzzy.iloc[:,0].shape[0]+1),result_fuzzy.iloc[:,0],label="Fuzzy data")
# plt.plot(range(1,result_fuzzy_FE.iloc[:,0].shape[0]+1),result_fuzzy_FE.iloc[:,0],label="Fuzzy with FE data")
# plt.legend()
# plt.title("Accuracy comparison between the three aproaches")
# plt.xlabel('Number of layers')
# plt.ylabel("Accuracy")
# plt.show()


##############################################
#Data visualization for the average properties
##############################################

# plt.figure()

# plt.plot(range(1,avg_result.iloc[:,0].shape[0]+1),avg_result.iloc[:,0],label="Accuracy (%)")
# plt.plot(range(1,avg_result.iloc[:,0].shape[0]+1),avg_result.iloc[:,1],label="F-Score")
# plt.plot(range(1,avg_result.iloc[:,0].shape[0]+1),avg_result.iloc[:,2],label="MCC")
# plt.plot(range(1,avg_result.iloc[:,0].shape[0]+1),avg_result.iloc[:,3],label="ROC AUC")
# plt.plot(range(1,avg_result.iloc[:,0].shape[0]+1),avg_result.iloc[:,4],label="Cohen's kappa")
# plt.plot(range(1,avg_result.iloc[:,0].shape[0]+1),avg_result.iloc[:,5],label="Recall")
# plt.legend()
# plt.title("Evolution of metrics with increase in depth of the tree, with initial features")
# plt.xlabel('Number of layers')
# plt.ylabel("Metrics")
# plt.show()

# plt.figure()

# plt.plot(range(1,avg_result_fuzzy.iloc[:,0].shape[0]+1),avg_result_fuzzy.iloc[:,0],label="Accuracy (%)")
# plt.plot(range(1,avg_result_fuzzy.iloc[:,0].shape[0]+1),avg_result_fuzzy.iloc[:,1],label="F-Score")
# plt.plot(range(1,avg_result_fuzzy.iloc[:,0].shape[0]+1),avg_result_fuzzy.iloc[:,2],label="MCC")
# plt.plot(range(1,avg_result_fuzzy.iloc[:,0].shape[0]+1),avg_result_fuzzy.iloc[:,3],label="ROC AUC")
# plt.plot(range(1,avg_result_fuzzy.iloc[:,0].shape[0]+1),avg_result_fuzzy.iloc[:,4],label="Cohen's kappa")
# plt.legend()
# plt.title("Evolution of metrics with increase in depth of the tree, with fuzzy features")
# plt.xlabel('Number of layers')
# plt.ylabel("Metrics")
# plt.show()

plt.figure()

plt.plot(range(1,avg_result_fuzzy_FE.iloc[:,0].shape[0]+1),avg_result_fuzzy_FE.iloc[:,0],label="Accuracy (%)")
plt.plot(range(1,avg_result_fuzzy_FE.iloc[:,0].shape[0]+1),avg_result_fuzzy_FE.iloc[:,1],label="F-Score")
plt.plot(range(1,avg_result_fuzzy_FE.iloc[:,0].shape[0]+1),avg_result_fuzzy_FE.iloc[:,2],label="MCC")
plt.plot(range(1,avg_result_fuzzy_FE.iloc[:,0].shape[0]+1),avg_result_fuzzy_FE.iloc[:,3],label="ROC AUC")
plt.plot(range(1,avg_result_fuzzy_FE.iloc[:,0].shape[0]+1),avg_result_fuzzy_FE.iloc[:,4],label="Cohen's kappa")
plt.plot(range(1,avg_result_fuzzy_FE.iloc[:,0].shape[0]+1),avg_result_fuzzy_FE.iloc[:,5],label="Recall")
plt.legend()
plt.title("Evolution of metrics with increase in depth of the tree, with fuzzy and reduced features")
plt.xlabel('Number of layers')
plt.ylabel("Metrics")
plt.show()

# plt.figure()

# plt.plot(range(1,avg_result.iloc[:,0].shape[0]+1),avg_result.iloc[:,0],label="Raw data")
# plt.plot(range(1,avg_result_fuzzy.iloc[:,0].shape[0]+1),avg_result_fuzzy.iloc[:,0],label="Fuzzy data")
# plt.plot(range(1,avg_result_fuzzy_FE.iloc[:,0].shape[0]+1),avg_result_fuzzy_FE.iloc[:,0],label="Fuzzy with FE data")
# plt.legend()
# plt.title("Accuracy comparison between the three aproaches")
# plt.xlabel('Number of layers')
# plt.ylabel("Average Accuracy")
# plt.show()




##############################################
#Tree Plot ###################################
##############################################

from sklearn.tree import plot_tree

# plt.figure(figsize=(40,30),dpi=200)
# plot_tree(DT_clf, max_depth=None, feature_names=pd.Index.tolist(inputs_normal.columns),class_names=['No complication','Complication'], label='all', filled=True, impurity=False, precision=3,fontsize=12)

# plt.figure(figsize=(40,30),dpi=200)
# plot_tree(DT_fuzzy_clf, max_depth=None, feature_names=pd.Index.tolist(fuzzified_inputs.columns),class_names=['No complication','Complication'], label='all', filled=True, impurity=False, precision=3,fontsize=12)

# plt.figure(figsize=(72,15),dpi=400)
# plot_tree(DT_fuzzy_FE_clf, max_depth=None, feature_names=pd.Index.tolist(fuzzified_inputs_FE.columns), class_names=['No complication','Complication'], label='all', filled=True, impurity=False, precision=3,fontsize=8)

# plt.figure(figsize=(35,15),dpi=400)
# plot_tree(DT_fuzzy_FE_clf, max_depth=3, feature_names=pd.Index.tolist(fuzzified_inputs_FE.columns), class_names=['No complication','Complication'], label='all', filled=True, impurity=False, precision=3,fontsize=8)