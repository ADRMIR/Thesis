#Fuzzy code
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

#sys.path.append(r"C:\Users\vm\Desktop\Andre\Thesis\Python\Feature_selection")   #Goes to the other directory and gets the desired files

sys.path.append(r"C:\Documents\MEMec_2015-2020\Tese\Python\Feature_selection")  #My PC


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
        
     
        
fuzzified_inputs=dataframe_append(Default_membership_x,mf,inputs)       #Obtains the fuzzified features

##############################
#ANN with fuzzified inputs
##############################

from sklearn.neural_network import MLPClassifier as MLPCla
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


# print('Beggining model training')
# print()


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

Choice=7

if Choice!=7:
    run_counter=1    #variable to choose the number of runs 
else:
    run_counter=7
    Choice=0

Number_of_features=10

#['nº dias na UCI', 'destino após UCI', 'complicação pós-cirúrgica',
#       'classificação clavien-dindo',
#       'óbito até 1 ano após cirurgia', ' reinternamento na UCI']    Pick one of these

#
#outcome=outputs['complicação pós-cirúrgica']
outcome=outputs['óbito até 1 ano após cirurgia']

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


t4=time()

for l in range(run_counter):
    
    #all_scored_compli, all_scored_obito, all_scored_dias, all_scored_clavien,
    #        all_scored_reint    choose accordingly
    
    if outcome.name=='complicação pós-cirúrgica':
        proper_feature_selection=big_boi_compli.iloc[:,Choice].sort_values(ascending=False)
    elif outcome.name=='óbito até 1 ano após cirurgia':
        proper_feature_selection=big_boi_obito.iloc[:,Choice].sort_values(ascending=False)
    elif outcome.name=='classificação clavien-dindo':
        proper_feature_selection=all_scored_clavien
    
    
    
    ########################################################################
    ############################ Model Training ############################   
    ########################################################################  
    
    
    fuzzified_inputs_FE=dataframe_append(Default_membership_x,mf,inputs[proper_feature_selection.index[:Number_of_features].values]) #Fuzzification of the Number_of_features best features for complication
    
    
    X_test_fuzzy_FE=dataframe_append(Default_membership_x2,mf2,test_inputs[proper_feature_selection.index[:Number_of_features].values]) #Fuzzification of the Number_of_features best features for complication
  
    
    avg_result=avg_result_fuzzy=avg_result_fuzzy_FE=pd.DataFrame(columns=["Avg Accuracy","Avg F1 Score","Avg MCC","Avg ROC AUC","Avg Cohen's Kappa","Avg Recall","Avg CM (1,1)","Avg CM (1,2)","Avg CM (2,1)","Avg CM (2,2)"])        #Initializes the scores dataframe
    
    best_model_result=pd.DataFrame(columns=["Accuracy","F1 Score","MCC","ROC AUC","Cohen's Kappa","Recall","CM (1,1)","CM (1,2)","CM (2,1)","CM (2,2)"])        #Initializes the scores dataframe
    std_results=pd.DataFrame(columns=["Std Accuracy","Std F1 Score","Std MCC","Std ROC AUC","Std Cohen's Kappa","Std Recall","CM (1,1)","CM (1,2)","CM (2,1)","CM (2,2)"])        #Initializes the scores dataframe
     
    
    
    
    Number_of_models=50
    
    maximum_number_of_neurons=1
    
    percent=0.5      #Percentage of positive cases to be used in training
    
    training=0.3        #Percentage of data to use in training
    
    
    # j=0
    k=1             #A layer number of neurons
    t1=time()
    
    
        
    # for j in range(maximum_number_of_neurons):                                      #Explores the network depth
    #     print('Training with ',j+1,' neurons')
    #     print()
    
    #     t0 = time()
    
    j=2
    
    result=result_fuzzy=result_fuzzy_FE=pd.DataFrame(columns=["Accuracy","F1 Score","MCC","ROC AUC","Cohen's Kappa","Recall","CM (1,1)","CM (1,2)","CM (2,1)","CM (2,2)"])        #Initializes the scores dataframe
    
    
    
    for i in range(Number_of_models):
        
        # ##############################
        # #ANN with normal inputs
        # ##############################
        
        # X_train_no_fuzzy, X_test_no_fuzzy, y_train_no_fuzzy, y_test_no_fuzzy = train_test_split(inputs, outcome, test_size=0.25, random_state=i)
        
        # hidden=round(math.sqrt(inputs.shape[1]))
        
        # clf_no_fuzzy=MLPCla(hidden_layer_sizes=(hidden,j+1),activation='logistic', solver='lbfgs',alpha=1e-5,max_iter=5000,early_stopping=True,verbose=0).fit(X_train_no_fuzzy,y_train_no_fuzzy)
        
        # y_predicted_no_fuzzy=clf_no_fuzzy.predict(X_test_no_fuzzy)
        
        # score_no_fuzzy=scores(y_test_no_fuzzy,y_predicted_no_fuzzy)   #Calculates the scores
    
        # result=result.append(score_no_fuzzy)    
            
        # ##############################
        # #ANN with fuzzy inputs
        # ##############################
    
        # X_train, X_test, y_train, y_test = train_test_split(fuzzified_inputs, outcome, test_size=0.25, random_state=i)
         
        # hidden_fuzzy=round(math.sqrt(fuzzified_inputs.shape[1]))
        # clf=MLPCla(hidden_layer_sizes=(hidden_fuzzy,j+1),activation='logistic', solver='lbfgs',alpha=1e-5,max_iter=5000,early_stopping=True,verbose=0).fit(X_train,y_train)
        
        # y_predicted_fuzzy=clf.predict(X_test)
        
        # score=scores(y_test_fuzzy,y_predicted_fuzzy)                              #Calculates the scores
    
        # result_fuzzy=result_fuzzy.append(score)   
      
    
        ##################################################################
        ##Model run with feature selection according to previous selection
        ##################################################################  
    
        #X_train_fuzzy_FE, X_test_fuzzy_FE, y_train_fuzzy_FE, y_test_fuzzy_FE = train_test_split(fuzzified_inputs_FE, outcome, test_size=0.25,random_state=i)
        
        X_train_fuzzy_FE=fuzzified_inputs_FE
        y_train_fuzzy_FE=outcome
        
        # sss=StratifiedShuffleSplit(1,test_size=0.25, random_state=i)
          
        # for train_index, test_index in sss.split(fuzzified_inputs_FE, outcome):             #Cycle to create the stratified training and testing sets
        #     X_train_fuzzy_FE, X_test_fuzzy_FE = fuzzified_inputs_FE.iloc[train_index], fuzzified_inputs_FE.iloc[test_index]
        #     y_train_fuzzy_FE, y_test_fuzzy_FE = outcome.iloc[train_index], outcome.iloc[test_index]
    
        
        
        #hidden=round(math.sqrt(fuzzified_inputs_FE.shape[1]))
        #hidden=(j+1)
        
        #Default
        #clf_fuzzy_with_FE=MLPCla().fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
        
        #Model creation and parameters exploration
        
        #clf_fuzzy_with_FE=MLPCla(hidden_layer_sizes=(hidden),activation='logistic', solver='lbfgs',alpha=1e-5,max_iter=5000,early_stopping=True,verbose=0).fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
        
        #Gridsearch results for 2 layers up to 30 neurons each for complication with emphasis on Recall
        
        #clf_fuzzy_with_FE=MLPCla(activation='logistic', alpha=0.0001, early_stopping=True,
        #           hidden_layer_sizes=(11, 10), max_iter=5000, solver='lbfgs',verbose=1).fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
        
        #Gridsearch results for 2 layers up to 30 neurons each for mortality with emphasis on Recall
        
        #clf_fuzzy_with_FE=MLPCla(activation='logistic', alpha=0.0001, early_stopping=True,
        #           hidden_layer_sizes=(3, 18), max_iter=5000, solver='lbfgs',verbose=0).fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
        
        #Gridsearch results for 2 layers up to 30 neurons each for complication with emphasis on MCC and pearson selection
        
        #clf_fuzzy_with_FE=MLPCla(activation='logistic', alpha=0.0001, batch_size='auto',
        #           beta_1=0.9, beta_2=0.999, early_stopping=True, epsilon=1e-08,
        #           hidden_layer_sizes=(2, 3), learning_rate='constant',
        #           learning_rate_init=0.001, max_fun=15000, max_iter=5000,
        #           momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
        #           power_t=0.5, random_state=None, shuffle=True, solver='lbfgs',
        #           tol=0.0001, validation_fraction=0.1, verbose=False,
        #           warm_start=False).fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
        
        #Gridsearch results for 2 layers up to 30 neurons each for mortality with emphasis on MCC and pearson selection
        
        #clf_fuzzy_with_FE=MLPCla(activation='logistic', alpha=0.1, batch_size='auto', beta_1=0.9,
        #           beta_2=0.999, early_stopping=True, epsilon=1e-08,
        #           hidden_layer_sizes=(2, 14), learning_rate='constant',
        #           learning_rate_init=0.001, max_fun=15000, max_iter=5000,
        #           momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
        #           power_t=0.5, random_state=None, shuffle=True, solver='lbfgs',
        #           tol=0.0001, validation_fraction=0.1, verbose=False,
        #           warm_start=False).fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
    


        ###########################################################################################################
        #Gridsearch results for 1 hidden layer up to 100 neurons for complications with emphasis on recall 
        ###########################################################################################################
    
        if outcome.name=='complicação pós-cirúrgica':    

            if l==0:
            #Pearson
    
                clf_fuzzy_with_FE=MLPCla().fit(X_train_fuzzy_FE,y_train_fuzzy_FE)           
            
            elif l==1:
            #Pearson
    
                clf_fuzzy_with_FE=MLPCla(activation='logistic', alpha=0.0001, batch_size='auto',
                                beta_1=0.9, beta_2=0.999, early_stopping=True, epsilon=1e-08,
                              hidden_layer_sizes=39, learning_rate='constant',
                              learning_rate_init=0.001, max_fun=15000, max_iter=5000,
                              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                              power_t=0.5, random_state=None, shuffle=True, solver='lbfgs',
                              tol=0.0001, validation_fraction=0.1, verbose=False,
                              warm_start=False).fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
        
            elif l==2:        
            #Spearman
    
                clf_fuzzy_with_FE=MLPCla(activation='logistic', alpha=0.0001, batch_size='auto',
                                beta_1=0.9, beta_2=0.999, early_stopping=True, epsilon=1e-08,
                                hidden_layer_sizes=100, learning_rate='constant',
                                learning_rate_init=0.001, max_fun=15000, max_iter=5000,
                                momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                                power_t=0.5, random_state=None, shuffle=True, solver='lbfgs',
                                tol=0.0001, validation_fraction=0.1, verbose=False,
                                warm_start=False).fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
        
            elif l==3:     
            #chi2
    
                clf_fuzzy_with_FE=MLPCla(activation='logistic', alpha=0.0001, batch_size='auto',
                              beta_1=0.9, beta_2=0.999, early_stopping=True, epsilon=1e-08,
                              hidden_layer_sizes=22, learning_rate='constant',
                              learning_rate_init=0.001, max_fun=15000, max_iter=5000,
                              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                              power_t=0.5, random_state=None, shuffle=True, solver='lbfgs',
                              tol=0.0001, validation_fraction=0.1, verbose=False,
                              warm_start=False).fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
        
            elif l==4: 
            #Lasso
    
                clf_fuzzy_with_FE=MLPCla(activation='logistic', alpha=0.0001, batch_size='auto',
                              beta_1=0.9, beta_2=0.999, early_stopping=True, epsilon=1e-08,
                              hidden_layer_sizes=29, learning_rate='constant',
                              learning_rate_init=0.001, max_fun=15000, max_iter=5000,
                              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                              power_t=0.5, random_state=None, shuffle=True, solver='lbfgs',
                              tol=0.0001, validation_fraction=0.1, verbose=False,
                              warm_start=False).fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
        
            elif l==5:         
            #Mutual information
    
                clf_fuzzy_with_FE=MLPCla(activation='logistic', alpha=0.0001, batch_size='auto',
                              beta_1=0.9, beta_2=0.999, early_stopping=True, epsilon=1e-08,
                              hidden_layer_sizes=22, learning_rate='constant',
                              learning_rate_init=0.001, max_fun=15000, max_iter=5000,
                              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                              power_t=0.5, random_state=None, shuffle=True, solver='lbfgs',
                              tol=0.0001, validation_fraction=0.1, verbose=False,
                              warm_start=False).fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
        
            elif l==6: 
            #weighted (all methods) method
    
                clf_fuzzy_with_FE=MLPCla(activation='logistic', alpha=0.0001, batch_size='auto',
                              beta_1=0.9, beta_2=0.999, early_stopping=True, epsilon=1e-08,
                              hidden_layer_sizes=8, learning_rate='constant',
                              learning_rate_init=0.001, max_fun=15000, max_iter=5000,
                              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                              power_t=0.5, random_state=None, shuffle=True, solver='lbfgs',
                              tol=0.0001, validation_fraction=0.1, verbose=False,
                              warm_start=False).fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
        


        ###########################################################################################################
        #Gridsearch results for 1 hidden layer up to 100 neurons for mortality with emphasis on recall
        ###########################################################################################################
    
        elif outcome.name=='óbito até 1 ano após cirurgia':   
    
            if l==0:
            #Pearson
    
                clf_fuzzy_with_FE=MLPCla().fit(X_train_fuzzy_FE,y_train_fuzzy_FE)           
            
            elif l==1:
            #Pearson
    
                clf_fuzzy_with_FE=MLPCla(activation='logistic', alpha=0.0001, batch_size='auto',
                              beta_1=0.9, beta_2=0.999, early_stopping=True, epsilon=1e-08,
                              hidden_layer_sizes=80, learning_rate='constant',
                              learning_rate_init=0.001, max_fun=15000, max_iter=5000,
                              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                              power_t=0.5, random_state=None, shuffle=True, solver='lbfgs',
                              tol=0.0001, validation_fraction=0.1, verbose=False,
                              warm_start=False).fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
    
    
            elif l==2:         
            #Spearman
        
                clf_fuzzy_with_FE=MLPCla(activation='logistic', alpha=0.0001, batch_size='auto',
                            beta_1=0.9, beta_2=0.999, early_stopping=True, epsilon=1e-08,
                            hidden_layer_sizes=49, learning_rate='constant',
                            learning_rate_init=0.001, max_fun=15000, max_iter=5000,
                            momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                            power_t=0.5, random_state=None, shuffle=True, solver='lbfgs',
                            tol=0.0001, validation_fraction=0.1, verbose=False,
                            warm_start=False).fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
    
            elif l==3:     
            #chi2
    
                clf_fuzzy_with_FE=MLPCla(activation='logistic', alpha=0.0001, batch_size='auto',
                            beta_1=0.9, beta_2=0.999, early_stopping=True, epsilon=1e-08,
                            hidden_layer_sizes=64, learning_rate='constant',
                            learning_rate_init=0.001, max_fun=15000, max_iter=5000,
                            momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                            power_t=0.5, random_state=None, shuffle=True, solver='lbfgs',
                            tol=0.0001, validation_fraction=0.1, verbose=False,
                            warm_start=False).fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
    
            elif l==4: 
            #Lasso
    
                clf_fuzzy_with_FE=MLPCla(activation='logistic', alpha=0.0001, batch_size='auto',
                            beta_1=0.9, beta_2=0.999, early_stopping=True, epsilon=1e-08,
                            hidden_layer_sizes=50, learning_rate='constant',
                            learning_rate_init=0.001, max_fun=15000, max_iter=5000,
                            momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                            power_t=0.5, random_state=None, shuffle=True, solver='lbfgs',
                            tol=0.0001, validation_fraction=0.1, verbose=False,
                            warm_start=False).fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
    
            elif l==5:         
            #Mutual information
    
                clf_fuzzy_with_FE=MLPCla(activation='logistic', alpha=0.0001, batch_size='auto',
                            beta_1=0.9, beta_2=0.999, early_stopping=True, epsilon=1e-08,
                            hidden_layer_sizes=12, learning_rate='constant',
                            learning_rate_init=0.001, max_fun=15000, max_iter=5000,
                            momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                            power_t=0.5, random_state=None, shuffle=True, solver='lbfgs',
                            tol=0.0001, validation_fraction=0.1, verbose=False,
                            warm_start=False).fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
    
            elif l==6: 
            #weighted (all methods) method

                clf_fuzzy_with_FE=MLPCla(activation='logistic', alpha=0.0001, batch_size='auto',
                            beta_1=0.9, beta_2=0.999, early_stopping=True, epsilon=1e-08,
                            hidden_layer_sizes=82, learning_rate='constant',
                            learning_rate_init=0.001, max_fun=15000, max_iter=5000,
                            momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                            power_t=0.5, random_state=None, shuffle=True, solver='lbfgs',
                            tol=0.0001, validation_fraction=0.1, verbose=False,
                            warm_start=False).fit(X_train_fuzzy_FE,y_train_fuzzy_FE)



        y_predicted_fuzzy_FE=clf_fuzzy_with_FE.predict(X_test_fuzzy_FE)
        
        score_fuzzy_FE=scores(y_test_fuzzy_FE,y_predicted_fuzzy_FE)   #Calculates the scores
        
        result_fuzzy_FE=result_fuzzy_FE.append(score_fuzzy_FE)     #Stacks the scores in a result variable
        
        
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
    
    # print()
    # # print( round((time() - t0)/60), " minutes to train ", Number_of_models,"  models with ", j+1, "hidden layers")
    # print( round((time() - t0)/60), " minutes to train ", Number_of_models,"  models with (", j+1,",",k+1, ") Neurons")
    # print()
    
    
    # avg_result_to_pass=pd.DataFrame([np.hstack((pd.DataFrame.mean(result).values,np.std(result.iloc[:,0].values,axis=0)))],columns=["Avg Accuracy","Avg F1 Score","Avg MCC","Avg ROC AUC","Avg Cohen's Kappa","Avg Recall","Avg CM (1,1)","Avg CM (1,2)","Avg CM (2,1)","Avg CM (2,2)","Accuracy Std"])       #Calculates mean values of the scores, and the std for accuracy
    # avg_result=avg_result.append(avg_result_to_pass)
    
    # avg_result_fuzzy_to_pass=pd.DataFrame([np.hstack((pd.DataFrame.mean(result_fuzzy).values,np.std(result_fuzzy.iloc[:,0].values,axis=0)))],columns=["Avg Accuracy","Avg F1 Score","Avg MCC","Avg ROC AUC","Avg Cohen's Kappa","Avg Recall","Avg CM (1,1)","Avg CM (1,2)","Avg CM (2,1)","Avg CM (2,2)","Accuracy Std"])
    # avg_result_fuzzy=avg_result_fuzzy.append(avg_result_fuzzy_to_pass)
    
    avg_result_fuzzy_FE_to_pass=pd.DataFrame([(pd.DataFrame.mean(result_fuzzy_FE).values)],columns=["Avg Accuracy","Avg F1 Score","Avg MCC","Avg ROC AUC","Avg Cohen's Kappa","Avg Recall","Avg CM (1,1)","Avg CM (1,2)","Avg CM (2,1)","Avg CM (2,2)"])
    avg_result_fuzzy_FE=avg_result_fuzzy_FE.append(avg_result_fuzzy_FE_to_pass)
    
    best_model_result=best_model_result.append(result_fuzzy_FE.iloc[np.argmax(result_fuzzy_FE.iloc[:,5]),:])
    
    std_results=std_results.append(pd.DataFrame([np.std(result_fuzzy_FE).values],columns=["Std Accuracy","Std F1 Score","Std MCC","Std ROC AUC","Std Cohen's Kappa","Std Recall","CM (1,1)","CM (1,2)","CM (2,1)","CM (2,2)"]))        #Initializes the scores dataframe
    
    
    
    print()
    print()
    print("Total elapsed time: ",round((time() - t1)/60)," minutes")                          
    # print("For ",i+1," different runs, with random weights, and early stopping to avoid overfitting:")
    # print()  
    
    big_boi_mean_and_std,big_boi_best=update(big_boi_mean_and_std,big_boi_best,avg_result_fuzzy_FE,std_results,best_model_result,l)
    
    Choice+=1


print()
print()
print("Total elapsed time: ",round((time() - t4)/60)," minutes")                          
print("For ",i+1," different runs, with random weights, and early stopping to avoid overfitting:")
print() 

###########################################
#Hypertuning with GridSearchCV
###########################################

# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer
# from itertools import permutations

# t5=time()

# for l in range(1,run_counter):

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
    
    
    
#     clf=GridSearchCV(MLPCla(),{'hidden_layer_sizes':list(permutations(range(5,30),2)),'activation':['logistic'], 'solver':['lbfgs'], 'alpha':[0.0001]
#                                 , 'max_iter':[10000], 'early_stopping':[True], 'validation_fraction':[0.1]
#                                 , 'n_iter_no_change':[10]}
#                       ,scoring={'Accuracy':'accuracy','ROC AUC':'roc_auc', 'MCC': make_scorer(matthews_corrcoef, greater_is_better=True, needs_proba=False)
#                                 ,'Recall':'recall'}, cv=5, verbose=0, refit='Recall', return_train_score=0)
    
#     # clf=GridSearchCV(MLPCla(),{'hidden_layer_sizes':list(permutations(range(1,100),2)),'activation':['logistic'], 'solver':['lbfgs'], 'alpha':[0.0001,0.1]
#     #                             , 'max_iter':[5000], 'early_stopping':[True], 'validation_fraction':[0.1]
#     #                             , 'n_iter_no_change':[10]}
#     #                   ,scoring={'Accuracy':'accuracy','ROC AUC':'roc_auc', 'MCC': make_scorer(matthews_corrcoef, greater_is_better=True, needs_proba=False)
#     #                             ,'Recall':'recall'}, cv=5, verbose=3, refit='MCC', return_train_score=0)
    
    
#     t_tuning=time()           
                                                                      
#     clf.fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
    
#     #Hyper_tuning_results=pd.DataFrame(clf.cv_results_)
    
#     print()
#     print()
#     print("Total tuning time: ",round((time() - t_tuning)/60)," minutes")  
    
#     #y_predicted_best_grid_fuzzy_FE=clf.best_estimator_.predict(X_test_fuzzy_FE)   #Obtains the predicted classes
        
#     #score_best_Grid_search_fuzzy_FE=scores(y_test_fuzzy_FE,y_predicted_best_grid_fuzzy_FE)   #Calculates the scores

#     print('Feature selection: ',l)
#     print(clf.best_estimator_)
#     Choice+=1
     

# print("Total elapsed time: ",round((time() - t5)/60)," minutes") 


# def score_print(result_1,result_2,result_3,string,index):
#     print("######################################################################################")
#     print()
#     print()
#     print("The avg.",string.format(),"w/o fuzz is:",np.mean(result_1[:,index]))
#     print()
#     print()    
#     print("The avg.",string.format(),"w/ fuzz is:",np.mean(result_2[:,index]))
#     print()
#     print()
#     print("The avg.",string.format(),"w/ fuzz and FE is:",np.mean(result_3[:,index]))
#     print()
#     print()
#     print("The max.",string.format(),"w/o fuzz is:",np.max(result_1[:,index])," corresponding to the ",np.argmax(result_1[:,index]),"th iteration of the model")
#     print()
#     print()
#     print("The max.",string.format(),"w/ fuzz is:",np.max(result_2[:,index])," corresponding to the ",np.argmax(result_2[:,index]),"th iteration of the model")
#     print()
#     print()
#     print("The max.",string.format(),"w/ fuzz and FE is:",np.max(result_3[:,index])," corresponding to the ",np.argmax(result_3[:,index]),"th iteration of the model")
#     print()
#     print()
#     print("######################################################################################")

# score_print(result,result_fuzzy,result_fuzzy_FE,'accu',0)       #Prints avg and max for accuracy across the models trained
# score_print(result,result_fuzzy,result_fuzzy_FE,'f1',1) 
# score_print(result,result_fuzzy,result_fuzzy_FE,'MCC',2) 
# score_print(result,result_fuzzy,result_fuzzy_FE,'ROC',3) 
# score_print(result,result_fuzzy,result_fuzzy_FE,'kappa',4) 


   
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
# plt.title("Diferent metrics across "+str(Number_of_models)+" models with normal inputs")
# plt.xlabel('Model Number')
# plt.ylabel("Metrics")
# plt.show()

# plt.figure()

# plt.plot(range(1,avg_result_fuzzy.iloc[:,0].shape[0]+1),avg_result_fuzzy.iloc[:,0],label="Accuracy (%)")
# plt.plot(range(1,avg_result_fuzzy.iloc[:,0].shape[0]+1),avg_result_fuzzy.iloc[:,1],label="F-Score")
# plt.plot(range(1,avg_result_fuzzy.iloc[:,0].shape[0]+1),avg_result_fuzzy.iloc[:,2],label="MCC")
# plt.plot(range(1,avg_result_fuzzy.iloc[:,0].shape[0]+1),avg_result_fuzzy.iloc[:,3],label="ROC AUC")
# plt.plot(range(1,avg_result_fuzzy.iloc[:,0].shape[0]+1),avg_result_fuzzy.iloc[:,4],label="Cohen's kappa")
# plt.legend()
# plt.title("Diferent metrics across "+str(Number_of_models)+" models with fuzzy features")
# plt.xlabel('Model Number')
# plt.ylabel("Metrics")
# plt.show()

# plt.figure()

# plt.plot(range(1,result_fuzzy_FE.iloc[:,0].shape[0]+1),result_fuzzy_FE.iloc[:,0],label="Accuracy (%)")
# plt.plot(range(1,result_fuzzy_FE.iloc[:,0].shape[0]+1),result_fuzzy_FE.iloc[:,1],label="F-Score")
# plt.plot(range(1,result_fuzzy_FE.iloc[:,0].shape[0]+1),result_fuzzy_FE.iloc[:,2],label="MCC")
# plt.plot(range(1,result_fuzzy_FE.iloc[:,0].shape[0]+1),result_fuzzy_FE.iloc[:,3],label="ROC AUC")
# plt.plot(range(1,result_fuzzy_FE.iloc[:,0].shape[0]+1),result_fuzzy_FE.iloc[:,4],label="Cohen's kappa")
# plt.plot(range(1,result_fuzzy_FE.iloc[:,0].shape[0]+1),result_fuzzy_FE.iloc[:,5],label="Recall")
# plt.legend()
# plt.title("Diferent metrics across "+str(Number_of_models)+" models with fuzzy and reduced features")
# plt.xlabel('Model Number')
# plt.ylabel("Metrics")
# plt.show()

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

# plt.figure()

# plt.plot(range(1,avg_result_fuzzy_FE.iloc[:,0].shape[0]+1),avg_result_fuzzy_FE.iloc[:,0],label="Accuracy (%)")
# plt.plot(range(1,avg_result_fuzzy_FE.iloc[:,0].shape[0]+1),avg_result_fuzzy_FE.iloc[:,1],label="F-Score")
# plt.plot(range(1,avg_result_fuzzy_FE.iloc[:,0].shape[0]+1),avg_result_fuzzy_FE.iloc[:,2],label="MCC")
# plt.plot(range(1,avg_result_fuzzy_FE.iloc[:,0].shape[0]+1),avg_result_fuzzy_FE.iloc[:,3],label="ROC AUC")
# plt.plot(range(1,avg_result_fuzzy_FE.iloc[:,0].shape[0]+1),avg_result_fuzzy_FE.iloc[:,4],label="Cohen's kappa")
# plt.plot(range(1,avg_result_fuzzy_FE.iloc[:,0].shape[0]+1),avg_result_fuzzy_FE.iloc[:,5],label="Recall")
# plt.legend()
# plt.title("Evolution of metrics with increase in depth of the tree, with fuzzy and reduced features")
# plt.xlabel('Number of layers')
# plt.ylabel("Metrics")
# plt.show()

# plt.figure()

# plt.plot(range(1,avg_result.iloc[:,0].shape[0]+1),avg_result.iloc[:,0],label="Raw data")
# plt.plot(range(1,avg_result_fuzzy.iloc[:,0].shape[0]+1),avg_result_fuzzy.iloc[:,0],label="Fuzzy data")
# plt.plot(range(1,avg_result_fuzzy_FE.iloc[:,0].shape[0]+1),avg_result_fuzzy_FE.iloc[:,0],label="Fuzzy with FE data")
# plt.legend()
# plt.title("Accuracy comparison between the three aproaches")
# plt.xlabel('Number of layers')
# plt.ylabel("Average Accuracy")
# plt.show()