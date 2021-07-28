#This is the code that evaluates the performance of the existing scorers gainst our data 
#Already implemented:
#Lasso
#Pearson Correlation
#Spearman Corrleation
#Average of methods
##
##
#Python 3.7 by André Miranda @ andre.lima.miranda@tecnico.ulisboa.pt


import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt

from Data_load_final import inputs,ignored, outputs        #4th version of database

from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedShuffleSplit



def scores(Y_Test,Y_Pred):              #This is the function that obtains all of the scores
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

def thresh(y,threshold):          #This is the function that converts into binary  
    out=y                   
    for i in range(y.shape[0]):     #estimates with a specified threshold
        if y.iloc[i]<threshold:
            out.iloc[i]=0
        else:
            out.iloc[i]=1
    
    return out

##########################################################################
# Output setup

Y_complication=outputs['complicação pós-cirúrgica']
Y_death=outputs['óbito até 1 ano após cirurgia']

# 'nº dias na UCI', 'destino após UCI', 'complicação pós-cirúrgica'
#             , 'classificação clavien-dindo', 'destino após IPO'
#             , 'óbito até 1 ano após cirurgia',' reinternamento na UCI'
#             ,'nº dias no  IPO'

# ##########################################################################
# ## P-Possum - BOTH
# ##########################################################################  
    

PP_complication=ignored['% morbilidade P-Possum'].astype(float)
PP_complication_thresh=thresh(PP_complication/100,0.5)
A_PP_complication_score=scores(Y_complication,PP_complication_thresh)

PP_death=ignored['% mortalidade P-Possum'].astype(float)
PP_death_thresh=thresh(PP_death/100,0.1)
A_PP_death_score=scores(Y_death,PP_death_thresh)


##########################################################################
## ACS risk scorer - BOTH
########################################################################## 


Z_ACS_complication=ignored['qualquer complicação (%)'].astype(float)
Z_ACS_complication_thresh=thresh(Z_ACS_complication/100,0.2)
ACS_complication_score=scores(Y_complication,Z_ACS_complication_thresh)

Z_ACS_death=ignored['morte (%)'].astype(float)
Z_ACS_death_thresh=thresh(Z_ACS_death/100,0.02)
ACS_death_score=scores(Y_death,Z_ACS_death_thresh)


##########################################################################
## ARISCAT risk scorer - COMPLICATIONS
########################################################################## 


Z_ARISCAT_complication=ignored['ARISCAT PONTUAÇÃO TOTAL'].astype(float)
Z_ARISCAT_complication_thresh=thresh(Z_ARISCAT_complication/max(Z_ARISCAT_complication),0.33)
ARISCAT_complication_score=scores(Y_complication,Z_ARISCAT_complication_thresh)

#ARISCAT DOES NOT PREDICT DEATH, ONLY COMPLICATIONS


##########################################################################
## Charlson risk scorer - DEATH
########################################################################## 


#Charlson DOES NOT PREDICT COMPLICATIONS, ONLY DEATH

Charlson_death=ignored['% Sobrevida estimada em 10 anos'].astype(float)
Charlson_death_thresh=thresh(Charlson_death/max(Charlson_death),0.5)
A_Charlson_death_score=scores(Y_death,Charlson_death_thresh)


##########################################################################
## Threshold exploring
########################################################################## 

# A_PP_complication_score=A_PP_death_score=ACS_complication_score=ACS_death_score=ARISCAT_complication_score=A_Charlson_death_score=pd.DataFrame(columns=["Accuracy","F1 Score","MCC","ROC AUC","Cohen's Kappa","Recall","CM (1,1)","CM (1,2)","CM (2,1)","CM (2,2)"])

# for i in range(100):
    
#     #######################################
#     ## P-Possum - BOTH
#     #######################################

    
#     PP_complication_thresh=thresh(PP_complication/100,(i)/100)
#     A_PP_complication_score=A_PP_complication_score.append(scores(Y_complication,PP_complication_thresh))
    
    
#     PP_death_thresh=thresh(PP_death/100,(i)/100)
#     A_PP_death_score=A_PP_death_score.append(scores(Y_death,PP_death_thresh))
    
    
#     #######################################
#     ## ACS risk scorer - BOTH
#     #######################################
    
    
#     Z_ACS_complication_thresh=thresh(Z_ACS_complication/100,(i)/100)
#     ACS_complication_score=ACS_complication_score.append(scores(Y_complication,Z_ACS_complication_thresh))
    
#     Z_ACS_death_thresh=thresh(Z_ACS_death/100,(i)/100)
#     ACS_death_score=ACS_death_score.append(scores(Y_death,Z_ACS_death_thresh))
    
    
#     #######################################
#     ## ARISCAT risk scorer - COMPLICATIONS
#     #######################################
    
    
#     Z_ARISCAT_complication_thresh=thresh(Z_ARISCAT_complication/max(Z_ARISCAT_complication),(i)/100)
#     ARISCAT_complication_score=ARISCAT_complication_score.append(scores(Y_complication,Z_ARISCAT_complication_thresh))
    
    
#     #######################################
#     ## Charlson risk scorer - DEATH
#     #######################################
    
#     Charlson_death=ignored['% Sobrevida estimada em 10 anos'].astype(float)
#     Charlson_death_thresh=thresh(Charlson_death/max(Charlson_death),(i)/100)
#     A_Charlson_death_score=A_Charlson_death_score.append(scores(Y_death,Charlson_death_thresh))
    

# plt.figure()

# plt.plot( np.linspace(0.01,1,100),A_PP_complication_score.iloc[:,0],label="Accuracy (%)")
# plt.plot( np.linspace(0.01,1,100),A_PP_complication_score.iloc[:,1],label="F-Score")
# plt.plot( np.linspace(0.01,1,100),A_PP_complication_score.iloc[:,2],label="MCC")
# plt.plot( np.linspace(0.01,1,100),A_PP_complication_score.iloc[:,3],label="ROC AUC")
# plt.plot( np.linspace(0.01,1,100),A_PP_complication_score.iloc[:,4],label="Cohen's kappa")
# plt.plot( np.linspace(0.01,1,100),A_PP_complication_score.iloc[:,5],label="Recall")
# plt.legend()
# #plt.title("Effect of threshold value on scorers performance - P-Possum Complications")
# plt.xlabel('Threshold')
# plt.ylabel("Metrics")
# plt.show()

# plt.figure()

# plt.plot( np.linspace(0.01,1,100),  A_PP_death_score.iloc[:,0],label="Accuracy (%)")
# plt.plot( np.linspace(0.01,1,100),  A_PP_death_score.iloc[:,1],label="F-Score")
# plt.plot( np.linspace(0.01,1,100),  A_PP_death_score.iloc[:,2],label="MCC")
# plt.plot( np.linspace(0.01,1,100),  A_PP_death_score.iloc[:,3],label="ROC AUC")
# plt.plot( np.linspace(0.01,1,100),  A_PP_death_score.iloc[:,4],label="Cohen's kappa")
# plt.plot( np.linspace(0.01,1,100),  A_PP_death_score.iloc[:,5],label="Recall")
# plt.legend()
# plt.legend()
# #plt.title("Effect of threshold value on scorers performance - P-Possum One-year mortality")
# plt.xlabel('Threshold')
# plt.ylabel("Metrics")
# plt.show()

# plt.figure()

# plt.plot( np.linspace(0.01,1,100),ACS_complication_score.iloc[:,0],label="Accuracy (%)")
# plt.plot( np.linspace(0.01,1,100),ACS_complication_score.iloc[:,1],label="F-Score")
# plt.plot( np.linspace(0.01,1,100),ACS_complication_score.iloc[:,2],label="MCC")
# plt.plot( np.linspace(0.01,1,100),ACS_complication_score.iloc[:,3],label="ROC AUC")
# plt.plot( np.linspace(0.01,1,100),ACS_complication_score.iloc[:,4],label="Cohen's kappa")
# plt.plot( np.linspace(0.01,1,100),ACS_complication_score.iloc[:,5],label="Recall")
# plt.legend()
# #plt.title("Effect of threshold value on scorers performance - ACS Complications")
# plt.xlabel('Threshold')
# plt.ylabel("Metrics")
# plt.show()

# plt.figure()

# plt.plot( np.linspace(0.01,1,100),  ACS_death_score.iloc[:,0],label="Accuracy (%)")
# plt.plot( np.linspace(0.01,1,100),  ACS_death_score.iloc[:,1],label="F-Score")
# plt.plot( np.linspace(0.01,1,100),  ACS_death_score.iloc[:,2],label="MCC")
# plt.plot( np.linspace(0.01,1,100),  ACS_death_score.iloc[:,3],label="ROC AUC")
# plt.plot( np.linspace(0.01,1,100),  ACS_death_score.iloc[:,4],label="Cohen's kappa")
# plt.plot( np.linspace(0.01,1,100),  ACS_death_score.iloc[:,5],label="Recall")
# plt.legend()
# plt.legend()
# #plt.title("Effect of threshold value on scorers performance - ACS One-year mortality")
# plt.xlabel('Threshold')
# plt.ylabel("Metrics")
# plt.show()
    
# plt.figure()

# plt.plot( np.linspace(0.01,1,100),ARISCAT_complication_score.iloc[:,0],label="Accuracy (%)")
# plt.plot( np.linspace(0.01,1,100),ARISCAT_complication_score.iloc[:,1],label="F-Score")
# plt.plot( np.linspace(0.01,1,100),ARISCAT_complication_score.iloc[:,2],label="MCC")
# plt.plot( np.linspace(0.01,1,100),ARISCAT_complication_score.iloc[:,3],label="ROC AUC")
# plt.plot( np.linspace(0.01,1,100),ARISCAT_complication_score.iloc[:,4],label="Cohen's kappa")
# plt.plot( np.linspace(0.01,1,100),ARISCAT_complication_score.iloc[:,5],label="Recall")
# plt.legend()
# #plt.title("Effect of threshold value on scorers performance - ARISCAT Complications")
# plt.xlabel('Threshold')
# plt.ylabel("Metrics")
# plt.show()

# plt.figure()

# plt.plot( np.linspace(0.01,1,100),A_Charlson_death_score.iloc[:,0],label="Accuracy (%)")
# plt.plot( np.linspace(0.01,1,100),A_Charlson_death_score.iloc[:,1],label="F-Score")
# plt.plot( np.linspace(0.01,1,100),A_Charlson_death_score.iloc[:,2],label="MCC")
# plt.plot( np.linspace(0.01,1,100),A_Charlson_death_score.iloc[:,3],label="ROC AUC")
# plt.plot( np.linspace(0.01,1,100),A_Charlson_death_score.iloc[:,4],label="Cohen's kappa")
# plt.plot( np.linspace(0.01,1,100),A_Charlson_death_score.iloc[:,5],label="Recall")
# plt.legend()
# plt.legend()
# #plt.title("Effect of threshold value on scorers performance - Charlson One-year mortality")
# plt.xlabel('Threshold')
# plt.ylabel("Metrics")
# plt.show()
    
##########################################################################
## 200 models estimation
########################################################################## 

def update(big_boi_mean_and_std,big_boi_best,avg_result_fuzzy_FE,std_results,best_model_result,l):    #Function to update the results with all feature selection methods
    big_boi_mean_and_std.iloc[2*l,:]=avg_result_fuzzy_FE.iloc[0,:6].values
    big_boi_mean_and_std.iloc[2*l+1,:]=std_results.iloc[0,:6].values
    big_boi_best.iloc[l,:]=best_model_result.iloc[0,:6].values
    return(big_boi_mean_and_std,big_boi_best)


def avg_hospital_scores(inputs,Y_pristine, Y_hospital):
    global all_scores,best_model_result
    all_scores=pd.DataFrame(columns=["Accuracy","F1 Score","MCC","ROC AUC","Cohen's Kappa","Recall","CM (1,1)","CM (1,2)","CM (2,1)","CM (2,2)"])
    
    for i in range(200):
        
        sss=StratifiedShuffleSplit(1,test_size=0.3, random_state=i)
         
        
        for train_index, test_index in sss.split(inputs, Y_hospital):             #Cycle to create the stratified training and testing sets
            x, x_test = inputs.iloc[train_index], inputs.iloc[test_index]
            y, y_test = Y_hospital.iloc[train_index], Y_hospital.iloc[test_index]
            n,  Y     = Y_pristine.iloc[train_index], Y_pristine.iloc[test_index]
        
        all_scores=all_scores.append(scores(Y, y_test))
    
    avg_result=pd.DataFrame([(pd.DataFrame.mean(all_scores).values)],columns=["Avg Accuracy","Avg F1 Score","Avg MCC","Avg ROC AUC","Avg Cohen's Kappa","Avg Recall","Avg CM (1,1)","Avg CM (1,2)","Avg CM (2,1)","Avg CM (2,2)"])
    
    best_model_result=pd.DataFrame(data=[all_scores.iloc[np.argmax(all_scores.iloc[:,5]),:].values],columns=["Best Accuracy","Best F1 Score","Best MCC","Best ROC AUC","Best Cohen's Kappa","Best Recall","Best CM (1,1)","Best CM (1,2)","Best CM (2,1)","Best CM (2,2)"])
    
    
    std_results=pd.DataFrame([np.std(all_scores).values],columns=["Std Accuracy","Std F1 Score","Std MCC","Std ROC AUC","Std Cohen's Kappa","Std Recall","CM (1,1)","CM (1,2)","CM (2,1)","CM (2,2)"])  
    
    return(avg_result,std_results,best_model_result)
    
big_boi_mean_and_std=pd.DataFrame(index=["PP-complications mean", "PP-complications std", "PP-death mean", "PP-death std"
                                         , "ACS-complications mean", "ACS-complications std", "ACS-death mean", "ACS-death std"
                                         , "ARISCAT-complications mean", "ARISCAT-complications std"
                                         , "Charlson-death mean", "Charlson-death std"],columns=["Accuracy","F1 Score","MCC","ROC AUC","Cohen's Kappa","Recall"])

big_boi_best=pd.DataFrame(index=["PP-complications", "PP-death"
                                 , "ACS-complications", "ACS-death"
                                 , "ARISCAT-complications"
                                 , "Charlson-death"],columns=["Accuracy","F1 Score","MCC","ROC AUC","Cohen's Kappa","Recall"])

all_threshes=list([PP_complication_thresh,PP_death_thresh
                   ,Z_ACS_complication_thresh,Z_ACS_death_thresh
                   ,Z_ARISCAT_complication_thresh
                   ,Charlson_death_thresh])

all_Y=list([Y_complication,Y_death,
            Y_complication,Y_death
            ,Y_complication
            ,Y_death])
l=0
for Y_hospital,Y in zip(all_threshes,all_Y):
    
    a,s,b=avg_hospital_scores(inputs,Y, Y_hospital)
    big_boi_mean_and_std,big_boi_best=update(big_boi_mean_and_std,big_boi_best,a,s,b,l)
    
    l+=1