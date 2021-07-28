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

sys.path.append("C:\Documents\MEMec_2015-2020\Tese\Python\Feature_selection")   #Goes to the other directory and gets the desired files


# Data_load import inputs, outputs
from Data_load_final import inputs, outputs
from sklearn import preprocessing as pre
from scipy import stats


######################################################################
#Pre processing with min-max
indexes_to_keep=(np.abs(stats.zscore(inputs[['idade', 'ACS altura', 'ACS peso']])) < 3).all(axis=1)
inputs=inputs[indexes_to_keep]
outputs=outputs[indexes_to_keep]

temporary = inputs.values #returns a numpy array
min_max_scaler = pre.MinMaxScaler()
temporary_scaled = min_max_scaler.fit_transform(temporary)
inputs_normal = pd.DataFrame(temporary_scaled,columns=inputs.columns)  #All normalized now
#######################################################################

# test=inputs.iloc[:,1].values   #Starting with ages


Default_membership_x=np.arange(inputs.shape[0])
#Standardizing
Default_membership_x=np.reshape(min_max_scaler.fit_transform(Default_membership_x.reshape(-1,1)),inputs.shape[0])   #Reshape due to function restrictions

#Don´t care membership function

mf_DC=np.empty(inputs.shape[0])
mf_DC.fill(1)

plt.plot(Default_membership_x, mf_DC, label='Do not Care')
plt.legend()
plt.ylim(0,1.2)
plt.xlim(0,1)

#2 membership functions from [0 to 1]

mf_2_small=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0,0.5),inputs.shape[0])     
mf_2_large=np.reshape(fuzz.membership.gaussmf(Default_membership_x,1,0.5),inputs.shape[0])     

plt.figure()

plt.plot(Default_membership_x,mf_2_small, label='Small')
plt.plot(Default_membership_x,mf_2_large, label='Large')
plt.legend()
plt.ylim(0,1.2)
plt.xlim(0,1)

#3 membership functions from [0 to 1]

mf_3_small=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0,0.2),inputs.shape[0])      
mf_3_median=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0.5,0.2),inputs.shape[0])    
mf_3_large=np.reshape(fuzz.membership.gaussmf(Default_membership_x,1,0.2),inputs.shape[0])     

plt.figure()

plt.plot(Default_membership_x,mf_3_small, label='Small')
plt.plot(Default_membership_x,mf_3_median, label='Median')
plt.plot(Default_membership_x,mf_3_large, label='Large')
plt.legend()
plt.ylim(0,1.2)
plt.xlim(0,1)

#4 membership functions from [0 to 1]

mf_4_small=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0,0.15),inputs.shape[0])      #Membership function for ages with a median of 0.5 and sigma of 0.1
mf_4_medium_small=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0.33,0.15),inputs.shape[0])      #Membership function for ages with a median of 0.5 and sigma of 0.1
mf_4_medium_large=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0.66,0.15),inputs.shape[0])    #Membership function for ages with a median of 0.5 and sigma of 0.1
mf_4_large=np.reshape(fuzz.membership.gaussmf(Default_membership_x,1,0.15),inputs.shape[0]) 

plt.figure()

plt.plot(Default_membership_x,mf_4_small, label='Small')
plt.plot(Default_membership_x,mf_4_medium_small, label='Medium Small')
plt.plot(Default_membership_x,mf_4_medium_large, label='Medium Large')
plt.plot(Default_membership_x,mf_4_large, label='Large')
plt.legend()
plt.ylim(0,1.2)
plt.xlim(0,1)

#5 membership functions from [0 to 1]

mf_5_small=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0,0.11),inputs.shape[0])     #Membership function for ages with a median of 0.5 and sigma of 0.1
mf_5_medium_small=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0.25,0.11),inputs.shape[0])      #Membership function for ages with a median of 0.5 and sigma of 0.1
mf_5_median=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0.5,0.11),inputs.shape[0])  
mf_5_medium_large=np.reshape(fuzz.membership.gaussmf(Default_membership_x,0.75,0.11),inputs.shape[0])      #Membership function for ages with a median of 0.5 and sigma of 0.1
mf_5_large=np.reshape(fuzz.membership.gaussmf(Default_membership_x,1,0.11),inputs.shape[0]) 

plt.figure()

plt.plot(Default_membership_x,mf_5_small, label='Small')
plt.plot(Default_membership_x,mf_5_medium_small, label='Medium Small')
plt.plot(Default_membership_x,mf_5_median, label='Median')
plt.plot(Default_membership_x,mf_5_medium_large, label='Medium Large')
plt.plot(Default_membership_x,mf_5_large, label='Large')
plt.legend()
plt.ylim(0,1.2)
plt.xlim(0,1)

mf=pd.DataFrame(data={'Dont Care':mf_DC,
                      'Small_2':mf_2_small,'Large_2':mf_2_large,
                      'Small_3':mf_3_small,'Median_3':mf_3_median,'Large_3':mf_3_large,
                      'Small_4':mf_4_small,'Medium_small_4':mf_4_medium_small,'Medium_large_4':mf_4_medium_large,'Large_4':mf_4_large,
                      'Small_5':mf_5_small,'Medium_small_5':mf_5_medium_small,'Medium_5':mf_5_median,'Medium_large_5':mf_5_medium_large,'Large_5':mf_5_large})
##############################
#Membership interpretation
##############################


def clc_memb(Default,mf,var):       #Calculates the membership value for the 15 different membership functions
    result=np.empty([var.shape[0],1])
    for i in range(mf.shape[1]):
        result=np.concatenate((result,fuzz.interp_membership(Default,mf.iloc[:,i].values,var).reshape([var.shape[0],1])),axis=1)
    return result
        
def dataframe_append(Default,mf,var):       #Calculates the membership value for all features
    result=np.empty([var.shape[0],1])
    for i in range(var.shape[1]):
        result=np.concatenate((result,clc_memb(Default,mf,var.iloc[:,i].values)),axis=1)
    return result
        
     
        
fuzzified_inputs=dataframe_append(Default_membership_x,mf,inputs_normal)       #Obtains the fuzzified features

##############################
#ANN with fuzzified inputs
##############################

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier as MLPCla
from sklearn.model_selection import train_test_split
from Feature_selection import all_scored_compli, all_scored_obito, all_scored_dias, all_scored_clavien, all_scored_reint
from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from time import time

def scores(Y_Test,Y_Pred,i):
    accu=accuracy_score(Y_Test,Y_Pred)
    f1=f1_score(Y_Test,Y_Pred)
    mathew=matthews_corrcoef(Y_Test,Y_Pred)
    roc=roc_auc_score(Y_Test,Y_Pred)
    kappa=cohen_kappa_score(Y_Test,Y_Pred)
    confusion=confusion_matrix(Y_Test,Y_Pred).flatten()
    
    score=np.hstack((accu,f1,mathew,roc,kappa,confusion,i))        #Stacks accuracy, F1-score, MCC, ROC, Kappa, confusion matrix in a single array
    return score


print('Beggining model training')
print()
fuzzified_inputs_FE=dataframe_append(Default_membership_x,mf,inputs_normal[all_scored_compli.index[:15].values]) #Fuzzification of the 15 best features for complication

avg_result_fuzzy_FE=np.empty([0,10])

maximum_range=10

maximum_range_2=30

# j=0
t1=time()
        
    
result_fuzzy_FE=np.empty([0,10])

hidden=round(math.sqrt(fuzzified_inputs_FE.shape[1]))
architecture=hidden
which='lbfgs'
    
for j in range(3):
    
    print('Training with ',j+1,' hidden layers of',hidden,' neurons each.')
    print()
    t0 = time()

    for i in range(maximum_range_2):
        
        ##################################################################
        ##Model run with feature selection according to previous selection
        ##################################################################  
    
        X_train_fuzzy_FE, X_test_fuzzy_FE, y_train_fuzzy_FE, y_test_fuzzy_FE = train_test_split(fuzzified_inputs_FE, outputs.values[:,2], test_size=0.25) #,random_state=i)
        
        
        clf_fuzzy_with_FE=MLPCla(hidden_layer_sizes=((16,16)),activation='logistic', solver=which,alpha=1e-5,max_iter=5000,early_stopping=True,verbose=0).fit(X_train_fuzzy_FE,y_train_fuzzy_FE)
        
        y_predicted_fuzzy_FE=clf_fuzzy_with_FE.predict(X_test_fuzzy_FE)
        
        score_fuzzy_FE=scores(y_test_fuzzy_FE,y_predicted_fuzzy_FE,i)
        
        score_fuzzy_FE=np.hstack((score_fuzzy_FE))
        #architecture=np.hstack((architecture,hidden))
        
        result_fuzzy_FE=np.vstack((result_fuzzy_FE,score_fuzzy_FE))
        
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
    if j==0:
        which='adam'
    elif j==1:
        which='sgd'
        
    print()
    print( round((time() - t0)/60), ' minutes to train ',i,' models with ',j+1,' hidden layers of ',hidden,' neurons each.')
    print()
    avg_result_fuzzy_FE=np.vstack((avg_result_fuzzy_FE,np.mean(result_fuzzy_FE,axis=0)))
    



print()
print()
print("Total elapsed time: ",round((time() - t1)/60)," minutes")                          
print("For ",i+1," different runs, with random weights, and early stopping to avoid overfitting:")
print()  


def score_print(result_1,result_2,result_3,string,index):
    print("######################################################################################")
    print()
    print()
    print("The avg.",string.format(),"w/o fuzz is:",np.mean(result_1[:,index]))
    print()
    print()    
    print("The avg.",string.format(),"w/ fuzz is:",np.mean(result_2[:,index]))
    print()
    print()
    print("The avg.",string.format(),"w/ fuzz and FE is:",np.mean(result_3[:,index]))
    print()
    print()
    print("The max.",string.format(),"w/o fuzz is:",np.max(result_1[:,index])," corresponding to the ",np.argmax(result_1[:,index]),"th iteration of the model")
    print()
    print()
    print("The max.",string.format(),"w/ fuzz is:",np.max(result_2[:,index])," corresponding to the ",np.argmax(result_2[:,index]),"th iteration of the model")
    print()
    print()
    print("The max.",string.format(),"w/ fuzz and FE is:",np.max(result_3[:,index])," corresponding to the ",np.argmax(result_3[:,index]),"th iteration of the model")
    print()
    print()
    print("######################################################################################")

# score_print(result,result_fuzzy,result_fuzzy_FE,'accu',0)       #Prints avg and max for accuracy across the models trained
# score_print(result,result_fuzzy,result_fuzzy_FE,'f1',1) 
# score_print(result,result_fuzzy,result_fuzzy_FE,'MCC',2) 
# score_print(result,result_fuzzy,result_fuzzy_FE,'ROC',3) 
# score_print(result,result_fuzzy,result_fuzzy_FE,'kappa',4) 


#Hyperparameters exploring
   
import matplotlib.pyplot as plt
    
# plt.plot(avg_result[:,0])
# plt.plot(avg_result_fuzzy[:,0])
# plt.plot(range(1,11),avg_result_fuzzy_FE[:,0],label="Accuracy (%)")
# plt.plot(range(1,11),avg_result_fuzzy_FE[:,1],label="F-Score")
# plt.plot(range(1,11),avg_result_fuzzy_FE[:,2],label="MCC")
# plt.plot(range(1,11),avg_result_fuzzy_FE[:,3],label="ROC AUC")
# plt.plot(range(1,11),avg_result_fuzzy_FE[:,4],label="Cohen's kappa")
# plt.legend()
# # plt.title("Evolution of accuracy with increase in hidden layers")
# # plt.xlabel("Number of hidden layers")
# plt.title("Evolution of accuracy with increase in Neurons")
# plt.xlabel("Number of Neurons")
# plt.ylabel("Metrics")
# plt.show()

# plt.plot(avg_result[:,0])
# plt.plot(avg_result_fuzzy[:,0])
plt.plot(range(1,avg_result_fuzzy_FE.shape[0]+1),avg_result_fuzzy_FE[:,0],label="Accuracy (%)")
plt.plot(range(1,avg_result_fuzzy_FE.shape[0]+1),avg_result_fuzzy_FE[:,1],label="F-Score")
plt.plot(range(1,avg_result_fuzzy_FE.shape[0]+1),avg_result_fuzzy_FE[:,2],label="MCC")
plt.plot(range(1,avg_result_fuzzy_FE.shape[0]+1),avg_result_fuzzy_FE[:,3],label="ROC AUC")
plt.plot(range(1,avg_result_fuzzy_FE.shape[0]+1),avg_result_fuzzy_FE[:,4],label="Cohen's kappa")
plt.legend()
# plt.title("Evolution of accuracy with increase in hidden layers")
# plt.xlabel("Number of hidden layers")
plt.title("Evolution of accuracy with increase in Neurons")
plt.xlabel("Number of model")
plt.ylabel("Metrics")
plt.show()