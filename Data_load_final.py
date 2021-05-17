#This is the code that reads the database contents and separates inputs to outputs
##
##
#Python 3.7 by André Miranda @ andre.lima.miranda@tecnico.ulisboa.pt

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
import pandas as pd
import matplotlib.pylab as pylt
from sklearn.feature_selection import VarianceThreshold

Location=r'C:\Documents\MEMec_2015-2020\Tese\Base_Dados\Database_4_without_header.csv'

alldata=pd.read_csv(Location, sep = ';',encoding='utf-8')

df1=pd.DataFrame(alldata)       #Data structure with all features
df1.replace(',','.',regex=True, inplace=True)   #replaces any comma for a dot,correcting common excel to number error 

       # 'data pedido pela anestesia', 'tipo pedido anestesia',
       # 'data admissão UCI', 'nº IPO', 'idade', 'género', 'proveniência',
       # 'motivo admissão UCI', 'tipo cirurgia', 'especialidade',
       # 'especialidade_COD', 'nº dias na UCI', 'nº dias no  IPO',
       # 'passou por SCI ao longo do internamento no IPO (antes ou após UCI)',
       # 'destino após UCI', 'total pontos NAS', 'média pontos NAS por dia',
       # 'realizou cirurgia', '1ª Cirurgia IPO', 'QT pré-operatória',
       # ' reinternamento na UCI', 'motivo de reinternamento na UCI', 'ASA',
       # 'LOCALIZAÇÃO ', 'diagnóstico pré-operatório', 'data cirurgia',
       # 'Operação efetuada', 'Intervenções_ICD10', 'procedimentos_COD',
       # 'PP idade', 'PP cardíaco', 'PP respiratório', 'PP ECG',
       # 'PP pressão arterial sistólica', 'PP pulsação arterial',
       # 'PP hemoglobina', 'PP leucócitos', 'PP ureia', 'PP sódio',
       # 'PP potássio', 'PP escala glasglow', 'PP tipo operação',
       # 'PP nº procedimentos', 'PP perda sangue', 'PP contaminação peritoneal',
       # 'PP estado da malignidade', 'PP CEPOD-classificação operação',
       # 'Score fisiológico P-Possum', 'Score gravidade cirúrgica P-Possum',
       # '% morbilidade P-Possum' '% mortalidade P-Possum', 'ACS_procedimento',
       # 'ACS idade', 'ACS género', 'ACS estado funcional', 'ACS emergência',
       # 'ACS ASA', 'ACS esteróides', 'ACS ascite', 'ACS sépsis sistémica',
       # 'ACS dependente ventilador', 'ACS cancro disseminado', 'ACS diabetes',
       # 'ACS hipertensão', 'ACS ICC', 'ACS dispneia', 'ACS fumador', 'ACS DPOC',
       # 'ACS diálise', 'ACS insuficiência renal aguda', 'ACS altura',
       # 'ACS peso', 'complicações sérias (%)', 'risco médio',
       # 'qualquer complicação (%)', 'risco médio.1', 'pneumonia (%)',
       # 'risco médio.2', 'complicações cardíacas (%)', 'risco médio.3',
       # 'infeção cirúrgica (%)', 'risco médio.4', 'ITU (%)', 'risco médio.5',
       # 'tromboembolismo venoso (%)', 'risco médio.6', 'falência renal (%)',
       # 'risco médio.7', 'ileus (%)', 'risco médio.8', 'fuga anastomótica (%)',
       # 'risco médio.9', 'readmissão (%)', 'risco médio.10', 'reoperação (%)',
       # 'risco médio.11', 'morte (%)', 'risco médio.12',
       # 'Discharge to Nursing or Rehab Facility (%)', 'risco médio.13',
       # 'ACS - previsão dias internamento', 'ARISCAT Idade', 'ARISCAT SpO2 ',
       # 'ARISCAT infeção respiratória último mês',
       # 'ARISCAT anemia pré-operativa', 'ARISCAT incisão cirúrgica',
       # 'ARISCAT duração cirurgia', 'ARISCAT procedimento emergente',
       # 'ARISCAT PONTUAÇÃO TOTAL', 'SCORE ARISCAT', 'CHARLSON Idade',
       # 'CHARLSON Diabetes Mellitus', 'CHARLSON Doença fígado',
       # 'CHARLSON Malignidade-Solid tumor', 'CHARLSON SIDA',
       # 'CHARLSON Doença Renal Crónica Moderada a Severa',
       # 'CHARLSON Insuficiência Cardíaca', 'CHARLSON Enfarte Miocárdio',
       # 'CHARLSON DPOC', 'CHARLSON Doença Vascular periférica',
       # 'CHARLSON AVC ou Ataque Isquémico Transitório', 'CHARLSON Demência',
       # 'CHARLSON Hemiplegia', 'CHARLSON Doença do Tecido Conjuntivo',
       # 'CHARLSON Úlcera Péptica', 'PONTOS - Charlson Comorbidity Index',
       # '% Sobrevida estimada em 10 anos', 'complicação pós-cirúrgica',
       # 'descrição complicação pós-cirúrgica', 'complicação_COD',
       # 'complicação principal_COD', 'classificação ACS complicações gerais',
       # 'classificação ACS complicações específicas',
       # 'classificação clavien-dindo', 'destino após IPO',
       # 'óbito até 1 ano após cirurgia', 'momento do óbito após a cirurgia ',
       # 'data óbito', 'Informação adicional do óbito',
       # 'Informação adicional do óbito traduzida para Inglês',
       # 'Co-morbilidades aquando internamento'



# print()
# print('Database read!!')
# print()
########################
###Inputs and Outputs###
########################

#Ignored data

ignored=df1[['data admissão UCI', 'nº IPO', 'motivo de reinternamento na UCI','realizou cirurgia'
             , 'procedimentos_COD', 'Score fisiológico P-Possum', 'Score gravidade cirúrgica P-Possum'
             , '% morbilidade P-Possum', '% mortalidade P-Possum', 'complicações sérias (%)', 'risco médio'
             , 'qualquer complicação (%)', 'risco médio.1', 'pneumonia (%)'
             , 'risco médio.2', 'complicações cardíacas (%)', 'risco médio.3'
             , 'infeção cirúrgica (%)', 'risco médio.4', 'ITU (%)', 'risco médio.5'
             , 'tromboembolismo venoso (%)', 'risco médio.6', 'falência renal (%)'
             , 'risco médio.7', 'ileus (%)', 'risco médio.8', 'fuga anastomótica (%)'
             , 'risco médio.9', 'readmissão (%)', 'risco médio.10', 'reoperação (%)'
             , 'risco médio.11', 'morte (%)', 'risco médio.12'
             , 'Discharge to Nursing or Rehab Facility (%)', 'risco médio.13'
             , 'ACS - previsão dias internamento', 'ARISCAT PONTUAÇÃO TOTAL', 'SCORE ARISCAT'
             , 'PONTOS - Charlson Comorbidity Index', '% Sobrevida estimada em 10 anos'
             , 'descrição complicação pós-cirúrgica', 'complicação_COD'
             , 'complicação principal_COD', 'classificação ACS complicações gerais'
             , 'classificação ACS complicações específicas', 'momento do óbito após a cirurgia '
             , 'data óbito', 'Informação adicional do óbito'
             , 'Informação adicional do óbito traduzida para Inglês','PP idade','ACS idade'
             , 'ARISCAT Idade', 'CHARLSON Idade', 'ACS género', 'PP CEPOD-classificação operação'
             , 'ACS emergência', 'ARISCAT procedimento emergente', 'ACS ASA'
             ,'CHARLSON Malignidade-Solid tumor', 'CHARLSON Diabetes Mellitus'
             , 'CHARLSON DPOC', 'total pontos NAS', 'média pontos NAS por dia'
             , 'passou por SCI ao longo do internamento no IPO (antes ou após UCI)']]


#Variables with strings
stringvariables=df1[['data pedido pela anestesia','especialidade','LOCALIZAÇÃO '
                     , 'diagnóstico pré-operatório', 'data cirurgia'
                     , 'Operação efetuada', 'Intervenções_ICD10'
                     , 'ACS_procedimento','Co-morbilidades aquando internamento']]


# print()
# print('Stringed variables separated!!')
# print()

#Treatment of some stringed variables
#ICD10
# import re
# import unidecode

# all_procedures_ICD10=[]

# for i in range(stringvariables.shape[0]):
#     s = stringvariables.iloc[i,6]
#     s=re.sub(r'[:-}]+', '', unidecode.unidecode(s))
#     s=re.sub(r'\(', '', unidecode.unidecode(s))
#     s=re.sub(r'\)', '', unidecode.unidecode(s))
#     s=re.sub(r'\.', '', unidecode.unidecode(s))
#     s=re.sub(r'\-', '', unidecode.unidecode(s))   
#     s=re.sub(r'/', '', unidecode.unidecode(s))
#     s=re.sub(r',', '', unidecode.unidecode(s))
#     s=re.sub(r' ', '', unidecode.unidecode(s))
#     s=re.sub(r'\n', '_', unidecode.unidecode(s)) 
#     stringvariables.iloc[i,6]=re.sub(r'\+', '_', unidecode.unidecode(s))
#     all_procedures_ICD10=np.hstack([all_procedures_ICD10, stringvariables.iloc[i,6].split("_")])


#Outputs

outputs=df1[['nº dias na UCI', 'destino após UCI', 'complicação pós-cirúrgica'
            , 'classificação clavien-dindo', 'destino após IPO'
            , 'óbito até 1 ano após cirurgia',' reinternamento na UCI','nº dias no  IPO' ]]


# print()
# print('Output variables separated!!')
# print()


#Inputs (everything else)


prepreinputs=df1.drop(stringvariables.columns, axis=1)
preinputs=prepreinputs.drop(ignored.columns,axis=1)    #Variables that are repeated or not relevant
inputs=preinputs.drop(outputs.columns,axis=1)

#Removing quasi constant features

filter_thresh=VarianceThreshold(0.02)
filter_thresh.fit(inputs)
zero_variance=inputs.drop(inputs.columns[filter_thresh.get_support()],axis=1)

inputs=inputs[inputs.columns[filter_thresh.get_support()]]

#
#        51 final input features before feature selection
#
#        Continuous
#
#        'idade', 'ACS altura', 'ACS peso'
#
#        Discrete and categorical
#
#        'tipo pedido anestesia',  'género', 'proveniência',
#        'motivo admissão UCI', 'tipo cirurgia', 'especialidade_COD',
#        '1ª Cirurgia IPO', 'QT pré-operatória', 'ASA', 'PP cardíaco',
#        'PP respiratório', 'PP ECG', 'PP pressão arterial sistólica',
#        'PP pulsação arterial', 'PP hemoglobina', 'PP leucócitos', 'PP ureia',
#        'PP sódio', 'PP potássio', 'PP nº procedimentos', 'PP perda sangue',
#        'PP contaminação peritoneal', 'PP estado da malignidade',
#        'ACS estado funcional', 'ACS esteróides', 'ACS ascite',
#        'ACS sépsis sistémica', 'ACS dependente ventilador',
#        'ACS cancro disseminado', 'ACS diabetes', 'ACS hipertensão', 'ACS ICC',
#        'ACS dispneia', 'ACS fumador', 'ACS DPOC',
#        'ACS insuficiência renal aguda',
#        'ARISCAT SpO2 ', 'ARISCAT infeção respiratória último mês',
#        'ARISCAT anemia pré-operativa', 'ARISCAT incisão cirúrgica',
#        'ARISCAT duração cirurgia', 'CHARLSON Doença fígado',
#        'CHARLSON Doença Renal Crónica Moderada a Severa',
#        'CHARLSON Insuficiência Cardíaca', 'CHARLSON Enfarte Miocárdio',
#        'CHARLSON Doença Vascular periférica',
#        'CHARLSON AVC ou Ataque Isquémico Transitório',
#        'CHARLSON Úlcera Péptica'

########################################
###Excel writing in Data_as_read.xlsx###
########################################


    
#Removing NAN from the dataframe
    
inputs=inputs.dropna()
outputs=outputs.iloc[inputs.index,:]
outputs=outputs.reset_index(drop=True).astype(float) 

ignored=ignored.iloc[inputs.index,:]
ignored=ignored.reset_index(drop=True)

stringvariables=stringvariables.iloc[inputs.index,:]
stringvariables=stringvariables.reset_index(drop=True)

#outputs=outputs.reset_index(drop=True).dropna()    #resets the indexing of the array, for proper reference towards input
inputs=inputs.iloc[outputs.index,:].reset_index(drop=True).astype(float)


with pd.ExcelWriter('C:\Documents\MEMec_2015-2020\Tese\Base_Dados\Data_as_read.xlsx') as writer: 
    
    inputs.to_excel(writer,sheet_name='Inputs')
    outputs.to_excel(writer,sheet_name='Outputs')
    stringvariables.to_excel(writer,sheet_name='Strings')
    ignored.to_excel(writer,sheet_name='Ignored')
    zero_variance.to_excel(writer,sheet_name='Variables with variance <= 2%')
#######################################################################


#IPO dataset visualization 

#Histograms

# fig = plt.figure(figsize = (5,5))
# ax=inputs['idade'].plot(kind='hist',figsize=(5,5),fontsize=16)#,title='Age distribution')
# plt.xlabel('Age in years',fontsize=16) 
# plt.ylabel('Frequency',fontsize=16) 
# plt.axis([0, 100,0, 210])

#Pie charts

# fig = plt.figure(figsize = (5,5))
# ax=inputs['especialidade_COD'].value_counts().plot(kind='pie',figsize=(5,5),labels=None
#                                                     , autopct='%1.1f%%', startangle=90,radius=0.9,center=(0.4,0.4),fontsize=16)  #,title='Specialty distribution'

# ax.legend(labels=['Digestive','Head and neck + Oral','Others','Thoracic'],bbox_to_anchor=(0.75,0.75),fontsize=16)
# plt.ylabel(None)

# fig = plt.figure(figsize = (5,5))
# ax=inputs['género'].value_counts().plot(kind='pie',figsize=(5,5),labels=None
#                                         , autopct='%1.1f%%', startangle=90,radius=0.9,center=(0.4,0.4),fontsize=16,colors=['#20B2AA','#DC143C']) #,title='Gender distribution in the dataset' 
# ax.legend(labels=['Male','Female'],bbox_to_anchor=(0.9,1),fontsize=16)
# plt.ylabel(None)

# fig = plt.figure(figsize = (5,5))
# ax=outputs['complicação pós-cirúrgica'].value_counts().plot(kind='pie',figsize=(5,5),labels=None
#                                                             , autopct='%1.1f%%', startangle=90,radius=0.9,center=(0.4,0.4),fontsize=16) #,title='Existence of complications after the procedure' 
# ax.legend(labels=['No complication','Complication'],bbox_to_anchor=(0.9,1),fontsize=16)
# plt.ylabel(None)

# fig = plt.figure(figsize = (5,5))
# ax=outputs['óbito até 1 ano após cirurgia'].value_counts().plot(kind='pie',figsize=(5,5),labels=None
#                                                                 , autopct='%1.1f%%', startangle=90,radius=0.9,center=(0.4,0.4),fontsize=16,colors=['#EEE8AA','#DA70D6'])  #,title='One-year Mortality'
# ax.legend(labels=['No death','Death'],bbox_to_anchor=(0.9,1),fontsize=16)
# plt.ylabel(None)

#Scatter plots

    #Complications

# fig = plt.figure(figsize= (7,7))
# ax=sns.scatterplot(data=pd.concat([inputs[['idade','ACS peso']],outputs['complicação pós-cirúrgica']],axis=1),x='idade',y='ACS peso',hue='complicação pós-cirúrgica',style='complicação pós-cirúrgica')
# plt.xlabel('Age',fontsize=16)
# plt.ylabel('Weight',fontsize=16)
# #plt.title('Age vs Weight and Complications')
# handles, garbage =  ax.get_legend_handles_labels()
# ax.legend(handles,['No complication','Complication'],fontsize=16)

# fig = plt.figure(figsize= (7,7))
# ax=sns.scatterplot(data=pd.concat([inputs[['ACS altura','ACS peso']],outputs['complicação pós-cirúrgica']],axis=1),x='ACS altura',y='ACS peso',hue='complicação pós-cirúrgica',style='complicação pós-cirúrgica')
# plt.xlabel('Height',fontsize=16)
# plt.ylabel('Weight',fontsize=16)
# #plt.title('Height vs Weight and Complications')
# handles, garbage =  ax.get_legend_handles_labels()
# ax.legend(handles,['No complication','Complication'],fontsize=16)


# data=pd.concat([inputs[['proveniência','ACS peso']],outputs['complicação pós-cirúrgica']],axis=1).sort_values(by=['proveniência']).round(0).astype(int)
# data['proveniência']=data['proveniência'].astype(str)

# fig = plt.figure(figsize= (5,5))
# ax=sns.scatterplot(data=data,x='proveniência',y='ACS peso',hue='complicação pós-cirúrgica',style='complicação pós-cirúrgica')
# plt.xlabel('Patient Origin',fontsize=16)
# plt.ylabel('Weight',fontsize=16)
# #plt.title('Incoming site vs Weight and Complications')
# ax.legend(handles,['No complication','Complication'],fontsize=16)


# data=pd.concat([inputs[['motivo admissão UCI','ACS peso']],outputs['complicação pós-cirúrgica']],axis=1).sort_values(by=['motivo admissão UCI']).round(0).astype(int)
# data['motivo admissão UCI']=data['motivo admissão UCI'].astype(str)

# fig = plt.figure(figsize= (5,5))
# ax=sns.scatterplot(data=data,x='motivo admissão UCI',y='ACS peso',hue='complicação pós-cirúrgica',style='complicação pós-cirúrgica')
# plt.xlabel('Reason for ICU admission',fontsize=16)
# plt.ylabel('Weight',fontsize=16)
# #plt.title('Reason of admission vs Weight and Complications')
# handles, garbage =  ax.get_legend_handles_labels()
# ax.legend(handles,['No complication','Complication'],fontsize=16)


# data=pd.concat([inputs[['tipo cirurgia','ACS peso']],outputs['complicação pós-cirúrgica']],axis=1).sort_values(by=['tipo cirurgia']).round(0).astype(int)
# data['tipo cirurgia']=data['tipo cirurgia'].astype(str)

# fig = plt.figure(figsize= (5,5))
# ax=sns.scatterplot(data=data,x='tipo cirurgia',y='ACS peso',hue='complicação pós-cirúrgica',style='complicação pós-cirúrgica')
# plt.xlabel('Surgery type',fontsize=16)
# plt.ylabel('Weight',fontsize=16)
# #plt.title('Surgery Type vs Weight and Complications')
# handles, garbage =  ax.get_legend_handles_labels()
# ax.legend(handles,['No complication','Complication'],fontsize=16)


# data=pd.concat([inputs[['PP respiratório','ACS peso']],outputs['complicação pós-cirúrgica']],axis=1).sort_values(by=['PP respiratório']).round(0).astype(int)
# data['PP respiratório']=data['PP respiratório'].astype(str)

# fig = plt.figure(figsize= (5,5))
# ax=sns.scatterplot(data=data,x='PP respiratório',y='ACS peso',hue='complicação pós-cirúrgica',style='complicação pós-cirúrgica')
# plt.xlabel('Shortness of breath',fontsize=16)
# plt.ylabel('Weight',fontsize=16)
# #plt.title('Shortness of breath vs Weight and Complications')
# handles, garbage =  ax.get_legend_handles_labels()
# ax.legend(handles,['No complication','Complication'],fontsize=16)


# data=pd.concat([inputs[['PP hemoglobina','ACS peso']],outputs['complicação pós-cirúrgica']],axis=1).sort_values(by=['PP hemoglobina']).round(0).astype(int)
# data['PP hemoglobina']=data['PP hemoglobina'].astype(str)

# fig = plt.figure(figsize= (5,5))
# ax=sns.scatterplot(data=data,x='PP hemoglobina',y='ACS peso',hue='complicação pós-cirúrgica',style='complicação pós-cirúrgica')
# plt.xlabel('Hemoglobin',fontsize=16)
# plt.ylabel('Weight',fontsize=16)
# #plt.title('Hemoglobin vs Weight and Complications')
# handles, garbage =  ax.get_legend_handles_labels()
# ax.legend(handles,['No complication','Complication'],fontsize=16)

#     #Death

# fig = plt.figure(figsize= (7,7))
# ax=sns.scatterplot(data=pd.concat([inputs[['idade','ACS peso']],outputs['óbito até 1 ano após cirurgia']],axis=1),x='idade',y='ACS peso',hue='óbito até 1 ano após cirurgia',style='óbito até 1 ano após cirurgia',palette=['#EEE8AA','#DA70D6'])
# plt.xlabel('Age',fontsize=16)
# plt.ylabel('Weight',fontsize=16)
# # plt.title('Age vs Weight and Death')
# handles, garbage =  ax.get_legend_handles_labels()
# ax.legend(handles,['No death','Death'],fontsize=16)

# fig = plt.figure(figsize= (7,7))
# ax=sns.scatterplot(data=pd.concat([inputs[['ACS altura','ACS peso']],outputs['óbito até 1 ano após cirurgia']],axis=1),x='ACS altura',y='ACS peso',hue='óbito até 1 ano após cirurgia',style='óbito até 1 ano após cirurgia',palette=['#EEE8AA','#DA70D6'])
# plt.xlabel('Height',fontsize=16)
# plt.ylabel('Weight',fontsize=16)
# # plt.title('Height vs Weight and Death')
# handles, garbage =  ax.get_legend_handles_labels()
# ax.legend(handles,['No death','Death'],fontsize=16)


# data=pd.concat([inputs[['proveniência','ACS peso']],outputs['óbito até 1 ano após cirurgia']],axis=1).sort_values(by=['proveniência']).round(0).astype(int)
# data['proveniência']=data['proveniência'].astype(str)

# fig = plt.figure(figsize= (5,5))
# ax=sns.scatterplot(data=data,x='proveniência',y='ACS peso',hue='óbito até 1 ano após cirurgia',style='óbito até 1 ano após cirurgia',palette=['#EEE8AA','#DA70D6'])
# plt.xlabel('Patient Origin',fontsize=16)
# plt.ylabel('Weight',fontsize=16)
# # plt.title('Incoming site vs Weight and Death')
# handles, garbage =  ax.get_legend_handles_labels()
# ax.legend(handles,['No death','Death'],fontsize=16)


# data=pd.concat([inputs[['motivo admissão UCI','ACS peso']],outputs['óbito até 1 ano após cirurgia']],axis=1).sort_values(by=['motivo admissão UCI']).round(0).astype(int)
# data['motivo admissão UCI']=data['motivo admissão UCI'].astype(str)

# fig = plt.figure(figsize= (5,5))
# ax=sns.scatterplot(data=data,x='motivo admissão UCI',y='ACS peso',hue='óbito até 1 ano após cirurgia',style='óbito até 1 ano após cirurgia',palette=['#EEE8AA','#DA70D6'])
# plt.xlabel('Reason for ICU admission',fontsize=16)
# plt.ylabel('Weight',fontsize=16)
# # plt.title('Reason of admission vs Weight and Death')
# handles, garbage =  ax.get_legend_handles_labels()
# ax.legend(handles,['No death','Death'],fontsize=16)


# data=pd.concat([inputs[['tipo cirurgia','ACS peso']],outputs['óbito até 1 ano após cirurgia']],axis=1).sort_values(by=['tipo cirurgia']).round(0).astype(int)
# data['tipo cirurgia']=data['tipo cirurgia'].astype(str)

# fig = plt.figure(figsize= (5,5))
# ax=sns.scatterplot(data=data,x='tipo cirurgia',y='ACS peso',hue='óbito até 1 ano após cirurgia',style='óbito até 1 ano após cirurgia',palette=['#EEE8AA','#DA70D6'])
# plt.xlabel('Surgery type',fontsize=16)
# plt.ylabel('Weight',fontsize=16)
# # plt.title('Surgery Type vs Weight and Death')
# handles, garbage =  ax.get_legend_handles_labels()
# ax.legend(handles,['No death','Death'],fontsize=16)


# data=pd.concat([inputs[['PP respiratório','ACS peso']],outputs['óbito até 1 ano após cirurgia']],axis=1).sort_values(by=['PP respiratório']).round(0).astype(int)
# data['PP respiratório']=data['PP respiratório'].astype(str)

# fig = plt.figure(figsize= (5,5))
# ax=sns.scatterplot(data=data,x='PP respiratório',y='ACS peso',hue='óbito até 1 ano após cirurgia',style='óbito até 1 ano após cirurgia',palette=['#EEE8AA','#DA70D6'])
# plt.xlabel('Shortness of breath',fontsize=16)
# plt.ylabel('Weight',fontsize=16)
# # plt.title('Shortness of breath vs Weight and Death')
# handles, garbage =  ax.get_legend_handles_labels()
# ax.legend(handles,['No death','Death'],fontsize=16)


# data=pd.concat([inputs[['PP hemoglobina','ACS peso']],outputs['óbito até 1 ano após cirurgia']],axis=1).sort_values(by=['PP hemoglobina']).round(0).astype(int)
# data['PP hemoglobina']=data['PP hemoglobina'].astype(str)

# fig = plt.figure(figsize= (5,5))
# ax=sns.scatterplot(data=data,x='PP hemoglobina',y='ACS peso',hue='óbito até 1 ano após cirurgia',style='óbito até 1 ano após cirurgia',palette=['#EEE8AA','#DA70D6'])
# plt.xlabel('Hemoglobin',fontsize=16)
# plt.ylabel('Weight',fontsize=16)
# # plt.title('Hemoglobin vs Weight and Death')
# handles, garbage =  ax.get_legend_handles_labels()
# ax.legend(handles,['No death','Death'],fontsize=16)


#################################################################
#Pairplots

# fig = plt.figure(figsize= (5,5))
# ax=sns.pairplot(pd.concat([inputs[['idade', 'ACS altura', 'ACS peso']],outputs['complicação pós-cirúrgica']],axis=1),hue='complicação pós-cirúrgica')
# ax._legend.remove()
# plt.legend(labels=['No complication','Complication'],title='Presence of complication',bbox_to_anchor=(0.85,0.6))
# ax.fig.suptitle('Pairplots between Age, height and weight with complications analysis',y=1.1,fontsize=16)

# fig = plt.figure(figsize= (5,5))
# ax=sns.pairplot(pd.concat([inputs[['idade', 'ACS altura', 'ACS peso']],outputs['óbito até 1 ano após cirurgia']],axis=1),hue='óbito até 1 ano após cirurgia')
# ax._legend.remove()
# plt.legend(labels=['No death','Death'],title='Death in the first year',bbox_to_anchor=(0.85,0.6))
# ax.fig.suptitle('Pairplots between Age, height and weight with death analysis',y=1.1,fontsize=16)

#################################################################
#Population pyramid for complications

# fig, (ax_fem, ax_mal) = plt.subplots(1,2,sharey=True,figsize= (8,5))  
# #fig.suptitle('Population Pyramid of Complications',fontsize=16)
# ax_fem.set_title('Female',fontsize=13)
# ax_mal.set_title('Male',fontsize=13)
# ax_mal.set_xlabel('Count',fontsize=13)

# ax_fem.set_ylabel('Age',fontsize=13)
# ax_fem.set_xlabel('Count',fontsize=13)
# ax_fem.yaxis.set_ticks_position("right")
# ax_fem.invert_xaxis()


# females=pd.concat([inputs[inputs['género']==1],outputs[inputs['género']==1]],axis=1)
# females=females[['idade','complicação pós-cirúrgica']].reset_index().drop('index',1)

# males=pd.concat([inputs[inputs['género']==2],outputs[inputs['género']==2]],axis=1)
# males=males[['idade','complicação pós-cirúrgica']].reset_index().drop('index',1)

# bin_ranges=[0,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,]

# sns.histplot(ax=ax_fem,data=females, y='idade', hue='complicação pós-cirúrgica', multiple="stack",bins=bin_ranges,legend=False,kde=True,line_kws={'linewidth': 3})

# sns.histplot(ax=ax_mal,data=males, y='idade', hue='complicação pós-cirúrgica', multiple="stack",bins=bin_ranges,kde=True,line_kws={'linewidth': 3})

# plt.legend(labels=['Complication','No complication'],fontsize=13,bbox_to_anchor=(0.2,0.2))

# ax_fem.set_xlim([80,0])
# ax_mal.set_xlim([0,80])
# ax_fem.set_ylim([0,100])

###################################################################
#Population pyramid for mortality


# fig, (ax_fem, ax_mal) = plt.subplots(1,2,sharey=True,figsize= (8,5))  
# #fig.suptitle('Population Pyramid of Mortality',fontsize=13)
# ax_fem.set_title('Female',fontsize=13)
# ax_mal.set_title('Male',fontsize=13)
# ax_mal.set_xlabel('Count',fontsize=13)

# ax_fem.set_ylabel('Age',fontsize=13)
# ax_fem.set_xlabel('Count',fontsize=13)
# ax_fem.yaxis.set_ticks_position("right")
# ax_fem.invert_xaxis()


# females=pd.concat([inputs[inputs['género']==1],outputs[inputs['género']==1]],axis=1)
# females=females[['idade','óbito até 1 ano após cirurgia']].reset_index().drop('index',1)

# males=pd.concat([inputs[inputs['género']==2],outputs[inputs['género']==2]],axis=1)
# males=males[['idade','óbito até 1 ano após cirurgia']].reset_index().drop('index',1)

# bin_ranges=[0,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96]

# sns.histplot(ax=ax_fem,data=females, y='idade', hue='óbito até 1 ano após cirurgia', multiple="stack",bins=bin_ranges,legend=False,kde=True,line_kws={'linewidth': 3},palette=['#EEE8AA','#DA70D6'])

# sns.histplot(ax=ax_mal,data=males, y='idade', hue='óbito até 1 ano após cirurgia', multiple="stack",bins=bin_ranges,kde=True,line_kws={'linewidth': 3},palette=['#EEE8AA','#DA70D6'])


# plt.legend(labels=['Death','No death'],fontsize=13,bbox_to_anchor=(0.2,0.2))

# ax_fem.set_xlim([80,0])
# ax_mal.set_xlim([0,80])
# ax_fem.set_ylim([0,100])


# ######################################################################
from sklearn import preprocessing as pre
from scipy import stats

#Pre processing with min-max
indexes_to_keep=(np.abs(stats.zscore(inputs[['idade', 'ACS altura', 'ACS peso']])) < 3).all(axis=1)
inputs=inputs[indexes_to_keep].reset_index()
ignored=ignored[indexes_to_keep].reset_index()
stringvariables=stringvariables[indexes_to_keep].reset_index()
outputs=outputs[indexes_to_keep].reset_index()

inputs_pristine=inputs.drop('index',1)

#Fitting the data into categorical bins [0-3]
inputs_pristine['idade']=pd.qcut(inputs_pristine['idade'],q=[0,0.4,0.6,0.8,1],labels=False,precision=1)
inputs_pristine['ACS altura']=pd.qcut(inputs_pristine['ACS altura'],q=[0,0.4,0.6,0.8,1],labels=False,precision=1)
inputs_pristine['ACS peso']=pd.qcut(inputs_pristine['ACS peso'],q=[0,0.4,0.6,0.8,1],labels=False,precision=1)

inputs['idade']=pd.qcut(inputs['idade'],q=[0,0.4,0.6,0.8,1],labels=False,precision=1)
inputs['ACS altura']=pd.qcut(inputs['ACS altura'],q=[0,0.4,0.6,0.8,1],labels=False,precision=1)
inputs['ACS peso']=pd.qcut(inputs['ACS peso'],q=[0,0.4,0.6,0.8,1],labels=False,precision=1)

temporary = inputs.values #returns a numpy array
min_max_scaler = pre.MinMaxScaler()
temporary_scaled = min_max_scaler.fit_transform(temporary)
inputs_normal = pd.DataFrame(temporary_scaled,columns=inputs.columns)  #All normalized now

inputs=inputs_normal.drop('index',1)


#####################################
#Normalized data
#####################################

with pd.ExcelWriter('C:\Documents\MEMec_2015-2020\Tese\Base_Dados\Data_as_read_normalized.xlsx') as writer: 
    
    inputs.to_excel(writer,sheet_name='Inputs')
    outputs.to_excel(writer,sheet_name='Outputs')
    stringvariables.to_excel(writer,sheet_name='Strings')
    ignored.to_excel(writer,sheet_name='Ignored')
    zero_variance.to_excel(writer,sheet_name='Variables with variance <= 2%')
#######################################################################




