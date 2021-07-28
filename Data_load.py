#This is the code that read sthe database contents and separates inputs to outputs
##
##
#Python 3.7 by André Miranda @ andre.lima.miranda@tecnico.ulisboa.pt

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import pandas as pd


Location=r'C:\Documents\MEMec_2015-2020\Tese\Base_Dados\Database_without_scores_outputs.csv'

alldata=pd.read_csv(Location, sep = ';', encoding='latin-1')

df1=pd.DataFrame(alldata)       #Data structure with all features apart from NAS points and scores results


#['data pedido pela anestesia', 'tipo pedido anestesia',
#       'data admissão UCI', 'numero IPO', 'idade', 'género', 'proveniência',
#       'motivo admissão UCI', 'tipo cirurgia', 'especialidade',
#       'especialidade_COD', 'dias na UCI', 'dias no  IPOP', 'destino após UCI',
#       'Primeira Cirurgia IPO', 'QT pré-operatória', 'reinternamento na UCI',
#       'ASA', 'LOCALIZAÇÃO ', 'diagnóstico pré-operatório', 'data cirurgia',
#       'Operação efetuada', 'Intervenções_ICD10', 'procedimentos_COD',
#       'PP idade', 'PP cardíaco', 'PP respiratório', 'PP ECG',
#       'PP pressão arterial sistólica', 'PP pulsação arterial',
#       'PP hemoglobina', 'PP leucócitos', 'PP ureia', 'PP sódio',
#       'PP potássio', 'PP escala glasglow', 'PP tipo operação',
#       'PP numero procedimentos', 'PP perda sangue',
#       'PP contaminação peritoneal', 'PP estado da malignidade',
#       'PP CEPOD-classificação operação', 'ACS_procedimento', 'ACS idade',
#       'ACS género', 'ACS estado funcional', 'ACS emergência', 'ACS ASA',
#       'ACS esteróides', 'ACS ascite', 'ACS sépsis sistémica',
#       'ACS dependente ventilador', 'ACS cancro disseminado', 'ACS diabetes',
#       'ACS hipertensão', 'ACS ICC', 'ACS dispneia', 'ACS fumador', 'ACS DPOC',
#       'ACS diálise', 'ACS insuficiência renal aguda', 'ACS altura',
#       'ACS peso', 'ARISCAT Idade', 'ARISCAT SpO2 ',
#       'ARISCAT infeção respiratória último mês',
#       'ARISCAT anemia pré-operativa', 'ARISCAT incisão cirúrgica',
#       'ARISCAT duração cirurgia', 'ARISCAT procedimento emergente',
#       'complicação pós-cirúrgica', 'descrição complicação pós-cirúrgica',
#       'complicação_COD', 'complicação principal_COD',
#       'classificação ACS complicações gerais',
#       'classificação ACS complicações específicas',
#       'classificação clavien-dindo', 'destino após IPO', 'óbito até 1 ano',
#       'óbito_tempo decorrido após data cirurgia (até 1 ano)', 'data óbito',
#       'Informação adicional', 'Comorbilidades pré-operatórias']

# print()
# print('Database read!!')
# print()
########################
###Inputs and Outputs###
########################

#Ignored data

ignored=df1[['numero IPO',
             'dias no  IPOP',
             'data cirurgia',
             'PP idade',
             'ACS idade',
             'classificação ACS complicações específicas',
             'ACS género',
             'ARISCAT Idade',
             'Informação adicional',
             'ARISCAT duração cirurgia',
             'motivo admissão UCI',
             'ASA',
             'PP CEPOD-classificação operação',
             'ACS emergência',
             'ARISCAT procedimento emergente',
             'PP escala glasglow',
             'proveniência',
             'dias no  IPOP',
             'procedimentos_COD',
             'PP tipo operação']]


#Variables with strings
stringvariables=df1[['data pedido pela anestesia',
            'data admissão UCI',
            'especialidade',
            'LOCALIZAÇÃO ', 'diagnóstico pré-operatório', 'Operação efetuada',
            'Intervenções_ICD10', 'ACS_procedimento',
            'descrição complicação pós-cirúrgica', 'complicação_COD',
            'data óbito',
            'Comorbilidades pré-operatórias']]


# print()
# print('Stringed variables separated!!')
# print()

#Treatment of some stringed variables
#ICD10
import re
import unidecode

all_procedures_ICD10=[]

for i in range(stringvariables.shape[0]):
    s = stringvariables.iloc[i,6]
    s=re.sub(r'[:-}]+', '', unidecode.unidecode(s))
    s=re.sub(r'\(', '', unidecode.unidecode(s))
    s=re.sub(r'\)', '', unidecode.unidecode(s))
    s=re.sub(r'\.', '', unidecode.unidecode(s))
    s=re.sub(r'\-', '', unidecode.unidecode(s))   
    s=re.sub(r'/', '', unidecode.unidecode(s))
    s=re.sub(r',', '', unidecode.unidecode(s))
    s=re.sub(r' ', '', unidecode.unidecode(s))
    s=re.sub(r'\n', '_', unidecode.unidecode(s)) 
    stringvariables.iloc[i,6]=re.sub(r'\+', '_', unidecode.unidecode(s))
    all_procedures_ICD10=np.hstack([all_procedures_ICD10, stringvariables.iloc[i,6].split("_")])


#Outputs

outputs=df1[['dias na UCI', 'destino após UCI', 'complicação pós-cirúrgica',
            'complicação principal_COD', 'classificação ACS complicações gerais',
            
            'classificação clavien-dindo', 'destino após IPO', 'óbito até 1 ano',
            'óbito_tempo decorrido após data cirurgia (até 1 ano)','reinternamento na UCI' ]]


# print()
# print('Output variables separated!!')
# print()


#Inputs (everything else)


prepreinputs=df1.drop(stringvariables.columns, axis=1)
preinputs=prepreinputs.drop(ignored.columns,axis=1)    #Variables that are repeated or not relevant
inputs=preinputs.drop(outputs.columns,axis=1)


# print()
# print('Input variables separated!!')
# print()

########################################
###Excel writing in Data_as_read.xlsx###
########################################

with pd.ExcelWriter('C:\Documents\MEMec_2015-2020\Tese\Base_Dados\Data_as_read.xlsx') as writer: 
    
    inputs.to_excel(writer,sheet_name='Inputs')
    outputs.to_excel(writer,sheet_name='Outputs')
    stringvariables.to_excel(writer,sheet_name='Strings')
    ignored.to_excel(writer,sheet_name='Ignored')