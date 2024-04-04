#%%conda o pip
import os
import numpy as np
#graficadores
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#dataframe
import pandas as pd
from pathlib import Path
import datetime as dt
import math
import warnings
from collections import Counter
import openpyxl
#entrenar modelos estadistico
import scipy
import scipy.stats as ss
import scipy.cluster.hierarchy as sch
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.formula.api import logit
#solo en pip es un graficador
import plot_likert

#%% open dataset 
# Set working directory file cell
cwd = Path.cwd()
file_I = Path("Data/Data_I.csv")
file_collabels = Path("Data/Colnames.csv")
file_defunciones = Path("Data/Defunciones.csv")
file_confirmados = Path("Data/Confirmados.csv")
#read files
data_I_HGM = pd.read_csv(cwd / file_I)
data_collabels = pd.read_csv(cwd / file_collabels)
data_defunciones = pd.read_csv(cwd / file_defunciones)
data_confirmados = pd.read_csv(cwd / file_confirmados)

#%% filter rows with more than one answer in hesitanci as 0
columnset = np.r_[0,40:47]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_pivot = dataset_test.melt('R0').pivot_table(index='R0', columns='value', aggfunc='size', fill_value=0) 
dataset_pivot = dataset_pivot.reset_index()
truelist = dataset_pivot[0] <= 1
data_I_HGM = data_I_HGM.loc[truelist,:] 

#%% Change date columns to the proper type
data_I_HGM['R2'] = pd.to_datetime(data_I_HGM['R2'], format='%d/%m/%Y')
data_I_HGM['R12'] = pd.to_datetime(data_I_HGM['R12'], format='%d/%m/%Y')
data_defunciones['Date'] = pd.to_datetime(data_defunciones['Date'], format='%d/%m/%Y')
data_confirmados['Date'] = pd.to_datetime(data_confirmados['Date'], format='%d/%m/%Y')

#%%filter by time
data_I_HGM = data_I_HGM[data_I_HGM.R2.between('2021-03', '2021-10')]

#%% Add confirmados y defunciones ass colums in data_I_HGM
data_I_HGM['R168'] = np.NAN
data_collabels.loc[len(data_collabels.index)] = ["R168","DefuncionesNacionales"]
data_I_HGM['R169'] = np.NAN
data_collabels.loc[len(data_collabels.index)] = ["R169","ConfirmadosNacionales"]
for index, row in data_I_HGM.iterrows():
    data_I_HGM.loc[index,'R168'] = data_defunciones.loc[data_defunciones['Date'] == row['R2'],'Nacional'].values
    data_I_HGM.loc[index,'R169'] = data_confirmados.loc[data_confirmados['Date'] == row['R2'],'Nacional'].values

#%% create groups of data 
# age groups
CutAge = pd.cut(data_I_HGM.R13, bins=[18, 23, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73, np.inf])
data_I_HGM['R170']=CutAge
data_collabels.loc[len(data_collabels.index)] = ["R170","CutAge"]
# date groups
time_cats = 12
CutTime =  pd.cut(data_I_HGM.R2,time_cats)
data_I_HGM['R171']=CutTime
data_collabels.loc[len(data_collabels.index)] = ["R171","CutDate"]
temp=data_I_HGM.R171.astype(str).str.extract(', (.+?)]').astype('datetime64[ns]')
data_I_HGM['R171c']=pd.to_datetime(temp[0]).dt.date
#education groups
CutEdu = pd.cut(data_I_HGM.R16, bins=[0, 6, 9, 12, 16, 22, np.inf])
data_I_HGM['R172']=CutEdu
data_collabels.loc[len(data_collabels.index)] = ["R172","CutEdu"]
#add media of incertidumbre
data_I_HGM['R173']=data_I_HGM.loc[:,['R70 - R77','R71 - R78', 'R72 - R79', 'R73 - R80', 'R74 - R83','R75 - R82', 'R76 - R81']].mean(axis=1)
data_collabels.loc[len(data_collabels.index)] = ["R173","HesitancyMean"]
#dichotomic media
data_I_HGM['R174'] = 0
data_I_HGM.loc[data_I_HGM['R173'] >= data_I_HGM.R173.quantile(0.75),'R174'] = 1

#%%dummies to categorical error no funciona
data_I_HGM['R175']=data_I_HGM.iloc[:,10:17].idxmax(axis=1)
data_collabels.loc[len(data_collabels.index)] = ["R175","SeguridadSocial"]
data_I_HGM['R176']=data_I_HGM.iloc[:,17:30].idxmax(axis=1)
data_collabels.loc[len(data_collabels.index)] = ["R176","EnfermedadReuma"]
data_I_HGM['R177']=data_I_HGM.iloc[:,73:84].idxmax(axis=1)
data_collabels.loc[len(data_collabels.index)] = ["R177","Cormo"]


#%%clean columns
data_I_HGM['R70 - R77c'] = data_I_HGM['R70 - R77'].map({0:"No sé", 5:"Definitivamente no", 4:"Probablemente no", 3:"Tal vez si o tal vez no", 2:"Probablemente", 1:"Seguramente"})
data_I_HGM['R71 - R78c'] = data_I_HGM['R71 - R78'].map({0:"No sé", 5:"Me negaré a aplicármela", 4:"Pospondré (retrasaré) su aplicación", 3:"No estoy seguro(a) de lo que haré", 2:"Me la aplicaría cuando me la ofrezcan", 1:"Me gustaría aplicármela lo antes posible"})
data_I_HGM['R72 - R79c'] = data_I_HGM['R72 - R79'].map({0:"No sé", 5:"En contra de la vacuna", 4:"Bastante preocupado(a)", 3:"Neutra", 2:"Bastante positiva", 1:"Muy entusiasta"})
data_I_HGM['R73 - R80c'] = data_I_HGM['R73 - R80'].map({0:"No sé", 5:"Nunca me la aplicaría", 4:"Evitaría aplicármela durante el mayor tiempo posible", 3:"Retrasaría su aplicación", 2:"Me la aplicaría cuando tenga tiempo", 1:"Me la aplicaría tan pronto como pueda"})
data_I_HGM['R74 - R83c'] = data_I_HGM['R74 - R83'].map({0:"No sé", 5:"Les sugeriría que no se vacunen", 4:"Les pediría que retrasen la vacuna", 3:"No les diría nada al respecto", 2:"Los animaría", 1:"Los animaría con entusiasmo"})
data_I_HGM['R75 - R82c'] = data_I_HGM['R75 - R82'].map({0:"No sé", 5:"En contra de la vacuna para la COVID-19", 4:"No dispuesto(a) a recibir la vacuna para la COVID-19", 3:"No preocupado(a) por recibir la vacuna para la COVID-19", 2:"Dispuesto(a) a recibir la vacuna para la COVID-19", 1:"Entusiasmado(a), por recibir la vacuna para la COVID-19"})
data_I_HGM['R76 - R81c'] = data_I_HGM['R76 - R81'].map({0:"No sé", 5:"Realmente no es importante", 4:"No es importante", 3:"Ni importante ni no importante", 2:"Importante", 1:"Realmente importante"})
#error no funciona
data_I_HGM['R175'] = data_I_HGM['R175'].map({'R20':'IMSS', 'R19':'Ninguna', 'R25':'Otro', 'R21':'ISSSTE', 'R24':'INSABI', 'R22':'Otro'})
data_I_HGM['R176'] = data_I_HGM['R176'].map({'R31':'Gota', 'R27':'ArtritisReumatoide', 'R38':'Otro', 'R30':'Esclerodermia', 'R33':'MiopatíaInflamatoria', 'R36':'Otro', 'R26':'Lupus', 'R32':'Sjögren', 'R28':'VasculitisANCA','R29':'EspondilitisAnquilosante', 'R34':'AAntifosfolípidos', 'R37':'Osteoartrosis', 'R35':'ArtrititsIdiopáticaJ'})
data_I_HGM['R177'] = data_I_HGM['R177'].map({'R157':'EnfermedadPulmonar', 'R161':'HipertensiónArterialSistémica', 'R165':'Otro', 'R164':'Depresión', 'R159':'OtraCardiovascular', 'R162':'HipertensiónArterialSistémica', 'R167':'OtrasGástricas', 'R160':'VascularCerebral','R163':'Otro', 'R166':'Ulcera', 'R158':'Otro'})

#%%likert scales plot
columnset = np.r_[40:47]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test = dataset_test.replace({0:'None',1: 'Strongly agree', 2:'Agree', 3: 'Neither agree nor disagree', 4:'Disagree', 5:'Strongly disagree'})
another_scale = \
    ['Strongly disagree',
    'Disagree',
    'Neither agree nor disagree',
    'Agree',
    'Strongly agree',
    'None']
colours5 = ['white','#f9665e','#fec9c9','#949494','#95b4cc','#799fcb'] 
plot_likert.plot_likert(dataset_test, another_scale,plot_percentage=True, colors=colours5) #another_scale plot_likert.scales.agree
plt.close()
image_format = 'svg'
file_O = Path("Data/outimg/likert_hesitancy.svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200)
plt.close()

columnset = np.r_[40:47,134]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test = dataset_test[dataset_test['R173'] >= dataset_test.R173.quantile(0.75)]
dataset_test.drop('R173', axis=1, inplace=True)
dataset_test = dataset_test.replace({0:'None',1: 'Strongly agree', 2:'Agree', 3: 'Neither agree nor disagree', 4:'Disagree', 5:'Strongly disagree'})
another_scale = \
    ['Strongly disagree',
    'Disagree',
    'Neither agree nor disagree',
    'Agree',
    'Strongly agree',
    'None']
plot_likert.plot_likert(dataset_test, another_scale,plot_percentage=True, colors=colours5) #another_scale plot_likert.scales.agree
image_format = 'svg'
file_O = Path("Data/outimg/likert_hesitancy_75per.svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200)
plt.close()

columnset = np.r_[40:47]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test=dataset_test[(dataset_test != 0).all(1)]
dataset_test=dataset_test[(dataset_test != 3).all(1)]
dataset_test = dataset_test.replace({1: 'Strongly agree', 2:'Agree', 4:'Disagree', 5:'Strongly disagree'})
another_scale = \
    ['Strongly disagree',
    'Disagree',
    'Agree',
    'Strongly agree']
colours4 = ['white','#f9665e','#fec9c9','#95b4cc','#799fcb'] 
plot_likert.plot_likert(dataset_test, another_scale,plot_percentage=True, colors=colours4) #another_scale plot_likert.scales.agree
image_format = 'svg'
file_O = Path("Data/outimg/likert_hesitancy4v.svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200)
plt.close()

columnset = np.r_[40:47,134]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test = dataset_test[dataset_test['R173'] >= dataset_test.R173.quantile(0.75)]
dataset_test.drop('R173', axis=1, inplace=True)
dataset_test=dataset_test[(dataset_test != 0).all(1)]
dataset_test=dataset_test[(dataset_test != 3).all(1)]
dataset_test = dataset_test.replace({1: 'Strongly agree', 2:'Agree', 4:'Disagree', 5:'Strongly disagree'})
another_scale = \
    ['Strongly disagree',
    'Disagree',
    'Agree',
    'Strongly agree']
plot_likert.plot_likert(dataset_test, another_scale,plot_percentage=True, colors=colours4) #another_scale plot_likert.scales.agree
image_format = 'svg'
file_O = Path("Data/outimg/likert_hesitancy_75per4v.svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200)
plt.close()

#%%factors
columnset = np.r_[47:65,66,69]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test = dataset_test.replace({0:'None',1: 'Strongly agree', 2:'Agree', 3: 'Neither agree nor disagree', 4:'Disagree', 5:'Strongly disagree'})
another_scale = \
    ['Strongly disagree',
    'Disagree',
    'Neither agree nor disagree',
    'Agree',
    'Strongly agree',
    'None']
colours5 = ['white','#f9665e','#fec9c9','#949494','#95b4cc','#799fcb'] 
plot_likert.plot_likert(dataset_test, another_scale,plot_percentage=True, colors=colours5) #another_scale plot_likert.scales.agree
image_format = 'svg'
file_O = Path("Data/outimg/likert_factors.svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200)

#%%final
columnset = np.r_[1,4:7,133,136,8,73:84,17:30,31:34,35,47:49,67:70,139:146]#,138:146]#17:89]#   ,]
dataset_test= data_I_HGM.iloc[:,columnset]
#dataset_test.columns.tolist
dataset_test.rename(columns={"R1":"Hospital", "R13": "Edad", "R14":"Sexo","R15":"Estado civil", "R172":"Educacion", "R17": "Ocupación", "R175":"Seguridad Social", "R176":"Diagnóstico", "R157":"Enfermedad pulmonar","R158":"Infarto agudo de miocardio","R159":"Otra enfermedad cardiovascular","R160":"Enfermedad vascular cerebral","R161":"Hipertensión arterial sistémica","R162":"Diabetes mellitus","R163":"Fractura de cadera/columna o pierna","R164":"Depresión","R165":"Cancer","R166":"Ulcera gastrointestinal","R167":"Otras enfermedades gástrica","R168":"Otro", "R60":"¿Usted requirió hospitalización?","R62":"¿Ya fue vacunado?","R63":"¿Ha tenido la oportunidad de vacunarse?","R65":"¿Cuenta con el esquema completo?","R70 - R77c":"¿Aceptaría alguna de las vacunas si se le ofreciera?","R71 - R78c":"Existen varias vacunas. Usted considera que","R72 - R79c":"Mi actitud con respecto a recibir la vacuna","R73 - R80c":"Si ya estuviera disponible, ¿qué haría?","R74 - R83c":"Si mi familia o amigos estuvieran pensando en vacunarse","R75 - R82c":"Con respecto a recibir la vacuna, yo me describiría como","R76 - R81c":"Recibir una vacuna es","R84":"¿Cree que se infectará?","R85":"Probabilidad que me funcione","R105":"Vacunado influenza","R148":"¿Cómo se siente en este momento?","R149":"Sistema inmune","R26":"Lupus eritematoso generalizado","R27":"Artritis reumatoide","R28":"Vasculitis ANCA","R29":"Espondilitis anquilosante","R30":"Esclerodermia","R31":"Gota","R32":"Síndrome de Sjögren","R33":"Miopatía inflamatoria","R34":"Síndrome de anticuerpos antifosfolípidos","R35":"Artritits idiopática juvenil","R36":"Enfermedad mixta del tejido conectivo","R37":"Osteoartrosis","R38":"Otro"}, inplace = True)
dataset_test.columns.tolist

dataset_test["Edad"].mean()
dataset_test["Edad"].std()
data_I_HGM["R16"].mean()
data_I_HGM["R16"].std()

dataset_test.loc[:, dataset_test.dtypes == 'object'] =\
        dataset_test.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
dataset_test.loc[:, dataset_test.isin([0,1]).all()] = dataset_test.loc[:, dataset_test.isin([0,1]).all()].apply(lambda x: x.astype('category'))
dataset_test.info()
ColNames = dataset_test.columns
ColGrouped = ColNames[0]
ColNames = np.delete(ColNames, [np.r_[0]], 0)
ColTitles = {'CHI', 'HGM', 'MTY', 'NUT'}
CQ  = pd.DataFrame(columns = ['Variable', 'CHI', 'HGM', 'MTY', 'NUT','p'])
temp_totales = dataset_test[ColGrouped].value_counts()
for coltitle in ColTitles:
    CQ.loc['Totales',coltitle] = temp_totales[coltitle]
for cols in ColNames:
    title_data = [cols]
    title = pd.DataFrame(title_data, index=[0])
    if(is_categorical(dataset_test[cols])):
        ctable = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped])
        ctablen = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped], normalize='columns')*100.00
        ctablen = ctablen.applymap(lambda x: " ({:.2f})".format(x))
        stat, pvalue, dof, expected = ss.chi2_contingency(ctable)
        ctable = ctable.astype(str)
        ctable = ctable + ctablen
        if((ctable.index == 0).any()):
            ctable = ctable.drop(0)
            ctable.rename({1:"Sí"}, inplace = True) 
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue) 
        #ctable.loc[ctable.index[0], 'Valor estadistica'] ="{:.2f}".format(stat)
    else:
        dataset_numeric = pd.concat([dataset_test[cols], dataset_test[ColGrouped]], axis=1)
        #correlation, pvalue = ss.pointbiserialr(dataset_test[ColGrouped], dataset_test[cols])
        x = []
        for coltitle in ColTitles:
            x.append(dataset_numeric.loc[dataset_numeric[ColGrouped] == coltitle,cols])
        correlation, pvalue = ss.kruskal(*x)
        ctable = dataset_numeric.groupby(ColGrouped).mean().T
        ctable = ctable.applymap(lambda x: "{:.2f}".format(x))
        ctabled = dataset_numeric.groupby(ColGrouped).std().T
        ctabled = ctabled.applymap(lambda x: " ({:.2f})".format(x))
        ctable = ctable + ctabled
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue)  
    title.rename(columns={0:"Variable"}, inplace = True)
    ctable = title.append(ctable)
    CQ  =  CQ.append(ctable)
CQ = CQ.replace(np.nan,'',regex = True)
file_O = Path("Data/out/crosstab.xlsx")
file_save = cwd / file_O
CQ.to_excel(file_save) 

columnset = np.r_[32,1,4:7,133,136,8,137:139,31:32,33:34,35,47:49,67:70,139:146]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test['R62'] = dataset_test['R62'].map({1:"Sí", 0:"No"})
dataset_test.rename(columns={"R1":"Hospital", "R13": "Edad", "R14":"Sexo","R15":"Estado civil", "R172":"Educacion", "R17": "Ocupación", "R175":"Seguridad Social", "R176":"Diagnóstico", "R177":"Comorbilidad", "R60":"¿Usted requirió hospitalización?","R62":"¿Ya fue vacunado?","R63":"¿Ha tenido la oportunidad de vacunarse?","R65":"¿Cuenta con el esquema completo?","R70 - R77c":"¿Aceptaría alguna de las vacunas si se le ofreciera?","R71 - R78c":"Existen varias vacunas. Usted considera que","R72 - R79c":"Mi actitud con respecto a recibir la vacuna","R73 - R80c":"Si ya estuviera disponible, ¿qué haría?","R74 - R83c":"Si mi familia o amigos estuvieran pensando en vacunarse","R75 - R82c":"Con respecto a recibir la vacuna, yo me describiría como","R76 - R81c":"Recibir una vacuna es","R84":"¿Cree que se infectará?","R85":"Probabilidad que me funcione","R105":"Vacunado influenza","R148":"¿Cómo se siente en este momento?","R149":"Sistema inmune"}, inplace = True)
dataset_test.loc[:, dataset_test.dtypes == 'object'] =\
        dataset_test.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
dataset_test.loc[:, dataset_test.isin([0,1]).all()] = dataset_test.loc[:, dataset_test.isin([0,1]).all()].apply(lambda x: x.astype('category'))
dataset_test.info()
ColNames = dataset_test.columns
ColGrouped = ColNames[0]
ColNames = np.delete(ColNames, [np.r_[0]], 0)
ColTitles = {'Sí', 'No'}
CQ  = pd.DataFrame(columns = ['Variable', 'Sí', 'No','p'])
temp_totales = dataset_test[ColGrouped].value_counts()
for coltitle in ColTitles:
    CQ.loc['Totales',coltitle] = temp_totales[coltitle]
for cols in ColNames:
    title_data = [cols]
    title = pd.DataFrame(title_data, index=[0])
    if(is_categorical(dataset_test[cols])):
        ctable = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped])
        ctablen = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped], normalize='columns')*100.00
        ctablen = ctablen.applymap(lambda x: " ({:.2f})".format(x))
        stat, pvalue, dof, expected = ss.chi2_contingency(ctable)
        ctable = ctable.astype(str)
        ctable = ctable + ctablen
        if((ctable.index == 0).any()):
            ctable = ctable.drop(0)
            ctable.rename({1:"Sí"}, inplace = True) 
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue) 
        #ctable.loc[ctable.index[0], 'Valor estadistica'] ="{:.2f}".format(stat)
    else:
        dataset_numeric = pd.concat([dataset_test[cols], dataset_test[ColGrouped]], axis=1)
        #correlation, pvalue = ss.pointbiserialr(dataset_test[ColGrouped], dataset_test[cols])
        x = []
        for coltitle in ColTitles:
            x.append(dataset_numeric.loc[dataset_numeric[ColGrouped] == coltitle,cols])
        correlation, pvalue = ss.kruskal(*x)
        ctable = dataset_numeric.groupby(ColGrouped).mean().T
        ctable = ctable.applymap(lambda x: "{:.2f}".format(x))
        ctabled = dataset_numeric.groupby(ColGrouped).std().T
        ctabled = ctabled.applymap(lambda x: " ({:.2f})".format(x))
        ctable = ctable + ctabled
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue)  
    title.rename(columns={0:"Variable"}, inplace = True)
    ctable = title.append(ctable)
    CQ  =  CQ.append(ctable)
CQ = CQ.replace(np.nan,'',regex = True)
file_O = Path("Data/out/crosstab.xlsx")
file_save = cwd / file_O
CQ.to_excel(file_save) 

columnset = np.r_[40:47]
dataset_dependents= data_I_HGM.iloc[:,columnset]
list_del = (dataset_dependents != 0).all(1)
dataset_dependents = dataset_dependents.replace({2:1, 3:0, 4:0, 5:0})
columnset = np.r_[1,4:7,133,136,8,137:139,31:34,35,47:49,67:70,139:146]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test.rename(columns={"R1":"Hospital", "R13": "Edad", "R14":"Sexo","R15":"Estado civil", "R172":"Educacion", "R17": "Ocupación", "R175":"Seguridad Social", "R176":"Diagnóstico", "R177":"Comorbilidad", "R60":"¿Usted requirió hospitalización?","R62":"¿Ya fue vacunado?","R63":"¿Ha tenido la oportunidad de vacunarse?","R65":"¿Cuenta con el esquema completo?","R70 - R77c":"¿Aceptaría alguna de las vacunas si se le ofreciera?","R71 - R78c":"Existen varias vacunas. Usted considera que","R72 - R79c":"Mi actitud con respecto a recibir la vacuna","R73 - R80c":"Si ya estuviera disponible, ¿qué haría?","R74 - R83c":"Si mi familia o amigos estuvieran pensando en vacunarse","R75 - R82c":"Con respecto a recibir la vacuna, yo me describiría como","R76 - R81c":"Recibir una vacuna es","R84":"¿Cree que se infectará?","R85":"Probabilidad que me funcione","R105":"Vacunado influenza","R148":"¿Cómo se siente en este momento?","R149":"Sistema inmune"}, inplace = True)
dataset_test.loc[:, dataset_test.dtypes == 'object'] =\
        dataset_test.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
dataset_test.loc[:, dataset_test.isin([0,1]).all()] = dataset_test.loc[:, dataset_test.isin([0,1]).all()].apply(lambda x: x.astype('category'))
dataset_test.info()
CBin = (dataset_dependents.sum(axis=1)>=4)*1
CBin= CBin.map({1:"Sí", 0:"No"})
dataset_test.insert(0, 'Aceptación', CBin)
ColNames = dataset_test.columns
ColGrouped = ColNames[0]
ColNames = np.delete(ColNames, [np.r_[0]], 0)
ColTitles = {'Sí', 'No'}
CQ  = pd.DataFrame(columns = ['Variable', 'Sí', 'No','p'])
dataset_test = dataset_test[list_del]
temp_totales = dataset_test[ColGrouped].value_counts()
for coltitle in ColTitles:
    CQ.loc['Totales',coltitle] = temp_totales[coltitle]
for cols in ColNames:
    title_data = [cols]
    title = pd.DataFrame(title_data, index=[0])
    if(is_categorical(dataset_test[cols])):
        ctable = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped])
        ctablen = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped], normalize='columns')*100.00
        ctablen = ctablen.applymap(lambda x: " ({:.2f})".format(x))
        stat, pvalue, dof, expected = ss.chi2_contingency(ctable)
        ctable = ctable.astype(str)
        ctable = ctable + ctablen
        if((ctable.index == 0).any()):
            ctable = ctable.drop(0)
            ctable.rename({1:"Sí"}, inplace = True) 
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue) 
        #ctable.loc[ctable.index[0], 'Valor estadistica'] ="{:.2f}".format(stat)
    else:
        dataset_numeric = pd.concat([dataset_test[cols], dataset_test[ColGrouped]], axis=1)
        #correlation, pvalue = ss.pointbiserialr(dataset_test[ColGrouped], dataset_test[cols])
        x = []
        for coltitle in ColTitles:
            x.append(dataset_numeric.loc[dataset_numeric[ColGrouped] == coltitle,cols])
        correlation, pvalue = ss.kruskal(*x)
        ctable = dataset_numeric.groupby(ColGrouped).mean().T
        ctable = ctable.applymap(lambda x: "{:.2f}".format(x))
        ctabled = dataset_numeric.groupby(ColGrouped).std().T
        ctabled = ctabled.applymap(lambda x: " ({:.2f})".format(x))
        ctable = ctable + ctabled
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue)  
    title.rename(columns={0:"Variable"}, inplace = True)
    ctable = title.append(ctable)
    CQ  =  CQ.append(ctable)
CQ = CQ.replace(np.nan,'',regex = True)
file_O = Path("Data/out/crosstab.xlsx")
file_save = cwd / file_O
CQ.to_excel(file_save)

columnsetsp = np.r_[46]
dataset_dependents= data_I_HGM.iloc[:,columnsetsp]
dataset_dependents = dataset_dependents.replace({2:1, 3:0, 4:0, 5:0})
columnset = np.r_[1,4:7,133,136,8,137:139,31:34,35,47:49,67:70,139:146]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test.rename(columns={"R1":"Hospital", "R13": "Edad", "R14":"Sexo","R15":"Estado civil", "R172":"Educacion", "R17": "Ocupación", "R175":"Seguridad Social", "R176":"Diagnóstico", "R177":"Comorbilidad", "R60":"¿Usted requirió hospitalización?","R62":"¿Ya fue vacunado?","R63":"¿Ha tenido la oportunidad de vacunarse?","R65":"¿Cuenta con el esquema completo?","R70 - R77c":"¿Aceptaría alguna de las vacunas si se le ofreciera?","R71 - R78c":"Existen varias vacunas. Usted considera que","R72 - R79c":"Mi actitud con respecto a recibir la vacuna","R73 - R80c":"Si ya estuviera disponible, ¿qué haría?","R74 - R83c":"Si mi familia o amigos estuvieran pensando en vacunarse","R75 - R82c":"Con respecto a recibir la vacuna, yo me describiría como","R76 - R81c":"Recibir una vacuna es","R84":"¿Cree que se infectará?","R85":"Probabilidad que me funcione","R105":"Vacunado influenza","R148":"¿Cómo se siente en este momento?","R149":"Sistema inmune"}, inplace = True)
dataset_test.loc[:, dataset_test.dtypes == 'object'] =\
        dataset_test.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
dataset_test.loc[:, dataset_test.isin([0,1]).all()] = dataset_test.loc[:, dataset_test.isin([0,1]).all()].apply(lambda x: x.astype('category'))
dataset_test.info()
CBin = dataset_dependents.iloc[:,0]
CBin= CBin.map({1:"Sí", 0:"No"})
dataset_test.insert(0, 'Aceptación', CBin)
ColNames = dataset_test.columns
ColGrouped = ColNames[0]
ColNames = np.delete(ColNames, [np.r_[0]], 0)
ColTitles = {'Sí', 'No'}
CQ  = pd.DataFrame(columns = ['Variable', 'Sí', 'No','p'])
dataset_test = dataset_test[list_del]
temp_totales = dataset_test[ColGrouped].value_counts()
for coltitle in ColTitles:
    CQ.loc['Totales',coltitle] = temp_totales[coltitle]
for cols in ColNames:
    title_data = [cols]
    title = pd.DataFrame(title_data, index=[0])
    if(is_categorical(dataset_test[cols])):
        ctable = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped])
        ctablen = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped], normalize='columns')*100.00
        ctablen = ctablen.applymap(lambda x: " ({:.2f})".format(x))
        stat, pvalue, dof, expected = ss.chi2_contingency(ctable)
        ctable = ctable.astype(str)
        ctable = ctable + ctablen
        if((ctable.index == 0).any()):
            ctable = ctable.drop(0)
            ctable.rename({1:"Sí"}, inplace = True) 
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue) 
        #ctable.loc[ctable.index[0], 'Valor estadistica'] ="{:.2f}".format(stat)
    else:
        dataset_numeric = pd.concat([dataset_test[cols], dataset_test[ColGrouped]], axis=1)
        #correlation, pvalue = ss.pointbiserialr(dataset_test[ColGrouped], dataset_test[cols])
        x = []
        for coltitle in ColTitles:
            x.append(dataset_numeric.loc[dataset_numeric[ColGrouped] == coltitle,cols])
        correlation, pvalue = ss.kruskal(*x)
        ctable = dataset_numeric.groupby(ColGrouped).mean().T
        ctable = ctable.applymap(lambda x: "{:.2f}".format(x))
        ctabled = dataset_numeric.groupby(ColGrouped).std().T
        ctabled = ctabled.applymap(lambda x: " ({:.2f})".format(x))
        ctable = ctable + ctabled
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue)  
    title.rename(columns={0:"Variable"}, inplace = True)
    ctable = title.append(ctable)
    CQ  =  CQ.append(ctable)
CQ = CQ.replace(np.nan,'',regex = True)
file_O = Path("Data/out/crosstab.xlsx")
file_save = cwd / file_O
CQ.to_excel(file_save)

#AR lupus
columnset = np.r_[137,1,4:7,133,136,8,73:84,138:139,31:34,35,47:49,67:70,139:146]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test.rename(columns={"R176":"Diagnóstico","R1":"Hospital", "R13": "Edad", "R14":"Sexo","R15":"Estado civil", "R172":"Educacion", "R17": "Ocupación", "R175":"Seguridad Social", "R176":"Diagnóstico", "R157":"Enfermedad pulmonar","R158":"Infarto agudo de miocardio","R159":"Otra enfermedad cardiovascular","R160":"Enfermedad vascular cerebral","R161":"Hipertensión arterial sistémica","R162":"Diabetes mellitus","R163":"Fractura de cadera/columna o pierna","R164":"Depresión","R165":"Cancer","R166":"Ulcera gastrointestinal","R167":"Otras enfermedades gástrica","R168":"Otro", "R60":"¿Usted requirió hospitalización?","R62":"¿Ya fue vacunado?","R63":"¿Ha tenido la oportunidad de vacunarse?","R65":"¿Cuenta con el esquema completo?","R70 - R77c":"¿Aceptaría alguna de las vacunas si se le ofreciera?","R71 - R78c":"Existen varias vacunas. Usted considera que","R72 - R79c":"Mi actitud con respecto a recibir la vacuna","R73 - R80c":"Si ya estuviera disponible, ¿qué haría?","R74 - R83c":"Si mi familia o amigos estuvieran pensando en vacunarse","R75 - R82c":"Con respecto a recibir la vacuna, yo me describiría como","R76 - R81c":"Recibir una vacuna es","R84":"¿Cree que se infectará?","R85":"Probabilidad que me funcione","R105":"Vacunado influenza","R148":"¿Cómo se siente en este momento?","R149":"Sistema inmune"}, inplace = True)

list_subset = (dataset_test['R176'] == 'Lupus') | (dataset_test['R176'] == 'ArtritisReumatoide')
dataset_test=dataset_test[list_subset]
dataset_test.loc[:, dataset_test.dtypes == 'object'] =\
        dataset_test.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
dataset_test.loc[:, dataset_test.isin([0,1]).all()] = dataset_test.loc[:, dataset_test.isin([0,1]).all()].apply(lambda x: x.astype('category'))
dataset_test.info()
ColNames = dataset_test.columns
ColGrouped = ColNames[0]
ColNames = np.delete(ColNames, [np.r_[0]], 0)
dataset_test[ColGrouped] = dataset_test[ColGrouped].replace("ArtritisReumatoide", "AR")
ColTitles = {'AR', 'Lupus'}
CQ  = pd.DataFrame(columns = ['Variable', 'AR', 'Lupus','p'])
temp_totales = dataset_test[ColGrouped].value_counts()
for coltitle in ColTitles:
    CQ.loc['Totales',coltitle] = temp_totales[coltitle]
for cols in ColNames:
    title_data = [cols]
    title = pd.DataFrame(title_data, index=[0])
    if(is_categorical(dataset_test[cols])):
        ctable = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped])
        ctablen = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped], normalize='columns')*100.00
        ctablen = ctablen.applymap(lambda x: " ({:.2f})".format(x))
        stat, pvalue, dof, expected = ss.chi2_contingency(ctable)
        ctable = ctable.astype(str)
        ctable = ctable + ctablen
        if((ctable.index == 0).any()):
            ctable = ctable.drop(0)
            ctable.rename({1:"Sí"}, inplace = True) 
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue) 
        #ctable.loc[ctable.index[0], 'Valor estadistica'] ="{:.2f}".format(stat)
    else:
        dataset_numeric = pd.concat([dataset_test[cols], dataset_test[ColGrouped]], axis=1)
        #correlation, pvalue = ss.pointbiserialr(dataset_test[ColGrouped], dataset_test[cols])
        x = []
        for coltitle in ColTitles:
            x.append(dataset_numeric.loc[dataset_numeric[ColGrouped] == coltitle,cols])
        correlation, pvalue = ss.kruskal(*x)
        ctable = dataset_numeric.groupby(ColGrouped).mean().T
        ctable = ctable.applymap(lambda x: "{:.2f}".format(x))
        ctabled = dataset_numeric.groupby(ColGrouped).std().T
        ctabled = ctabled.applymap(lambda x: " ({:.2f})".format(x))
        ctable = ctable + ctabled
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue)  
    title.rename(columns={0:"Variable"}, inplace = True)
    ctable = title.append(ctable)
    CQ  =  CQ.append(ctable)
CQ = CQ.replace(np.nan,'',regex = True)
file_O = Path("Data/out/crosstab.xlsx")
file_save = cwd / file_O
CQ.to_excel(file_save)

#numero de dosis
columnsetsp = np.r_[35]
dataset_temp1= data_I_HGM.iloc[:,columnsetsp]
columnsetsp = np.r_[32]
dataset_temp2= data_I_HGM.iloc[:,columnsetsp]
dataset_dependents = dataset_temp2.iloc[:,0] + dataset_temp1.iloc[:,0]
dataset_dependents = dataset_dependents.map({1:"Una dosis", 0:"No vacunado", 2:"Dos dosis"})
#columnset = np.r_[1,4:7,133,136,8,137:139,31,47:49,67:70,139:146]
columnset = np.r_[1,4:7,133,136,8,73:84,17:30,31,47:49,67:70,139:146]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test.rename(columns={"R1":"Hospital", "R13": "Edad", "R14":"Sexo","R15":"Estado civil", "R172":"Educacion", "R17": "Ocupación", "R175":"Seguridad Social", "R176":"Diagnóstico", "R157":"Enfermedad pulmonar","R158":"Infarto agudo de miocardio","R159":"Otra enfermedad cardiovascular","R160":"Enfermedad vascular cerebral","R161":"Hipertensión arterial sistémica","R162":"Diabetes mellitus","R163":"Fractura de cadera/columna o pierna","R164":"Depresión","R165":"Cancer","R166":"Ulcera gastrointestinal","R167":"Otras enfermedades gástrica","R168":"Otro", "R60":"¿Usted requirió hospitalización?","R62":"¿Ya fue vacunado?","R63":"¿Ha tenido la oportunidad de vacunarse?","R65":"¿Cuenta con el esquema completo?","R70 - R77c":"¿Aceptaría alguna de las vacunas si se le ofreciera?","R71 - R78c":"Existen varias vacunas. Usted considera que","R72 - R79c":"Mi actitud con respecto a recibir la vacuna","R73 - R80c":"Si ya estuviera disponible, ¿qué haría?","R74 - R83c":"Si mi familia o amigos estuvieran pensando en vacunarse","R75 - R82c":"Con respecto a recibir la vacuna, yo me describiría como","R76 - R81c":"Recibir una vacuna es","R84":"¿Cree que se infectará?","R85":"Probabilidad que me funcione","R105":"Vacunado influenza","R148":"¿Cómo se siente en este momento?","R149":"Sistema inmune","R26":"Lupus eritematoso generalizado","R27":"Artritis reumatoide","R28":"Vasculitis ANCA","R29":"Espondilitis anquilosante","R30":"Esclerodermia","R31":"Gota","R32":"Síndrome de Sjögren","R33":"Miopatía inflamatoria","R34":"Síndrome de anticuerpos antifosfolípidos","R35":"Artritits idiopática juvenil","R36":"Enfermedad mixta del tejido conectivo","R37":"Osteoartrosis","R38":"Otro"}, inplace = True)
dataset_test.columns.tolist
dataset_test.loc[:, dataset_test.dtypes == 'object'] =\
        dataset_test.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
dataset_test.loc[:, dataset_test.isin([0,1]).all()] = dataset_test.loc[:, dataset_test.isin([0,1]).all()].apply(lambda x: x.astype('category'))
dataset_test.info()
dataset_test.insert(0, 'Dosis', dataset_dependents)

ColNames = dataset_test.columns
ColGrouped = ColNames[0]
ColNames = np.delete(ColNames, [np.r_[0]], 0)
ColTitles = {"No vacunado", "Una dosis","Dos dosis"}
CQ  = pd.DataFrame(columns = ['Variable', "No vacunado", "Una dosis","Dos dosis",'p'])
temp_totales = dataset_test[ColGrouped].value_counts()
for coltitle in ColTitles:
    CQ.loc['Totales',coltitle] = temp_totales[coltitle]
for cols in ColNames:
    title_data = [cols]
    title = pd.DataFrame(title_data, index=[0])
    if(is_categorical(dataset_test[cols])):
        ctable = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped])
        ctablen = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped], normalize='columns')*100.00
        ctablen = ctablen.applymap(lambda x: " ({:.2f})".format(x))
        stat, pvalue, dof, expected = ss.chi2_contingency(ctable)
        ctable = ctable.astype(str)
        ctable = ctable + ctablen
        if((ctable.index == 0).any()):
            ctable = ctable.drop(0)
            ctable.rename({1:"Sí"}, inplace = True) 
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue) 
        #ctable.loc[ctable.index[0], 'Valor estadistica'] ="{:.2f}".format(stat)
    else:
        dataset_numeric = pd.concat([dataset_test[cols], dataset_test[ColGrouped]], axis=1)
        #correlation, pvalue = ss.pointbiserialr(dataset_test[ColGrouped], dataset_test[cols])
        x = []
        for coltitle in ColTitles:
            x.append(dataset_numeric.loc[dataset_numeric[ColGrouped] == coltitle,cols])
        correlation, pvalue = ss.kruskal(*x)
        ctable = dataset_numeric.groupby(ColGrouped).mean().T
        ctable = ctable.applymap(lambda x: "{:.2f}".format(x))
        ctabled = dataset_numeric.groupby(ColGrouped).std().T
        ctabled = ctabled.applymap(lambda x: " ({:.2f})".format(x))
        ctable = ctable + ctabled
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue)  
    title.rename(columns={0:"Variable"}, inplace = True)
    ctable = title.append(ctable)
    CQ  =  CQ.append(ctable)
CQ = CQ.replace(np.nan,'',regex = True)
file_O = Path("Data/out/crosstab.xlsx")
file_save = cwd / file_O
CQ.to_excel(file_save)


#time
columnset = np.r_[40:47]#40:47
dataset_dependents= data_I_HGM.iloc[:,columnset]
list_del = (dataset_dependents != 0).all(1)
max_column = dataset_dependents.apply(pd.Series.value_counts, axis=1).idxmax(axis=1)
max_column.name = "MaxFrec"
columnset = np.r_[132,40:47]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents = pd.concat([dataset_dependents, max_column], axis=1)
dataset_dependents.loc[:, dataset_dependents.dtypes == 'object'] =\
        dataset_dependents.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
ColNames = dataset_dependents.columns
ColNames = np.delete(ColNames, [np.r_[0]], 0)
for cols in ColNames:
    datset_time = pd.DataFrame()
    dataset_sub = dataset_dependents[['R171c',cols]]
    for cat_id in range(0, time_cats):
        dataset_test = dataset_sub.loc[dataset_sub.R171c == dataset_sub.R171c.cat.categories[cat_id],cols]
        datset_time = pd.concat([datset_time, dataset_test.rename(str(dataset_sub.R171c.cat.categories[cat_id]))], axis=1)
    dataset_test = datset_time.fillna(0)
    dataset_test = dataset_test.replace({0:'None',1: 'Strongly agree', 2:'Agree', 3: 'Neither agree nor disagree', 4:'Disagree', 5:'Strongly disagree'})
    another_scale = \
        ['Strongly disagree',
        'Disagree',
        'Neither agree nor disagree',
        'Agree',
        'Strongly agree',
        'None']
    colours5 = ['white','#f9665e','#fec9c9','#949494','#95b4cc','#799fcb'] 
    ax = plot_likert.plot_likert(dataset_test, another_scale,plot_percentage=True, colors=colours5) #another_scale plot_likert.scales.agree
    ax.set_xlim([0, 30]);
    ax.xaxis.set_label_text('Porcentaje de respuestas: Población completa');
    ax.get_legend().remove()
    image_format = 'svg'
    file_O = Path("Data/outimg/likert_hesitancy_"+cols+".svg")
    file_save = cwd / file_O
    plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
    plt.close()

columnset = np.r_[132,32,35]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents.rename(columns={"R171c":"Date","R62":"1 dosis","R65":"2 dosis"}, inplace = True)
agg_dict = {
    '1 dosis':['sum'],
    '2 dosis':['sum']
}
dataset_dependents_count=dataset_dependents.groupby(by=["Date"]).agg(agg_dict)
sns.heatmap(dataset_dependents_count,annot = dataset_dependents_count, fmt=".0f", linewidths=.5, cmap="flare") 
image_format = 'svg'
file_O = Path("Data/outimg/dosis_totales"+".svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
plt.close()

columnset = np.r_[132,32,35,128:130]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents.rename(columns={"R171c":"Date","R62":"¿1 dosis?","R65":"¿2 dosis?","R168":"Muertes", "R169": "Casos"}, inplace = True)
dataset_dependents_norm= dataset_dependents.groupby(by=["Date"]).mean()/(1.0,1.0,dataset_dependents.iloc[:,3].max(),dataset_dependents.iloc[:,4].max())
agg_dict = {
    '1 dosis':['sum'],
    '2 dosis':['sum'],
    'Muertes':['mean'],
    'Casos':['mean'],
}
dataset_dependents_count=dataset_dependents.groupby(by=["Date"]).agg(agg_dict)
sns.heatmap(dataset_dependents_norm,annot = dataset_dependents_count, fmt=".0f", linewidths=.5, cmap="flare") 
image_format = 'svg'
file_O = Path("Data/outimg/Cases_totales"+".svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
plt.close()


#enfermedad ArtritisReumatoide Lupus
columnset = np.r_[137]#40:47
dataset_dependents= data_I_HGM.iloc[:,columnset]
list_subset = dataset_dependents['R176'] == 'ArtritisReumatoide'
columnset = np.r_[40:47]#40:47
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents=dataset_dependents[list_subset]
list_del = (dataset_dependents != 0).all(1)
max_column = dataset_dependents.apply(pd.Series.value_counts, axis=1).idxmax(axis=1)
max_column.name = "MaxFrec"
max_column=max_column[list_subset]
columnset = np.r_[132,40:47]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents=dataset_dependents[list_subset]
dataset_dependents = pd.concat([dataset_dependents, max_column], axis=1)
dataset_dependents.loc[:, dataset_dependents.dtypes == 'object'] =\
        dataset_dependents.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
ColNames = dataset_dependents.columns
ColNames = np.delete(ColNames, [np.r_[0]], 0)
for cols in ColNames:
    datset_time = pd.DataFrame()
    dataset_sub = dataset_dependents[['R171c',cols]]
    for cat_id in range(0, time_cats):
        dataset_test = dataset_sub.loc[dataset_sub.R171c == dataset_sub.R171c.cat.categories[cat_id],cols]
        datset_time = pd.concat([datset_time, dataset_test.rename(str(dataset_sub.R171c.cat.categories[cat_id]))], axis=1)
    dataset_test = datset_time.fillna(0)
    dataset_test = dataset_test.replace({0:'None',1: 'Strongly agree', 2:'Agree', 3: 'Neither agree nor disagree', 4:'Disagree', 5:'Strongly disagree'})
    another_scale = \
        ['Strongly disagree',
        'Disagree',
        'Neither agree nor disagree',
        'Agree',
        'Strongly agree',
        'None']
    colours5 = ['white','#f9665e','#fec9c9','#949494','#95b4cc','#799fcb'] 
    ax = plot_likert.plot_likert(dataset_test, another_scale,plot_percentage=True, colors=colours5) #another_scale plot_likert.scales.agree
    ax.set_xlim([0, 30]);
    ax.xaxis.set_label_text('Porcentaje de respuestas: Artritis Reumatoide');
    ax.get_legend().remove()
    image_format = 'svg'
    file_O = Path("Data/outimg/likert_hesitancy_ArtritisReumatoide_"+cols+".svg")
    file_save = cwd / file_O
    plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
    plt.close()

columnset = np.r_[132,32,35]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents.rename(columns={"R171c":"Date","R62":"1 dosis","R65":"2 dosis"}, inplace = True)
agg_dict = {
    '1 dosis':['sum'],
    '2 dosis':['sum']
}
dataset_dependents_count=dataset_dependents.groupby(by=["Date"]).agg(agg_dict)
sns.heatmap(dataset_dependents_count,annot = dataset_dependents_count, fmt=".0f", linewidths=.5, cmap="flare") 
image_format = 'svg'
file_O = Path("Data/outimg/dosis_artritis"+".svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
plt.close()
    
columnset = np.r_[132,32,35,128:130]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents=dataset_dependents[list_subset]
dataset_dependents.rename(columns={"R171c":"Date","R62":"¿1 dosis?","R65":"¿2 dosis?","R168":"Muertes", "R169": "Casos"}, inplace = True)
dataset_dependents_norm= dataset_dependents.groupby(by=["Date"]).mean()/(1.0,1.0,dataset_dependents.iloc[:,3].max(),dataset_dependents.iloc[:,4].max())
agg_dict = {
    '¿1 dosis?':['sum'],
    '¿2 dosis?':['sum'],
    'Muertes':['mean'],
    'Casos':['mean'],
}
dataset_dependents_count=dataset_dependents.groupby(by=["Date"]).agg(agg_dict)
sns.heatmap(dataset_dependents_norm,annot = dataset_dependents_count, fmt=".0f", linewidths=.5, cmap="flare") 
image_format = 'svg'
file_O = Path("Data/outimg/Cases_artritis"+".svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
plt.close()

columnset = np.r_[137]#40:47
dataset_dependents= data_I_HGM.iloc[:,columnset]
list_subset = dataset_dependents['R176'] == 'Lupus'
columnset = np.r_[40:47]#40:47
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents=dataset_dependents[list_subset]
list_del = (dataset_dependents != 0).all(1)
max_column = dataset_dependents.apply(pd.Series.value_counts, axis=1).idxmax(axis=1)
max_column.name = "MaxFrec"
max_column=max_column[list_subset]
columnset = np.r_[132,40:47]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents=dataset_dependents[list_subset]
dataset_dependents = pd.concat([dataset_dependents, max_column], axis=1)
dataset_dependents.loc[:, dataset_dependents.dtypes == 'object'] =\
        dataset_dependents.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
ColNames = dataset_dependents.columns
ColNames = np.delete(ColNames, [np.r_[0]], 0)
for cols in ColNames:
    datset_time = pd.DataFrame()
    dataset_sub = dataset_dependents[['R171c',cols]]
    for cat_id in range(0, time_cats):
        dataset_test = dataset_sub.loc[dataset_sub.R171c == dataset_sub.R171c.cat.categories[cat_id],cols]
        datset_time = pd.concat([datset_time, dataset_test.rename(str(dataset_sub.R171c.cat.categories[cat_id]))], axis=1)
    dataset_test = datset_time.fillna(0)
    dataset_test = dataset_test.replace({0:'None',1: 'Strongly agree', 2:'Agree', 3: 'Neither agree nor disagree', 4:'Disagree', 5:'Strongly disagree'})
    another_scale = \
        ['Strongly disagree',
        'Disagree',
        'Neither agree nor disagree',
        'Agree',
        'Strongly agree',
        'None']
    colours5 = ['white','#f9665e','#fec9c9','#949494','#95b4cc','#799fcb'] 
    ax = plot_likert.plot_likert(dataset_test, another_scale,plot_percentage=True, colors=colours5) #another_scale plot_likert.scales.agree
    ax.set_xlim([0, 30]);
    ax.xaxis.set_label_text('Porcentaje de respuestas: Lupus');
    ax.get_legend().remove()
    image_format = 'svg'
    file_O = Path("Data/outimg/likert_hesitancy_Lupus_"+cols+".svg")
    file_save = cwd / file_O
    plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
    plt.close()

columnset = np.r_[132,32,35]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents.rename(columns={"R171c":"Date","R62":"1 dosis","R65":"2 dosis"}, inplace = True)
agg_dict = {
    '1 dosis':['sum'],
    '2 dosis':['sum']
}
dataset_dependents_count=dataset_dependents.groupby(by=["Date"]).agg(agg_dict)
sns.heatmap(dataset_dependents_count,annot = dataset_dependents_count, fmt=".0f", linewidths=.5, cmap="flare") 
image_format = 'svg'
file_O = Path("Data/outimg/dosis_lupus"+".svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
plt.close()

columnset = np.r_[132,32,35,128:130]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents=dataset_dependents[list_subset]
dataset_dependents.rename(columns={"R171c":"Date","R62":"¿1 dosis?","R65":"¿2 dosis?","R168":"Muertes", "R169": "Casos"}, inplace = True)
dataset_dependents_norm= dataset_dependents.groupby(by=["Date"]).mean()/(1.0,1.0,dataset_dependents.iloc[:,3].max(),dataset_dependents.iloc[:,4].max())
agg_dict = {
    '¿1 dosis?':['sum'],
    '¿2 dosis?':['sum'],
    'Muertes':['mean'],
    'Casos':['mean'],
}
dataset_dependents_count=dataset_dependents.groupby(by=["Date"]).agg(agg_dict)
sns.heatmap(dataset_dependents_norm,annot = dataset_dependents_count, fmt=".0f", linewidths=.5, cmap="flare") 
image_format = 'svg'
file_O = Path("Data/outimg/Cases_lupus"+".svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
plt.close()

#time2
columnset = np.r_[40:47]#40:47
dataset_dependents= data_I_HGM.iloc[:,columnset]
list_del = (dataset_dependents != 0).all(1)
columnset = np.r_[132,40:47]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents.loc[:, dataset_dependents.dtypes == 'object'] =\
        dataset_dependents.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
for cat_id in range(0, time_cats):
    dataset_test = dataset_dependents[dataset_dependents.R171c == dataset_dependents.R171c.cat.categories[cat_id]]
    date_str=str(dataset_test.iloc[0,0])
    dataset_test.drop(columns=dataset_test.columns[0], axis=1, inplace=True)
    dataset_test = dataset_test.replace({0:'None',1: 'Strongly agree', 2:'Agree', 3: 'Neither agree nor disagree', 4:'Disagree', 5:'Strongly disagree'})
    another_scale = \
        ['Strongly disagree',
        'Disagree',
        'Neither agree nor disagree',
        'Agree',
        'Strongly agree',
        'None']
    colours5 = ['white','#f9665e','#fec9c9','#949494','#95b4cc','#799fcb'] 
    plot_likert.plot_likert(dataset_test, another_scale,plot_percentage=True, colors=colours5) #another_scale plot_likert.scales.agree
    image_format = 'svg'
    file_O = Path("Data/outimg/likert_hesitancy_"+date_str+".svg")
    file_save = cwd / file_O
    plt.savefig(file_save, format=image_format, dpi=1200)
    plt.close()

#cosine simmilitued
cosine_similarity(df)

#matriz de asociacion de cramer correlacion chi-square
columnset = np.r_[139:146]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test.rename(columns={"R70 - R77c":"¿Aceptaría alguna de las vacunas si se le ofreciera?","R71 - R78c":"Existen varias vacunas. Usted considera que","R72 - R79c":"Mi actitud con respecto a recibir la vacuna","R73 - R80c":"Si ya estuviera disponible, ¿qué haría?","R74 - R83c":"Si mi familia o amigos estuvieran pensando en vacunarse","R75 - R82c":"Con respecto a recibir la vacuna, yo me describiría como","R76 - R81c":"Recibir una vacuna es"}, inplace = True)
dataset_test.loc[:, dataset_test.dtypes == 'object'] =\
        dataset_test.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
dataset_test.info()        
dataset_test.loc[:, dataset_test.isin([0,1]).all()] = dataset_test.loc[:, dataset_test.isin([0,1]).all()].apply(lambda x: x.astype('category'))
ColNames = dataset_test.columns
datast_c = pd.DataFrame(columns=ColNames, index=ColNames,dtype="float")
datast_t = pd.DataFrame(columns=ColNames, index=ColNames,dtype="float")
datast_p = pd.DataFrame(columns=ColNames, index=ColNames,dtype="float")
for cols_x in ColNames:
    for cols_y in ColNames:
        c_value = cramers_v(dataset_test[cols_x],dataset_test[cols_y],bias_correction=False)
        t_value = theils_u(dataset_test[cols_x],dataset_test[cols_y])
        ctable = pd.crosstab(dataset_test[cols_x],dataset_test[cols_y])
        stat, p_value, dof, expected = ss.chi2_contingency(ctable)
        datast_c.loc[cols_x,cols_y] = c_value
        datast_t.loc[cols_x,cols_y] = t_value
        if(pvalue < 0.001):
            datast_p.loc[cols_x,cols_y] = '<0.001'
        else:
            datast_p.loc[cols_x,cols_y] = "{:.3f}".format

sns.heatmap(datast_c, annot = True, fmt=".2f", linewidths=.5, cmap="flare") 
image_format = 'svg'
file_O = Path("Data/outimg/cramer"+".svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
plt.close()

sns.heatmap(datast_t, annot = True, fmt=".2f", linewidths=.5, cmap="flare") 
image_format = 'svg'
file_O = Path("Data/outimg/theils"+".svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
plt.close()

#logistic regresion
#new set
columnset = np.r_[1,4:7,136,8,137:139,31:34,35,47:49,67:70]#,138:146]#17:89]#   ,]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test['R62'] = dataset_test['R62'].map({1:"Sí", 0:"No"})
dataset_test.columns.tolist()
dataset_test.rename(columns={"R1":"Hospital", "R13": "Edad", "R14":"Sexo","R15":"EstadoCivil", "R17": "Ocupacion", "R175":"SeguridadSocial", "R176":"Diagnostico", "R177":"Comorbilidad", "R60":"RequirioHospitalización","R62":"Vacunado","R63":"OportunidadVacunarse","R65":"EsquemaCompleto","R84":"Infectara","R85":"ProbabilidadFuncione","R105":"VacunadoInfluenza","R148":"SienteMomento","R149":"SistemaInmune"}, inplace = True)
dataset_test.loc[:, dataset_test.dtypes == 'object'] =\
        dataset_test.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
dataset_test.loc[:, dataset_test.isin([0,1]).all()] = dataset_test.loc[:, dataset_test.isin([0,1]).all()].apply(lambda x: x.astype('category'))
#
columnset = np.r_[40:47]
dataset_dependents= data_I_HGM.iloc[:,columnset]
list_del = (dataset_dependents != 0).all(1)
dataset_dependents = dataset_dependents.replace({2:1, 3:0, 4:0, 5:0})
dataset_dependents['bin'] = (dataset_dependents.sum(axis=1)>=4)*1
columnset = np.r_[134]
dataset_temp= data_I_HGM.iloc[:,columnset]
dataset_dependents['per'] = (dataset_temp['R173'] >= dataset_temp.R173.quantile(0.75))*1
dataset_dependents=dataset_dependents[list_del]
dataset_odds= pd.DataFrame(columns=['ODDs', 'coeff', 'pvals', '0.025', '0.975', 'name'])
dataset_odds= pd.DataFrame()
for name, values in dataset_dependents.iteritems():
    print(name)
    dataset_results_final = pd.DataFrame()
    dataset_dependent = dataset_dependents.loc[:,name]
    dataset_independent= dataset_test
    dataset_independent=dataset_independent[list_del]
    dataset_model = pd.concat([dataset_dependent, dataset_independent], axis=1)
    dataset_model.rename(columns={ dataset_model.columns[0]: "D" }, inplace = True)
    train_data, test_data = train_test_split(dataset_model, test_size=0.20, random_state= 42)
    # formula = ('D ~ Hospital + Edad + Sexo + EstadoCivil + SeguridadSocial + Ocupacion + Diagnostico + Comorbilidad + RequirioHospitalización + Vacunado + OportunidadVacunarse + EsquemaCompleto + Infectara + ProbabilidadFuncione + VacunadoInfluenza + SienteMomento + SistemaInmune')
    # formula = ('D ~ Hospital + Edad + Sexo + EstadoCivil + SeguridadSocial + Ocupacion + RequirioHospitalización + Vacunado + OportunidadVacunarse + EsquemaCompleto + Infectara + ProbabilidadFuncione + VacunadoInfluenza + SienteMomento + SistemaInmune')
    formula = ('D ~ Hospital + Edad + Sexo + EstadoCivil  + OportunidadVacunarse + EsquemaCompleto + Infectara + ProbabilidadFuncione + VacunadoInfluenza + SienteMomento + SistemaInmune')
    model = logit(formula = formula, data = train_data).fit()
    dataset_results = results_summary_to_dataframe(model)
    dataset_results_final['OR (95% CI two-sided)'] = dataset_results['ODDs'].fillna('').apply('{:.2f}'.format).astype(str) + ' (' + dataset_results['0.05'].fillna('').apply('{:.2f}'.format).astype(str) + ' - ' + dataset_results['0.95'].fillna('').apply('{:.2f}'.format).astype(str) +  ' )'
    dataset_results['ps'] = ''
    dataset_results.loc[dataset_results['pvals']< 0.05,'ps'] = '< 0.05' 
    dataset_results.loc[dataset_results['pvals']>= 0.05,'ps'] = dataset_results.loc[dataset_results['pvals']>= 0.05,'pvals'].fillna('').apply('{:.3f}'.format).astype(str)
    dataset_results_final['p'] = dataset_results['ps']
    dataset_results_final.columns = pd.MultiIndex.from_product([[name],dataset_results_final.columns])
    dataset_odds = pd.concat([dataset_odds, dataset_results_final], axis=1)
file_O = Path("Data/out/" + "OddsAll.xlsx")
file_save = cwd / file_O
dataset_odds.to_excel(file_save)
