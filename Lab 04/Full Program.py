print('############## ATIVIDADE 1 ##############')

import pandas as pd

#### PARAMETROS ####
proporcao_treino_teste = 0.2
rdm_state = 0
neuronios = (150,100,50)
entrada_manual_teste = ''

url_dataset = 'data/heart.csv'
url_trickbag = 'data/trickbag'
url_ds_clean = 'data/heart_clean_1.csv'
url_ds_scaled = 'data/scale_data_2.csv'
url_train = 'data/heart_train_3.csv'
url_validation = 'data/heart_validation_3.csv'

#### FIM PARAMETROS ####

dataset = pd.read_csv(url_dataset)
cnt = 0
for x in dataset.isna().any():
    if x == True:
        cnt += 1
if cnt == 0:
    try:
        dataset.to_csv(url_ds_clean, index=False)
        print('===== Dataset salvo sem Missing Values =====')
    except: print('Houve um erro na gravação')
else:
    print ('!!!!! Dataset não salvo pois existem Missing Values !!!!!')

#######################################################################################
print('############## ATIVIDADE 2 ##############')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

ds_clean = pd.read_csv(url_ds_clean)
trickbag = Pipeline([('StandartScaler', StandardScaler())])

ds_clean.iloc[:,0:-1] = trickbag['StandartScaler'].fit_transform(ds_clean.iloc[:,0:-1])

try:
    ds_clean.to_csv(url_ds_scaled, index = False)
except: print('!!!!!! Houve um erro em importar o Dataset !!!!!!!')

try:
    trickbag_file = open(url_trickbag, 'wb')
    joblib.dump(trickbag,trickbag_file)
    print('===== Modelo e Dataset Gravados =====')
except: print('!!!!!! Houve um erro em importar o Modelo StandartScaler !!!!!!')

#######################################################################################
print('############## ATIVIDADE 3 ##############')

from sklearn.model_selection import train_test_split

dataset = pd.read_csv(url_ds_scaled)
features, features_validation= train_test_split(dataset,
                                          test_size=proporcao_treino_teste,
                                          random_state=rdm_state)

try:
    features.to_csv(url_train, index = False)
    features_validation.to_csv(url_validation, index=False)
    print('===== Datasets separados e gravados com sucesso =====')
except : print('!!!!! Houve um problema na gravação dos datasets !!!!!')

#######################################################################################
print('############## ATIVIDADE 4 ##############')

from sklearn.neural_network import MLPClassifier

#### PARAMETROS ####
url_features_train = 'data/heart_train_3.csv'
rdm_state = 0
url_trickbag_std = 'data/predictorMLPC_heart_4.sav'
#### FIM PARAMETROS ####

dataset = pd.read_csv(url_features_train)

trickbag.steps.append(['MLPC', MLPClassifier(hidden_layer_sizes=neuronios,
                                             max_iter=100000,
                                             activation= 'relu',
                                             solver='adam',
                                             random_state=rdm_state)])

trickbag['MLPC'].fit(dataset.iloc[:,:-1], dataset.iloc[:,-1])
try:
    trickbag_file = open(url_trickbag_std, 'wb')
    joblib.dump(trickbag,trickbag_file)
    print('===== Predictor gravado com sucesso =====')
except: print('!!!!!! Houve um erro ao gravar o predictor !!!!!!')

#######################################################################################
print('############## ATIVIDADE 5 ##############')

from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv(url_validation)
x_ss = dataset.iloc[:,:-1]
y_validation = dataset.iloc[:,-1]

y_predicted = trickbag['MLPC'].predict(x_ss)
y_predicted = pd.DataFrame(y_predicted)

cm = confusion_matrix(y_predicted,y_validation)

print('Traço = %.2f || Matriz confusão = %.2f || Score = %.2f || Precisao do MLPC = %.2f'% (cm.trace(), cm.sum(), trickbag['MLPC'].score(x_ss, y_validation), accuracy_score(y_validation, y_predicted)))

#######################################################################################
print('############## ATIVIDADE 6 ##############')

import random
if entrada_manual_teste != '':
    resultado = trickbag['MLPC'].predict(entrada_manual_teste)
    print(f'Resultado da Previsão: {resultado}')
else:
    dataset_teste = pd.read_csv(url_validation)
    rng = int(random.random()*(len(dataset_teste)))
    amostra = pd.DataFrame(dataset_teste.iloc[rng,:-1]).T
    resultado = trickbag['MLPC'].predict(amostra)
    esperado = dataset_teste.iloc[rng,-1]
    print(f'Resultado da Previsão: {resultado} || Resultado Esperado: {esperado} || Linha: {rng+1}')