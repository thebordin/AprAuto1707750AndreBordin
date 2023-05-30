print('############## ATIVIDADE 1 ##############')

import pandas as pd

#### PARAMETROS ####
proporcao_treino_teste = 0.2
rdm_state = 0
neuronios = (175,120,100)
entrada_manual_teste = ''

url_dataset = 'data/heart1.csv'
url_trickbag = 'data/trickbag2'

#### FIM PARAMETROS ####

dataset = pd.read_csv(url_dataset)
cnt = 0
for x in dataset.isna().any():
    if x == True:
        cnt += 1
if cnt == 0:
    print('===== Dataset sem Missing Values =====')
else:
    print ('!!!!! Dataset possui Missing Values !!!!!')
#######################################################################################
print('############## ATIVIDADE 3 ##############')

from sklearn.model_selection import train_test_split

features, features_validation, labels, label_validation= train_test_split(dataset.iloc[:,:-1],
                                                                          dataset.iloc[:,-1],
                                                                          test_size=proporcao_treino_teste,
                                                                          random_state=rdm_state)
print('===== Datasets separados com sucesso =====')


#########################################################################################
######################Codificando as features
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler

trickbag = Pipeline([('StandardScaler', StandardScaler()),
                     ('LabelE', LabelEncoder())])
for nomes in dataset.iloc[:,:-1].columns.values:
    trickbag['LabelE'].fit(dataset[f'{nomes}'])
    features[f'{nomes}'] = trickbag['LabelE'].transform(features[f'{nomes}'])
    trickbag.steps.append([f'{nomes}', trickbag['LabelE'].classes_])
print('============= Features Codificadas ==============')
#######################################################################################
print('############## ATIVIDADE 2 ##############')

import joblib

features_ss = trickbag['StandardScaler'].fit_transform(pd.DataFrame(features))

trickbag_file = open(url_trickbag, 'wb')
joblib.dump(trickbag,trickbag_file)
print('===== Modelo e Transformador Gravado com sucesso =====')

#######################################################################################
print('############## ATIVIDADE 4 ##############')

from sklearn.neural_network import MLPClassifier

trickbag.steps.append(['MLPC', MLPClassifier(hidden_layer_sizes=neuronios,
                                             max_iter=100000,
                                             activation= 'relu',
                                             solver='adam',
                                             random_state=rdm_state)])

trickbag['MLPC'].fit(features_ss, labels)

print('===== Predictor treinado com sucesso =====')

#######################################################################################
print('############## ATIVIDADE 5 ##############')

from sklearn.metrics import confusion_matrix, accuracy_score
features_validation_le = features_validation.copy()
for nomes in features_validation_le.columns.values:
    trickbag['LabelE'].classes_ = trickbag[f'{nomes}']
    features_validation_le[nomes] = trickbag['LabelE'].transform(features_validation_le[f'{nomes}'])
features_validation_ss = trickbag['StandardScaler'].transform(features_validation_le)
label_predicted = trickbag['MLPC'].predict(features_validation_ss)
label_predicted = pd.DataFrame(label_predicted)

cm = confusion_matrix(label_validation,label_predicted)

print('Traço = %.2f || Matriz confusão = %.2f || Score = %.2f || Precisao do MLPC = %.2f'% (cm.trace(), cm.sum(), trickbag['MLPC'].score(features_validation_ss, label_validation), accuracy_score(label_validation, label_predicted)))

#######################################################################################
print('############## ATIVIDADE 6 ##############')
import random

if entrada_manual_teste != '':
    for nomes in entrada_manual_teste.columns.values:
        trickbag['LabelE'].classes_ = trickbag[f'{nomes}']
        features_validation_le[nomes] = trickbag['LabelE'].transform(entrada_manual_teste[nomes])
    resultado = trickbag['StandardScaler'].transform(entrada_manual_teste)
    resultado = trickbag['MLPC'].predict(entrada_manual_teste)
    print(f'Resultado da Previsão: {resultado}')
else:
    rng = int(random.random() * (len(features_validation)))
    amostra = features_validation.iloc[rng:rng+1,:].copy()
    print('AMOSTRA: \n',amostra)
    for nomes in amostra.columns.values:
        trickbag['LabelE'].classes_ = trickbag[f'{nomes}']
        amostra[nomes] = trickbag['LabelE'].transform(amostra[nomes])
    esperado = label_validation.iloc[rng]
    resultado = trickbag['StandardScaler'].transform(amostra)
    resultado = trickbag['MLPC'].predict(resultado)
    print(f'Resultado da Previsão: {resultado} || Resultado Esperado: {esperado} || Linha: {rng}')


trickbag_file = open(url_trickbag, 'wb')
joblib.dump(trickbag,trickbag_file)
print('===== Modelo e Transformador Gravado com sucesso =====')