import pandas as pd
import joblib
import random

#### PARAMETROS ####
url_features_validation = 'data/heart_validation_3.csv'
url_trickbag_std = 'data/predictorMLPC_heart_4.sav'
trickbag = joblib.load(open(url_trickbag_std, 'rb'))
#### FIM PARAMETROS ####

dataset_teste = pd.read_csv(url_features_validation)
rng = int(random.random()*(len(dataset_teste)))
amostra = pd.DataFrame(dataset_teste.iloc[rng,:-1]).T
resultado = trickbag['MLPC'].predict(amostra)
esperado = dataset_teste.iloc[rng,-1]
print(f'Resultado da Previsão: {resultado} || Resultado Esperado: {esperado} || Linha: {rng}')