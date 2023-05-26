import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier

#### PARAMETROS ####
url_features_train = 'data/heart_train_3.csv'
rdm_state = 0
url_trickbag = 'data/trickbag'
url_trickbag_std = 'data/predictorMLPC_heart_4.sav'
trickbag = joblib.load(open(url_trickbag, 'rb'))
#### FIM PARAMETROS ####

dataset = pd.read_csv(url_features_train)

trickbag.steps.append(['MLPC', MLPClassifier(hidden_layer_sizes=(150,100,50),
                                             max_iter=100000,
                                             activation= 'relu',
                                             solver='adam',
                                             random_state=rdm_state)])

trickbag['MLPC'].fit(dataset.iloc[:,:-1], dataset.iloc[:,-1])
try:
    trickbag_file = open(url_trickbag_std, 'wb')
    joblib.dump(trickbag,trickbag_file)
    print('Predictor gravado com sucesso')
except: print('Houve um erro ao gravar o predictor')
