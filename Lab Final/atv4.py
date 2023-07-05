import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
import sys

#### PARAMETROS ####
input = sys.argv[1]
trickbag_url = sys.argv[2]
url_predictor = sys.argv[3]
rdm_state = 42
trickbag = joblib.load(open(trickbag_url, 'rb'))
dataset = pd.read_csv(input, header=None, sep=';')
#### FIM PARAMETROS ####

dataset = pd.read_csv(input)

trickbag.steps.append(['MLPC', MLPClassifier(hidden_layer_sizes=(150,100,50),
                                             max_iter=100000,
                                             activation= 'relu',
                                             solver='adam',
                                             random_state=rdm_state)])

trickbag['MLPC'].fit(dataset)
try:
    trickbag_file = open(url_predictor, 'wb')
    joblib.dump(trickbag,trickbag_file)
    print('Predictor gravado com sucesso')
except: print('Houve um erro ao gravar o predictor')
