import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import sys

#### PARAMETROS ####
input = sys.argv[1]
output = sys.argv[2]
trickbag_url = sys.argv[3]
dataset = pd.read_csv(input, header=None, sep=';')
#### FIM PARAMETROS ####
trickbag = Pipeline([('StandardScaler', StandardScaler())])

#dataset_ss = trickbag['StandardScaler'].fit_transform(dataset)
dataframe_ss = pd.DataFrame(trickbag['StandardScaler'].fit_transform(dataset))

try:
    dataframe_ss.to_csv(output, index = False)
except: print('Houve um erro em importar o Dataset')

try:
    trickbag_file = open(trickbag_url, 'wb')
    joblib.dump(trickbag,trickbag_file)
    print('===== Modelo e Dataset Gravados =====')
except: print('Houve um erro em importar o Modelo StandardScaler')