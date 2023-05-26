import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

#### PARAMETROS ####
url_ds_clean = 'data/heart_clean_1.csv'
url_ds_scaled = 'data/scale_data_2.csv'
url_trickbag = 'data/trickbag'
#### FIM PARAMETROS ####

ds_clean = pd.read_csv(url_ds_clean)
trickbag = Pipeline([('StandartScaler', StandardScaler())])

ds_clean.iloc[:,0:-1] = trickbag['StandartScaler'].fit_transform(ds_clean.iloc[:,0:-1])

try:
    ds_clean.to_csv(url_ds_scaled, index = False)
except: print('Houve um erro em importar o Dataset')

try:
    trickbag_file = open(url_trickbag, 'wb')
    joblib.dump(trickbag,trickbag_file)
    print('===== Modelo e Dataset Gravados =====')
except: print('Houve um erro em importar o Modelo StandartScaler')
