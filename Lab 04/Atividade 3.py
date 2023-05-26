import pandas as pd
from sklearn.model_selection import train_test_split
#### PARAMETROS ####
url_ds_scaled = 'data/scale_data_2.csv'
url_train = 'data/heart_train_3.csv'
url_validation = 'data/heart_validation_3.csv'
proporcao_treino_teste = 0.1
rdm_state = 0
'''url_trickbag = 'data/trickbag'
trickbag = joblib.load(open(url_trickbag, 'rb'))'''
#### FIM PARAMETROS ####

dataset = pd.read_csv(url_ds_scaled)
features, features_teste= train_test_split(dataset,
                                          test_size=proporcao_treino_teste,
                                          random_state=rdm_state)