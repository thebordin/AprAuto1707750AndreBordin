import pandas as pd
from sklearn.model_selection import train_test_split

#### PARAMETROS ####
url_ds_scaled = 'data/scale_data_2.csv'
url_train = 'data/heart_train_3.csv'
url_validation = 'data/heart_validation_3.csv'
url_features_train = 'data/heart_train_3.csv'
url_features_validation = 'data/heart_validation_3.csv'
proporcao_treino_teste = 0.15
rdm_state = 0

#### FIM PARAMETROS ####

dataset = pd.read_csv(url_ds_scaled)
features, features_validation= train_test_split(dataset,
                                          test_size=proporcao_treino_teste,
                                          random_state=rdm_state)

try:
    features.to_csv(url_features_train, index = False)
    features_validation.to_csv(url_features_validation, index=False)
    print('Datasets separados e gravados com sucesso')
except : print('Houve um problema na gravação dos datasets')
