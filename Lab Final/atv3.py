import pandas as pd
from sklearn.model_selection import train_test_split
import sys

#### PARAMETROS ####
input = sys.argv[1]
trickbag_url = sys.argv[2]
url_features_train = sys.argv[3]
url_features_validation = sys.argv[4]
proporcao_treino_teste = float(sys.argv[5])
rdm_state = 42
dataset = pd.read_csv(input, header=None, sep=';')
#### FIM PARAMETROS ####

dataset = pd.read_csv(input)
features, features_validation= train_test_split(dataset,
                                          test_size=proporcao_treino_teste,
                                          random_state=rdm_state)

try:
    features.to_csv(url_features_train, index = False)
    features_validation.to_csv(url_features_validation, index=False)
    print('Datasets separados e gravados com sucesso')
except : print('Houve um problema na gravação dos datasets')
