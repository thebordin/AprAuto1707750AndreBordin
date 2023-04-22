#import matplotlib.pyplot as plt #Importa a biblioteca matplotlib associada ao comando plt com a função de desenhar gráficos
import pickle as p1 #importa a biblioteca pickle associada ao comando p1 com a função de serializar o código.
#import numpy as np #importa a biblioteca NumPy associada ao comando np com a função de suporte a grandes Datasets multi-dimensionais e matrizes assim como uma grande coleção de funções matemáticas de alto nível.
import pandas as pd #importa a biblioteca pandas associada ao comando pd com a função de facilitar a manipulação de dados a partir de diversos tipos de entrada.
from sklearn import linear_model#importa a biblioteca sklearn com as funções 1- Armazenar conjunto de datasets que podem ser importados. 2- de implementar varios modelos de calculos lineares
#from sklearn.metrics import mean_squared_error, r2_score #importa a biblioteca como ferramentas de coeficiente de corelação

data = pd.read_csv('dataset/optdigits.tra', sep=',', header=None)
train_data=data[:]
data_X=train_data.iloc[:,0:64]
data_Y=train_data.iloc[:,64:65]
print(train_data.columns)
print(data_X)
print(data_Y)
regr = linear_model.LinearRegression()
preditor_linear_model: object=regr.fit(data_X, data_Y)
preditor_Pickle= open('dataset/character_predictor', 'wb')
print('===== CHARACTER PREDICTOR =====')
p1.dump(preditor_linear_model, preditor_Pickle)